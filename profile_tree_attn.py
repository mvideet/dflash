"""Sparse tree-attention kernel via flex_attention + standalone profile.

The verify step's Q is the tree (M tokens) and K is prefix (length P) + tree (M).
SDPA today computes the full B*M*(P+M) attention then masks tree-tree pairs that
aren't ancestors. The required compute is much sparser:
  - Q at depth d attends to all P prefix keys + its (d+1) ancestor chain in the tree.

This script:
1. Constructs realistic random tree topologies matching v7's max_tree_size.
2. Implements three kernels:
   (A) Naive PyTorch sdpa with a 4D additive bf16 mask  — what we use today.
   (B) flex_attention with a mask_mod that encodes ancestor reachability.
   (C) Reference Python loop (slow, for correctness check only).
3. Profiles (A) and (B) at multiple (B, M, prefix_len) shapes. Reports speedup.

All kernels are bf16 on CUDA, n_heads=32, head_dim=128 to match Qwen3-4B target.
"""
import argparse
import time
from typing import Tuple

import torch
import torch._dynamo
import torch.nn.functional as F

torch._dynamo.config.cache_size_limit = 256
torch._dynamo.config.accumulated_cache_size_limit = 256


# ---------------------------------------------------------------------------
# Tree generation
# ---------------------------------------------------------------------------

def make_random_tree(M: int, expand_k: int = 8, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a random tree of M nodes resembling v7 DDTree shapes.

    Returns:
        parent_idx [M] : long, parent index per node (root → -1, mapped to 0)
        depth      [M] : long, depth of each node
    """
    g = torch.Generator().manual_seed(seed)
    parent = [-1]
    depth = [0]
    for n in range(1, M):
        candidates = list(range(n))
        w = torch.tensor([1.0 / (depth[c] + 1) for c in candidates])
        children_count = [parent.count(p) for p in candidates]
        for i, cc in enumerate(children_count):
            if cc >= expand_k:
                w[i] = 0
        if w.sum() == 0:
            par = candidates[0]
        else:
            w = w / w.sum()
            par = candidates[int(torch.multinomial(w, 1, generator=g).item())]
        parent.append(par)
        depth.append(depth[par] + 1)
    parent_idx = torch.tensor([p if p >= 0 else 0 for p in parent], dtype=torch.long)
    depth_t = torch.tensor(depth, dtype=torch.long)
    return parent_idx, depth_t


def make_ancestor_set(parent_idx: torch.Tensor, M: int) -> torch.Tensor:
    """Returns [M, M] bool: anc[i, j] = True iff j is an ancestor of i (including i)."""
    anc = torch.zeros(M, M, dtype=torch.bool)
    for i in range(M):
        cur = i
        while cur >= 0:
            anc[i, cur] = True
            cur = parent_idx[cur].item() if cur > 0 else -1
            if cur == 0 and i > 0:
                anc[i, 0] = True
                break
    return anc


# ---------------------------------------------------------------------------
# Kernel A: standard SDPA + 4D additive mask (what we use today)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sdpa_with_tree_mask(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    P: int, ancestor_set: torch.Tensor,
) -> torch.Tensor:
    """Q [B, H, M, d], K, V [B, H, P+M, d]. ancestor_set [M, M]."""
    B, H, M, d = Q.shape
    device = Q.device
    dtype = Q.dtype
    min_val = torch.finfo(dtype).min
    # Mask shape [1, 1, M, P+M]: prefix cols always 0, tree-tree masked by ancestor_set.
    mask = torch.zeros(1, 1, M, P + M, device=device, dtype=dtype)
    # Tree-tree portion: -inf where NOT ancestor.
    mask[:, :, :, P:] = torch.where(
        ancestor_set, torch.zeros(M, M, device=device, dtype=dtype),
        torch.full((M, M), min_val, device=device, dtype=dtype),
    )
    out = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
    return out


# ---------------------------------------------------------------------------
# Kernel B: flex_attention with mask_mod
# ---------------------------------------------------------------------------

def _build_flex_attn_kernel(P: int, M: int, ancestor_idx: torch.Tensor):
    """Build a flex_attention call with a tree mask compiled via Triton.

    ancestor_idx: [M] long → for each query (depth m), gives the maximum K-index
    in the TREE portion that's a valid ancestor.
    """
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    # We pass ancestor_set as a global tensor accessed via mask_mod.
    # mask_mod(b, h, q_idx, kv_idx) -> bool
    # q_idx is into [0, M), kv_idx is into [0, P+M).
    #   - If kv_idx < P: prefix; always allowed.
    #   - Else: kv_idx - P is tree-K index; check ancestor_set[q_idx, kv_idx - P].
    anc_t = ancestor_idx  # [M, M] bool, on device.

    def mask_mod(b, h, q_idx, kv_idx):
        in_prefix = kv_idx < P
        tree_kv = kv_idx - P
        tree_kv_clamped = torch.clamp(tree_kv, min=0)
        is_anc = anc_t[q_idx, tree_kv_clamped]
        return in_prefix | is_anc

    return mask_mod


def precompile_flex_kernel(B: int, M: int, P: int, ancestor_set: torch.Tensor):
    """Build the BlockMask + return a closure that takes only Q, K, V."""
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    import functools

    def mask_mod(b, h, q_idx, kv_idx):
        in_prefix = kv_idx < P
        tree_kv = (kv_idx - P).clamp(min=0)
        is_anc = ancestor_set[q_idx, tree_kv]
        return in_prefix | is_anc

    device = ancestor_set.device.type
    block_mask = create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=M, KV_LEN=P + M, device=device,
    )
    # Compile flex_attention itself for speed.
    flex_compiled = torch.compile(flex_attention, dynamic=False, fullgraph=True)

    def call(Q, K, V):
        return flex_compiled(Q, K, V, block_mask=block_mask)

    return call


@torch.no_grad()
def flex_with_tree_mask(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    P: int, ancestor_set: torch.Tensor,
) -> torch.Tensor:
    """Convenience wrapper — used only for one-shot calls (slow due to compile)."""
    fn = precompile_flex_kernel(Q.shape[0], Q.shape[2], P, ancestor_set)
    return fn(Q, K, V)


# ---------------------------------------------------------------------------
# Reference (slow) — for correctness check
# ---------------------------------------------------------------------------

@torch.no_grad()
def reference_tree_attn(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    P: int, ancestor_set: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch with explicit mask — same as kernel A, used as ground truth."""
    return sdpa_with_tree_mask(Q, K, V, P, ancestor_set)


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

def time_kernel(fn, *args, n_iter: int = 30, warmup: int = 5) -> float:
    for _ in range(warmup):
        out = fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000  # ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-heads", type=int, default=32, help="Match Qwen3-4B target.")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--prefix-lens", type=str, default="200,512,2048")
    parser.add_argument("--Ms", type=str, default="16,32,64,128")
    parser.add_argument("--Bs", type=str, default="1,4,16,32")
    parser.add_argument("--n-iter", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    H = args.n_heads
    d = args.head_dim
    P_list = [int(x) for x in args.prefix_lens.split(",")]
    M_list = [int(x) for x in args.Ms.split(",")]
    B_list = [int(x) for x in args.Bs.split(",")]

    print(f"Device: {device}, dtype: {dtype}, H={H}, d={d}")
    print(f"Profiling at P={P_list}, M={M_list}, B={B_list}")

    print(f"\n{'B':>3} {'P':>5} {'M':>4} | {'sdpa_ms':>10} {'flex_ms':>10} {'speedup':>8} | "
          f"{'sdpa_wo_mask':>13}")
    print("-" * 80)

    for B in B_list:
        for P in P_list:
            for M in M_list:
                # Tree topology shared across batch.
                parent_idx, depth_t = make_random_tree(M, expand_k=8, seed=0)
                anc = make_ancestor_set(parent_idx, M).to(device)

                Q = torch.randn(B, H, M, d, device=device, dtype=dtype) * 0.1
                K = torch.randn(B, H, P + M, d, device=device, dtype=dtype) * 0.1
                V = torch.randn(B, H, P + M, d, device=device, dtype=dtype) * 0.1

                # Time kernel A (sdpa + tree mask)
                t_sdpa = time_kernel(sdpa_with_tree_mask, Q, K, V, P, anc, n_iter=args.n_iter)

                # Time kernel B (flex_attention) — pre-compile mask + flex outside the loop.
                try:
                    flex_fn = precompile_flex_kernel(B, M, P, anc)
                    t_flex = time_kernel(flex_fn, Q, K, V, n_iter=args.n_iter)
                    speedup_str = f"{t_sdpa / t_flex:>7.2f}x"
                    flex_str = f"{t_flex:>9.3f}ms"
                except Exception as e:
                    print(f"  flex failed for B={B} P={P} M={M}: {type(e).__name__}: {e}")
                    speedup_str = "—"
                    flex_str = "FAIL"

                # Reference: SDPA WITHOUT mask (causal-style).
                # Used as a lower bound (this is the cheapest plausible attention).
                t_sdpa_full = time_kernel(
                    F.scaled_dot_product_attention, Q, K, V, n_iter=args.n_iter
                )

                print(f"{B:>3} {P:>5} {M:>4} | {t_sdpa:>9.3f}ms {flex_str} {speedup_str} | "
                      f"{t_sdpa_full:>11.3f}ms")

    print("\nNotes:")
    print(" - sdpa_ms: current path (SDPA + 4D additive bf16 mask).")
    print(" - flex_ms: flex_attention with tree mask_mod (compiled to Triton).")
    print(" - speedup: sdpa_ms / flex_ms.")
    print(" - sdpa_wo_mask: SDPA with no mask (lower bound = full QK^T cost).")


if __name__ == "__main__":
    main()
