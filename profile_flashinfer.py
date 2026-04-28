"""FlashInfer tree-attention profile.

FlashInfer 0.6.4 is the production-grade kernel mentioned in the roadmap.
This script profiles its `BatchPrefillWithRaggedKVCacheWrapper` with a
tree-aware `custom_mask` against:
  (A) PyTorch SDPA + 4D additive mask  — what the model uses today.
  (B) FlashInfer with a tree-aware custom_mask.

Hardware shapes match Qwen3-4B target: H=32 heads, head_dim=128, bf16.
"""
import argparse
import time

import torch
import torch.nn.functional as F


def make_random_tree(M: int, expand_k: int = 8, seed: int = 0):
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
    return parent, depth


def make_ancestor_mask(parent, M: int):
    """Returns [M, M] bool: anc[i, j] = True iff j is ancestor of i (including i)."""
    anc = torch.zeros(M, M, dtype=torch.bool)
    for i in range(M):
        cur = i
        while cur >= 0:
            anc[i, cur] = True
            if cur == 0:
                if i > 0:
                    anc[i, 0] = True
                break
            cur = parent[cur]
    return anc


@torch.no_grad()
def run_sdpa(Q, K, V, P, anc):
    B, H, M, d = Q.shape
    device = Q.device
    dtype = Q.dtype
    min_val = torch.finfo(dtype).min
    mask = torch.zeros(1, 1, M, P + M, device=device, dtype=dtype)
    mask[:, :, :, P:] = torch.where(
        anc, torch.zeros(M, M, device=device, dtype=dtype),
        torch.full((M, M), min_val, device=device, dtype=dtype),
    )
    return F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)


def precompile_flashinfer(B: int, M: int, P: int, H: int, head_dim: int,
                          anc: torch.Tensor, device: torch.device):
    """Build a BatchPrefillWithRaggedKVCacheWrapper with custom mask.

    FlashInfer ragged mode: each "request" b has q_len=M and kv_len=P+M.
    All requests share kv_indptr layout via prefix sum.
    """
    import flashinfer

    # 128 MB workspace.
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")

    # Indptrs: each of B requests has q_len=M, kv_len=P+M.
    q_indptr = torch.tensor([i * M for i in range(B + 1)], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([i * (P + M) for i in range(B + 1)],
                              dtype=torch.int32, device=device)

    # custom_mask: bool flat, length = sum_b (q_len[b] * kv_len[b]) = B * M * (P+M).
    # mask[b, q_i, k_i] in flat form. For each (b, q_i, k_i):
    #   k_i < P → allow.
    #   k_i >= P → allow if anc[q_i, k_i - P].
    # Since all elements share same tree, we replicate.
    full_mask_per = torch.zeros(M, P + M, dtype=torch.bool, device=device)
    full_mask_per[:, :P] = True
    full_mask_per[:, P:] = anc.to(device)
    custom_mask = full_mask_per.repeat(B, 1, 1).reshape(-1)  # [B * M * (P+M)]

    # plan() compiles the kernel for these shapes.
    wrapper.plan(
        qo_indptr=q_indptr,
        kv_indptr=kv_indptr,
        custom_mask=custom_mask,
        num_qo_heads=H,
        num_kv_heads=H,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        causal=False,
        sm_scale=1.0 / (head_dim ** 0.5),
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )

    def call(Q_BHMD, K_BHND, V_BHND):
        # FlashInfer NHD = [N, H, D] flat-batched.
        # Q: [B*M, H, D]; KV: [B*(P+M), H, D].
        Q_nhd = Q_BHMD.permute(0, 2, 1, 3).contiguous().view(B * M, H, head_dim)
        K_nhd = K_BHND.permute(0, 2, 1, 3).contiguous().view(B * (P + M), H, head_dim)
        V_nhd = V_BHND.permute(0, 2, 1, 3).contiguous().view(B * (P + M), H, head_dim)
        out = wrapper.run(Q_nhd, K_nhd, V_nhd)
        # back to [B, H, M, D].
        return out.view(B, M, H, head_dim).permute(0, 2, 1, 3).contiguous()

    return call


def time_kernel(fn, *args, n_iter=30, warmup=5):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--prefix-lens", type=str, default="200,1024,4096")
    parser.add_argument("--Ms", type=str, default="16,32,64,128")
    parser.add_argument("--Bs", type=str, default="1,4,8,16,32")
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
    print(f"Profiling B={B_list}, P={P_list}, M={M_list}")

    print(f"\n{'B':>3} {'P':>5} {'M':>4} | {'sdpa_ms':>10} {'fi_ms':>10} {'speedup':>8}")
    print("-" * 60)

    for B in B_list:
        for P in P_list:
            for M in M_list:
                parent, depth = make_random_tree(M, expand_k=8, seed=0)
                anc = make_ancestor_mask(parent, M).to(device)

                Q = torch.randn(B, H, M, d, device=device, dtype=dtype) * 0.1
                K = torch.randn(B, H, P + M, d, device=device, dtype=dtype) * 0.1
                V = torch.randn(B, H, P + M, d, device=device, dtype=dtype) * 0.1

                # SDPA baseline.
                t_sdpa = time_kernel(run_sdpa, Q, K, V, P, anc, n_iter=args.n_iter)

                # FlashInfer.
                try:
                    fi_fn = precompile_flashinfer(B, M, P, H, d, anc, device)
                    t_fi = time_kernel(fi_fn, Q, K, V, n_iter=args.n_iter)
                    speedup = f"{t_sdpa / t_fi:>7.2f}x"
                    fi_str = f"{t_fi:>9.3f}ms"
                except Exception as e:
                    fi_str = "FAIL"
                    speedup = "—"
                    print(f"  fi failed B={B} P={P} M={M}: {type(e).__name__}: {str(e)[:80]}")

                print(f"{B:>3} {P:>5} {M:>4} | {t_sdpa:>9.3f}ms {fi_str} {speedup}")


if __name__ == "__main__":
    main()
