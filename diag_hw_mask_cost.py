"""HW Idea 1 feasibility: mask materialization cost.

Measures the time and memory of constructing the tree-attention mask
[B, 1, M, prefix+M] of bf16 at varying B/M/prefix. Compares to the
attention-compute time on the same shape. Tells us how much the
"compressed-mask kernel" (Idea 1) could save.
"""
import time, json, os
import torch

DEVICE = torch.device("cuda:0")


def time_op(fn, n_iter=10):
    # warmup
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000  # ms


def main():
    # Realistic shapes from our measured math500 sweep.
    configs = [
        # (B, M, prefix, label)
        (1, 64, 600, "B=1 M=64 prefix=600"),
        (4, 32, 600, "B=4 M=32 prefix=600"),
        (8, 16, 600, "B=8 M=16 prefix=600"),
        (32, 16, 800, "B=32 M=16 prefix=800"),
        (64, 16, 1000, "B=64 M=16 prefix=1000"),
    ]
    out = []
    for B, M, prefix, label in configs:
        kv_len = prefix + M
        # 1) Mask construction: dense [B, 1, M, kv_len]
        def build_mask():
            min_val = torch.finfo(torch.bfloat16).min
            m = torch.zeros(B, 1, M, kv_len, dtype=torch.bfloat16, device=DEVICE)
            # Random ancestor pattern (worst-case fill)
            mask = torch.rand(B, M, kv_len, device=DEVICE) > 0.5
            m[~mask.unsqueeze(1)] = min_val
            return m

        # 2) Attention with that mask: simulate target's attention compute
        # Single-head, single-layer proxy — actual model has 32 layers × 32 heads.
        H = 32
        D = 128
        def attn_op():
            q = torch.randn(B, H, M, D, dtype=torch.bfloat16, device=DEVICE)
            k = torch.randn(B, H, kv_len, D, dtype=torch.bfloat16, device=DEVICE)
            v = torch.randn(B, H, kv_len, D, dtype=torch.bfloat16, device=DEVICE)
            mask = torch.zeros(B, 1, M, kv_len, dtype=torch.bfloat16, device=DEVICE)
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask)

        mask_ms = time_op(build_mask, n_iter=20)
        attn_ms = time_op(attn_op, n_iter=20)

        mask_mb = B * M * kv_len * 2 / 1024 / 1024  # bf16

        print(f"\n=== {label} ===")
        print(f"  mask shape: [{B}, 1, {M}, {kv_len}] = {mask_mb:.2f} MB bf16")
        print(f"  mask construction (per call): {mask_ms:.3f} ms")
        print(f"  one-layer one-head attention: {attn_ms:.3f} ms")
        print(f"  full verify (32 layer × 32 head est): ~{attn_ms*32:.0f} ms attention")
        # Idea 1 saves the mask construction. At 32 layers, mask passed 32x but is
        # constructed ONCE per round in our code. So actual savings = 1× mask_ms.
        # If the kernel reads mask 32 times (once per layer) — savings could be more
        # if the mask is bandwidth-limited.
        # Estimated savings:
        full_round_ms_est = attn_ms * 32 * 2  # 32 layers × 2 heads-per-call (rough)
        savings_pct = mask_ms / full_round_ms_est * 100 if full_round_ms_est > 0 else 0
        print(f"  Idea 1 ceiling: ~{savings_pct:.1f}% of round time")
        out.append({
            "B": B, "M": M, "prefix": prefix,
            "mask_ms": mask_ms, "attn_ms": attn_ms,
            "mask_mb": mask_mb, "idea1_pct_ceiling": savings_pct,
        })

    os.makedirs("logs", exist_ok=True)
    with open("logs/hw_mask_cost.json", "w") as f:
        json.dump({"results": out}, f, indent=2)
    print("\nWrote logs/hw_mask_cost.json")


if __name__ == "__main__":
    main()
