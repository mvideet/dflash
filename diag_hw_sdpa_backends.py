"""HW Idea 2 feasibility: which SDPA backend handles tree-attention best?

PyTorch's scaled_dot_product_attention has 3 backends:
  - math: explicit (Q @ K^T) + softmax + (× V)
  - mem-efficient: xformers-style chunked
  - flash: FA-2 (sometimes available)

For our use (custom mask, small q_len), one might be faster. Test all 3 at
realistic shapes to see if a "free" 2× verify speedup is sitting in a
backend switch.
"""
import time, json, os
import torch

DEVICE = torch.device("cuda:0")


def time_fn(fn, n_iter=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000


def make_qkv(B, H, M, kv_len, D):
    q = torch.randn(B, H, M, D, dtype=torch.bfloat16, device=DEVICE)
    k = torch.randn(B, H, kv_len, D, dtype=torch.bfloat16, device=DEVICE)
    v = torch.randn(B, H, kv_len, D, dtype=torch.bfloat16, device=DEVICE)
    mask = torch.zeros(B, 1, M, kv_len, dtype=torch.bfloat16, device=DEVICE)
    # Random ancestor pattern
    masked = torch.rand(B, 1, M, kv_len, device=DEVICE) > 0.7
    mask[masked] = torch.finfo(torch.bfloat16).min
    return q, k, v, mask


def main():
    configs = [
        (1, 64, 600, "B=1 M=64 prefix=600"),
        (4, 32, 600, "B=4 M=32 prefix=600"),
        (8, 16, 600, "B=8 M=16 prefix=600"),
        (32, 16, 800, "B=32 M=16 prefix=800"),
        (64, 16, 1000, "B=64 M=16 prefix=1000"),
    ]
    H, D = 32, 128

    backends = [
        ("default", None),
        ("math", torch.nn.attention.SDPBackend.MATH),
        ("efficient", torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION),
        ("flash", torch.nn.attention.SDPBackend.FLASH_ATTENTION),
    ]

    out = []
    for B, M, prefix, label in configs:
        kv_len = prefix + M
        q, k, v, mask = make_qkv(B, H, M, kv_len, D)
        print(f"\n=== {label} ===")
        results = {}
        for name, be in backends:
            try:
                if be is None:
                    fn = lambda: torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, attn_mask=mask)
                else:
                    def fn():
                        with torch.nn.attention.sdpa_kernel([be]):
                            return torch.nn.functional.scaled_dot_product_attention(
                                q, k, v, attn_mask=mask)
                ms = time_fn(fn, n_iter=30)
                results[name] = ms
                print(f"  {name:>10}: {ms:.3f} ms")
            except (RuntimeError, ValueError) as e:
                results[name] = None
                print(f"  {name:>10}: failed ({str(e)[:80]})")
        out.append({"label": label, "B": B, "M": M, "prefix": prefix, "ms": results})

    print("\n=== Summary (best-case speedup over 'default' backend) ===")
    print(f"  {'config':>30}  default  math  efficient  flash  best  speedup")
    for r in out:
        ms = r["ms"]
        defm = ms.get("default")
        valid = [(k, v) for k, v in ms.items() if v is not None and k != "default"]
        if not defm or not valid: continue
        best_name, best_ms = min(valid, key=lambda kv: kv[1])
        speedup = defm / best_ms if best_ms else 0
        print(f"  {r['label']:>30}  {defm:>6.3f}  "
              f"{ms.get('math', 0) or 0:>4.2f}  {ms.get('efficient', 0) or 0:>9.3f}  "
              f"{ms.get('flash', 0) or 0:>5.3f}  {best_name:>5}  {speedup:>5.2f}×")

    os.makedirs("logs", exist_ok=True)
    with open("logs/hw_sdpa_backends.json", "w") as f:
        json.dump({"results": out}, f, indent=2)
    print("\nWrote logs/hw_sdpa_backends.json")


if __name__ == "__main__":
    main()
