"""HW Idea G: spine-first verify with torch.compile.

Earlier finding: spine-first lost because per-call overhead was ~30ms; two
sequential calls cost ~60ms vs 35ms single call → -25%.

With torch.compile cutting per-call overhead 1.81× (34→19ms), retesting:
- single full forward (q=32): ~19ms compiled
- spine forward (q=16): ~12-15ms compiled
- side forward (q=16): ~12-15ms compiled
- spine-first avg: 0.3 × spine + 0.7 × (spine + side)

If spine + side adds < single full forward, spine-first wins net on the
30% full-accept rounds.

Micro-benchmark (no full pipeline integration) — just measures verify-call
timings under different configurations.
"""
import argparse, json, os, time, copy
import torch
from transformers import AutoModelForCausalLM
from transformers import DynamicCache

DEVICE = torch.device("cuda:0")


def time_fn(fn, n=20, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def main():
    print("Loading model...", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B", attn_implementation="sdpa", dtype=torch.bfloat16,
    ).to(DEVICE).eval()

    B = 4
    block_size = 16
    prefix_len = 200

    # Pre-fill target KV
    pre = torch.randint(0, 50000, (B, prefix_len), device=DEVICE)
    pre_attn = torch.ones(B, prefix_len, dtype=torch.long, device=DEVICE)
    base_kv = DynamicCache()
    with torch.inference_mode():
        target.model(pre, attention_mask=pre_attn, past_key_values=base_kv,
                     use_cache=True)

    def fresh_kv(): return copy.deepcopy(base_kv)

    # Three q_len configurations:
    # (a) Full tree at M=32 (verify the full M-budget)
    # (b) Spine at q=16 (verify the rank-0 chain)
    # (c) Side at q=16 (verify only side branches; needs prefix+spine in KV)

    def make_inputs(q_len):
        ids = torch.randint(0, 50000, (B, q_len), device=DEVICE)
        pos = torch.arange(prefix_len, prefix_len + q_len, device=DEVICE).unsqueeze(0).expand(B, -1)
        # Add spine to KV first if simulating side verify
        attn_mask = torch.zeros(B, 1, q_len, prefix_len + q_len,
                                 dtype=torch.bfloat16, device=DEVICE)
        return ids, pos, attn_mask

    print("\n=== Verify call timings ===")
    print(f"  config: B={B}, prefix={prefix_len}, dtype=bf16")

    for q_len in [16, 32]:
        ids, pos, attn = make_inputs(q_len)

        def eager_call():
            kv = fresh_kv()
            with torch.inference_mode():
                return target.model(ids, position_ids=pos, past_key_values=kv,
                                    use_cache=False, attention_mask=attn)

        eager_ms = time_fn(eager_call, n=20)
        print(f"\n  q={q_len} eager:    {eager_ms:.2f} ms/call")

    print(f"\n  Compiling target.model (mode=reduce-overhead)...", flush=True)
    compiled = torch.compile(target.model, mode="reduce-overhead", fullgraph=False)

    timings = {}
    for q_len in [16, 32]:
        ids, pos, attn = make_inputs(q_len)

        def compiled_call():
            kv = fresh_kv()
            with torch.inference_mode():
                return compiled(ids, position_ids=pos, past_key_values=kv,
                                use_cache=False, attention_mask=attn)

        # Warmup heavy first call (triggers compile on this q_len)
        print(f"  q={q_len} compile warmup...", flush=True)
        for i in range(5):
            t0 = time.perf_counter()
            compiled_call()
            torch.cuda.synchronize()
            print(f"    iter {i}: {(time.perf_counter()-t0)*1000:.0f}ms")

        ms = time_fn(compiled_call, n=20, warmup=2)
        print(f"  q={q_len} compiled: {ms:.2f} ms/call")
        timings[q_len] = ms

    # Predicted spine-first cost
    print("\n=== Spine-first economics ===")
    spine_ms = timings[16]
    full_ms = timings[32]
    side_ms = spine_ms  # same q_len
    full_accept_rate = 0.30

    spine_first_avg = full_accept_rate * spine_ms + (1 - full_accept_rate) * (spine_ms + side_ms)
    print(f"  Single full verify (q=32, compiled):     {full_ms:.2f} ms")
    print(f"  Spine verify (q=16, compiled):           {spine_ms:.2f} ms")
    print(f"  Side verify (q=16, compiled):            {side_ms:.2f} ms")
    print(f"  Spine-first avg ({full_accept_rate*100:.0f}% spine-only + "
          f"{(1-full_accept_rate)*100:.0f}% spine+side):")
    print(f"    {full_accept_rate} × {spine_ms:.1f} + {1-full_accept_rate} × "
          f"({spine_ms:.1f} + {side_ms:.1f}) = {spine_first_avg:.2f} ms")
    delta = (full_ms - spine_first_avg) / full_ms * 100
    print(f"  vs single full: {delta:+.1f}% saving")
    if delta > 0:
        print(f"  → SPINE-FIRST WINS with compile (math flipped)")
    else:
        print(f"  → spine-first still loses; per-call overhead remains too large")

    # Higher full-accept rate scenarios
    print(f"\n  Sensitivity to full-accept rate:")
    for fa in [0.2, 0.3, 0.4, 0.5]:
        sf = fa * spine_ms + (1 - fa) * (spine_ms + side_ms)
        d = (full_ms - sf) / full_ms * 100
        print(f"    full_accept={fa}: spine-first {sf:.1f} ms vs full {full_ms:.1f} ms → {d:+.1f}%")

    out = {"B": B, "prefix": prefix_len,
           "q16_compiled_ms": timings[16],
           "q32_compiled_ms": timings[32],
           "spine_first_avg_ms_at_30pct": spine_first_avg,
           "single_full_ms": full_ms,
           "delta_pct_at_30pct": delta}
    os.makedirs("logs", exist_ok=True)
    with open("logs/hw_spine_compile.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote logs/hw_spine_compile.json")


if __name__ == "__main__":
    main()
