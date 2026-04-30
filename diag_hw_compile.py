"""HW Idea 3 feasibility: does torch.compile reduce per-call overhead on verify?

The persistent-kernel idea (Idea 3) is months of CUDA work. A cheap proxy is
torch.compile(target.model) — PyTorch's graph-compile fuses kernels and can
drastically reduce launch overhead. If compile gives a 2-4× verify speedup,
the persistent-kernel direction is validated. If compile gives nothing,
either compile is hitting graph-break issues OR the per-call overhead isn't
the bottleneck.
"""
import argparse, json, os, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import DFlashDraftModel, load_and_process_dataset
from dflash_batched import dflash_generate_batched, vanilla_ar_generate_batched
from benchmark_batched import tokenize_prompts, make_padded_batch, chunk_rows_list

DEVICE = torch.device("cuda:0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    torch.manual_seed(0)
    target = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B", attn_implementation="sdpa", dtype=torch.bfloat16,
    ).to(DEVICE).eval()
    draft = DFlashDraftModel.from_pretrained(
        "z-lab/Qwen3-4B-DFlash-b16", attn_implementation="sdpa", dtype=torch.bfloat16,
    ).to(DEVICE).eval()
    block_size = draft.block_size
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_ids = [tokenizer.eos_token_id]
    mid = draft.mask_token_id

    ds = load_and_process_dataset("math500").shuffle(seed=0)
    raw = [ds[i]["turns"][0] for i in range(min(len(ds), args.max_samples * 4))]
    prompts_list = tokenize_prompts(
        raw, tokenizer, max_samples=args.max_samples,
        max_prompt_tokens=256, device=DEVICE,
    )

    # Test direct verify-call latency at a fixed shape
    # using target.model directly (not the full generate loop).
    B, M = 4, 32
    prefix_len = 200
    print(f"\n=== Verify-call latency at B={B}, M={M}, prefix={prefix_len} ===")

    # Construct sample inputs
    input_ids = torch.randint(0, 50000, (B, M), device=DEVICE)
    pos_ids = torch.arange(prefix_len, prefix_len + M, device=DEVICE).unsqueeze(0).expand(B, -1)
    attn_mask = torch.zeros(B, 1, M, prefix_len + M, dtype=torch.bfloat16, device=DEVICE)
    from transformers import DynamicCache

    # Pre-fill cache by running prefill
    pre_input = torch.randint(0, 50000, (B, prefix_len), device=DEVICE)
    pre_attn = torch.ones(B, prefix_len, dtype=torch.long, device=DEVICE)
    past_kv = DynamicCache()
    with torch.inference_mode():
        target.model(pre_input, attention_mask=pre_attn, past_key_values=past_kv,
                     use_cache=True)

    def verify_call(model, past_kv_local):
        with torch.inference_mode():
            return model(input_ids, position_ids=pos_ids,
                         past_key_values=past_kv_local, use_cache=False,
                         attention_mask=attn_mask)

    # Snapshot KV cache state for reuse
    import copy
    kv_snapshot = copy.deepcopy(past_kv)

    def fresh_kv():
        return copy.deepcopy(kv_snapshot)

    # Eager (default) path
    def eager_fn():
        kv = fresh_kv()
        return verify_call(target.model, kv)

    # Time eager
    for _ in range(3): eager_fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20): eager_fn()
    torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - t0) / 20 * 1000
    print(f"  eager:  {eager_ms:.2f} ms/call")

    # Compile path
    print(f"  compiling...", flush=True)
    try:
        compiled = torch.compile(target.model, mode="reduce-overhead", fullgraph=False)
        # Warmup compile (first few calls trigger compilation)
        for i in range(5):
            t0 = time.perf_counter()
            kv = fresh_kv()
            with torch.inference_mode():
                compiled(input_ids, position_ids=pos_ids,
                         past_key_values=kv, use_cache=False,
                         attention_mask=attn_mask)
            torch.cuda.synchronize()
            t = (time.perf_counter() - t0) * 1000
            print(f"    warmup {i}: {t:.2f} ms")
        # Time compiled
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            kv = fresh_kv()
            with torch.inference_mode():
                compiled(input_ids, position_ids=pos_ids,
                         past_key_values=kv, use_cache=False,
                         attention_mask=attn_mask)
        torch.cuda.synchronize()
        compile_ms = (time.perf_counter() - t0) / 20 * 1000
        print(f"  compiled: {compile_ms:.2f} ms/call")
        speedup = eager_ms / compile_ms
        print(f"  → torch.compile speedup: {speedup:.2f}×")
    except Exception as e:
        print(f"  compile failed: {e}")
        compile_ms = None
        speedup = None

    out = {"B": B, "M": M, "prefix": prefix_len,
           "eager_ms": eager_ms, "compile_ms": compile_ms, "speedup": speedup}
    os.makedirs("logs", exist_ok=True)
    with open("logs/hw_compile.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote logs/hw_compile.json")


if __name__ == "__main__":
    main()
