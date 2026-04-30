"""HW Idea 4 feasibility: tile-aligned tree shapes.

Idea: constrain heap-built tree so per-depth widths are multiples of 4 (the
FA tile size). Each query attends to exactly ⌈max_depth/4⌉ key tiles.
Trades some tau for kernel efficiency.

This is the only purely-algorithmic HW idea — no kernel work needed. Test
by forcing per-depth widths to multiples of 4 via the existing
expand_k_per_depth knob, compare τ and speedup vs vanilla heap.
"""
import argparse, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import DFlashDraftModel, load_and_process_dataset
from dflash_batched import dflash_generate_batched, vanilla_ar_generate_batched
from benchmark_batched import tokenize_prompts, make_padded_batch, chunk_rows_list, _resolve_mts

SCHEDULE = {1: 64, 2: 32, 4: 32, 8: 16, 16: 16, 32: 16}

# Per-depth K: tile-aligned (multiples of 4).
# Two variants: aggressive (K=4 everywhere) and balanced (4 at edges, 8 in middle)
TILE_K_AGGRESSIVE = [4] * 16   # all depths use K=4
TILE_K_BALANCED = [4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.manual_seed(0)
    target = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B", attn_implementation="sdpa", dtype=torch.bfloat16,
    ).to(device).eval()
    draft = DFlashDraftModel.from_pretrained(
        "z-lab/Qwen3-4B-DFlash-b16", attn_implementation="sdpa", dtype=torch.bfloat16,
    ).to(device).eval()
    block_size = draft.block_size
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_ids = [tokenizer.eos_token_id]
    mid = draft.mask_token_id

    ds = load_and_process_dataset("math500").shuffle(seed=0)
    raw = [ds[i]["turns"][0] for i in range(min(len(ds), args.max_samples * 4))]
    prompts_list = tokenize_prompts(
        raw, tokenizer, max_samples=args.max_samples,
        max_prompt_tokens=256, device=device,
    )

    warm_ids, warm_attn = make_padded_batch(prompts_list[:1], pad_id, device)
    _ = vanilla_ar_generate_batched(
        target=target, input_ids=warm_ids, attention_mask=warm_attn,
        eos_token_ids=eos_ids, max_new_tokens=8,
    )
    torch.cuda.empty_cache()

    rows = []
    for B in [1, 4]:
        M = _resolve_mts(B, 16, SCHEDULE)
        van_t, van_o = 0.0, 0
        for chunk in chunk_rows_list(prompts_list, B):
            ids, attn = make_padded_batch(chunk, pad_id, device)
            v_out = vanilla_ar_generate_batched(
                target=target, input_ids=ids, attention_mask=attn,
                eos_token_ids=eos_ids, max_new_tokens=args.max_new_tokens,
            )
            van_t += v_out.total_decode_time
            van_o += sum(v_out.num_output_tokens)
            del v_out, ids, attn; torch.cuda.empty_cache()
        van_tps = van_o / van_t

        for variant, kw in [
            ("baseline_K8", dict()),
            ("tile_K4",     dict(expand_k_per_depth=TILE_K_AGGRESSIVE)),
            ("tile_balanced", dict(expand_k_per_depth=TILE_K_BALANCED)),
        ]:
            torch.manual_seed(0)
            v7_t, v7_o, accs = 0.0, 0, []
            for chunk in chunk_rows_list(prompts_list, B):
                ids, attn = make_padded_batch(chunk, pad_id, device)
                s_out = dflash_generate_batched(
                    draft=draft, target=target, input_ids=ids, attention_mask=attn,
                    mask_token_id=mid, eos_token_ids=eos_ids,
                    max_new_tokens=args.max_new_tokens, block_size=block_size,
                    max_tree_size=M, expand_k=8, temperature=0.0,
                    **kw,
                )
                v7_t += s_out.total_decode_time
                v7_o += sum(s_out.num_output_tokens)
                for lst in s_out.acceptance_lengths_per_elem:
                    accs.extend(lst)
                del s_out, ids, attn; torch.cuda.empty_cache()
            v7_tps = v7_o / v7_t
            tau = float(np.mean(accs)) if accs else float("nan")
            speedup = v7_tps / van_tps
            rows.append({"B": B, "variant": variant, "M": M,
                         "v7_tps": round(v7_tps, 1),
                         "speedup": round(speedup, 3), "tau": round(tau, 3)})
            print(f"  B={B} {variant:>15}: tau={tau:.2f} speedup={speedup:.3f}×",
                  flush=True)

    print("\n=== Summary ===")
    for B in [1, 4]:
        rows_B = [r for r in rows if r["B"] == B]
        baseline = next((r for r in rows_B if r["variant"] == "baseline_K8"), None)
        if not baseline: continue
        print(f"\nB={B}:")
        for r in rows_B:
            gain = 100 * (r["speedup"] - baseline["speedup"]) / baseline["speedup"]
            print(f"  {r['variant']:>15}: speedup={r['speedup']:.3f}×  "
                  f"tau={r['tau']:.2f}  Δ={gain:+5.1f}%")

    os.makedirs("logs", exist_ok=True)
    with open("logs/hw_tile_aligned.json", "w") as f:
        json.dump({"results": rows}, f, indent=2)
    print(f"\nWrote logs/hw_tile_aligned.json")


if __name__ == "__main__":
    main()
