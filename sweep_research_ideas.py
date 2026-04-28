"""Single in-process sweep over inference-only research ideas.

Modes tested:
  - baseline                : current best (adaptive M static schedule + batched draft)
  - pwls                    : Posterior-Weighted Leaf Selection
  - bqat50                  : Bonus-Quality-Aware Truncation, threshold 0.5
  - bqat70                  : BQAT threshold 0.7
  - pwls+bqat50             : both stacked
  - cppr_l05                : Cross-Path Posterior Re-Ranking, lambda=0.5
  - cppr_l10                : CPPR lambda=1.0

Each mode runs at B=1, 4, 16 with the current best static M schedule.
"""
import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import DFlashDraftModel, load_and_process_dataset
from dflash_batched import dflash_generate_batched, vanilla_ar_generate_batched
from benchmark_batched import tokenize_prompts, make_padded_batch, chunk_rows_list, _resolve_mts


MODES = [
    ("baseline",      dict()),
    ("pwls",          dict(pwls=True)),
    ("bqat50",        dict(bqat_threshold=0.5)),
    ("bqat70",        dict(bqat_threshold=0.7)),
    ("pwls+bqat50",   dict(pwls=True, bqat_threshold=0.5)),
    ("cppr_l05",      dict(cppr=True, cppr_lambda=0.5)),
    ("cppr_l10",      dict(cppr=True, cppr_lambda=1.0)),
]
# Static schedule from earlier engineering work.
SCHEDULE = {1: 64, 2: 32, 4: 32, 8: 16, 16: 16, 32: 16}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--batch-sizes", type=str, default="1,4,16")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--output-json", type=str, default="logs/research_sweep.json")
    args = parser.parse_args()

    Bs = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    print("Loading models...")
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

    ds = load_and_process_dataset("math500").shuffle(seed=0)
    raw = [ds[i]["turns"][0] for i in range(min(len(ds), args.max_samples * 4))]
    prompts_list = tokenize_prompts(
        raw, tokenizer, max_samples=args.max_samples,
        max_prompt_tokens=args.max_prompt_tokens, device=device,
    )
    print(f"Tokenized {len(prompts_list)} prompts.")

    # GPU warmup.
    warm_ids, warm_attn = make_padded_batch(prompts_list[:1], pad_id, device)
    _ = vanilla_ar_generate_batched(
        target=target, input_ids=warm_ids, attention_mask=warm_attn,
        eos_token_ids=eos_ids, max_new_tokens=8,
    )
    torch.cuda.empty_cache()

    results = []

    # Vanilla AR baseline (run once per B; same across all modes).
    print("\n=== Vanilla AR baseline ===")
    van_by_B = {}
    for B in Bs:
        van_total_t, van_total_o = 0.0, 0
        for chunk in chunk_rows_list(prompts_list, B):
            ids, attn = make_padded_batch(chunk, pad_id, device)
            v_out = vanilla_ar_generate_batched(
                target=target, input_ids=ids, attention_mask=attn,
                eos_token_ids=eos_ids, max_new_tokens=args.max_new_tokens,
            )
            van_total_t += v_out.total_decode_time
            van_total_o += sum(v_out.num_output_tokens)
            del v_out, ids, attn
            torch.cuda.empty_cache()
        van_tps = van_total_o / van_total_t
        van_by_B[B] = van_tps
        print(f"  B={B:>2}: vanilla {van_tps:.1f} tok/s")

    # Now each mode.
    for mode_name, mode_kwargs in MODES:
        print(f"\n=== mode: {mode_name} ===")
        for B in Bs:
            eff_mts = _resolve_mts(B, 16, SCHEDULE)
            v7_total_t, v7_total_o = 0.0, 0
            v7_acc = []
            for chunk in chunk_rows_list(prompts_list, B):
                ids, attn = make_padded_batch(chunk, pad_id, device)
                s_out = dflash_generate_batched(
                    draft=draft, target=target, input_ids=ids, attention_mask=attn,
                    mask_token_id=draft.mask_token_id, eos_token_ids=eos_ids,
                    max_new_tokens=args.max_new_tokens, block_size=block_size,
                    max_tree_size=eff_mts, expand_k=8, temperature=0.0,
                    **mode_kwargs,
                )
                v7_total_t += s_out.total_decode_time
                v7_total_o += sum(s_out.num_output_tokens)
                for lst in s_out.acceptance_lengths_per_elem:
                    v7_acc.extend(lst)
                del s_out, ids, attn
                torch.cuda.empty_cache()
            v7_tps = v7_total_o / v7_total_t
            tau = float(np.mean(v7_acc)) if v7_acc else float("nan")
            speedup = v7_tps / van_by_B[B]
            row = {
                "mode": mode_name, "B": B, "M": eff_mts,
                "v7_tps": round(v7_tps, 1), "vanilla_tps": round(van_by_B[B], 1),
                "speedup": round(speedup, 3), "tau": round(tau, 3),
            }
            results.append(row)
            print(f"  B={B:>2} M={eff_mts:>3}: v7 {v7_tps:>7.1f} tok/s, tau={tau:.2f}, speedup={speedup:.3f}×")

    # Per-(mode, B) speedup table.
    print("\n=== summary speedup (vs vanilla AR) ===")
    by_B = {B: {} for B in Bs}
    for r in results:
        by_B[r["B"]][r["mode"]] = r["speedup"]
    header = "  B   " + " | ".join(f"{m:>13}" for m, _ in MODES)
    print(header)
    for B in Bs:
        row = f"{B:>3}   " + " | ".join(f"{by_B[B].get(m, float('nan')):>13.3f}" for m, _ in MODES)
        print(row)

    print("\n=== summary tau ===")
    by_B_tau = {B: {} for B in Bs}
    for r in results:
        by_B_tau[r["B"]][r["mode"]] = r["tau"]
    print(header)
    for B in Bs:
        row = f"{B:>3}   " + " | ".join(f"{by_B_tau[B].get(m, float('nan')):>13.3f}" for m, _ in MODES)
        print(row)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"modes": [m for m, _ in MODES], "results": results}, f, indent=2)
    print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
