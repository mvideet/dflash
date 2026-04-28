"""Interleaved-trials sweep over inference-only research ideas.

Critical fix vs earlier sweep: each mode runs the same fixed (B, prompts) WITH
WARMUP and INTERLEAVED across 3 trials, so warm-cache effects affect all modes
equally. Each (mode, B) is the median of 3 trials.

Modes tested:
  - baseline    : current best
  - pwls        : Posterior-Weighted Leaf Selection
  - bqat50      : Bonus-Quality-Aware Truncation t=0.5
  - cppr_l05    : Cross-Path Posterior Re-Ranking λ=0.5
"""
import argparse
import json
import os
import statistics

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import DFlashDraftModel, load_and_process_dataset
from dflash_batched import dflash_generate_batched, vanilla_ar_generate_batched
from benchmark_batched import tokenize_prompts, make_padded_batch, chunk_rows_list, _resolve_mts


MODES = [
    ("baseline", dict()),
    ("pwls",     dict(pwls=True)),
    ("bqat50",   dict(bqat_threshold=0.5)),
    ("cppr_l05", dict(cppr=True, cppr_lambda=0.5)),
]
SCHEDULE = {1: 64, 2: 32, 4: 32, 8: 16, 16: 16, 32: 16}


def measure_mode(target, draft, prompts_list, B, mode_kwargs, *,
                 max_new_tokens, max_prompt_tokens, eff_mts, block_size,
                 mask_token_id, pad_id, eos_ids, device):
    v7_total_t, v7_total_o = 0.0, 0
    v7_acc = []
    for chunk in chunk_rows_list(prompts_list, B):
        ids, attn = make_padded_batch(chunk, pad_id, device)
        s_out = dflash_generate_batched(
            draft=draft, target=target, input_ids=ids, attention_mask=attn,
            mask_token_id=mask_token_id, eos_token_ids=eos_ids,
            max_new_tokens=max_new_tokens, block_size=block_size,
            max_tree_size=eff_mts, expand_k=8, temperature=0.0,
            **mode_kwargs,
        )
        v7_total_t += s_out.total_decode_time
        v7_total_o += sum(s_out.num_output_tokens)
        for lst in s_out.acceptance_lengths_per_elem:
            v7_acc.extend(lst)
        del s_out, ids, attn
        torch.cuda.empty_cache()
    return v7_total_o / v7_total_t, float(np.mean(v7_acc)) if v7_acc else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--batch-sizes", type=str, default="1,4,16")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--output-json", type=str, default="logs/research_interleaved.json")
    args = parser.parse_args()

    Bs = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    print("Loading models...", flush=True)
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
    mask_token_id = draft.mask_token_id

    ds = load_and_process_dataset("math500").shuffle(seed=0)
    raw = [ds[i]["turns"][0] for i in range(min(len(ds), args.max_samples * 4))]
    prompts_list = tokenize_prompts(
        raw, tokenizer, max_samples=args.max_samples, max_prompt_tokens=256, device=device,
    )
    print(f"Tokenized {len(prompts_list)} prompts.", flush=True)

    # Vanilla AR baseline (one trial each B is enough; vanilla doesn't vary).
    van_by_B = {}
    print("\n=== Vanilla AR baseline ===", flush=True)
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
        van_by_B[B] = van_total_o / van_total_t
        print(f"  B={B:>2}: vanilla {van_by_B[B]:.1f} tok/s", flush=True)

    # WARMUP: run each mode once at each B to populate GPU caches.
    print("\n=== Warmup (all modes, all B) ===", flush=True)
    for B in Bs:
        eff_mts = _resolve_mts(B, 16, SCHEDULE)
        for mode_name, mode_kwargs in MODES:
            _ = measure_mode(target, draft, prompts_list, B, mode_kwargs,
                             max_new_tokens=64, max_prompt_tokens=256,
                             eff_mts=eff_mts, block_size=block_size,
                             mask_token_id=mask_token_id, pad_id=pad_id,
                             eos_ids=eos_ids, device=device)
            torch.cuda.empty_cache()
        print(f"  warmed B={B}", flush=True)

    # INTERLEAVED TRIALS: for each B, do n_trials of [mode_a, mode_b, ...] in random order.
    results_by_mode_B = {(m, B): {"tps": [], "tau": []} for m, _ in MODES for B in Bs}
    print(f"\n=== Trials (n={args.n_trials}, interleaved) ===", flush=True)
    import random
    rng = random.Random(0)
    for trial in range(args.n_trials):
        mode_order = list(MODES)
        rng.shuffle(mode_order)
        for B in Bs:
            eff_mts = _resolve_mts(B, 16, SCHEDULE)
            for mode_name, mode_kwargs in mode_order:
                tps, tau = measure_mode(
                    target, draft, prompts_list, B, mode_kwargs,
                    max_new_tokens=args.max_new_tokens, max_prompt_tokens=256,
                    eff_mts=eff_mts, block_size=block_size,
                    mask_token_id=mask_token_id, pad_id=pad_id,
                    eos_ids=eos_ids, device=device,
                )
                results_by_mode_B[(mode_name, B)]["tps"].append(tps)
                results_by_mode_B[(mode_name, B)]["tau"].append(tau)
                print(f"  trial={trial} B={B:>2} mode={mode_name:>12}: {tps:.1f} tok/s, tau={tau:.2f}",
                      flush=True)

    # Aggregate
    summary = []
    print("\n=== Median speedup (vs vanilla, n_trials trials) ===", flush=True)
    header = "  B   " + " | ".join(f"{m:>11}" for m, _ in MODES)
    print(header, flush=True)
    for B in Bs:
        row_speedups = []
        for m, _ in MODES:
            tps_med = statistics.median(results_by_mode_B[(m, B)]["tps"])
            spd = tps_med / van_by_B[B]
            row_speedups.append(spd)
            summary.append({
                "mode": m, "B": B,
                "v7_tps_median": round(tps_med, 1),
                "v7_tps_mean": round(statistics.mean(results_by_mode_B[(m, B)]["tps"]), 1),
                "v7_tps_stdev": round(statistics.stdev(results_by_mode_B[(m, B)]["tps"]) if args.n_trials > 1 else 0, 2),
                "tau_median": round(statistics.median(results_by_mode_B[(m, B)]["tau"]), 3),
                "speedup_median": round(spd, 3),
                "vanilla_tps": round(van_by_B[B], 1),
            })
        row = f"{B:>3}   " + " | ".join(f"{s:>11.3f}" for s in row_speedups)
        print(row, flush=True)

    print("\n=== Per-trial tps stdev (noise floor) ===", flush=True)
    print(header, flush=True)
    for B in Bs:
        row_std = []
        for m, _ in MODES:
            std_v = (statistics.stdev(results_by_mode_B[(m, B)]["tps"])
                     if args.n_trials > 1 else 0.0)
            row_std.append(std_v)
        row = f"{B:>3}   " + " | ".join(f"{s:>11.2f}" for s in row_std)
        print(row, flush=True)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"vanilla_tps": van_by_B, "summary": summary}, f, indent=2)
    print(f"\nWrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
