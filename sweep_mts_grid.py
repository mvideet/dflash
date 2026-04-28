"""Sweep (B, max_tree_size) grid in-process.

Loads target+draft once, then iterates over (B, mts) configs calling
dflash_generate_batched and vanilla_ar_generate_batched directly. Saves
summary JSON for picking an adaptive schedule.
"""
import argparse
import json
import os
from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import DFlashDraftModel, load_and_process_dataset
from dflash_batched import dflash_generate_batched, vanilla_ar_generate_batched
from benchmark_batched import tokenize_prompts, make_padded_batch, chunk_rows_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--draft-name-or-path", type=str, default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--dataset", type=str, default="math500")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--expand-k", type=int, default=8)
    parser.add_argument("--output-json", type=str, default="logs/mts_sweep_summary.json")
    parser.add_argument("--configs", type=str,
                        default="1:128,1:64,1:32,2:128,2:64,2:32,4:128,4:64,4:32,8:128,8:64,8:32,8:16,16:128,16:64,16:32,16:16",
                        help="Comma-separated B:mts pairs.")
    args = parser.parse_args()

    configs = []
    for tok in args.configs.split(","):
        b, m = tok.split(":")
        configs.append((int(b), int(m)))

    device = torch.device("cuda:0")
    torch.manual_seed(0)

    print(f"Loading target  : {args.model_name_or_path}")
    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, attn_implementation="sdpa", dtype=torch.bfloat16,
    ).to(device).eval()
    print(f"Loading draft   : {args.draft_name_or_path}")
    draft = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path, attn_implementation="sdpa", dtype=torch.bfloat16,
    ).to(device).eval()
    block_size = draft.block_size
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    mask_token_id = draft.mask_token_id
    eos_token_ids = [tokenizer.eos_token_id]

    ds = load_and_process_dataset(args.dataset).shuffle(seed=0)
    raw_prompts = [ds[i]["turns"][0] for i in range(min(len(ds), args.max_samples * 4))]
    prompts_list = tokenize_prompts(
        raw_prompts, tokenizer, max_samples=args.max_samples,
        max_prompt_tokens=args.max_prompt_tokens, device=device,
    )
    print(f"Tokenized {len(prompts_list)} prompts")

    # GPU warmup.
    warm_ids, warm_attn = make_padded_batch(prompts_list[:1], pad_token_id, device)
    _ = vanilla_ar_generate_batched(
        target=target, input_ids=warm_ids, attention_mask=warm_attn,
        eos_token_ids=eos_token_ids, max_new_tokens=8,
    )
    torch.cuda.empty_cache()

    results = []
    for B, mts in configs:
        print(f"\n=== B={B}, mts={mts} ===")
        van_total_time, van_total_out = 0.0, 0
        v7_total_time, v7_total_out = 0.0, 0
        v7_acc_lengths = []
        for chunk in chunk_rows_list(prompts_list, B):
            ids, attn = make_padded_batch(chunk, pad_token_id, device)
            v_out = vanilla_ar_generate_batched(
                target=target, input_ids=ids, attention_mask=attn,
                eos_token_ids=eos_token_ids, max_new_tokens=args.max_new_tokens,
            )
            van_total_time += v_out.total_decode_time
            van_total_out += sum(v_out.num_output_tokens)

            s_out = dflash_generate_batched(
                draft=draft, target=target, input_ids=ids, attention_mask=attn,
                mask_token_id=mask_token_id, eos_token_ids=eos_token_ids,
                max_new_tokens=args.max_new_tokens, block_size=block_size,
                max_tree_size=mts, expand_k=args.expand_k, temperature=0.0,
            )
            v7_total_time += s_out.total_decode_time
            v7_total_out += sum(s_out.num_output_tokens)
            for lst in s_out.acceptance_lengths_per_elem:
                v7_acc_lengths.extend(lst)
            del v_out, s_out, ids, attn
            torch.cuda.empty_cache()

        van_tps = van_total_out / van_total_time
        v7_tps = v7_total_out / v7_total_time
        speedup = v7_tps / van_tps
        tau = float(np.mean(v7_acc_lengths)) if v7_acc_lengths else float("nan")
        row = {
            "B": B, "mts": mts, "tau": round(tau, 2),
            "speedup": round(speedup, 2), "v7_tps": round(v7_tps, 1),
            "van_tps": round(van_tps, 1),
            "v7_time_s": round(v7_total_time, 2),
            "van_time_s": round(van_total_time, 2),
        }
        results.append(row)
        print(f"  → tau={row['tau']}, v7_tps={row['v7_tps']}, van_tps={row['van_tps']}, speedup={row['speedup']}×")

    print("\n=== SUMMARY (best mts per B) ===")
    by_B = {}
    for r in results:
        by_B.setdefault(r["B"], []).append(r)
    for B in sorted(by_B):
        rows = by_B[B]
        best = max(rows, key=lambda r: r["speedup"])
        print(f"B={B}: best mts={best['mts']}, speedup={best['speedup']}× (tau={best['tau']})")

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"configs": configs, "results": results}, f, indent=2)
    print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
