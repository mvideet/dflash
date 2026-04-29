"""Rejection-rank diagnostic.

At every rejection point, log target.argmax's rank in draft q_i. Histogram
the result. Tells us:
  - rank=0 (top-1)        : impossible (top-1 is what got rejected — sanity check)
  - rank=1-2 (top 2-3)    : heap could find with K=3
  - rank=3-7 (top 4-8)    : heap could find with K=8 (current)
  - rank=8-31             : heap couldn't find at any reasonable K
  - rank=32+              : long tail

If most rejections are at rank ≤ 7, the heap has the right tokens — issue is
elsewhere. If a substantial chunk is at rank 8+, *selective K expansion* (idea
#6) is live.
"""
import argparse, json, os
from collections import Counter
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import DFlashDraftModel, load_and_process_dataset
from dflash_batched import dflash_generate_batched, vanilla_ar_generate_batched
from benchmark_batched import tokenize_prompts, make_padded_batch, chunk_rows_list, _resolve_mts


SCHEDULE = {1: 64, 2: 32, 4: 32, 8: 16, 16: 16, 32: 16}


CONFIGS = [
    ("B=1 baseline",  1, dict()),
    ("B=4 ewma",      4, dict(ewma_adaptive=True, ewma_min_M=8, ewma_max_ek=8, ewma_min_ek=2)),
    ("B=8 specdecpp", 8, dict(specdecpp_threshold=0.05)),
]


def rank_bin(r):
    if r == 0: return "0"
    if r == 1: return "1"
    if r == 2: return "2"
    if r <= 4: return "3-4"
    if r <= 7: return "5-7"
    if r <= 15: return "8-15"
    if r <= 31: return "16-31"
    if r <= 63: return "32-63"
    return "64+"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--output-json", type=str, default="logs/rejection_rank.json")
    args = parser.parse_args()

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
    mid = draft.mask_token_id

    ds = load_and_process_dataset("math500").shuffle(seed=0)
    raw = [ds[i]["turns"][0] for i in range(min(len(ds), args.max_samples * 4))]
    prompts_list = tokenize_prompts(
        raw, tokenizer, max_samples=args.max_samples,
        max_prompt_tokens=args.max_prompt_tokens, device=device,
    )
    print(f"Tokenized {len(prompts_list)} prompts.", flush=True)

    out = {"configs": []}
    for label, B, mode_kwargs in CONFIGS:
        M = _resolve_mts(B, 16, SCHEDULE)
        all_ranks = []  # list of (depth, rank)
        for chunk in chunk_rows_list(prompts_list, B):
            ids, attn = make_padded_batch(chunk, pad_id, device)
            s_out = dflash_generate_batched(
                draft=draft, target=target, input_ids=ids, attention_mask=attn,
                mask_token_id=mid, eos_token_ids=eos_ids,
                max_new_tokens=args.max_new_tokens, block_size=block_size,
                max_tree_size=M, expand_k=8, temperature=0.0,
                log_rejection_ranks=True,
                **mode_kwargs,
            )
            all_ranks.extend(s_out.rejection_ranks)
            del s_out, ids, attn
            torch.cuda.empty_cache()

        # Aggregate.
        rank_only = [r for _, r in all_ranks]
        bin_count = Counter(rank_bin(r) for r in rank_only)
        depth_x_rank = Counter(all_ranks)  # (depth, rank) → count

        total = max(len(rank_only), 1)
        # Print bin histogram.
        print(f"\n=== {label} (M={M}, n_rejections={len(rank_only)}) ===", flush=True)
        print(f"  median rank={int(np.median(rank_only)) if rank_only else 0}, "
              f"mean rank={float(np.mean(rank_only)):.2f}, "
              f"p90={int(np.percentile(rank_only, 90)) if rank_only else 0}", flush=True)
        order = ["0", "1", "2", "3-4", "5-7", "8-15", "16-31", "32-63", "64+"]
        for k in order:
            v = bin_count.get(k, 0)
            pct = 100 * v / total
            bar = "#" * int(round(60 * v / total))
            print(f"    rank {k:>5}: {pct:>5.1f}% ({v:>3}) {bar}", flush=True)

        # Top-K reach analysis.
        in_top_2 = sum(1 for r in rank_only if r <= 1)
        in_top_4 = sum(1 for r in rank_only if r <= 3)
        in_top_8 = sum(1 for r in rank_only if r <= 7)
        in_top_16 = sum(1 for r in rank_only if r <= 15)
        in_top_32 = sum(1 for r in rank_only if r <= 31)
        print(f"\n  reachable cumulative:", flush=True)
        print(f"    K≥2  catches {100*in_top_2/total:.1f}% of rejections", flush=True)
        print(f"    K≥4  catches {100*in_top_4/total:.1f}% of rejections", flush=True)
        print(f"    K≥8  catches {100*in_top_8/total:.1f}% of rejections (current)", flush=True)
        print(f"    K≥16 catches {100*in_top_16/total:.1f}% of rejections", flush=True)
        print(f"    K≥32 catches {100*in_top_32/total:.1f}% of rejections", flush=True)

        # Rank by depth (which depths have heavy-tail failures).
        print(f"\n  mean rank per depth:", flush=True)
        depth_groups: dict = {}
        for d, r in all_ranks:
            depth_groups.setdefault(d, []).append(r)
        for d in sorted(depth_groups.keys()):
            rs = depth_groups[d]
            print(f"    depth {d:>2}: n={len(rs):>3}, "
                  f"median={int(np.median(rs))}, "
                  f"%>7={100*sum(1 for r in rs if r > 7)/len(rs):>5.1f}%", flush=True)

        out["configs"].append({
            "label": label, "B": B, "M": M,
            "n_rejections": len(rank_only),
            "rank_histogram": dict(bin_count),
            "in_top_2_pct": round(100*in_top_2/total, 2),
            "in_top_4_pct": round(100*in_top_4/total, 2),
            "in_top_8_pct": round(100*in_top_8/total, 2),
            "in_top_16_pct": round(100*in_top_16/total, 2),
            "in_top_32_pct": round(100*in_top_32/total, 2),
            "all_ranks": all_ranks,
        })

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
