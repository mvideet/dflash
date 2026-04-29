"""Verify each claimed v7 failure mode with direct measurement.

Failure modes claimed:
  FM1: Rank-0 rejections (22-35%) caused by heap walking non-rank-0 sibling chain
  FM2: Late-depth heavy tails — depths 10-15 have higher rejection-rank
  FM3: Out-of-K rejections (rank ≥ 8): 6-15% of rejections
  FM4: Bimodal accept distribution (~30% n=15 + spread n=2..13)
  FM5: At B=8 M=block_size, tree degenerates to chain (vs heap structure)

This harness runs at B={1, 4, 8} with current best configs, captures everything,
then aggregates per-failure-mode evidence.
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
    ("B=1", 1, dict()),
    ("B=4", 4, dict(ewma_adaptive=True, ewma_min_M=8, ewma_max_ek=8, ewma_min_ek=2)),
    ("B=8", 8, dict(specdecpp_threshold=0.05)),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--output-json", type=str, default="logs/failure_mode_verify.json")
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

    out = {"configs": [], "block_size": block_size}
    for label, B, mode_kwargs in CONFIGS:
        M = _resolve_mts(B, 16, SCHEDULE)
        print(f"\n=== {label} M={M} ===", flush=True)
        rejection_ranks_list = []      # for FM2, FM3
        path_trajs_list = []            # for FM1
        accept_lens_list = []           # for FM4 (n_accepted distribution)
        tree_node_counts = []           # for FM5
        for chunk in chunk_rows_list(prompts_list, B):
            ids, attn = make_padded_batch(chunk, pad_id, device)
            s_out = dflash_generate_batched(
                draft=draft, target=target, input_ids=ids, attention_mask=attn,
                mask_token_id=mid, eos_token_ids=eos_ids,
                max_new_tokens=args.max_new_tokens, block_size=block_size,
                max_tree_size=M, expand_k=8, temperature=0.0,
                log_rejection_ranks=True,
                log_path_trajectory=True,
                **mode_kwargs,
            )
            rejection_ranks_list.extend(s_out.rejection_ranks)
            path_trajs_list.extend(s_out.path_trajectories)
            for lst in s_out.acceptance_lengths_per_elem:
                accept_lens_list.extend([n_plus_1 - 1 for n_plus_1 in lst])  # n = n_plus_1 - 1
            tree_node_counts.extend(s_out.tree_node_counts)
            del s_out, ids, attn
            torch.cuda.empty_cache()

        # ---- FM1: rank=0 rejection mechanism ----
        # For each round (in path_trajs_list), check:
        #   - what was the best leaf's path-of-ranks?
        #   - was best leaf the rank-0 chain (all-rank-0)?
        #   - rejection rank at best leaf
        n_rounds = len(path_trajs_list)
        # Classification
        leaf_kind_counts = Counter()      # 'pure_rank0' / 'mixed' / 'no_rank0'
        rej0_with_mixed = 0  # count of (rejection_rank == 0) AND best leaf used non-rank-0 child
        rej0_total = 0       # count of rejection_rank == 0 events
        rejN_total = 0       # any rejection
        rank0_could_have_won_count = 0   # rank-0 chain accept length > best leaf accept length
        rank0_tied_count = 0
        rank0_short_count = 0
        for tj in path_trajs_list:
            best_path = tj["best_path_ranks"]
            best_n = tj["best_n_accepted"]
            r0_n = tj["rank0_n_accepted"]
            rej_r = tj["rejection_rank_at_best"]
            best_leaf_idx = tj["best_leaf_idx"]
            # Classify best leaf
            ranks_in_accepted = best_path[:max(0, best_n)]  # ranks at depths 1..best_n
            if len(ranks_in_accepted) == 0:
                leaf_kind = "trivial"
            elif all(r == 0 for r in ranks_in_accepted):
                leaf_kind = "pure_rank0"
            else:
                leaf_kind = "mixed"
            leaf_kind_counts[leaf_kind] += 1
            # Rank-0 chain comparison
            if r0_n > best_n:
                rank0_could_have_won_count += 1
            elif r0_n == best_n:
                rank0_tied_count += 1
            else:
                rank0_short_count += 1
            # Rejection-cause classification
            if rej_r >= 0:
                rejN_total += 1
                if rej_r == 0:
                    rej0_total += 1
                    if leaf_kind == "mixed":
                        rej0_with_mixed += 1

        # ---- FM2, FM3: rejection rank distribution by depth ----
        rej_by_depth = {}  # depth -> list of ranks
        for d, r in rejection_ranks_list:
            rej_by_depth.setdefault(d, []).append(r)
        rank_buckets = {"0": 0, "1-2": 0, "3-7": 0, "8-15": 0, "16+": 0}
        for _, r in rejection_ranks_list:
            if r == 0: rank_buckets["0"] += 1
            elif r <= 2: rank_buckets["1-2"] += 1
            elif r <= 7: rank_buckets["3-7"] += 1
            elif r <= 15: rank_buckets["8-15"] += 1
            else: rank_buckets["16+"] += 1
        n_rej = max(sum(rank_buckets.values()), 1)
        # Out-of-K: rank ≥ 8
        out_of_k_pct = 100 * (rank_buckets["8-15"] + rank_buckets["16+"]) / n_rej

        # ---- FM4: accept length distribution ----
        accept_buckets = Counter()
        for n in accept_lens_list:
            n = max(0, min(int(n), block_size - 1))
            accept_buckets[n] += 1
        n_rounds_total = max(sum(accept_buckets.values()), 1)

        # ---- FM5: tree degeneration to chain ----
        node_count_dist = Counter(tree_node_counts)

        config_data = {
            "label": label, "B": B, "M": M, "n_rounds": n_rounds,
            "fm1_leaf_kind_counts": dict(leaf_kind_counts),
            "fm1_pure_rank0_pct": round(100 * leaf_kind_counts["pure_rank0"] / max(n_rounds, 1), 1),
            "fm1_mixed_pct": round(100 * leaf_kind_counts["mixed"] / max(n_rounds, 1), 1),
            "fm1_rank0_could_have_won": rank0_could_have_won_count,
            "fm1_rank0_tied": rank0_tied_count,
            "fm1_rank0_short": rank0_short_count,
            "fm1_rej0_total": rej0_total,
            "fm1_rej0_with_mixed_best_leaf": rej0_with_mixed,
            "fm1_rej0_pct_with_mixed": round(100 * rej0_with_mixed / max(rej0_total, 1), 1),
            "fm1_rej0_fraction_of_all_rejections": round(100 * rej0_total / max(rejN_total, 1), 1),
            "fm2_rejection_rank_buckets_pct": {k: round(100 * v / n_rej, 1) for k, v in rank_buckets.items()},
            "fm2_rejection_rank_by_depth": {
                d: {"n": len(rs), "median": int(np.median(rs)), "pct_above_7": round(100 * sum(1 for r in rs if r > 7) / len(rs), 1)}
                for d, rs in sorted(rej_by_depth.items())
            },
            "fm3_out_of_k_pct": round(out_of_k_pct, 1),
            "fm4_accept_distribution_pct": {
                str(n): round(100 * accept_buckets[n] / n_rounds_total, 1)
                for n in range(block_size)
            },
            "fm4_full_block_pct": round(100 * accept_buckets[block_size - 1] / n_rounds_total, 1),
            "fm4_total_rounds": n_rounds_total,
            "fm5_tree_node_counts": dict(node_count_dist),
        }
        out["configs"].append(config_data)

        # Console summary per-config
        print(f"\n  --- FM1 (rank=0 rejection mechanism) ---", flush=True)
        print(f"    Total rejections: {rejN_total}", flush=True)
        print(f"    Rank-0 rejections: {rej0_total} ({config_data['fm1_rej0_fraction_of_all_rejections']:.1f}% of all)", flush=True)
        print(f"    Of those rank-0 rejections, % where best leaf walked non-rank-0 child: "
              f"{config_data['fm1_rej0_pct_with_mixed']:.1f}%", flush=True)
        print(f"    Best leaf is pure-rank-0 chain: {config_data['fm1_pure_rank0_pct']:.1f}% of rounds", flush=True)
        print(f"    Best leaf is mixed (≥1 non-rank-0): {config_data['fm1_mixed_pct']:.1f}% of rounds", flush=True)
        print(f"    Rank-0 chain could have accepted MORE than best leaf in: "
              f"{rank0_could_have_won_count}/{n_rounds} rounds "
              f"({100*rank0_could_have_won_count/max(n_rounds,1):.1f}%)", flush=True)

        print(f"\n  --- FM2/FM3 (rejection rank distribution) ---", flush=True)
        for k, v in rank_buckets.items():
            print(f"    rank {k:>5}: {100*v/n_rej:>5.1f}% ({v})", flush=True)
        print(f"    Out-of-K (rank ≥ 8): {out_of_k_pct:.1f}%", flush=True)

        print(f"\n  --- FM4 (accept distribution) ---", flush=True)
        print(f"    n=15 (full block): {config_data['fm4_full_block_pct']:.1f}%", flush=True)
        # Print buckets where rate > 0.5%
        for n in range(block_size):
            pct = config_data['fm4_accept_distribution_pct'][str(n)]
            if pct > 0.5:
                print(f"    n={n:>2}: {pct:>5.1f}%", flush=True)

        print(f"\n  --- FM5 (tree shape) ---", flush=True)
        for nc in sorted(node_count_dist.keys()):
            cnt = node_count_dist[nc]
            print(f"    nodes={nc}: {cnt} rounds ({100*cnt/n_rounds_total:.1f}%)", flush=True)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
