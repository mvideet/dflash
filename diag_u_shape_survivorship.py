"""Disambiguate: is the U-shape real or a heap-survivorship artifact?

Method: for every node visited in the tree (across many rounds), record:
  - depth d of the node
  - drafter joint log-prob along the path TO this node (sum of log_q's from
    anchor to parent of this node, i.e. node's path-prob)
  - whether target.argmax(parent) == realized child token (= "accepted" at
    this depth, conditional on path being the realized chain)

Then bucket by (depth, log-prob bin) and compute P(accept | depth, bin).

If U-shape persists ACROSS log-prob bins (i.e., at any fixed log-prob, deep
depths have higher accept rate than middle depths), the phenomenon is real.

If U-shape disappears when controlling for log-prob (i.e., conditional on
log-prob, accept rate is monotone-decreasing in depth), the apparent U was
selection effect — only high-prob deep nodes survived the heap.

We use full_tree_dumps to get every leaf in every round, with full path-of-ranks.
Note: the heap still selects which nodes appear, but we now condition on
joint log-prob rather than just measuring rank-0-chain survival.
"""
import argparse, json, os
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import DFlashDraftModel, load_and_process_dataset
from dflash_batched import dflash_generate_batched, vanilla_ar_generate_batched
from benchmark_batched import tokenize_prompts, make_padded_batch, chunk_rows_list, _resolve_mts

SCHEDULE = {1: 64, 2: 32, 4: 32, 8: 16}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--datasets", type=str, default="math500,aime24,gsm8k")
    parser.add_argument("--output-json", type=str, default="logs/u_shape_survivorship.json")
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

    out = {"datasets": []}
    for ds_name in args.datasets.split(","):
        ds_name = ds_name.strip()
        print(f"\n========== {ds_name} ==========", flush=True)
        try:
            ds = load_and_process_dataset(ds_name).shuffle(seed=0)
        except Exception as e:
            print(f"  Failed: {e}"); continue
        raw = [ds[i]["turns"][0] for i in range(min(len(ds), args.max_samples * 4))]
        prompts_list = tokenize_prompts(
            raw, tokenizer, max_samples=args.max_samples,
            max_prompt_tokens=args.max_prompt_tokens, device=device,
        )
        if not prompts_list: continue

        warm_ids, warm_attn = make_padded_batch(prompts_list[:1], pad_id, device)
        _ = vanilla_ar_generate_batched(
            target=target, input_ids=warm_ids, attention_mask=warm_attn,
            eos_token_ids=eos_ids, max_new_tokens=8,
        )
        torch.cuda.empty_cache()

        B = 4
        M = _resolve_mts(B, 16, SCHEDULE)

        # Per-depth, per-log-prob-bin counts.
        # Bins: log-prob along path to parent of node. We discretize into
        # 8 bins from quantiles of observed log-probs.
        all_observations = []  # list of (depth, path_logprob_to_parent, accepted_bool)

        for chunk in chunk_rows_list(prompts_list, B):
            ids, attn = make_padded_batch(chunk, pad_id, device)
            s_out = dflash_generate_batched(
                draft=draft, target=target, input_ids=ids, attention_mask=attn,
                mask_token_id=mid, eos_token_ids=eos_ids,
                max_new_tokens=args.max_new_tokens, block_size=block_size,
                max_tree_size=M, expand_k=8, temperature=0.0,
                log_full_tree_dump=True,
                ewma_adaptive=True, ewma_min_M=8, ewma_max_ek=8, ewma_min_ek=2,
            )
            # Each round dumps full tree. For each leaf:
            #   - path-of-ranks shows each child's rank in the draft top-K
            #   - accept_n = how many initial depths matched target.argmax
            # We approximate "drafter joint log-prob to parent of depth-d node"
            # via the SUM of the path-ranks-converted-to-log-prob.
            # (Rank-r at depth d ≈ log(p_d^(r)). Without exact log-probs we use
            # rank as proxy: lower rank → higher log-prob.)
            # We use rank-sum-to-d as a coarse proxy for path log-prob:
            #   bin 0 (highest prob) = all-rank-0 path
            #   bin K = path sum of ranks to here
            for d_dump in s_out.full_tree_dumps:
                for leaf in d_dump["leaves"]:
                    ranks = leaf["ranks"]
                    accept_n = leaf["accept_n"]
                    cum_rank = 0
                    for d_idx, r in enumerate(ranks):
                        depth = d_idx + 1   # tree depth of this child
                        # accept at this child depth = (d_idx < accept_n)
                        accepted = (d_idx < accept_n)
                        # path log-prob proxy = rank sum over ancestors (lower=better)
                        # we count parent's rank-sum (excluding current child rank)
                        all_observations.append({
                            "depth": depth, "path_rank_sum_to_parent": cum_rank,
                            "accepted": int(accepted),
                            "this_rank": r,
                        })
                        cum_rank += r
            del s_out, ids, attn; torch.cuda.empty_cache()

        # Aggregate: bucket by depth and rank-sum quantile.
        # Compute global quantiles of path_rank_sum_to_parent, bucket into 4 bins.
        rank_sums = [o["path_rank_sum_to_parent"] for o in all_observations]
        if not rank_sums:
            continue
        q = np.percentile(rank_sums, [25, 50, 75])
        def bin_for_sum(s):
            if s <= q[0]: return 0   # high-prob (low rank-sum)
            if s <= q[1]: return 1
            if s <= q[2]: return 2
            return 3                  # low-prob

        # Per-depth, per-bin: count of observations + count accepted.
        cell = defaultdict(lambda: [0, 0])  # (depth, bin) -> [count, accepted]
        for o in all_observations:
            b = bin_for_sum(o["path_rank_sum_to_parent"])
            d = o["depth"]
            cell[(d, b)][0] += 1
            cell[(d, b)][1] += o["accepted"]

        # Compute per-depth accept rate, AND per-depth-per-bin accept rate.
        per_depth_overall = defaultdict(lambda: [0, 0])
        for (d, b), (c, a) in cell.items():
            per_depth_overall[d][0] += c
            per_depth_overall[d][1] += a

        print(f"\n  {ds_name}: per-depth stratified P(accept) "
              f"(rank-sum bin: 0=highest-prob, 3=lowest)", flush=True)
        print(f"  {'depth':>5} {'overall':>8} {'bin0':>7} {'bin1':>7} {'bin2':>7} {'bin3':>7} "
              f"{'n0':>5} {'n1':>5} {'n2':>5} {'n3':>5}", flush=True)
        per_depth_table = {}
        for d in range(1, block_size):
            cnt_overall, acc_overall = per_depth_overall.get(d, (0, 0))
            if cnt_overall < 5: continue
            row = {"depth": d, "n_overall": cnt_overall, "p_overall": acc_overall / cnt_overall}
            line = f"  {d:>5} {acc_overall/cnt_overall:>7.3f}"
            for b in range(4):
                c, a = cell.get((d, b), (0, 0))
                p = a / c if c > 0 else float("nan")
                row[f"bin{b}_n"] = c
                row[f"bin{b}_p"] = p if c > 0 else None
                line += f" {p:>6.3f}" if c > 0 else f" {'--':>6}"
            for b in range(4):
                c, _ = cell.get((d, b), (0, 0))
                line += f" {c:>5}"
            per_depth_table[d] = row
            print(line, flush=True)

        # Verdict per dataset:
        # Look at bin 0 (high-prob): is its P(accept) U-shaped or monotone-decreasing?
        # If high-prob bin still shows U-shape → real
        # If high-prob bin is monotone-decreasing → survivorship
        bin0_curve = []
        for d in range(1, block_size):
            row = per_depth_table.get(d, {})
            p = row.get("bin0_p")
            n = row.get("bin0_n", 0)
            if p is not None and n >= 5:
                bin0_curve.append((d, p, n))

        if len(bin0_curve) >= 5:
            # Check if there's recovery at deep depths
            mid_p = np.mean([p for d, p, n in bin0_curve if 4 <= d <= 9])
            deep_p = np.mean([p for d, p, n in bin0_curve if 12 <= d <= 15])
            shallow_p = np.mean([p for d, p, n in bin0_curve if d <= 3])
            print(f"\n  bin0 (high-prob) shape:")
            print(f"    shallow d=1-3: {shallow_p:.3f}")
            print(f"    middle  d=4-9: {mid_p:.3f}")
            print(f"    deep   d=12-15: {deep_p:.3f}")
            if deep_p > mid_p + 0.05:
                verdict = "U-SHAPE PRESERVED — phenomenon is real (not survivorship)"
            elif deep_p < mid_p - 0.05:
                verdict = "SURVIVORSHIP CONFIRMED — deep accuracy was selection bias"
            else:
                verdict = "INCONCLUSIVE — no clear recovery or decline at deep"
            print(f"  → {verdict}", flush=True)
        else:
            verdict = "Insufficient bin0 samples"

        out["datasets"].append({
            "dataset": ds_name,
            "n_observations": len(all_observations),
            "rank_sum_quantiles": list(q),
            "per_depth_stratified": per_depth_table,
            "verdict": verdict,
        })

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
