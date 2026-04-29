"""Compare tree shapes at decay=1.0 vs decay=0.85, identical prompts/seeds.

Diagnostic dimensions:
  1. Per-round n_accepted distribution (does decay improve or shift it?)
  2. Best-leaf deviation-depth distribution (does decay shift toward shallower deviations?)
  3. Best-leaf path-of-ranks (do decayed best leaves use rank-2+ children differently?)
  4. Per-depth draft top-1 accuracy (= P(target.argmax(parent on leaf) == draft top-1 at masked pos d))
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


def run_at_decay(target, draft, prompts_list, B, M, block_size, mid, eos_ids, pad_id,
                 mode_kwargs, decay, max_new_tokens, device):
    """Run one decay setting, return aggregated diagnostics."""
    full_tree_dumps_all = []
    accept_lens_all = []
    rejection_ranks_all = []
    total_t, total_o = 0.0, 0
    for chunk in chunk_rows_list(prompts_list, B):
        ids, attn = make_padded_batch(chunk, pad_id, device)
        s_out = dflash_generate_batched(
            draft=draft, target=target, input_ids=ids, attention_mask=attn,
            mask_token_id=mid, eos_token_ids=eos_ids,
            max_new_tokens=max_new_tokens, block_size=block_size,
            max_tree_size=M, expand_k=8, temperature=0.0,
            log_rejection_ranks=True,
            log_full_tree_dump=True,
            score_decay=decay,
            **mode_kwargs,
        )
        total_t += s_out.total_decode_time
        total_o += sum(s_out.num_output_tokens)
        full_tree_dumps_all.extend(s_out.full_tree_dumps)
        rejection_ranks_all.extend(s_out.rejection_ranks)
        for lst in s_out.acceptance_lengths_per_elem:
            for n_plus_1 in lst:
                accept_lens_all.append(max(0, n_plus_1 - 1))
        del s_out, ids, attn
        torch.cuda.empty_cache()
    return {
        "tps": total_o / total_t,
        "full_tree_dumps": full_tree_dumps_all,
        "accept_lens": accept_lens_all,
        "rejection_ranks": rejection_ranks_all,
    }


def aggregate(dump_a, dump_b, label_a, label_b):
    """Diff between two decay runs."""
    # Best-leaf deviation depth distribution
    def deviation_dist(dumps):
        c = Counter()
        for d in dumps:
            best_leaf_idx = d["best_leaf_idx"]
            best_leaf = next((l for l in d["leaves"] if l["leaf_idx"] == best_leaf_idx), None)
            if best_leaf is None:
                c["NA"] += 1
            else:
                dev = best_leaf["deviation_depth"]
                if dev < 0:
                    c["pure_rank0"] += 1
                else:
                    c[f"dev@{dev+1}"] += 1
        return c

    # n_accepted distribution
    def accept_dist(accs):
        c = Counter()
        for n in accs:
            c[n] += 1
        return c

    # Per-round delta in n_accepted (paired by round_idx)
    n_a = aggregate_n_accepted(dump_a["full_tree_dumps"])
    n_b = aggregate_n_accepted(dump_b["full_tree_dumps"])

    return {
        f"{label_a}_dev_dist": dict(deviation_dist(dump_a["full_tree_dumps"])),
        f"{label_b}_dev_dist": dict(deviation_dist(dump_b["full_tree_dumps"])),
        f"{label_a}_accept_dist": dict(accept_dist(dump_a["accept_lens"])),
        f"{label_b}_accept_dist": dict(accept_dist(dump_b["accept_lens"])),
        f"{label_a}_mean_tau": float(np.mean(dump_a["accept_lens"])),
        f"{label_b}_mean_tau": float(np.mean(dump_b["accept_lens"])),
        f"{label_a}_n_rejections": len(dump_a["rejection_ranks"]),
        f"{label_b}_n_rejections": len(dump_b["rejection_ranks"]),
    }


def aggregate_n_accepted(dumps):
    return [d["best_n_accepted"] for d in dumps]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--output-json", type=str, default="logs/decay_compare.json")
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

    # B=4 with ewma_adaptive (where decay=0.85 had cleanest signal)
    B = 4
    M = _resolve_mts(B, 16, SCHEDULE)
    mode_kwargs = dict(ewma_adaptive=True, ewma_min_M=8, ewma_max_ek=8, ewma_min_ek=2)

    print(f"\n=== B={B} M={M}, comparing decay=1.0 vs decay=0.85 ===", flush=True)
    print("Running decay=1.00...", flush=True)
    res_baseline = run_at_decay(target, draft, prompts_list, B, M, block_size, mid, eos_ids, pad_id,
                                  mode_kwargs, 1.0, args.max_new_tokens, device)
    print(f"  tps={res_baseline['tps']:.1f}, mean_tau={np.mean(res_baseline['accept_lens']):.2f}", flush=True)

    torch.manual_seed(0)  # reset for identical sampling
    print("Running decay=0.85...", flush=True)
    res_decay = run_at_decay(target, draft, prompts_list, B, M, block_size, mid, eos_ids, pad_id,
                               mode_kwargs, 0.85, args.max_new_tokens, device)
    print(f"  tps={res_decay['tps']:.1f}, mean_tau={np.mean(res_decay['accept_lens']):.2f}", flush=True)

    agg = aggregate(res_baseline, res_decay, "decay_1", "decay_085")

    # --- Print diagnostic comparison ---
    print("\n=== Best-leaf deviation distribution (decay=1.0 vs decay=0.85) ===", flush=True)
    devs_1 = agg["decay_1_dev_dist"]
    devs_085 = agg["decay_085_dev_dist"]
    all_keys = sorted(set(list(devs_1.keys()) + list(devs_085.keys())))
    n1 = sum(devs_1.values())
    n085 = sum(devs_085.values())
    print(f"  {'kind':>15}  {'decay=1':>10}  {'decay=0.85':>12}", flush=True)
    for k in all_keys:
        c1 = devs_1.get(k, 0)
        c085 = devs_085.get(k, 0)
        print(f"  {k:>15}  {100*c1/max(n1,1):>9.1f}%  {100*c085/max(n085,1):>11.1f}%", flush=True)

    print("\n=== Best-leaf accept-len distribution ===", flush=True)
    print(f"  {'n':>3}  {'decay=1':>10}  {'decay=0.85':>12}", flush=True)
    accept_1 = agg["decay_1_accept_dist"]
    accept_085 = agg["decay_085_accept_dist"]
    nr1 = sum(accept_1.values())
    nr085 = sum(accept_085.values())
    for n in range(block_size):
        c1 = accept_1.get(n, 0)
        c085 = accept_085.get(n, 0)
        if c1 == 0 and c085 == 0:
            continue
        print(f"  {n:>3}  {100*c1/max(nr1,1):>9.1f}%  {100*c085/max(nr085,1):>11.1f}%", flush=True)

    # --- Per-depth draft top-1 accuracy (averaged over all leaves visited) ---
    # FM6 test: P(rank-0 child = target.argmax(parent)) per depth
    def per_depth_top1_accuracy(dumps):
        """For each depth, count: (a) rounds with a rank-0-child at this depth,
           (b) rounds where that rank-0 child was accepted (= target.argmax matched)."""
        # We have leaf info: for each leaf, ranks[d] for d in 0..len-1.
        # accept_n = how many INITIAL depths matched.
        # We want: at each depth d, P(rank-0 child accepted), aggregated over rounds.
        # To compute: for each round, for the rank-0 chain (= leaf with all ranks=0 if exists),
        # accept_n along that chain tells us how many depths the rank-0 chain accepted.
        depth_alive = Counter()    # rank-0 chain reached this depth
        depth_accepted = Counter() # rank-0 chain matched at this depth
        for d in dumps:
            r0_leaf = next((l for l in d["leaves"] if all(r == 0 for r in l["ranks"])), None)
            if r0_leaf is None:
                continue
            n_acc = r0_leaf["accept_n"]
            chain_len = len(r0_leaf["ranks"])
            for depth_idx in range(chain_len):
                depth_alive[depth_idx + 1] += 1
                if depth_idx < n_acc:
                    depth_accepted[depth_idx + 1] += 1
        return depth_alive, depth_accepted

    a_alive, a_acc = per_depth_top1_accuracy(res_baseline["full_tree_dumps"])
    print("\n=== FM6 test: rank-0 chain top-1 accept rate by depth (decay=1.0) ===", flush=True)
    print(f"  {'depth':>5}  {'n':>5}  {'P(accept)':>10}", flush=True)
    for d in sorted(a_alive.keys()):
        if a_alive[d] < 5:
            continue
        p = a_acc[d] / a_alive[d]
        print(f"  {d:>5}  {a_alive[d]:>5}  {p:>9.3f}", flush=True)

    out = {
        "B": B, "M": M, "block_size": block_size,
        "decay_1_tps": res_baseline["tps"],
        "decay_085_tps": res_decay["tps"],
        "decay_1_mean_tau": float(np.mean(res_baseline["accept_lens"])),
        "decay_085_mean_tau": float(np.mean(res_decay["accept_lens"])),
        "deviation_dist": agg,
        "depth_alive_baseline": dict(a_alive),
        "depth_accepted_baseline": dict(a_acc),
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
