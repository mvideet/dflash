"""End-to-end batched benchmark on a real dataset (math500 by default).

For each batch size B, partitions the dataset into ceil(N/B) micro-batches
and runs both vanilla AR and v7 (DDTree) on each. Measures aggregate
tokens/sec and computes the speedup of v7 over vanilla AR.

Prompts are filtered+truncated to a uniform length so we can avoid prefill
padding. Per-element divergence is fully exercised at the tree+verify+trim
layers (different draft logits → different trees → different accept lengths).

Usage:
    python benchmark_batched.py --dataset math500 --max-samples 16 \
        --batch-sizes 1,2,4,8 --max-new-tokens 256 --max-tree-size 128 \
        --prompt-len 100
"""
import argparse
import json
import os
from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import DFlashDraftModel, load_and_process_dataset
from dflash_batched import (
    dflash_generate_batched,
    vanilla_ar_generate_batched,
)


def tokenize_prompts(
    prompts: List[str], tokenizer, max_samples: int, max_prompt_tokens: int, device,
) -> List[torch.Tensor]:
    """Tokenize each prompt with chat template. Drop those longer than
    max_prompt_tokens (truncating would chop the chat-template suffix and
    break tau). Return a list of variable-length [1, S_b] long tensors.
    """
    rows = []
    for p in prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        ids = tokenizer.encode(text, return_tensors="pt").to(device)
        if ids.shape[1] <= max_prompt_tokens:
            rows.append(ids)
            if len(rows) >= max_samples:
                break
    if not rows:
        raise RuntimeError(f"No prompt has length <= {max_prompt_tokens}.")
    return rows


def make_padded_batch(rows: List[torch.Tensor], pad_token_id: int, device):
    """Right-pad a list of [1, S_b] tensors to [B, S_max] with attention_mask."""
    B = len(rows)
    S_max = max(int(t.shape[1]) for t in rows)
    ids = torch.full((B, S_max), pad_token_id, dtype=torch.long, device=device)
    attn = torch.zeros(B, S_max, dtype=torch.long, device=device)
    for b, t in enumerate(rows):
        L = int(t.shape[1])
        ids[b, :L] = t[0]
        attn[b, :L] = 1
    return ids, attn


def chunk_rows_list(rows: List[torch.Tensor], B: int):
    for i in range(0, len(rows), B):
        yield rows[i:i + B]


def _resolve_mts(B: int, default_mts: int, schedule: dict) -> int:
    """Pick max_tree_size for a given B from a {B: mts} schedule. Falls back to
    default_mts if no schedule entry matches; otherwise uses the largest B' ≤ B
    in the schedule."""
    if not schedule:
        return default_mts
    eligible = [b for b in schedule if b <= B]
    if not eligible:
        return default_mts
    return schedule[max(eligible)]


def run_one_batch_size(
    target, draft, prompts_list: List[torch.Tensor], B: int,
    max_new_tokens: int, max_tree_size: int, expand_k: int, block_size: int,
    mask_token_id: int, pad_token_id: int, eos_token_ids: List[int], device,
    mts_schedule: dict = None,
    online_mts: bool = False,
    online_mts_candidates: tuple = (8, 16, 32, 64, 128),
    pwls: bool = False,
    bqat_threshold: float = 0.0,
    cppr: bool = False,
    cppr_lambda: float = 0.5,
    pdrr_k1: float = 0.0, pdrr_k2: float = 0.0, pdrr_k3: float = 0.0,
    cgdb_shallow_depth: int = 0, cgdb_high_thresh: float = 0.0,
    cgdb_low_thresh: float = 0.0, cgdb_mid_k: int = 0,
    ewma_adaptive: bool = False, ewma_decay: float = 0.8,
    ewma_min_M: int = 12, ewma_min_ek: int = 2, ewma_max_ek: int = 8,
    anchor_signal: str = "none", anchor_min_M: int = 32, anchor_gamma: float = 1.0,
    use_target_q1: bool = False,
    adaedl_B: bool = False, adaedl_min_M: int = 8, adaedl_max_M: int = 128,
    entropy_width: bool = False, entropy_width_min_ek: int = 1, entropy_width_max_ek: int = 8,
):
    eff_mts = _resolve_mts(B, max_tree_size, mts_schedule or {})
    per_step_M_all = []
    vanilla_total_time = 0.0
    vanilla_total_out = 0
    v7_total_time = 0.0
    v7_total_out = 0
    v7_acc_lengths_all = []
    v7_tree_nodes_all = []

    for chunk in chunk_rows_list(prompts_list, B):
        ids, attn = make_padded_batch(chunk, pad_token_id, device)

        v_out = vanilla_ar_generate_batched(
            target=target, input_ids=ids, attention_mask=attn,
            eos_token_ids=eos_token_ids, max_new_tokens=max_new_tokens,
        )
        vanilla_total_time += v_out.total_decode_time
        vanilla_total_out += sum(v_out.num_output_tokens)

        s_out = dflash_generate_batched(
            draft=draft, target=target, input_ids=ids, attention_mask=attn,
            mask_token_id=mask_token_id, eos_token_ids=eos_token_ids,
            max_new_tokens=max_new_tokens, block_size=block_size,
            max_tree_size=eff_mts, expand_k=expand_k, temperature=0.0,
            online_mts=online_mts, online_mts_candidates=online_mts_candidates,
            pwls=pwls, bqat_threshold=bqat_threshold,
            cppr=cppr, cppr_lambda=cppr_lambda,
            pdrr_k1=pdrr_k1, pdrr_k2=pdrr_k2, pdrr_k3=pdrr_k3,
            cgdb_shallow_depth=cgdb_shallow_depth,
            cgdb_high_thresh=cgdb_high_thresh,
            cgdb_low_thresh=cgdb_low_thresh,
            cgdb_mid_k=cgdb_mid_k,
            ewma_adaptive=ewma_adaptive, ewma_decay=ewma_decay,
            ewma_min_M=ewma_min_M, ewma_min_ek=ewma_min_ek, ewma_max_ek=ewma_max_ek,
            anchor_signal=anchor_signal, anchor_min_M=anchor_min_M, anchor_gamma=anchor_gamma,
            use_target_q1=use_target_q1,
            adaedl_B=adaedl_B, adaedl_min_M=adaedl_min_M, adaedl_max_M=adaedl_max_M,
            entropy_width=entropy_width,
            entropy_width_min_ek=entropy_width_min_ek,
            entropy_width_max_ek=entropy_width_max_ek,
        )
        v7_total_time += s_out.total_decode_time
        v7_total_out += sum(s_out.num_output_tokens)
        for lst in s_out.acceptance_lengths_per_elem:
            v7_acc_lengths_all.extend(lst)
        v7_tree_nodes_all.extend(s_out.tree_node_counts)
        if hasattr(s_out, "per_step_M_choices"):
            per_step_M_all.extend(s_out.per_step_M_choices)

        del v_out, s_out, ids, attn
        torch.cuda.empty_cache()

    vanilla_tps = vanilla_total_out / vanilla_total_time
    v7_tps = v7_total_out / v7_total_time
    speedup = v7_tps / vanilla_tps
    tau = float(np.mean(v7_acc_lengths_all)) if v7_acc_lengths_all else float("nan")
    avg_nodes = float(np.mean(v7_tree_nodes_all)) if v7_tree_nodes_all else float("nan")

    M_hist = {}
    if per_step_M_all:
        for m in per_step_M_all:
            M_hist[m] = M_hist.get(m, 0) + 1
        M_mean = float(np.mean(per_step_M_all))
    else:
        M_mean = float(eff_mts)
    return {
        "batch_size": B,
        "max_tree_size": eff_mts,
        "online_mts": online_mts,
        "M_mean": round(M_mean, 2),
        "M_histogram": M_hist,
        "vanilla_tps": vanilla_tps,
        "v7_tps": v7_tps,
        "speedup": speedup,
        "tau": tau,
        "avg_nodes": avg_nodes,
        "vanilla_total_time_s": vanilla_total_time,
        "v7_total_time_s": v7_total_time,
        "vanilla_total_out_tokens": vanilla_total_out,
        "v7_total_out_tokens": v7_total_out,
    }


def make_plot(results, out_png: str, meta: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bs = [r["batch_size"] for r in results]
    speedups = [r["speedup"] for r in results]
    vanilla_tps = [r["vanilla_tps"] for r in results]
    v7_tps = [r["v7_tps"] for r in results]
    tau_avg = float(np.mean([r["tau"] for r in results]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(bs, speedups, marker="o", color="C0", linewidth=2)
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(bs)
    ax.set_xticklabels([str(b) for b in bs])
    ax.set_xlabel("batch size")
    ax.set_ylabel("v7 ddtree speedup over vanilla AR")
    ax.set_title(f"End-to-end speedup (tau≈{tau_avg:.2f}, B_tree={meta['max_tree_size']})")
    ax.grid(True, alpha=0.3)
    for x, y in zip(bs, speedups):
        ax.annotate(f"{y:.2f}×", xy=(x, y), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=8)

    ax = axes[1]
    ax.plot(bs, vanilla_tps, marker="s", label="vanilla AR", color="C1", linewidth=2)
    ax.plot(bs, v7_tps, marker="o", label="v7 ddtree", color="C0", linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(bs)
    ax.set_xticklabels([str(b) for b in bs])
    ax.set_xlabel("batch size")
    ax.set_ylabel("tokens / sec (aggregate)")
    ax.set_title("End-to-end throughput")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.suptitle(
        f"DFlash v7 batched benchmark — {meta['model']} on {meta['dataset']}",
        fontsize=11,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  → {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--draft-name-or-path", type=str, default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--dataset", type=str, default="math500")
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-tree-size", type=int, default=128,
                        help="Default max_tree_size if no schedule matches.")
    parser.add_argument("--mts-schedule", type=str, default="",
                        help="B-aware schedule for max_tree_size, e.g. '1:64,2:32,4:32,8:16,16:16'. "
                             "For each B we use the largest scheduled B' ≤ B.")
    parser.add_argument("--online-mts", action="store_true",
                        help="Online goodput-driven M* prediction (per step). Overrides "
                             "the schedule's M but uses it as the initial guess.")
    parser.add_argument("--online-mts-candidates", type=str, default="8,16,32,64,128",
                        help="Comma-sep candidate Ms for online M* selection.")
    parser.add_argument("--pwls", action="store_true",
                        help="Posterior-Weighted Leaf Selection — break ties on bonus conf.")
    parser.add_argument("--bqat-threshold", type=float, default=0.0,
                        help="Bonus-Quality-Aware Truncation threshold. If bonus conf < t, "
                             "truncate by 1. 0.0 disables.")
    parser.add_argument("--cppr", action="store_true",
                        help="Cross-Path Posterior Re-Ranking — score leaves by length + "
                             "lambda * mean target log-prob along accepted prefix.")
    parser.add_argument("--cppr-lambda", type=float, default=0.5,
                        help="CPPR weight on mean target log-prob (default 0.5).")
    # v8 score adjustments (CGDB + PDRR; ported from v7 sweep's headline config).
    parser.add_argument("--pdrr-k1", type=float, default=0.0,
                        help="PDRR boost at child_depth = dev_depth + 1.")
    parser.add_argument("--pdrr-k2", type=float, default=0.0,
                        help="PDRR boost at child_depth = dev_depth + 2.")
    parser.add_argument("--pdrr-k3", type=float, default=0.0,
                        help="PDRR boost at child_depth = dev_depth + 3.")
    parser.add_argument("--cgdb-shallow-depth", type=int, default=0,
                        help="CGDB: full expand_k always at depths ≤ this.")
    parser.add_argument("--cgdb-high-thresh", type=float, default=0.0,
                        help="CGDB: at depth > shallow, p_path < this → mid_k.")
    parser.add_argument("--cgdb-low-thresh", type=float, default=0.0,
                        help="CGDB: at depth > shallow, p_path < this → 1 (argmax tail).")
    parser.add_argument("--cgdb-mid-k", type=int, default=0,
                        help="CGDB: mid-tier expand_k.")
    parser.add_argument("--v8", action="store_true",
                        help="Enable v8 with the canonical CGDB+PDRR config from "
                             "paper/results_apr27_b224_headline.md (overrides individual flags).")
    # EWMA adaptive (ported from benchmark.py --adaptive-block).
    parser.add_argument("--ewma-adaptive", action="store_true",
                        help="EWMA-based adaptive M and expand_k.")
    parser.add_argument("--ewma-decay", type=float, default=0.8)
    parser.add_argument("--ewma-min-M", type=int, default=12)
    parser.add_argument("--ewma-min-ek", type=int, default=2)
    parser.add_argument("--ewma-max-ek", type=int, default=8)
    # Anchor-entropy / draft-conf adaptive (ported from --adaptive-budget-mode).
    parser.add_argument("--anchor-signal", type=str, default="none",
                        choices=["none", "entropy", "draft-conf"],
                        help="Per-step forward-signal-driven M sizing.")
    parser.add_argument("--anchor-min-M", type=int, default=32)
    parser.add_argument("--anchor-gamma", type=float, default=1.0)
    # Idea 1e — exact target depth-1 distribution reuse.
    parser.add_argument("--use-target-q1", action="store_true",
                        help="Replace draft's q_1 with target's logits at bonus (lossless).")
    # Idea 1b — AdaEDL closed-form adaptive M.
    parser.add_argument("--adaedl", action="store_true",
                        help="AdaEDL: M_t scales with (1 - prev anchor conf).")
    parser.add_argument("--adaedl-min-M", type=int, default=8)
    parser.add_argument("--adaedl-max-M", type=int, default=128)
    # Idea 1c — entropy-guided per-position expand_k.
    parser.add_argument("--entropy-width", action="store_true",
                        help="Per-position expand_k from current draft entropy at that depth.")
    parser.add_argument("--entropy-width-min-ek", type=int, default=1)
    parser.add_argument("--entropy-width-max-ek", type=int, default=8)
    parser.add_argument("--expand-k", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=256,
                        help="Drop prompts longer than this many tokens (truncating would chop "
                             "the chat-template suffix and break tau). Shorter prompts are kept "
                             "intact and right-padded across the batch with attention_mask.")
    parser.add_argument("--output-json", type=str, default="logs/batched_benchmark.json")
    parser.add_argument("--output-png", type=str, default="paper/fig/speedup_vs_batch_size_batched.png")
    args = parser.parse_args()

    bs_list = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    mts_schedule = {}
    if args.mts_schedule:
        for tok in args.mts_schedule.split(","):
            b, m = tok.split(":")
            mts_schedule[int(b)] = int(m)
    # v8 canonical config (from paper/results_apr27_b224_headline.md).
    if args.v8:
        if args.pdrr_k1 == 0.0:
            args.pdrr_k1 = 0.5
        if args.pdrr_k2 == 0.0:
            args.pdrr_k2 = 0.25
        if args.cgdb_shallow_depth == 0:
            args.cgdb_shallow_depth = 4
        if args.cgdb_high_thresh == 0.0:
            args.cgdb_high_thresh = 0.1
        if args.cgdb_low_thresh == 0.0:
            args.cgdb_low_thresh = 0.01
        if args.cgdb_mid_k == 0:
            args.cgdb_mid_k = 4
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
    block_size = args.block_size if args.block_size is not None else draft.block_size
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    mask_token_id = draft.mask_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_token_ids = [tokenizer.eos_token_id]
    print(f"block_size={block_size}, max_tree_size={args.max_tree_size}, expand_k={args.expand_k}")

    ds = load_and_process_dataset(args.dataset)
    ds = ds.shuffle(seed=0)
    raw_prompts = [ds[i]["turns"][0] for i in range(min(len(ds), args.max_samples * 4))]
    prompts_list = tokenize_prompts(
        raw_prompts, tokenizer, max_samples=args.max_samples,
        max_prompt_tokens=args.max_prompt_tokens, device=device,
    )
    lens = [int(t.shape[1]) for t in prompts_list]
    print(f"Tokenized {len(prompts_list)} prompts (lens min/median/max = {min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)})")

    print("Warming up GPU...")
    warm_ids, warm_attn = make_padded_batch(prompts_list[:1], pad_token_id, device)
    _ = vanilla_ar_generate_batched(
        target=target, input_ids=warm_ids, attention_mask=warm_attn,
        eos_token_ids=eos_token_ids, max_new_tokens=8,
    )
    torch.cuda.empty_cache()

    results = []
    for B in bs_list:
        print(f"\n=== batch_size = {B} ===")
        try:
            online_cands = tuple(int(x) for x in args.online_mts_candidates.split(",") if x.strip())
            r = run_one_batch_size(
                target=target, draft=draft, prompts_list=prompts_list, B=B,
                max_new_tokens=args.max_new_tokens, max_tree_size=args.max_tree_size,
                expand_k=args.expand_k, block_size=block_size,
                mask_token_id=mask_token_id, pad_token_id=pad_token_id,
                eos_token_ids=eos_token_ids, device=device,
                mts_schedule=mts_schedule,
                online_mts=args.online_mts, online_mts_candidates=online_cands,
                pwls=args.pwls, bqat_threshold=args.bqat_threshold,
                cppr=args.cppr, cppr_lambda=args.cppr_lambda,
                pdrr_k1=args.pdrr_k1, pdrr_k2=args.pdrr_k2, pdrr_k3=args.pdrr_k3,
                cgdb_shallow_depth=args.cgdb_shallow_depth,
                cgdb_high_thresh=args.cgdb_high_thresh,
                cgdb_low_thresh=args.cgdb_low_thresh,
                cgdb_mid_k=args.cgdb_mid_k,
                ewma_adaptive=args.ewma_adaptive, ewma_decay=args.ewma_decay,
                ewma_min_M=args.ewma_min_M, ewma_min_ek=args.ewma_min_ek,
                ewma_max_ek=args.ewma_max_ek,
                anchor_signal=args.anchor_signal,
                anchor_min_M=args.anchor_min_M, anchor_gamma=args.anchor_gamma,
                use_target_q1=args.use_target_q1,
                adaedl_B=args.adaedl,
                adaedl_min_M=args.adaedl_min_M, adaedl_max_M=args.adaedl_max_M,
                entropy_width=args.entropy_width,
                entropy_width_min_ek=args.entropy_width_min_ek,
                entropy_width_max_ek=args.entropy_width_max_ek,
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM at B={B}: stopping. ({e})")
            torch.cuda.empty_cache()
            break
        print(f"  mts (effective): {r['max_tree_size']} (online={r['online_mts']}, mean M={r['M_mean']})")
        if r.get('M_histogram'):
            print(f"  M histogram    : {dict(sorted(r['M_histogram'].items()))}")
        print(f"  vanilla AR     : {r['vanilla_tps']:8.1f} tok/s  ({r['vanilla_total_out_tokens']} tok / {r['vanilla_total_time_s']:.2f}s)")
        print(f"  v7 ddtree      : {r['v7_tps']:8.1f} tok/s  ({r['v7_total_out_tokens']} tok / {r['v7_total_time_s']:.2f}s)")
        print(f"  tau            : {r['tau']:.2f}, avg_nodes: {r['avg_nodes']:.1f}")
        print(f"  speedup        : {r['speedup']:.2f}×")
        results.append(r)

    if not results:
        print("No results — aborting.")
        return

    meta = {
        "model": args.model_name_or_path,
        "draft": args.draft_name_or_path,
        "dataset": args.dataset,
        "max_samples": len(prompts_list),
        "max_new_tokens": args.max_new_tokens,
        "max_tree_size": args.max_tree_size,
        "expand_k": args.expand_k,
        "block_size": block_size,
        "max_prompt_tokens": args.max_prompt_tokens,
        "prompt_lens": [int(t.shape[1]) for t in prompts_list],
    }
    payload = {"meta": meta, "results": results}
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {args.output_json}")
    make_plot(results, out_png=args.output_png, meta=meta)


if __name__ == "__main__":
    main()
