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


def run_one_batch_size(
    target, draft, prompts_list: List[torch.Tensor], B: int,
    max_new_tokens: int, max_tree_size: int, expand_k: int, block_size: int,
    mask_token_id: int, pad_token_id: int, eos_token_ids: List[int], device,
):
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
            max_tree_size=max_tree_size, expand_k=expand_k, temperature=0.0,
        )
        v7_total_time += s_out.total_decode_time
        v7_total_out += sum(s_out.num_output_tokens)
        for lst in s_out.acceptance_lengths_per_elem:
            v7_acc_lengths_all.extend(lst)
        v7_tree_nodes_all.extend(s_out.tree_node_counts)

        del v_out, s_out, ids, attn
        torch.cuda.empty_cache()

    vanilla_tps = vanilla_total_out / vanilla_total_time
    v7_tps = v7_total_out / v7_total_time
    speedup = v7_tps / vanilla_tps
    tau = float(np.mean(v7_acc_lengths_all)) if v7_acc_lengths_all else float("nan")
    avg_nodes = float(np.mean(v7_tree_nodes_all)) if v7_tree_nodes_all else float("nan")

    return {
        "batch_size": B,
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
    parser.add_argument("--max-tree-size", type=int, default=128)
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
            r = run_one_batch_size(
                target=target, draft=draft, prompts_list=prompts_list, B=B,
                max_new_tokens=args.max_new_tokens, max_tree_size=args.max_tree_size,
                expand_k=args.expand_k, block_size=block_size,
                mask_token_id=mask_token_id, pad_token_id=pad_token_id,
                eos_token_ids=eos_token_ids, device=device,
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM at B={B}: stopping. ({e})")
            torch.cuda.empty_cache()
            break
        print(f"  vanilla AR  : {r['vanilla_tps']:8.1f} tok/s  ({r['vanilla_total_out_tokens']} tok / {r['vanilla_total_time_s']:.2f}s)")
        print(f"  v7 ddtree   : {r['v7_tps']:8.1f} tok/s  ({r['v7_total_out_tokens']} tok / {r['v7_total_time_s']:.2f}s)")
        print(f"  tau         : {r['tau']:.2f}, avg_nodes: {r['avg_nodes']:.1f}")
        print(f"  speedup     : {r['speedup']:.2f}×")
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
