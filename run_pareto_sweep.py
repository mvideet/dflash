#!/usr/bin/env python3
"""
Pareto sweep: τ (mean accepted length) vs. actual trie node count
for all four tree-building strategies across multiple benchmarks.

Outputs:
  logs/pareto_results.json       — structured results (incrementally saved)
  logs/pareto_tau_vs_nodes.pdf   — publication figure (one panel per benchmark)
  logs/pareto_tau_vs_nodes.png   — same, rasterized

Usage:
  python run_pareto_sweep.py \
    --model-name-or-path Qwen/Qwen3-4B \
    --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16

  # Override defaults:
  python run_pareto_sweep.py \
    --model-name-or-path Qwen/Qwen3-4B \
    --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 \
    --benchmarks gsm8k humaneval alpaca \
    --budgets 4 8 16 32 48 64 \
    --max-samples 50
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "paper"))
from plot_config import TREE_LABELS_INT as TREE_LABELS, TREE_COLORS_INT as TREE_COLORS, TREE_MARKERS_INT as TREE_MARKERS

from model import DFlashDraftModel, load_and_process_dataset
from model.freq_vocab import load_freq_mapping, get_reduced_lm_head
from benchmark import dflash_generate

DEFAULT_BENCHMARKS = ["mt-bench", "humaneval", "math500"]
DEFAULT_BUDGETS = [4, 8, 16, 32, 48, 64]

# Dense budgets for fair v2-vs-v4 Pareto comparison.
# v2 produces nodes = max_tree_size + 1, so its range is [2, 257].
# v4 produces nodes ~ 2*max_tree_size at low budgets, ~2*mts at high,
# so budget=1 gives ~5 nodes and budget=128 gives ~256 nodes.
# This grid ensures both methods are sampled across [5, 260] nodes.
V2_V4_BUDGETS = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256]


# ---------------------------------------------------------------------------
# Core sweep logic
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_one_config(
    draft_model,
    target,
    tokenizer,
    dataset,
    tree_version,
    max_tree_size,
    block_size,
    max_new_tokens,
    expand_k,
    top_k,
    freq_used_tokens,
    freq_reduced_weight,
    freq_reduced_bias,
):
    """Run one (tree_version, max_tree_size) config across all samples.

    Returns dict with mean_tau, std_tau, mean_nodes, std_nodes, n_steps.
    """
    all_taus = []
    all_nodes = []

    for idx in range(len(dataset)):
        instance = dataset[idx]
        messages = []
        for user_content in instance["turns"]:
            messages.append({"role": "user", "content": user_content})
            try:
                input_text = tokenizer.apply_chat_template(
                    messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False,
                )
            except TypeError:
                input_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            input_ids = tokenizer.encode(
                input_text, return_tensors="pt"
            ).to(target.device)

            result = dflash_generate(
                model=draft_model,
                target=target,
                input_ids=input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=max_new_tokens,
                block_size=block_size,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=0.0,
                chain_attention=True,
                top_k=top_k,
                dynamic_branching=True,
                tree_version=tree_version,
                theta_uni=0.9,
                theta_bi=0.3,
                theta_tri=0.1,
                max_tree_size=max_tree_size,
                expand_k=expand_k,
                freq_used_tokens=freq_used_tokens,
                freq_reduced_weight=freq_reduced_weight,
                freq_reduced_bias=freq_reduced_bias,
            )

            all_taus.extend(result.acceptance_lengths)
            all_nodes.extend(result.tree_node_counts)

            generated_ids = result.output_ids[0, result.num_input_tokens:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})

    return {
        "mean_tau": float(np.mean(all_taus)) if all_taus else 0.0,
        "std_tau": float(np.std(all_taus)) if all_taus else 0.0,
        "mean_nodes": float(np.mean(all_nodes)) if all_nodes else 0.0,
        "std_nodes": float(np.std(all_nodes)) if all_nodes else 0.0,
        "n_steps": len(all_taus),
        "n_samples": len(dataset),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_pareto(all_results, output_dir):
    """Publication-quality τ vs. node-count Pareto figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    benchmarks = list(all_results.keys())
    n = len(benchmarks)

    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), squeeze=False)
    axes = axes[0]

    for ax, bench in zip(axes, benchmarks):
        for tv in sorted(TREE_LABELS.keys()):
            key = f"v{tv}"
            if key not in all_results[bench]:
                continue
            entries = all_results[bench][key]
            nodes = [e["mean_nodes"] for e in entries]
            taus = [e["mean_tau"] for e in entries]
            sems = [
                e["std_tau"] / np.sqrt(max(e["n_steps"], 1)) for e in entries
            ]

            ax.errorbar(
                nodes, taus, yerr=sems,
                label=TREE_LABELS[tv],
                color=TREE_COLORS[tv],
                marker=TREE_MARKERS[tv],
                markersize=7,
                linewidth=1.8,
                capsize=3,
                capthick=1.2,
            )

        ax.set_xlabel("Mean trie node count (verification cost)", fontsize=11)
        ax.set_ylabel("Mean accepted length (\u03c4)", fontsize=11)
        ax.set_title(bench, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    pdf_path = os.path.join(output_dir, "pareto_tau_vs_nodes.pdf")
    png_path = os.path.join(output_dir, "pareto_tau_vs_nodes.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {pdf_path} and {png_path}")


def print_summary_table(all_results):
    """Compact table for quick inspection / copy-paste into LaTeX."""
    hdr = (
        f"{'Benchmark':<12} {'Method':<22} {'Budget':>6} "
        f"{'tau':>8} {'+-':>6} {'Nodes':>8} {'+-':>6} {'Steps':>6}"
    )
    print(f"\n{'=' * len(hdr)}")
    print(hdr)
    print(f"{'=' * len(hdr)}")
    for bench, versions in all_results.items():
        for key in sorted(versions.keys()):
            tv = int(key[1])
            for entry in versions[key]:
                print(
                    f"{bench:<12} {TREE_LABELS[tv]:<22} "
                    f"{entry['max_tree_size']:>6} "
                    f"{entry['mean_tau']:>8.3f} {entry['std_tau']:>6.2f} "
                    f"{entry['mean_nodes']:>8.1f} {entry['std_nodes']:>6.1f} "
                    f"{entry['n_steps']:>6}"
                )
        print(f"{'-' * len(hdr)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Pareto sweep: tau vs. trie node count across tree strategies",
    )
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument(
        "--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS,
        help=f"Datasets to evaluate (default: {DEFAULT_BENCHMARKS})",
    )
    parser.add_argument(
        "--budgets", nargs="+", type=int, default=None,
        help=f"max_tree_size values to sweep (default: {DEFAULT_BUDGETS})",
    )
    parser.add_argument(
        "--dense-v2-v4", action="store_true",
        help="Use dense budget grid optimized for v2-vs-v4 Pareto comparison "
             "(21 budgets from 1 to 256, only v2 and v4)",
    )
    parser.add_argument(
        "--tree-versions", nargs="+", type=int, default=None,
        choices=[1, 2, 3, 4],
        help="Tree versions to include (default: 1 2 3 4)",
    )
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--expand-k", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--freq-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="logs")
    args = parser.parse_args()

    if args.dense_v2_v4:
        if args.budgets is None:
            args.budgets = V2_V4_BUDGETS
        if args.tree_versions is None:
            args.tree_versions = [2, 4]
        print(f"Dense v2-vs-v4 mode: {len(args.budgets)} budgets, versions {args.tree_versions}")
    else:
        if args.budgets is None:
            args.budgets = DEFAULT_BUDGETS
        if args.tree_versions is None:
            args.tree_versions = [1, 2, 3, 4]

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0")

    def has_flash_attn():
        try:
            import flash_attn  # noqa: F401
            return True
        except ImportError:
            return False

    attn_impl = "flash_attention_2" if has_flash_attn() else "sdpa"

    print("Loading target model ...")
    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
    ).to(device).eval()

    print("Loading draft model ...")
    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
    ).to(device).eval()

    block_size = args.block_size if args.block_size else draft_model.block_size
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    freq_used_tokens = None
    freq_reduced_weight = None
    freq_reduced_bias = None
    if args.freq_path:
        _, freq_used_tokens = load_freq_mapping(args.freq_path)
        freq_reduced_weight, freq_reduced_bias = get_reduced_lm_head(
            target.lm_head, freq_used_tokens, device,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    json_name = "pareto_v2_vs_v4_dense.json" if args.dense_v2_v4 else "pareto_results.json"
    json_path = os.path.join(args.output_dir, json_name)

    all_results: dict = {}
    total_configs = len(args.benchmarks) * len(args.tree_versions) * len(args.budgets)
    config_idx = 0
    sweep_start = time.time()

    for bench in args.benchmarks:
        print(f"\n{'=' * 60}")
        print(f"  Benchmark: {bench}")
        print(f"{'=' * 60}")

        dataset = load_and_process_dataset(bench)
        if len(dataset) > args.max_samples:
            dataset = dataset.shuffle(seed=0).select(range(args.max_samples))
        print(f"  Samples: {len(dataset)}")

        all_results[bench] = {}

        for tv in args.tree_versions:
            key = f"v{tv}"
            all_results[bench][key] = []

            for mts in args.budgets:
                config_idx += 1
                elapsed = time.time() - sweep_start
                eta = (elapsed / config_idx) * (total_configs - config_idx)

                print(
                    f"\n  [{config_idx}/{total_configs}] {TREE_LABELS[tv]}  "
                    f"max_tree_size={mts}  (ETA: {eta / 60:.0f} min)"
                )

                try:
                    stats = run_one_config(
                        draft_model=draft_model,
                        target=target,
                        tokenizer=tokenizer,
                        dataset=dataset,
                        tree_version=tv,
                        max_tree_size=mts,
                        block_size=block_size,
                        max_new_tokens=args.max_new_tokens,
                        expand_k=args.expand_k,
                        top_k=args.top_k,
                        freq_used_tokens=freq_used_tokens,
                        freq_reduced_weight=freq_reduced_weight,
                        freq_reduced_bias=freq_reduced_bias,
                    )
                except Exception as e:
                    print(f"  ERROR: {e}")
                    stats = {
                        "mean_tau": 0.0, "std_tau": 0.0,
                        "mean_nodes": 0.0, "std_nodes": 0.0,
                        "n_steps": 0, "n_samples": 0, "error": str(e),
                    }

                stats["max_tree_size"] = mts
                stats["tree_version"] = tv
                all_results[bench][key].append(stats)

                print(
                    f"  -> tau = {stats['mean_tau']:.3f} +/- {stats['std_tau']:.3f}  |  "
                    f"nodes = {stats['mean_nodes']:.1f} +/- {stats['std_nodes']:.1f}  |  "
                    f"{stats['n_steps']} steps"
                )

                torch.cuda.empty_cache()

                with open(json_path, "w") as f:
                    json.dump({"meta": vars(args), "results": all_results}, f, indent=2)

    total_time = time.time() - sweep_start
    print(f"\n\nSweep complete in {total_time / 60:.1f} minutes")
    print(f"Results saved to {json_path}")

    print_summary_table(all_results)

    try:
        plot_pareto(all_results, args.output_dir)
    except Exception as e:
        print(f"\nPlot generation failed: {e}")
        print("Install matplotlib to generate plots: pip install matplotlib")


if __name__ == "__main__":
    main()
