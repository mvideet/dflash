#!/usr/bin/env python3
"""
Generate publication figures from experiment results.

Produces:
  paper/fig_pareto.pdf          — tau vs node count (Pareto curves), one panel per benchmark
  paper/fig_main_bar.pdf        — grouped bar chart of tau and speedup across all 11 datasets
"""

import json
import os
import re
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")
OUT_DIR = SCRIPT_DIR

TREE_LABELS = {
    "v1": "Threshold (v1)",
    "v2": "EAGLE-2 (v2)",
    "v3": "Best-First (v3)",
    "v4": "Prefix-Aware (v4, ours)",
}
TREE_COLORS = {"v1": "#d62728", "v2": "#1f77b4", "v3": "#2ca02c", "v4": "#9467bd"}
TREE_MARKERS = {"v1": "o", "v2": "s", "v3": "^", "v4": "D"}

BENCH_DISPLAY = {
    "mt-bench": "MT-Bench",
    "humaneval": "HumanEval",
    "math500": "MATH-500",
}


# ======================================================================
# Figure 1: Pareto curves (tau vs node count)
# ======================================================================
def fig_pareto():
    path = os.path.join(LOGS_DIR, "pareto_results.json")
    with open(path) as f:
        data = json.load(f)
    results = data["results"]

    benchmarks = list(results.keys())
    n = len(benchmarks)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2), squeeze=False)
    axes = axes[0]

    for ax, bench in zip(axes, benchmarks):
        for vk in ["v1", "v2", "v3", "v4"]:
            if vk not in results[bench]:
                continue
            entries = results[bench][vk]
            nodes = [e["mean_nodes"] for e in entries]
            taus = [e["mean_tau"] for e in entries]
            sems = [e["std_tau"] / np.sqrt(max(e["n_steps"], 1)) for e in entries]

            ax.errorbar(
                nodes, taus, yerr=sems,
                label=TREE_LABELS[vk],
                color=TREE_COLORS[vk],
                marker=TREE_MARKERS[vk],
                markersize=7, linewidth=2.0,
                capsize=3, capthick=1.2,
                zorder=5 if vk == "v4" else 3,
            )

        ax.set_xlabel("Mean trie nodes (verification cost)", fontsize=12)
        ax.set_ylabel("Mean accepted length ($\\bar{\\tau}$)", fontsize=12)
        ax.set_title(BENCH_DISPLAY.get(bench, bench), fontsize=14, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(labelsize=10)

    fig.tight_layout(pad=1.5)
    for ext in ["pdf", "png"]:
        out = os.path.join(OUT_DIR, f"fig_pareto.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  Saved {out}")
    plt.close(fig)


# ======================================================================
# Figure 2: Grouped bar chart (all 11 datasets)
# ======================================================================
def parse_summary():
    """Parse paper_four_trees_summary.txt into structured data."""
    path = os.path.join(LOGS_DIR, "paper_four_trees_summary.txt")
    rows = []
    with open(path) as f:
        for line in f:
            m = re.match(
                r"^(\S+)\s+n=\d+\s+temp=[\d.]+\s+(\S+)\s+mts=\d+\s+"
                r"speedup=([\d.]+)\s+avg_accept=([\d.]+)\s+avg_nodes=([\d.]+)\s+exit=0",
                line.strip(),
            )
            if m:
                rows.append({
                    "bench": m.group(1),
                    "method": m.group(2),
                    "speedup": float(m.group(3)),
                    "tau": float(m.group(4)),
                    "nodes": float(m.group(5)),
                })
    return rows


def fig_main_bar():
    rows = parse_summary()
    if not rows:
        print("  No summary data found, skipping bar chart.")
        return

    bench_order = [
        "gsm8k", "math500", "aime24", "aime25",
        "alpaca", "mt-bench",
        "humaneval", "mbpp", "lbpp", "swe-bench", "livecodebench",
    ]
    bench_labels = [
        "GSM8K", "MATH-500", "AIME'24", "AIME'25",
        "Alpaca", "MT-Bench",
        "HumanEval", "MBPP", "LBPP", "SWE-Bench", "LiveCode",
    ]

    method_order = ["v1_thresh", "v2_eagle2", "v3_bestfirst", "v4_prefixaware"]
    method_labels = ["Threshold (v1)", "EAGLE-2 (v2)", "Best-First (v3)", "Prefix-Aware (v4)"]
    method_colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    method_hatches = ["//", "\\\\", "xx", ""]

    lookup = {}
    for r in rows:
        lookup[(r["bench"], r["method"])] = r

    n_bench = len(bench_order)
    n_methods = len(method_order)
    x = np.arange(n_bench)
    width = 0.19

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True)

    for i, (meth, label, color, hatch) in enumerate(
        zip(method_order, method_labels, method_colors, method_hatches)
    ):
        taus = []
        speedups = []
        for b in bench_order:
            r = lookup.get((b, meth))
            taus.append(r["tau"] if r else 0)
            speedups.append(r["speedup"] if r else 0)

        offset = (i - n_methods / 2 + 0.5) * width
        bars1 = ax1.bar(
            x + offset, taus, width,
            label=label, color=color, hatch=hatch,
            edgecolor="white", linewidth=0.5,
        )
        bars2 = ax2.bar(
            x + offset, speedups, width,
            label=label, color=color, hatch=hatch,
            edgecolor="white", linewidth=0.5,
        )

    ax1.set_ylabel("Mean accepted length ($\\bar{\\tau}$)", fontsize=12)
    ax1.legend(fontsize=9, ncol=4, loc="upper left", framealpha=0.9)
    ax1.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax1.tick_params(labelsize=10)

    ax2.set_ylabel("Speedup over autoregressive", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bench_labels, rotation=35, ha="right", fontsize=10)
    ax2.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax2.tick_params(labelsize=10)

    # Category separators
    for ax in [ax1, ax2]:
        ax.axvline(x=3.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axvline(x=5.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    ax1.text(1.5, ax1.get_ylim()[1] * 0.95, "Math", ha="center", fontsize=10, fontstyle="italic", alpha=0.6)
    ax1.text(4.5, ax1.get_ylim()[1] * 0.95, "Chat", ha="center", fontsize=10, fontstyle="italic", alpha=0.6)
    ax1.text(8.0, ax1.get_ylim()[1] * 0.95, "Code", ha="center", fontsize=10, fontstyle="italic", alpha=0.6)

    fig.tight_layout(pad=1.5)
    for ext in ["pdf", "png"]:
        out = os.path.join(OUT_DIR, f"fig_main_bar.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating Figure 1: Pareto curves ...")
    fig_pareto()
    print("Generating Figure 2: Main bar chart ...")
    fig_main_bar()
    print("Done.")
