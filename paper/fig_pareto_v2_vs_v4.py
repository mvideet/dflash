#!/usr/bin/env python3
"""
Generate focused Pareto comparison figure: v4 (Prefix-Aware) vs v2 (EAGLE-2).

Produces:
  paper/fig_pareto_v2_vs_v4.pdf
  paper/fig_pareto_v2_vs_v4.png
  logs/pareto_v2_vs_v4.pdf
  logs/pareto_v2_vs_v4.png
"""

import json
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")

from plot_config import BENCH_DISPLAY

V2_STYLE = dict(
    color="#1f77b4",
    marker="s",
    markersize=7,
    linewidth=2.2,
    capsize=4,
    capthick=1.2,
    label="v2 (EAGLE-2)",
    zorder=3,
)
V4_STYLE = dict(
    color="#9467bd",
    marker="D",
    markersize=7,
    linewidth=2.2,
    capsize=4,
    capthick=1.2,
    label="v4 (Prefix-Aware, ours)",
    zorder=5,
)


def load_data(dense=False):
    """Load results, preferring the dense v2-vs-v4 sweep if available."""
    dense_path = os.path.join(LOGS_DIR, "pareto_v2_vs_v4_dense.json")
    default_path = os.path.join(LOGS_DIR, "pareto_results.json")

    if dense and os.path.exists(dense_path):
        path = dense_path
        print(f"  Using dense sweep data: {path}")
    else:
        path = default_path
        print(f"  Using standard sweep data: {path}")

    with open(path) as f:
        data = json.load(f)
    return data["results"]


def fig_pareto_v2_vs_v4(dense=False):
    results = load_data(dense=dense)
    benchmarks = list(results.keys())
    n = len(benchmarks)

    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.8), squeeze=False)
    axes = axes[0]

    for ax, bench in zip(axes, benchmarks):
        for vk, style in [("v2", V2_STYLE), ("v4", V4_STYLE)]:
            if vk not in results[bench]:
                continue
            entries = results[bench][vk]
            nodes = [e["mean_nodes"] for e in entries]
            taus = [e["mean_tau"] for e in entries]
            sems = [e["std_tau"] / np.sqrt(max(e["n_steps"], 1)) for e in entries]

            ax.errorbar(nodes, taus, yerr=sems, **style)

        # Annotate delta-tau at matched budgets
        if "v2" in results[bench] and "v4" in results[bench]:
            v2_entries = results[bench]["v2"]
            v4_entries = results[bench]["v4"]
            for v4e in v4_entries:
                v2_match = next(
                    (e for e in v2_entries if e["max_tree_size"] == v4e["max_tree_size"]),
                    None,
                )
                if v2_match is None:
                    continue
                delta = v4e["mean_tau"] - v2_match["mean_tau"]
                if abs(delta) < 0.01:
                    continue
                color = "#2ca02c" if delta > 0 else "#d62728"
                ax.annotate(
                    f"{delta:+.2f}",
                    xy=(v4e["mean_nodes"], v4e["mean_tau"]),
                    xytext=(0, 12),
                    textcoords="offset points",
                    fontsize=7.5,
                    fontweight="bold",
                    color=color,
                    ha="center",
                    va="bottom",
                )
                # Dashed connector between matched-budget points
                ax.plot(
                    [v2_match["mean_nodes"], v4e["mean_nodes"]],
                    [v2_match["mean_tau"], v4e["mean_tau"]],
                    color="gray",
                    linewidth=0.6,
                    linestyle=":",
                    alpha=0.4,
                    zorder=1,
                )

        ax.set_xlabel("Mean verification nodes", fontsize=12)
        ax.set_ylabel(r"Mean accepted length ($\bar{\tau}$)", fontsize=12)
        ax.set_title(BENCH_DISPLAY.get(bench, bench), fontsize=14, fontweight="bold")
        ax.legend(fontsize=9.5, loc="lower right", framealpha=0.9)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(labelsize=10)

    fig.tight_layout(pad=1.8)

    for out_dir in [SCRIPT_DIR, LOGS_DIR]:
        for ext in ["pdf", "png"]:
            out = os.path.join(out_dir, f"pareto_v2_vs_v4.{ext}")
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"  Saved {out}")
    plt.close(fig)


def print_comparison_table(dense=False):
    """Print a concise comparison table to stdout."""
    results = load_data(dense=dense)
    benchmarks = list(results.keys())

    hdr = (
        f"{'Benchmark':<12} {'Budget':>6} "
        f"{'v2 nodes':>9} {'v2 tau':>8} "
        f"{'v4 nodes':>9} {'v4 tau':>8} "
        f"{'dtau':>8} {'dtau%':>7} "
        f"{'eff v2':>8} {'eff v4':>8}"
    )
    print(f"\n{'=' * len(hdr)}")
    print("  Pareto Comparison: v4 (Prefix-Aware) vs v2 (EAGLE-2)")
    print(f"{'=' * len(hdr)}")
    print(hdr)
    print(f"{'-' * len(hdr)}")

    for bench in benchmarks:
        v2_entries = results[bench].get("v2", [])
        v4_entries = results[bench].get("v4", [])
        for v2e in v2_entries:
            v4e = next(
                (e for e in v4_entries if e["max_tree_size"] == v2e["max_tree_size"]),
                None,
            )
            if v4e is None:
                continue
            dtau = v4e["mean_tau"] - v2e["mean_tau"]
            dpct = (dtau / v2e["mean_tau"]) * 100 if v2e["mean_tau"] > 0 else 0
            eff_v2 = v2e["mean_tau"] / max(v2e["mean_nodes"], 1)
            eff_v4 = v4e["mean_tau"] / max(v4e["mean_nodes"], 1)
            print(
                f"{BENCH_DISPLAY.get(bench, bench):<12} {v2e['max_tree_size']:>6} "
                f"{v2e['mean_nodes']:>9.1f} {v2e['mean_tau']:>8.3f} "
                f"{v4e['mean_nodes']:>9.1f} {v4e['mean_tau']:>8.3f} "
                f"{dtau:>+8.3f} {dpct:>+6.1f}% "
                f"{eff_v2:>8.4f} {eff_v4:>8.4f}"
            )
        print(f"{'-' * len(hdr)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", action="store_true",
                    help="Use dense v2-vs-v4 sweep data if available")
    cli = ap.parse_args()

    print("Generating Pareto comparison: v4 vs v2 ...")
    fig_pareto_v2_vs_v4(dense=cli.dense)
    print_comparison_table(dense=cli.dense)
    print("\nDone.")
