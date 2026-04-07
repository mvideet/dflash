#!/usr/bin/env python3
"""
Parse ablation summary files and produce:
  - Terminal-friendly tables
  - LaTeX tables (paper-ready)
  - Figures (PDF + PNG)

Usage:
  python paper/format_ablation_results.py          # format all available ablations
  python paper/format_ablation_results.py --only A  # format only ablation A
"""

import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")
OUT_DIR = SCRIPT_DIR

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

METHOD_DISPLAY = {
    "v1_thresh": "Threshold (v1)",
    "v2_eagle2": "EAGLE-2 (v2)",
    "v3_bestfirst": "Best-First (v3)",
    "v4_prefixaware": "Prefix-Aware (v4)",
}
METHOD_COLORS = {
    "v1_thresh": "#d62728",
    "v2_eagle2": "#1f77b4",
    "v3_bestfirst": "#2ca02c",
    "v4_prefixaware": "#9467bd",
}
BENCH_DISPLAY = {
    "gsm8k": "GSM8K", "math500": "MATH-500", "aime24": "AIME'24",
    "aime25": "AIME'25", "alpaca": "Alpaca", "mt-bench": "MT-Bench",
    "humaneval": "HumanEval", "mbpp": "MBPP", "lbpp": "LBPP",
    "swe-bench": "SWE-Bench", "livecodebench": "LiveCode",
}


def parse_summary(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            m = re.match(
                r"^(\S+)\s+temp=([\d.]+)\s+(\S+)\s+"
                r"speedup=([\d.]+|N/A)\s+avg_accept=([\d.]+|N/A)\s+"
                r"avg_nodes=([\d.]+|N/A)\s+exit=(\d+)",
                line.strip(),
            )
            if m:
                rows.append({
                    "bench": m.group(1),
                    "temp": m.group(2),
                    "method": m.group(3),
                    "speedup": float(m.group(4)) if m.group(4) != "N/A" else None,
                    "tau": float(m.group(5)) if m.group(5) != "N/A" else None,
                    "nodes": float(m.group(6)) if m.group(6) != "N/A" else None,
                    "exit": int(m.group(7)),
                })
    return rows


def parse_histograms(path):
    """Parse histograms from per-run log files (handles multi-line output)."""
    hists = {}
    if not os.path.exists(path):
        return hists

    log_dir = os.path.join(os.path.dirname(path))
    for fname in sorted(os.listdir(log_dir)):
        if not fname.startswith("ablE_") or not fname.endswith(".log"):
            continue
        m = re.match(r"ablE_(.+?)_(v\d+_\w+)_t([\d.]+)\.log", fname)
        if not m:
            continue
        bench, method, temp = m.group(1), m.group(2), m.group(3)
        log_path = os.path.join(log_dir, fname)

        with open(log_path) as f:
            full_text = f.read()
        hm = re.search(r"Acceptance length histogram:\s*\[([^\]]+)\]", full_text)
        if hm:
            vals = [float(x.strip().strip("'").rstrip("%")) / 100.0
                    for x in hm.group(1).split(",") if x.strip()]
            hists[(bench, temp, method)] = vals

    return hists


def parse_profile(path):
    """Parse profile data from per-run log files for more reliable extraction."""
    profiles = {}
    if not os.path.exists(path):
        return profiles

    log_dir = os.path.dirname(path)
    for fname in sorted(os.listdir(log_dir)):
        if not fname.startswith("ablD_") or not fname.endswith(".log"):
            continue
        m = re.match(r"ablD_(.+?)_v4_profile\.log", fname)
        if not m:
            continue
        bench = m.group(1)
        log_path = os.path.join(log_dir, fname)
        profiles[bench] = {}

        with open(log_path) as f:
            for line in f:
                pm = re.match(
                    r"^\s*(\w+):\s+([\d.]+)s total\s+\(([\d.]+)%\s*(?:of decode)?\)",
                    line,
                )
                if pm:
                    profiles[bench][pm.group(1)] = {
                        "total_s": float(pm.group(2)),
                        "pct_decode": float(pm.group(3)),
                    }

    return profiles


# ======================================================================
# A) Temperature sensitivity
# ======================================================================
def format_A():
    path = os.path.join(LOGS_DIR, "ablation_A_temperature.txt")
    rows = parse_summary(path)
    if not rows:
        print("  [A] No data found.")
        return

    print("\n" + "=" * 80)
    print("  Ablation A: Temperature Sensitivity (temp=0.6)")
    print("=" * 80)

    benchmarks = list(dict.fromkeys(r["bench"] for r in rows))
    methods = ["v1_thresh", "v2_eagle2", "v3_bestfirst", "v4_prefixaware"]
    lookup = {(r["bench"], r["method"]): r for r in rows}

    hdr = f"{'Benchmark':<14}"
    for m in methods:
        hdr += f" {'τ':>6} {'Spd':>6}"
    print(hdr)
    print("-" * len(hdr))

    for b in benchmarks:
        line = f"{BENCH_DISPLAY.get(b, b):<14}"
        best_tau = max((lookup.get((b, m), {}).get("tau") or 0) for m in methods)
        for m in methods:
            r = lookup.get((b, m))
            tau_s = f"{r['tau']:.2f}" if r and r["tau"] else "—"
            spd_s = f"{r['speedup']:.2f}×" if r and r["speedup"] else "—"
            if r and r["tau"] and r["tau"] >= best_tau - 0.005:
                tau_s = f"*{tau_s}*"
            line += f" {tau_s:>6} {spd_s:>6}"
        print(line)

    # LaTeX
    tex_path = os.path.join(OUT_DIR, "table_ablation_A.tex")
    with open(tex_path, "w") as f:
        f.write("% Ablation A: Temperature sensitivity (temp=0.6)\n")
        f.write("\\begin{table}[t]\n\\centering\\small\n")
        f.write("\\caption{Temperature sensitivity ($T{=}0.6$). Bold: best $\\bar{\\tau}$ per row.}\n")
        f.write("\\label{tab:ablation-temp}\n")
        f.write("\\begin{tabular}{@{}l" + " rr" * len(methods) + "@{}}\n\\toprule\n")
        f.write("& " + " & ".join(
            f"\\multicolumn{{2}}{{c}}{{{METHOD_DISPLAY[m]}}}" for m in methods
        ) + " \\\\\n")
        f.write("".join(f"\\cmidrule(lr){{{2*i+2}-{2*i+3}}}" for i in range(len(methods))) + "\n")
        f.write("Benchmark" + " & $\\bar{\\tau}$ & Speed" * len(methods) + " \\\\\n\\midrule\n")
        for b in benchmarks:
            best_tau = max((lookup.get((b, m), {}).get("tau") or 0) for m in methods)
            line = BENCH_DISPLAY.get(b, b)
            for m in methods:
                r = lookup.get((b, m))
                tau_v = r["tau"] if r and r["tau"] else 0
                spd_v = r["speedup"] if r and r["speedup"] else 0
                tau_s = f"{tau_v:.2f}" if tau_v else "—"
                if tau_v and tau_v >= best_tau - 0.005:
                    tau_s = f"\\textbf{{{tau_s}}}"
                line += f" & {tau_s} & {spd_v:.2f}$\\times$"
            f.write(line + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"\n  LaTeX: {tex_path}")


# ======================================================================
# B) v3 vs v4 ablation
# ======================================================================
def format_B():
    path = os.path.join(LOGS_DIR, "ablation_B_v3_vs_v4.txt")
    rows = parse_summary(path)
    if not rows:
        print("  [B] No data found.")
        return

    print("\n" + "=" * 80)
    print("  Ablation B: v3 vs v4 (Phase 2 contribution)")
    print("=" * 80)

    benchmarks = list(dict.fromkeys(r["bench"] for r in rows))
    lookup = {(r["bench"], r["method"]): r for r in rows}

    hdr = f"{'Benchmark':<14} {'v3 τ':>7} {'v3 Nodes':>9} {'v4 τ':>7} {'v4 Nodes':>9} {'Δτ':>7} {'Δτ%':>6}"
    print(hdr)
    print("-" * len(hdr))

    for b in benchmarks:
        r3 = lookup.get((b, "v3_bestfirst"))
        r4 = lookup.get((b, "v4_prefixaware"))
        t3 = r3["tau"] if r3 and r3["tau"] else 0
        t4 = r4["tau"] if r4 and r4["tau"] else 0
        n3 = r3["nodes"] if r3 and r3["nodes"] else 0
        n4 = r4["nodes"] if r4 and r4["nodes"] else 0
        delta = t4 - t3
        pct = (delta / t3 * 100) if t3 > 0 else 0
        print(f"{BENCH_DISPLAY.get(b, b):<14} {t3:>7.2f} {n3:>9.0f} {t4:>7.2f} {n4:>9.0f} {delta:>+7.2f} {pct:>+5.1f}%")

    tex_path = os.path.join(OUT_DIR, "table_ablation_B.tex")
    with open(tex_path, "w") as f:
        f.write("% Ablation B: v3 vs v4\n")
        f.write("\\begin{table}[t]\n\\centering\\small\n")
        f.write("\\caption{Phase~2 ablation: Best-First (v3) vs.\\ Prefix-Aware (v4) at \\texttt{max\\_tree\\_size}$\\,{=}\\,32$.}\n")
        f.write("\\label{tab:ablation-v3v4}\n")
        f.write("\\begin{tabular}{@{}l rr rr r@{}}\n\\toprule\n")
        f.write("& \\multicolumn{2}{c}{Best-First (v3)} & \\multicolumn{2}{c}{Prefix-Aware (v4)} & \\\\\n")
        f.write("\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n")
        f.write("Benchmark & $\\bar{\\tau}$ & $\\bar{N}$ & $\\bar{\\tau}$ & $\\bar{N}$ & $\\Delta\\bar{\\tau}$ \\\\\n\\midrule\n")
        for b in benchmarks:
            r3 = lookup.get((b, "v3_bestfirst"))
            r4 = lookup.get((b, "v4_prefixaware"))
            t3 = r3["tau"] if r3 and r3["tau"] else 0
            t4 = r4["tau"] if r4 and r4["tau"] else 0
            n3 = r3["nodes"] if r3 and r3["nodes"] else 0
            n4 = r4["nodes"] if r4 and r4["nodes"] else 0
            delta = t4 - t3
            f.write(f"{BENCH_DISPLAY.get(b, b)} & {t3:.2f} & {n3:.0f} & \\textbf{{{t4:.2f}}} & {n4:.0f} & {delta:+.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"\n  LaTeX: {tex_path}")


# ======================================================================
# C) expand_k sensitivity
# ======================================================================
def format_C():
    path = os.path.join(LOGS_DIR, "ablation_C_expand_k.txt")
    rows = parse_summary(path)
    if not rows:
        print("  [C] No data found.")
        return

    print("\n" + "=" * 80)
    print("  Ablation C: expand_k Sensitivity")
    print("=" * 80)

    benchmarks = list(dict.fromkeys(r["bench"] for r in rows))
    ks = sorted(set(int(r["method"].split("k")[-1]) for r in rows))
    lookup = {(r["bench"], r["method"]): r for r in rows}

    hdr = f"{'Benchmark':<14}"
    for k in ks:
        hdr += f"  K={k} τ  K={k} Nodes"
    print(hdr)
    print("-" * len(hdr))

    for b in benchmarks:
        line = f"{BENCH_DISPLAY.get(b, b):<14}"
        for k in ks:
            r = lookup.get((b, f"v4_k{k}"))
            tau = r["tau"] if r and r["tau"] else 0
            nodes = r["nodes"] if r and r["nodes"] else 0
            line += f"  {tau:>5.2f}  {nodes:>7.0f}"
        print(line)

    if HAS_MPL:
        fig, axes = plt.subplots(1, len(benchmarks), figsize=(5 * len(benchmarks), 4), squeeze=False)
        axes = axes[0]
        for ax, b in zip(axes, benchmarks):
            taus = []
            nodes = []
            for k in ks:
                r = lookup.get((b, f"v4_k{k}"))
                taus.append(r["tau"] if r and r["tau"] else 0)
                nodes.append(r["nodes"] if r and r["nodes"] else 0)
            ax.plot(ks, taus, "D-", color="#9467bd", linewidth=2, markersize=8)
            ax.set_xlabel("expand_k", fontsize=12)
            ax.set_ylabel("$\\bar{\\tau}$", fontsize=12)
            ax.set_title(BENCH_DISPLAY.get(b, b), fontsize=13, fontweight="bold")
            ax.set_xticks(ks)
            ax.grid(True, alpha=0.25)

            ax2 = ax.twinx()
            ax2.bar(ks, nodes, width=0.3, alpha=0.25, color="#9467bd", label="Nodes")
            ax2.set_ylabel("$\\bar{N}$ (nodes)", fontsize=10, alpha=0.6)

        fig.tight_layout()
        for ext in ["pdf", "png"]:
            out = os.path.join(OUT_DIR, f"fig_ablation_C_expand_k.{ext}")
            fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Figure: {os.path.join(OUT_DIR, 'fig_ablation_C_expand_k.pdf')}")


# ======================================================================
# D) Profile overhead
# ======================================================================
def format_D():
    path = os.path.join(LOGS_DIR, "ablation_D_profile.txt")
    profiles = parse_profile(path)
    if not profiles:
        print("  [D] No data found.")
        return

    print("\n" + "=" * 80)
    print("  Ablation D: v4 Profiling Breakdown")
    print("=" * 80)

    for bench, timings in profiles.items():
        print(f"\n  {BENCH_DISPLAY.get(bench, bench)}:")
        total_decode = sum(v["total_s"] for k, v in timings.items() if k != "prefill_target")
        for name in sorted(timings.keys()):
            t = timings[name]
            bar = "█" * int(t["pct_decode"] / 2) if "pct_decode" in t else ""
            print(f"    {name:<22} {t['total_s']:>7.2f}s  {t.get('pct_decode', 0):>5.1f}%  {bar}")

        tree_build = timings.get("tree_build", {})
        if tree_build:
            print(f"    ── tree_build is {tree_build.get('pct_decode', 0):.1f}% of decode time")


# ======================================================================
# E) Acceptance length histograms
# ======================================================================
def format_E():
    path = os.path.join(LOGS_DIR, "ablation_E_histograms.txt")
    hists = parse_histograms(path)
    if not hists:
        print("  [E] No data found.")
        return

    print("\n" + "=" * 80)
    print("  Ablation E: Acceptance Length Histograms (v3 vs v4)")
    print("=" * 80)

    benchmarks = sorted(set(k[0] for k in hists.keys()))

    for b in benchmarks:
        h3 = hists.get((b, "0.0", "v3_bestfirst"))
        h4 = hists.get((b, "0.0", "v4_prefixaware"))
        if not h3 or not h4:
            continue
        max_len = max(len(h3), len(h4))
        print(f"\n  {BENCH_DISPLAY.get(b, b)}:")
        print(f"    {'Len':>4}  {'v3':>7}  {'v4':>7}  {'Δ':>7}")
        for i in range(max_len):
            p3 = h3[i] if i < len(h3) else 0
            p4 = h4[i] if i < len(h4) else 0
            delta = p4 - p3
            bar3 = "▓" * int(p3 * 50)
            bar4 = "█" * int(p4 * 50)
            print(f"    {i:>4}  {p3*100:>6.1f}%  {p4*100:>6.1f}%  {delta*100:>+6.1f}%  {bar3}|{bar4}")

    if HAS_MPL and benchmarks:
        fig, axes = plt.subplots(1, len(benchmarks), figsize=(5.5 * len(benchmarks), 4), squeeze=False)
        axes = axes[0]
        for ax, b in zip(axes, benchmarks):
            h3 = hists.get((b, "0.0", "v3_bestfirst"), [])
            h4 = hists.get((b, "0.0", "v4_prefixaware"), [])
            max_len = max(len(h3), len(h4))
            x = np.arange(max_len)
            w = 0.35
            vals3 = [h3[i] * 100 if i < len(h3) else 0 for i in range(max_len)]
            vals4 = [h4[i] * 100 if i < len(h4) else 0 for i in range(max_len)]
            ax.bar(x - w / 2, vals3, w, label="Best-First (v3)", color="#2ca02c", alpha=0.8)
            ax.bar(x + w / 2, vals4, w, label="Prefix-Aware (v4)", color="#9467bd", alpha=0.8)
            ax.set_xlabel("Accepted length", fontsize=11)
            ax.set_ylabel("Frequency (%)", fontsize=11)
            ax.set_title(BENCH_DISPLAY.get(b, b), fontsize=13, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        for ext in ["pdf", "png"]:
            out = os.path.join(OUT_DIR, f"fig_ablation_E_histograms.{ext}")
            fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Figure: {os.path.join(OUT_DIR, 'fig_ablation_E_histograms.pdf')}")


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated list of ablations to format (e.g. A,B)")
    args = parser.parse_args()

    exps = args.only.upper().split(",") if args.only else ["A", "B", "C", "D", "E"]

    for exp in exps:
        if exp == "A":
            format_A()
        elif exp == "B":
            format_B()
        elif exp == "C":
            format_C()
        elif exp == "D":
            format_D()
        elif exp == "E":
            format_E()
        else:
            print(f"Unknown ablation: {exp}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
