"""Generate key figures for the paper from session apr18-19 data.

Outputs SVG files to paper/fig/ .
"""

import os
os.makedirs("paper/fig", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---- Figure 1: OOD cliff vs VB plateau (math500 32 samples) ----

b_values = np.array([16, 20, 24, 28])
stock_speedup = [8.42, 7.55, 6.26, np.nan]
stock_tau    = [10.38, 9.59, 7.66, np.nan]
vb_speedup   = [8.31, 8.68, 8.58, 8.40]
vb_tau       = [10.27, 10.91, 10.85, 10.92]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.8))

ax1.plot(b_values, stock_speedup, "o-", label="stock b=16 trained", color="#d62728", linewidth=2, markersize=8)
ax1.plot(b_values, vb_speedup,    "s-", label="VB v1 (b∈{12,16,20,24} trained)", color="#2ca02c", linewidth=2, markersize=8)
ax1.axhline(8.33, ls="--", c="gray", alpha=0.5, label="stock b=16 (256 samples)")
ax1.set_xlabel("inference block size b")
ax1.set_ylabel("speedup (wall-clock ratio)")
ax1.set_title("(a) speedup vs inference block size")
ax1.set_xticks(b_values)
ax1.legend(loc="lower left", fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.plot(b_values, stock_tau, "o-", label="stock b=16 trained", color="#d62728", linewidth=2, markersize=8)
ax2.plot(b_values, vb_tau,    "s-", label="VB v1 b∈{12..24} trained", color="#2ca02c", linewidth=2, markersize=8)
ax2.axhline(10.08, ls="--", c="gray", alpha=0.5, label="stock b=16 (256 samples)")
ax2.set_xlabel("inference block size b")
ax2.set_ylabel("acceptance length τ")
ax2.set_title("(b) τ vs inference block size")
ax2.set_xticks(b_values)
ax2.legend(loc="lower left", fontsize=9)
ax2.grid(True, alpha=0.3)

fig.suptitle("Stock's OOD cliff vs. VB's plateau at b ≥ 20 (math500, 32 samples)",
             y=1.02, fontsize=12)
fig.tight_layout()
fig.savefig("paper/fig/ood_cliff.svg", bbox_inches="tight")
print("wrote paper/fig/ood_cliff.svg")
plt.close(fig)


# ---- Figure 2: Cross-dataset gains (VB b=20 vs stock b=16) ----

datasets = ["math500\n(256s)", "mt-bench\n(80s)", "gsm8k\n(128s)", "humaneval\n(164s)"]
stock_vals = [8.33, 4.41, 7.25, 7.46]
vb_vals    = [8.52, 4.20, 7.32, 7.59]

x = np.arange(len(datasets))
width = 0.36

fig, ax = plt.subplots(figsize=(7.5, 4))
b1 = ax.bar(x - width/2, stock_vals, width, label="stock b=16", color="#d62728", alpha=0.85)
b2 = ax.bar(x + width/2, vb_vals,    width, label="VB v1 b=20", color="#2ca02c", alpha=0.85)

for i, (s, v) in enumerate(zip(stock_vals, vb_vals)):
    delta = v - s
    sign = "+" if delta >= 0 else ""
    color = "#2ca02c" if delta > 0 else "#d62728"
    ax.text(i + width/2, v + 0.08, f"{sign}{delta:+.2f}", ha="center",
            fontsize=9, color=color, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylabel("speedup")
ax.set_title("Cross-dataset speedup: VB v1 at b=20 vs stock at b=16")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig("paper/fig/cross_dataset.svg", bbox_inches="tight")
print("wrote paper/fig/cross_dataset.svg")
plt.close(fig)


# ---- Figure 3: Learning curve (VB v1 at b=16 across training steps) ----

steps = [0, 500, 2000, 5000, 9000, 14000, 18000, 18500]
# At step 0, the drafter is the same as stock
speedup_at_b16 = [8.37, 8.34, 8.18, 8.17, 8.27, 8.24, 8.17, 8.28]
tau_at_b16     = [10.38, 10.35, 10.20, 10.21, 10.26, 10.21, 10.25, 10.27]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.8))

ax1.plot(steps, speedup_at_b16, "o-", color="#1f77b4", linewidth=2, markersize=7)
ax1.axhline(8.37, ls="--", c="gray", alpha=0.5, label="stock")
ax1.set_xlabel("training step")
ax1.set_ylabel("speedup at b=16")
ax1.set_title("(a) speedup at b=16 (in-distribution for stock)")
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.plot(steps, tau_at_b16, "o-", color="#1f77b4", linewidth=2, markersize=7)
ax2.axhline(10.38, ls="--", c="gray", alpha=0.5, label="stock")
ax2.set_xlabel("training step")
ax2.set_ylabel("τ at b=16")
ax2.set_title("(b) τ at b=16")
ax2.grid(True, alpha=0.3)
ax2.legend()

fig.suptitle("VB v1 at b=16 regresses vs stock (broad-mix dilution)",
             y=1.02, fontsize=12)
fig.tight_layout()
fig.savefig("paper/fig/learning_curve_b16.svg", bbox_inches="tight")
print("wrote paper/fig/learning_curve_b16.svg")
plt.close(fig)


# ---- Figure 4: Tau ceiling break ----

fig, ax = plt.subplots(figsize=(6.5, 4))
conditions = ["stock b=16\n(prior SOTA)", "VB v1 b=16", "VB v1 b=20\n(new SOTA)", "VB v1 b=24"]
taus = [10.08, 9.89, 10.43, 10.49]
colors = ["#d62728", "#ff7f0e", "#2ca02c", "#17becf"]

bars = ax.bar(conditions, taus, color=colors, alpha=0.85)
for bar, t in zip(bars, taus):
    ax.text(bar.get_x() + bar.get_width()/2, t + 0.05, f"{t:.2f}",
            ha="center", fontsize=10, fontweight="bold")

ax.axhline(10.08, ls="--", c="gray", alpha=0.5, label="prior ceiling")
ax.axhline(16,    ls=":",  c="gray", alpha=0.3, label="block_size=16 ceiling")
ax.axhline(24,    ls=":",  c="lightgray", alpha=0.3, label="block_size=24 ceiling")

ax.set_ylim([9.5, 25])
ax.set_ylabel("τ (math500 256 samples)")
ax.set_title("Breaking the τ=10.08 prior ceiling via variable-block training")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig("paper/fig/tau_ceiling_break.svg", bbox_inches="tight")
print("wrote paper/fig/tau_ceiling_break.svg")
plt.close(fig)


print("\nAll figures written to paper/fig/")
