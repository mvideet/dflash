"""Plot v7 speedup vs batch size, including the new large-B data."""
import json, os
import matplotlib.pyplot as plt
import numpy as np

# Compiled from earlier mts_sweep_summary.json + batched_benchmark_largeB.json + new v7_largeB.json
import json
# math500 standard config: 512 prompt tokens + 1024 generation tokens.
src = json.load(open("logs/v7_math500_standard.json"))
data = [(r["B"], r["M"], r["speedup"], r["tau"], "standard")
        for r in src["results"] if "speedup" in r]

Bs = [d[0] for d in data]
speedups = [d[2] for d in data]
taus = [d[3] for d in data]
sources = [d[4] for d in data]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Speedup curve
ax = axes[0]
ax.scatter(Bs, speedups, color="#1f77b4", s=80, zorder=3, label="v7 DDTree")
# Highlight peak
peak_idx = int(np.argmax(speedups))
ax.scatter([Bs[peak_idx]], [speedups[peak_idx]], color="#d62728", s=160, zorder=4,
           edgecolor="black", linewidth=2, label=f"peak (B={Bs[peak_idx]})")
ax.plot(Bs, speedups, "-", color="#888", alpha=0.5, zorder=1)
for b, s in zip(Bs, speedups):
    ax.annotate(f"{s:.2f}×", (b, s), textcoords="offset points", xytext=(0, 8),
                fontsize=8, ha="center")
ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1, label="vanilla AR")
ax.axhline(y=np.mean(speedups[-3:]), color="orange", linestyle=":", linewidth=1, alpha=0.5,
           label=f"large-B plateau (~{np.mean(speedups[-3:]):.2f}×)")
ax.set_xscale("log", base=2)
ax.set_xticks(Bs)
ax.set_xticklabels(Bs)
ax.set_xlabel("Batch size B")
ax.set_ylabel("Speedup over vanilla AR")
ax.set_title("v7 (DDTree) speedup vs batch size on math500 — STANDARD CONFIG\n"
             "max_prompt=512, max_new=1024, N=64, Qwen3-4B + DFlash-b16, A6000")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")
ax.set_ylim(0, max(speedups) * 1.15)

# Tau curve (right axis)
ax = axes[1]
ax.scatter(Bs, taus, color="#1f77b4", s=80, zorder=3)
ax.plot(Bs, taus, "-", color="#888", alpha=0.5, zorder=1)
for b, t, m in zip(Bs, taus, [d[1] for d in data]):
    ax.annotate(f"M={m}", (b, t), textcoords="offset points", xytext=(0, -14),
                fontsize=7, ha="center", color="gray")
ax.set_xscale("log", base=2)
ax.set_xticks(Bs)
ax.set_xticklabels(Bs)
ax.set_xlabel("Batch size B")
ax.set_ylabel("Acceptance length (τ)")
ax.set_title("v7 acceptance length vs batch size\n"
             "(τ drops as M shrinks at high B due to memory)")
ax.grid(True, alpha=0.3)
ax.axhline(y=15, color="green", linestyle=":", linewidth=1, alpha=0.4,
           label="block_size ceiling")
ax.legend(loc="upper right")
ax.set_ylim(0, 16)

plt.tight_layout()
out_path = "logs/v7_batch_curve.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"Saved {out_path}")

# Also save data
with open("logs/v7_batch_curve.json", "w") as f:
    json.dump({"data": [{"B": b, "M": m, "speedup": s, "tau": t, "source": src}
                        for b, m, s, t, src in data]}, f, indent=2)
print("Saved logs/v7_batch_curve.json")
