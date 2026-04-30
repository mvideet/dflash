"""HW Idea 7 feasibility: how much memory could KV-quantization save?

Idea: prefix KV in INT4/FP8, tree-page KV in bf16. Memory ceiling is set by
the prefix-vs-tree memory ratio. Measure both.

Per-token KV size:
  bf16: 32 layers × 32 heads × 128 dim × 2 (K+V) × 2 bytes = 524 KB/token
  INT4: 32 layers × 32 heads × 128 dim × 2 (K+V) × 0.5 bytes = 131 KB/token (4× saving)

At realistic shapes, what fraction of total KV is prefix?
"""
import json, os
import numpy as np

# Configurations from real measured runs
configs = [
    (1, 64, 600, "B=1 M=64 prefix=600 (math500 standard)"),
    (4, 32, 600, "B=4 M=32 prefix=600 (math500 standard)"),
    (8, 16, 600, "B=8 M=16 prefix=600"),
    (32, 16, 800, "B=32 M=16 prefix=800"),
    (64, 16, 1200, "B=64 M=16 prefix=1200 (after generation)"),
    (128, 8, 1500, "B=128 M=8 prefix=1500 (likely OOM)"),
]

KV_BYTES_PER_TOKEN_BF16 = 32 * 32 * 128 * 2 * 2  # ~524 KB
INT4_RATIO = 4.0  # 4× memory saving with INT4

print(f"Per-token KV (bf16): {KV_BYTES_PER_TOKEN_BF16 / 1024:.0f} KB")
print(f"Per-token KV (INT4): {KV_BYTES_PER_TOKEN_BF16 / 1024 / INT4_RATIO:.0f} KB")

print(f"\n{'config':>50}  {'prefix_MB':>9}  {'tree_MB':>8}  {'total_MB':>8}  {'INT4_save_MB':>12}  {'INT4_save_%':>10}")
out = []
for B, M, prefix, label in configs:
    prefix_mb = B * prefix * KV_BYTES_PER_TOKEN_BF16 / 1024 / 1024
    tree_mb = B * M * KV_BYTES_PER_TOKEN_BF16 / 1024 / 1024
    total_mb = prefix_mb + tree_mb
    # If prefix is INT4, save 75% of prefix portion
    int4_save_mb = prefix_mb * (1 - 1/INT4_RATIO)
    int4_save_pct = int4_save_mb / total_mb * 100

    print(f"{label:>50}  {prefix_mb:>8.0f}M  {tree_mb:>7.0f}M  {total_mb:>7.0f}M  "
          f"{int4_save_mb:>11.0f}M  {int4_save_pct:>9.1f}%")
    out.append({
        "B": B, "M": M, "prefix": prefix, "label": label,
        "prefix_mb": prefix_mb, "tree_mb": tree_mb,
        "total_mb": total_mb,
        "int4_save_mb": int4_save_mb, "int4_save_pct": int4_save_pct,
    })

# Total A6000 = 48GB. Model = ~16GB. Available for KV: ~32GB.
print(f"\nA6000 KV budget: ~32GB")
print(f"\nAt B=128 prefix=1500: {out[-1]['total_mb']:.0f} MB needed; INT4-prefix saves "
      f"{out[-1]['int4_save_mb']:.0f} MB → drops to {out[-1]['total_mb']-out[-1]['int4_save_mb']:.0f} MB")
print(f"  → INT4 prefix could enable B=128 fitting in memory cleanly")

print("\n=== Composition with prefix-sharing (Idea 5) ===")
print("Prefix-sharing = reuse prefix across batch; effectively divides prefix_mb by B")
for r in out:
    if r["B"] >= 8:
        shared_prefix_mb = r["prefix_mb"] / r["B"]
        # Idea 5 alone
        idea5_save = r["prefix_mb"] - shared_prefix_mb
        # Idea 5 + INT4 stacked
        stacked_save = r["prefix_mb"] - shared_prefix_mb / INT4_RATIO
        print(f"  {r['label']}: Idea5 alone saves {idea5_save:.0f}MB "
              f"({idea5_save/r['total_mb']*100:.0f}%); +INT4 saves {stacked_save:.0f}MB "
              f"({stacked_save/r['total_mb']*100:.0f}%)")

os.makedirs("logs", exist_ok=True)
with open("logs/hw_kv_quant.json", "w") as f:
    json.dump({"results": out, "kv_per_token_bf16_kb": KV_BYTES_PER_TOKEN_BF16/1024,
               "int4_ratio": INT4_RATIO}, f, indent=2)
print("\nWrote logs/hw_kv_quant.json")
