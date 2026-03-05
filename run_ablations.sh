#!/bin/bash
# Ablation study: block_size (8,16), temperature, theta (pruning/branching thresholds), max_tree_size
# Output: logs/ablation_results.csv

# No set -e: continue on benchmark failures so we can collect partial results
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

mkdir -p logs
VANILLA_RESULTS=logs/ablation_results_a100_vanilla.csv
RESULTS=logs/ablation_results_a100.csv

# Grid (must be defined before vanilla loop)
BLOCK_SIZES=(8 16)
DATASET="mt-bench"
MAX_SAMPLES=80
PORT=29600

# ------------------------------------------------------------------
# Baseline: Vanilla DFlash (no dynamic branching) at each block size and temperature
# Writes to ablation_results_vanilla.csv (does not touch ablation_results.csv)
# ------------------------------------------------------------------
echo ""
echo "========================================================"
echo "BASELINE: Vanilla DFlash (block_size 8 and 16, multiple temperatures)"
echo "========================================================"

echo 'block_size,mode,temperature,theta_uni,theta_bi,theta_tri,max_tree_size,avg_acceptance,speedup' > "$VANILLA_RESULTS"

VANILLA_TEMPERATURES=(0.0 0.2 0.4 0.6)

for BS in "${BLOCK_SIZES[@]}"; do
  for T in "${VANILLA_TEMPERATURES[@]}"; do
    echo "[vanilla] block_size=$BS temperature=$T"
    unset TORCH_LOGS
    TMPOUT=$(mktemp)
    (PYTHONUNBUFFERED=1 torchrun \
      --nproc_per_node=8 \
      --master_port=$PORT \
      benchmark.py \
      --dataset "$DATASET" \
      --max-samples "$MAX_SAMPLES" \
      --model-name-or-path Qwen/Qwen3-4B \
      --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 \
      --block-size "$BS" \
      --temperature "$T" \
      --max-new-tokens 2048 \
      2>&1 > "$TMPOUT") || true

    SPEEDUP=$(grep "Decoding speedup:" "$TMPOUT" 2>/dev/null | grep -oE "[0-9]+\.[0-9]+" | tail -1)
    ACC=$(grep "Average Acceptance length:" "$TMPOUT" 2>/dev/null | grep -oE "[0-9]+\.[0-9]+" | tail -1)

    if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
      echo "[WARN] Vanilla block_size=$BS temp=$T failed (no speedup/acceptance in output)" >&2
      tail -80 "$TMPOUT" > logs/ablation_debug_vanilla_bs${BS}_t${T}.txt
      echo "--- Last 80 lines of output (see logs/ablation_debug_vanilla_bs${BS}_t${T}.txt for full): ---" >&2
      tail -30 "$TMPOUT" >&2
      SPEEDUP="N/A"
      ACC="N/A"
    fi
    rm -f "$TMPOUT"

    echo "$BS,vanilla,$T,-,-,-,-,$ACC,$SPEEDUP" >> "$VANILLA_RESULTS"
    PORT=$((PORT + 1))
  done
done

echo ""
echo "========================================================"
echo "ABLATIONS: Dynamic branching grid"
echo "========================================================"
echo 'block_size,mode,temperature,theta_uni,theta_bi,theta_tri,max_tree_size,avg_acceptance,speedup' > "$RESULTS"

# # Grid (edit to change search space)
# # Quick: TEMPERATURES=(0.0 0.4) THETA_UNI_VALS=(0.9) THETA_BI_VALS=(0.3) -> 2*2*1*1*1*3=12 configs
# TEMPERATURES=(0.0 0.2 0.4 0.6)
# THETA_UNI_VALS=(0.85 0.9 0.92)
# THETA_BI_VALS=(0.25 0.3 0.35)
# THETA_TRI_VALS=(0.1)
# MAX_TREE_SIZES=(4 6 8)

# TOTAL=0

# for BS in "${BLOCK_SIZES[@]}"; do
#   for T in "${TEMPERATURES[@]}"; do
#     for TU in "${THETA_UNI_VALS[@]}"; do
#       for TB in "${THETA_BI_VALS[@]}"; do
#         for TT in "${THETA_TRI_VALS[@]}"; do
#           for MTS in "${MAX_TREE_SIZES[@]}"; do
#             TOTAL=$((TOTAL + 1))
#             echo "[$TOTAL] block_size=$BS temp=$T theta=($TU,$TB,$TT) max_tree=$MTS"

#             TMPOUT=$(mktemp)
#             (PYTHONUNBUFFERED=1 torchrun \
#               --nproc_per_node=8 \
#               --master_port=$PORT \
#               benchmark.py \
#               --dataset "$DATASET" \
#               --max-samples "$MAX_SAMPLES" \
#               --model-name-or-path Qwen/Qwen3-4B \
#               --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 \
#               --block-size "$BS" \
#               --temperature "$T" \
#               --max-new-tokens 2048 \
#               --dynamic-branching \
#               --theta-uni "$TU" \
#               --theta-bi "$TB" \
#               --theta-tri "$TT" \
#               --max-tree-size "$MTS" \
#               2>&1 > "$TMPOUT") || true

#             SPEEDUP=$(grep "Decoding speedup:" "$TMPOUT" 2>/dev/null | grep -oE "[0-9]+\.[0-9]+" | tail -1)
#             ACC=$(grep "Average Acceptance length:" "$TMPOUT" 2>/dev/null | grep -oE "[0-9]+\.[0-9]+" | tail -1)

#             if [ -z "$SPEEDUP" ] || [ -z "$ACC" ]; then
#               echo "[WARN] Config failed, writing N/A" >&2
#               tail -30 "$TMPOUT" > logs/ablation_debug_last.txt
#               SPEEDUP="N/A"
#               ACC="N/A"
#             fi
#             rm -f "$TMPOUT"

#             echo "$BS,dynamic,$T,$TU,$TB,$TT,$MTS,$ACC,$SPEEDUP" >> "$RESULTS"
#             PORT=$((PORT + 1))
#           done
#         done
#       done
#     done
#   done
# done

# echo ""
# echo "Done. Results: $RESULTS"
# echo "Total configs: $TOTAL"
