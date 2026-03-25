#!/usr/bin/env bash
# Full comparison: v1 vs v2 (EAGLE-2) vs v3 (best-first)
# across datasets and temperatures.
#
# Usage:  ./run_full_comparison.sh
#
# Results: logs/full_comparison.txt (compact table)
#          logs/experiment_runs.txt  (detailed ledger via run_benchmark.sh)
set -uo pipefail

cd "$(dirname "$0")"
mkdir -p logs

SUMMARY="logs/full_comparison.txt"

# Shared config
NPROC="${NPROC_PER_NODE:-8}"
PORT="${MASTER_PORT:-29600}"
MODEL="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B}"
DRAFT="${DRAFT_NAME_OR_PATH:-z-lab/Qwen3-4B-DFlash-b16}"
MAX_TOKENS="${MAX_NEW_TOKENS:-2048}"

# Grid
TREE_VERSIONS=(1 2 3)
TREE_NAMES=("v1_thresh" "v2_eagle2" "v3_bestfirst")
MTS=32
EXPAND_K=3
TOP_K=3

DATASETS=("mt-bench:80" "gsm8k:128")
TEMPERATURES=("0.0" "0.6")

TOTAL=$(( ${#TREE_VERSIONS[@]} * ${#DATASETS[@]} * ${#TEMPERATURES[@]} ))
RUN=0

{
  echo ""
  echo "========================================================================"
  echo "Full comparison sweep — $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "Tree versions: ${TREE_NAMES[*]}"
  echo "Datasets: ${DATASETS[*]}"
  echo "Temperatures: ${TEMPERATURES[*]}"
  echo "max_tree_size=${MTS}  expand_k=${EXPAND_K}  top_k=${TOP_K}"
  echo "Total runs: ${TOTAL}"
  echo "========================================================================"
} | tee -a "$SUMMARY"

for TEMP in "${TEMPERATURES[@]}"; do
  for DS_SPEC in "${DATASETS[@]}"; do
    IFS=':' read -r DS_NAME DS_SAMPLES <<< "$DS_SPEC"

    for i in "${!TREE_VERSIONS[@]}"; do
      TV="${TREE_VERSIONS[$i]}"
      TNAME="${TREE_NAMES[$i]}"
      RUN=$((RUN + 1))
      TAG="${TNAME}_mts${MTS}_t${TEMP}"
      LOG_FILE="logs/${DS_NAME}_${TAG}.log"

      echo ""
      echo ">>> [${RUN}/${TOTAL}] ${DS_NAME} temp=${TEMP} ${TNAME} mts=${MTS}"
      echo ">>> [${RUN}/${TOTAL}] ${DS_NAME} temp=${TEMP} ${TNAME} mts=${MTS}" >> "$SUMMARY"

      EXTRA_ARGS=(
        --dynamic-branching
        --tree-version "$TV"
        --max-tree-size "$MTS"
        --top-k "$TOP_K"
        --expand-k "$EXPAND_K"
        --theta-uni 0.9 --theta-bi 0.3 --theta-tri 0.1
        --profile
      )

      torchrun \
        --nproc_per_node="$NPROC" \
        --master_port="$PORT" \
        benchmark.py \
        --dataset "$DS_NAME" \
        --max-samples "$DS_SAMPLES" \
        --model-name-or-path "$MODEL" \
        --draft-name-or-path "$DRAFT" \
        --max-new-tokens "$MAX_TOKENS" \
        --temperature "$TEMP" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "$LOG_FILE"
      RC="${PIPESTATUS[0]}"

      SPEEDUP=$(grep "Decoding speedup:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP '[\d.]+$' || echo "N/A")
      ACC=$(grep "Average Acceptance length:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP '[\d.]+$' || echo "N/A")

      LINE="${DS_NAME}  temp=${TEMP}  ${TNAME}  mts=${MTS}  speedup=${SPEEDUP}  avg_accept=${ACC}  exit=${RC}"
      echo "$LINE"
      echo "$LINE" >> "$SUMMARY"
    done
  done
done

echo ""
echo "========================================================================"
echo "Sweep complete. Results table:"
echo "========================================================================"
echo ""
echo "dataset       temp  method        mts  speedup  avg_accept"
echo "------------- ----  ------------- ---  -------  ----------"
grep -E "^(mt-bench|gsm8k)" "$SUMMARY" | while IFS= read -r line; do
  echo "  $line"
done
echo ""
echo "Full details: $SUMMARY"
echo "Per-run ledger: logs/experiment_runs.txt"
