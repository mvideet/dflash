#!/usr/bin/env bash
# Full v1 / v2 (EAGLE-2) / v3 (best-first) comparison on every dataset supported by
# model/utils.py:load_and_process_dataset — intended for overnight runs.
#
# Usage:  ./run_full_comparison_overnight.sh
#         OVERNIGHT_PROFILE=false ./run_full_comparison_overnight.sh   # faster, no --profile
#
# Results: logs/full_comparison_overnight.txt
set -uo pipefail

cd "$(dirname "$0")"
mkdir -p logs
unset TORCH_LOGS

SUMMARY="logs/full_comparison_overnight.txt"

NPROC="${NPROC_PER_NODE:-8}"
PORT="${MASTER_PORT:-29600}"
MODEL="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B}"
DRAFT="${DRAFT_NAME_OR_PATH:-z-lab/Qwen3-4B-DFlash-b16}"
MAX_TOKENS="${MAX_NEW_TOKENS:-2048}"

TREE_VERSIONS=(1 2 3 4)
TREE_NAMES=("v1_thresh" "v2_eagle2" "v3_bestfirst" "v4_prefixaware")
MTS="${OVERNIGHT_MTS:-32}"
EXPAND_K="${OVERNIGHT_EXPAND_K:-3}"
TOP_K="${OVERNIGHT_TOP_K:-3}"
TEMPERATURES=("0.0" "0.6")

# dataset_name:max_samples — caps keep wall-clock reasonable on huge splits (alpaca, livecodebench, etc.)
DATASETS=(
  "gsm8k:256"
  "math500:256"
  "aime24:64"
  "aime25:64"
  "alpaca:256"
  "mt-bench:80"
  "humaneval:164"
  "mbpp:256"
  "lbpp:256"
  "swe-bench:128"
  "livecodebench:256"
)

USE_PROFILE="${OVERNIGHT_PROFILE:-true}"
PROFILE_FLAG=()
if [ "${USE_PROFILE}" = "true" ]; then
  PROFILE_FLAG=(--profile)
fi

TOTAL=$(( ${#TREE_VERSIONS[@]} * ${#DATASETS[@]} * ${#TEMPERATURES[@]} ))
RUN=0

{
  echo ""
  echo "========================================================================"
  echo "Overnight full-dataset sweep — $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "Tree versions: ${TREE_NAMES[*]}"
  echo "Datasets (${#DATASETS[@]}): ${DATASETS[*]}"
  echo "Temperatures: ${TEMPERATURES[*]}"
  echo "max_tree_size=${MTS}  expand_k=${EXPAND_K}  top_k=${TOP_K}  profile=${USE_PROFILE}"
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
      LOG_FILE="logs/overnight_${DS_NAME}_${TAG}.log"

      echo ""
      echo ">>> [${RUN}/${TOTAL}] ${DS_NAME} n=${DS_SAMPLES} temp=${TEMP} ${TNAME}"
      echo ">>> [${RUN}/${TOTAL}] ${DS_NAME} n=${DS_SAMPLES} temp=${TEMP} ${TNAME}" >> "$SUMMARY"

      EXTRA_ARGS=(
        --dynamic-branching
        --tree-version "$TV"
        --max-tree-size "$MTS"
        --top-k "$TOP_K"
        --expand-k "$EXPAND_K"
        --theta-uni 0.9 --theta-bi 0.3 --theta-tri 0.1
        "${PROFILE_FLAG[@]}"
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

      LINE="${DS_NAME}  n=${DS_SAMPLES}  temp=${TEMP}  ${TNAME}  mts=${MTS}  speedup=${SPEEDUP}  avg_accept=${ACC}  exit=${RC}"
      echo "$LINE"
      echo "$LINE" >> "$SUMMARY"

      PORT=$((PORT + 1))
    done
  done
done

echo ""
echo "========================================================================"
echo "Sweep complete — result rows (grep speedup=):"
echo "========================================================================"
grep "speedup=" "$SUMMARY" | grep "avg_accept=" || true
echo ""
echo "Full log: $SUMMARY"