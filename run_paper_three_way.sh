#!/usr/bin/env bash
# Paper comparison across all datasets supported by model/utils.py:
#
#   1) v1_thresh       — dynamic tree, threshold + cartesian cap
#   2) v2_eagle2       — EAGLE-2 style expand + rerank
#   3) v3_bestfirst    — best-first by cumulative log-probability
#   4) v4_prefixaware  — prefix-aware submodular selection
#
# Usage:
#   ./run_paper_three_way.sh
#   MODEL_NAME_OR_PATH=Qwen/Qwen3-4B DRAFT_NAME_OR_PATH=z-lab/Qwen3-4B-DFlash-b16 ./run_paper_three_way.sh
#   PAPER4_TEMPERATURES="0.0" OVERNIGHT_PROFILE=false ./run_paper_three_way.sh
#
# Results: logs/paper_four_trees_summary.txt (one line per run + per-log files under logs/)
set -uo pipefail

cd "$(dirname "$0")"
mkdir -p logs
unset TORCH_LOGS

SUMMARY="${PAPER4_SUMMARY:-${PAPER3_SUMMARY:-logs/paper_four_trees_summary.txt}}"

NPROC="${NPROC_PER_NODE:-8}"
PORT="${MASTER_PORT:-29600}"
MODEL="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B}"
DRAFT="${DRAFT_NAME_OR_PATH:-z-lab/Qwen3-4B-DFlash-b16}"
MAX_TOKENS="${MAX_NEW_TOKENS:-2048}"

MTS="${PAPER4_MTS:-${PAPER3_MTS:-${OVERNIGHT_MTS:-32}}}"
EXPAND_K="${PAPER4_EXPAND_K:-${PAPER3_EXPAND_K:-${OVERNIGHT_EXPAND_K:-3}}}"
TOP_K="${PAPER4_TOP_K:-${PAPER3_TOP_K:-${OVERNIGHT_TOP_K:-3}}}"

# Space-separated list (default: greedy only; set to "0.0 0.6" to match overnight)
read -r -a TEMPERATURES <<< "${PAPER4_TEMPERATURES:-${PAPER3_TEMPERATURES:-0.0}}"

# dataset_name:max_samples — same caps as run_full_comparison_overnight.sh
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

USE_PROFILE="${PAPER4_PROFILE:-${PAPER3_PROFILE:-${OVERNIGHT_PROFILE:-false}}}"
PROFILE_FLAG=()
if [ "${USE_PROFILE}" = "true" ]; then
  PROFILE_FLAG=(--profile)
fi

# --- build mode list: tag  extra_args_as_bash_words ---
# We use a function to emit torchrun args per mode to avoid quoting hell.

run_one() {
  local DS_NAME="$1"
  local DS_SAMPLES="$2"
  local TEMP="$3"
  local MODE_TAG="$4"
  shift 4
  # remaining "$@" are extra args for benchmark.py

  local TAG="${MODE_TAG}_mts${MTS}_t${TEMP}"
  local LOG_FILE="logs/paper4_${DS_NAME}_${TAG}.log"

  echo ""
  echo ">>> ${DS_NAME}  n=${DS_SAMPLES}  temp=${TEMP}  ${MODE_TAG}"
  echo ">>> ${DS_NAME}  n=${DS_SAMPLES}  temp=${TEMP}  ${MODE_TAG}" >> "$SUMMARY"

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
    "$@" \
    "${PROFILE_FLAG[@]}" \
    2>&1 | tee "$LOG_FILE"
  local RC="${PIPESTATUS[0]}"

  local SPEEDUP ACC NODES
  SPEEDUP=$(grep "Decoding speedup:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+$' || echo "N/A")
  ACC=$(grep "Average Acceptance length:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+$' || echo "N/A")
  NODES=$(grep "Average tree node count:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+$' || echo "N/A")

  local LINE="${DS_NAME}  n=${DS_SAMPLES}  temp=${TEMP}  ${MODE_TAG}  mts=${MTS}  speedup=${SPEEDUP}  avg_accept=${ACC}  avg_nodes=${NODES}  exit=${RC}"
  echo "$LINE"
  echo "$LINE" >> "$SUMMARY"

  PORT=$((PORT + 1))
  return "$RC"
}

MODES_PER_TEMP=4
TOTAL=$(( MODES_PER_TEMP * ${#DATASETS[@]} * ${#TEMPERATURES[@]} ))
RUN=0

{
  echo ""
  echo "========================================================================"
  echo "Paper four-tree sweep — $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "Modes: v1_thresh | v2_eagle2 | v3_bestfirst | v4_prefixaware"
  echo "Datasets (${#DATASETS[@]}): ${DATASETS[*]}"
  echo "Temperatures: ${TEMPERATURES[*]}"
  echo "max_tree_size=${MTS}  expand_k=${EXPAND_K}  top_k=${TOP_K}  profile=${USE_PROFILE}"
  echo "Total runs: ${TOTAL}"
  echo "========================================================================"
} | tee -a "$SUMMARY"

for TEMP in "${TEMPERATURES[@]}"; do
  for DS_SPEC in "${DATASETS[@]}"; do
    IFS=':' read -r DS_NAME DS_SAMPLES <<< "$DS_SPEC"

    # (1) v1 — threshold + cartesian cap
    RUN=$((RUN + 1))
    echo ""
    echo ">>> [${RUN}/${TOTAL}] v1_thresh"
    run_one "$DS_NAME" "$DS_SAMPLES" "$TEMP" "v1_thresh" \
      --dynamic-branching \
      --tree-version 1 \
      --max-tree-size "$MTS" \
      --top-k "$TOP_K" \
      --theta-uni 0.9 --theta-bi 0.3 --theta-tri 0.1

    # (2) v2 — EAGLE-2
    RUN=$((RUN + 1))
    echo ""
    echo ">>> [${RUN}/${TOTAL}] v2_eagle2"
    run_one "$DS_NAME" "$DS_SAMPLES" "$TEMP" "v2_eagle2" \
      --dynamic-branching \
      --tree-version 2 \
      --max-tree-size "$MTS" \
      --expand-k "$EXPAND_K"

    # (3) v3 — best-first
    RUN=$((RUN + 1))
    echo ""
    echo ">>> [${RUN}/${TOTAL}] v3_bestfirst"
    run_one "$DS_NAME" "$DS_SAMPLES" "$TEMP" "v3_bestfirst" \
      --dynamic-branching \
      --tree-version 3 \
      --max-tree-size "$MTS" \
      --expand-k "$EXPAND_K"

    # (4) v4 — prefix-aware greedy
    RUN=$((RUN + 1))
    echo ""
    echo ">>> [${RUN}/${TOTAL}] v4_prefixaware"
    run_one "$DS_NAME" "$DS_SAMPLES" "$TEMP" "v4_prefixaware" \
      --dynamic-branching \
      --tree-version 4 \
      --max-tree-size "$MTS" \
      --expand-k "$EXPAND_K"
  done
done

echo ""
echo "========================================================================"
echo "Sweep complete — summary lines:"
echo "========================================================================"
grep "speedup=" "$SUMMARY" | grep "avg_accept=" || true
echo ""
echo "Full log: $SUMMARY"
