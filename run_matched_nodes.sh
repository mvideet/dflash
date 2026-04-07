#!/usr/bin/env bash
# ======================================================================
# Matched-node comparison: v4 at v2's and v3's node counts
#
# Runs v4 at mts=8 (~32 nodes, matching v2@mts=32) and
#       v4 at mts=16 (~46 nodes, close to v3@mts=32's 49 nodes)
# alongside v2@mts=32 and v3@mts=32 on all 11 datasets.
#
# Usage: ./run_matched_nodes.sh
# Results: logs/matched_nodes_summary.txt
# ======================================================================
set -uo pipefail

cd "$(dirname "$0")"
mkdir -p logs
unset TORCH_LOGS

SUMMARY="logs/matched_nodes_summary.txt"

NPROC="${NPROC_PER_NODE:-8}"
PORT="${MASTER_PORT:-29800}"
MODEL="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B}"
DRAFT="${DRAFT_NAME_OR_PATH:-z-lab/Qwen3-4B-DFlash-b16}"
MAX_TOKENS="${MAX_NEW_TOKENS:-2048}"
EXPAND_K=3
TOP_K=3

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

run_one() {
  local DS_NAME="$1"
  local DS_SAMPLES="$2"
  local MODE_TAG="$3"
  shift 3

  local LOG_FILE="logs/matched_${DS_NAME}_${MODE_TAG}.log"

  echo ""
  echo ">>> ${DS_NAME}  ${MODE_TAG}"
  echo ">>> ${DS_NAME}  ${MODE_TAG}" >> "$SUMMARY"

  torchrun \
    --nproc_per_node="$NPROC" \
    --master_port="$PORT" \
    benchmark.py \
    --dataset "$DS_NAME" \
    --max-samples "$DS_SAMPLES" \
    --model-name-or-path "$MODEL" \
    --draft-name-or-path "$DRAFT" \
    --max-new-tokens "$MAX_TOKENS" \
    --temperature 0.0 \
    "$@" \
    2>&1 | tee "$LOG_FILE"
  local RC="${PIPESTATUS[0]}"

  local SPEEDUP ACC NODES
  SPEEDUP=$(grep "Decoding speedup:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+$' || echo "N/A")
  ACC=$(grep "Average Acceptance length:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+$' || echo "N/A")
  NODES=$(grep "Average tree node count:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+$' || echo "N/A")

  local LINE="${DS_NAME}  ${MODE_TAG}  speedup=${SPEEDUP}  avg_accept=${ACC}  avg_nodes=${NODES}  exit=${RC}"
  echo "$LINE"
  echo "$LINE" >> "$SUMMARY"

  PORT=$((PORT + 1))
}

TOTAL=$(( 4 * ${#DATASETS[@]} ))
RUN=0

{
  echo ""
  echo "========================================================================"
  echo "Matched-node comparison — $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "  v2@mts=32 (~33 nodes)  vs  v4@mts=8 (~32 nodes)"
  echo "  v3@mts=32 (~49 nodes)  vs  v4@mts=16 (~46 nodes)"
  echo "Total runs: ${TOTAL}"
  echo "========================================================================"
} | tee -a "$SUMMARY"

for DS_SPEC in "${DATASETS[@]}"; do
  IFS=':' read -r DS_NAME DS_SAMPLES <<< "$DS_SPEC"

  # v2 @ mts=32 -> ~33 nodes
  RUN=$((RUN + 1))
  echo ">>> [${RUN}/${TOTAL}]"
  run_one "$DS_NAME" "$DS_SAMPLES" "v2_mts32" \
    --dynamic-branching --tree-version 2 --max-tree-size 32 --expand-k "$EXPAND_K"

  # v4 @ mts=8 -> ~32 nodes (matched to v2)
  RUN=$((RUN + 1))
  echo ">>> [${RUN}/${TOTAL}]"
  run_one "$DS_NAME" "$DS_SAMPLES" "v4_mts8" \
    --dynamic-branching --tree-version 4 --max-tree-size 8 --expand-k "$EXPAND_K"

  # v3 @ mts=32 -> ~49 nodes
  RUN=$((RUN + 1))
  echo ">>> [${RUN}/${TOTAL}]"
  run_one "$DS_NAME" "$DS_SAMPLES" "v3_mts32" \
    --dynamic-branching --tree-version 3 --max-tree-size 32 --expand-k "$EXPAND_K"

  # v4 @ mts=16 -> ~46 nodes (matched to v3)
  RUN=$((RUN + 1))
  echo ">>> [${RUN}/${TOTAL}]"
  run_one "$DS_NAME" "$DS_SAMPLES" "v4_mts16" \
    --dynamic-branching --tree-version 4 --max-tree-size 16 --expand-k "$EXPAND_K"
done

echo ""
echo "========================================================================"
echo "Sweep complete — summary:"
echo "========================================================================"
echo ""
echo "--- v4@mts=8 (~32 nodes) vs v2@mts=32 (~33 nodes) ---"
grep "v2_mts32\|v4_mts8" "$SUMMARY" | grep "avg_accept" || true
echo ""
echo "--- v4@mts=16 (~46 nodes) vs v3@mts=32 (~49 nodes) ---"
grep "v3_mts32\|v4_mts16" "$SUMMARY" | grep "avg_accept" || true
echo ""
echo "Full log: $SUMMARY"
