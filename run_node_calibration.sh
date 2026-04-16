#!/usr/bin/env bash
# Quick calibration: find the max_tree_size that makes v2 and v4 average ~70 trie nodes.
# Runs on a small mt-bench slice (20 samples) to be fast (~3 min per config).
#
# Usage:  ./run_node_calibration.sh
set -uo pipefail

cd "$(dirname "$0")"
mkdir -p logs
unset TORCH_LOGS

SUMMARY="logs/node_calibration.txt"
NPROC="${NPROC_PER_NODE:-8}"
PORT="${MASTER_PORT:-29800}"
MODEL="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B}"
DRAFT="${DRAFT_NAME_OR_PATH:-z-lab/Qwen3-4B-DFlash-b16}"
EXPAND_K=3

# Small slice — just enough to get a stable avg_nodes estimate
SAMPLES=20
TEMP="0.0"

# MTS values to probe for each tree version
MTS_VALUES=(24 28 32 36 40 44)

{
  echo ""
  echo "================================================================"
  echo "Node calibration — $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "target: ~70 avg trie nodes"
  echo "samples=${SAMPLES}  temp=${TEMP}  expand_k=${EXPAND_K}"
  echo "mts values: ${MTS_VALUES[*]}"
  echo "================================================================"
} | tee -a "$SUMMARY"

for TV in 2 4; do
  TV_NAME="v${TV}"
  [ "$TV" -eq 2 ] && TV_NAME="v2_eagle2"
  [ "$TV" -eq 4 ] && TV_NAME="v4_prefixaware"

  for MTS in "${MTS_VALUES[@]}"; do
    LOG_FILE="logs/calib_${TV_NAME}_mts${MTS}.log"

    echo ""
    echo ">>> ${TV_NAME}  mts=${MTS}"
    echo ">>> ${TV_NAME}  mts=${MTS}" >> "$SUMMARY"

    torchrun \
      --nproc_per_node="$NPROC" \
      --master_port="$PORT" \
      benchmark.py \
      --dataset mt-bench \
      --max-samples "$SAMPLES" \
      --model-name-or-path "$MODEL" \
      --draft-name-or-path "$DRAFT" \
      --max-new-tokens 512 \
      --temperature "$TEMP" \
      --dynamic-branching \
      --tree-version "$TV" \
      --max-tree-size "$MTS" \
      --expand-k "$EXPAND_K" \
      2>&1 | tee "$LOG_FILE"

    NODES=$(grep "Average tree node count:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+$' || echo "N/A")
    ACC=$(  grep "Average Acceptance length:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+$' || echo "N/A")

    LINE="${TV_NAME}  mts=${MTS}  avg_nodes=${NODES}  avg_accept=${ACC}"
    echo "$LINE"
    echo "$LINE" >> "$SUMMARY"

    PORT=$((PORT + 1))
  done
done

echo ""
echo "================================================================"
echo "Calibration complete. Node counts by config:"
echo "================================================================"
grep "avg_nodes=" "$SUMMARY" | grep -v "^>>>" || true
echo ""
echo "Pick the mts closest to 70 nodes for each version."
echo "Full log: $SUMMARY"
