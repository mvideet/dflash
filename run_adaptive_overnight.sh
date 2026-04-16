#!/usr/bin/env bash
# Overnight sweep: v4 + EWMA adaptive tree sizing + adaptive expand_k
# across all standard datasets and temperatures.
#
# Best config from sweep: decay=0.8, min_tree=12, expand_k 2→5
#
# Usage:
#   ./run_adaptive_overnight.sh
#   MODEL_NAME_OR_PATH=... DRAFT_NAME_OR_PATH=... ./run_adaptive_overnight.sh
#
# Results: logs/adaptive_overnight_summary.txt
set -uo pipefail

cd "$(dirname "$0")"
mkdir -p logs
unset TORCH_LOGS

SUMMARY="logs/adaptive_overnight_summary.txt"

NPROC="${NPROC_PER_NODE:-8}"
PORT="${MASTER_PORT:-29700}"
MODEL="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B}"
DRAFT="${DRAFT_NAME_OR_PATH:-z-lab/Qwen3-4B-DFlash-b16}"
MAX_TOKENS="${MAX_NEW_TOKENS:-2048}"

MTS="${ADAPTIVE_MTS:-32}"
EXPAND_K="${ADAPTIVE_EXPAND_K:-3}"
MIN_TREE="${ADAPTIVE_MIN_TREE:-12}"
EWMA_DECAY="${ADAPTIVE_EWMA_DECAY:-0.8}"
MIN_EXPAND_K="${ADAPTIVE_MIN_EXPAND_K:-2}"
MAX_EXPAND_K="${ADAPTIVE_MAX_EXPAND_K:-5}"

TEMPERATURES=("0.0" "0.6")

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

TOTAL=$(( ${#DATASETS[@]} * ${#TEMPERATURES[@]} ))
RUN=0

{
  echo ""
  echo "========================================================================"
  echo "Adaptive overnight sweep — $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "Strategy: v4 + EWMA (tree_size + expand_k)"
  echo "Datasets (${#DATASETS[@]}): ${DATASETS[*]}"
  echo "Temperatures: ${TEMPERATURES[*]}"
  echo "max_tree_size=${MTS}  min_tree_size=${MIN_TREE}"
  echo "expand_k=${EXPAND_K}  min_expand_k=${MIN_EXPAND_K}  max_expand_k=${MAX_EXPAND_K}"
  echo "ewma_decay=${EWMA_DECAY}"
  echo "Total runs: ${TOTAL}"
  echo "========================================================================"
} | tee -a "$SUMMARY"

for TEMP in "${TEMPERATURES[@]}"; do
  for DS_SPEC in "${DATASETS[@]}"; do
    IFS=':' read -r DS_NAME DS_SAMPLES <<< "$DS_SPEC"

    RUN=$((RUN + 1))
    TAG="v4_ewma_ek${MIN_EXPAND_K}-${MAX_EXPAND_K}_t${TEMP}"
    LOG_FILE="logs/adaptive_${DS_NAME}_${TAG}.log"

    echo ""
    echo ">>> [${RUN}/${TOTAL}] ${DS_NAME}  n=${DS_SAMPLES}  temp=${TEMP}  v4+ewma+ek"
    echo ">>> [${RUN}/${TOTAL}] ${DS_NAME}  n=${DS_SAMPLES}  temp=${TEMP}  v4+ewma+ek" >> "$SUMMARY"

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
      --tree-version 4 \
      --max-tree-size "$MTS" \
      --expand-k "$EXPAND_K" \
      --adaptive-block \
      --adaptive-block-ewma-decay "$EWMA_DECAY" \
      --adaptive-block-min-tree-size "$MIN_TREE" \
      --adaptive-block-min-expand-k "$MIN_EXPAND_K" \
      --adaptive-block-max-expand-k "$MAX_EXPAND_K" \
      2>&1 | tee "$LOG_FILE"
    RC="${PIPESTATUS[0]}"

    SPEEDUP=$(grep "Decoding speedup:"         "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP '[\d.]+$' || echo "N/A")
    ACC=$(    grep "Average Acceptance length:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP '[\d.]+$' || echo "N/A")
    NODES=$(  grep "Average tree node count:"  "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP '[\d.]+$' || echo "N/A")

    LINE="${DS_NAME}  n=${DS_SAMPLES}  temp=${TEMP}  v4_ewma_ek  mts=${MTS}  speedup=${SPEEDUP}  avg_accept=${ACC}  avg_nodes=${NODES}  exit=${RC}"
    echo "$LINE"
    echo "$LINE" >> "$SUMMARY"

    PORT=$((PORT + 1))
  done
done

echo ""
echo "========================================================================"
echo "Sweep complete — result rows:"
echo "========================================================================"
grep "speedup=" "$SUMMARY" | grep "avg_accept=" || true
echo ""
echo "Full log: $SUMMARY"
