#!/usr/bin/env bash
# Hyperparameter sweep: find the best EWMA (decay, min_tree_size) config
# that maximises speedup at fixed verification node counts.
#
# Baselines to beat (~70 nodes, greedy):
#   v4 fixed  mts=32  avg_nodes=70  tau≈5.60  speedup≈3.84
#   v2 fixed  mts=69  avg_nodes=70  tau=5.56   speedup=3.80
#
# Grid: 5 ewma_decay × 6 min_tree_size = 30 configs
# Datasets: mt-bench (80), gsm8k (256), humaneval (164) — 3 datasets × 30 = 90 runs
# ETA: ~10 min/run × 90 = ~15 hours
#
# Usage:  nohup ./run_ewma_sweep.sh > logs/ewma_sweep_driver.log 2>&1 &
set -uo pipefail

cd "$(dirname "$0")"
mkdir -p logs
unset TORCH_LOGS

SUMMARY="logs/ewma_sweep_summary.txt"
NPROC="${NPROC_PER_NODE:-8}"
PORT="${MASTER_PORT:-29900}"
MODEL="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B}"
DRAFT="${DRAFT_NAME_OR_PATH:-z-lab/Qwen3-4B-DFlash-b16}"

MAX_TOKENS=2048
TEMP="0.0"
MTS=32       # max ceiling — same as v4 fixed
EXPAND_K=3

# Three datasets: chat, math reasoning, code
DATASETS=(
  "mt-bench:80"
  "gsm8k:256"
  "humaneval:164"
)

# Sweep grid
EWMA_DECAYS=(0.5 0.6 0.7 0.8 0.9)
MIN_TREE_SIZES=(4 8 12 16 20 24)

TOTAL=$(( ${#EWMA_DECAYS[@]} * ${#MIN_TREE_SIZES[@]} * ${#DATASETS[@]} ))
RUN=0

{
  echo ""
  echo "========================================================================"
  echo "EWMA hyperparameter sweep — $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "Grid: ewma_decay × min_tree_size × datasets = ${TOTAL} runs"
  echo "EWMA decays:     ${EWMA_DECAYS[*]}"
  echo "Min tree sizes:  ${MIN_TREE_SIZES[*]}"
  echo "Datasets:        ${DATASETS[*]}"
  echo "max_tokens=${MAX_TOKENS}  temp=${TEMP}  mts_ceiling=${MTS}"
  echo "Baselines to beat at ~70 nodes:"
  echo "  v4 fixed  tau≈5.60  speedup≈3.84  nodes=70"
  echo "  v2 fixed  tau=5.56  speedup=3.80  nodes=70"
  echo "========================================================================"
} | tee -a "$SUMMARY"

for DS_SPEC in "${DATASETS[@]}"; do
  IFS=':' read -r DS_NAME DS_SAMPLES <<< "$DS_SPEC"

  echo ""
  echo "######## Dataset: ${DS_NAME}  n=${DS_SAMPLES} ########"
  echo "######## Dataset: ${DS_NAME}  n=${DS_SAMPLES} ########" >> "$SUMMARY"

  for DECAY in "${EWMA_DECAYS[@]}"; do
    for MIN_TS in "${MIN_TREE_SIZES[@]}"; do
      RUN=$((RUN + 1))
      TAG="${DS_NAME}_decay${DECAY}_min${MIN_TS}"
      LOG_FILE="logs/ewma_sweep_${TAG}.log"

      echo ""
      echo ">>> [${RUN}/${TOTAL}]  ${DS_NAME}  ewma_decay=${DECAY}  min_tree_size=${MIN_TS}"
      echo ">>> [${RUN}/${TOTAL}]  ${DS_NAME}  ewma_decay=${DECAY}  min_tree_size=${MIN_TS}" >> "$SUMMARY"

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
      --adaptive-block-ewma-decay "$DECAY" \
        --adaptive-block-min-tree-size "$MIN_TS" \
        2>&1 | tee "$LOG_FILE"
      RC="${PIPESTATUS[0]}"

      SPEEDUP=$(grep "Decoding speedup:"         "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP '[\d.]+$' || echo "N/A")
      TAU=$(    grep "Average Acceptance length:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP '[\d.]+$' || echo "N/A")
      NODES=$(  grep "Average tree node count:"  "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP '[\d.]+$' || echo "N/A")

      LINE="ds=${DS_NAME}  decay=${DECAY}  min_ts=${MIN_TS}  avg_nodes=${NODES}  tau=${TAU}  speedup=${SPEEDUP}  exit=${RC}"
      echo "$LINE"
      echo "$LINE" >> "$SUMMARY"

      PORT=$((PORT + 1))
    done
  done
done

echo ""
echo "========================================================================"
echo "Sweep complete. All results (sorted by speedup DESC):"
echo "========================================================================"
echo ""
printf "%-12s %-7s %-7s %-10s %-6s %s\n" "dataset" "decay" "min_ts" "avg_nodes" "tau" "speedup"
printf "%-12s %-7s %-7s %-10s %-6s %s\n" "------------" "-------" "-------" "----------" "------" "-------"
grep "^ds=" "$SUMMARY" | sort -t= -k7 -rn | \
  awk -F'[= ]+' '{printf "%-12s %-7s %-7s %-10s %-6s %s\n", $2, $4, $6, $8, $10, $12}'
echo ""

echo ""
echo "========================================================================"
echo "Per-dataset winners (best speedup with tau >= 5.40):"
echo "========================================================================"
for DS_SPEC in "${DATASETS[@]}"; do
  IFS=':' read -r DS_NAME _ <<< "$DS_SPEC"
  echo ""
  echo "  --- ${DS_NAME} ---"
  grep "^ds=${DS_NAME}" "$SUMMARY" | awk -F'[= ]+' '
    $10+0 >= 5.40 {
      printf "  decay=%-5s min_ts=%-4s nodes=%-8s tau=%-6s speedup=%s\n", $4,$6,$8,$10,$12
    }
  ' | sort -t= -k6 -rn | head -5
done
echo ""
echo "Full log: $SUMMARY"
