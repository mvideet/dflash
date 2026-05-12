#!/bin/bash
# Eval RL v3 step_10000_hf at the iter23 config (math500 N=32, v7, B=160,
# b=16, ek=8, T=0). Runs paired stock control immediately after on same
# hardware for clean A/B (eliminates HW noise).
set -uo pipefail
cd /homes/videetm/dflash

GPUS=4,5,6,7
NPROC=4
PORT=29561
MODEL=Qwen/Qwen3-4B
LOGDIR=logs/target_conf_adapt
PY=/homes/videetm/miniforge3/envs/dflash312/bin/torchrun

run() {
  local TAG=$1 DRAFT=$2
  local LOG="${LOGDIR}/iter24_${TAG}.log"
  echo "=== ${TAG} ==="
  CUDA_VISIBLE_DEVICES=$GPUS timeout 1200 ${PY} \
    --nproc_per_node=$NPROC --master_port=$PORT benchmark.py \
    --dataset math500 --max-samples 32 \
    --model-name-or-path "$MODEL" --draft-name-or-path "$DRAFT" \
    --max-new-tokens 1024 --temperature 0.0 \
    --tree-version 7 --max-tree-size 160 --expand-k 8 \
    --block-size 16 > "$LOG" 2>&1 || echo "(crash)"
  grep -E "Decoding speedup|Acceptance length|tree node count" "$LOG" | sed 's/^/  /'
}

run "rl_v3_step10000" "trainingto/dflash_rl_v3/step_10000_hf"
run "stock_control"   "z-lab/Qwen3-4B-DFlash-b16"
