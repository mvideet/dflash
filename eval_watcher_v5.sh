#!/bin/bash
# Eval watcher: poll trainingto/dflash_rl_v5/ for new step_*_hf checkpoints
# and bench each on mt-bench (N=80, v7+B=160). Append to TSV.
#
# Usage: bash eval_watcher_v5.sh [gpus]
# Default GPUs: 4,5,6,7 (v5 trains on 0).
set -uo pipefail
cd /homes/videetm/dflash

GPUS=${1:-4,5,6,7}
NPROC=$(echo $GPUS | awk -F, '{print NF}')
PORT=29571
MODEL=Qwen/Qwen3-4B
SAVEDIR=trainingto/dflash_rl_v5
LOGDIR=logs/v5_mt_eval
TSV=$LOGDIR/v5_mt_results.tsv
PY=/homes/videetm/miniforge3/envs/dflash312/bin/torchrun

mkdir -p "$LOGDIR"
[ -f "$TSV" ] || echo -e "step\tspeedup\ttau\tnodes\tts" > "$TSV"

bench() {
  local CKPT=$1 TAG=$2
  local LOG="$LOGDIR/${TAG}.log"
  [ -f "$LOG" ] && grep -q "Decoding speedup:" "$LOG" && { echo "[skip] $TAG"; return; }
  echo "[$(date -u +%H:%M:%S)] === bench $TAG ==="
  CUDA_VISIBLE_DEVICES=$GPUS timeout 1500 $PY \
    --nproc_per_node=$NPROC --master_port=$PORT benchmark.py \
    --dataset mt-bench --max-samples 80 \
    --model-name-or-path "$MODEL" --draft-name-or-path "$CKPT" \
    --max-new-tokens 1024 --temperature 0.0 \
    --tree-version 7 --max-tree-size 160 --expand-k 8 \
    --block-size 16 > "$LOG" 2>&1 || echo "  (crash)"
  local SP=$(grep "Decoding speedup:" "$LOG" | tail -1 | awk '{print $3}')
  local TAU=$(grep "Average Acceptance length:" "$LOG" | tail -1 | awk '{print $4}')
  local ND=$(grep "Average tree node count:" "$LOG" | tail -1 | awk '{print $5}')
  local STEP=$(echo "$TAG" | grep -oE '[0-9]+' | tail -1)
  printf "%s\t%s\t%s\t%s\t%s\n" "$STEP" "$SP" "$TAU" "$ND" "$(date -u +%H:%M:%S)" >> "$TSV"
  echo "  step=$STEP spd=$SP tau=$TAU nodes=$ND"
}

# Stock baseline once at start
bench "z-lab/Qwen3-4B-DFlash-b16" "stock_baseline_mt_b16"

echo "[$(date -u +%H:%M:%S)] watching $SAVEDIR ..."
seen=""
while true; do
  # Stop if v5 process is gone and we've evaluated final_hf
  if ! pgrep -f "train_rl.py.*dflash_rl_v5" >/dev/null 2>&1; then
    if [ -d "$SAVEDIR/final_hf" ] && echo "$seen" | grep -q "final_hf"; then
      echo "[$(date -u +%H:%M:%S)] training done + final_hf evaluated, exiting"
      break
    fi
  fi
  for D in "$SAVEDIR"/step_*_hf "$SAVEDIR"/final_hf; do
    [ -d "$D" ] || continue
    TAG=$(basename "$D")
    echo "$seen" | grep -q "$TAG" && continue
    bench "$D" "$TAG"
    seen="$seen $TAG"
  done
  sleep 60
done
