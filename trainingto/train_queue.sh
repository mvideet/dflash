#!/bin/bash
# Sequential training queue: runs multiple training variants on the same train
# GPUs, each saving to its own savedir.  Creates a .DONE_TRAINING sentinel
# in each savedir when its run completes so the watcher can exit.
#
# Usage: ./train_queue.sh <train_gpus> <recipe_list_file>
#
# recipe_list_file has one line per variant:
#   SAVEDIR|TRAINPATH|EXTRA_ARGS
# e.g.
#   dflash_mix_10k_v2|data/nemotron_math_10k/train.jsonl|--block-sizes 12,16 --ctr-weight 0.3
#
# Each variant gets --num_epochs 1 and saves every 500 steps.

set -uo pipefail

TRAIN_GPUS="$1"
RECIPES_FILE="$2"
N_GPUS=$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)

cd /homes/videetm/dflash/trainingto

while IFS='|' read -r SAVEDIR TRAINPATH EXTRA; do
  [ -z "${SAVEDIR:-}" ] && continue
  [[ "$SAVEDIR" == \#* ]] && continue

  echo "[queue] === $SAVEDIR ==="
  LOG="/homes/videetm/dflash/logs/session_apr18/train_${SAVEDIR}.log"

  deepspeed --include "localhost:${TRAIN_GPUS}" --master_port=29702 main_mix.py \
    --basepath Qwen/Qwen3-4B \
    --draftpath z-lab/Qwen3-4B-DFlash-b16 \
    --trainpath "$TRAINPATH" \
    --testpath  data/nemotron_math_10k/test.jsonl \
    --savedir   "$SAVEDIR" \
    --deepspeed_config ds_config.json \
    --num_epochs 1 \
    --gamma-loss 7.0 \
    --random-anchors --anchors-per-seq 16 \
    --save-every 500 \
    $EXTRA > "$LOG" 2>&1 || echo "[queue] $SAVEDIR FAILED (exit=$?)"

  touch "${SAVEDIR}/.DONE_TRAINING"
  echo "[queue] $SAVEDIR done."
done < "$RECIPES_FILE"

echo "[queue] all recipes complete"
