#!/bin/bash
# Master pipeline.  Runs end-to-end:
#   1. Waits for the currently-running v1 training to finish.
#   2. Kicks off v2/v3/v4 training variants (via train_queue.sh) on GPUs 0,1.
#   3. Per-variant: starts a watcher that auto-evals every step_* checkpoint
#      as soon as it appears, on eval GPUs 6,7.
#   4. When all variants done, picks the best checkpoint by math500-16 score
#      and runs a "finalist" evaluation (math500 256 samples).
#   5. If the finalist beats the stock baseline (8.27 on math500), runs
#      cross-dataset eval (mt-bench, gsm8k, humaneval) and writes a summary.
#
# Usage (run in the background):
#   nohup ./master_pipeline.sh > master.log 2>&1 &

set -uo pipefail

ROOT=/homes/videetm/dflash
cd "$ROOT"

LOGDIR="$ROOT/logs/session_apr18"
RESULTS_TSV="$LOGDIR/mix_eval.tsv"
RECIPES="$LOGDIR/recipes.txt"
TRAIN_GPUS="0,1"
EVAL_GPUS="6,7"
STOCK_SPEEDUP=8.27  # math500 256-sample baseline for v7 B=128 ek=8

log() { echo "[pipeline] $*" | tee -a "$LOGDIR/pipeline.log"; }

# --- 1. Wait for v1 training to finish --------------------------------------
V1_LOG="$LOGDIR/mix_10k_marg.log"
log "Waiting for v1 training to complete..."
# v1 is complete when the final epoch checkpoint appears.
until [ -d "$ROOT/trainingto/dflash_mix_10k/epoch_0" ]; do
  sleep 30
done
touch "$ROOT/trainingto/dflash_mix_10k/.DONE_TRAINING"
log "v1 training done."

# --- 2. Kick off v2/v3/v4 queue --------------------------------------------
log "Starting train_queue for v2/v3/v4 on gpus $TRAIN_GPUS..."
cd "$ROOT/trainingto"
./train_queue.sh "$TRAIN_GPUS" "$RECIPES" > "$LOGDIR/train_queue.log" 2>&1 &
QUEUE_PID=$!
log "train_queue pid=$QUEUE_PID"

# Start per-variant watchers as soon as the savedir appears.
for VARIANT in dflash_mix_10k_v2_varblock dflash_mix_10k_v3_ctrlite dflash_mix_10k_v4_ttlite; do
  # Start the watcher eagerly; it will loop even before the dir exists.
  mkdir -p "$ROOT/trainingto/$VARIANT"
  nohup "$ROOT/trainingto/watch_and_eval.sh" "$VARIANT" "$EVAL_GPUS" 16 "$RESULTS_TSV" \
    > "$LOGDIR/watcher_${VARIANT}.log" 2>&1 &
  log "started watcher for $VARIANT pid=$!"
done

# --- 3. Wait for all training to finish ------------------------------------
log "Waiting for all variants to finish..."
wait "$QUEUE_PID" || true
log "All training done."

# Give watchers time to eval the final epoch checkpoints.
sleep 60

# Kill the watchers (sentinels are set by train_queue).
pkill -f watch_and_eval.sh || true

# --- 4. Pick best checkpoint -----------------------------------------------
log "Picking best checkpoint by speedup on math500 (16-sample)..."
BEST=$(awk -F'\t' 'NR>1 && $1 !~ /^STOCK/ && $4 > 0 {print $4"\t"$1}' "$RESULTS_TSV" \
       | sort -rn | head -1)
BEST_SPEEDUP=$(echo "$BEST" | awk -F'\t' '{print $1}')
BEST_CKPT=$(echo "$BEST" | awk -F'\t' '{print $2}')
log "Best: $BEST_CKPT speedup=$BEST_SPEEDUP"

if [ -z "$BEST_CKPT" ] || [ "$BEST_CKPT" = "" ]; then
  log "No valid checkpoint found.  Exiting."
  exit 1
fi

# --- 5. Finalist eval ------------------------------------------------------
BEST_HF="${BEST_CKPT}_hf"
if [ ! -d "$BEST_HF" ]; then
  log "Building HF dir for $BEST_CKPT..."
  python "$ROOT/trainingto/convert_ckpt_to_hf.py" \
    --base z-lab/Qwen3-4B-DFlash-b16 \
    --ckpt "${BEST_CKPT}/pytorch_model.bin" \
    --out  "$BEST_HF"
fi

log "Finalist eval: math500 256 samples..."
( cd "$ROOT" && CUDA_VISIBLE_DEVICES="$EVAL_GPUS" torchrun \
  --nproc_per_node=2 --master_port=29700 \
  benchmark.py \
  --dataset math500 --max-samples 256 \
  --model-name-or-path Qwen/Qwen3-4B \
  --draft-name-or-path "$BEST_HF" \
  --tree-version 7 --max-tree-size 128 --expand-k 8 \
  --temperature 0.0 > "$LOGDIR/finalist_math500.log" 2>&1 )

FINAL_SPEEDUP=$(grep "Decoding speedup:" "$LOGDIR/finalist_math500.log" | tail -1 | awk '{print $NF}')
FINAL_TAU=$(grep "Average Acceptance length:" "$LOGDIR/finalist_math500.log" | tail -1 | awk '{print $NF}')
log "Finalist math500 256s: speedup=$FINAL_SPEEDUP tau=$FINAL_TAU"

# Compare to stock.
if python -c "exit(0 if float('$FINAL_SPEEDUP') >= $STOCK_SPEEDUP else 1)"; then
  log "Finalist BEATS or MATCHES stock ($STOCK_SPEEDUP).  Running cross-dataset eval..."
  for DS_CFG in "mt-bench 80" "gsm8k 128" "humaneval 164"; do
    DS=$(echo $DS_CFG | awk '{print $1}')
    N=$(echo $DS_CFG | awk '{print $2}')
    log "  $DS n=$N..."
    ( cd "$ROOT" && CUDA_VISIBLE_DEVICES="$EVAL_GPUS" torchrun \
      --nproc_per_node=2 --master_port=29700 \
      benchmark.py --dataset "$DS" --max-samples "$N" \
      --model-name-or-path Qwen/Qwen3-4B \
      --draft-name-or-path "$BEST_HF" \
      --tree-version 7 --max-tree-size 128 --expand-k 8 \
      --temperature 0.0 > "$LOGDIR/finalist_${DS}.log" 2>&1 )
    SP=$(grep "Decoding speedup:" "$LOGDIR/finalist_${DS}.log" | tail -1 | awk '{print $NF}')
    TAU=$(grep "Average Acceptance length:" "$LOGDIR/finalist_${DS}.log" | tail -1 | awk '{print $NF}')
    log "    $DS speedup=$SP tau=$TAU"
  done
else
  log "Finalist $FINAL_SPEEDUP < stock $STOCK_SPEEDUP.  Not running cross-dataset."
fi

log "=== PIPELINE COMPLETE ==="
log "Best ckpt: $BEST_CKPT  (finalist math500: $FINAL_SPEEDUP)"
log "Results in: $RESULTS_TSV"
