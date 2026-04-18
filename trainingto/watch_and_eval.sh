#!/bin/bash
# Watch a training savedir for new step_* checkpoints and auto-eval each.
# Polls every 30s.  Once a checkpoint is eval'd it is renamed to step_N_done
# so the next iteration skips it.
#
# Usage: ./watch_and_eval.sh <savedir> <eval_gpus> <n_samples> <results_tsv>

set -uo pipefail

SAVEDIR="$1"
GPUS="$2"
N="${3:-16}"
OUT_TSV="${4:-/homes/videetm/dflash/logs/session_apr18/mix_eval.tsv}"

SENTINEL="${SAVEDIR}/.DONE_TRAINING"

echo "[watch] watching $SAVEDIR, gpus=$GPUS, n_samples=$N"

while true; do
  # List candidate checkpoints — step_* dirs that are NOT yet evaluated.
  for d in "$SAVEDIR"/step_*; do
    [ -d "$d" ] || continue
    [[ "$d" == *_hf ]] && continue
    MARKER="${d}/.evaled"
    [ -f "$MARKER" ] && continue
    [ -f "${d}/pytorch_model.bin" ] || continue
    # Make sure the file isn't currently being written — check size stability.
    SZ1=$(stat -c %s "${d}/pytorch_model.bin")
    sleep 2
    SZ2=$(stat -c %s "${d}/pytorch_model.bin")
    [ "$SZ1" != "$SZ2" ] && continue  # still writing
    echo "[watch] eval $d"
    if /homes/videetm/dflash/trainingto/eval_ckpt.sh "$d" "$GPUS" "$N" "$OUT_TSV" >> /homes/videetm/dflash/logs/session_apr18/watcher.log 2>&1; then
      touch "$MARKER"
    else
      echo "[watch] eval FAILED for $d" >&2
      touch "$MARKER"  # don't retry
    fi
  done
  # Stop if the training sentinel is set.
  if [ -f "$SENTINEL" ]; then
    echo "[watch] training sentinel seen; exiting watcher"
    break
  fi
  sleep 30
done
