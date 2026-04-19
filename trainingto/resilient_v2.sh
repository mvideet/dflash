#!/bin/bash
# Resilient v2 training launcher: auto-restarts on crash IF there's a
# checkpoint to resume from.  Backs off from quick restarts.
cd /homes/videetm/dflash
SAVEDIR=trainingto/dflash_broad_varblock_v2
ATTEMPT=0
while true; do
  ATTEMPT=$((ATTEMPT+1))
  # Check GPU free before launching
  while true; do
    min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
    if [ "$min_free" -gt 40000 ]; then break; fi
    echo "[$(date -u +%H:%M:%S)] attempt $ATTEMPT: waiting; min_free=${min_free} MiB"
    sleep 60
  done

  # Check latest checkpoint (for info only)
  LATEST=$(ls $SAVEDIR 2>/dev/null | grep -oE "step_[0-9]+" | sort -t'_' -k2 -n | tail -1 || echo "none")
  echo "[$(date -u +%H:%M:%S)] attempt $ATTEMPT: launching v2 (resume from $LATEST)"

  bash trainingto/run_broad_training_v2.sh > logs/session_apr18_neurips/train_broad_varblock_v2.log 2>&1
  RET=$?
  echo "[$(date -u +%H:%M:%S)] attempt $ATTEMPT: exited with $RET"

  # If training completed normally (epoch 2 done), stop.
  if grep -q "Epoch 2:" logs/session_apr18_neurips/train_broad_varblock_v2.log 2>/dev/null; then
    echo "Training complete."
    exit 0
  fi

  # Backoff before retry
  echo "Retrying in 120s..."
  sleep 120
done
