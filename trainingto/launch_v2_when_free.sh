#!/bin/bash
# Waits for all 8 GPUs to have >40 GiB free, then launches VB v2.
cd /homes/videetm/dflash
while true; do
  min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
  ts=$(date -u +"%H:%M:%S")
  if [ "$min_free" -gt 40000 ]; then
    echo "[$ts] GPUs free (min_free=${min_free} MiB); launching VB v2..."
    rm -rf trainingto/dflash_broad_varblock_v2
    bash trainingto/run_broad_training_v2.sh > logs/session_apr18_neurips/train_broad_varblock_v2.log 2>&1 &
    echo "[$ts] Training PID: $!"
    exit 0
  fi
  echo "[$ts] waiting; min_free=${min_free} MiB"
  sleep 60
done
