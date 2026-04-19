#!/bin/bash
# Sweep math500-32 at b=20 across VB v1 training checkpoints.
# Gives a learning curve for the BEST inference setting (b=20), not b=16.
set -u
cd /homes/videetm/dflash

wait_for_8_gpus() {
  while true; do
    min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
    if [ "$min_free" -gt 60000 ]; then
      sleep 30
      min_free_2=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
      [ "$min_free_2" -gt 60000 ] && return 0
    fi
    echo "[$(date -u +%H:%M:%S)] waiting; min_free=${min_free} MiB"
    sleep 60
  done
}

RESULT=logs/session_apr18_neurips/vb_ckpt_sweep_b20.tsv
[ ! -f "$RESULT" ] && echo -e "step\tspeedup\ttau\tnodes" > $RESULT

for STEP in 500 2000 4000 6000 8000 10000 12000 14000 16000 18500; do
  CKPT=trainingto/dflash_broad_varblock_v1/step_${STEP}_hf
  if [ ! -d "$CKPT" ]; then
    echo "[skip] $STEP: checkpoint dir missing"; continue
  fi
  LOG=logs/session_apr18_neurips/cs20_step${STEP}.log
  if [ -f "$LOG" ] && grep -q "Decoding speedup:" "$LOG"; then
    echo "[skip] step $STEP already done"; continue
  fi
  wait_for_8_gpus
  echo "=== step_${STEP} b=20 ==="
  /homes/videetm/miniforge3/envs/dflash312/bin/torchrun --nproc_per_node=8 --master_port=29501 benchmark.py \
    --dataset math500 --max-samples 32 \
    --model-name-or-path Qwen/Qwen3-4B \
    --draft-name-or-path "$CKPT" \
    --tree-version 7 --max-tree-size 128 --expand-k 8 \
    --temperature 0.0 \
    --block-size 20 > "$LOG" 2>&1 || echo "(crash)"
  SP=$(grep "Decoding speedup:" "$LOG" | tail -1 | awk '{print $NF}')
  TAU=$(grep "Average Acceptance length:" "$LOG" | tail -1 | awk '{print $NF}')
  ND=$(grep "Average tree node count:" "$LOG" | tail -1 | awk '{print $NF}')
  printf "%s\t%s\t%s\t%s\n" "$STEP" "$SP" "$TAU" "$ND" >> $RESULT
  echo "step_${STEP}: speedup=$SP tau=$TAU nodes=$ND"
done
echo "=== ckpt sweep b=20 complete ==="
cat $RESULT
