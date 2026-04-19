#!/bin/bash
# Adaptive resilient v2 launcher: uses WHATEVER GPUs are currently free.
# Dynamically picks GPUs with >40 GB free and launches deepspeed with
# --include on those.  Scales gradient_accumulation inversely with GPU
# count to keep effective batch constant.
cd /homes/videetm/dflash
SAVEDIR=trainingto/dflash_broad_varblock_v2
ATTEMPT=0
MIN_GPUS=2
TARGET_EFFECTIVE_BS=16  # 8 GPUs × 1 × 2 grad_accum = 16
while true; do
  ATTEMPT=$((ATTEMPT+1))
  # Discover free GPUs (>40 GB free)
  FREE_GPUS=""
  while IFS=, read -r idx free; do
    idx=$(echo $idx | tr -d ' ')
    free=$(echo $free | tr -d ' ')
    if [ "$free" -gt 40000 ]; then
      FREE_GPUS="$FREE_GPUS,$idx"
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  FREE_GPUS="${FREE_GPUS#,}"  # strip leading comma
  NGPUS=$(echo "$FREE_GPUS" | tr -cd , | wc -c)
  NGPUS=$((NGPUS+1))

  if [ -z "$FREE_GPUS" ] || [ "$NGPUS" -lt "$MIN_GPUS" ]; then
    echo "[$(date -u +%H:%M:%S)] attempt $ATTEMPT: only $NGPUS GPU(s) free ($FREE_GPUS); waiting..."
    sleep 60
    continue
  fi

  # Scale grad_accum: desired_eff_bs / ngpus
  GRAD_ACCUM=$((TARGET_EFFECTIVE_BS / NGPUS))
  [ "$GRAD_ACCUM" -lt 1 ] && GRAD_ACCUM=1

  echo "[$(date -u +%H:%M:%S)] attempt $ATTEMPT: $NGPUS GPUs free ($FREE_GPUS); grad_accum=$GRAD_ACCUM"

  # Patch ds_config.json temporarily for this launch
  cat > /tmp/ds_config_adapt.json <<EOF
{
    "bf16": {"enabled": true},
    "zero_allow_untested_optimizer": true,
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {"warmup_min_lr": 1e-7, "warmup_max_lr": 5e-6,
                   "warmup_num_steps": 200, "total_num_steps": 100000}
    },
    "zero_optimization": {
        "stage": 1, "allgather_partitions": true,
        "allgather_bucket_size": 2e8, "overlap_comm": true,
        "reduce_scatter": true, "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": ${GRAD_ACCUM},
    "gradient_clipping": 0.5,
    "steps_per_print": 2000,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": false
}
EOF

  cd /homes/videetm/dflash/trainingto
  export PATH=/homes/videetm/miniforge3/envs/dflash/bin:$PATH
  export DFLASH_ATTN_IMPL=sdpa
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
  export NCCL_TIMEOUT=3600000

  deepspeed --include "localhost:${FREE_GPUS}" --master_port=29703 main_mix.py \
      --basepath Qwen/Qwen3-4B \
      --draftpath z-lab/Qwen3-4B-DFlash-b16 \
      --trainpath data/nemotron_broad_150k/train.jsonl \
      --testpath  data/nemotron_broad_150k/test.jsonl \
      --savedir   dflash_broad_varblock_v2 \
      --deepspeed_config /tmp/ds_config_adapt.json \
      --num_epochs 3 \
      --gamma-loss 10.0 \
      --random-anchors --anchors-per-seq 32 \
      --block-sizes 12,16,20,24,28,32 \
      --block-size-probs 1,2,2,2,1,1 \
      --ctr-weight 0.3 \
      --ttt-weight 0.0 \
      --max-tree-size 16 \
      --tree-expand-k 5 \
      --save-every 1000 \
      > /homes/videetm/dflash/logs/session_apr18_neurips/train_broad_varblock_v2.log 2>&1
  RET=$?
  cd /homes/videetm/dflash
  echo "[$(date -u +%H:%M:%S)] attempt $ATTEMPT: exited with $RET (used ${NGPUS} GPUs)"

  if grep -q "Epoch 2:" logs/session_apr18_neurips/train_broad_varblock_v2.log 2>/dev/null; then
    echo "Training complete."
    exit 0
  fi

  sleep 60
done
