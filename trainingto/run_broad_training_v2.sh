#!/bin/bash
# VB v2: Extended-block curriculum.
#   - Block sizes: b ∈ {12,16,20,24,28,32} weighted {1,2,2,2,1,1}
#     (more sizes, emphasize mid-range, some exposure to b=28,32 OOD)
#   - Epochs: 3 (vs 1 in v1)
#   - Data: same 148k broad-mix
#   - Goal: unlock b=28/32 inference for even higher tau
set -euo pipefail

ROOT=/homes/videetm/dflash
cd "$ROOT/trainingto"

DATA=data/nemotron_broad_150k
SAVE=dflash_broad_varblock_v2
GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}

mkdir -p /homes/videetm/dflash/logs/session_apr18_neurips

export PATH=/homes/videetm/miniforge3/envs/dflash/bin:$PATH
export DFLASH_ATTN_IMPL=sdpa
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_TIMEOUT=3600000

deepspeed --include "localhost:${GPUS}" --master_port=29703 main_mix.py \
    --basepath Qwen/Qwen3-4B \
    --draftpath z-lab/Qwen3-4B-DFlash-b16 \
    --trainpath "$DATA/train.jsonl" \
    --testpath  "$DATA/test.jsonl" \
    --savedir   "$SAVE" \
    --deepspeed_config ds_config.json \
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
    2>&1 | tee /homes/videetm/dflash/logs/session_apr18_neurips/train_broad_varblock_v2.log
