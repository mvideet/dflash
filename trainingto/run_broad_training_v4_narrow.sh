#!/bin/bash
# V4-narrow: EXACT replay of dflash_mix_10k_v3_ctrlite winning recipe, 
# scaled up to our 148k broad-mix data.
# Math-only would be better per prior winner, but we use broad to keep 
# cross-dataset exposure.
set -euo pipefail
cd /homes/videetm/dflash/trainingto
export PATH=/homes/videetm/miniforge3/envs/dflash/bin:$PATH
export DFLASH_ATTN_IMPL=sdpa
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT=3600000
DATA=data/nemotron_broad_150k
deepspeed --include "localhost:0,1,2,3,4,5,6,7" --master_port=29705 main_mix.py \
    --basepath Qwen/Qwen3-4B \
    --draftpath z-lab/Qwen3-4B-DFlash-b16 \
    --trainpath "$DATA/train.jsonl" \
    --testpath  "$DATA/test.jsonl" \
    --savedir   dflash_broad_v4_narrow \
    --deepspeed_config ds_config.json \
    --num_epochs 1 \
    --gamma-loss 7.0 \
    --random-anchors --anchors-per-seq 16 \
    --block-sizes 16 \
    --ctr-weight 0.3 \
    --ttt-weight 0.0 \
    --max-tree-size 16 \
    --tree-expand-k 5 \
    --save-every 1000 \
    2>&1 | tee /homes/videetm/dflash/logs/session_apr18_neurips/train_broad_v4_narrow.log
