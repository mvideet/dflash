#!/bin/bash
# V6 partial-mask fine-tune from V5 (best checkpoint).
# 1 epoch (adjust by --num_epochs) with 50% prob partial-mask batches.
# Cheap: ~5% of VB v1 cost per user's recipe.
set -euo pipefail
cd /homes/videetm/dflash/trainingto
export PATH=/homes/videetm/miniforge3/envs/dflash/bin:$PATH
export DFLASH_ATTN_IMPL=sdpa
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT=3600000
DATA=data/nemotron_broad_150k
# Warm-start from V5 (session SOTA)
DRAFT_INIT=dflash_v5_aggressive/step_37000_hf
deepspeed --include "localhost:0,1,2,3,4,5,6,7" --master_port=29708 main_mix.py \
    --basepath Qwen/Qwen3-4B \
    --draftpath "$DRAFT_INIT" \
    --trainpath "$DATA/train.jsonl" \
    --testpath  "$DATA/test.jsonl" \
    --savedir   dflash_v6_partial_mask \
    --deepspeed_config ds_config.json \
    --num_epochs 1 \
    --gamma-loss 10.0 \
    --random-anchors --anchors-per-seq 24 \
    --block-sizes 16,20,24,28,32 \
    --block-size-probs 1,2,2,2,1 \
    --mask-fill-prob 0.5 \
    --fill-fraction-lo 0.2 \
    --fill-fraction-hi 0.6 \
    --ctr-weight 0.3 \
    --ttt-weight 0.0 \
    --max-tree-size 16 \
    --tree-expand-k 5 \
    --save-every 1000 \
    2>&1 | tee /homes/videetm/dflash/logs/session_apr18_neurips/train_v6_partial_mask.log
