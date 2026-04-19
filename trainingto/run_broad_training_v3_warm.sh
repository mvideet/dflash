#!/bin/bash
# VB v3 (warm start): continue from VB v1 step_18500 rather than stock.
# This reuses the +3% speedup v1 already gained and adds another 1-2
# epochs of extended-curriculum training for further refinement.
set -euo pipefail

ROOT=/homes/videetm/dflash
cd "$ROOT/trainingto"

DATA=data/nemotron_broad_150k
SAVE=dflash_broad_varblock_v3_warm
GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}

mkdir -p /homes/videetm/dflash/logs/session_apr18_neurips

export PATH=/homes/videetm/miniforge3/envs/dflash/bin:$PATH
export DFLASH_ATTN_IMPL=sdpa
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT=3600000

# WARM START: use VB v1's final checkpoint as initial draft weights
DRAFT_INIT=dflash_broad_varblock_v1/step_18500_hf

deepspeed --include "localhost:${GPUS}" --master_port=29704 main_mix.py \
    --basepath Qwen/Qwen3-4B \
    --draftpath "$DRAFT_INIT" \
    --trainpath "$DATA/train.jsonl" \
    --testpath  "$DATA/test.jsonl" \
    --savedir   "$SAVE" \
    --deepspeed_config ds_config.json \
    --num_epochs 2 \
    --gamma-loss 10.0 \
    --random-anchors --anchors-per-seq 32 \
    --block-sizes 12,16,20,24,28,32 \
    --block-size-probs 1,2,2,2,1,1 \
    --ctr-weight 0.3 \
    --ttt-weight 0.0 \
    --max-tree-size 16 \
    --tree-expand-k 5 \
    --save-every 1000 \
    2>&1 | tee /homes/videetm/dflash/logs/session_apr18_neurips/train_broad_varblock_v3_warm.log
