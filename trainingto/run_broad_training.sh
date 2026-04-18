#!/bin/bash
# Launch variable-block DFlash training on broad-mix data.
# Recipe (NeurIPS push):
#   - Variable block: b ∈ {12, 16, 20, 24} (ADD b=24 to break ceiling)
#   - Random anchors (+13% tau, DFlash Table 9)
#   - Exp-weighted CE with γ=7
#   - Mild CTR (weight 0.3, ctrlite recipe — prior winner)
#   - No TTT (prior tt-lite slightly regressed)
#   - Data: broad mix (math + chat + stem + code)
set -euo pipefail

ROOT=/homes/videetm/dflash
cd "$ROOT/trainingto"

DATA=data/nemotron_broad_150k
SAVE=dflash_broad_varblock_v1
NUM_GPUS=${NUM_GPUS:-8}

mkdir -p /homes/videetm/dflash/logs/session_apr18_neurips

deepspeed --num_gpus=$NUM_GPUS --master_port=29702 main_mix.py \
    --basepath Qwen/Qwen3-4B \
    --draftpath z-lab/Qwen3-4B-DFlash-b16 \
    --trainpath "$DATA/train.jsonl" \
    --testpath  "$DATA/test.jsonl" \
    --savedir   "$SAVE" \
    --deepspeed_config ds_config.json \
    --num_epochs 2 \
    --gamma-loss 7.0 \
    --random-anchors --anchors-per-seq 32 \
    --block-sizes 12,16,20,24 \
    --block-size-probs 1,2,2,1 \
    --ctr-weight 0.3 \
    --ttt-weight 0.0 \
    --max-tree-size 16 \
    --tree-expand-k 5 \
    --save-every 500 \
    2>&1 | tee /homes/videetm/dflash/logs/session_apr18_neurips/train_broad_varblock_v1.log
