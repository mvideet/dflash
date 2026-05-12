#!/bin/bash
# v5: v4 (TV + LOO + asym-ent) + DVI-style warmup. First --warmup-steps
# steps are pure aux-TV (PG+asym-ent off), then RL mixes in.
# Pairs with eval_watcher_v5.sh (every-checkpoint mt-bench eval).
set -euo pipefail
cd /homes/videetm/dflash

GPU=${1:-0}
OUTDIR=trainingto/dflash_rl_v5
RUN=rl-v5-dvi-warmup-tv-loo-asym-ent-M8-10k

mkdir -p "$OUTDIR"

CUDA_VISIBLE_DEVICES=$GPU /homes/videetm/miniforge3/envs/dflash312/bin/python train_rl.py \
    --num-steps 10000 \
    --batch-size 4 \
    --rollouts-per-prompt 8 \
    --max-tree-size 160 \
    --gumbel-scale 0.5 \
    --lr 2.5e-5 \
    --entropy-coef 0 \
    --entropy-sharp-coef 0.005 \
    --entropy-flat-coef 0.01 \
    --aux-ce-coef 0 \
    --aux-tv-coef 1.0 \
    --loo-baseline \
    --warmup-steps 2000 \
    --device cuda:0 \
    --out-dir "$OUTDIR" \
    --save-every 1000 \
    --wandb-project dflash-rl \
    --wandb-run-name "$RUN" \
    > /tmp/train_rl_v5.log 2>&1
