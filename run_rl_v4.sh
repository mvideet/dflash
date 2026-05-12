#!/bin/bash
# v4: TV-distance auxiliary loss (DistillSpec ICLR'24) + asymmetric entropy +
# leave-one-out (RLOO, Ahmadian et al. 2024) baseline for variance reduction.
# Mirror of v3 except aux_ce_coef -> aux_tv_coef and group-mean -> LOO baseline.
# TV(p, q) is a direct lower bound on speculative-decoding acceptance prob
# (α >= 1 - TV); LOO baseline is strictly lower variance than the group mean.
set -euo pipefail
cd /homes/videetm/dflash

GPU=${1:-0}
OUTDIR=trainingto/dflash_rl_v4
RUN=rl-v4-tv-loo-asym-ent-M8-10k

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
    --device cuda:0 \
    --out-dir "$OUTDIR" \
    --save-every 500 \
    --wandb-project dflash-rl \
    --wandb-run-name "$RUN" \
    > /tmp/train_rl_v4.log 2>&1
