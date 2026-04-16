#!/bin/bash
# CTR (Conditional Tree Refinement) training for DFlash draft model
# Target: Qwen/Qwen3-4B  |  Draft: z-lab/Qwen3-4B-DFlash-b16
# Data: nemotron_math (225K train, 2.3K test)

NUM_GPUS="${NUM_GPUS:-3}"
MASTER_PORT="${MASTER_PORT:-29700}"

deepspeed --num_gpus=$NUM_GPUS --master_port=$MASTER_PORT main_ctr.py \
    --basepath Qwen/Qwen3-4B \
    --draftpath z-lab/Qwen3-4B-DFlash-b16 \
    --trainpath data/nemotron_math/train_nemotron_math.jsonl \
    --testpath  data/nemotron_math/test_nemotron_math.jsonl \
    --savedir   dflash_ctr_checkpoints \
    --deepspeed_config ds_config.json \
    --num_epochs 5 \
    --ctr-weight 0.5 \
    --max-tree-size 16 \
    --tree-expand-k 5 \
    --save-every 500 \
    "$@"
