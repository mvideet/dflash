#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mkdir -p logs

TASKS=("mt-bench:80")
BLOCK_SIZE=16              # draft block size (default from model; 8, 16, etc.)
THETA_UNI=0.9
THETA_BI=0.3
THETA_TRI=0.1
MAX_TREE_SIZE=8

for task in "${TASKS[@]}"; do
  IFS=':' read -r DATASET_NAME MAX_SAMPLES <<< "$task"

  # Vanilla
  torchrun \
    --nproc_per_node=8 \
    --master_port=29600 \
    benchmark.py \
    --dataset "$DATASET_NAME" \
    --max-samples "$MAX_SAMPLES" \
    --model-name-or-path Qwen/Qwen3-4B \
    --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 \
    --max-new-tokens 2048 \
    --temperature 0.0 \
    --block-size ${BLOCK_SIZE} \
    2>&1 | tee "logs/${DATASET_NAME}_vanilla.log"

  # Optimized
  torchrun \
    --nproc_per_node=8 \
    --master_port=29601 \
    benchmark.py \
    --dataset "$DATASET_NAME" \
    --max-samples "$MAX_SAMPLES" \
    --model-name-or-path Qwen/Qwen3-4B \
    --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 \
    --max-new-tokens 2048 \
    --temperature 0.0 \
    --block-size ${BLOCK_SIZE} \
    --dynamic-branching \
    --theta-uni ${THETA_UNI} \
    --theta-bi ${THETA_BI} \
    --theta-tri ${THETA_TRI} \
    --max-tree-size ${MAX_TREE_SIZE} \
    2>&1 | tee "logs/${DATASET_NAME}_optimized.log"

  VANILLA_SPEEDUP=$(grep "Decoding speedup:" "logs/${DATASET_NAME}_vanilla.log" | tail -1 | awk '{print $3}')
  VANILLA_ACCEPTANCE=$(grep "Average Acceptance length:" "logs/${DATASET_NAME}_vanilla.log" | tail -1 | awk '{print $4}')
  OPTIMIZED_SPEEDUP=$(grep "Decoding speedup:" "logs/${DATASET_NAME}_optimized.log" | tail -1 | awk '{print $3}')
  OPTIMIZED_ACCEPTANCE=$(grep "Average Acceptance length:" "logs/${DATASET_NAME}_optimized.log" | tail -1 | awk '{print $4}')

  echo "Summary: $DATASET_NAME (block_size=${BLOCK_SIZE})"
  echo "Vanilla      speedup: ${VANILLA_SPEEDUP:-N/A}, acceptance: ${VANILLA_ACCEPTANCE:-N/A}"
  echo "Optimized    speedup: ${OPTIMIZED_SPEEDUP:-N/A}, acceptance: ${OPTIMIZED_ACCEPTANCE:-N/A}"

  if [ -n "$VANILLA_SPEEDUP" ] && [ -n "$OPTIMIZED_SPEEDUP" ]; then
    echo "Logs: logs/${DATASET_NAME}_vanilla.log  |  logs/${DATASET_NAME}_optimized.log"
  fi
done
