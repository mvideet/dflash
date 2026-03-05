export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

mkdir -p logs

# TASKS=(
#   "gsm8k:128"
#   "math500:128"
#   "aime24:30"
#   "aime25:30"
#   "humaneval:164"
#   "mbpp:128"
#   "livecodebench:128"
#   "swe-bench:128"
#   "mt-bench:80"
#   "alpaca:128"
# )
TASKS=(
  # "gsm8k:128"
  "mt-bench:80"
)

CHAIN_ATTENTION=false   # set to false to use standard draft-KV-cache mode
TOP_K=5                # branching factor K (only used when CHAIN_ATTENTION=true, dynamic_branching=false)
DYNAMIC_BRANCHING=true # adaptive K in {1,2,3} per position (implies chain_attention)
THETA_UNI=0.9
THETA_BI=0.3
THETA_TRI=0.1
MAX_TREE_SIZE=8

for task in "${TASKS[@]}"; do
  IFS=':' read -r DATASET_NAME MAX_SAMPLES <<< "$task"

  echo "========================================================"
  echo "Running Benchmark: $DATASET_NAME with $MAX_SAMPLES samples"
  echo "chain_attention=${CHAIN_ATTENTION}  top_k=${TOP_K}  dynamic_branching=${DYNAMIC_BRANCHING}"
  echo "========================================================"

  EXTRA_ARGS=""
  if [ "$CHAIN_ATTENTION" = "true" ]; then
    EXTRA_ARGS="--chain-attention --top-k ${TOP_K}"
  fi
  if [ "$DYNAMIC_BRANCHING" = "true" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --dynamic-branching --theta-uni ${THETA_UNI} --theta-bi ${THETA_BI} --theta-tri ${THETA_TRI} --max-tree-size ${MAX_TREE_SIZE}"
  fi

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
    $EXTRA_ARGS \
    2>&1 | tee "logs/${DATASET_NAME}.log"

done