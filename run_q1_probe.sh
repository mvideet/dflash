#!/bin/bash
# Q1: Power-scaled scoring probe sweep.
# Targets mts=256 where v7 baseline collapsed from 7.98 (mts=128) -> 7.29.
# If any (alpha, beta) config beats 7.29 materially, expand that direction.

set -o pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dflash312

MODEL="Qwen/Qwen3-4B"
DRAFT="z-lab/Qwen3-4B-DFlash-b16"
DATASET="math500"
SAMPLES=256
PORT=29501
EK=8
TEMP=0.0
MTS=256
PER_RUN_TIMEOUT=2400

mkdir -p logs

# Configs: (alpha, beta)
CONFIGS=(
    "1.0 0.5"
    "1.0 1.0"
    "0.95 0.0"
    "0.9 0.5"
    "0.95 0.5"
)

for CFG in "${CONFIGS[@]}"; do
    read A B <<< "${CFG}"
    TAG="a${A}_b${B}_mts${MTS}"
    LOG="logs/q1_probe_${TAG}.log"

    if [ -f "${LOG}" ] && grep -q "Decoding speedup:" "${LOG}"; then
        echo "--- ${TAG} already done, skip ---"
        grep "Decoding speedup:"         "${LOG}" | tail -1
        grep "Average Acceptance length:" "${LOG}" | tail -1
        grep "Average tree node count:"  "${LOG}" | tail -1
        continue
    fi

    echo "=== Running alpha=${A} beta=${B} mts=${MTS} ==="
    timeout ${PER_RUN_TIMEOUT} torchrun --nproc_per_node=8 --master_port=${PORT} \
        benchmark.py \
        --dataset ${DATASET} \
        --max-samples ${SAMPLES} \
        --model-name-or-path ${MODEL} \
        --draft-name-or-path ${DRAFT} \
        --tree-version 7 \
        --max-tree-size ${MTS} \
        --expand-k ${EK} \
        --temperature ${TEMP} \
        --score-alpha ${A} \
        --score-beta ${B} \
        2>&1 | tee "${LOG}"

    echo ""
    echo "--- ${TAG} results ---"
    grep "Decoding speedup:"         "${LOG}" | tail -1 || echo "  crash"
    grep "Average Acceptance length:" "${LOG}" | tail -1
    grep "Average tree node count:"  "${LOG}" | tail -1
    echo ""

    sleep 10
done

echo "=== Q1 PROBE SWEEP DONE ==="
