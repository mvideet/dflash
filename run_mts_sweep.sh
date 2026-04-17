#!/bin/bash
# Re-sweep mts values after tree_attn_mask vectorization. Peak may have shifted.

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
TIMEOUT=1800

mkdir -p logs

for MTS in 96 160 192; do
    LOG="logs/optmask_${DATASET}_mts${MTS}.log"
    if [ -f "${LOG}" ] && grep -q "Decoding speedup:" "${LOG}"; then
        echo "--- mts=${MTS} done ---"
        grep "Decoding speedup" "${LOG}"
        continue
    fi
    echo "=== mts=${MTS} ==="
    timeout ${TIMEOUT} torchrun --nproc_per_node=8 --master_port=${PORT} \
        benchmark.py \
        --dataset ${DATASET} --max-samples ${SAMPLES} \
        --model-name-or-path ${MODEL} --draft-name-or-path ${DRAFT} \
        --tree-version 7 --max-tree-size ${MTS} --expand-k ${EK} \
        --temperature ${TEMP} 2>&1 | tee "${LOG}"
    grep "Decoding speedup" "${LOG}"
    sleep 5
done
echo "=== DONE ==="
