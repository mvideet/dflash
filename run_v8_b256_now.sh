#!/bin/bash
# Test v8 at the v7-collapse regime (B=256, B=512). The phantom-path
# hypothesis predicts v7 collapses here but v8's entropy-gated penalty
# should preserve more speedup by reducing phantom selections.

set -o pipefail
unset TORCH_LOGS
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-4B"
DRAFT="z-lab/Qwen3-4B-DFlash-b16"
DATASET="math500"
SAMPLES=64
PORT=29503
EK=8
TEMP=0.0
GPUS="0,2,3,4,5,6,7"
NPROC=7
LOGDIR="logs/v8"
PY=/homes/videetm/miniforge3/envs/dflash312/bin/torchrun

mkdir -p "${LOGDIR}"

run_cfg() {
    local TAG=$1; local VER=$2; local MTS=$3
    local BETA=$4; local GAMMA=$5; local LAMBDA=$6; local POOLMULT=$7
    local LOG="${LOGDIR}/${TAG}.log"
    if [ -f "${LOG}" ] && grep -q "Decoding speedup:" "${LOG}"; then
        echo "[skip] ${TAG}"
        return 0
    fi
    echo "[run ] ${TAG}"
    CUDA_VISIBLE_DEVICES=${GPUS} ${PY} \
        --nproc_per_node=${NPROC} --master_port=${PORT} benchmark.py \
        --dataset ${DATASET} --max-samples ${SAMPLES} \
        --model-name-or-path ${MODEL} --draft-name-or-path ${DRAFT} \
        --max-new-tokens 1024 --temperature ${TEMP} \
        --tree-version ${VER} --max-tree-size ${MTS} --expand-k ${EK} \
        --v8-entropy-beta ${BETA} --v8-leaf-gamma ${GAMMA} \
        --v8-overlap-lambda ${LAMBDA} --v8-pool-multiplier ${POOLMULT} \
        > "${LOG}" 2>&1
    RC=$?
    grep -E "Decoding|Acceptance|tree node" "${LOG}" | head -3 | sed "s/^/   /"
}

# v7 baselines at B=256, B=512 (known to collapse per program.md)
run_cfg "v7_b256" 7 256 0.0 0.0 0.0 1
run_cfg "v7_b512" 7 512 0.0 0.0 0.0 1

# v8 with strong entropy β at B=256, B=512
run_cfg "v8_b256_beta4"  8 256 4.0 0.0 0.0 1
run_cfg "v8_b256_beta8"  8 256 8.0 0.0 0.0 1
run_cfg "v8_b256_beta16" 8 256 16.0 0.0 0.0 1
run_cfg "v8_b512_beta8"  8 512 8.0 0.0 0.0 1
run_cfg "v8_b512_beta16" 8 512 16.0 0.0 0.0 1

echo "=== Summary ==="
for LOG in ${LOGDIR}/v7_b256.log ${LOGDIR}/v8_b256_*.log ${LOGDIR}/v7_b512.log ${LOGDIR}/v8_b512_*.log; do
    [ -f "${LOG}" ] || continue
    TAG=$(basename "${LOG}" .log)
    SPD=$(grep "Decoding speedup:" "${LOG}" | tail -1 | awk '{print $3}')
    TAU=$(grep "Average Acceptance length:" "${LOG}" | tail -1 | awk '{print $4}')
    NODES=$(grep "Average tree node count:" "${LOG}" | tail -1 | awk '{print $5}')
    printf "%-30s spd=%s tau=%s nodes=%s\n" "${TAG}" "${SPD:-FAIL}" "${TAU:-FAIL}" "${NODES:-FAIL}"
done
