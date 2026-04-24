#!/bin/bash
# Cross-dataset confirmation: v7 and v8 at B=128 on mt-bench-40.

set -o pipefail
unset TORCH_LOGS
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-4B"
DRAFT="z-lab/Qwen3-4B-DFlash-b16"
PORT=29506
EK=8
TEMP=0.0
GPUS="0,2,3,4,5,6,7"
NPROC=7
LOGDIR="logs/v8"
PY=/homes/videetm/miniforge3/envs/dflash312/bin/torchrun

run_mtb() {
    local TAG=$1; local VER=$2; local MTS=$3
    local BETA=$4; local GAMMA=$5; local LAMBDA=$6; local POOLMULT=$7; local DEVCOST=$8
    local LOG="${LOGDIR}/${TAG}.log"
    if [ -f "${LOG}" ] && grep -q "Decoding speedup:" "${LOG}"; then
        echo "[skip] ${TAG}"
        return 0
    fi
    echo "[run ] ${TAG}"
    if [ "${VER}" = "7" ]; then
        EXTRA=""
    else
        EXTRA="--v8-entropy-beta ${BETA} --v8-leaf-gamma ${GAMMA} --v8-overlap-lambda ${LAMBDA} --v8-pool-multiplier ${POOLMULT} --v8-dev-depth-cost ${DEVCOST}"
    fi
    CUDA_VISIBLE_DEVICES=${GPUS} ${PY} \
        --nproc_per_node=${NPROC} --master_port=${PORT} benchmark.py \
        --dataset mt-bench --max-samples 40 \
        --model-name-or-path ${MODEL} --draft-name-or-path ${DRAFT} \
        --max-new-tokens 1024 --temperature ${TEMP} \
        --tree-version ${VER} --max-tree-size ${MTS} --expand-k ${EK} \
        ${EXTRA} > "${LOG}" 2>&1
    grep -E "Decoding|Acceptance|tree node" "${LOG}" | head -3 | sed "s/^/   /"
}

# v7 baseline on mt-bench
run_mtb mtb_v7_b128      7 128 0.0 0.0 0.0 1 0

# Closest-to-success v8 variants at B=128
run_mtb mtb_v8_ident     8 128 0.0 0.0 0.0 1 0
run_mtb mtb_v8_hdc_d1    8 128 0.0 0.0 0.0 1 1

# Also test HDC at B=256 (best v8 @ B=256 on math500)
run_mtb mtb_v8_hdc_b256  8 256 0.0 0.0 0.0 1 1
run_mtb mtb_v7_b256      7 256 0.0 0.0 0.0 1 0

echo "=== mt-bench summary ==="
for LOG in ${LOGDIR}/mtb_*.log; do
    [ -f "${LOG}" ] || continue
    TAG=$(basename "${LOG}" .log)
    SPD=$(grep "Decoding speedup:" "${LOG}" | tail -1 | awk '{print $3}')
    TAU=$(grep "Average Acceptance length:" "${LOG}" | tail -1 | awk '{print $4}')
    NODES=$(grep "Average tree node count:" "${LOG}" | tail -1 | awk '{print $5}')
    printf "%-25s spd=%s tau=%s nodes=%s\n" "${TAG}" "${SPD:-FAIL}" "${TAU:-FAIL}" "${NODES:-FAIL}"
done
