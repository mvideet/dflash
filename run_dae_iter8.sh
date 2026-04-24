#!/bin/bash
# Iter 8: Depth-Adaptive Expansion. Shallow-broad, deep-narrow expand_k.
# Tau-first evaluation.

set -o pipefail
unset TORCH_LOGS
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-4B"
DRAFT="z-lab/Qwen3-4B-DFlash-b16"
PORT=29515
EK=8
TEMP=0.0
GPUS="0,1,2,3,4,5,6,7"
NPROC=8
LOGDIR="logs/v8"
PY=/homes/videetm/miniforge3/envs/dflash312/bin/torchrun

run_cfg() {
    local TAG=$1; local DATASET=$2; local SAMPLES=$3; local MTS=$4; local VER=$5
    local SK=$6; local SD=$7; local DK=$8
    local LOG="${LOGDIR}/${TAG}.log"
    if [ -f "${LOG}" ] && grep -q "Decoding speedup:" "${LOG}"; then
        echo "[skip] ${TAG}"
        return 0
    fi
    echo "[run ] ${TAG}  ver=${VER} sk=${SK} sd=${SD} dk=${DK}"
    if [ "${VER}" = "7" ]; then EXTRA=""
    else EXTRA="--v8-dae-shallow-k ${SK} --v8-dae-shallow-depth ${SD} --v8-dae-deep-k ${DK}"
    fi
    CUDA_VISIBLE_DEVICES=${GPUS} ${PY} \
        --nproc_per_node=${NPROC} --master_port=${PORT} benchmark.py \
        --dataset ${DATASET} --max-samples ${SAMPLES} \
        --model-name-or-path ${MODEL} --draft-name-or-path ${DRAFT} \
        --max-new-tokens 1024 --temperature ${TEMP} \
        --tree-version ${VER} --max-tree-size ${MTS} --expand-k ${EK} \
        ${EXTRA} > "${LOG}" 2>&1
    grep -E "Decoding|Acceptance|tree node" "${LOG}" | head -3 | sed "s/^/   /"
}

echo "=== math500-128 ==="
run_cfg ir8_m500_v7            math500 128 128 7 0 3 0
run_cfg ir8_m500_dae_12_3_4    math500 128 128 8 12 3 4
run_cfg ir8_m500_dae_16_3_4    math500 128 128 8 16 3 4
run_cfg ir8_m500_dae_12_4_6    math500 128 128 8 12 4 6
run_cfg ir8_m500_dae_16_2_4    math500 128 128 8 16 2 4
run_cfg ir8_m500_dae_12_3_6    math500 128 128 8 12 3 6

echo ""
echo "=== mt-bench-40 (winners from math500) ==="
run_cfg ir8_mtb_v7             mt-bench 40 128 7 0 3 0
run_cfg ir8_mtb_dae_12_3_4     mt-bench 40 128 8 12 3 4
run_cfg ir8_mtb_dae_16_3_4     mt-bench 40 128 8 16 3 4

echo ""
echo "=== Summary (tau-first) ==="
for LOG in ${LOGDIR}/ir8_*.log; do
    [ -f "${LOG}" ] || continue
    TAG=$(basename "${LOG}" .log)
    SPD=$(grep "Decoding speedup:" "${LOG}" | tail -1 | awk '{print $3}')
    TAU=$(grep "Average Acceptance length:" "${LOG}" | tail -1 | awk '{print $4}')
    NODES=$(grep "Average tree node count:" "${LOG}" | tail -1 | awk '{print $5}')
    printf "%-30s tau=%s spd=%s nodes=%s\n" "${TAG}" "${TAU:-FAIL}" "${SPD:-FAIL}" "${NODES:-FAIL}"
done
