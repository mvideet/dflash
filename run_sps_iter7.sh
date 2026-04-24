#!/bin/bash
# Iter 7: SPS. Reward low-prob alternatives via q -> (1-λ)q + λ/K smoothing.
# TAU-FIRST evaluation — user said acceptance rate takes precedence.

set -o pipefail
unset TORCH_LOGS
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-4B"
DRAFT="z-lab/Qwen3-4B-DFlash-b16"
PORT=29514
EK=8
TEMP=0.0
GPUS="0,1,2,3,4,5,6,7"
NPROC=8
LOGDIR="logs/v8"
PY=/homes/videetm/miniforge3/envs/dflash312/bin/torchrun

run_cfg() {
    local TAG=$1; local DATASET=$2; local SAMPLES=$3; local MTS=$4; local VER=$5; local LAM=$6
    local LOG="${LOGDIR}/${TAG}.log"
    if [ -f "${LOG}" ] && grep -q "Decoding speedup:" "${LOG}"; then
        echo "[skip] ${TAG}"
        return 0
    fi
    echo "[run ] ${TAG}  ver=${VER} B=${MTS} λ=${LAM}"
    if [ "${VER}" = "7" ]; then EXTRA=""
    else EXTRA="--v8-sps-lambda ${LAM}"
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

echo "=== math500-128 B=128 (tau-first sweep) ==="
run_cfg ir7_m500_v7         math500 128 128 7 0.0
run_cfg ir7_m500_lam0p05    math500 128 128 8 0.05
run_cfg ir7_m500_lam0p1     math500 128 128 8 0.1
run_cfg ir7_m500_lam0p2     math500 128 128 8 0.2
run_cfg ir7_m500_lam0p3     math500 128 128 8 0.3
run_cfg ir7_m500_lam0p5     math500 128 128 8 0.5

echo ""
echo "=== mt-bench-40 (cross-dataset) ==="
run_cfg ir7_mtb_v7          mt-bench 40 128 7 0.0
run_cfg ir7_mtb_lam0p1      mt-bench 40 128 8 0.1
run_cfg ir7_mtb_lam0p2      mt-bench 40 128 8 0.2
run_cfg ir7_mtb_lam0p3      mt-bench 40 128 8 0.3

echo ""
echo "=== Summary (tau first) ==="
for LOG in ${LOGDIR}/ir7_*.log; do
    [ -f "${LOG}" ] || continue
    TAG=$(basename "${LOG}" .log)
    SPD=$(grep "Decoding speedup:" "${LOG}" | tail -1 | awk '{print $3}')
    TAU=$(grep "Average Acceptance length:" "${LOG}" | tail -1 | awk '{print $4}')
    NODES=$(grep "Average tree node count:" "${LOG}" | tail -1 | awk '{print $5}')
    printf "%-25s tau=%s spd=%s nodes=%s\n" "${TAG}" "${TAU:-FAIL}" "${SPD:-FAIL}" "${NODES:-FAIL}"
done
