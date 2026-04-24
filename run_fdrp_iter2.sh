#!/bin/bash
# Iteration 2: First-Deviation Rank Penalty — β · j^exp applied ONCE at the
# first-deviation depth. Targets deep-rank phantoms (rank-3..7) whose product
# probability misleadingly survives v7's top-B sort.

set -o pipefail
unset TORCH_LOGS
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-4B"
DRAFT="z-lab/Qwen3-4B-DFlash-b16"
PORT=29508
EK=8
TEMP=0.0
GPUS="0,2,3,4,5,6,7"
NPROC=7
LOGDIR="logs/v8"
PY=/homes/videetm/miniforge3/envs/dflash312/bin/torchrun

run_cfg() {
    local TAG=$1; local DATASET=$2; local SAMPLES=$3
    local VER=$4; local MTS=$5; local BETA=$6; local EXPO=$7
    local LOG="${LOGDIR}/${TAG}.log"
    if [ -f "${LOG}" ] && grep -q "Decoding speedup:" "${LOG}"; then
        echo "[skip] ${TAG}"
        return 0
    fi
    echo "[run ] ${TAG}  ver=${VER} B=${MTS} β=${BETA} exp=${EXPO}"
    if [ "${VER}" = "7" ]; then EXTRA=""
    else EXTRA="--v8-fdrp-beta ${BETA} --v8-fdrp-exp ${EXPO}"
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

echo "=== math500-64, B=128 (v7 baseline: 7.85 / 9.98) ==="
run_cfg ir2_m500_v7_b128        math500 64 7 128 0.0 2.0
run_cfg ir2_m500_fd0p2_e2_b128  math500 64 8 128 0.2 2.0
run_cfg ir2_m500_fd0p5_e2_b128  math500 64 8 128 0.5 2.0
run_cfg ir2_m500_fd1_e2_b128    math500 64 8 128 1.0 2.0
run_cfg ir2_m500_fd0p5_e1_b128  math500 64 8 128 0.5 1.0
run_cfg ir2_m500_fd0p2_e3_b128  math500 64 8 128 0.2 3.0

echo ""
echo "=== mt-bench-40 (best β,exp on math500) ==="
run_cfg ir2_mtb_v7_b128      mt-bench 40 7 128 0.0 2.0
run_cfg ir2_mtb_fd0p5_b128   mt-bench 40 8 128 0.5 2.0

echo ""
echo "=== Pareto summary ==="
for LOG in ${LOGDIR}/ir2_*.log; do
    [ -f "${LOG}" ] || continue
    TAG=$(basename "${LOG}" .log)
    SPD=$(grep "Decoding speedup:" "${LOG}" | tail -1 | awk '{print $3}')
    TAU=$(grep "Average Acceptance length:" "${LOG}" | tail -1 | awk '{print $4}')
    NODES=$(grep "Average tree node count:" "${LOG}" | tail -1 | awk '{print $5}')
    printf "%-30s spd=%s tau=%s nodes=%s\n" "${TAG}" "${SPD:-FAIL}" "${TAU:-FAIL}" "${NODES:-FAIL}"
done
