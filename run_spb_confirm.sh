#!/bin/bash
# Iter 6 confirm: stack SPB α=0.5 with B=96 (iter-5 sweet spot), and do
# interleaved repeats to nail down whether +0.07-0.08 is real.

set -o pipefail
unset TORCH_LOGS
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-4B"
DRAFT="z-lab/Qwen3-4B-DFlash-b16"
PORT=29513
EK=8
TEMP=0.0
GPUS="0,1,2,3,4,5,6,7"
NPROC=8
LOGDIR="logs/v8"
PY=/homes/videetm/miniforge3/envs/dflash312/bin/torchrun

run_cfg() {
    local TAG=$1; local DATASET=$2; local SAMPLES=$3; local MTS=$4; local VER=$5; local ALPHA=$6
    local LOG="${LOGDIR}/${TAG}.log"
    if [ -f "${LOG}" ] && grep -q "Decoding speedup:" "${LOG}"; then
        echo "[skip] ${TAG}"
        return 0
    fi
    echo "[run ] ${TAG}  ver=${VER} B=${MTS} α=${ALPHA}"
    if [ "${VER}" = "7" ]; then EXTRA=""
    else EXTRA="--v8-spb-alpha ${ALPHA}"
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

# Interleaved confirmation: v7_B128, spb_B128, v7_B96, spb_B96, rinse repeat.
echo "=== Interleaved math500-128 confirmations (4 pairs) ==="
run_cfg ir6c_v7_b128_r1    math500 128 128 7 0.0
run_cfg ir6c_spb_b128_r1   math500 128 128 8 0.5
run_cfg ir6c_v7_b96_r1     math500 128 96  7 0.0
run_cfg ir6c_spb_b96_r1    math500 128 96  8 0.5

run_cfg ir6c_v7_b128_r2    math500 128 128 7 0.0
run_cfg ir6c_spb_b128_r2   math500 128 128 8 0.5
run_cfg ir6c_v7_b96_r2     math500 128 96  7 0.0
run_cfg ir6c_spb_b96_r2    math500 128 96  8 0.5

echo ""
echo "=== mt-bench-40 confirmations ==="
run_cfg ir6c_mtb_v7_b128_r1   mt-bench 40 128 7 0.0
run_cfg ir6c_mtb_spb_b128_r1  mt-bench 40 128 8 0.5
run_cfg ir6c_mtb_v7_b112      mt-bench 40 112 7 0.0
run_cfg ir6c_mtb_spb_b112     mt-bench 40 112 8 0.5

echo ""
echo "=== Summary ==="
for LOG in ${LOGDIR}/ir6c_*.log; do
    [ -f "${LOG}" ] || continue
    TAG=$(basename "${LOG}" .log)
    SPD=$(grep "Decoding speedup:" "${LOG}" | tail -1 | awk '{print $3}')
    TAU=$(grep "Average Acceptance length:" "${LOG}" | tail -1 | awk '{print $4}')
    NODES=$(grep "Average tree node count:" "${LOG}" | tail -1 | awk '{print $5}')
    printf "%-30s spd=%s tau=%s nodes=%s\n" "${TAG}" "${SPD:-FAIL}" "${TAU:-FAIL}" "${NODES:-FAIL}"
done
