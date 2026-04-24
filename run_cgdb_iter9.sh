#!/bin/bash
# Iter 9: Confidence-Gated Deep Branching. Shallow wide, deep gated by path prob.

set -o pipefail
unset TORCH_LOGS
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-4B"
DRAFT="z-lab/Qwen3-4B-DFlash-b16"
PORT=29517
EK=8
TEMP=0.0
GPUS="0,1,2,3,4,5,6,7"
NPROC=8
LOGDIR="logs/v8"
PY=/homes/videetm/miniforge3/envs/dflash312/bin/torchrun

run_cfg() {
    local TAG=$1; local DATASET=$2; local SAMPLES=$3; local VER=$4
    local SD=$5; local HI=$6; local LO=$7; local MK=$8
    local LOG="${LOGDIR}/${TAG}.log"
    if [ -f "${LOG}" ] && grep -q "Decoding speedup:" "${LOG}"; then
        echo "[skip] ${TAG}"
        return 0
    fi
    echo "[run ] ${TAG} ver=${VER} sd=${SD} hi=${HI} lo=${LO} mk=${MK}"
    if [ "${VER}" = "7" ]; then EXTRA=""
    else EXTRA="--v8-cgdb-shallow-depth ${SD} --v8-cgdb-high-thresh ${HI} --v8-cgdb-low-thresh ${LO} --v8-cgdb-mid-k ${MK}"
    fi
    CUDA_VISIBLE_DEVICES=${GPUS} ${PY} \
        --nproc_per_node=${NPROC} --master_port=${PORT} benchmark.py \
        --dataset ${DATASET} --max-samples ${SAMPLES} \
        --model-name-or-path ${MODEL} --draft-name-or-path ${DRAFT} \
        --max-new-tokens 1024 --temperature ${TEMP} \
        --tree-version ${VER} --max-tree-size 128 --expand-k ${EK} \
        ${EXTRA} > "${LOG}" 2>&1
    grep -E "Decoding|Acceptance|tree node" "${LOG}" | head -3 | sed "s/^/   /"
}

echo "=== math500-128 ==="
run_cfg ir9_m500_v7              math500 128 7 0 0.0 0.0 0
# Aggressive path-prob gating (cut low prob tails, keep high-prob full)
run_cfg ir9_m500_sd3_hi0p1_lo0p01 math500 128 8 3 0.1 0.01 4
run_cfg ir9_m500_sd3_hi0p3_lo0p05 math500 128 8 3 0.3 0.05 4
run_cfg ir9_m500_sd2_hi0p1_lo0p01 math500 128 8 2 0.1 0.01 4
run_cfg ir9_m500_sd4_hi0p1_lo0p01 math500 128 8 4 0.1 0.01 4
run_cfg ir9_m500_sd3_hi0p05_lo0p005 math500 128 8 3 0.05 0.005 2
# Extreme: hard cut low-prob past depth 3 (no mid zone)
run_cfg ir9_m500_sd3_lo0p05       math500 128 8 3 0.0 0.05 0

echo ""
echo "=== mt-bench-40 (best math500 config) ==="
run_cfg ir9_mtb_v7                mt-bench 40 7 0 0.0 0.0 0
run_cfg ir9_mtb_sd3_hi0p1_lo0p01  mt-bench 40 8 3 0.1 0.01 4

echo ""
echo "=== Summary (tau-first) ==="
for LOG in ${LOGDIR}/ir9_*.log; do
    [ -f "${LOG}" ] || continue
    TAG=$(basename "${LOG}" .log)
    SPD=$(grep "Decoding speedup:" "${LOG}" | tail -1 | awk '{print $3}')
    TAU=$(grep "Average Acceptance length:" "${LOG}" | tail -1 | awk '{print $4}')
    NODES=$(grep "Average tree node count:" "${LOG}" | tail -1 | awk '{print $5}')
    printf "%-35s tau=%s spd=%s nodes=%s\n" "${TAG}" "${TAU:-FAIL}" "${SPD:-FAIL}" "${NODES:-FAIL}"
done
