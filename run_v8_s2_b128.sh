#!/bin/bash
# Test Stage-2 reselection at B=128 (the current SOTA peak).
# Hypothesis: γ-leaf-bonus + pool=2 shifts selection toward better-leaf
# patterns, raising tau at same node count. Tested at B=256 (tau 10.17→10.29).
# Does the same hold at B=128 (tau 9.98 → ?)?

set -o pipefail
unset TORCH_LOGS
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-4B"
DRAFT="z-lab/Qwen3-4B-DFlash-b16"
DATASET="math500"
SAMPLES=64
PORT=29505
EK=8
TEMP=0.0
GPUS="0,2,3,4,5,6,7"
NPROC=7
LOGDIR="logs/v8"
PY=/homes/videetm/miniforge3/envs/dflash312/bin/torchrun

run_v8() {
    local TAG=$1; local MTS=$2
    local BETA=$3; local GAMMA=$4; local LAMBDA=$5; local POOLMULT=$6; local DEVCOST=$7
    local LOG="${LOGDIR}/${TAG}.log"
    if [ -f "${LOG}" ] && grep -q "Decoding speedup:" "${LOG}"; then
        echo "[skip] ${TAG}"
        return 0
    fi
    echo "[run ] ${TAG} (mts=${MTS} β=${BETA} γ=${GAMMA} λ=${LAMBDA} pm=${POOLMULT} dev=${DEVCOST})"
    CUDA_VISIBLE_DEVICES=${GPUS} ${PY} \
        --nproc_per_node=${NPROC} --master_port=${PORT} benchmark.py \
        --dataset ${DATASET} --max-samples ${SAMPLES} \
        --model-name-or-path ${MODEL} --draft-name-or-path ${DRAFT} \
        --max-new-tokens 1024 --temperature ${TEMP} \
        --tree-version 8 --max-tree-size ${MTS} --expand-k ${EK} \
        --v8-entropy-beta ${BETA} --v8-leaf-gamma ${GAMMA} \
        --v8-overlap-lambda ${LAMBDA} --v8-pool-multiplier ${POOLMULT} \
        --v8-dev-depth-cost ${DEVCOST} \
        > "${LOG}" 2>&1
    grep -E "Decoding|Acceptance|tree node" "${LOG}" | head -3 | sed "s/^/   /"
}

run_v8 s2_b128_g0p5_pm2 128 0.0 0.5 0.0 2 0
run_v8 s2_b128_g1_pm2   128 0.0 1.0 0.0 2 0
run_v8 s2_b128_g2_pm2   128 0.0 2.0 0.0 2 0
run_v8 s2_b128_g1_pm3   128 0.0 1.0 0.0 3 0
run_v8 s2_b128_g1_pm4   128 0.0 1.0 0.0 4 0
run_v8 s2_b128_g2_pm4   128 0.0 2.0 0.0 4 0

echo ""
echo "=== Summary vs v7 B=128 baseline (7.84 / 9.98) ==="
for LOG in ${LOGDIR}/s2_b128_*.log; do
    [ -f "${LOG}" ] || continue
    TAG=$(basename "${LOG}" .log)
    SPD=$(grep "Decoding speedup:" "${LOG}" | tail -1 | awk '{print $3}')
    TAU=$(grep "Average Acceptance length:" "${LOG}" | tail -1 | awk '{print $4}')
    NODES=$(grep "Average tree node count:" "${LOG}" | tail -1 | awk '{print $5}')
    printf "%-25s spd=%s tau=%s nodes=%s\n" "${TAG}" "${SPD:-FAIL}" "${TAU:-FAIL}" "${NODES:-FAIL}"
done
