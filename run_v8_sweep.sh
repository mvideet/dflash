#!/bin/bash
# v8 hyperparameter sweep — math500-64, A6000 x 7.
# v7 baseline (this hardware, this sample count): 7.84 / 9.98 / 129.
#
# Phase A: entropy-gated beta only (additive, pool=1) — does the new scoring
#          signal shift the top-B selection?
# Phase B: + leaf-bonus γ with pool oversampling — does Stage 2 add anything?
# Phase C: + overlap λ — does sibling-redundancy penalty help at B=128?
#
# Each config writes its own log under logs/v8/ so we can grep later.

set -o pipefail
unset TORCH_LOGS
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-4B"
DRAFT="z-lab/Qwen3-4B-DFlash-b16"
DATASET="math500"
SAMPLES=64
PORT=29502
MTS=128
EK=8
TEMP=0.0
GPUS="0,2,3,4,5,6,7"
NPROC=7
LOGDIR="logs/v8"
PY=/homes/videetm/miniforge3/envs/dflash312/bin/torchrun

mkdir -p "${LOGDIR}"

run_v8() {
    local TAG=$1
    local BETA=$2
    local GAMMA=$3
    local LAMBDA=$4
    local POOLMULT=$5
    local LOG="${LOGDIR}/v8_${TAG}.log"
    if [ -f "${LOG}" ] && grep -q "Decoding speedup:" "${LOG}"; then
        echo "[skip] ${TAG} already done"
        return 0
    fi
    echo "[run ] ${TAG}  β=${BETA} γ=${GAMMA} λ=${LAMBDA} pool=${POOLMULT}"
    CUDA_VISIBLE_DEVICES=${GPUS} ${PY} \
        --nproc_per_node=${NPROC} --master_port=${PORT} benchmark.py \
        --dataset ${DATASET} --max-samples ${SAMPLES} \
        --model-name-or-path ${MODEL} --draft-name-or-path ${DRAFT} \
        --max-new-tokens 1024 --temperature ${TEMP} \
        --tree-version 8 --max-tree-size ${MTS} --expand-k ${EK} \
        --v8-entropy-beta ${BETA} --v8-leaf-gamma ${GAMMA} \
        --v8-overlap-lambda ${LAMBDA} --v8-pool-multiplier ${POOLMULT} \
        > "${LOG}" 2>&1
    RC=$?
    if [ ${RC} -ne 0 ]; then
        echo "[FAIL] ${TAG}  (exit ${RC}) — see ${LOG}"
    else
        grep -E "Decoding speedup|Average Acceptance|Average tree" "${LOG}" | head -3 \
            | sed "s/^/   /"
    fi
}

echo "=== Phase A: entropy-gated β (additive, pool=1) ==="
run_v8 "A_beta2"  2.0  0.0 0.0 1
run_v8 "A_beta4"  4.0  0.0 0.0 1
run_v8 "A_beta8"  8.0  0.0 0.0 1
run_v8 "A_beta16" 16.0 0.0 0.0 1
run_v8 "A_beta32" 32.0 0.0 0.0 1

echo ""
echo "=== Phase B: + leaf-bonus γ, pool=2 (Stage 2 swap search) ==="
# γ applied only to LEAVES; requires pool > B for any swap to be useful.
run_v8 "B_g0p5_pool2"  0.0  0.5 0.0 2
run_v8 "B_g1_pool2"    0.0  1.0 0.0 2
run_v8 "B_g2_pool2"    0.0  2.0 0.0 2
run_v8 "B_b4_g1_pool2" 4.0  1.0 0.0 2
run_v8 "B_b4_g2_pool2" 4.0  2.0 0.0 2

echo ""
echo "=== Phase C: + sibling overlap λ ==="
run_v8 "C_b4_g1_L0p3_pool2" 4.0  1.0 0.3 2
run_v8 "C_b4_g1_L1_pool2"   4.0  1.0 1.0 2

echo ""
echo "=== DONE ==="
echo ""
echo "Summary:"
for LOG in ${LOGDIR}/v8_*.log; do
    TAG=$(basename "${LOG}" .log)
    SPD=$(grep "Decoding speedup:" "${LOG}" | tail -1 | awk '{print $3}')
    TAU=$(grep "Average Acceptance length:" "${LOG}" | tail -1 | awk '{print $4}')
    NODES=$(grep "Average tree node count:" "${LOG}" | tail -1 | awk '{print $5}')
    printf "%-30s  spd=%-6s tau=%-6s nodes=%s\n" "${TAG}" "${SPD:-FAIL}" "${TAU:-FAIL}" "${NODES:-FAIL}"
done
