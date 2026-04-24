#!/bin/bash
# Cross-dataset confirmation: run the best v8 config on mt-bench + gsm8k.
# Takes BETA_BEST, GAMMA_BEST, LAMBDA_BEST, POOLMULT_BEST from env; defaults
# pick the pure β_e = 4 configuration if not set.

set -o pipefail
unset TORCH_LOGS
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-4B"
DRAFT="z-lab/Qwen3-4B-DFlash-b16"
PORT=29502
MTS=128
EK=8
TEMP=0.0
GPUS="0,2,3,4,5,6,7"
NPROC=7
LOGDIR="logs/v8"
PY=/homes/videetm/miniforge3/envs/dflash312/bin/torchrun

BETA_BEST=${BETA_BEST:-4.0}
GAMMA_BEST=${GAMMA_BEST:-0.0}
LAMBDA_BEST=${LAMBDA_BEST:-0.0}
POOLMULT_BEST=${POOLMULT_BEST:-1}

mkdir -p "${LOGDIR}"

run_cd() {
    local TAG=$1
    local DATASET=$2
    local SAMPLES=$3
    local VER=$4
    local LOG="${LOGDIR}/${TAG}.log"
    if [ -f "${LOG}" ] && grep -q "Decoding speedup:" "${LOG}"; then
        echo "[skip] ${TAG}"
        return 0
    fi
    echo "[run ] ${TAG}"
    if [ "${VER}" = "7" ]; then
        EXTRA=""
    else
        EXTRA="--v8-entropy-beta ${BETA_BEST} --v8-leaf-gamma ${GAMMA_BEST} --v8-overlap-lambda ${LAMBDA_BEST} --v8-pool-multiplier ${POOLMULT_BEST}"
    fi
    CUDA_VISIBLE_DEVICES=${GPUS} ${PY} \
        --nproc_per_node=${NPROC} --master_port=${PORT} benchmark.py \
        --dataset ${DATASET} --max-samples ${SAMPLES} \
        --model-name-or-path ${MODEL} --draft-name-or-path ${DRAFT} \
        --max-new-tokens 1024 --temperature ${TEMP} \
        --tree-version ${VER} --max-tree-size ${MTS} --expand-k ${EK} \
        ${EXTRA} \
        > "${LOG}" 2>&1
    RC=$?
    if [ ${RC} -ne 0 ]; then
        echo "[FAIL] ${TAG}"
    else
        grep -E "Decoding speedup|Average Acceptance|Average tree" "${LOG}" | head -3 | sed "s/^/   /"
    fi
}

# v7 baselines
run_cd "v7_mtbench_40"  mt-bench 40  7
run_cd "v7_gsm8k_64"    gsm8k    64  7

# v8 best
run_cd "v8_mtbench_40"  mt-bench 40  8
run_cd "v8_gsm8k_64"    gsm8k    64  8

echo ""
echo "=== Cross-dataset summary (v8 config: β=${BETA_BEST} γ=${GAMMA_BEST} λ=${LAMBDA_BEST} pool=${POOLMULT_BEST}) ==="
for LOG in ${LOGDIR}/v7_mtbench_40.log ${LOGDIR}/v8_mtbench_40.log ${LOGDIR}/v7_gsm8k_64.log ${LOGDIR}/v8_gsm8k_64.log; do
    [ -f "${LOG}" ] || continue
    TAG=$(basename "${LOG}" .log)
    SPD=$(grep "Decoding speedup:" "${LOG}" | tail -1 | awk '{print $3}')
    TAU=$(grep "Average Acceptance length:" "${LOG}" | tail -1 | awk '{print $4}')
    NODES=$(grep "Average tree node count:" "${LOG}" | tail -1 | awk '{print $5}')
    printf "%-25s spd=%s tau=%s nodes=%s\n" "${TAG}" "${SPD:-FAIL}" "${TAU:-FAIL}" "${NODES:-FAIL}"
done
