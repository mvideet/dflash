#!/bin/bash
# Iter 16: CMG. Cumulative-margin gating.

set -o pipefail
unset TORCH_LOGS
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-4B"
DRAFT="z-lab/Qwen3-4B-DFlash-b16"
PORT=29546
EK=8
TEMP=0.0
GPUS="0,1,2,3,4,5,6,7"
NPROC=8
LOGDIR="logs/v8"
PY=/homes/videetm/miniforge3/envs/dflash312/bin/torchrun

run_cfg() {
    local TAG=$1; local DS=$2; local N=$3; local MTS=$4; local CONF=$5
    local LOG="${LOGDIR}/${TAG}.log"
    if [ -f "${LOG}" ] && grep -q "Decoding speedup:" "${LOG}"; then
        echo "[skip] ${TAG}"
        return 0
    fi
    echo "[run ] ${TAG} B=${MTS} ${CONF}"
    timeout 1800 \
      env CUDA_VISIBLE_DEVICES=${GPUS} ${PY} \
        --nproc_per_node=${NPROC} --master_port=${PORT} benchmark.py \
        --dataset ${DS} --max-samples ${N} \
        --model-name-or-path ${MODEL} --draft-name-or-path ${DRAFT} \
        --max-new-tokens 1024 --temperature ${TEMP} \
        --tree-version 8 --max-tree-size ${MTS} --expand-k ${EK} \
        --v8-cgdb-shallow-depth 4 ${CONF} \
        > "${LOG}" 2>&1
    grep -E "Decoding|Acceptance" "${LOG}" | head -2 | sed "s/^/   /"
}

declare -A NSAMPLES=(
    [math500]=256
    [mt-bench]=80
    [gsm8k]=256
    [humaneval]=164
)

# CMG configs (3 thresholds) + stack with CGDB.
declare -a CONFIGS=(
    "cmgA   --v8-cmg-high 8.0  --v8-cmg-low 3.0  --v8-cmg-mid-k 4 --v8-cmg-shallow-depth 4"
    "cmgB   --v8-cmg-high 12.0 --v8-cmg-low 5.0  --v8-cmg-mid-k 4 --v8-cmg-shallow-depth 4"
    "cmgC_stack --v8-cmg-high 8.0 --v8-cmg-low 3.0 --v8-cmg-mid-k 4 --v8-cmg-shallow-depth 4 --v8-cgdb-high-thresh 0.1 --v8-cgdb-low-thresh 0.01 --v8-cgdb-mid-k 4"
)

for DS in math500 mt-bench gsm8k humaneval; do
    N=${NSAMPLES[${DS}]}
    echo "=== ${DS} (N=${N}) ==="
    for B in 128 192; do
        for CFG in "${CONFIGS[@]}"; do
            CFG_NAME=$(echo "${CFG}" | awk '{print $1}')
            CFG_FLAGS=$(echo "${CFG}" | cut -d' ' -f2-)
            run_cfg "iter16_${DS}_b${B}_${CFG_NAME}" ${DS} ${N} ${B} "${CFG_FLAGS}"
        done
    done
done

echo ""
echo "=== ITER 16 CMG Summary vs CGDB baseline ==="
for DS in math500 mt-bench gsm8k humaneval; do
    for B in 128 192; do
        CGDB_LOG="${LOGDIR}/pareto_${DS}_cgdb_b${B}.log"
        BASE_TAU=$(grep "Average Acceptance length:" "${CGDB_LOG}" | tail -1 | awk '{print $4}')
        BASE_SPD=$(grep "Decoding speedup:" "${CGDB_LOG}" | tail -1 | awk '{print $3}')
        printf "%-30s baseline tau=%s spd=%s\n" "${DS}_b${B}" "${BASE_TAU}" "${BASE_SPD}"
        for CFG_NAME in cmgA cmgB cmgC_stack; do
            LOG="${LOGDIR}/iter16_${DS}_b${B}_${CFG_NAME}.log"
            [ -f "${LOG}" ] || continue
            SPD=$(grep "Decoding speedup:" "${LOG}" | tail -1 | awk '{print $3}')
            TAU=$(grep "Average Acceptance length:" "${LOG}" | tail -1 | awk '{print $4}')
            printf "%-30s ${CFG_NAME} tau=%s spd=%s\n" "${DS}_b${B}" "${TAU}" "${SPD}"
        done
    done
done
