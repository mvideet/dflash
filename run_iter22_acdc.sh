#!/bin/bash
# Iter 22: ACDC — argmax chain + (depth, rank) edit grid. Topology different.
# Tests at B=64, 96, 128, 192, 256 with both --v9-emp-prior on/off.

set -o pipefail
unset TORCH_LOGS
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-4B"
DRAFT="z-lab/Qwen3-4B-DFlash-b16"
PORT=29552
EK=8
TEMP=0.0
GPUS="0,1,2,3,4,5,6,7"
NPROC=8
LOGDIR="logs/v8"
PY=/homes/videetm/miniforge3/envs/dflash312/bin/torchrun

run_cfg() {
    local TAG=$1; local DS=$2; local N=$3; local MTS=$4; local EMP=$5
    local LOG="${LOGDIR}/${TAG}.log"
    if [ -f "${LOG}" ] && grep -q "Decoding speedup:" "${LOG}"; then
        echo "[skip] ${TAG}"
        return 0
    fi
    EXTRA=""
    if [ "${EMP}" = "1" ]; then EXTRA="--v9-emp-prior"; fi
    echo "[run ] ${TAG} B=${MTS} emp=${EMP}"
    timeout 1800 \
      env CUDA_VISIBLE_DEVICES=${GPUS} ${PY} \
        --nproc_per_node=${NPROC} --master_port=${PORT} benchmark.py \
        --dataset ${DS} --max-samples ${N} \
        --model-name-or-path ${MODEL} --draft-name-or-path ${DRAFT} \
        --max-new-tokens 1024 --temperature ${TEMP} \
        --tree-version 9 --max-tree-size ${MTS} --expand-k ${EK} ${EXTRA} \
        > "${LOG}" 2>&1
    grep -E "Decoding|Acceptance" "${LOG}" | head -2 | sed "s/^/   /"
}

declare -A NSAMPLES=(
    [math500]=256
    [mt-bench]=80
    [gsm8k]=256
    [humaneval]=164
)

# Test on math500 first across multiple B with both modes.
DS=math500
N=256
echo "=== ${DS} (N=${N}) ACDC quick scan ==="
for B in 64 96 128 192; do
    run_cfg "iter22_${DS}_b${B}_q" ${DS} ${N} ${B} 0
    run_cfg "iter22_${DS}_b${B}_emp" ${DS} ${N} ${B} 1
done

echo ""
echo "=== ITER 22 ACDC vs CGDB baseline ==="
for B in 64 96 128 192; do
    CGDB_LOG="${LOGDIR}/pareto_${DS}_cgdb_b${B}.log"
    BASE_TAU=$(grep "Average Acceptance length:" "${CGDB_LOG}" | tail -1 | awk '{print $4}')
    BASE_SPD=$(grep "Decoding speedup:" "${CGDB_LOG}" | tail -1 | awk '{print $3}')
    printf "%-30s baseline_cgdb tau=%s spd=%s\n" "${DS}_b${B}" "${BASE_TAU}" "${BASE_SPD}"
    for SUFFIX in q emp; do
        LOG="${LOGDIR}/iter22_${DS}_b${B}_${SUFFIX}.log"
        [ -f "${LOG}" ] || continue
        SPD=$(grep "Decoding speedup:" "${LOG}" | tail -1 | awk '{print $3}')
        TAU=$(grep "Average Acceptance length:" "${LOG}" | tail -1 | awk '{print $4}')
        printf "%-30s acdc_${SUFFIX} tau=%s spd=%s\n" "${DS}_b${B}" "${TAU}" "${SPD}"
    done
done
