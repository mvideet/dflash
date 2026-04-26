#!/bin/bash
# Iter 23: ACDC-T — argmax + (d, j) edit leaves WITH T-token argmax tail.

set -o pipefail
unset TORCH_LOGS
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-4B"
DRAFT="z-lab/Qwen3-4B-DFlash-b16"
PORT=29553
EK=8
TEMP=0.0
GPUS="0,1,2,3,4,5,6,7"
NPROC=8
LOGDIR="logs/v8"
PY=/homes/videetm/miniforge3/envs/dflash312/bin/torchrun

run_cfg() {
    local TAG=$1; local DS=$2; local N=$3; local MTS=$4; local TAIL=$5
    local LOG="${LOGDIR}/${TAG}.log"
    if [ -f "${LOG}" ] && grep -q "Decoding speedup:" "${LOG}"; then
        echo "[skip] ${TAG}"
        return 0
    fi
    echo "[run ] ${TAG} B=${MTS} tail=${TAIL}"
    timeout 1800 \
      env CUDA_VISIBLE_DEVICES=${GPUS} ${PY} \
        --nproc_per_node=${NPROC} --master_port=${PORT} benchmark.py \
        --dataset ${DS} --max-samples ${N} \
        --model-name-or-path ${MODEL} --draft-name-or-path ${DRAFT} \
        --max-new-tokens 1024 --temperature ${TEMP} \
        --tree-version 9 --max-tree-size ${MTS} --expand-k ${EK} \
        --v9-tail-len ${TAIL} \
        > "${LOG}" 2>&1
    grep -E "Decoding|Acceptance" "${LOG}" | head -2 | sed "s/^/   /"
}

DS=math500
N=256
echo "=== ${DS} (N=${N}) ACDC-T tail sweep at B=128 ==="
for TAIL in 1 2 4 8 14; do
    run_cfg "iter23_${DS}_b128_tail${TAIL}" ${DS} ${N} 128 ${TAIL}
done

echo ""
echo "=== ITER 23 ACDC-T Summary vs CGDB baseline ==="
CGDB_LOG="${LOGDIR}/pareto_${DS}_cgdb_b128.log"
BASE_TAU=$(grep "Average Acceptance length:" "${CGDB_LOG}" | tail -1 | awk '{print $4}')
BASE_SPD=$(grep "Decoding speedup:" "${CGDB_LOG}" | tail -1 | awk '{print $3}')
printf "%-30s baseline_cgdb tau=%s spd=%s\n" "${DS}_b128" "${BASE_TAU}" "${BASE_SPD}"
for TAIL in 1 2 4 8 14; do
    LOG="${LOGDIR}/iter23_${DS}_b128_tail${TAIL}.log"
    [ -f "${LOG}" ] || continue
    SPD=$(grep "Decoding speedup:" "${LOG}" | tail -1 | awk '{print $3}')
    TAU=$(grep "Average Acceptance length:" "${LOG}" | tail -1 | awk '{print $4}')
    printf "%-30s tail=${TAIL} tau=%s spd=%s\n" "${DS}_b128" "${TAU}" "${SPD}"
done
