#!/bin/bash
# Iter 25: ACDC-V — asymmetric tail-per-rank.

set -o pipefail
unset TORCH_LOGS
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-4B"
DRAFT="z-lab/Qwen3-4B-DFlash-b16"
PORT=29555
EK=8
TEMP=0.0
GPUS="0,1,2,3,4,5,6,7"
NPROC=8
LOGDIR="logs/v8"
PY=/homes/videetm/miniforge3/envs/dflash312/bin/torchrun

run_cfg() {
    local TAG=$1; local DS=$2; local N=$3; local MTS=$4; local TPR=$5
    local LOG="${LOGDIR}/${TAG}.log"
    if [ -f "${LOG}" ] && grep -q "Decoding speedup:" "${LOG}"; then
        echo "[skip] ${TAG}"
        return 0
    fi
    echo "[run ] ${TAG} B=${MTS} tpr=${TPR}"
    timeout 1800 \
      env CUDA_VISIBLE_DEVICES=${GPUS} ${PY} \
        --nproc_per_node=${NPROC} --master_port=${PORT} benchmark.py \
        --dataset ${DS} --max-samples ${N} \
        --model-name-or-path ${MODEL} --draft-name-or-path ${DRAFT} \
        --max-new-tokens 1024 --temperature ${TEMP} \
        --tree-version 9 --max-tree-size ${MTS} --expand-k ${EK} \
        --v9-tail-per-rank "${TPR}" \
        > "${LOG}" 2>&1
    grep -E "Decoding|Acceptance|tree node" "${LOG}" | head -3 | sed "s/^/   /"
}

DS=math500
N=256
echo "=== ${DS} (N=${N}) ACDC-V at B=128 ==="
# Aggressive long rank-1, gentle rank-2, point-only rank-3+
run_cfg "iter25_${DS}_b128_v1"  ${DS} ${N} 128 "14,4,1,0,0,0,0"
# rank-1 still long but rank-2 also extended
run_cfg "iter25_${DS}_b128_v2"  ${DS} ${N} 128 "14,8,2,0,0,0,0"
# All ranks moderate tail
run_cfg "iter25_${DS}_b128_v3"  ${DS} ${N} 128 "8,4,2,1,1,0,0"
# Deep rank-1, short rank-2, point rank-3+
run_cfg "iter25_${DS}_b128_v4"  ${DS} ${N} 128 "14,2,1,1,0,0,0"

echo ""
echo "=== ITER 25 ACDC-V Summary vs CGDB baseline ==="
CGDB_LOG="${LOGDIR}/pareto_${DS}_cgdb_b128.log"
BASE_TAU=$(grep "Average Acceptance length:" "${CGDB_LOG}" | tail -1 | awk '{print $4}')
BASE_SPD=$(grep "Decoding speedup:" "${CGDB_LOG}" | tail -1 | awk '{print $3}')
printf "%-30s baseline tau=%s spd=%s\n" "${DS}_b128" "${BASE_TAU}" "${BASE_SPD}"
for V in v1 v2 v3 v4; do
    LOG="${LOGDIR}/iter25_${DS}_b128_${V}.log"
    [ -f "${LOG}" ] || continue
    SPD=$(grep "Decoding speedup:" "${LOG}" | tail -1 | awk '{print $3}')
    TAU=$(grep "Average Acceptance length:" "${LOG}" | tail -1 | awk '{print $4}')
    NODES=$(grep "Average tree node count:" "${LOG}" | tail -1 | awk '{print $5}')
    printf "%-30s ${V} tau=%s spd=%s nodes=%s\n" "${DS}_b128" "${TAU}" "${SPD}" "${NODES}"
done
