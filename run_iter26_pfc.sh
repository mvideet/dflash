#!/bin/bash
# Iter 26: PFC — Parent-First-Dev-Rank-Conditional expand_k.

set -o pipefail
unset TORCH_LOGS
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-4B"
DRAFT="z-lab/Qwen3-4B-DFlash-b16"
PORT=29556
EK=8
TEMP=0.0
GPUS="0,1,2,3,4,5,6,7"
NPROC=8
LOGDIR="logs/v8"
PY=/homes/videetm/miniforge3/envs/dflash312/bin/torchrun

run_cfg() {
    local TAG=$1; local DS=$2; local N=$3; local MTS=$4; local F1=$5; local F2=$6
    local LOG="${LOGDIR}/${TAG}.log"
    if [ -f "${LOG}" ] && grep -q "Decoding speedup:" "${LOG}"; then
        echo "[skip] ${TAG}"
        return 0
    fi
    echo "[run ] ${TAG} B=${MTS} fdr1_k=${F1} fdr2_k=${F2}"
    timeout 1800 \
      env CUDA_VISIBLE_DEVICES=${GPUS} ${PY} \
        --nproc_per_node=${NPROC} --master_port=${PORT} benchmark.py \
        --dataset ${DS} --max-samples ${N} \
        --model-name-or-path ${MODEL} --draft-name-or-path ${DRAFT} \
        --max-new-tokens 1024 --temperature ${TEMP} \
        --tree-version 8 --max-tree-size ${MTS} --expand-k ${EK} \
        --v8-cgdb-shallow-depth 4 --v8-cgdb-high-thresh 0.1 --v8-cgdb-low-thresh 0.01 --v8-cgdb-mid-k 4 \
        --v8-pfc-fdr1-k ${F1} --v8-pfc-fdr2plus-k ${F2} \
        > "${LOG}" 2>&1
    grep -E "Decoding|Acceptance|tree node" "${LOG}" | head -3 | sed "s/^/   /"
}

DS=math500
N=256
echo "=== ${DS} (N=${N}) PFC at B=128 ==="
run_cfg "iter26_${DS}_b128_p2_1"  ${DS} ${N} 128 2 1
run_cfg "iter26_${DS}_b128_p3_1"  ${DS} ${N} 128 3 1
run_cfg "iter26_${DS}_b128_p4_2"  ${DS} ${N} 128 4 2
run_cfg "iter26_${DS}_b128_p2_0"  ${DS} ${N} 128 2 0

echo ""
echo "=== ITER 26 PFC Summary vs CGDB baseline ==="
CGDB_LOG="${LOGDIR}/pareto_${DS}_cgdb_b128.log"
BASE_TAU=$(grep "Average Acceptance length:" "${CGDB_LOG}" | tail -1 | awk '{print $4}')
BASE_SPD=$(grep "Decoding speedup:" "${CGDB_LOG}" | tail -1 | awk '{print $3}')
printf "%-30s baseline tau=%s spd=%s\n" "${DS}_b128" "${BASE_TAU}" "${BASE_SPD}"
for V in p2_1 p3_1 p4_2 p2_0; do
    LOG="${LOGDIR}/iter26_${DS}_b128_${V}.log"
    [ -f "${LOG}" ] || continue
    SPD=$(grep "Decoding speedup:" "${LOG}" | tail -1 | awk '{print $3}')
    TAU=$(grep "Average Acceptance length:" "${LOG}" | tail -1 | awk '{print $4}')
    NODES=$(grep "Average tree node count:" "${LOG}" | tail -1 | awk '{print $5}')
    printf "%-30s ${V} tau=%s spd=%s nodes=%s\n" "${DS}_b128" "${TAU}" "${SPD}" "${NODES}"
done
