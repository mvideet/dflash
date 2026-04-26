#!/bin/bash
# Iter 24: FCH — Forced-Coverage Heap. v8/CGDB + force (d, j) leaves at every depth.

set -o pipefail
unset TORCH_LOGS
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-4B"
DRAFT="z-lab/Qwen3-4B-DFlash-b16"
PORT=29554
EK=8
TEMP=0.0
GPUS="0,1,2,3,4,5,6,7"
NPROC=8
LOGDIR="logs/v8"
PY=/homes/videetm/miniforge3/envs/dflash312/bin/torchrun

run_cfg() {
    local TAG=$1; local DS=$2; local N=$3; local MTS=$4; local R=$5
    local LOG="${LOGDIR}/${TAG}.log"
    if [ -f "${LOG}" ] && grep -q "Decoding speedup:" "${LOG}"; then
        echo "[skip] ${TAG}"
        return 0
    fi
    echo "[run ] ${TAG} B=${MTS} fch_max_rank=${R}"
    timeout 1800 \
      env CUDA_VISIBLE_DEVICES=${GPUS} ${PY} \
        --nproc_per_node=${NPROC} --master_port=${PORT} benchmark.py \
        --dataset ${DS} --max-samples ${N} \
        --model-name-or-path ${MODEL} --draft-name-or-path ${DRAFT} \
        --max-new-tokens 1024 --temperature ${TEMP} \
        --tree-version 8 --max-tree-size ${MTS} --expand-k ${EK} \
        --v8-cgdb-shallow-depth 4 --v8-cgdb-high-thresh 0.1 --v8-cgdb-low-thresh 0.01 --v8-cgdb-mid-k 4 \
        --v8-fch-max-rank ${R} \
        > "${LOG}" 2>&1
    grep -E "Decoding|Acceptance|tree node" "${LOG}" | head -3 | sed "s/^/   /"
}

DS=math500
N=256
echo "=== ${DS} (N=${N}) FCH at B=128 ==="
for R in 1 2 3; do
    run_cfg "iter24_${DS}_b128_fch${R}" ${DS} ${N} 128 ${R}
done

echo ""
echo "=== ITER 24 FCH Summary vs CGDB baseline ==="
CGDB_LOG="${LOGDIR}/pareto_${DS}_cgdb_b128.log"
BASE_TAU=$(grep "Average Acceptance length:" "${CGDB_LOG}" | tail -1 | awk '{print $4}')
BASE_SPD=$(grep "Decoding speedup:" "${CGDB_LOG}" | tail -1 | awk '{print $3}')
printf "%-30s baseline tau=%s spd=%s\n" "${DS}_b128" "${BASE_TAU}" "${BASE_SPD}"
for R in 1 2 3; do
    LOG="${LOGDIR}/iter24_${DS}_b128_fch${R}.log"
    [ -f "${LOG}" ] || continue
    SPD=$(grep "Decoding speedup:" "${LOG}" | tail -1 | awk '{print $3}')
    TAU=$(grep "Average Acceptance length:" "${LOG}" | tail -1 | awk '{print $4}')
    NODES=$(grep "Average tree node count:" "${LOG}" | tail -1 | awk '{print $5}')
    printf "%-30s fch=${R} tau=%s spd=%s nodes=%s\n" "${DS}_b128" "${TAU}" "${SPD}" "${NODES}"
done
