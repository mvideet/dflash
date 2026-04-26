#!/bin/bash
# Iter 18: ECS — empirical-calibrated score using P_emp from offline profile.

set -o pipefail
unset TORCH_LOGS
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-4B"
DRAFT="z-lab/Qwen3-4B-DFlash-b16"
PORT=29548
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
        ${CONF} \
        > "${LOG}" 2>&1
    grep -E "Decoding|Acceptance" "${LOG}" | head -2 | sed "s/^/   /"
}

# 4 datasets x B=128 (the sweet spot) x 2 configs (ECS-only, ECS+CGDB)
for DS in math500 mt-bench gsm8k humaneval; do
    case ${DS} in
        math500)   N=256;;
        mt-bench)  N=80;;
        gsm8k)     N=256;;
        humaneval) N=164;;
    esac
    echo "=== ${DS} (N=${N}) ==="
    # ECS only
    run_cfg "iter18_${DS}_b128_ecs" ${DS} ${N} 128 \
        "--v8-cgdb-shallow-depth 4 --v8-ecs"
    # ECS + CGDB
    run_cfg "iter18_${DS}_b128_ecs_cgdb" ${DS} ${N} 128 \
        "--v8-cgdb-shallow-depth 4 --v8-cgdb-high-thresh 0.1 --v8-cgdb-low-thresh 0.01 --v8-cgdb-mid-k 4 --v8-ecs"
done

echo ""
echo "=== ITER 18 ECS Summary vs CGDB baseline ==="
for DS in math500 mt-bench gsm8k humaneval; do
    CGDB_LOG="${LOGDIR}/pareto_${DS}_cgdb_b128.log"
    BASE_TAU=$(grep "Average Acceptance length:" "${CGDB_LOG}" | tail -1 | awk '{print $4}')
    BASE_SPD=$(grep "Decoding speedup:" "${CGDB_LOG}" | tail -1 | awk '{print $3}')
    printf "%-30s baseline_cgdb tau=%s spd=%s\n" "${DS}_b128" "${BASE_TAU}" "${BASE_SPD}"
    for CFG in ecs ecs_cgdb; do
        LOG="${LOGDIR}/iter18_${DS}_b128_${CFG}.log"
        [ -f "${LOG}" ] || continue
        SPD=$(grep "Decoding speedup:" "${LOG}" | tail -1 | awk '{print $3}')
        TAU=$(grep "Average Acceptance length:" "${LOG}" | tail -1 | awk '{print $4}')
        printf "%-30s ${CFG} tau=%s spd=%s\n" "${DS}_b128" "${TAU}" "${SPD}"
    done
done
