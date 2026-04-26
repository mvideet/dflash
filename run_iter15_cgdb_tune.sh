#!/bin/bash
# Iter 15: CGDB hyperparameter tune — sweep sd & mid_k around iter-9 best.

set -o pipefail
unset TORCH_LOGS
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-4B"
DRAFT="z-lab/Qwen3-4B-DFlash-b16"
PORT=29545
EK=8
TEMP=0.0
GPUS="0,1,2,3,4,5,6,7"
NPROC=8
LOGDIR="logs/v8"
PY=/homes/videetm/miniforge3/envs/dflash312/bin/torchrun

run_cfg() {
    local TAG=$1; local DS=$2; local N=$3; local MTS=$4; local SD=$5; local MK=$6
    local LOG="${LOGDIR}/${TAG}.log"
    if [ -f "${LOG}" ] && grep -q "Decoding speedup:" "${LOG}"; then
        echo "[skip] ${TAG}"
        return 0
    fi
    echo "[run ] ${TAG} B=${MTS} sd=${SD} mid_k=${MK}"
    timeout 1800 \
      env CUDA_VISIBLE_DEVICES=${GPUS} ${PY} \
        --nproc_per_node=${NPROC} --master_port=${PORT} benchmark.py \
        --dataset ${DS} --max-samples ${N} \
        --model-name-or-path ${MODEL} --draft-name-or-path ${DRAFT} \
        --max-new-tokens 1024 --temperature ${TEMP} \
        --tree-version 8 --max-tree-size ${MTS} --expand-k ${EK} \
        --v8-cgdb-shallow-depth ${SD} --v8-cgdb-high-thresh 0.1 --v8-cgdb-low-thresh 0.01 --v8-cgdb-mid-k ${MK} \
        > "${LOG}" 2>&1
    grep -E "Decoding|Acceptance" "${LOG}" | head -2 | sed "s/^/   /"
}

declare -A NSAMPLES=(
    [math500]=256
    [mt-bench]=80
    [gsm8k]=256
    [humaneval]=164
)

# Sweep sd in {2, 3, 5, 6} (4 was original baseline) at mid_k=4.
# Plus sweep mid_k in {2, 6} at sd=4.
for DS in math500 mt-bench gsm8k humaneval; do
    N=${NSAMPLES[${DS}]}
    echo "=== ${DS} (N=${N}) ==="
    for B in 128 192; do
        # sd sweep at mid_k=4
        for SD in 2 3 5 6; do
            run_cfg "iter15_${DS}_b${B}_sd${SD}_mk4" ${DS} ${N} ${B} ${SD} 4
        done
        # mid_k sweep at sd=4
        for MK in 2 6; do
            run_cfg "iter15_${DS}_b${B}_sd4_mk${MK}" ${DS} ${N} ${B} 4 ${MK}
        done
    done
done

echo ""
echo "=== ITER 15 CGDB tune Summary vs CGDB sd=4,mk=4 baseline ==="
for DS in math500 mt-bench gsm8k humaneval; do
    for B in 128 192; do
        CGDB_LOG="${LOGDIR}/pareto_${DS}_cgdb_b${B}.log"
        BASE_TAU=$(grep "Average Acceptance length:" "${CGDB_LOG}" | tail -1 | awk '{print $4}')
        BASE_SPD=$(grep "Decoding speedup:" "${CGDB_LOG}" | tail -1 | awk '{print $3}')
        printf "%-30s baseline_sd4mk4 tau=%s spd=%s\n" "${DS}_b${B}" "${BASE_TAU}" "${BASE_SPD}"
        for SD in 2 3 5 6; do
            LOG="${LOGDIR}/iter15_${DS}_b${B}_sd${SD}_mk4.log"
            [ -f "${LOG}" ] || continue
            SPD=$(grep "Decoding speedup:" "${LOG}" | tail -1 | awk '{print $3}')
            TAU=$(grep "Average Acceptance length:" "${LOG}" | tail -1 | awk '{print $4}')
            printf "%-30s sd=${SD},mk=4 tau=%s spd=%s\n" "${DS}_b${B}" "${TAU}" "${SPD}"
        done
        for MK in 2 6; do
            LOG="${LOGDIR}/iter15_${DS}_b${B}_sd4_mk${MK}.log"
            [ -f "${LOG}" ] || continue
            SPD=$(grep "Decoding speedup:" "${LOG}" | tail -1 | awk '{print $3}')
            TAU=$(grep "Average Acceptance length:" "${LOG}" | tail -1 | awk '{print $4}')
            printf "%-30s sd=4,mk=${MK} tau=%s spd=%s\n" "${DS}_b${B}" "${TAU}" "${SPD}"
        done
    done
done
