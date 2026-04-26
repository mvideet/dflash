#!/bin/bash
# Iter 20: CGDB + FDRC stacked. Hard cap rank-j past first deviation.

set -o pipefail
unset TORCH_LOGS
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-4B"
DRAFT="z-lab/Qwen3-4B-DFlash-b16"
PORT=29550
EK=8
TEMP=0.0
GPUS="0,1,2,3,4,5,6,7"
NPROC=8
LOGDIR="logs/v8"
PY=/homes/videetm/miniforge3/envs/dflash312/bin/torchrun

run_cfg() {
    local TAG=$1; local DS=$2; local N=$3; local MTS=$4; local CAP=$5
    local LOG="${LOGDIR}/${TAG}.log"
    if [ -f "${LOG}" ] && grep -q "Decoding speedup:" "${LOG}"; then
        echo "[skip] ${TAG}"
        return 0
    fi
    echo "[run ] ${TAG} B=${MTS} cap=${CAP}"
    timeout 1800 \
      env CUDA_VISIBLE_DEVICES=${GPUS} ${PY} \
        --nproc_per_node=${NPROC} --master_port=${PORT} benchmark.py \
        --dataset ${DS} --max-samples ${N} \
        --model-name-or-path ${MODEL} --draft-name-or-path ${DRAFT} \
        --max-new-tokens 1024 --temperature ${TEMP} \
        --tree-version 8 --max-tree-size ${MTS} --expand-k ${EK} \
        --v8-cgdb-shallow-depth 4 --v8-cgdb-high-thresh 0.1 --v8-cgdb-low-thresh 0.01 --v8-cgdb-mid-k 4 \
        --v8-fdrc-cap ${CAP} \
        > "${LOG}" 2>&1
    grep -E "Decoding|Acceptance" "${LOG}" | head -2 | sed "s/^/   /"
}

declare -A NSAMPLES=(
    [math500]=256
    [mt-bench]=80
    [gsm8k]=256
    [humaneval]=164
)

for DS in math500 mt-bench gsm8k humaneval; do
    N=${NSAMPLES[${DS}]}
    echo "=== ${DS} (N=${N}) ==="
    for CAP in 2 3 4; do
        run_cfg "iter20_${DS}_b128_cap${CAP}" ${DS} ${N} 128 ${CAP}
    done
done

echo ""
echo "=== ITER 20 CGDB+FDRC Summary vs CGDB baseline ==="
for DS in math500 mt-bench gsm8k humaneval; do
    CGDB_LOG="${LOGDIR}/pareto_${DS}_cgdb_b128.log"
    BASE_TAU=$(grep "Average Acceptance length:" "${CGDB_LOG}" | tail -1 | awk '{print $4}')
    BASE_SPD=$(grep "Decoding speedup:" "${CGDB_LOG}" | tail -1 | awk '{print $3}')
    printf "%-30s baseline_cgdb tau=%s spd=%s\n" "${DS}_b128" "${BASE_TAU}" "${BASE_SPD}"
    for CAP in 2 3 4; do
        LOG="${LOGDIR}/iter20_${DS}_b128_cap${CAP}.log"
        [ -f "${LOG}" ] || continue
        SPD=$(grep "Decoding speedup:" "${LOG}" | tail -1 | awk '{print $3}')
        TAU=$(grep "Average Acceptance length:" "${LOG}" | tail -1 | awk '{print $4}')
        printf "%-30s cap=${CAP} tau=%s spd=%s\n" "${DS}_b128" "${TAU}" "${SPD}"
    done
done
