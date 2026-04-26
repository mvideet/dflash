#!/bin/bash
# Iter 10: TT-CGDB. Stack tail-truncation on top of CGDB.
# Test 3 tail_depth values vs the CGDB best baseline.

set -o pipefail
unset TORCH_LOGS
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-4B"
DRAFT="z-lab/Qwen3-4B-DFlash-b16"
PORT=29540
EK=8
TEMP=0.0
GPUS="0,1,2,3,4,5,6,7"
NPROC=8
LOGDIR="logs/v8"
PY=/homes/videetm/miniforge3/envs/dflash312/bin/torchrun

# CGDB defaults from iter 9.
CGDB_SD=4
CGDB_HI=0.1
CGDB_LO=0.01
CGDB_MK=4

run_cfg() {
    local TAG=$1; local DS=$2; local N=$3; local MTS=$4; local TT=$5
    local LOG="${LOGDIR}/${TAG}.log"
    if [ -f "${LOG}" ] && grep -q "Decoding speedup:" "${LOG}"; then
        echo "[skip] ${TAG}"
        return 0
    fi
    echo "[run ] ${TAG} B=${MTS} tt=${TT}"
    EXTRA="--v8-cgdb-shallow-depth ${CGDB_SD} --v8-cgdb-high-thresh ${CGDB_HI} --v8-cgdb-low-thresh ${CGDB_LO} --v8-cgdb-mid-k ${CGDB_MK}"
    if [ "${TT}" != "0" ]; then EXTRA="${EXTRA} --v8-tt-depth ${TT}"; fi
    timeout 1800 \
      env CUDA_VISIBLE_DEVICES=${GPUS} ${PY} \
        --nproc_per_node=${NPROC} --master_port=${PORT} benchmark.py \
        --dataset ${DS} --max-samples ${N} \
        --model-name-or-path ${MODEL} --draft-name-or-path ${DRAFT} \
        --max-new-tokens 1024 --temperature ${TEMP} \
        --tree-version 8 --max-tree-size ${MTS} --expand-k ${EK} \
        ${EXTRA} > "${LOG}" 2>&1
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
    for B in 128 192; do
        for TT in 8 10 12; do
            run_cfg "iter10_${DS}_b${B}_tt${TT}" ${DS} ${N} ${B} ${TT}
        done
    done
done

echo ""
echo "=== ITER 10 TT-CGDB Summary ==="
for DS in math500 mt-bench gsm8k humaneval; do
    for B in 128 192; do
        # Get CGDB baseline from prior pareto run for reference.
        CGDB_LOG="${LOGDIR}/pareto_${DS}_cgdb_b${B}.log"
        BASE_TAU=$(grep "Average Acceptance length:" "${CGDB_LOG}" | tail -1 | awk '{print $4}')
        BASE_SPD=$(grep "Decoding speedup:" "${CGDB_LOG}" | tail -1 | awk '{print $3}')
        printf "%-30s baseline_cgdb tau=%s spd=%s\n" "${DS}_b${B}" "${BASE_TAU}" "${BASE_SPD}"
        for TT in 8 10 12; do
            LOG="${LOGDIR}/iter10_${DS}_b${B}_tt${TT}.log"
            [ -f "${LOG}" ] || continue
            SPD=$(grep "Decoding speedup:" "${LOG}" | tail -1 | awk '{print $3}')
            TAU=$(grep "Average Acceptance length:" "${LOG}" | tail -1 | awk '{print $4}')
            printf "%-30s tt=%s tau=%s spd=%s\n" "${DS}_b${B}" "${TT}" "${TAU}" "${SPD}"
        done
    done
done
