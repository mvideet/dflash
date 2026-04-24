#!/bin/bash
# Iteration 4: First-Deviation Rank Cap (FDRC). Hard cap on rank of first dev.
# If cap=3 is a no-op (no rank-4+ in v7's top-128), tau stays at 10.00.
# If tau shifts up, we've found a structural phantom reallocation.

set -o pipefail
unset TORCH_LOGS
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-4B"
DRAFT="z-lab/Qwen3-4B-DFlash-b16"
PORT=29510
EK=8
TEMP=0.0
GPUS="0,2,3,4,5,6,7"
NPROC=7
LOGDIR="logs/v8"
PY=/homes/videetm/miniforge3/envs/dflash312/bin/torchrun

run_cfg() {
    local TAG=$1; local DATASET=$2; local SAMPLES=$3
    local VER=$4; local MTS=$5; local CAP=$6
    local LOG="${LOGDIR}/${TAG}.log"
    if [ -f "${LOG}" ] && grep -q "Decoding speedup:" "${LOG}"; then
        echo "[skip] ${TAG}"
        return 0
    fi
    echo "[run ] ${TAG} ver=${VER} B=${MTS} cap=${CAP}"
    if [ "${VER}" = "7" ]; then EXTRA=""
    else EXTRA="--v8-fdrc-cap ${CAP}"
    fi
    CUDA_VISIBLE_DEVICES=${GPUS} ${PY} \
        --nproc_per_node=${NPROC} --master_port=${PORT} benchmark.py \
        --dataset ${DATASET} --max-samples ${SAMPLES} \
        --model-name-or-path ${MODEL} --draft-name-or-path ${DRAFT} \
        --max-new-tokens 1024 --temperature ${TEMP} \
        --tree-version ${VER} --max-tree-size ${MTS} --expand-k ${EK} \
        ${EXTRA} > "${LOG}" 2>&1
    grep -E "Decoding|Acceptance|tree node" "${LOG}" | head -3 | sed "s/^/   /"
}

echo "=== FDRC sweep math500-128 (v7 ref ~ 8.3 / 10.00) ==="
run_cfg ir4_v7_128s        math500 128 7 128 0
run_cfg ir4_fdrc1_128s     math500 128 8 128 1
run_cfg ir4_fdrc2_128s     math500 128 8 128 2
run_cfg ir4_fdrc3_128s     math500 128 8 128 3
run_cfg ir4_fdrc4_128s     math500 128 8 128 4
run_cfg ir4_fdrc5_128s     math500 128 8 128 5

echo ""
echo "=== mt-bench-40 if math500 shows signal ==="
run_cfg ir4_mtb_v7         mt-bench 40 7 128 0
run_cfg ir4_mtb_fdrc3      mt-bench 40 8 128 3

echo ""
echo "=== Summary ==="
for LOG in ${LOGDIR}/ir4_*.log; do
    [ -f "${LOG}" ] || continue
    TAG=$(basename "${LOG}" .log)
    SPD=$(grep "Decoding speedup:" "${LOG}" | tail -1 | awk '{print $3}')
    TAU=$(grep "Average Acceptance length:" "${LOG}" | tail -1 | awk '{print $4}')
    NODES=$(grep "Average tree node count:" "${LOG}" | tail -1 | awk '{print $5}')
    printf "%-25s spd=%s tau=%s nodes=%s\n" "${TAG}" "${SPD:-FAIL}" "${TAU:-FAIL}" "${NODES:-FAIL}"
done
