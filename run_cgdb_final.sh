#!/bin/bash
# Final CGDB confirmation at 256 samples + B=96 stacked with iter-5 finding.

set -o pipefail
unset TORCH_LOGS
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-4B"
DRAFT="z-lab/Qwen3-4B-DFlash-b16"
PORT=29520
EK=8
TEMP=0.0
GPUS="0,1,2,3,4,5,6,7"
NPROC=8
LOGDIR="logs/v8"
PY=/homes/videetm/miniforge3/envs/dflash312/bin/torchrun

run_cfg() {
    local TAG=$1; local DS=$2; local N=$3; local MTS=$4; local VER=$5
    local SD=$6; local HI=$7; local LO=$8; local MK=$9
    local LOG="${LOGDIR}/${TAG}.log"
    if [ -f "${LOG}" ] && grep -q "Decoding speedup:" "${LOG}"; then
        echo "[skip] ${TAG}"
        return 0
    fi
    echo "[run ] ${TAG} ver=${VER} B=${MTS} sd=${SD}"
    if [ "${VER}" = "7" ]; then EXTRA=""
    else EXTRA="--v8-cgdb-shallow-depth ${SD} --v8-cgdb-high-thresh ${HI} --v8-cgdb-low-thresh ${LO} --v8-cgdb-mid-k ${MK}"
    fi
    CUDA_VISIBLE_DEVICES=${GPUS} ${PY} \
        --nproc_per_node=${NPROC} --master_port=${PORT} benchmark.py \
        --dataset ${DS} --max-samples ${N} \
        --model-name-or-path ${MODEL} --draft-name-or-path ${DRAFT} \
        --max-new-tokens 1024 --temperature ${TEMP} \
        --tree-version ${VER} --max-tree-size ${MTS} --expand-k ${EK} \
        ${EXTRA} > "${LOG}" 2>&1
    grep -E "Decoding|Acceptance|tree node" "${LOG}" | head -3 | sed "s/^/   /"
}

echo "=== math500 256 samples ==="
run_cfg irfinal_m500_v7_b128       math500 256 128 7 0 0.0 0.0 0
run_cfg irfinal_m500_cgdb_b128     math500 256 128 8 4 0.1 0.01 4
run_cfg irfinal_m500_v7_b96        math500 256 96  7 0 0.0 0.0 0
run_cfg irfinal_m500_cgdb_b96      math500 256 96  8 4 0.1 0.01 4

echo ""
echo "=== mt-bench 80 samples (full) ==="
run_cfg irfinal_mtb_v7_b128        mt-bench 80 128 7 0 0.0 0.0 0
run_cfg irfinal_mtb_cgdb_b128      mt-bench 80 128 8 3 0.1 0.01 4

echo ""
echo "=== Summary ==="
for LOG in ${LOGDIR}/irfinal_*.log; do
    [ -f "${LOG}" ] || continue
    TAG=$(basename "${LOG}" .log)
    SPD=$(grep "Decoding speedup:" "${LOG}" | tail -1 | awk '{print $3}')
    TAU=$(grep "Average Acceptance length:" "${LOG}" | tail -1 | awk '{print $4}')
    NODES=$(grep "Average tree node count:" "${LOG}" | tail -1 | awk '{print $5}')
    printf "%-30s tau=%s spd=%s nodes=%s\n" "${TAG}" "${TAU:-FAIL}" "${SPD:-FAIL}" "${NODES:-FAIL}"
done
