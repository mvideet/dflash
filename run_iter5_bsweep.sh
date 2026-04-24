#!/bin/bash
# Iteration 5: B-Pareto scan. All my selection tweaks (iters 1-4) regressed
# tau. Only unexplored lever: is v7 B=128 genuinely optimal, or does a smaller
# B give equal tau at lower comm cost (and thus higher speedup on this comm-
# bound hardware)?

set -o pipefail
unset TORCH_LOGS
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-4B"
DRAFT="z-lab/Qwen3-4B-DFlash-b16"
PORT=29511
EK=8
TEMP=0.0
# Use all 8 GPUs now (previous tenant finished)
GPUS="0,1,2,3,4,5,6,7"
NPROC=8
LOGDIR="logs/v8"
PY=/homes/videetm/miniforge3/envs/dflash312/bin/torchrun

run_cfg() {
    local TAG=$1; local DATASET=$2; local SAMPLES=$3
    local MTS=$4
    local LOG="${LOGDIR}/${TAG}.log"
    if [ -f "${LOG}" ] && grep -q "Decoding speedup:" "${LOG}"; then
        echo "[skip] ${TAG}"
        return 0
    fi
    echo "[run ] ${TAG} B=${MTS}"
    CUDA_VISIBLE_DEVICES=${GPUS} ${PY} \
        --nproc_per_node=${NPROC} --master_port=${PORT} benchmark.py \
        --dataset ${DATASET} --max-samples ${SAMPLES} \
        --model-name-or-path ${MODEL} --draft-name-or-path ${DRAFT} \
        --max-new-tokens 1024 --temperature ${TEMP} \
        --tree-version 7 --max-tree-size ${MTS} --expand-k ${EK} \
        > "${LOG}" 2>&1
    grep -E "Decoding|Acceptance|tree node" "${LOG}" | head -3 | sed "s/^/   /"
}

echo "=== math500-128 B-Pareto scan (8 GPUs) ==="
run_cfg ir5_b64_m500      math500 128 64
run_cfg ir5_b96_m500      math500 128 96
run_cfg ir5_b112_m500     math500 128 112
run_cfg ir5_b128_m500     math500 128 128
run_cfg ir5_b144_m500     math500 128 144
run_cfg ir5_b160_m500     math500 128 160

echo ""
echo "=== mt-bench-40 ==="
run_cfg ir5_b64_mtb       mt-bench 40 64
run_cfg ir5_b96_mtb       mt-bench 40 96
run_cfg ir5_b112_mtb      mt-bench 40 112
run_cfg ir5_b128_mtb      mt-bench 40 128

echo ""
echo "=== Summary ==="
for LOG in ${LOGDIR}/ir5_*.log; do
    [ -f "${LOG}" ] || continue
    TAG=$(basename "${LOG}" .log)
    SPD=$(grep "Decoding speedup:" "${LOG}" | tail -1 | awk '{print $3}')
    TAU=$(grep "Average Acceptance length:" "${LOG}" | tail -1 | awk '{print $4}')
    NODES=$(grep "Average tree node count:" "${LOG}" | tail -1 | awk '{print $5}')
    printf "%-25s spd=%s tau=%s nodes=%s\n" "${TAG}" "${SPD:-FAIL}" "${TAU:-FAIL}" "${NODES:-FAIL}"
done
