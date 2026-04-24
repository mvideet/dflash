#!/bin/bash
# Iteration 1: Post-Deviation Depth Penalty (PDDP) theoretically motivated
# β · max(depth − first_dev, 0). Pareto comparison vs v7 at matched B.

set -o pipefail
unset TORCH_LOGS
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-4B"
DRAFT="z-lab/Qwen3-4B-DFlash-b16"
PORT=29507
EK=8
TEMP=0.0
GPUS="0,2,3,4,5,6,7"
NPROC=7
LOGDIR="logs/v8"
PY=/homes/videetm/miniforge3/envs/dflash312/bin/torchrun

mkdir -p "${LOGDIR}"

run_cfg() {
    local TAG=$1; local DATASET=$2; local SAMPLES=$3
    local VER=$4; local MTS=$5; local PDDPB=$6
    local LOG="${LOGDIR}/${TAG}.log"
    if [ -f "${LOG}" ] && grep -q "Decoding speedup:" "${LOG}"; then
        echo "[skip] ${TAG}"
        return 0
    fi
    echo "[run ] ${TAG} (ds=${DATASET} N=${SAMPLES} ver=${VER} B=${MTS} β_pd=${PDDPB})"
    if [ "${VER}" = "7" ]; then EXTRA=""
    else EXTRA="--v8-postdev-beta ${PDDPB}"
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

echo "=== math500-64, B=128 Pareto ==="
run_cfg ir1_m500_v7_b128     math500 64 7 128 0.0
run_cfg ir1_m500_pd0p5_b128  math500 64 8 128 0.5
run_cfg ir1_m500_pd1_b128    math500 64 8 128 1.0
run_cfg ir1_m500_pd2_b128    math500 64 8 128 2.0
run_cfg ir1_m500_pd4_b128    math500 64 8 128 4.0

echo ""
echo "=== math500-64, B=256 Pareto ==="
run_cfg ir1_m500_v7_b256     math500 64 7 256 0.0
run_cfg ir1_m500_pd1_b256    math500 64 8 256 1.0
run_cfg ir1_m500_pd2_b256    math500 64 8 256 2.0

echo ""
echo "=== mt-bench-40 (cross-dataset) ==="
run_cfg ir1_mtb_v7_b128      mt-bench 40 7 128 0.0
run_cfg ir1_mtb_pd1_b128     mt-bench 40 8 128 1.0
run_cfg ir1_mtb_pd2_b128     mt-bench 40 8 128 2.0

echo ""
echo "=== Summary vs v7 at matched B ==="
for LOG in ${LOGDIR}/ir1_*.log; do
    [ -f "${LOG}" ] || continue
    TAG=$(basename "${LOG}" .log)
    SPD=$(grep "Decoding speedup:" "${LOG}" | tail -1 | awk '{print $3}')
    TAU=$(grep "Average Acceptance length:" "${LOG}" | tail -1 | awk '{print $4}')
    NODES=$(grep "Average tree node count:" "${LOG}" | tail -1 | awk '{print $5}')
    printf "%-30s spd=%s tau=%s nodes=%s\n" "${TAG}" "${SPD:-FAIL}" "${TAU:-FAIL}" "${NODES:-FAIL}"
done
