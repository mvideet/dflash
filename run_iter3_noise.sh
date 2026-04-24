#!/bin/bash
# Iteration 3: noise-floor measurement. Run v7 baseline 3× at N=128 for
# variance estimation, and PDDP / FDRP at N=128 to see whether their
# regressions are real vs cluster-noise.

set -o pipefail
unset TORCH_LOGS
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-4B"
DRAFT="z-lab/Qwen3-4B-DFlash-b16"
PORT=29509
EK=8
TEMP=0.0
GPUS="0,2,3,4,5,6,7"
NPROC=7
LOGDIR="logs/v8"
PY=/homes/videetm/miniforge3/envs/dflash312/bin/torchrun

run_cfg() {
    local TAG=$1; local VER=$2; local MTS=$3; local EXTRA=$4
    local LOG="${LOGDIR}/${TAG}.log"
    if [ -f "${LOG}" ] && grep -q "Decoding speedup:" "${LOG}"; then
        echo "[skip] ${TAG}"
        return 0
    fi
    echo "[run ] ${TAG}"
    CUDA_VISIBLE_DEVICES=${GPUS} ${PY} \
        --nproc_per_node=${NPROC} --master_port=${PORT} benchmark.py \
        --dataset math500 --max-samples 128 \
        --model-name-or-path ${MODEL} --draft-name-or-path ${DRAFT} \
        --max-new-tokens 1024 --temperature ${TEMP} \
        --tree-version ${VER} --max-tree-size ${MTS} --expand-k ${EK} \
        ${EXTRA} > "${LOG}" 2>&1
    grep -E "Decoding|Acceptance|tree node" "${LOG}" | head -3 | sed "s/^/   /"
}

run_cfg ir3_v7_128s_run1      7 128 ""
run_cfg ir3_v7_128s_run2      7 128 ""
run_cfg ir3_v7_128s_run3      7 128 ""
run_cfg ir3_pddp0p5_128s      8 128 "--v8-postdev-beta 0.5"
run_cfg ir3_fdrp0p5_128s      8 128 "--v8-fdrp-beta 0.5 --v8-fdrp-exp 2.0"
run_cfg ir3_fdrp0p2_128s      8 128 "--v8-fdrp-beta 0.2 --v8-fdrp-exp 2.0"

echo "=== Summary (N=128 math500) ==="
for LOG in ${LOGDIR}/ir3_*.log; do
    [ -f "${LOG}" ] || continue
    TAG=$(basename "${LOG}" .log)
    SPD=$(grep "Decoding speedup:" "${LOG}" | tail -1 | awk '{print $3}')
    TAU=$(grep "Average Acceptance length:" "${LOG}" | tail -1 | awk '{print $4}')
    NODES=$(grep "Average tree node count:" "${LOG}" | tail -1 | awk '{print $5}')
    printf "%-25s spd=%s tau=%s nodes=%s\n" "${TAG}" "${SPD:-FAIL}" "${TAU:-FAIL}" "${NODES:-FAIL}"
done
