#!/bin/bash
# Post-hoc: if any v8 phase-A config beat v7 baseline, this sweep tests
# whether the same config UNLOCKS the v7 B=256 plateau (where phantom paths
# are supposed to dominate).
#
# v7 @ B=128 baseline on this hardware: 7.84 / 9.98 / 129.
# v7 @ B=256 baseline on 8×A100 from program.md: 7.29 / 10.37 / 257.
# We expect v7@B=256 here to drop similarly.

set -o pipefail
unset TORCH_LOGS
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-4B"
DRAFT="z-lab/Qwen3-4B-DFlash-b16"
DATASET="math500"
SAMPLES=64
PORT=29502
EK=8
TEMP=0.0
GPUS="0,2,3,4,5,6,7"
NPROC=7
LOGDIR="logs/v8"
PY=/homes/videetm/miniforge3/envs/dflash312/bin/torchrun

mkdir -p "${LOGDIR}"

run_cfg() {
    local TAG=$1
    local VER=$2
    local MTS=$3
    local BETA=$4
    local GAMMA=$5
    local LAMBDA=$6
    local POOLMULT=$7
    local LOG="${LOGDIR}/${TAG}.log"
    if [ -f "${LOG}" ] && grep -q "Decoding speedup:" "${LOG}"; then
        echo "[skip] ${TAG}"
        return 0
    fi
    echo "[run ] ${TAG}"
    CUDA_VISIBLE_DEVICES=${GPUS} ${PY} \
        --nproc_per_node=${NPROC} --master_port=${PORT} benchmark.py \
        --dataset ${DATASET} --max-samples ${SAMPLES} \
        --model-name-or-path ${MODEL} --draft-name-or-path ${DRAFT} \
        --max-new-tokens 1024 --temperature ${TEMP} \
        --tree-version ${VER} --max-tree-size ${MTS} --expand-k ${EK} \
        --v8-entropy-beta ${BETA} --v8-leaf-gamma ${GAMMA} \
        --v8-overlap-lambda ${LAMBDA} --v8-pool-multiplier ${POOLMULT} \
        > "${LOG}" 2>&1
    RC=$?
    if [ ${RC} -ne 0 ]; then
        echo "[FAIL] ${TAG}"
    else
        grep -E "Decoding speedup|Average Acceptance|Average tree" "${LOG}" | head -3 | sed "s/^/   /"
    fi
}

# v7 baselines at larger B (known to collapse)
run_cfg "v7_b256"      7 256 0.0 0.0 0.0 1
run_cfg "v7_b512"      7 512 0.0 0.0 0.0 1

# v8 with best-of-phase-A (substituted after sweep finishes; default β_e=4 if unset)
BETA_BEST=${BETA_BEST:-4.0}
run_cfg "v8_b256_bb"   8 256 ${BETA_BEST} 0.0 0.0 1
run_cfg "v8_b512_bb"   8 512 ${BETA_BEST} 0.0 0.0 1

echo ""
echo "=== Summary ==="
for LOG in ${LOGDIR}/v7_b256.log ${LOGDIR}/v7_b512.log ${LOGDIR}/v8_b256_bb.log ${LOGDIR}/v8_b512_bb.log; do
    [ -f "${LOG}" ] || continue
    TAG=$(basename "${LOG}" .log)
    SPD=$(grep "Decoding speedup:" "${LOG}" | tail -1 | awk '{print $3}')
    TAU=$(grep "Average Acceptance length:" "${LOG}" | tail -1 | awk '{print $4}')
    NODES=$(grep "Average tree node count:" "${LOG}" | tail -1 | awk '{print $5}')
    printf "%-20s spd=%s tau=%s nodes=%s\n" "${TAG}" "${SPD:-FAIL}" "${TAU:-FAIL}" "${NODES:-FAIL}"
done
