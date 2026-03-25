#!/usr/bin/env bash
# DFlash benchmark driver — edit the CONFIG section below, then:  ./run_benchmark.sh
#
# Each run appends a block to EXPERIMENT_LOG (full config + key metrics).
# Per-task full stdout/stderr: logs/<dataset>[_<RUN_TAG>].log
#
# -u: unset vars error; no -e so failed benchmarks still get a ledger line (see exit_code)
set -uo pipefail

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset TORCH_LOGS

mkdir -p logs

# =============================================================================
# CONFIG — change experiments here
# =============================================================================

# Models & generation
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B}"
DRAFT_NAME_OR_PATH="${DRAFT_NAME_OR_PATH:-z-lab/Qwen3-4B-DFlash-b16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
# Leave empty to use draft model default block size from checkpoint
BLOCK_SIZE="${BLOCK_SIZE:-}"

# Optional: FR-Spec / freq map (leave empty to disable)
FREQ_PATH="${FREQ_PATH:-}"

# Tasks: "dataset_name:max_samples" per line
TASKS=(
  # "gsm8k:128"
  "mt-bench:80"
)

# Tree / speculative decoding
CHAIN_ATTENTION="${CHAIN_ATTENTION:-false}"
DYNAMIC_BRANCHING="${DYNAMIC_BRANCHING:-true}"
TOP_K="${TOP_K:-3}"
# Tree version: 1=threshold+cap (v1), 2=EAGLE-2 expand+rerank, 3=best-first (recommended)
TREE_VERSION="${TREE_VERSION:-3}"
THETA_UNI="${THETA_UNI:-0.9}"                # v1 only
THETA_BI="${THETA_BI:-0.3}"                  # v1 only
THETA_TRI="${THETA_TRI:-0.1}"                # v1 only
MAX_TREE_SIZE="${MAX_TREE_SIZE:-32}"
ADAPTIVE_DEPTH="${ADAPTIVE_DEPTH:-false}"    # v1 only
ADAPTIVE_DEPTH_THRESHOLD="${ADAPTIVE_DEPTH_THRESHOLD:-0.1}"
PROFILE="${PROFILE:-false}"

# torchrun
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29600}"

# Logging: suffix avoids overwriting logs when re-running the same dataset with different settings
# Example: RUN_TAG="k3_adapt" -> logs/mt-bench_k3_adapt.log
RUN_TAG="${RUN_TAG:-}"
# Append-only ledger of configs + metrics (easy to diff / paste into papers)
EXPERIMENT_LOG="${EXPERIMENT_LOG:-logs/experiment_runs.txt}"

# =============================================================================
# Build CLI
# =============================================================================

append_ledger() {
  local full_log="$1"
  local exit_code="${2:-0}"
  {
    echo ""
    echo "################################################################################"
    echo "RUN $(date -u +"%Y-%m-%dT%H:%M:%SZ")  host=$(hostname -s 2>/dev/null || echo unknown)"
    echo "  exit_code=${exit_code}"
    echo "---- experiment ----"
    echo "  dataset=${DATASET_NAME}  max_samples=${MAX_SAMPLES}"
    echo "  MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
    echo "  DRAFT_NAME_OR_PATH=${DRAFT_NAME_OR_PATH}"
    echo "  MAX_NEW_TOKENS=${MAX_NEW_TOKENS}  TEMPERATURE=${TEMPERATURE}  BLOCK_SIZE=${BLOCK_SIZE:-<default from draft>}"
    echo "  CHAIN_ATTENTION=${CHAIN_ATTENTION}  DYNAMIC_BRANCHING=${DYNAMIC_BRANCHING}"
    echo "  TOP_K=${TOP_K}  TREE_VERSION=${TREE_VERSION}  MAX_TREE_SIZE=${MAX_TREE_SIZE}"
    echo "  THETA_UNI=${THETA_UNI}  THETA_BI=${THETA_BI}  THETA_TRI=${THETA_TRI}"
    echo "  ADAPTIVE_DEPTH=${ADAPTIVE_DEPTH}  ADAPTIVE_DEPTH_THRESHOLD=${ADAPTIVE_DEPTH_THRESHOLD}  PROFILE=${PROFILE}"
    echo "  FREQ_PATH=${FREQ_PATH:-<none>}"
    echo "  full_log=${full_log}"
    echo "---- metrics (grep from log) ----"
    grep "Decoding speedup:" "$full_log" 2>/dev/null | tail -1 || echo "  (missing Decoding speedup)"
    grep "Average Acceptance length:" "$full_log" 2>/dev/null | tail -1 || echo "  (missing Average Acceptance length)"
    grep "Acceptance length histogram:" "$full_log" 2>/dev/null | tail -1 || echo "  (missing histogram)"
    if grep -q "^--- Profile" "$full_log" 2>/dev/null; then
      echo "---- profile (last lines) ----"
      grep -A99 "^--- Profile" "$full_log" 2>/dev/null | tail -n 25 | sed 's/^/  /' || true
    fi
    echo ""
  } >> "$EXPERIMENT_LOG"
}

for task in "${TASKS[@]}"; do
  [[ -z "${task// }" ]] && continue
  [[ "$task" =~ ^# ]] && continue
  IFS=':' read -r DATASET_NAME MAX_SAMPLES <<< "$task"

  echo "========================================================"
  echo "Benchmark: ${DATASET_NAME}  (max_samples=${MAX_SAMPLES})"
  echo "  tree_version=${TREE_VERSION}  mts=${MAX_TREE_SIZE}  k=${TOP_K}  temp=${TEMPERATURE}"
  echo "========================================================"

  EXTRA_ARGS=()
  if [ "${CHAIN_ATTENTION}" = "true" ]; then
    EXTRA_ARGS+=(--chain-attention)
  fi
  if [ "${DYNAMIC_BRANCHING}" = "true" ]; then
    EXTRA_ARGS+=(--dynamic-branching --theta-uni "${THETA_UNI}" --theta-bi "${THETA_BI}" --theta-tri "${THETA_TRI}" --max-tree-size "${MAX_TREE_SIZE}" --tree-version "${TREE_VERSION}")
  fi
  if [ "${ADAPTIVE_DEPTH}" = "true" ]; then
    EXTRA_ARGS+=(--adaptive-depth --adaptive-depth-threshold "${ADAPTIVE_DEPTH_THRESHOLD}")
  fi
  if [ "${CHAIN_ATTENTION}" = "true" ] || [ "${DYNAMIC_BRANCHING}" = "true" ]; then
    EXTRA_ARGS+=(--top-k "${TOP_K}" --expand-k "${TOP_K}")
  fi
  if [ "${PROFILE}" = "true" ]; then
    EXTRA_ARGS+=(--profile)
  fi
  if [ -n "${BLOCK_SIZE}" ]; then
    EXTRA_ARGS+=(--block-size "${BLOCK_SIZE}")
  fi
  if [ -n "${FREQ_PATH}" ]; then
    EXTRA_ARGS+=(--freq-path "${FREQ_PATH}")
  fi

  LOG_BASENAME="${DATASET_NAME}"
  [ -n "${RUN_TAG}" ] && LOG_BASENAME="${DATASET_NAME}_${RUN_TAG}"
  FULL_LOG="logs/${LOG_BASENAME}.log"

  torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    benchmark.py \
    --dataset "${DATASET_NAME}" \
    --max-samples "${MAX_SAMPLES}" \
    --model-name-or-path "${MODEL_NAME_OR_PATH}" \
    --draft-name-or-path "${DRAFT_NAME_OR_PATH}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${FULL_LOG}"
  RC="${PIPESTATUS[0]}"

  append_ledger "${FULL_LOG}" "${RC}"
  echo "Appended summary -> ${EXPERIMENT_LOG}  (exit_code=${RC})"
done

echo "Done. Ledger: ${EXPERIMENT_LOG}"
