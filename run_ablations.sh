#!/usr/bin/env bash
# ======================================================================
# Paper ablation suite for v4 (prefix-aware tree)
#
# Experiments:
#   A) Temperature sensitivity      — v1/v2/v3/v4 at temp=0.6, all 11 datasets
#   B) v3 vs v4 ablation            — isolate Phase 2 contribution, all 11 datasets
#   C) expand_k sensitivity         — K=2,3,4,5 on 3 benchmarks
#   D) Profile overhead             — v4 with --profile on 3 benchmarks
#   E) Acceptance length histograms — v3 vs v4 on 3 benchmarks (parsed from logs)
#
# Usage:
#   ./run_ablations.sh              # run all experiments
#   ABLATION=A ./run_ablations.sh   # run only experiment A
#   ABLATION=B,C ./run_ablations.sh # run B and C
#
# Results: logs/ablation_*.txt summary files + per-run logs under logs/
# ======================================================================
set -uo pipefail

cd "$(dirname "$0")"
mkdir -p logs
unset TORCH_LOGS

NPROC="${NPROC_PER_NODE:-8}"
PORT="${MASTER_PORT:-29700}"
MODEL="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B}"
DRAFT="${DRAFT_NAME_OR_PATH:-z-lab/Qwen3-4B-DFlash-b16}"
MAX_TOKENS="${MAX_NEW_TOKENS:-2048}"
MTS="${ABLATION_MTS:-32}"
EXPAND_K_DEFAULT="${ABLATION_EXPAND_K:-3}"
TOP_K="${ABLATION_TOP_K:-3}"

ABLATION="${ABLATION:-A,B,C,D,E}"

ALL_DATASETS=(
  "gsm8k:256"
  "math500:256"
  "aime24:64"
  "aime25:64"
  "alpaca:256"
  "mt-bench:80"
  "humaneval:164"
  "mbpp:256"
  "lbpp:256"
  "swe-bench:128"
  "livecodebench:256"
)

REPR_DATASETS=(
  "math500:256"
  "mt-bench:80"
  "humaneval:164"
)

# ======================================================================
# Helper: run one benchmark config
# ======================================================================
run_one() {
  local SUMMARY="$1"
  local DS_NAME="$2"
  local DS_SAMPLES="$3"
  local TEMP="$4"
  local MODE_TAG="$5"
  local LOG_PREFIX="$6"
  shift 6

  local LOG_FILE="logs/${LOG_PREFIX}_${DS_NAME}_${MODE_TAG}_t${TEMP}.log"

  echo ""
  echo ">>> ${DS_NAME}  n=${DS_SAMPLES}  temp=${TEMP}  ${MODE_TAG}"
  echo ">>> ${DS_NAME}  n=${DS_SAMPLES}  temp=${TEMP}  ${MODE_TAG}" >> "$SUMMARY"

  torchrun \
    --nproc_per_node="$NPROC" \
    --master_port="$PORT" \
    benchmark.py \
    --dataset "$DS_NAME" \
    --max-samples "$DS_SAMPLES" \
    --model-name-or-path "$MODEL" \
    --draft-name-or-path "$DRAFT" \
    --max-new-tokens "$MAX_TOKENS" \
    --temperature "$TEMP" \
    "$@" \
    2>&1 | tee "$LOG_FILE"
  local RC="${PIPESTATUS[0]}"

  local SPEEDUP ACC NODES HIST
  SPEEDUP=$(grep "Decoding speedup:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+$' || echo "N/A")
  ACC=$(grep "Average Acceptance length:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+$' || echo "N/A")
  NODES=$(grep "Average tree node count:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+$' || echo "N/A")
  HIST=$(grep "Acceptance length histogram:" "$LOG_FILE" 2>/dev/null | tail -1 || echo "N/A")

  local LINE="${DS_NAME}  temp=${TEMP}  ${MODE_TAG}  speedup=${SPEEDUP}  avg_accept=${ACC}  avg_nodes=${NODES}  exit=${RC}"
  echo "$LINE"
  echo "$LINE" >> "$SUMMARY"
  echo "HIST: ${HIST}" >> "$SUMMARY"

  PORT=$((PORT + 1))
}

# ======================================================================
# A) Temperature sensitivity: v1/v2/v3/v4 at temp=0.6, all datasets
# ======================================================================
run_A() {
  local SUMMARY="logs/ablation_A_temperature.txt"
  echo "" > "$SUMMARY"
  echo "========================================================================" >> "$SUMMARY"
  echo "Ablation A: Temperature sensitivity (temp=0.6) — $(date -u)" >> "$SUMMARY"
  echo "========================================================================" >> "$SUMMARY"

  for DS_SPEC in "${ALL_DATASETS[@]}"; do
    IFS=':' read -r DS_NAME DS_SAMPLES <<< "$DS_SPEC"

    run_one "$SUMMARY" "$DS_NAME" "$DS_SAMPLES" "0.6" "v1_thresh" "ablA" \
      --dynamic-branching --tree-version 1 --max-tree-size "$MTS" \
      --top-k "$TOP_K" --theta-uni 0.9 --theta-bi 0.3 --theta-tri 0.1

    run_one "$SUMMARY" "$DS_NAME" "$DS_SAMPLES" "0.6" "v2_eagle2" "ablA" \
      --dynamic-branching --tree-version 2 --max-tree-size "$MTS" --expand-k "$EXPAND_K_DEFAULT"

    run_one "$SUMMARY" "$DS_NAME" "$DS_SAMPLES" "0.6" "v3_bestfirst" "ablA" \
      --dynamic-branching --tree-version 3 --max-tree-size "$MTS" --expand-k "$EXPAND_K_DEFAULT"

    run_one "$SUMMARY" "$DS_NAME" "$DS_SAMPLES" "0.6" "v4_prefixaware" "ablA" \
      --dynamic-branching --tree-version 4 --max-tree-size "$MTS" --expand-k "$EXPAND_K_DEFAULT"
  done

  echo ""
  echo "=== Ablation A complete. Summary: $SUMMARY ==="
}

# ======================================================================
# B) v3 vs v4 ablation: isolate Phase 2, all datasets, temp=0.0
# ======================================================================
run_B() {
  local SUMMARY="logs/ablation_B_v3_vs_v4.txt"
  echo "" > "$SUMMARY"
  echo "========================================================================" >> "$SUMMARY"
  echo "Ablation B: v3 vs v4 (Phase 2 contribution) — $(date -u)" >> "$SUMMARY"
  echo "max_tree_size=${MTS}  expand_k=${EXPAND_K_DEFAULT}" >> "$SUMMARY"
  echo "========================================================================" >> "$SUMMARY"

  for DS_SPEC in "${ALL_DATASETS[@]}"; do
    IFS=':' read -r DS_NAME DS_SAMPLES <<< "$DS_SPEC"

    run_one "$SUMMARY" "$DS_NAME" "$DS_SAMPLES" "0.0" "v3_bestfirst" "ablB" \
      --dynamic-branching --tree-version 3 --max-tree-size "$MTS" --expand-k "$EXPAND_K_DEFAULT"

    run_one "$SUMMARY" "$DS_NAME" "$DS_SAMPLES" "0.0" "v4_prefixaware" "ablB" \
      --dynamic-branching --tree-version 4 --max-tree-size "$MTS" --expand-k "$EXPAND_K_DEFAULT"
  done

  echo ""
  echo "=== Ablation B complete. Summary: $SUMMARY ==="
}

# ======================================================================
# C) expand_k sensitivity: K=2,3,4,5 on representative benchmarks
# ======================================================================
run_C() {
  local SUMMARY="logs/ablation_C_expand_k.txt"
  echo "" > "$SUMMARY"
  echo "========================================================================" >> "$SUMMARY"
  echo "Ablation C: expand_k sensitivity (K=2,3,4,5) — $(date -u)" >> "$SUMMARY"
  echo "max_tree_size=${MTS}  tree_version=4" >> "$SUMMARY"
  echo "========================================================================" >> "$SUMMARY"

  for K in 2 3 4 5; do
    for DS_SPEC in "${REPR_DATASETS[@]}"; do
      IFS=':' read -r DS_NAME DS_SAMPLES <<< "$DS_SPEC"

      run_one "$SUMMARY" "$DS_NAME" "$DS_SAMPLES" "0.0" "v4_k${K}" "ablC" \
        --dynamic-branching --tree-version 4 --max-tree-size "$MTS" --expand-k "$K"
    done
  done

  echo ""
  echo "=== Ablation C complete. Summary: $SUMMARY ==="
}

# ======================================================================
# D) Profile overhead: v4 with --profile on representative benchmarks
# ======================================================================
run_D() {
  local SUMMARY="logs/ablation_D_profile.txt"
  echo "" > "$SUMMARY"
  echo "========================================================================" >> "$SUMMARY"
  echo "Ablation D: v4 profiling breakdown — $(date -u)" >> "$SUMMARY"
  echo "========================================================================" >> "$SUMMARY"

  for DS_SPEC in "${REPR_DATASETS[@]}"; do
    IFS=':' read -r DS_NAME DS_SAMPLES <<< "$DS_SPEC"

    local LOG_FILE="logs/ablD_${DS_NAME}_v4_profile.log"

    echo ""
    echo ">>> ${DS_NAME}  n=${DS_SAMPLES}  temp=0.0  v4_profile"
    echo ">>> ${DS_NAME}  n=${DS_SAMPLES}  temp=0.0  v4_profile" >> "$SUMMARY"

    torchrun \
      --nproc_per_node="$NPROC" \
      --master_port="$PORT" \
      benchmark.py \
      --dataset "$DS_NAME" \
      --max-samples "$DS_SAMPLES" \
      --model-name-or-path "$MODEL" \
      --draft-name-or-path "$DRAFT" \
      --max-new-tokens "$MAX_TOKENS" \
      --temperature 0.0 \
      --dynamic-branching \
      --tree-version 4 \
      --max-tree-size "$MTS" \
      --expand-k "$EXPAND_K_DEFAULT" \
      --profile \
      2>&1 | tee "$LOG_FILE"

    echo "--- Profile from ${DS_NAME} ---" >> "$SUMMARY"
    grep -A 20 "Profile (CUDA-synced)" "$LOG_FILE" >> "$SUMMARY" 2>/dev/null || echo "  (no profile data)" >> "$SUMMARY"
    echo "" >> "$SUMMARY"

    PORT=$((PORT + 1))
  done

  echo ""
  echo "=== Ablation D complete. Summary: $SUMMARY ==="
}

# ======================================================================
# E) Acceptance length histograms: v3 vs v4 on representative benchmarks
#    (data is captured from the "Acceptance length histogram:" line in logs)
# ======================================================================
run_E() {
  local SUMMARY="logs/ablation_E_histograms.txt"
  echo "" > "$SUMMARY"
  echo "========================================================================" >> "$SUMMARY"
  echo "Ablation E: Acceptance length histograms (v3 vs v4) — $(date -u)" >> "$SUMMARY"
  echo "========================================================================" >> "$SUMMARY"

  for DS_SPEC in "${REPR_DATASETS[@]}"; do
    IFS=':' read -r DS_NAME DS_SAMPLES <<< "$DS_SPEC"

    run_one "$SUMMARY" "$DS_NAME" "$DS_SAMPLES" "0.0" "v3_bestfirst" "ablE" \
      --dynamic-branching --tree-version 3 --max-tree-size "$MTS" --expand-k "$EXPAND_K_DEFAULT"

    run_one "$SUMMARY" "$DS_NAME" "$DS_SAMPLES" "0.0" "v4_prefixaware" "ablE" \
      --dynamic-branching --tree-version 4 --max-tree-size "$MTS" --expand-k "$EXPAND_K_DEFAULT"
  done

  echo ""
  echo "=== Ablation E complete. Summary: $SUMMARY ==="
}

# ======================================================================
# Dispatch
# ======================================================================
echo ""
echo "========================================================================"
echo "  DFlash Ablation Suite"
echo "  Experiments: ${ABLATION}"
echo "  Model: ${MODEL}  Draft: ${DRAFT}"
echo "  max_tree_size=${MTS}  expand_k=${EXPAND_K_DEFAULT}  top_k=${TOP_K}"
echo "========================================================================"

IFS=',' read -r -a EXPS <<< "$ABLATION"
for EXP in "${EXPS[@]}"; do
  case "$EXP" in
    A) echo ""; echo ">>> Running Ablation A: Temperature sensitivity ..."; run_A ;;
    B) echo ""; echo ">>> Running Ablation B: v3 vs v4 (Phase 2) ..."; run_B ;;
    C) echo ""; echo ">>> Running Ablation C: expand_k sensitivity ..."; run_C ;;
    D) echo ""; echo ">>> Running Ablation D: Profile overhead ..."; run_D ;;
    E) echo ""; echo ">>> Running Ablation E: Histograms ..."; run_E ;;
    *) echo "Unknown experiment: $EXP (valid: A,B,C,D,E)" ;;
  esac
done

echo ""
echo "========================================================================"
echo "  All requested ablations complete."
echo "  Summary files:"
for EXP in "${EXPS[@]}"; do
  case "$EXP" in
    A) echo "    A: logs/ablation_A_temperature.txt" ;;
    B) echo "    B: logs/ablation_B_v3_vs_v4.txt" ;;
    C) echo "    C: logs/ablation_C_expand_k.txt" ;;
    D) echo "    D: logs/ablation_D_profile.txt" ;;
    E) echo "    E: logs/ablation_E_histograms.txt" ;;
  esac
done
echo "========================================================================"
