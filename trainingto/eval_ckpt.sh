#!/bin/bash
# Auto-eval a trained checkpoint: convert to HF, run math500 bench on a small
# sample count, extract speedup, append to a results tsv.
#
# Usage: ./eval_ckpt.sh <ckpt_dir> <gpus> <n_samples> <results_tsv>
#   ckpt_dir: path to the dir containing pytorch_model.bin
#   gpus:     comma-separated CUDA_VISIBLE_DEVICES (e.g. "6,7")
#   n_samples: math500 sample count (16 = fast/noisy, 64 = slow/stable)
#   results_tsv: append-target file (tab-separated)

set -euo pipefail

CKPT="$1"
GPUS="$2"
N="${3:-16}"
OUT_TSV="${4:-/homes/videetm/dflash/logs/session_apr18/mix_eval.tsv}"

# Normalise CKPT to absolute so later cd doesn't break it.
if [[ "$CKPT" != /* ]]; then
  CKPT="$(readlink -f "$CKPT")"
fi
HF_DIR="${CKPT}_hf"
cd /homes/videetm/dflash

# File lock: serialize GPU-bench work across concurrent watchers/callers.
LOCK="/tmp/dflash_eval_gpu_${GPUS//,/_}.lock"
exec 9> "$LOCK"
flock 9

if [ ! -d "$HF_DIR" ]; then
  python trainingto/convert_ckpt_to_hf.py \
    --base z-lab/Qwen3-4B-DFlash-b16 \
    --ckpt "${CKPT}/pytorch_model.bin" \
    --out  "$HF_DIR" 2>&1 | tail -1 >&2
fi

N_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
LOG="$(dirname "$CKPT")/$(basename "$CKPT")_bench.log"

CUDA_VISIBLE_DEVICES="$GPUS" torchrun \
  --nproc_per_node="$N_GPUS" \
  --master_port=$((29600 + RANDOM % 100)) \
  benchmark.py \
  --dataset math500 --max-samples "$N" \
  --model-name-or-path Qwen/Qwen3-4B \
  --draft-name-or-path "$HF_DIR" \
  --tree-version 7 --max-tree-size 128 --expand-k 8 \
  --temperature 0.0 > "$LOG" 2>&1

SPEEDUP=$(grep "Decoding speedup:" "$LOG" | tail -1 | awk '{print $NF}')
TAU=$(grep "Average Acceptance length:" "$LOG" | tail -1 | awk '{print $NF}')
NODES=$(grep "Average tree node count:" "$LOG" | tail -1 | awk '{print $NF}')

if [ -z "${SPEEDUP:-}" ]; then
  echo "BENCH FAILED for $CKPT" >&2
  tail -20 "$LOG" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_TSV")"
if [ ! -f "$OUT_TSV" ]; then
  echo -e "ckpt\tn_samples\tgpus\tspeedup\ttau\tnodes\ttimestamp" > "$OUT_TSV"
fi
echo -e "${CKPT}\t${N}\t${GPUS}\t${SPEEDUP}\t${TAU}\t${NODES}\t$(date -Iseconds)" >> "$OUT_TSV"

echo "RESULT  ckpt=$CKPT  speedup=$SPEEDUP  tau=$TAU  nodes=$NODES"
