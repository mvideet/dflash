#!/bin/bash
# Run after all 3 splits are regenerated.  Merges into broad mix and kicks
# off training.  Idempotent — safe to re-run.
set -euo pipefail

ROOT=/homes/videetm/dflash
cd "$ROOT"

PY=/homes/videetm/miniforge3/envs/vllm_gen/bin/python
DATA=trainingto/data

LOGDIR=logs/session_apr18_neurips
mkdir -p "$LOGDIR"

echo "== Merge broad mix =="
# math: 225k rows (existing) — subsample 45% = ~100k
# chat: 19k, stem: 17k, code: ~9.5k expected
$PY trainingto/mix_split_jsonls.py \
    --inputs \
        "$DATA/nemotron_math/train_nemotron_math.jsonl:0.45" \
        "$DATA/nemotron_chat_20k/train_nemotron_chat.jsonl:1.0" \
        "$DATA/nemotron_stem_20k/train_nemotron_stem.jsonl:1.0" \
        "$DATA/nemotron_code_10k/train_nemotron_code.jsonl:1.0" \
    --output-dir "$DATA/nemotron_broad_150k" \
    --output-name train \
    --test-ratio 0.005 \
    2>&1 | tee "$LOGDIR/merge_broad.log"

wc -l "$DATA/nemotron_broad_150k"/*.jsonl

echo "== Launch training =="
bash trainingto/run_broad_training.sh
