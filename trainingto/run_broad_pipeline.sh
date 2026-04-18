#!/bin/bash
# Broad-mix data pipeline for NeurIPS variable-block DFlash training.
# 1. Regenerate chat/stem/code prompts with Qwen3-4B (vLLM, 8 GPUs).
# 2. Merge with existing math -> broad train set.
# Math is already regenerated at trainingto/data/nemotron_math/.
set -euo pipefail

ROOT=/homes/videetm/dflash
cd "$ROOT"

PY=/homes/videetm/miniforge3/envs/dflash312/bin/python
DATA=trainingto/data

mkdir -p logs/session_apr18_neurips

echo "== Regen chat (20k) =="
$PY trainingto/regen_nemotron_broad.py \
    --split chat --max-samples 20000 \
    --output-dir "$DATA/nemotron_chat_20k" \
    --tensor-parallel-size 1 --num-gpus 8 \
    2>&1 | tee logs/session_apr18_neurips/regen_chat_20k.log

echo "== Regen stem (20k) =="
$PY trainingto/regen_nemotron_broad.py \
    --split stem --max-samples 20000 \
    --output-dir "$DATA/nemotron_stem_20k" \
    --tensor-parallel-size 1 --num-gpus 8 \
    2>&1 | tee logs/session_apr18_neurips/regen_stem_20k.log

echo "== Regen code (10k) =="
$PY trainingto/regen_nemotron_broad.py \
    --split code --max-samples 10000 \
    --output-dir "$DATA/nemotron_code_10k" \
    --tensor-parallel-size 1 --num-gpus 8 \
    2>&1 | tee logs/session_apr18_neurips/regen_code_10k.log

echo "== Merge broad mix =="
$PY trainingto/mix_split_jsonls.py \
    --inputs \
        "$DATA/nemotron_math/train_nemotron_math.jsonl:0.45" \
        "$DATA/nemotron_chat_20k/train_nemotron_chat.jsonl:1.0" \
        "$DATA/nemotron_stem_20k/train_nemotron_stem.jsonl:1.0" \
        "$DATA/nemotron_code_10k/train_nemotron_code.jsonl:1.0" \
    --output-dir "$DATA/nemotron_broad_150k" \
    --output-name train \
    --test-ratio 0.005 \
    2>&1 | tee logs/session_apr18_neurips/merge_broad.log

echo "== Broad mix ready =="
wc -l "$DATA/nemotron_broad_150k"/*.jsonl
