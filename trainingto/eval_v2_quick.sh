#!/bin/bash
# Quick eval of a v2 checkpoint as it becomes available.
# Used by watch_v2_eval.sh for on-arrival benchmarking.
# Usage: eval_v2_quick.sh <step_N_hf_dir> [dataset] [samples]
set -e
cd /homes/videetm/dflash

CKPT=$1
DS=${2:-math500}
N=${3:-32}
BS_LIST=${BS_LIST:-"16 20 24 28"}
LABEL=$(basename "$CKPT")

for BS in $BS_LIST; do
  LOG=logs/session_apr18_neurips/v2eval_${LABEL}_${DS}${N}_b${BS}.log
  if [ -f "$LOG" ] && grep -q "Decoding speedup:" "$LOG"; then
    echo "[skip] $LABEL b=$BS already done"; continue
  fi
  echo "=== $LABEL $DS-${N} b=$BS ==="
  /homes/videetm/miniforge3/envs/dflash312/bin/torchrun --nproc_per_node=8 --master_port=29501 benchmark.py \
    --dataset $DS --max-samples $N \
    --model-name-or-path Qwen/Qwen3-4B \
    --draft-name-or-path "$CKPT" \
    --tree-version 7 --max-tree-size 128 --expand-k 8 \
    --temperature 0.0 \
    --block-size $BS > "$LOG" 2>&1 || echo "(crash)"
  SP=$(grep "Decoding speedup:" "$LOG" | tail -1 | awk '{print $NF}')
  TAU=$(grep "Average Acceptance length:" "$LOG" | tail -1 | awk '{print $NF}')
  ND=$(grep "Average tree node count:" "$LOG" | tail -1 | awk '{print $NF}')
  echo "  -> speedup=$SP tau=$TAU nodes=$ND"
  RESULT=logs/session_apr18_neurips/v2_ckpt_results.tsv
  if [ ! -f "$RESULT" ]; then echo -e "ckpt\tdataset\tn\tblock_size\tspeedup\ttau\tnodes" > $RESULT; fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$LABEL" "$DS" "$N" "$BS" "$SP" "$TAU" "$ND" >> $RESULT
done
