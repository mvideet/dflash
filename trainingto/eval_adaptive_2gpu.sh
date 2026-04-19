#!/bin/bash
# 2-GPU eval of adaptive-block-sizes on VB v1, using GPUs 0 and 4.
# Direct usable result even during junxiang's co-tenant chaos.
set -e
cd /homes/videetm/dflash
RESULT=logs/session_apr18_neurips/vb_adaptive_2gpu.tsv
echo -e "label\tdataset\tn\tspeedup\ttau\tnodes" > $RESULT

VB=trainingto/dflash_broad_varblock_v1/step_18500_hf
STOCK=z-lab/Qwen3-4B-DFlash-b16

run_2gpu() {
  local LABEL=$1 DS=$2 N=$3 DRAFT=$4 BS=$5 EXTRA=${6:-}
  LOG=logs/session_apr18_neurips/a2g_${LABEL}.log
  echo "=== $LABEL b=$BS extra=$EXTRA ==="
  CUDA_VISIBLE_DEVICES=0,4 /homes/videetm/miniforge3/envs/dflash312/bin/torchrun \
    --nproc_per_node=2 --master_port=29505 benchmark.py \
    --dataset $DS --max-samples $N \
    --model-name-or-path Qwen/Qwen3-4B \
    --draft-name-or-path "$DRAFT" \
    --tree-version 7 --max-tree-size 128 --expand-k 8 \
    --temperature 0.0 \
    --block-size $BS $EXTRA > "$LOG" 2>&1 || echo "(crash)"
  SP=$(grep "Decoding speedup:" "$LOG" | tail -1 | awk '{print $NF}')
  TAU=$(grep "Average Acceptance length:" "$LOG" | tail -1 | awk '{print $NF}')
  ND=$(grep "Average tree node count:" "$LOG" | tail -1 | awk '{print $NF}')
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$LABEL" "$DS" "$N" "$SP" "$TAU" "$ND" >> $RESULT
  echo "$LABEL: speedup=$SP tau=$TAU nodes=$ND"
}

# Paired 2-GPU comparisons — all same HW state, so deltas are meaningful.
run_2gpu stock_b16_mt  mt-bench 32 "$STOCK" 16
run_2gpu vb_b20_mt     mt-bench 32 "$VB" 20
run_2gpu vb_adap_mt    mt-bench 32 "$VB" 24 "--adaptive-block-sizes 16,20,24 --adaptive-block-thresholds 0.55,0.75"

run_2gpu stock_b16_math  math500 32 "$STOCK" 16
run_2gpu vb_b20_math     math500 32 "$VB" 20

echo "=== 2-GPU adaptive eval complete ==="
cat $RESULT
