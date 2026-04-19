#!/bin/bash
# Test whether VB v1 handles large B (256, 512) better than stock.
# Hypothesis: VB's reduced phantom paths → better large-B scaling.
set -e
cd /homes/videetm/dflash
RESULT=logs/session_apr18_neurips/vb_large_budget.tsv
echo -e "draft\tblock_size\tmts\tek\tspeedup\ttau\tnodes" > $RESULT

VB=trainingto/dflash_broad_varblock_v1/step_18500_hf
STOCK=z-lab/Qwen3-4B-DFlash-b16

run() {
  local LABEL=$1 DRAFT=$2 BS=$3 MTS=$4 EK=${5:-8}
  LOG=logs/session_apr18_neurips/lb_${LABEL}.log
  echo "=== $LABEL (b=$BS mts=$MTS ek=$EK) ==="
  /homes/videetm/miniforge3/envs/dflash312/bin/torchrun --nproc_per_node=8 --master_port=29501 benchmark.py \
    --dataset math500 --max-samples 256 \
    --model-name-or-path Qwen/Qwen3-4B \
    --draft-name-or-path "$DRAFT" \
    --tree-version 7 --max-tree-size $MTS --expand-k $EK \
    --temperature 0.0 \
    --block-size $BS > "$LOG" 2>&1 || echo "(crash)"
  SP=$(grep "Decoding speedup:" "$LOG" | tail -1 | awk '{print $NF}')
  TAU=$(grep "Average Acceptance length:" "$LOG" | tail -1 | awk '{print $NF}')
  ND=$(grep "Average tree node count:" "$LOG" | tail -1 | awk '{print $NF}')
  printf "%s\tb=%s\t%s\t%s\t%s\t%s\t%s\n" "$LABEL" "$BS" "$MTS" "$EK" "$SP" "$TAU" "$ND" >> $RESULT
  echo "$LABEL: speedup=$SP tau=$TAU nodes=$ND"
}

# Stock vs VB at increasing budgets
# Stock historical: B=128 7.98/10.08, B=256 7.29/10.37, B=512 5.38/10.59
run stock_B128 "$STOCK" 16 128 8
run stock_B256 "$STOCK" 16 256 8
run stock_B512 "$STOCK" 16 512 8
run vb_b16_B128 "$VB" 16 128 8
run vb_b16_B256 "$VB" 16 256 8
run vb_b16_B512 "$VB" 16 512 8
run vb_b20_B128 "$VB" 20 128 8
run vb_b20_B256 "$VB" 20 256 8
run vb_b20_B512 "$VB" 20 512 8
# With VB potentially-reduced phantoms, bigger ek at large budget might help
run vb_b20_B256_ek12 "$VB" 20 256 12
run vb_b20_B512_ek12 "$VB" 20 512 12

echo "=== large-budget scan complete ==="
cat $RESULT
