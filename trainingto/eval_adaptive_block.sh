#!/bin/bash
# Test adaptive block-size on mt-bench to see if it recovers the regression.
set -e
cd /homes/videetm/dflash
RESULT=logs/session_apr18_neurips/vb_adaptive.tsv
echo -e "label\tdataset\tn\tspeedup\ttau\tnodes" > $RESULT

VB=trainingto/dflash_broad_varblock_v1/step_18500_hf
STOCK=z-lab/Qwen3-4B-DFlash-b16

run_adaptive() {
  local LABEL=$1 DS=$2 N=$3 DRAFT=$4 SIZES=$5 THRESH=$6 START_B=$7
  LOG=logs/session_apr18_neurips/adaptive_${LABEL}.log
  echo "=== $LABEL sizes=$SIZES thresh=$THRESH ==="
  /homes/videetm/miniforge3/envs/dflash312/bin/torchrun --nproc_per_node=8 --master_port=29501 benchmark.py \
    --dataset $DS --max-samples $N \
    --model-name-or-path Qwen/Qwen3-4B \
    --draft-name-or-path "$DRAFT" \
    --tree-version 7 --max-tree-size 128 --expand-k 8 \
    --temperature 0.0 \
    --block-size $START_B \
    --adaptive-block-sizes "$SIZES" \
    --adaptive-block-thresholds "$THRESH" > "$LOG" 2>&1 || echo "(crash)"
  SP=$(grep "Decoding speedup:" "$LOG" | tail -1 | awk '{print $NF}')
  TAU=$(grep "Average Acceptance length:" "$LOG" | tail -1 | awk '{print $NF}')
  ND=$(grep "Average tree node count:" "$LOG" | tail -1 | awk '{print $NF}')
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$LABEL" "$DS" "$N" "$SP" "$TAU" "$ND" >> $RESULT
  echo "$LABEL: speedup=$SP tau=$TAU nodes=$ND"
}

# VB v1 with adaptive blocks on mt-bench
run_adaptive vb_adap_mt_conservative  mt-bench  80 "$VB" "16,20"      "0.75"       20
run_adaptive vb_adap_mt_balanced      mt-bench  80 "$VB" "16,20,24"   "0.55,0.75"  24
run_adaptive vb_adap_mt_aggressive    mt-bench  80 "$VB" "16,20,24"   "0.45,0.65"  24

# Same on math500 to verify no regression
run_adaptive vb_adap_math_balanced    math500  128 "$VB" "16,20,24"   "0.55,0.75"  24

# gsm8k too
run_adaptive vb_adap_gsm_balanced     gsm8k    128 "$VB" "16,20,24"   "0.55,0.75"  24

# humaneval
run_adaptive vb_adap_he_balanced      humaneval 164 "$VB" "16,20,24"  "0.55,0.75"  24

echo "=== adaptive eval complete ==="
cat $RESULT
