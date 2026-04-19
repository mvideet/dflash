#!/bin/bash
# Eval queue: runs a list of benchmark configs sequentially, waiting for
# 8 free GPUs before each launch.  Skips configs whose result log already
# shows a speedup number (idempotent — rerunning is safe).
set -u
cd /homes/videetm/dflash

VB=trainingto/dflash_broad_varblock_v1/step_18500_hf
STOCK=z-lab/Qwen3-4B-DFlash-b16

wait_for_8_gpus() {
  while true; do
    min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
    if [ "$min_free" -gt 60000 ]; then return 0; fi
    echo "[$(date -u +%H:%M:%S)] waiting for 8 GPUs; min_free=${min_free} MiB"
    sleep 60
  done
}

run() {
  local LABEL=$1 DS=$2 N=$3 DRAFT=$4 BS=$5 MTS=${6:-128} EK=${7:-8}
  local EXTRA=${8:-}
  LOG=logs/session_apr18_neurips/eq_${LABEL}.log
  if [ -f "$LOG" ] && grep -q "Decoding speedup:" "$LOG"; then
    echo "[skip] $LABEL already done"
    return
  fi
  wait_for_8_gpus
  echo "=== $LABEL b=$BS mts=$MTS ek=$EK ==="
  /homes/videetm/miniforge3/envs/dflash312/bin/torchrun --nproc_per_node=8 --master_port=29501 benchmark.py \
    --dataset $DS --max-samples $N \
    --model-name-or-path Qwen/Qwen3-4B \
    --draft-name-or-path "$DRAFT" \
    --tree-version 7 --max-tree-size $MTS --expand-k $EK \
    --temperature 0.0 \
    --block-size $BS $EXTRA > "$LOG" 2>&1 || echo "(crash - will retry later)"
  SP=$(grep "Decoding speedup:" "$LOG" | tail -1 | awk '{print $NF}')
  TAU=$(grep "Average Acceptance length:" "$LOG" | tail -1 | awk '{print $NF}')
  ND=$(grep "Average tree node count:" "$LOG" | tail -1 | awk '{print $NF}')
  echo "$LABEL: speedup=$SP tau=$TAU nodes=$ND"
  RESULT=logs/session_apr18_neurips/eval_queue_results.tsv
  if [ ! -f "$RESULT" ]; then echo -e "label\tdataset\tn\tblock_size\tmts\tek\tspeedup\ttau\tnodes\textra" > $RESULT; fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$LABEL" "$DS" "$N" "$BS" "$MTS" "$EK" "$SP" "$TAU" "$ND" "$EXTRA" >> $RESULT
}

# --- HIGH PRIORITY: adaptive block on mt-bench (fix the regression) ---
run vb_adap_mt_80_aggressive  mt-bench   80 "$VB" 24 128 8 '--adaptive-block-sizes 16,20,24 --adaptive-block-thresholds 0.45,0.65'
run vb_adap_mt_80_balanced    mt-bench   80 "$VB" 24 128 8 '--adaptive-block-sizes 16,20,24 --adaptive-block-thresholds 0.55,0.75'
run vb_adap_mt_80_conservative mt-bench  80 "$VB" 20 128 8 '--adaptive-block-sizes 16,20     --adaptive-block-thresholds 0.75'

# --- Adaptive block elsewhere: confirm no regression ---
run vb_adap_math_256_balanced math500   256 "$VB" 24 128 8 '--adaptive-block-sizes 16,20,24 --adaptive-block-thresholds 0.55,0.75'
run vb_adap_gsm_128_balanced  gsm8k     128 "$VB" 24 128 8 '--adaptive-block-sizes 16,20,24 --adaptive-block-thresholds 0.55,0.75'

# --- LARGE BUDGET with VB (hypothesis: phantom reduction) ---
run stock_B256_math  math500  256 "$STOCK" 16 256 8
run stock_B512_math  math500  256 "$STOCK" 16 512 8
run vb_b20_B256_math math500  256 "$VB"    20 256 8
run vb_b20_B512_math math500  256 "$VB"    20 512 8
run vb_b24_B256_math math500  256 "$VB"    24 256 8

# --- Wider expand_k at larger budget ---
run vb_b20_B256_ek12 math500  256 "$VB"    20 256 12
run vb_b20_B512_ek12 math500  256 "$VB"    20 512 12

# --- Temperature robustness: T=0.6 ---
run vb_b20_math_t06  math500  128 "$VB"    20 128 8 '--temperature 0.6'
run stock_b16_math_t06 math500 128 "$STOCK" 16 128 8 '--temperature 0.6'

echo "=== EVAL QUEUE COMPLETE ==="
cat logs/session_apr18_neurips/eval_queue_results.tsv
