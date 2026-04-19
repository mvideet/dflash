#!/bin/bash
# Run this after VB v2 training completes (or has enough checkpoints).
# Eval best v2 checkpoints vs stock + v1 across block sizes and datasets.
set -e
cd /homes/videetm/dflash
RESULT=logs/session_apr18_neurips/vb_v2_eval.tsv
echo -e "label\tdataset\tn\tblock_size\tbudget\tspeedup\ttau\tnodes" > $RESULT

# Set via env var before calling:
#   CKPT_HF=trainingto/dflash_broad_varblock_v2/step_N_hf ./eval_v2_full.sh
CKPT_HF="${CKPT_HF:-}"
[ -z "$CKPT_HF" ] && { echo "CKPT_HF env var required"; exit 1; }

run() {
  local LABEL=$1 DS=$2 N=$3 DRAFT=$4 BS=$5 MTS=${6:-128} EK=${7:-8}
  LOG=logs/session_apr18_neurips/v2_${LABEL}.log
  echo "=== $LABEL (b=$BS mts=$MTS ek=$EK) ==="
  /homes/videetm/miniforge3/envs/dflash312/bin/torchrun --nproc_per_node=8 --master_port=29501 benchmark.py \
    --dataset $DS --max-samples $N \
    --model-name-or-path Qwen/Qwen3-4B \
    --draft-name-or-path "$DRAFT" \
    --tree-version 7 --max-tree-size $MTS --expand-k $EK \
    --temperature 0.0 \
    --block-size $BS > "$LOG" 2>&1 || echo "(crash)"
  SP=$(grep "Decoding speedup:" "$LOG" | tail -1 | awk '{print $NF}')
  TAU=$(grep "Average Acceptance length:" "$LOG" | tail -1 | awk '{print $NF}')
  ND=$(grep "Average tree node count:" "$LOG" | tail -1 | awk '{print $NF}')
  printf "%s\t%s\t%s\tb=%s\tB=%s\t%s\t%s\t%s\n" "$LABEL" "$DS" "$N" "$BS" "$MTS" "$SP" "$TAU" "$ND" >> $RESULT
  echo "$LABEL: speedup=$SP tau=$TAU nodes=$ND"
}

# Core block-size sweep on math500 32s (quick triage)
run v2_math32_b16  math500 32 "$CKPT_HF" 16
run v2_math32_b20  math500 32 "$CKPT_HF" 20
run v2_math32_b24  math500 32 "$CKPT_HF" 24
run v2_math32_b28  math500 32 "$CKPT_HF" 28
run v2_math32_b32  math500 32 "$CKPT_HF" 32
run v2_math32_b40  math500 32 "$CKPT_HF" 40

# Best block size confirmation at 256 samples
run v2_math256_b20  math500 256 "$CKPT_HF" 20
run v2_math256_b24  math500 256 "$CKPT_HF" 24
run v2_math256_b28  math500 256 "$CKPT_HF" 28
run v2_math256_b32  math500 256 "$CKPT_HF" 32

# Cross-dataset at the winning block size (assume b=20; override post-facto if different)
BS_CROSS=${BS_CROSS:-20}
run v2_mtbench_bC   mt-bench    80 "$CKPT_HF" $BS_CROSS
run v2_gsm8k_bC     gsm8k      128 "$CKPT_HF" $BS_CROSS
run v2_humaneval_bC humaneval  164 "$CKPT_HF" $BS_CROSS

# Aggressive budget: VB may handle B=256 where stock collapsed
run v2_math256_B256  math500 256 "$CKPT_HF" 20 256 8
run v2_math256_B512  math500 256 "$CKPT_HF" 20 512 8

echo "=== v2 eval complete ==="
cat $RESULT
