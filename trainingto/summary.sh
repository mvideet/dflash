#!/bin/bash
# Print a live summary of the master pipeline state.

ROOT=/homes/videetm/dflash
LOGDIR=$ROOT/logs/session_apr18
TSV=$LOGDIR/mix_eval.tsv

echo "=== Pipeline summary $(date +%H:%M:%S) ==="

# Training states
for V in dflash_mix_10k dflash_mix_10k_v2_varblock dflash_mix_10k_v3_ctrlite dflash_mix_10k_v4_ttlite; do
  DIR=$ROOT/trainingto/$V
  if [ ! -d "$DIR" ]; then continue; fi
  DONE_FILE=$DIR/.DONE_TRAINING
  LAST_STEP=$(ls -d $DIR/step_* 2>/dev/null | grep -oE '[0-9]+$' | sort -n | tail -1)
  LAST_STEP=${LAST_STEP:-"(none yet)"}
  if [ -f "$DONE_FILE" ]; then
    echo "  $V : TRAINING DONE  (last ckpt: step_$LAST_STEP)"
  else
    echo "  $V : training in flight  (latest ckpt: step_$LAST_STEP)"
  fi
done

echo ""
echo "=== Eval results (sorted by speedup desc, top 10) ==="
if [ -f "$TSV" ]; then
  head -1 "$TSV"
  awk -F'\t' 'NR>1 {print}' "$TSV" | sort -t$'\t' -k4 -rn | head -10
fi

echo ""
echo "=== Active watchers ==="
ps -ef | grep watch_and_eval.sh | grep -v grep | awk '{print $NF}' | tr '\n' ' '
echo ""
