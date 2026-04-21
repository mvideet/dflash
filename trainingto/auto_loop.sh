#!/bin/bash
# Infinite auto-research loop. Keeps firing experiments from the queue
# until user interrupts. Each experiment logs to results.tsv.
#
# Queue design:
#   - Train/eval alternates based on what's up
#   - New training experiments added as ideas arise
#   - Adaptive-block variants, large-budget sweeps, cross-dataset
set -u
cd /homes/videetm/dflash

SESSION_LOG=logs/session_apr18_neurips/auto_loop.log
mkdir -p logs/session_apr18_neurips

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$SESSION_LOG"; }

wait_8_gpus() {
  while true; do
    min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
    if [ "$min_free" -gt 60000 ]; then
      sleep 30
      min_free_2=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
      [ "$min_free_2" -gt 60000 ] && return 0
    fi
    log "waiting for 8 GPUs (min_free=${min_free}M)"
    sleep 60
  done
}

# Skip condition: log exists and has "Decoding speedup:"
already_done() {
  [ -f "$1" ] && grep -q "Decoding speedup:" "$1"
}

bench_v3_at() {
  local CKPT=$1 BS=$2 MTS=${3:-128} EK=${4:-8} DS=${5:-math500} N=${6:-256} EXTRA=${7:-}
  local NAME="$(basename $CKPT)_${DS}_${N}_b${BS}_B${MTS}_ek${EK}"
  local LOG=logs/session_apr18_neurips/al_${NAME}.log
  already_done "$LOG" && { log "[skip] $NAME"; return; }
  wait_8_gpus
  log "=== $NAME ==="
  /homes/videetm/miniforge3/envs/dflash312/bin/torchrun --nproc_per_node=8 --master_port=29501 benchmark.py \
    --dataset $DS --max-samples $N \
    --model-name-or-path Qwen/Qwen3-4B \
    --draft-name-or-path "$CKPT" \
    --tree-version 7 --max-tree-size $MTS --expand-k $EK \
    --temperature 0.0 --block-size $BS $EXTRA > "$LOG" 2>&1 || log "  (crash)"
  SP=$(grep "Decoding speedup:" "$LOG" | tail -1 | awk '{print $NF}')
  TAU=$(grep "Average Acceptance length:" "$LOG" | tail -1 | awk '{print $NF}')
  ND=$(grep "Average tree node count:" "$LOG" | tail -1 | awk '{print $NF}')
  log "  $NAME: speedup=$SP tau=$TAU nodes=$ND"
  local RESULT=logs/session_apr18_neurips/auto_loop_results.tsv
  [ ! -f "$RESULT" ] && echo -e "name\tspeedup\ttau\tnodes" > $RESULT
  printf "%s\t%s\t%s\t%s\n" "$NAME" "$SP" "$TAU" "$ND" >> $RESULT
}

VB=trainingto/dflash_broad_varblock_v1/step_18500_hf
V3=trainingto/dflash_broad_varblock_v3_warm/step_37000_hf
V4=trainingto/dflash_broad_v4_narrow/epoch_0
STOCK=z-lab/Qwen3-4B-DFlash-b16

# ============ MAIN LOOP ============
ITER=0
while true; do
  ITER=$((ITER+1))
  log "====== Iter $ITER ======"

  # Wait for v4-narrow to finish if it's still running
  while ps -ef | grep "main_mix.*v4_narrow" | grep -v grep >/dev/null; do
    log "waiting for v4-narrow to finish training"
    sleep 300
  done

  # ---- Eval batch for iteration N ----
  # V4-narrow (if exists) at b=16, 20, 24
  if [ -d "$V4" ] && [ -f "$V4/pytorch_model.bin" ]; then
    V4HF="${V4}_hf"
    if [ ! -d "$V4HF" ]; then
      log "converting v4 epoch_0"
      /homes/videetm/miniforge3/envs/dflash/bin/python trainingto/convert_ckpt_to_hf.py \
        --base z-lab/Qwen3-4B-DFlash-b16 --ckpt "$V4/pytorch_model.bin" --out "$V4HF" 2>&1 | tail -1
    fi
    bench_v3_at "$V4HF" 16
    bench_v3_at "$V4HF" 20
    bench_v3_at "$V4HF" 24
    # Cross-dataset at best block size
    bench_v3_at "$V4HF" 20 128 8 mt-bench  80
    bench_v3_at "$V4HF" 20 128 8 gsm8k    128
    bench_v3_at "$V4HF" 20 128 8 humaneval 164
  fi

  # Anchor-entropy adaptive (decay=0.0) on mt-bench
  bench_v3_at "$VB" 24 128 8 mt-bench 80 "--adaptive-block-sizes 16,20,24 --adaptive-block-thresholds 0.55,0.75 --adaptive-block-ewma-decay 0.0"
  bench_v3_at "$V3" 24 128 8 mt-bench 80 "--adaptive-block-sizes 16,20,24,28 --adaptive-block-thresholds 0.50,0.65,0.80 --adaptive-block-ewma-decay 0.0"
  bench_v3_at "$V3" 28 256 8 math500 256 "--adaptive-block-sizes 20,24,28 --adaptive-block-thresholds 0.70,0.85 --adaptive-block-ewma-decay 0.0"

  # V3 large budget comparisons (if not done)
  bench_v3_at "$V3" 28 256 8  math500 256
  bench_v3_at "$V3" 24 256 8  math500 256
  bench_v3_at "$V3" 28 512 8  math500 256

  # V3 cross-dataset at b=28 (best block)
  bench_v3_at "$V3" 28 128 8  gsm8k 128
  bench_v3_at "$V3" 28 128 8  humaneval 164

  # ---- Plan next training if we have the savedirs needed ----
  if [ $ITER -eq 1 ]; then
    log "planning iter-2 training: v5_aggressive (warm from v3, epochs=3, blocks {16,24,28,32,40})"
    cat > trainingto/run_v5_aggressive.sh << 'EOS'
#!/bin/bash
set -euo pipefail
cd /homes/videetm/dflash/trainingto
export PATH=/homes/videetm/miniforge3/envs/dflash/bin:$PATH
export DFLASH_ATTN_IMPL=sdpa
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT=3600000
deepspeed --include "localhost:0,1,2,3,4,5,6,7" --master_port=29706 main_mix.py \
    --basepath Qwen/Qwen3-4B \
    --draftpath dflash_broad_varblock_v3_warm/step_37000_hf \
    --trainpath data/nemotron_broad_150k/train.jsonl \
    --testpath  data/nemotron_broad_150k/test.jsonl \
    --savedir   dflash_v5_aggressive \
    --deepspeed_config ds_config.json \
    --num_epochs 2 \
    --gamma-loss 12.0 \
    --random-anchors --anchors-per-seq 24 \
    --block-sizes 16,24,28,32,40 \
    --block-size-probs 1,2,2,2,1 \
    --ctr-weight 0.3 \
    --ttt-weight 0.0 \
    --max-tree-size 16 \
    --tree-expand-k 5 \
    --save-every 1000 \
    2>&1 | tee /homes/videetm/dflash/logs/session_apr18_neurips/train_v5_aggressive.log
EOS
    chmod +x trainingto/run_v5_aggressive.sh
  fi

  # Launch planned training (after first iteration's evals)
  if [ $ITER -eq 1 ]; then
    log "launching v5 aggressive training"
    wait_8_gpus
    nohup bash trainingto/run_v5_aggressive.sh > logs/session_apr18_neurips/v5_launch.log 2>&1 &
    V5_PID=$!
    log "v5 PID: $V5_PID"
    # Wait for v5 completion before continuing
    while ps -p $V5_PID >/dev/null 2>&1; do
      sleep 300
      last=$(grep -oE "\[MIX\] step=[0-9]+" logs/session_apr18_neurips/train_v5_aggressive.log 2>/dev/null | tail -1 || echo "none")
      log "v5 training: last=$last"
    done
    log "v5 completed"
  fi

  # After iter 2 plan more
  if [ $ITER -gt 3 ]; then
    log "iter > 3; idle-looping with random useful exps"
    # Pick random unexplored budget/block combo
    BS_OPTS=(16 20 24 28 32)
    MTS_OPTS=(128 256)
    BS=${BS_OPTS[$((RANDOM % 5))]}
    MTS=${MTS_OPTS[$((RANDOM % 2))]}
    bench_v3_at "$V3" $BS $MTS 8 math500 256
    sleep 120
  fi

  log "iter $ITER complete"
done
