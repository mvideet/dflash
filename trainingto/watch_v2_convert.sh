#!/bin/bash
# Poll VB v2 savedir for new step_N dirs and convert each to step_N_hf.
# CPU-only conversion, doesn't steal GPU from training.
SAVEDIR=/homes/videetm/dflash/trainingto/dflash_broad_varblock_v3_warm
while true; do
  for d in "$SAVEDIR"/step_*; do
    [ -d "$d" ] || continue
    [[ "$d" == *_hf ]] && continue
    HF_DIR="${d}_hf"
    [ -d "$HF_DIR" ] && continue
    [ -f "$d/pytorch_model.bin" ] || continue
    # Check file-size stability (not mid-write)
    SZ1=$(stat -c %s "$d/pytorch_model.bin")
    sleep 2
    SZ2=$(stat -c %s "$d/pytorch_model.bin")
    [ "$SZ1" != "$SZ2" ] && continue
    echo "[watch-v2] converting $d"
    /homes/videetm/miniforge3/envs/dflash/bin/python /homes/videetm/dflash/trainingto/convert_ckpt_to_hf.py \
      --base z-lab/Qwen3-4B-DFlash-b16 \
      --ckpt "$d/pytorch_model.bin" \
      --out "$HF_DIR" 2>&1 | tail -1
  done
  sleep 60
done
