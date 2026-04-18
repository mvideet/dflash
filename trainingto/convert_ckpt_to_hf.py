"""Convert a DeepSpeed save_16bit_model checkpoint to a HF-loadable
DFlashDraftModel directory.

    python convert_ckpt_to_hf.py \
        --base z-lab/Qwen3-4B-DFlash-b16 \
        --ckpt dflash_mix_smoke/epoch_0/pytorch_model.bin \
        --out  dflash_mix_smoke/epoch_0_hf

The output dir can then be passed to benchmark.py via --draft-name-or-path.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.dflash import DFlashDraftModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True,
                   help="Base draft path to load config from (e.g. z-lab/Qwen3-4B-DFlash-b16).")
    p.add_argument("--ckpt", required=True,
                   help="Path to pytorch_model.bin produced by DeepSpeed save_16bit_model.")
    p.add_argument("--out", required=True,
                   help="Output directory for the HF-format checkpoint.")
    args = p.parse_args()

    print(f"Loading base model {args.base} (for config)…")
    model = DFlashDraftModel.from_pretrained(
        args.base, torch_dtype=torch.bfloat16,
    )

    print(f"Loading trained state dict from {args.ckpt}…")
    raw = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    draft_sd = {}
    skipped = 0
    for k, v in raw.items():
        if k.startswith("draft_model."):
            draft_sd[k[len("draft_model."):]] = v
        else:
            skipped += 1
    print(f"  {len(draft_sd)} draft parameters loaded; skipped {skipped} non-draft keys")

    missing, unexpected = model.load_state_dict(draft_sd, strict=False)
    print(f"  missing: {len(missing)}  unexpected: {len(unexpected)}")
    if missing[:5]:
        print(f"  first missing: {missing[:5]}")
    if unexpected[:5]:
        print(f"  first unexpected: {unexpected[:5]}")

    os.makedirs(args.out, exist_ok=True)
    print(f"Saving to {args.out}…")
    model.save_pretrained(args.out, safe_serialization=False)

    # DFlashDraftModel.from_pretrained needs the config's dflash_config key +
    # custom code.  model.save_pretrained writes the model + config but does
    # NOT copy trust_remote_code.  Copy config.json + modeling files from base
    # just in case, so `from_pretrained` with trust_remote_code still works.
    print("Done. Now you can benchmark via --draft-name-or-path", args.out)


if __name__ == "__main__":
    main()
