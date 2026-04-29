"""Test whether the U-shaped per-depth accuracy curve is universal across B.

We measured at B=4 (decay_compare). Here verify at B=1 and B=8.
If U-shape holds at all B, we know the UST design generalizes.
If U-shape varies by B, we need per-B weight schedules.
"""
import argparse, json, os
from collections import Counter
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import DFlashDraftModel, load_and_process_dataset
from dflash_batched import dflash_generate_batched, vanilla_ar_generate_batched
from benchmark_batched import tokenize_prompts, make_padded_batch, chunk_rows_list, _resolve_mts

SCHEDULE = {1: 64, 2: 32, 4: 32, 8: 16, 16: 16, 32: 16}

CONFIGS = [
    ("B=1", 1, dict()),
    ("B=4", 4, dict(ewma_adaptive=True, ewma_min_M=8, ewma_max_ek=8, ewma_min_ek=2)),
    ("B=8", 8, dict(specdecpp_threshold=0.05)),
]


def per_depth_accuracy(dumps):
    depth_alive = Counter()
    depth_acc = Counter()
    for d in dumps:
        r0_leaf = next((l for l in d["leaves"] if all(r == 0 for r in l["ranks"])), None)
        if r0_leaf is None:
            continue
        n_acc = r0_leaf["accept_n"]
        for depth_idx in range(len(r0_leaf["ranks"])):
            depth_alive[depth_idx + 1] += 1
            if depth_idx < n_acc:
                depth_acc[depth_idx + 1] += 1
    return depth_alive, depth_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--output-json", type=str, default="logs/u_shape_universality.json")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.manual_seed(0)
    print("Loading models...", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B", attn_implementation="sdpa", dtype=torch.bfloat16,
    ).to(device).eval()
    draft = DFlashDraftModel.from_pretrained(
        "z-lab/Qwen3-4B-DFlash-b16", attn_implementation="sdpa", dtype=torch.bfloat16,
    ).to(device).eval()
    block_size = draft.block_size
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_ids = [tokenizer.eos_token_id]
    mid = draft.mask_token_id

    ds = load_and_process_dataset("math500").shuffle(seed=0)
    raw = [ds[i]["turns"][0] for i in range(min(len(ds), args.max_samples * 4))]
    prompts_list = tokenize_prompts(
        raw, tokenizer, max_samples=args.max_samples,
        max_prompt_tokens=args.max_prompt_tokens, device=device,
    )

    out = {"configs": []}
    for label, B, mode_kwargs in CONFIGS:
        M = _resolve_mts(B, 16, SCHEDULE)
        dumps = []
        for chunk in chunk_rows_list(prompts_list, B):
            ids, attn = make_padded_batch(chunk, pad_id, device)
            s_out = dflash_generate_batched(
                draft=draft, target=target, input_ids=ids, attention_mask=attn,
                mask_token_id=mid, eos_token_ids=eos_ids,
                max_new_tokens=args.max_new_tokens, block_size=block_size,
                max_tree_size=M, expand_k=8, temperature=0.0,
                log_full_tree_dump=True, **mode_kwargs,
            )
            dumps.extend(s_out.full_tree_dumps)
            del s_out, ids, attn
            torch.cuda.empty_cache()
        alive, acc = per_depth_accuracy(dumps)

        print(f"\n=== {label} M={M} ({len(dumps)} rounds) ===", flush=True)
        print(f"  {'depth':>5}  {'n':>5}  {'P(accept)':>10}", flush=True)
        curve = {}
        for d in sorted(alive.keys(), key=int):
            if alive[d] < 5:
                continue
            p = acc[d] / alive[d]
            curve[d] = p
            print(f"  {d:>5}  {alive[d]:>5}  {p:>9.3f}", flush=True)
        out["configs"].append({
            "label": label, "B": B, "M": M, "n_rounds": len(dumps),
            "depth_alive": dict(alive),
            "depth_accepted": dict(acc),
            "depth_p_accept": curve,
        })

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
