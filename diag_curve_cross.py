"""Per-depth rank-0 chain P(accept) curve across datasets.

Verifies whether the U-shape per-depth accuracy curve we measured on math500
(0.94→0.61→0.77) is universal or dataset-specific. If different, it explains
why CDDT generalization fails on gsm8k/humaneval.
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
DATASETS = ["math500", "gsm8k", "humaneval"]


def per_depth_p_accept(dumps):
    alive, acc = Counter(), Counter()
    for d in dumps:
        r0_leaf = next((l for l in d["leaves"] if all(r == 0 for r in l["ranks"])), None)
        if r0_leaf is None:
            continue
        n_acc = r0_leaf["accept_n"]
        for d_idx in range(len(r0_leaf["ranks"])):
            alive[d_idx + 1] += 1
            if d_idx < n_acc:
                acc[d_idx + 1] += 1
    return alive, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--output-json", type=str, default="logs/curve_cross.json")
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

    out = {"configs": []}
    for ds_name in DATASETS:
        try:
            ds = load_and_process_dataset(ds_name).shuffle(seed=0)
        except Exception as e:
            print(f"{ds_name} failed: {e}"); continue
        raw = [ds[i]["turns"][0] for i in range(min(len(ds), args.max_samples * 4))]
        prompts_list = tokenize_prompts(
            raw, tokenizer, max_samples=args.max_samples,
            max_prompt_tokens=args.max_prompt_tokens, device=device,
        )
        if not prompts_list: continue

        # Use B=4 baseline
        B = 4
        M = _resolve_mts(B, 16, SCHEDULE)
        dumps = []
        for chunk in chunk_rows_list(prompts_list, B):
            ids, attn = make_padded_batch(chunk, pad_id, device)
            s_out = dflash_generate_batched(
                draft=draft, target=target, input_ids=ids, attention_mask=attn,
                mask_token_id=mid, eos_token_ids=eos_ids,
                max_new_tokens=args.max_new_tokens, block_size=block_size,
                max_tree_size=M, expand_k=8, temperature=0.0,
                log_full_tree_dump=True,
                ewma_adaptive=True, ewma_min_M=8, ewma_max_ek=8, ewma_min_ek=2,
            )
            dumps.extend(s_out.full_tree_dumps)
            del s_out, ids, attn; torch.cuda.empty_cache()
        alive, acc = per_depth_p_accept(dumps)

        print(f"\n=== {ds_name} (B={B} M={M}, {len(dumps)} rounds) ===", flush=True)
        print(f"  {'depth':>5}  {'n':>5}  {'P(accept)':>10}", flush=True)
        curve = {}
        for d in sorted(alive.keys(), key=int):
            if alive[d] < 5: continue
            p = acc[d] / alive[d]
            curve[d] = round(p, 3)
            print(f"  {d:>5}  {alive[d]:>5}  {p:>9.3f}", flush=True)
        out["configs"].append({"dataset": ds_name, "B": B, "M": M, "n_rounds": len(dumps),
                               "depth_p_accept": curve})

    print("\n=== Comparison summary ===", flush=True)
    print(f"  {'depth':>5}  " + "  ".join(f"{c['dataset']:>10}" for c in out["configs"]), flush=True)
    all_depths = sorted({d for c in out["configs"] for d in c["depth_p_accept"].keys()}, key=int)
    for d in all_depths:
        line = f"  {d:>5}  "
        for c in out["configs"]:
            p = c["depth_p_accept"].get(d)
            if p is None:
                line += f"{'--':>10}  "
            else:
                line += f"{p:>10.3f}  "
        print(line, flush=True)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
