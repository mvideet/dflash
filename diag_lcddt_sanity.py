"""Sanity check: verify L-CDDT learns the same per-depth curve we measured offline.

For each dataset, run L-CDDT with power=1.0 (raw curve) and dump the final
learned curve. Compare to:
  - the offline-measured curve from diag_curve_cross.py / curve_cross.json
  - basic shape sanity (math500 U-shape, gsm8k monotone, humaneval weak U)

Reports pass/fail per dataset.
"""
import argparse, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import DFlashDraftModel, load_and_process_dataset
from dflash_batched import dflash_generate_batched, vanilla_ar_generate_batched
from benchmark_batched import tokenize_prompts, make_padded_batch, chunk_rows_list, _resolve_mts

SCHEDULE = {1: 64, 2: 32, 4: 32, 8: 16, 16: 16, 32: 16}
DATASETS = ["math500", "gsm8k", "humaneval"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--output-json", type=str, default="logs/lcddt_sanity.json")
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

    # Load offline-measured curves for comparison.
    try:
        offline = json.load(open("logs/curve_cross.json"))
        offline_curves = {c["dataset"]: c["depth_p_accept"] for c in offline["configs"]}
    except Exception:
        offline_curves = {}

    out = {"configs": []}
    for ds_name in DATASETS:
        print(f"\n========== Dataset: {ds_name} ==========", flush=True)
        try:
            ds = load_and_process_dataset(ds_name).shuffle(seed=0)
        except Exception as e:
            print(f"  Failed: {e}"); continue
        raw = [ds[i]["turns"][0] for i in range(min(len(ds), args.max_samples * 4))]
        prompts_list = tokenize_prompts(
            raw, tokenizer, max_samples=args.max_samples,
            max_prompt_tokens=args.max_prompt_tokens, device=device,
        )
        if not prompts_list: continue

        warm_ids, warm_attn = make_padded_batch(prompts_list[:1], pad_id, device)
        _ = vanilla_ar_generate_batched(
            target=target, input_ids=warm_ids, attention_mask=warm_attn,
            eos_token_ids=eos_ids, max_new_tokens=8,
        )
        torch.cuda.empty_cache()

        B = 4
        M = _resolve_mts(B, 16, SCHEDULE)
        # Run L-CDDT power=1.0 (raw curve, no power amplification — should match offline directly)
        # Use a single chunk so curve persists across batches. Don't reset state.
        torch.manual_seed(0)
        learned_curves_per_chunk = []
        steps_per_chunk = []
        for chunk in chunk_rows_list(prompts_list, B):
            ids, attn = make_padded_batch(chunk, pad_id, device)
            s_out = dflash_generate_batched(
                draft=draft, target=target, input_ids=ids, attention_mask=attn,
                mask_token_id=mid, eos_token_ids=eos_ids,
                max_new_tokens=args.max_new_tokens, block_size=block_size,
                max_tree_size=M, expand_k=8, temperature=0.0,
                ewma_adaptive=True, ewma_min_M=8, ewma_max_ek=8, ewma_min_ek=2,
                learned_curve_decay=True, learned_curve_power=1.0,
                learned_curve_warmup=5, learned_curve_alpha=0.9,
            )
            learned_curves_per_chunk.append(list(s_out.lcddt_final_curve))
            steps_per_chunk.append(s_out.lcddt_step_count)
            del s_out, ids, attn; torch.cuda.empty_cache()

        # Final learned curve = average over chunks (each is independent because state resets per call)
        # Most informative: report all chunks plus mean.
        last_curve = learned_curves_per_chunk[-1] if learned_curves_per_chunk else []
        offline = offline_curves.get(ds_name, {})

        print(f"  steps_per_chunk={steps_per_chunk}", flush=True)
        print(f"  {'depth':>5}  {'learned':>10}  {'offline':>10}  {'diff':>10}", flush=True)
        ok = True
        for d in range(1, block_size):
            learned = last_curve[d] if d < len(last_curve) else None
            off = offline.get(str(d))
            if learned is None or off is None:
                print(f"  {d:>5}  {'--':>10}  {off if off is not None else '--':>10}  {'--':>10}", flush=True)
                continue
            diff = abs(learned - off)
            flag = "✓" if diff < 0.15 else "✗"
            if diff >= 0.15:
                ok = False
            print(f"  {d:>5}  {learned:>10.3f}  {off:>10.3f}  {diff:>+10.3f}  {flag}", flush=True)

        # Shape sanity: U-shape iff value at depth 12-15 > value at depth 5-8 by some margin
        if last_curve:
            mid_avg = np.mean([last_curve[d] for d in range(5, 9) if d < len(last_curve)])
            deep_avg = np.mean([last_curve[d] for d in range(12, 16) if d < len(last_curve)])
            shape = "U-shape" if deep_avg > mid_avg + 0.05 else (
                "monotone-decay" if last_curve[1] > deep_avg + 0.2 else "flat/other")
            print(f"  → mid({mid_avg:.3f}) deep({deep_avg:.3f}) → shape: {shape}", flush=True)
        else:
            shape = "no data"

        # Cross-check against expected
        expected = {
            "math500": "U-shape",
            "gsm8k": "monotone-decay",
            "humaneval": "weak U-shape (any)",
        }
        print(f"  Expected shape: {expected.get(ds_name, '?')}", flush=True)

        out["configs"].append({
            "dataset": ds_name, "B": B, "M": M,
            "learned_curve_last_chunk": last_curve,
            "all_chunk_curves": learned_curves_per_chunk,
            "steps_per_chunk": steps_per_chunk,
            "offline_curve": offline,
            "shape_observed": shape,
            "shape_expected": expected.get(ds_name, "?"),
            "all_within_0.15_of_offline": ok,
        })

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
