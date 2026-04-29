"""Consolidated CDDT production benchmark.

Final headline experiment after the autonomous loop. Tests whether CDDT
composes with the previously-validated pieces:
  - B=1: v7 baseline (M=64) vs CDDT γ=0.95
  - B=2: v7 baseline (M=32) vs CDDT γ=0.88
  - B=4: ewma_adaptive (M=32) vs ewma_adaptive + CDDT γ=0.85
  - B=8: specdecpp (M=16) vs specdecpp + chain_mode

Larger N (24 prompts) and 192 max_new_tokens to denoise.
"""
import argparse, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import DFlashDraftModel, load_and_process_dataset
from dflash_batched import dflash_generate_batched, vanilla_ar_generate_batched
from benchmark_batched import tokenize_prompts, make_padded_batch, chunk_rows_list, _resolve_mts

SCHEDULE = {1: 64, 2: 32, 4: 32, 8: 16, 16: 16, 32: 16}

# Paired comparisons: (label, B, baseline_kw, cddt_kw)
COMPARISONS = [
    ("B=1",  1, dict(),
                dict(score_decay=0.95)),
    ("B=2",  2, dict(ewma_adaptive=True, ewma_min_M=8, ewma_max_ek=8, ewma_min_ek=2),
                dict(ewma_adaptive=True, ewma_min_M=8, ewma_max_ek=8, ewma_min_ek=2, score_decay=0.88)),
    ("B=4",  4, dict(ewma_adaptive=True, ewma_min_M=8, ewma_max_ek=8, ewma_min_ek=2),
                dict(ewma_adaptive=True, ewma_min_M=8, ewma_max_ek=8, ewma_min_ek=2, score_decay=0.85)),
    ("B=8",  8, dict(specdecpp_threshold=0.05),
                dict(specdecpp_threshold=0.05, chain_mode=True)),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--output-json", type=str, default="logs/cddt_production.json")
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
    print(f"Tokenized {len(prompts_list)} prompts.", flush=True)

    warm_ids, warm_attn = make_padded_batch(prompts_list[:1], pad_id, device)
    _ = vanilla_ar_generate_batched(
        target=target, input_ids=warm_ids, attention_mask=warm_attn,
        eos_token_ids=eos_ids, max_new_tokens=8,
    )
    torch.cuda.empty_cache()

    print("\n=== Vanilla AR baseline ===", flush=True)
    van_by_B = {}
    for label, B, _, _ in COMPARISONS:
        if B in van_by_B:
            continue
        van_t, van_o = 0.0, 0
        for chunk in chunk_rows_list(prompts_list, B):
            ids, attn = make_padded_batch(chunk, pad_id, device)
            v_out = vanilla_ar_generate_batched(
                target=target, input_ids=ids, attention_mask=attn,
                eos_token_ids=eos_ids, max_new_tokens=args.max_new_tokens,
            )
            van_t += v_out.total_decode_time
            van_o += sum(v_out.num_output_tokens)
            del v_out, ids, attn
            torch.cuda.empty_cache()
        van_by_B[B] = van_o / van_t
        print(f"  B={B}: {van_by_B[B]:.1f} tok/s", flush=True)

    rows = []
    for label, B, base_kw, cddt_kw in COMPARISONS:
        M = _resolve_mts(B, 16, SCHEDULE)
        print(f"\n=== {label} M={M} ===", flush=True)
        for variant, var_kwargs in [("v7_baseline", base_kw), ("cddt_prod", cddt_kw)]:
            torch.manual_seed(0)
            v7_t, v7_o, accs = 0.0, 0, []
            for chunk in chunk_rows_list(prompts_list, B):
                ids, attn = make_padded_batch(chunk, pad_id, device)
                s_out = dflash_generate_batched(
                    draft=draft, target=target, input_ids=ids, attention_mask=attn,
                    mask_token_id=mid, eos_token_ids=eos_ids,
                    max_new_tokens=args.max_new_tokens, block_size=block_size,
                    max_tree_size=M, expand_k=8, temperature=0.0,
                    **var_kwargs,
                )
                v7_t += s_out.total_decode_time
                v7_o += sum(s_out.num_output_tokens)
                for lst in s_out.acceptance_lengths_per_elem:
                    accs.extend(lst)
                del s_out, ids, attn
                torch.cuda.empty_cache()
            v7_tps = v7_o / v7_t
            tau = float(np.mean(accs)) if accs else float("nan")
            speedup = v7_tps / van_by_B[B]
            row = {"variant": variant, "B": B, "M": M,
                   "v7_tps": round(v7_tps, 1), "vanilla_tps": round(van_by_B[B], 1),
                   "speedup": round(speedup, 3), "tau": round(tau, 3)}
            rows.append(row)
            print(f"  {variant:>13}: tau={tau:>5.2f} speedup={speedup:>5.3f}× tps={v7_tps:.0f}",
                  flush=True)

    print("\n=== Headline summary ===", flush=True)
    print(f"  {'B':>3} {'M':>3} {'baseline':>10} {'cddt':>10} {'gain':>8}", flush=True)
    for label, B, _, _ in COMPARISONS:
        b_row = next((r for r in rows if r['B'] == B and r['variant'] == 'v7_baseline'), None)
        c_row = next((r for r in rows if r['B'] == B and r['variant'] == 'cddt_prod'), None)
        if b_row and c_row:
            gain_pct = 100 * (c_row['speedup'] - b_row['speedup']) / b_row['speedup']
            print(f"  {B:>3} {b_row['M']:>3} {b_row['speedup']:>9.3f}× {c_row['speedup']:>9.3f}× "
                  f"{gain_pct:>+6.1f}%", flush=True)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"vanilla_by_B": van_by_B, "results": rows}, f, indent=2)
    print(f"\nWrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
