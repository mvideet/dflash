"""Final headline benchmark: CDDT γ=0.95 production config across all B and datasets.

Per-B production recipe:
  B=1 : score_decay=0.95
  B=4 : ewma_adaptive + score_decay=0.95
  B=8 : specdecpp + chain_mode (decay no-op at M=block_size)

Tests on math500, gsm8k, humaneval at N=24. Compares to v7 baseline.
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
B_VALUES = [1, 4, 8]


def production_kwargs(B):
    if B == 1:
        return dict(), dict(score_decay=0.95)
    elif B == 4:
        return (dict(ewma_adaptive=True, ewma_min_M=8, ewma_max_ek=8, ewma_min_ek=2),
                dict(ewma_adaptive=True, ewma_min_M=8, ewma_max_ek=8, ewma_min_ek=2,
                     score_decay=0.95))
    elif B == 8:
        return (dict(specdecpp_threshold=0.05),
                dict(specdecpp_threshold=0.05, chain_mode=True))
    return dict(), dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--output-json", type=str, default="logs/final_headline.json")
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

    rows = []
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

        for B in B_VALUES:
            M = _resolve_mts(B, 16, SCHEDULE)
            base_kw, prod_kw = production_kwargs(B)
            van_t, van_o = 0.0, 0
            for chunk in chunk_rows_list(prompts_list, B):
                ids, attn = make_padded_batch(chunk, pad_id, device)
                v_out = vanilla_ar_generate_batched(
                    target=target, input_ids=ids, attention_mask=attn,
                    eos_token_ids=eos_ids, max_new_tokens=args.max_new_tokens,
                )
                van_t += v_out.total_decode_time
                van_o += sum(v_out.num_output_tokens)
                del v_out, ids, attn; torch.cuda.empty_cache()
            van_tps = van_o / van_t

            for variant, kw in [("v7", base_kw), ("cddt_g095", prod_kw)]:
                torch.manual_seed(0)
                v7_t, v7_o, accs = 0.0, 0, []
                for chunk in chunk_rows_list(prompts_list, B):
                    ids, attn = make_padded_batch(chunk, pad_id, device)
                    s_out = dflash_generate_batched(
                        draft=draft, target=target, input_ids=ids, attention_mask=attn,
                        mask_token_id=mid, eos_token_ids=eos_ids,
                        max_new_tokens=args.max_new_tokens, block_size=block_size,
                        max_tree_size=M, expand_k=8, temperature=0.0, **kw,
                    )
                    v7_t += s_out.total_decode_time
                    v7_o += sum(s_out.num_output_tokens)
                    for lst in s_out.acceptance_lengths_per_elem:
                        accs.extend(lst)
                    del s_out, ids, attn; torch.cuda.empty_cache()
                v7_tps = v7_o / v7_t
                tau = float(np.mean(accs)) if accs else float("nan")
                speedup = v7_tps / van_tps
                rows.append({"dataset": ds_name, "variant": variant, "B": B, "M": M,
                             "v7_tps": round(v7_tps, 1), "vanilla_tps": round(van_tps, 1),
                             "speedup": round(speedup, 3), "tau": round(tau, 3)})
                print(f"  B={B:>2} {variant:>10}: tau={tau:>5.2f} speedup={speedup:>5.3f}× "
                      f"tps={v7_tps:.0f}", flush=True)

    print("\n=== FINAL HEADLINE: CDDT γ=0.95 production gain over v7 ===", flush=True)
    print(f"  {'dataset':>11}  " + "  ".join(f"B={B:>2}" for B in B_VALUES), flush=True)
    for ds_name in DATASETS:
        line = f"  {ds_name:>11}  "
        for B in B_VALUES:
            base = next((r for r in rows if r['dataset']==ds_name and r['variant']=='v7' and r['B']==B), None)
            cddt = next((r for r in rows if r['dataset']==ds_name and r['variant']=='cddt_g095' and r['B']==B), None)
            if base and cddt:
                gain = 100 * (cddt['speedup'] - base['speedup']) / base['speedup']
                line += f"{gain:>+5.1f}%  "
            else:
                line += f"{'--':>5}   "
        print(line, flush=True)

    print("\n=== Absolute speedups ===", flush=True)
    print(f"  {'dataset':>11}  " + "  ".join(f"{'B='+str(B):>14}" for B in B_VALUES), flush=True)
    for ds_name in DATASETS:
        line = f"  {ds_name:>11}  "
        for B in B_VALUES:
            base = next((r for r in rows if r['dataset']==ds_name and r['variant']=='v7' and r['B']==B), None)
            cddt = next((r for r in rows if r['dataset']==ds_name and r['variant']=='cddt_g095' and r['B']==B), None)
            if base and cddt:
                line += f"{base['speedup']:>5.2f}→{cddt['speedup']:>5.2f}  "
            else:
                line += f"{'--':>14}"
        print(line, flush=True)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"results": rows}, f, indent=2)
    print(f"\nWrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
