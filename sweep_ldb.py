"""Late-Deviation Bias (LDB) — directly penalize early-deviation children in heap.

Insight: comparative diagnostic showed score_decay shifts deviation depths from
middle (dev@5: 9.7%→4.2%) to late (dev@7: 3.2%→9.2%), increasing full-block rate.
LDB applies this directly via a depth-conditional bonus on the FIRST deviation.

Schedule: dev_depth_bonus[d] = (d - pivot) * lambda
  d < pivot  → negative (penalty for early deviation)
  d > pivot  → positive (bonus for late deviation)

Compared to baseline (no bias) and decay_085 (current best at B=4).
"""
import argparse, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import DFlashDraftModel, load_and_process_dataset
from dflash_batched import dflash_generate_batched, vanilla_ar_generate_batched
from benchmark_batched import tokenize_prompts, make_padded_batch, chunk_rows_list, _resolve_mts

SCHEDULE = {1: 64, 2: 32, 4: 32, 8: 16, 16: 16, 32: 16}

def make_ldb_schedule(pivot=8, lam=0.5, length=17):
    return [(d - pivot) * lam for d in range(length)]

CONFIGS = [
    ("B=1", 1, dict()),
    ("B=4", 4, dict(ewma_adaptive=True, ewma_min_M=8, ewma_max_ek=8, ewma_min_ek=2)),
    ("B=8", 8, dict(specdecpp_threshold=0.05)),
]

VARIANTS = [
    ("baseline",         dict()),
    ("decay_085",        dict(score_decay=0.85)),
    ("ldb_pivot8_l0.5",  dict(dev_depth_bonus=make_ldb_schedule(pivot=8, lam=0.5))),
    ("ldb_pivot10_l0.5", dict(dev_depth_bonus=make_ldb_schedule(pivot=10, lam=0.5))),
    ("ldb_pivot8_l1.0",  dict(dev_depth_bonus=make_ldb_schedule(pivot=8, lam=1.0))),
    ("decay_085+ldb",    dict(score_decay=0.85, dev_depth_bonus=make_ldb_schedule(pivot=8, lam=0.5))),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--output-json", type=str, default="logs/ldb_sweep.json")
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

    warm_ids, warm_attn = make_padded_batch(prompts_list[:1], pad_id, device)
    _ = vanilla_ar_generate_batched(
        target=target, input_ids=warm_ids, attention_mask=warm_attn,
        eos_token_ids=eos_ids, max_new_tokens=8,
    )
    torch.cuda.empty_cache()

    print("\n=== Vanilla AR baseline ===", flush=True)
    van_by_B = {}
    for label, B, _ in CONFIGS:
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
        van_tps = van_o / van_t
        van_by_B[B] = van_tps
        print(f"  {label}: {van_tps:.1f} tok/s", flush=True)

    rows = []
    for label, B, mode_kwargs in CONFIGS:
        M = _resolve_mts(B, 16, SCHEDULE)
        print(f"\n=== {label} M={M} ===", flush=True)
        for variant, var_kwargs in VARIANTS:
            torch.manual_seed(0)
            v7_t, v7_o, accs = 0.0, 0, []
            try:
                for chunk in chunk_rows_list(prompts_list, B):
                    ids, attn = make_padded_batch(chunk, pad_id, device)
                    s_out = dflash_generate_batched(
                        draft=draft, target=target, input_ids=ids, attention_mask=attn,
                        mask_token_id=mid, eos_token_ids=eos_ids,
                        max_new_tokens=args.max_new_tokens, block_size=block_size,
                        max_tree_size=M, expand_k=8, temperature=0.0,
                        **mode_kwargs, **var_kwargs,
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
                       "v7_tps": round(v7_tps, 1),
                       "vanilla_tps": round(van_by_B[B], 1),
                       "speedup": round(speedup, 3), "tau": round(tau, 3)}
                rows.append(row)
                print(f"  {variant:>20}: tau={tau:>5.2f} speedup={speedup:>5.3f}× tps={v7_tps:.0f}",
                      flush=True)
            except Exception as e:
                print(f"  {variant:>20}: CRASHED — {e}", flush=True)
                rows.append({"variant": variant, "B": B, "error": str(e)[:200]})

    print("\n=== speedup matrix ===", flush=True)
    print(f"  {'variant':>20}  " + "  ".join(f"{l:>6}" for l, _, _ in CONFIGS), flush=True)
    by_v = {(r['variant'], r['B']): r['speedup'] for r in rows if 'speedup' in r}
    for v_name, _ in VARIANTS:
        line = f"  {v_name:>20}  " + "  ".join(f"{by_v.get((v_name, B), float('nan')):>6.3f}" for _, B, _ in CONFIGS)
        print(line, flush=True)

    print("\n=== tau matrix ===", flush=True)
    by_v_tau = {(r['variant'], r['B']): r['tau'] for r in rows if 'tau' in r}
    print(f"  {'variant':>20}  " + "  ".join(f"{l:>6}" for l, _, _ in CONFIGS), flush=True)
    for v_name, _ in VARIANTS:
        line = f"  {v_name:>20}  " + "  ".join(f"{by_v_tau.get((v_name, B), float('nan')):>6.2f}" for _, B, _ in CONFIGS)
        print(line, flush=True)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"vanilla_by_B": van_by_B, "results": rows}, f, indent=2)
    print(f"\nWrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
