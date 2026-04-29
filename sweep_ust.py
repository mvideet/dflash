"""U-Shape Tree (UST) — first novel algorithm test.

Insight: P(target.argmax = draft top-1) along rank-0 chain is U-shaped:
  depth 1   :  0.94
  depth 4-8 :  0.60-0.69 (DIP — drafter most uncertain mid-block)
  depth 12-15: 0.68-0.77 (rises — survivor bias on harder prompts)

UST encodes this via:
  C1: per-depth heap-score weight w(d). Mid-depths get LOW weight (so the
      heap doesn't over-penalize paths that pass through low-q middle tokens),
      shallow/deep depths get HIGH weight (heap trusts these signals).
  C2: per-depth expand_k K(d). Mid-depths get HIGH K (explore more siblings
      where drafter is uncertain), shallow/deep depths get LOW K (trust top-1).

Schedules tested (each [d=0..15]):
  baseline:     w = [1.0]*16,                K=8 everywhere
  decay_085:    w = 0.85^d (current best),    K=8 everywhere
  ust_w_only:   w = U-shape (high@edges,low@mid), K=8
  ust_K_only:   w = 1.0,                       K=mid-bias
  ust_full:     w = U-shape,                   K=mid-bias
"""
import argparse, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import DFlashDraftModel, load_and_process_dataset
from dflash_batched import dflash_generate_batched, vanilla_ar_generate_batched
from benchmark_batched import tokenize_prompts, make_padded_batch, chunk_rows_list, _resolve_mts

SCHEDULE = {1: 64, 2: 32, 4: 32, 8: 16, 16: 16, 32: 16}

# Per-depth weights for UST. Indexed [0, 1, 2, ..., 15] for tree depths.
# Inverse of (1 - p(d)): depths where draft is reliable get HIGH weight (heap trusts their log_q),
# depths where draft is uncertain get LOW weight (heap doesn't over-penalize low q tokens there).
# From measured U-shape: d=1: 0.94, d=5: 0.60, d=15: 0.77.
UST_WEIGHTS = [
    1.0,  # 0 (anchor, unused)
    1.0,  # 1: 0.94 reliable
    0.95, # 2
    0.85, # 3
    0.6,  # 4: dipping
    0.5,  # 5: bottom of dip
    0.5,  # 6
    0.55, # 7
    0.6,  # 8
    0.65, # 9
    0.7,  # 10
    0.75, # 11
    0.8,  # 12: recovering
    0.85, # 13
    0.9,  # 14
    1.0,  # 15: 0.77 reliable (survivor)
]

# Per-position K. Index d (masked position d, predicting depth d+1) → K cap.
# At the heap, when expanding a node at tree-depth d, we look at top-K children
# at masked position d. So index 0 of expand_k_per_depth controls children at depth 1, etc.
UST_K_MID_BIAS = [
    4,    # d=0: predict depth 1. Reliable (94%). Narrow K.
    4,    # d=1: predict depth 2.
    8,    # d=2
    12,   # d=3: predict depth 4 — entering dip. Wide K.
    16,   # d=4: dip
    16,   # d=5: dip
    16,   # d=6: dip
    16,   # d=7: dip
    12,   # d=8
    8,    # d=9: recovering
    8,    # d=10
    8,    # d=11
    4,    # d=12: reliable
    4,    # d=13
    4,    # d=14
    # (d=15 doesn't predict — block ends)
]

CONFIGS = [
    ("B=1", 1, dict()),
    ("B=4", 4, dict(ewma_adaptive=True, ewma_min_M=8, ewma_max_ek=8, ewma_min_ek=2)),
    ("B=8", 8, dict(specdecpp_threshold=0.05)),
]

VARIANTS = [
    ("baseline",   dict()),
    ("decay_085",  dict(score_decay=0.85)),
    ("ust_w",      dict(score_weights_per_depth=UST_WEIGHTS)),
    ("ust_K",      dict(expand_k_per_depth=UST_K_MID_BIAS)),
    ("ust_full",   dict(score_weights_per_depth=UST_WEIGHTS, expand_k_per_depth=UST_K_MID_BIAS)),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--output-json", type=str, default="logs/ust_sweep.json")
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
                row = {"variant": variant, "B": B, "M": M, "v7_tps": round(v7_tps, 1),
                       "vanilla_tps": round(van_by_B[B], 1),
                       "speedup": round(speedup, 3), "tau": round(tau, 3)}
                rows.append(row)
                print(f"  {variant:>12}: tau={tau:>5.2f} speedup={speedup:>5.3f}× tps={v7_tps:.0f}",
                      flush=True)
            except Exception as e:
                print(f"  {variant:>12}: CRASHED — {e}", flush=True)
                rows.append({"variant": variant, "B": B, "error": str(e)[:200]})

    print("\n=== speedup matrix ===", flush=True)
    print(f"  {'variant':>12}  " + "  ".join(f"{l:>6}" for l, _, _ in CONFIGS), flush=True)
    by_v = {(r['variant'], r['B']): r['speedup'] for r in rows if 'speedup' in r}
    for v_name, _ in VARIANTS:
        line = f"  {v_name:>12}  " + "  ".join(f"{by_v.get((v_name, B), float('nan')):>6.3f}" for _, B, _ in CONFIGS)
        print(line, flush=True)

    print("\n=== tau matrix ===", flush=True)
    by_v_tau = {(r['variant'], r['B']): r['tau'] for r in rows if 'tau' in r}
    print(f"  {'variant':>12}  " + "  ".join(f"{l:>6}" for l, _, _ in CONFIGS), flush=True)
    for v_name, _ in VARIANTS:
        line = f"  {v_name:>12}  " + "  ".join(f"{by_v_tau.get((v_name, B), float('nan')):>6.2f}" for _, B, _ in CONFIGS)
        print(line, flush=True)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"vanilla_by_B": van_by_B, "results": rows,
                   "ust_weights": UST_WEIGHTS, "ust_K": UST_K_MID_BIAS}, f, indent=2)
    print(f"\nWrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
