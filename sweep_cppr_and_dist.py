"""Re-test CPPR vs current baseline + dump τ histogram at best config.

(1) CPPR re-test — current baseline contains the RoPE fix and other engineering
    work that the original CPPR sweep didn't have. Test at B={1,4,8} with
    cppr_lambda ∈ {0.5, 1.0}.

(2) τ histogram — for the *best per-B mode* (specdecpp at B=8, ewma at B=4,
    baseline at B=1), dump every round's accept-length and bin into
    {0, 1-3, 4-8, 9-15, 16}.
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

# (1) CPPR sweep modes.
MODES = [
    ("baseline",       dict()),
    ("cppr_l05",       dict(cppr=True, cppr_lambda=0.5)),
    ("cppr_l10",       dict(cppr=True, cppr_lambda=1.0)),
]

# (2) τ-distribution best-per-B configs.
DIST_CONFIGS = [
    ("B=1 baseline",    1, dict()),
    ("B=4 ewma",        4, dict(ewma_adaptive=True, ewma_min_M=8, ewma_max_ek=8, ewma_min_ek=2)),
    ("B=8 specdecpp",   8, dict(specdecpp_threshold=0.05)),
]


def histo(accs, block_size=16):
    """Per-value histogram of accept lengths in {0..block_size-1}."""
    bins = {str(n): 0 for n in range(block_size)}
    for n in accs:
        n = max(0, min(int(n), block_size - 1))
        bins[str(n)] += 1
    return bins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--output-json", type=str, default="logs/cppr_and_dist.json")
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

    out = {"cppr_sweep": [], "tau_distribution": []}

    # CPPR sweep skipped — already in logs/cppr_and_dist.json. Reuse those numbers.

    # ------------- (2) τ histograms -------------
    print("\n=== (2) Per-round τ histograms (best per-B configs) ===", flush=True)
    for label, B, mode_kwargs in DIST_CONFIGS:
        M = _resolve_mts(B, 16, SCHEDULE)
        all_accs = []
        for chunk in chunk_rows_list(prompts_list, B):
            ids, attn = make_padded_batch(chunk, pad_id, device)
            s_out = dflash_generate_batched(
                draft=draft, target=target, input_ids=ids, attention_mask=attn,
                mask_token_id=mid, eos_token_ids=eos_ids,
                max_new_tokens=args.max_new_tokens, block_size=block_size,
                max_tree_size=M, expand_k=8, temperature=0.0,
                **mode_kwargs,
            )
            for lst in s_out.acceptance_lengths_per_elem:
                # acceptance_lengths_per_elem holds n+1 (including bonus). Subtract 1
                # to get "real" accept-length n in {0..block_size}.
                for n_plus_1 in lst:
                    all_accs.append(max(0, n_plus_1 - 1))
            del s_out, ids, attn
            torch.cuda.empty_cache()
        h = histo(all_accs, block_size=block_size)
        total = max(sum(h.values()), 1)
        pct = {k: round(100 * v / total, 1) for k, v in h.items()}
        mean_tau = float(np.mean(all_accs)) if all_accs else float("nan")
        median_tau = float(np.median(all_accs)) if all_accs else float("nan")
        p10 = float(np.percentile(all_accs, 10)) if all_accs else float("nan")
        p90 = float(np.percentile(all_accs, 90)) if all_accs else float("nan")
        row = {
            "label": label, "B": B, "M": M, "n_rounds": len(all_accs),
            "mean_tau": round(mean_tau, 2), "median_tau": median_tau,
            "p10": p10, "p90": p90,
            "histogram_count": h, "histogram_pct": pct,
        }
        out["tau_distribution"].append(row)
        print(f"\n  {label} (n={len(all_accs)} rounds): "
              f"mean τ={mean_tau:.2f}, median={median_tau:.0f}, "
              f"p10={p10:.0f}, p90={p90:.0f}", flush=True)
        for k, v in h.items():
            bar = "#" * int(round(60 * v / total))
            print(f"    n={k:>2} | {pct[k]:>5.1f}% ({v:>4}) {bar}", flush=True)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
