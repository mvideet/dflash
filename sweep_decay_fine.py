"""Verify the B=4 decay=0.85 sweet-spot. Finer grid + larger N to denoise."""
import argparse, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import DFlashDraftModel, load_and_process_dataset
from dflash_batched import dflash_generate_batched, vanilla_ar_generate_batched
from benchmark_batched import tokenize_prompts, make_padded_batch, chunk_rows_list, _resolve_mts

SCHEDULE = {1: 64, 2: 32, 4: 32, 8: 16, 16: 16, 32: 16}
DECAYS = [1.0, 0.92, 0.88, 0.85, 0.82, 0.78]

CONFIGS = [
    ("B=2",      2, dict(ewma_adaptive=True, ewma_min_M=8, ewma_max_ek=8, ewma_min_ek=2)),
    ("B=4",      4, dict(ewma_adaptive=True, ewma_min_M=8, ewma_max_ek=8, ewma_min_ek=2)),
    ("B=4 nokw", 4, dict()),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--output-json", type=str, default="logs/score_decay_fine.json")
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
        print(f"  B={B}: {van_tps:.1f} tok/s", flush=True)

    rows = []
    for label, B, mode_kwargs in CONFIGS:
        M = _resolve_mts(B, 16, SCHEDULE)
        print(f"\n=== {label} M={M} mode={mode_kwargs} ===", flush=True)
        for decay in DECAYS:
            v7_t, v7_o, accs = 0.0, 0, []
            all_ranks = []
            for chunk in chunk_rows_list(prompts_list, B):
                ids, attn = make_padded_batch(chunk, pad_id, device)
                s_out = dflash_generate_batched(
                    draft=draft, target=target, input_ids=ids, attention_mask=attn,
                    mask_token_id=mid, eos_token_ids=eos_ids,
                    max_new_tokens=args.max_new_tokens, block_size=block_size,
                    max_tree_size=M, expand_k=8, temperature=0.0,
                    log_rejection_ranks=True,
                    score_decay=decay,
                    **mode_kwargs,
                )
                v7_t += s_out.total_decode_time
                v7_o += sum(s_out.num_output_tokens)
                for lst in s_out.acceptance_lengths_per_elem:
                    accs.extend(lst)
                all_ranks.extend(s_out.rejection_ranks)
                del s_out, ids, attn
                torch.cuda.empty_cache()
            v7_tps = v7_o / v7_t
            tau = float(np.mean(accs)) if accs else float("nan")
            speedup = v7_tps / van_by_B[B]
            ranks = [r for _, r in all_ranks]
            n_rej = max(len(ranks), 1)
            rank0_pct = 100 * sum(1 for r in ranks if r == 0) / n_rej
            row = {
                "label": label, "decay": decay, "B": B, "M": M,
                "v7_tps": round(v7_tps, 1), "vanilla_tps": round(van_by_B[B], 1),
                "speedup": round(speedup, 3), "tau": round(tau, 3),
                "n_rejections": len(ranks), "rank0_pct": round(rank0_pct, 1),
            }
            rows.append(row)
            print(f"  decay={decay:.2f}: tau={tau:>5.2f} speedup={speedup:>5.3f}× "
                  f"rank0%={rank0_pct:>4.1f}", flush=True)

    print("\n=== Speedup matrix ===", flush=True)
    by_d = {(r['label'], r['decay']): r['speedup'] for r in rows if 'speedup' in r}
    print(f"  {'decay':>6}  " + "  ".join(f"{l:>9}" for l, _, _ in CONFIGS), flush=True)
    for d in DECAYS:
        line = f"  {d:>6.2f}  " + "  ".join(f"{by_d.get((l, d), float('nan')):>9.3f}" for l, _, _ in CONFIGS)
        print(line, flush=True)

    print("\n=== Tau matrix ===", flush=True)
    by_d_tau = {(r['label'], r['decay']): r['tau'] for r in rows if 'tau' in r}
    print(f"  {'decay':>6}  " + "  ".join(f"{l:>9}" for l, _, _ in CONFIGS), flush=True)
    for d in DECAYS:
        line = f"  {d:>6.2f}  " + "  ".join(f"{by_d_tau.get((l, d), float('nan')):>9.2f}" for l, _, _ in CONFIGS)
        print(line, flush=True)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"vanilla_by_B": van_by_B, "results": rows}, f, indent=2)
    print(f"\nWrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
