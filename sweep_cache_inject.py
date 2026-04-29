"""Rejection-derived cache injection sweep.

Compares baseline (no cache) vs cache_inject with cache_slots ∈ {4, 8} at the
current best per-B configs. Reports:
  - tau split between cache-hit rounds and no-hit rounds (proxy: mean tau on
    rounds where cache_round_had_hit fired vs not)
  - per-slot acceptance rate by depth-distance from previous failure
  - end-to-end speedup vs vanilla AR

The user's expected gain estimate: +0.5 to +1.0 τ if cache survival ~30% per
slot, decaying with distance. If per-slot rate <25% the experiment is negative.
"""
import argparse, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import DFlashDraftModel, load_and_process_dataset
from dflash_batched import dflash_generate_batched, vanilla_ar_generate_batched
from benchmark_batched import tokenize_prompts, make_padded_batch, chunk_rows_list, _resolve_mts


SCHEDULE = {1: 64, 2: 32, 4: 32, 8: 16, 16: 16, 32: 16}

# Best-per-B current configs.
BEST_KW = {
    1: dict(),                                                              # baseline
    4: dict(ewma_adaptive=True, ewma_min_M=8, ewma_max_ek=8, ewma_min_ek=2),
    8: dict(specdecpp_threshold=0.05),
}

CACHE_VARIANTS = [
    ("baseline",      dict(cache_inject=False)),
    ("cache_4",       dict(cache_inject=True, cache_slots=4)),
    ("cache_8",       dict(cache_inject=True, cache_slots=8)),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--batch-sizes", type=str, default="1,4,8")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--output-json", type=str, default="logs/cache_inject.json")
    args = parser.parse_args()

    Bs = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
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

    rows = []
    print("\n=== Vanilla AR baseline ===", flush=True)
    van_by_B = {}
    for B in Bs:
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
        print(f"  B={B:>2}: {van_tps:.1f} tok/s", flush=True)

    for B in Bs:
        best_kw = BEST_KW.get(B, dict())
        M = _resolve_mts(B, 16, SCHEDULE)
        print(f"\n=== B={B}, best_kw={best_kw} ===", flush=True)
        for variant, cache_kw in CACHE_VARIANTS:
            v7_t, v7_o, accs = 0.0, 0, []
            hits_per_slot = [0] * 16
            attempts_per_slot = [0] * 16
            tot_round, hit_round = 0, 0
            for chunk in chunk_rows_list(prompts_list, B):
                ids, attn = make_padded_batch(chunk, pad_id, device)
                kw = {**best_kw, **cache_kw}
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
                # Aggregate cache stats.
                if hasattr(s_out, "cache_hits_per_slot"):
                    for c in range(min(len(s_out.cache_hits_per_slot), len(hits_per_slot))):
                        hits_per_slot[c] += s_out.cache_hits_per_slot[c]
                        attempts_per_slot[c] += s_out.cache_attempts_per_slot[c]
                    tot_round += s_out.cache_round_total
                    hit_round += s_out.cache_round_had_hit
                del s_out, ids, attn
                torch.cuda.empty_cache()
            v7_tps = v7_o / v7_t
            tau = float(np.mean(accs)) if accs else float("nan")
            speedup = v7_tps / van_by_B[B]
            slot_acc_rate = [
                round(hits_per_slot[c] / max(attempts_per_slot[c], 1) * 100, 1)
                for c in range(min(8, len(hits_per_slot)))
            ]
            row = {
                "variant": variant, "B": B, "M": M,
                "v7_tps": round(v7_tps, 1), "vanilla_tps": round(van_by_B[B], 1),
                "speedup": round(speedup, 3), "tau": round(tau, 3),
                "cache_round_total": tot_round,
                "cache_round_had_hit": hit_round,
                "cache_round_hit_rate_pct": (
                    round(100 * hit_round / tot_round, 1) if tot_round else 0.0
                ),
                "per_slot_accept_pct": slot_acc_rate,
                "per_slot_attempts": attempts_per_slot[:8],
                "per_slot_hits": hits_per_slot[:8],
            }
            rows.append(row)
            print(f"  {variant:>10}: {v7_tps:>7.1f} tok/s, tau={tau:.2f}, "
                  f"speedup={speedup:.3f}×", flush=True)
            if tot_round:
                print(f"    cache fired in {hit_round}/{tot_round} rounds "
                      f"({row['cache_round_hit_rate_pct']:.1f}%)", flush=True)
                print(f"    per-slot accept rate: {slot_acc_rate}", flush=True)

    print("\n=== summary ===", flush=True)
    print(f"  {'variant':>10} {'B':>3} {'M':>3} {'tau':>6} {'speedup':>8} "
          f"{'hit%':>6} slot_accept_rate", flush=True)
    for r in rows:
        print(f"  {r['variant']:>10} {r['B']:>3} {r['M']:>3} {r['tau']:>6.2f} "
              f"{r['speedup']:>7.3f}× {r.get('cache_round_hit_rate_pct',0):>5.1f}% "
              f"{r.get('per_slot_accept_pct',[])}", flush=True)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"vanilla_by_B": van_by_B, "results": rows}, f, indent=2)
    print(f"\nWrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
