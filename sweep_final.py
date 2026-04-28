"""Final consolidated sweep — all 15 research items vs baseline.

Each row: B=1, 4, 8 with one mode applied. Vanilla AR is the speedup denom.
The goal is to produce a single results table that captures everything we've
tried so the user can read the headline at a glance.

Modes (most are flag-only; if a flag depends on another, we stack):
  baseline             : v7 with current best static M schedule
  online_mts           : Item via _pick_m_online (cost-model M*)
  ewma_adaptive        : repo-style EWMA acceptance → M, expand_k
  anchor_entropy       : forward-look anchor entropy → M
  heap_conc_B          : Item 2
  optree_term          : Item 3 OPT-Tree termination
  eagle2_overbuild     : Item 9 over-build then trim
  specdecpp            : Item 10 rejection-prob threshold
  online_sequoia       : Item 14 fixed-width tree from learned p_d
  roofline_pid         : Item 15 hill-climb on M
"""
import argparse, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import DFlashDraftModel, load_and_process_dataset
from dflash_batched import dflash_generate_batched, vanilla_ar_generate_batched
from benchmark_batched import tokenize_prompts, make_padded_batch, chunk_rows_list, _resolve_mts


SCHEDULE = {1: 64, 2: 32, 4: 32, 8: 16, 16: 16, 32: 16}

MODES = [
    ("baseline",         dict()),
    ("online_mts",       dict(online_mts=True)),
    ("ewma_adaptive",    dict(ewma_adaptive=True, ewma_min_M=8, ewma_max_ek=8, ewma_min_ek=2)),
    ("anchor_entropy",   dict(anchor_signal="entropy", anchor_min_M=8, anchor_gamma=1.0)),
    ("heap_conc_B",      dict(heap_conc_B=True, heap_conc_min_M=8, heap_conc_max_M=64)),
    ("optree_term",      dict(optree_termination=True, optree_threshold=0.05)),
    ("eagle2_overbuild", dict(eagle2_overbuild=True, eagle2_pool_mult=2)),
    ("specdecpp",        dict(specdecpp_threshold=0.05)),
    ("online_sequoia",   dict(online_sequoia=True, sequoia_recompute_every=3)),
    ("roofline_pid",     dict(roofline_pid=True)),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--batch-sizes", type=str, default="1,4,8")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--output-json", type=str, default="logs/final_sweep.json")
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

    for mode_name, mode_kwargs in MODES:
        print(f"\n=== mode: {mode_name} ===", flush=True)
        for B in Bs:
            M = _resolve_mts(B, 16, SCHEDULE)
            v7_t, v7_o, accs = 0.0, 0, []
            try:
                for chunk in chunk_rows_list(prompts_list, B):
                    ids, attn = make_padded_batch(chunk, pad_id, device)
                    s_out = dflash_generate_batched(
                        draft=draft, target=target, input_ids=ids, attention_mask=attn,
                        mask_token_id=mid, eos_token_ids=eos_ids,
                        max_new_tokens=args.max_new_tokens, block_size=block_size,
                        max_tree_size=M, expand_k=8, temperature=0.0,
                        **mode_kwargs,
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
                rows.append({
                    "mode": mode_name, "B": B, "M": M,
                    "v7_tps": round(v7_tps, 1), "vanilla_tps": round(van_by_B[B], 1),
                    "speedup": round(speedup, 3), "tau": round(tau, 3),
                })
                print(f"  B={B:>2} M={M:>3}: {v7_tps:>7.1f} tok/s, tau={tau:.2f}, "
                      f"speedup={speedup:.3f}×", flush=True)
            except Exception as e:
                rows.append({"mode": mode_name, "B": B, "error": str(e)[:200]})
                print(f"  B={B:>2}: CRASHED — {e}", flush=True)

    # Speedup matrix.
    print("\n=== speedup × matrix (rows = mode, cols = B) ===", flush=True)
    print(f"  {'mode':>20} | " + " | ".join(f"B={B:>2}" for B in Bs), flush=True)
    by_mode = {m: {B: float("nan") for B in Bs} for m, _ in MODES}
    for r in rows:
        if "speedup" in r:
            by_mode[r["mode"]][r["B"]] = r["speedup"]
    for mode_name, _ in MODES:
        line = f"  {mode_name:>20} | " + " | ".join(
            f"{by_mode[mode_name][B]:>5.2f}" for B in Bs
        )
        print(line, flush=True)

    print("\n=== tau matrix ===", flush=True)
    by_mode_tau = {m: {B: float("nan") for B in Bs} for m, _ in MODES}
    for r in rows:
        if "tau" in r:
            by_mode_tau[r["mode"]][r["B"]] = r["tau"]
    print(f"  {'mode':>20} | " + " | ".join(f"B={B:>2}" for B in Bs), flush=True)
    for mode_name, _ in MODES:
        line = f"  {mode_name:>20} | " + " | ".join(
            f"{by_mode_tau[mode_name][B]:>5.2f}" for B in Bs
        )
        print(line, flush=True)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"vanilla_by_B": van_by_B, "results": rows}, f, indent=2)
    print(f"\nWrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
