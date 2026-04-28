"""5-curve comparison: vanilla AR baseline + v7 (fixed/adaptive M) + v8 (fixed/adaptive M).

Single in-process sweep — loads model once.
"""
import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import DFlashDraftModel, load_and_process_dataset
from dflash_batched import dflash_generate_batched, vanilla_ar_generate_batched
from benchmark_batched import tokenize_prompts, make_padded_batch, chunk_rows_list, _resolve_mts


SCHEDULE = {1: 64, 2: 32, 4: 32, 8: 16, 16: 16, 32: 16}

V8_KW = dict(
    pdrr_k1=0.5, pdrr_k2=0.25, pdrr_k3=0.0,
    cgdb_shallow_depth=4, cgdb_high_thresh=0.1,
    cgdb_low_thresh=0.01, cgdb_mid_k=4,
)

# v8 + EWMA-adaptive M & expand_k.
V8_EWMA_KW = dict(
    **V8_KW,
    ewma_adaptive=True, ewma_decay=0.8,
    ewma_min_M=12, ewma_min_ek=2, ewma_max_ek=8,
)

# v8 + anchor-entropy adaptive M.
V8_ANCHOR_KW = dict(
    **V8_KW,
    anchor_signal="entropy", anchor_min_M=16, anchor_gamma=1.0,
)


def run_v7(target, draft, prompts_list, B, M, *, mn, bs, mid, pid, eos, dev, **extra):
    v7_t, v7_o, accs = 0.0, 0, []
    for chunk in chunk_rows_list(prompts_list, B):
        ids, attn = make_padded_batch(chunk, pid, dev)
        s_out = dflash_generate_batched(
            draft=draft, target=target, input_ids=ids, attention_mask=attn,
            mask_token_id=mid, eos_token_ids=eos,
            max_new_tokens=mn, block_size=bs, max_tree_size=M,
            expand_k=8, temperature=0.0, **extra,
        )
        v7_t += s_out.total_decode_time
        v7_o += sum(s_out.num_output_tokens)
        for lst in s_out.acceptance_lengths_per_elem:
            accs.extend(lst)
        del s_out, ids, attn
        torch.cuda.empty_cache()
    return v7_o / v7_t, float(np.mean(accs)) if accs else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-prompt-tokens", type=int, default=200)
    parser.add_argument("--output-json", type=str, default="logs/v7_vs_v8.json")
    parser.add_argument("--output-png", type=str, default="paper/fig/v7_vs_v8.png")
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

    # Warmup.
    warm_ids, warm_attn = make_padded_batch(prompts_list[:1], pad_id, device)
    _ = vanilla_ar_generate_batched(
        target=target, input_ids=warm_ids, attention_mask=warm_attn,
        eos_token_ids=eos_ids, max_new_tokens=8,
    )
    torch.cuda.empty_cache()

    rows = []
    for B in Bs:
        print(f"\n=== B={B} ===", flush=True)
        # Vanilla AR baseline.
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
        print(f"  vanilla AR     : {van_tps:7.1f} tok/s", flush=True)

        # v7 fixed M=128.
        tps, tau = run_v7(target, draft, prompts_list, B, 128,
                          mn=args.max_new_tokens, bs=block_size,
                          mid=mid, pid=pad_id, eos=eos_ids, dev=device)
        rows.append({"variant": "v7_fixed",  "B": B, "M": 128, "tau": round(tau, 2),
                     "v7_tps": round(tps, 1), "speedup": round(tps / van_tps, 3),
                     "vanilla_tps": round(van_tps, 1)})
        print(f"  v7 fixed M=128 : {tps:7.1f} tok/s, tau={tau:.2f}, speedup={tps/van_tps:.2f}×", flush=True)

        # v7 adaptive M.
        M_adapt = _resolve_mts(B, 16, SCHEDULE)
        tps, tau = run_v7(target, draft, prompts_list, B, M_adapt,
                          mn=args.max_new_tokens, bs=block_size,
                          mid=mid, pid=pad_id, eos=eos_ids, dev=device)
        rows.append({"variant": "v7_adapt",  "B": B, "M": M_adapt, "tau": round(tau, 2),
                     "v7_tps": round(tps, 1), "speedup": round(tps / van_tps, 3),
                     "vanilla_tps": round(van_tps, 1)})
        print(f"  v7 adaptive M={M_adapt:>3}: {tps:7.1f} tok/s, tau={tau:.2f}, speedup={tps/van_tps:.2f}×", flush=True)

        # v8 fixed M=128.
        tps, tau = run_v7(target, draft, prompts_list, B, 128,
                          mn=args.max_new_tokens, bs=block_size,
                          mid=mid, pid=pad_id, eos=eos_ids, dev=device, **V8_KW)
        rows.append({"variant": "v8_fixed",  "B": B, "M": 128, "tau": round(tau, 2),
                     "v7_tps": round(tps, 1), "speedup": round(tps / van_tps, 3),
                     "vanilla_tps": round(van_tps, 1)})
        print(f"  v8 fixed M=128 : {tps:7.1f} tok/s, tau={tau:.2f}, speedup={tps/van_tps:.2f}×", flush=True)

        # v8 adaptive M.
        tps, tau = run_v7(target, draft, prompts_list, B, M_adapt,
                          mn=args.max_new_tokens, bs=block_size,
                          mid=mid, pid=pad_id, eos=eos_ids, dev=device, **V8_KW)
        rows.append({"variant": "v8_adapt",  "B": B, "M": M_adapt, "tau": round(tau, 2),
                     "v7_tps": round(tps, 1), "speedup": round(tps / van_tps, 3),
                     "vanilla_tps": round(van_tps, 1)})
        print(f"  v8 adaptive M={M_adapt:>3}: {tps:7.1f} tok/s, tau={tau:.2f}, speedup={tps/van_tps:.2f}×", flush=True)

        # v8 + EWMA adaptive (max budget = 128).
        tps, tau = run_v7(target, draft, prompts_list, B, 128,
                          mn=args.max_new_tokens, bs=block_size,
                          mid=mid, pid=pad_id, eos=eos_ids, dev=device, **V8_EWMA_KW)
        rows.append({"variant": "v8_ewma",  "B": B, "M": "ewma", "tau": round(tau, 2),
                     "v7_tps": round(tps, 1), "speedup": round(tps / van_tps, 3),
                     "vanilla_tps": round(van_tps, 1)})
        print(f"  v8 EWMA adapt  : {tps:7.1f} tok/s, tau={tau:.2f}, speedup={tps/van_tps:.2f}×", flush=True)

        # v8 + anchor-entropy adaptive (max budget = 128).
        tps, tau = run_v7(target, draft, prompts_list, B, 128,
                          mn=args.max_new_tokens, bs=block_size,
                          mid=mid, pid=pad_id, eos=eos_ids, dev=device, **V8_ANCHOR_KW)
        rows.append({"variant": "v8_anchor",  "B": B, "M": "anchor", "tau": round(tau, 2),
                     "v7_tps": round(tps, 1), "speedup": round(tps / van_tps, 3),
                     "vanilla_tps": round(van_tps, 1)})
        print(f"  v8 anchor-H    : {tps:7.1f} tok/s, tau={tau:.2f}, speedup={tps/van_tps:.2f}×", flush=True)

    # Save & plot.
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"results": rows, "schedule": SCHEDULE, "v8_kwargs": V8_KW}, f, indent=2)
    print(f"\nWrote {args.output_json}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    by_var = {}
    for r in rows:
        by_var.setdefault(r["variant"], []).append(r)

    styles = {
        "v7_fixed":  ("v7 fixed M=128", "C3", "s", "--"),
        "v7_adapt":  ("v7 adaptive M",  "C2", "D", "-"),
        "v8_fixed":  ("v8 (CGDB+PDRR) fixed M=128", "C4", "^", "--"),
        "v8_adapt":  ("v8 (CGDB+PDRR) adaptive M",  "C0", "o", "-"),
        "v8_ewma":   ("v8 + EWMA adaptive",         "C5", "v", "-"),
        "v8_anchor": ("v8 + anchor-entropy",        "C6", "P", "-"),
    }

    ax = axes[0]
    for var in ["v7_fixed", "v7_adapt", "v8_fixed", "v8_adapt", "v8_ewma", "v8_anchor"]:
        rs = sorted(by_var[var], key=lambda r: r["B"])
        bs = [r["B"] for r in rs]
        spd = [r["speedup"] for r in rs]
        lbl, c, m, ls = styles[var]
        ax.plot(bs, spd, marker=m, color=c, linestyle=ls, linewidth=2.0, label=lbl, markersize=7)
    ax.axhline(1.0, color="k", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Bs); ax.set_xticklabels([str(b) for b in Bs])
    ax.set_xlabel("batch size"); ax.set_ylabel("speedup over vanilla AR")
    ax.set_title("Speedup: v7 vs v8 (CGDB+PDRR), fixed vs adaptive M")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    # Vanilla absolute throughput.
    van_by_B = {r["B"]: r["vanilla_tps"] for r in rows}
    bs_van = sorted(van_by_B)
    ax.plot(bs_van, [van_by_B[b] for b in bs_van], marker="X", color="C1", linestyle=":",
            linewidth=2.0, label="vanilla AR", markersize=7)
    for var in ["v7_fixed", "v7_adapt", "v8_fixed", "v8_adapt", "v8_ewma", "v8_anchor"]:
        if var not in by_var:
            continue
        rs = sorted(by_var[var], key=lambda r: r["B"])
        bs = [r["B"] for r in rs]
        v7 = [r["v7_tps"] for r in rs]
        lbl, c, m, ls = styles[var]
        ax.plot(bs, v7, marker=m, color=c, linestyle=ls, linewidth=2.0, label=lbl, markersize=7)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Bs); ax.set_xticklabels([str(b) for b in Bs])
    ax.set_xlabel("batch size"); ax.set_ylabel("tokens / sec (aggregate)")
    ax.set_title("Aggregate throughput")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper left", fontsize=9)

    plt.suptitle("DFlash v7 vs v8 (CGDB+PDRR) batched benchmark — Qwen/Qwen3-4B on math500", fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output_png) or ".", exist_ok=True)
    plt.savefig(args.output_png, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.output_png}", flush=True)


if __name__ == "__main__":
    main()
