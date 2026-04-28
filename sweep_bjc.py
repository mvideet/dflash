"""BJC-Tree headline sweep: v7 vs v8 vs v7+BJC at multiple B's.

Three configs at each B:
  - v7 plain (no score adjustments)
  - v8 (hand-tuned PDRR + CGDB) — current best non-training algorithmic
  - v7 + BJC (online Bayesian joint correction, α=2.0)

Single in-process run; one model load.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import DFlashDraftModel, load_and_process_dataset
from model.joint_dist_calib import JointCalib
from dflash_batched import dflash_generate_batched, vanilla_ar_generate_batched
from benchmark_batched import tokenize_prompts, make_padded_batch, chunk_rows_list, _resolve_mts


SCHEDULE = {1: 64, 2: 32, 4: 32, 8: 16, 16: 16, 32: 16}

V8_KW = dict(pdrr_k1=0.5, pdrr_k2=0.25,
             cgdb_shallow_depth=4, cgdb_high_thresh=0.1,
             cgdb_low_thresh=0.01, cgdb_mid_k=4)


def run_v7(target, draft, prompts_list, B, M, *, mn, bs, mid, pad_id, eos, dev,
           bjc_alpha=None, **extra):
    v7_t, v7_o, accs = 0.0, 0, []
    # NEW JointCalib per "run" so we don't carry across configs.
    calib = None
    if bjc_alpha is not None:
        calib = JointCalib(max_depth=bs, K=8, alpha_prior=bjc_alpha,
                           min_count=20.0, device=dev)

    for chunk in chunk_rows_list(prompts_list, B):
        ids, attn = make_padded_batch(chunk, pad_id, dev)
        s_out = dflash_generate_batched(
            draft=draft, target=target, input_ids=ids, attention_mask=attn,
            mask_token_id=mid, eos_token_ids=eos,
            max_new_tokens=mn, block_size=bs, max_tree_size=M,
            expand_k=8, temperature=0.0, bjc_calib=calib, **extra,
        )
        v7_t += s_out.total_decode_time
        v7_o += sum(s_out.num_output_tokens)
        for lst in s_out.acceptance_lengths_per_elem:
            accs.extend(lst)
        del s_out, ids, attn
        torch.cuda.empty_cache()
    n_obs = calib.num_observations() if calib is not None else 0
    return v7_o / v7_t, float(np.mean(accs)) if accs else float("nan"), n_obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--batch-sizes", type=str, default="1,4,8,16,32")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-prompt-tokens", type=int, default=200)
    parser.add_argument("--bjc-alpha", type=float, default=2.0)
    parser.add_argument("--output-json", type=str, default="logs/bjc_sweep.json")
    parser.add_argument("--output-png", type=str, default="paper/fig/bjc_sweep.png")
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

        # Use adaptive M from static schedule for all v7/v8/BJC configs.
        M = _resolve_mts(B, 16, SCHEDULE)

        # v7 plain.
        tps, tau, _ = run_v7(target, draft, prompts_list, B, M,
                             mn=args.max_new_tokens, bs=block_size,
                             mid=mid, pad_id=pad_id, eos=eos_ids, dev=device)
        rows.append({"variant": "v7",   "B": B, "M": M, "tau": round(tau, 3),
                     "v7_tps": round(tps, 1), "speedup": round(tps / van_tps, 3),
                     "vanilla_tps": round(van_tps, 1), "n_obs": 0})
        print(f"  v7 plain  M={M:>3}: {tps:7.1f} tok/s, tau={tau:.3f}, speedup={tps/van_tps:.3f}×", flush=True)

        # v8 (hand-tuned).
        tps, tau, _ = run_v7(target, draft, prompts_list, B, M,
                             mn=args.max_new_tokens, bs=block_size,
                             mid=mid, pad_id=pad_id, eos=eos_ids, dev=device, **V8_KW)
        rows.append({"variant": "v8",   "B": B, "M": M, "tau": round(tau, 3),
                     "v7_tps": round(tps, 1), "speedup": round(tps / van_tps, 3),
                     "vanilla_tps": round(van_tps, 1), "n_obs": 0})
        print(f"  v8 hand   M={M:>3}: {tps:7.1f} tok/s, tau={tau:.3f}, speedup={tps/van_tps:.3f}×", flush=True)

        # v7 + BJC-Tree (online).
        tps, tau, n_obs = run_v7(target, draft, prompts_list, B, M,
                                 mn=args.max_new_tokens, bs=block_size,
                                 mid=mid, pad_id=pad_id, eos=eos_ids, dev=device,
                                 bjc_alpha=args.bjc_alpha)
        rows.append({"variant": "bjc",  "B": B, "M": M, "tau": round(tau, 3),
                     "v7_tps": round(tps, 1), "speedup": round(tps / van_tps, 3),
                     "vanilla_tps": round(van_tps, 1), "n_obs": int(n_obs)})
        print(f"  v7+BJC    M={M:>3}: {tps:7.1f} tok/s, tau={tau:.3f}, speedup={tps/van_tps:.3f}× (α={args.bjc_alpha}, n_obs={n_obs:.0f})", flush=True)

    # Save & plot.
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"results": rows, "schedule": SCHEDULE,
                   "v8_kwargs": V8_KW, "bjc_alpha": args.bjc_alpha}, f, indent=2)
    print(f"\nWrote {args.output_json}", flush=True)

    # Plot.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    by_var = {}
    for r in rows:
        by_var.setdefault(r["variant"], []).append(r)

    styles = {
        "v7":  ("v7 (DDTree, no correction)",                "C3", "s", "-"),
        "v8":  ("v8 (CGDB+PDRR, hand-tuned)",                "C2", "D", "-"),
        "bjc": (f"v7 + BJC-Tree (online, α={args.bjc_alpha})", "C0", "o", "-"),
    }

    ax = axes[0]
    for var in ["v7", "v8", "bjc"]:
        rs = sorted(by_var[var], key=lambda r: r["B"])
        bs = [r["B"] for r in rs]
        spd = [r["speedup"] for r in rs]
        lbl, c, m, ls = styles[var]
        ax.plot(bs, spd, marker=m, color=c, linestyle=ls, linewidth=2.0, label=lbl, markersize=7)
        for x, y in zip(bs, spd):
            ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=7, color=c)
    ax.axhline(1.0, color="k", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Bs); ax.set_xticklabels([str(b) for b in Bs])
    ax.set_xlabel("batch size"); ax.set_ylabel("speedup over vanilla AR")
    ax.set_title("Speedup: v7 vs v8 vs BJC-Tree (adaptive M schedule)")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    for var in ["v7", "v8", "bjc"]:
        rs = sorted(by_var[var], key=lambda r: r["B"])
        bs = [r["B"] for r in rs]
        tau = [r["tau"] for r in rs]
        lbl, c, m, ls = styles[var]
        ax.plot(bs, tau, marker=m, color=c, linestyle=ls, linewidth=2.0, label=lbl, markersize=7)
        for x, y in zip(bs, tau):
            ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=7, color=c)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Bs); ax.set_xticklabels([str(b) for b in Bs])
    ax.set_xlabel("batch size"); ax.set_ylabel("acceptance length τ")
    ax.set_title("Acceptance length τ (higher is better)")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper right", fontsize=9)

    plt.suptitle("BJC-Tree (online Bayesian) vs v7 vs v8 — Qwen/Qwen3-4B on math500", fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output_png) or ".", exist_ok=True)
    plt.savefig(args.output_png, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.output_png}", flush=True)


if __name__ == "__main__":
    main()
