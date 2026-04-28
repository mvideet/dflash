"""Sweep all roadmap enhancements vs baselines across batch sizes.

Configs:
  baseline_v7   : v7 plain at static-schedule M
  v8_handtuned  : v8 (CGDB+PDRR) at static-schedule M
  adaedl        : AdaEDL adaptive M from anchor entropy
  entropy_width : per-position expand_k from current draft entropy
  combined      : v8 + AdaEDL + entropy_width
"""
import argparse, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import DFlashDraftModel, load_and_process_dataset
from dflash_batched import dflash_generate_batched, vanilla_ar_generate_batched
from benchmark_batched import tokenize_prompts, make_padded_batch, chunk_rows_list, _resolve_mts

SCHEDULE = {1: 64, 2: 32, 4: 32, 8: 16, 16: 16, 32: 16}
V8_KW = dict(pdrr_k1=0.5, pdrr_k2=0.25, cgdb_shallow_depth=4,
             cgdb_high_thresh=0.1, cgdb_low_thresh=0.01, cgdb_mid_k=4)


def run_v7(target, draft, prompts_list, B, M, *, mn, bs, mid, pad_id, eos, dev, **extra):
    tot_t, tot_o, accs = 0.0, 0, []
    for chunk in chunk_rows_list(prompts_list, B):
        ids, attn = make_padded_batch(chunk, pad_id, dev)
        s_out = dflash_generate_batched(
            draft=draft, target=target, input_ids=ids, attention_mask=attn,
            mask_token_id=mid, eos_token_ids=eos,
            max_new_tokens=mn, block_size=bs, max_tree_size=M,
            expand_k=8, temperature=0.0, **extra,
        )
        tot_t += s_out.total_decode_time
        tot_o += sum(s_out.num_output_tokens)
        for lst in s_out.acceptance_lengths_per_elem:
            accs.extend(lst)
        del s_out, ids, attn
        torch.cuda.empty_cache()
    return tot_o / tot_t, float(np.mean(accs)) if accs else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--batch-sizes", type=str, default="1,4,8,16,32")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-prompt-tokens", type=int, default=200)
    parser.add_argument("--output-json", type=str, default="logs/enhancements_sweep.json")
    parser.add_argument("--output-png", type=str, default="paper/fig/enhancements_sweep.png")
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
    for B in Bs:
        print(f"\n=== B={B} ===", flush=True)
        # Vanilla AR.
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
        M = _resolve_mts(B, 16, SCHEDULE)
        print(f"  vanilla AR     : {van_tps:7.1f} tok/s  (M for v7/v8 = {M})", flush=True)

        configs = [
            ("baseline_v7",   M, dict()),
            ("v8_handtuned",  M, V8_KW),
            ("adaedl",        M, dict(adaedl_B=True, adaedl_min_M=4, adaedl_max_M=M)),
            ("entropy_width", M, dict(entropy_width=True, entropy_width_min_ek=1,
                                       entropy_width_max_ek=8)),
            ("combined",      M, dict(**V8_KW, adaedl_B=True, adaedl_min_M=4,
                                       adaedl_max_M=M, entropy_width=True,
                                       entropy_width_min_ek=1, entropy_width_max_ek=8)),
        ]

        for name, M_use, kw in configs:
            tps, tau = run_v7(target, draft, prompts_list, B, M_use,
                              mn=args.max_new_tokens, bs=block_size,
                              mid=mid, pad_id=pad_id, eos=eos_ids, dev=device, **kw)
            speedup = tps / van_tps
            rows.append({"variant": name, "B": B, "M": M_use,
                         "tps": round(tps, 1), "tau": round(tau, 3),
                         "speedup": round(speedup, 3),
                         "vanilla_tps": round(van_tps, 1)})
            print(f"  {name:>14}: {tps:7.1f} tok/s, tau={tau:.3f}, speedup={speedup:.3f}×",
                  flush=True)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"results": rows, "schedule": SCHEDULE, "v8_kwargs": V8_KW}, f, indent=2)
    print(f"\nWrote {args.output_json}", flush=True)

    # Plot.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    by_var = {}
    for r in rows:
        by_var.setdefault(r["variant"], []).append(r)
    styles = {
        "baseline_v7":   ("v7 baseline",         "C3", "s"),
        "v8_handtuned":  ("v8 (CGDB+PDRR)",      "C2", "D"),
        "adaedl":        ("AdaEDL (anchor-conf)", "C0", "o"),
        "entropy_width": ("Entropy-width per-pos ek", "C5", "v"),
        "combined":      ("v8 + AdaEDL + entropy_width", "C6", "P"),
    }
    ax = axes[0]
    for var, (lbl, c, m) in styles.items():
        if var not in by_var:
            continue
        rs = sorted(by_var[var], key=lambda r: r["B"])
        ax.plot([r["B"] for r in rs], [r["speedup"] for r in rs],
                marker=m, color=c, linewidth=2.0, label=lbl, markersize=7)
    ax.axhline(1.0, color="k", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Bs); ax.set_xticklabels([str(b) for b in Bs])
    ax.set_xlabel("batch size"); ax.set_ylabel("speedup over vanilla AR")
    ax.set_title("Speedup with roadmap enhancements")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    for var, (lbl, c, m) in styles.items():
        if var not in by_var:
            continue
        rs = sorted(by_var[var], key=lambda r: r["B"])
        ax.plot([r["B"] for r in rs], [r["tau"] for r in rs],
                marker=m, color=c, linewidth=2.0, label=lbl, markersize=7)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Bs); ax.set_xticklabels([str(b) for b in Bs])
    ax.set_xlabel("batch size"); ax.set_ylabel("acceptance length τ")
    ax.set_title("τ across enhancements")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper right", fontsize=9)
    plt.suptitle("Roadmap enhancements — Qwen/Qwen3-4B on math500", fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output_png) or ".", exist_ok=True)
    plt.savefig(args.output_png, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.output_png}", flush=True)


if __name__ == "__main__":
    main()
