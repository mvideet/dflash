"""Batch-size scaling micro-benchmark for DFlash v7 (ddtree) vs vanilla AR.

Why this exists
---------------
`benchmark.py` and the v7 sweep scripts can only run at batch size 1 — the tree
builders (`build_node_budget_tree`, `create_tree_attention_mask_dynamic`) hard-
assert B=1, and `dflash_generate` itself is single-stream. In real serving you
batch many requests through one target forward, which eats into spec-decoding's
margin: the AR baseline gets cheaper per-token as B grows (memory-bound), while
the tree-verify forward also gets faster per-batch-element but starts from a
much larger N_tokens. So the speedup of v7 over vanilla AR is a function of B,
and we currently can't measure it.

What this script measures
-------------------------
Per-step kernel cost at varying batch sizes, then converts to a tokens/sec
speedup curve via tau (acceptance length, ~B-invariant under our assumptions).

For each B in --batch-sizes:
  1. Replicate the same prompt across the batch dim and prefill (B, S).
  2. Time T_AR(B): one target.model([B, 1]) forward (vanilla autoregressive).
  3. Time T_v7(B): draft.forward([B, block_size]) + tree-build + target.model
     ([B, L_tree], with tree attention mask). The tree is built once from
     element-0's draft logits and replicated across the batch — this is the
     padding-free realistic case (v7's node budget caps every element's tree
     to the same size, so identical trees match real batched decoding kernel
     shape exactly).
  4. speedup(B) = tau * T_AR(B) / T_v7(B).

Tau is measured once at B=1 by running real `dflash_generate` on a few prompts
(or passed via --tau-fixed). Tau is essentially independent of B in the kernel
timing we model here.

Steady-state caveat: every timed iteration restores the draft+target KV caches
to their "after first decode step" lengths so that we measure steady-state cost,
not the one-shot long first-step cost (where the draft processes positions
0..S+block instead of just S..S+block).
"""
import argparse
import json
import os
import time
from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from model import (
    DFlashDraftModel,
    extract_context_feature,
    load_and_process_dataset,
)
from model.dflash_tree import (
    build_node_budget_tree,
    create_tree_attention_mask_dynamic,
)


# --------------------------------------------------------------------------
# DynamicCache helpers (DynamicCache lacks a public crop-to-length method)
# --------------------------------------------------------------------------

def _crop_kv(past_kv: DynamicCache, target_len: int) -> None:
    for layer in past_kv.layers:
        layer.keys = layer.keys[:, :, :target_len, :].contiguous()
        layer.values = layer.values[:, :, :target_len, :].contiguous()


# --------------------------------------------------------------------------
# Kernel-timing primitives
# --------------------------------------------------------------------------

@torch.inference_mode()
def time_vanilla_ar_step(
    target,
    target_kv: DynamicCache,
    prev_tok: torch.Tensor,
    n_iter: int,
    warmup: int,
) -> float:
    """Time one vanilla AR step at the prefill state.

    Each step: target.model([B, 1]) → grow KV by 1 → crop back. Steady-state
    shape (Q=1, K=S+1, then truncated) is identical across iters.
    """
    prefix_len = target_kv.get_seq_length()
    B = prev_tok.shape[0]
    pos_ids = torch.full((B, 1), prefix_len, dtype=torch.long, device=prev_tok.device)

    def step():
        target.model(
            prev_tok,
            position_ids=pos_ids,
            past_key_values=target_kv,
            use_cache=True,
        )
        _crop_kv(target_kv, prefix_len)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter


@torch.inference_mode()
def time_v7_step(
    target,
    draft,
    target_hidden_steady: torch.Tensor,   # [B, ctx, D'] — cross-context for draft
    prev_tok: torch.Tensor,                # [B, 1] last accepted token
    target_kv: DynamicCache,               # warm at prefix_len_target = S
    draft_kv: DynamicCache,                # warm at prefix_len_draft = S (after long-first-step)
    block_size: int,
    max_tree_size: int,
    expand_k: int,
    mask_token_id: int,
    n_iter: int,
    warmup: int,
) -> dict:
    """Time one v7 step: draft fwd + tree-build (B=1, replicated) + target tree-verify."""
    prefix_len_target = target_kv.get_seq_length()
    prefix_len_draft = draft_kv.get_seq_length()
    B = prev_tok.shape[0]
    device = prev_tok.device

    # Build the input block: [anchor, mask, mask, ..., mask] of length block_size.
    block_output_ids = torch.full(
        (B, block_size), mask_token_id, dtype=torch.long, device=device,
    )
    block_output_ids[:, 0] = prev_tok[:, 0]
    block_pos_ids = torch.arange(
        prefix_len_draft, prefix_len_draft + block_size, device=device, dtype=torch.long,
    ).unsqueeze(0).expand(B, -1).contiguous()

    sub_times = {"draft_fwd": 0.0, "tree_build": 0.0, "target_verify": 0.0}

    def step(record: bool = False):
        if record:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        # 1) Draft forward
        noise_embedding = target.model.embed_tokens(block_output_ids)
        draft_hidden = draft(
            target_hidden=target_hidden_steady,
            noise_embedding=noise_embedding,
            position_ids=block_pos_ids,
            past_key_values=draft_kv,
            use_cache=True,
            is_causal=False,
        )[:, -block_size + 1:, :]
        _crop_kv(draft_kv, prefix_len_draft)

        if record:
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            sub_times["draft_fwd"] += (t1 - t0)

        # lm_head on element 0 only — tree builder is B=1.
        draft_logits_one = target.lm_head(draft_hidden[:1, :, :])

        # 2) Tree build (Python/CPU, single elem)
        packed_ids_1, packed_pos_relative_1, parent_idx, _leaf_paths, _leaf_tokens = build_node_budget_tree(
            draft_logits=draft_logits_one,
            anchor_token_ids=block_output_ids[:1, :1],
            max_tree_size=max_tree_size,
            expand_k=expand_k,
        )
        packed_ids_B = packed_ids_1.expand(B, -1).contiguous()
        packed_pos_B = (packed_pos_relative_1 + prefix_len_target).expand(B, -1).contiguous()
        attn_mask = create_tree_attention_mask_dynamic(
            packed_pos_relative_1, parent_idx, prefix_len_target,
        )

        if record:
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            sub_times["tree_build"] += (t2 - t1)

        # 3) Target verify (Q=L_tree, K=S+L_tree). Use sdpa for the custom mask.
        saved_attn = target.config._attn_implementation
        target.config._attn_implementation = "sdpa"
        out = target.model(
            packed_ids_B,
            position_ids=packed_pos_B,
            past_key_values=target_kv,
            use_cache=True,
            attention_mask=attn_mask,
        )
        target.lm_head(out.last_hidden_state)
        target.config._attn_implementation = saved_attn
        _crop_kv(target_kv, prefix_len_target)

        if record:
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            sub_times["target_verify"] += (t3 - t2)

    for _ in range(warmup):
        step(record=False)
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    for _ in range(n_iter):
        step(record=True)
    torch.cuda.synchronize()
    total = time.perf_counter() - t_start

    return {
        "total_per_step": total / n_iter,
        "draft_fwd_per_step": sub_times["draft_fwd"] / n_iter,
        "tree_build_per_step": sub_times["tree_build"] / n_iter,
        "target_verify_per_step": sub_times["target_verify"] / n_iter,
    }


# --------------------------------------------------------------------------
# Tau measurement (one-time, B=1)
# --------------------------------------------------------------------------

@torch.inference_mode()
def measure_tau_b1(
    target, draft, tokenizer, prompts: List[str],
    max_new_tokens: int, block_size: int, max_tree_size: int, expand_k: int,
) -> float:
    """Run real dflash_generate at B=1 on a few prompts to estimate tau."""
    from benchmark import dflash_generate
    accs = []
    for p in prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        ids = tokenizer.encode(text, return_tensors="pt").to(target.device)
        r = dflash_generate(
            model=draft, target=target, input_ids=ids,
            mask_token_id=draft.mask_token_id,
            max_new_tokens=max_new_tokens, block_size=block_size,
            stop_token_ids=[tokenizer.eos_token_id], temperature=0.0,
            tree_version=7, max_tree_size=max_tree_size, expand_k=expand_k,
        )
        if r.acceptance_lengths:
            accs.append(float(np.mean(r.acceptance_lengths)))
    return float(np.mean(accs)) if accs else float("nan")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

@torch.inference_mode()
def run_one_batch_size(
    target, draft, tokenizer, ids: torch.Tensor, B: int,
    block_size: int, max_tree_size: int, expand_k: int,
    n_iter: int, warmup: int, device: torch.device,
) -> dict:
    """Prefill at batch B and time both vanilla AR and v7."""
    S = ids.shape[1]
    ids_B = ids.expand(B, -1).contiguous()
    pos_B = torch.arange(S, device=device).unsqueeze(0).expand(B, -1).contiguous()

    target_kv = DynamicCache()
    out = target(
        ids_B, position_ids=pos_B, past_key_values=target_kv,
        use_cache=True, logits_to_keep=1, output_hidden_states=True,
    )
    target_hidden_full = extract_context_feature(out.hidden_states, draft.target_layer_ids)
    # Steady-state target_hidden ≈ post-first-step shape ([B, tau+1, D']). We
    # use last block_size positions of prefill output as a stand-in (slight
    # overestimate of ctx_len; same shape across B for fair comparison).
    target_hidden_steady = target_hidden_full[:, -block_size:, :].contiguous()
    first_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]
    del out, target_hidden_full

    # Time vanilla AR (uses target_kv at prefix_len = S, restored each iter).
    t_ar = time_vanilla_ar_step(
        target=target, target_kv=target_kv, prev_tok=first_tok,
        n_iter=n_iter, warmup=warmup,
    )

    # Set up draft KV: simulate "after long first decode step" by running one
    # long forward (positions [0, S+block_size]) then cropping back to S. This
    # is an untimed setup so the timed iterations reflect steady state.
    draft_kv = DynamicCache()
    long_pos = torch.arange(0, S + block_size, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1).contiguous()
    long_block = torch.full(
        (B, S + block_size), draft.mask_token_id, dtype=torch.long, device=device,
    )
    long_block[:, :S] = ids_B
    # Set the anchor token at position S (start of the synthetic block).
    long_block[:, S] = first_tok[:, 0]
    long_noise = target.model.embed_tokens(long_block)
    _ = draft(
        target_hidden=target_hidden_steady,
        noise_embedding=long_noise,
        position_ids=long_pos,
        past_key_values=draft_kv,
        use_cache=True,
        is_causal=False,
    )
    _crop_kv(draft_kv, S)
    del long_noise, long_block, long_pos

    # Time v7 step.
    v7_times = time_v7_step(
        target=target, draft=draft,
        target_hidden_steady=target_hidden_steady,
        prev_tok=first_tok,
        target_kv=target_kv, draft_kv=draft_kv,
        block_size=block_size, max_tree_size=max_tree_size, expand_k=expand_k,
        mask_token_id=draft.mask_token_id,
        n_iter=n_iter, warmup=warmup,
    )

    # Free this batch's KV before next B.
    del target_kv, draft_kv, target_hidden_steady, first_tok, ids_B, pos_B
    torch.cuda.empty_cache()

    return {
        "batch_size": B,
        "t_ar_ms": t_ar * 1000,
        "t_v7_ms": v7_times["total_per_step"] * 1000,
        "t_v7_draft_ms": v7_times["draft_fwd_per_step"] * 1000,
        "t_v7_treebuild_ms": v7_times["tree_build_per_step"] * 1000,
        "t_v7_verify_ms": v7_times["target_verify_per_step"] * 1000,
    }


def make_plot(results: List[dict], tau: float, out_png: str, meta: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bs = [r["batch_size"] for r in results]
    speedups = [tau * r["t_ar_ms"] / r["t_v7_ms"] for r in results]
    vanilla_tps = [B * 1000 / r["t_ar_ms"] for B, r in zip(bs, results)]
    v7_tps = [B * tau * 1000 / r["t_v7_ms"] for B, r in zip(bs, results)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Speedup vs batch size
    ax = axes[0]
    ax.plot(bs, speedups, marker="o", color="C0", linewidth=2)
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(bs)
    ax.set_xticklabels([str(b) for b in bs])
    ax.set_xlabel("batch size")
    ax.set_ylabel("v7 ddtree speedup over vanilla AR")
    ax.set_title(f"Speedup vs batch size (tau={tau:.2f}, B_tree={meta['max_tree_size']})")
    ax.grid(True, alpha=0.3)
    for x, y in zip(bs, speedups):
        ax.annotate(f"{y:.2f}×", xy=(x, y), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=8)

    # tokens/s vs batch size (per algorithm)
    ax = axes[1]
    ax.plot(bs, vanilla_tps, marker="s", label="vanilla AR", color="C1", linewidth=2)
    ax.plot(bs, v7_tps, marker="o", label="v7 ddtree", color="C0", linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(bs)
    ax.set_xticklabels([str(b) for b in bs])
    ax.set_xlabel("batch size")
    ax.set_ylabel("tokens / sec (aggregate across batch)")
    ax.set_title("Aggregate throughput")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.suptitle(f"DFlash v7 batch-size scaling — {meta['model']}", fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  → {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--draft-name-or-path", type=str, default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32",
                        help="Comma-separated list of batch sizes to profile.")
    parser.add_argument("--max-tree-size", type=int, default=128)
    parser.add_argument("--expand-k", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=None,
                        help="Override draft.block_size if set.")
    parser.add_argument("--n-iter", type=int, default=20,
                        help="Timed iterations per batch size.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--prompt", type=str,
                        default="Solve the integral: ∫(x² + 3x + 2) dx from 0 to 5. Show every step.",
                        help="Single prompt to replicate across the batch.")
    parser.add_argument("--max-prompt-tokens", type=int, default=256,
                        help="Truncate prompt so prefill memory stays bounded.")
    parser.add_argument("--tau-prompts", type=int, default=0,
                        help="If >0, measure tau via dflash_generate on this many math500 prompts.")
    parser.add_argument("--tau-fixed", type=float, default=6.5,
                        help="Tau used in speedup formula when --tau-prompts=0. "
                             "6.5 ≈ math500 v7 mts=128 ek=8.")
    parser.add_argument("--tau-max-new-tokens", type=int, default=256)
    parser.add_argument("--output-json", type=str, default="logs/profile_batch_size.json")
    parser.add_argument("--output-png", type=str, default="paper/fig/speedup_vs_batch_size.png")
    args = parser.parse_args()

    bs_list = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    print(f"Loading target  : {args.model_name_or_path}")
    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, attn_implementation="sdpa", dtype=torch.bfloat16,
    ).to(device).eval()
    print(f"Loading draft   : {args.draft_name_or_path}")
    draft = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path, attn_implementation="sdpa", dtype=torch.bfloat16,
    ).to(device).eval()
    block_size = args.block_size if args.block_size is not None else draft.block_size
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    print(f"block_size={block_size}, max_tree_size={args.max_tree_size}, expand_k={args.expand_k}")

    # Tau (acceptance length) — used in the speedup formula.
    if args.tau_prompts > 0:
        ds = load_and_process_dataset("math500").shuffle(seed=0).select(range(args.tau_prompts))
        prompts = [ds[i]["turns"][0] for i in range(args.tau_prompts)]
        tau = measure_tau_b1(
            target, draft, tokenizer, prompts,
            max_new_tokens=args.tau_max_new_tokens, block_size=block_size,
            max_tree_size=args.max_tree_size, expand_k=args.expand_k,
        )
        print(f"Measured tau (B=1, n={args.tau_prompts}) = {tau:.3f}")
    else:
        tau = args.tau_fixed
        print(f"Using fixed tau={tau:.3f} (skip --tau-prompts to override)")

    # Tokenize prompt.
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    ids = tokenizer.encode(text, return_tensors="pt").to(device)
    if ids.shape[1] > args.max_prompt_tokens:
        ids = ids[:, :args.max_prompt_tokens]
    S = ids.shape[1]
    print(f"Prompt length S={S}\n")

    results = []
    for B in bs_list:
        print(f"=== batch_size = {B} ===")
        try:
            r = run_one_batch_size(
                target=target, draft=draft, tokenizer=tokenizer, ids=ids, B=B,
                block_size=block_size, max_tree_size=args.max_tree_size, expand_k=args.expand_k,
                n_iter=args.n_iter, warmup=args.warmup, device=device,
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM at B={B}: {e}. Stopping.")
            torch.cuda.empty_cache()
            break
        speedup = tau * r["t_ar_ms"] / r["t_v7_ms"]
        vanilla_tps = B * 1000 / r["t_ar_ms"]
        v7_tps = B * tau * 1000 / r["t_v7_ms"]
        print(f"  T_AR        = {r['t_ar_ms']:6.2f} ms/step  → {vanilla_tps:7.1f} tok/s")
        print(f"  T_v7        = {r['t_v7_ms']:6.2f} ms/step  → {v7_tps:7.1f} tok/s")
        print(f"    draft fwd       = {r['t_v7_draft_ms']:6.2f} ms")
        print(f"    tree build (CPU)= {r['t_v7_treebuild_ms']:6.2f} ms")
        print(f"    target verify   = {r['t_v7_verify_ms']:6.2f} ms")
        print(f"  speedup     = {speedup:.2f}×\n")
        results.append(r)

    if not results:
        print("No results — aborting plot.")
        return

    meta = {
        "model": args.model_name_or_path,
        "draft": args.draft_name_or_path,
        "block_size": block_size,
        "max_tree_size": args.max_tree_size,
        "expand_k": args.expand_k,
        "prompt_len": S,
        "n_iter": args.n_iter,
        "warmup": args.warmup,
    }
    payload = {"tau": tau, "meta": meta, "results": results}
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {args.output_json}")

    make_plot(results, tau=tau, out_png=args.output_png, meta=meta)


if __name__ == "__main__":
    main()
