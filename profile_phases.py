"""Per-phase timing breakdown — figure out where the round time actually goes.

Phases:
  draft       : DFlash forward + cache append/crop
  tree_build  : Python heap + tensor pack
  verify      : target.model() + lm_head
  select      : leaf accept-path search + bonus arg-max
  trim        : KV cache trim
  total       : sum across all rounds for the run

Run B={1,4,8} at current best config. Note total decode time isn't sum of
phases (Python interspersed accounting). Phase totals dominated by big-3:
draft/verify/tree_build.
"""
import argparse, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import DFlashDraftModel, load_and_process_dataset
from dflash_batched import dflash_generate_batched, vanilla_ar_generate_batched
from benchmark_batched import tokenize_prompts, make_padded_batch, chunk_rows_list, _resolve_mts

SCHEDULE = {1: 64, 2: 32, 4: 32, 8: 16, 16: 16, 32: 16}

CONFIGS = [
    ("B=1 baseline",    1, dict()),
    ("B=4 ewma",        4, dict(ewma_adaptive=True, ewma_min_M=8, ewma_max_ek=8, ewma_min_ek=2)),
    ("B=8 specdecpp",   8, dict(specdecpp_threshold=0.05)),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--output-json", type=str, default="logs/phase_profile.json")
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

    # Warmup.
    warm_ids, warm_attn = make_padded_batch(prompts_list[:1], pad_id, device)
    _ = vanilla_ar_generate_batched(
        target=target, input_ids=warm_ids, attention_mask=warm_attn,
        eos_token_ids=eos_ids, max_new_tokens=8,
    )
    torch.cuda.empty_cache()

    out = {"configs": []}
    for label, B, mode_kwargs in CONFIGS:
        M = _resolve_mts(B, 16, SCHEDULE)
        agg = {"draft": 0.0, "tree_build": 0.0, "verify": 0.0, "select": 0.0, "trim": 0.0, "steps": 0}
        total_decode_t = 0.0
        for chunk in chunk_rows_list(prompts_list, B):
            ids, attn = make_padded_batch(chunk, pad_id, device)
            s_out = dflash_generate_batched(
                draft=draft, target=target, input_ids=ids, attention_mask=attn,
                mask_token_id=mid, eos_token_ids=eos_ids,
                max_new_tokens=args.max_new_tokens, block_size=block_size,
                max_tree_size=M, expand_k=8, temperature=0.0,
                log_phase_timings=True,
                **mode_kwargs,
            )
            for k in agg:
                agg[k] += s_out.phase_timings_ms[k]
            total_decode_t += s_out.total_decode_time
            del s_out, ids, attn
            torch.cuda.empty_cache()

        total_phases = sum(v for k, v in agg.items() if k != "steps")
        per_step_ms = {k: round(v / max(agg["steps"], 1), 3) for k, v in agg.items() if k != "steps"}
        pct = {k: round(100 * v / max(total_phases, 1), 1) for k, v in agg.items() if k != "steps"}

        print(f"\n=== {label} (B={B}, M={M}, {agg['steps']} rounds, "
              f"total_decode={total_decode_t*1000:.0f}ms) ===", flush=True)
        print(f"  {'phase':>10} {'total_ms':>10} {'per_step_ms':>14} {'pct':>6}", flush=True)
        order = ["draft", "tree_build", "verify", "select", "trim"]
        for k in order:
            print(f"  {k:>10} {agg[k]:>10.1f} {per_step_ms[k]:>14.3f} {pct[k]:>5.1f}%", flush=True)
        print(f"  {'(sum)':>10} {total_phases:>10.1f}", flush=True)
        accounted_pct = 100 * total_phases / (total_decode_t * 1000)
        print(f"  accounted: {accounted_pct:.1f}% of total decode time "
              f"({total_phases:.0f}ms / {total_decode_t*1000:.0f}ms)", flush=True)

        out["configs"].append({
            "label": label, "B": B, "M": M,
            "n_steps": agg["steps"],
            "total_decode_ms": round(total_decode_t * 1000, 1),
            "phase_total_ms": agg,
            "phase_pct": pct,
            "phase_per_step_ms": per_step_ms,
            "accounted_pct": round(accounted_pct, 1),
        })

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
