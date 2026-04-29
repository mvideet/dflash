"""Profile spine-first sub-phase timings.

Goal: confirm whether spine-first slowdown is from Python orchestration or
from compute. Compare:
  - Baseline single-pass forward(M)
  - Spine-first stage1 forward(P) + stage2 forward(M-P)

Reports: total verify time, spine time, side time, full-spine-accept rate.
"""
import argparse, json, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import DFlashDraftModel, load_and_process_dataset
from dflash_batched import dflash_generate_batched, vanilla_ar_generate_batched
from benchmark_batched import tokenize_prompts, make_padded_batch, chunk_rows_list, _resolve_mts

SCHEDULE = {1: 64, 2: 32, 4: 32, 8: 16, 16: 16, 32: 16}

CONFIGS = [
    ("B=1 baseline",       1, dict()),
    ("B=1 spine_first",    1, dict(spine_first=True)),
    ("B=4 ewma",           4, dict(ewma_adaptive=True, ewma_min_M=8, ewma_max_ek=8, ewma_min_ek=2)),
    ("B=4 ewma+spine",     4, dict(ewma_adaptive=True, ewma_min_M=8, ewma_max_ek=8, ewma_min_ek=2, spine_first=True)),
    ("B=8 specdecpp",      8, dict(specdecpp_threshold=0.05)),
    ("B=8 specdecpp+spine",8, dict(specdecpp_threshold=0.05, spine_first=True)),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--output-json", type=str, default="logs/spine_first_profile.json")
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

    rows = []
    for label, B, mode_kwargs in CONFIGS:
        M = _resolve_mts(B, 16, SCHEDULE)
        agg = {"draft": 0.0, "tree_build": 0.0, "verify": 0.0,
               "spine_verify": 0.0, "side_verify": 0.0,
               "spine_full_count": 0, "side_runs": 0,
               "select": 0.0, "trim": 0.0, "steps": 0}
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
                if k in s_out.phase_timings_ms:
                    agg[k] += s_out.phase_timings_ms[k]
            total_decode_t += s_out.total_decode_time
            del s_out, ids, attn
            torch.cuda.empty_cache()

        steps = max(agg["steps"], 1)
        per_step = {k: round(v / steps, 3) for k, v in agg.items()
                    if k not in ("steps", "spine_full_count", "side_runs")}
        print(f"\n=== {label} (B={B}, M={M}, {steps} rounds, "
              f"total={total_decode_t*1000:.0f}ms) ===", flush=True)
        print(f"  per-round phase_ms:", flush=True)
        for k in ["draft", "tree_build", "verify", "spine_verify", "side_verify", "select", "trim"]:
            print(f"    {k:>12}: {per_step[k]:>7.3f}", flush=True)
        if "spine" in label:
            print(f"  spine_full_accept: {agg['spine_full_count']}/{steps} "
                  f"({100*agg['spine_full_count']/steps:.1f}%)", flush=True)
            print(f"  side_runs: {agg['side_runs']}/{steps} "
                  f"({100*agg['side_runs']/steps:.1f}%)", flush=True)

        rows.append({
            "label": label, "B": B, "M": M, "n_steps": steps,
            "total_decode_ms": round(total_decode_t * 1000, 1),
            "phase_per_step_ms": per_step,
            "spine_full_count": agg["spine_full_count"],
            "side_runs": agg["side_runs"],
        })

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"results": rows}, f, indent=2)
    print(f"\nWrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
