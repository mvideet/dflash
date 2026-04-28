"""Long-context regime test — pad math500 prompts with synthetic context to ~P tokens.

Tests whether the kernel/quant levers (FlashInfer, FP8 KV) have a chance
of working on this stack at all. Prefix=200 (default math500) is too short
for verify-attention to be a bottleneck; at P>=4096 it dominates.

Pads each prompt with random tokens before the actual user question. Output
is meaningless but kernel timings reflect the real cost ratio.
"""
import argparse, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import DFlashDraftModel, load_and_process_dataset
from dflash_batched import dflash_generate_batched, vanilla_ar_generate_batched
from benchmark_batched import tokenize_prompts, make_padded_batch, chunk_rows_list, _resolve_mts


SCHEDULE = {1: 64, 2: 32, 4: 32, 8: 16, 16: 16, 32: 16}


def pad_prompts_to_long_context(prompts_list, target_len, tokenizer, device):
    """Prepend random in-vocab tokens to each prompt to hit target_len."""
    padded = []
    V = tokenizer.vocab_size
    for ids in prompts_list:
        L = ids.shape[1]
        if L >= target_len:
            padded.append(ids[:, :target_len])
        else:
            n_pad = target_len - L
            # Random tokens from [1000, 50000) to avoid special tokens.
            random_pad = torch.randint(1000, 50000, (1, n_pad), device=device, dtype=torch.long)
            new_ids = torch.cat([random_pad, ids], dim=1)
            padded.append(new_ids)
    return padded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--batch-sizes", type=str, default="1,4,8,16")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--target-prefix", type=int, default=4096)
    parser.add_argument("--output-json", type=str, default="logs/long_context.json")
    args = parser.parse_args()

    Bs = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    print(f"Loading models (prefix target={args.target_prefix})...", flush=True)
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
        max_prompt_tokens=args.target_prefix, device=device,
    )
    prompts_list = pad_prompts_to_long_context(prompts_list, args.target_prefix, tokenizer, device)
    print(f"Padded {len(prompts_list)} prompts to length ~{args.target_prefix}", flush=True)
    warm_ids, warm_attn = make_padded_batch(prompts_list[:1], pad_id, device)
    _ = vanilla_ar_generate_batched(
        target=target, input_ids=warm_ids, attention_mask=warm_attn,
        eos_token_ids=eos_ids, max_new_tokens=8,
    )
    torch.cuda.empty_cache()

    rows = []
    for B in Bs:
        print(f"\n=== B={B} ===", flush=True)
        # Vanilla.
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
        print(f"  vanilla AR: {van_tps:.1f} tok/s, M={M}", flush=True)

        # v7 plain.
        v7_t, v7_o, accs = 0.0, 0, []
        for chunk in chunk_rows_list(prompts_list, B):
            ids, attn = make_padded_batch(chunk, pad_id, device)
            s_out = dflash_generate_batched(
                draft=draft, target=target, input_ids=ids, attention_mask=attn,
                mask_token_id=mid, eos_token_ids=eos_ids,
                max_new_tokens=args.max_new_tokens, block_size=block_size,
                max_tree_size=M, expand_k=8, temperature=0.0,
            )
            v7_t += s_out.total_decode_time
            v7_o += sum(s_out.num_output_tokens)
            for lst in s_out.acceptance_lengths_per_elem:
                accs.extend(lst)
            del s_out, ids, attn
            torch.cuda.empty_cache()
        v7_tps = v7_o / v7_t
        tau = float(np.mean(accs)) if accs else float("nan")
        speedup = v7_tps / van_tps
        rows.append({"B": B, "M": M, "v7_tps": round(v7_tps, 1),
                     "vanilla_tps": round(van_tps, 1), "speedup": round(speedup, 3),
                     "tau": round(tau, 3), "prefix_target": args.target_prefix})
        print(f"  v7 plain  : {v7_tps:.1f} tok/s, tau={tau:.2f}, speedup={speedup:.3f}×",
              flush=True)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"prefix": args.target_prefix, "results": rows}, f, indent=2)
    print(f"\nWrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
