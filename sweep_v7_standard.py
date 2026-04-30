"""v7 curve at math500 standard token budget: 512 prompt + 1024 generation.

This is the proper benchmark configuration — long enough for full step-by-step
reasoning solutions. Memory-bound at high B; sweep only goes as far as fits.
"""
import argparse, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import DFlashDraftModel, load_and_process_dataset
from dflash_batched import dflash_generate_batched, vanilla_ar_generate_batched
from benchmark_batched import tokenize_prompts, make_padded_batch, chunk_rows_list

SCHEDULE = {1: 64, 2: 32, 4: 32, 8: 16, 16: 16, 32: 16, 64: 16, 128: 8, 192: 8, 256: 8}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--output-json", type=str, default="logs/v7_math500_standard.json")
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
    print(f"Tokenized {len(prompts_list)} prompts (max_prompt={args.max_prompt_tokens}).",
          flush=True)

    warm_ids, warm_attn = make_padded_batch(prompts_list[:1], pad_id, device)
    _ = vanilla_ar_generate_batched(
        target=target, input_ids=warm_ids, attention_mask=warm_attn,
        eos_token_ids=eos_ids, max_new_tokens=8,
    )
    torch.cuda.empty_cache()

    Bs = [1, 2, 4, 8, 16, 32, 64]
    rows = []
    for B in Bs:
        M = SCHEDULE.get(B, 16)
        if B > len(prompts_list):
            prompts_for_B = (prompts_list * ((B // len(prompts_list)) + 1))[:B]
        else:
            prompts_for_B = prompts_list[:B]
        try:
            torch.cuda.empty_cache()
            van_t, van_o = 0.0, 0
            for chunk in chunk_rows_list(prompts_for_B, B):
                ids, attn = make_padded_batch(chunk, pad_id, device)
                v_out = vanilla_ar_generate_batched(
                    target=target, input_ids=ids, attention_mask=attn,
                    eos_token_ids=eos_ids, max_new_tokens=args.max_new_tokens,
                )
                van_t += v_out.total_decode_time
                van_o += sum(v_out.num_output_tokens)
                del v_out, ids, attn; torch.cuda.empty_cache()
            van_tps = van_o / van_t

            v7_t, v7_o, accs = 0.0, 0, []
            for chunk in chunk_rows_list(prompts_for_B, B):
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
                del s_out, ids, attn; torch.cuda.empty_cache()
            v7_tps = v7_o / v7_t
            tau = float(np.mean(accs)) if accs else float("nan")
            speedup = v7_tps / van_tps
            rows.append({"B": B, "M": M,
                         "vanilla_tps": round(van_tps, 1), "v7_tps": round(v7_tps, 1),
                         "speedup": round(speedup, 3), "tau": round(tau, 3)})
            print(f"  B={B:>3} M={M:>3}: van={van_tps:.0f} v7={v7_tps:.0f} "
                  f"tau={tau:.2f} speedup={speedup:.3f}×", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  B={B}: OOM ({str(e)[:120]})", flush=True)
            rows.append({"B": B, "M": M, "error": "OOM"})
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"  B={B}: ERROR {str(e)[:120]}", flush=True)
            rows.append({"B": B, "M": M, "error": str(e)[:200]})
            torch.cuda.empty_cache()

    print("\n=== math500 standard config (max_prompt=512, max_new=1024) ===", flush=True)
    print(f"  {'B':>3} {'M':>3} {'vanilla':>9} {'v7':>9} {'tau':>6} {'speedup':>8}", flush=True)
    for r in rows:
        if "error" in r:
            print(f"  {r['B']:>3} {r['M']:>3} {'--':>9} {'--':>9} {'--':>6} {r['error']:>8}", flush=True)
        else:
            print(f"  {r['B']:>3} {r['M']:>3} {r['vanilla_tps']:>8.1f} {r['v7_tps']:>8.1f} "
                  f"{r['tau']:>6.2f} {r['speedup']:>7.3f}×", flush=True)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"results": rows, "config": {
            "max_new_tokens": args.max_new_tokens,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_samples": args.max_samples,
        }}, f, indent=2)
    print(f"\nWrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
