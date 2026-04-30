"""v7 (DDTree) at very large batch sizes: B=128, 192, 256, 384, 512.

Earlier data ended at B=128 (1.96×). Pushing higher to map the curve.
Uses small M (8 or 4) to fit in A6000 48GB memory.
"""
import argparse, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import DFlashDraftModel, load_and_process_dataset
from dflash_batched import dflash_generate_batched, vanilla_ar_generate_batched
from benchmark_batched import tokenize_prompts, make_padded_batch, chunk_rows_list

# Per-B schedule: smaller M at higher B to avoid OOM.
# At B=128 we used M=8 successfully.
B_M_CONFIGS = [
    (128, 8),
    (192, 8),
    (256, 8),
    (384, 4),
    (512, 4),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--output-json", type=str, default="logs/v7_largeB.json")
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
    print(f"Tokenized {len(prompts_list)} prompts.", flush=True)

    warm_ids, warm_attn = make_padded_batch(prompts_list[:1], pad_id, device)
    _ = vanilla_ar_generate_batched(
        target=target, input_ids=warm_ids, attention_mask=warm_attn,
        eos_token_ids=eos_ids, max_new_tokens=8,
    )
    torch.cuda.empty_cache()

    rows = []
    for B, M in B_M_CONFIGS:
        if B > len(prompts_list):
            # Cycle prompts to reach B samples (with replacement).
            print(f"\n=== B={B} M={M}: cycling {len(prompts_list)} prompts to {B} ===", flush=True)
            prompts_for_B = (prompts_list * ((B // len(prompts_list)) + 1))[:B]
        else:
            prompts_for_B = prompts_list[:B]
        # use prompts_for_B (single chunk of exactly B) for measurement
        print(f"\n=== B={B} M={M} ===", flush=True)
        try:
            torch.cuda.empty_cache()
            # Vanilla baseline
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
            print(f"  vanilla AR: {van_tps:.1f} tok/s", flush=True)

            # v7 DDTree
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
            print(f"  v7 DDTree:  {v7_tps:.1f} tok/s, tau={tau:.2f}, speedup={speedup:.3f}×",
                  flush=True)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM at B={B} M={M}: {e}", flush=True)
            rows.append({"B": B, "M": M, "error": "OOM"})
            torch.cuda.empty_cache()
            break  # if we OOM, larger B will also OOM
        except Exception as e:
            print(f"  ERROR at B={B} M={M}: {e}", flush=True)
            rows.append({"B": B, "M": M, "error": str(e)[:200]})
            torch.cuda.empty_cache()

    print("\n=== large-B v7 summary ===", flush=True)
    print(f"  {'B':>5} {'M':>3} {'vanilla':>10} {'v7':>10} {'tau':>6} {'speedup':>8}", flush=True)
    for r in rows:
        if "error" in r:
            print(f"  {r['B']:>5} {r['M']:>3} {'--':>10} {'--':>10} {'--':>6} {r['error']:>8}", flush=True)
        else:
            print(f"  {r['B']:>5} {r['M']:>3} {r['vanilla_tps']:>9.1f} {r['v7_tps']:>9.1f} "
                  f"{r['tau']:>6.2f} {r['speedup']:>7.3f}×", flush=True)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"results": rows}, f, indent=2)
    print(f"\nWrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
