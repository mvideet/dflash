"""γ sweep × datasets: does each dataset have a sweet-spot γ?"""
import argparse, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import DFlashDraftModel, load_and_process_dataset
from dflash_batched import dflash_generate_batched, vanilla_ar_generate_batched
from benchmark_batched import tokenize_prompts, make_padded_batch, chunk_rows_list, _resolve_mts

SCHEDULE = {1: 64, 2: 32, 4: 32, 8: 16, 16: 16, 32: 16}
DATASETS = ["math500", "gsm8k", "humaneval"]
GAMMAS = [1.0, 0.95, 0.85, 0.75, 0.65, 0.55]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--output-json", type=str, default="logs/gamma_cross.json")
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

    rows = []
    for ds_name in DATASETS:
        print(f"\n========== Dataset: {ds_name} ==========", flush=True)
        try:
            ds = load_and_process_dataset(ds_name).shuffle(seed=0)
        except Exception as e:
            print(f"  Failed: {e}"); continue
        raw = [ds[i]["turns"][0] for i in range(min(len(ds), args.max_samples * 4))]
        prompts_list = tokenize_prompts(
            raw, tokenizer, max_samples=args.max_samples,
            max_prompt_tokens=args.max_prompt_tokens, device=device,
        )
        if not prompts_list: continue

        warm_ids, warm_attn = make_padded_batch(prompts_list[:1], pad_id, device)
        _ = vanilla_ar_generate_batched(
            target=target, input_ids=warm_ids, attention_mask=warm_attn,
            eos_token_ids=eos_ids, max_new_tokens=8,
        )
        torch.cuda.empty_cache()

        B = 4
        M = _resolve_mts(B, 16, SCHEDULE)
        base_kw = dict(ewma_adaptive=True, ewma_min_M=8, ewma_max_ek=8, ewma_min_ek=2)
        van_t, van_o = 0.0, 0
        for chunk in chunk_rows_list(prompts_list, B):
            ids, attn = make_padded_batch(chunk, pad_id, device)
            v_out = vanilla_ar_generate_batched(
                target=target, input_ids=ids, attention_mask=attn,
                eos_token_ids=eos_ids, max_new_tokens=args.max_new_tokens,
            )
            van_t += v_out.total_decode_time
            van_o += sum(v_out.num_output_tokens)
            del v_out, ids, attn; torch.cuda.empty_cache()
        van_tps = van_o / van_t

        for gamma in GAMMAS:
            torch.manual_seed(0)
            v7_t, v7_o, accs = 0.0, 0, []
            for chunk in chunk_rows_list(prompts_list, B):
                ids, attn = make_padded_batch(chunk, pad_id, device)
                s_out = dflash_generate_batched(
                    draft=draft, target=target, input_ids=ids, attention_mask=attn,
                    mask_token_id=mid, eos_token_ids=eos_ids,
                    max_new_tokens=args.max_new_tokens, block_size=block_size,
                    max_tree_size=M, expand_k=8, temperature=0.0,
                    score_decay=gamma, **base_kw,
                )
                v7_t += s_out.total_decode_time
                v7_o += sum(s_out.num_output_tokens)
                for lst in s_out.acceptance_lengths_per_elem:
                    accs.extend(lst)
                del s_out, ids, attn; torch.cuda.empty_cache()
            v7_tps = v7_o / v7_t
            tau = float(np.mean(accs)) if accs else float("nan")
            speedup = v7_tps / van_tps
            rows.append({"dataset": ds_name, "gamma": gamma, "B": B, "M": M,
                         "v7_tps": round(v7_tps, 1), "vanilla_tps": round(van_tps, 1),
                         "speedup": round(speedup, 3), "tau": round(tau, 3)})
            print(f"  γ={gamma}: tau={tau:>5.2f} speedup={speedup:>5.3f}× tps={v7_tps:.0f}",
                  flush=True)

    print("\n=== Speedup matrix (rows=dataset, cols=γ) ===", flush=True)
    print(f"  {'dataset':>11}  " + "  ".join(f"γ={g:>4}" for g in GAMMAS), flush=True)
    for ds_name in DATASETS:
        line = f"  {ds_name:>11}  "
        for g in GAMMAS:
            r = next((r for r in rows if r['dataset'] == ds_name and r['gamma'] == g), None)
            if r:
                line += f"{r['speedup']:>6.3f}  "
        print(line, flush=True)

    print("\n=== Gain over γ=1.0 baseline ===", flush=True)
    print(f"  {'dataset':>11}  " + "  ".join(f"γ={g:>4}" for g in GAMMAS), flush=True)
    for ds_name in DATASETS:
        baseline = next((r for r in rows if r['dataset'] == ds_name and r['gamma'] == 1.0), None)
        if baseline is None: continue
        line = f"  {ds_name:>11}  "
        for g in GAMMAS:
            r = next((r for r in rows if r['dataset'] == ds_name and r['gamma'] == g), None)
            if r:
                gain = 100 * (r['speedup'] - baseline['speedup']) / baseline['speedup']
                line += f"{gain:>+5.1f}%  "
        print(line, flush=True)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"results": rows}, f, indent=2)
    print(f"\nWrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
