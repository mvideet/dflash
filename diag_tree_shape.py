"""At B=8 M=16, does the heap actually build a tree or a chain?

For each round, count: total tree nodes, max depth reached, leaves count.
A degenerate chain has: N=16, max_depth=15, n_leaves=1.
A real tree has: N=16, max_depth<15, n_leaves>1.
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter

from model import DFlashDraftModel, load_and_process_dataset
from dflash_batched import dflash_generate_batched, vanilla_ar_generate_batched
from benchmark_batched import tokenize_prompts, make_padded_batch, chunk_rows_list, _resolve_mts


SCHEDULE = {1: 64, 2: 32, 4: 32, 8: 16, 16: 16, 32: 16}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
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
        max_prompt_tokens=256, device=device,
    )

    # Hook into the tree-build to dump shape stats. Easiest: monkeypatch via
    # global counter. Since we can't easily inject without changing the generator,
    # do something simpler: change kw to also set log_phase_timings=True so we get
    # all data, then read tree_node_counts from output.
    for B in [1, 4, 8]:
        M = _resolve_mts(B, 16, SCHEDULE)
        kw = dict(specdecpp_threshold=0.05) if B == 8 else (
            dict(ewma_adaptive=True, ewma_min_M=8, ewma_max_ek=8, ewma_min_ek=2) if B == 4 else dict()
        )
        node_counts = []
        for chunk in chunk_rows_list(prompts_list, B):
            ids, attn = make_padded_batch(chunk, pad_id, device)
            s_out = dflash_generate_batched(
                draft=draft, target=target, input_ids=ids, attention_mask=attn,
                mask_token_id=mid, eos_token_ids=eos_ids,
                max_new_tokens=args.max_new_tokens, block_size=block_size,
                max_tree_size=M, expand_k=8, temperature=0.0, **kw,
            )
            node_counts.extend(s_out.tree_node_counts)
            del s_out, ids, attn
            torch.cuda.empty_cache()
        c = Counter(node_counts)
        print(f"\nB={B} M={M}: tree_node_count distribution (avg = mean nodes per round)")
        for k in sorted(c.keys()):
            print(f"  nodes={k}: {c[k]} rounds")


if __name__ == "__main__":
    main()
