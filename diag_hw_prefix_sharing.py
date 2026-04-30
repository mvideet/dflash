"""HW Idea 5 feasibility: how much of math500 prompts share a system prefix?

The prefix-sharing payoff comes from common tokens at the start of every prompt.
Measure the longest common prefix (LCP) across math500 prompts, expressed as:
  - LCP token length
  - LCP fraction of average prompt length
  - Memory savings if shared across batch B

Result tells us: at B=128, what fraction of KV cache could be merged into one
shared page? This is the upper bound on Idea 5's memory unlock.
"""
import json, os
import numpy as np
from transformers import AutoTokenizer

from model import load_and_process_dataset


def lcp(prompts):
    """Longest common prefix of a list of token-id sequences."""
    if not prompts: return 0
    min_len = min(len(p) for p in prompts)
    n_match = 0
    for i in range(min_len):
        if all(p[i] == prompts[0][i] for p in prompts):
            n_match += 1
        else:
            break
    return n_match


def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    out = {}
    for ds_name in ["math500", "aime24", "gsm8k"]:
        try:
            ds = load_and_process_dataset(ds_name)
        except Exception as e:
            print(f"{ds_name}: {e}"); continue

        # Tokenize each prompt's "turns[0]"
        all_prompts = []
        for i in range(min(len(ds), 100)):
            text = ds[i]["turns"][0]
            ids = tokenizer.encode(text, add_special_tokens=True)
            all_prompts.append(ids)

        prompt_lens = [len(p) for p in all_prompts]
        n_lcp = lcp(all_prompts)
        n_lcp_first10 = lcp(all_prompts[:10])

        # Pairwise LCP (averaged)
        pairwise = []
        for i in range(min(len(all_prompts), 50)):
            for j in range(i+1, min(len(all_prompts), 50)):
                pairwise.append(lcp([all_prompts[i], all_prompts[j]]))

        # Estimate at B=128: if we use the LCP across the batch as the "shared page",
        # what's the saving?
        avg_len = np.mean(prompt_lens)
        full_lcp_pct = n_lcp / avg_len * 100
        pair_lcp_avg = np.mean(pairwise) if pairwise else 0
        pair_lcp_pct = pair_lcp_avg / avg_len * 100

        # KV-cache memory savings at B=128:
        # without sharing: 128 × avg_len × 32 layers × 32 heads × 128 dim × 2 (K+V) × 2B
        # with sharing on LCP: (128-1) × LCP saved + 128 × (avg_len - LCP) tail
        # saving = 127 × LCP × layer-bytes
        layer_bytes = 32 * 32 * 128 * 2 * 2  # 32 layers × 32 heads × 128 dim × K+V × bf16
        # But model only has 32 layers once; KV is per-layer.
        # KV cache = num_layers × num_heads × seq_len × head_dim × 2 (K+V) × dtype_bytes
        kv_bytes_per_token = 32 * 32 * 128 * 2 * 2  # ~524 KB per token across all layers
        kv_per_token_kb = kv_bytes_per_token / 1024

        # at B=128 avg_len without sharing
        baseline_mb = 128 * avg_len * kv_bytes_per_token / 1024 / 1024
        # with sharing on full LCP (if all prompts share it)
        shared_mb = (1 + 127 * (avg_len - n_lcp) / avg_len) * avg_len * kv_bytes_per_token / 1024 / 1024
        # alternative interpretation: shared page = LCP × kv_per_token, plus 128 × (len - LCP)
        shared_mb_v2 = (n_lcp + 128 * (avg_len - n_lcp)) * kv_bytes_per_token / 1024 / 1024
        savings_mb = baseline_mb - shared_mb_v2
        savings_pct = savings_mb / baseline_mb * 100

        print(f"\n=== {ds_name} ===")
        print(f"  N prompts measured: {len(all_prompts)}")
        print(f"  Avg prompt length: {avg_len:.1f} tokens")
        print(f"  LCP across all {len(all_prompts)} prompts: {n_lcp} tokens ({full_lcp_pct:.1f}%)")
        print(f"  LCP across first 10: {n_lcp_first10} tokens")
        print(f"  Pairwise LCP avg: {pair_lcp_avg:.1f} tokens ({pair_lcp_pct:.1f}%)")
        print(f"  KV bytes per token (32-layer × 32-head × 128-dim × K+V × bf16): "
              f"{kv_per_token_kb:.1f} KB")
        print(f"  KV memory at B=128, len={avg_len:.0f}: baseline={baseline_mb:.0f} MB,")
        print(f"                                          shared on LCP={shared_mb_v2:.0f} MB,")
        print(f"                                          saving={savings_mb:.0f} MB ({savings_pct:.1f}%)")
        if n_lcp >= 20:
            print(f"  → Sharing the {n_lcp}-token LCP across batch saves {savings_pct:.1f}% of KV memory")

        # Show the LCP text:
        if n_lcp > 0:
            shared_text = tokenizer.decode(all_prompts[0][:n_lcp])
            print(f"  LCP text: {shared_text!r}")

        out[ds_name] = {
            "n_prompts": len(all_prompts),
            "avg_len": float(avg_len),
            "lcp_all": n_lcp,
            "lcp_first10": n_lcp_first10,
            "lcp_pairwise_avg": float(pair_lcp_avg),
            "lcp_pct_of_avg": float(full_lcp_pct),
            "kv_baseline_mb_b128": float(baseline_mb),
            "kv_shared_mb_b128": float(shared_mb_v2),
            "kv_savings_mb_b128": float(savings_mb),
            "kv_savings_pct_b128": float(savings_pct),
        }

    os.makedirs("logs", exist_ok=True)
    with open("logs/hw_prefix_sharing.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote logs/hw_prefix_sharing.json")


if __name__ == "__main__":
    main()
