#!/usr/bin/env python3
"""
Sanity check: run DFlash speculative decoding and print actual prompts + outputs.
Verifies the tree mechanism produces coherent text.

Losslessness check (temp=0): vanilla and DFlash must produce identical outputs.
Usage:
  python sanity_check_decode.py --dataset gsm8k --max-samples 3
  python sanity_check_decode.py --verify-lossless --dataset gsm8k --max-samples 20
  python sanity_check_decode.py --verify-lossless-no-tree --dataset gsm8k  # vanilla vs DFlash without tree
"""
import argparse
import os
import random

# Avoid PyTorch logging init error (get_log_level_pairs)
os.environ.pop("TORCH_LOGS", None)

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Single-GPU: avoid torchrun / distributed
os.environ.pop("RANK", None)
os.environ.pop("WORLD_SIZE", None)
os.environ.pop("LOCAL_RANK", None)
os.environ.pop("LOCAL_WORLD_SIZE", None)

from benchmark import dflash_generate
from model import DFlashDraftModel, load_and_process_dataset


def main():
    parser = argparse.ArgumentParser(description="Sanity check: print prompts and decoded outputs")
    parser.add_argument("--model-name-or-path", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--draft-name-or-path", type=str, default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dynamic-branching", action="store_true", default=True)
    parser.add_argument("--no-dynamic-branching", dest="dynamic_branching", action="store_false")
    parser.add_argument("--theta-uni", type=float, default=0.9)
    parser.add_argument("--theta-bi", type=float, default=0.3)
    parser.add_argument("--theta-tri", type=float, default=0.1)
    parser.add_argument("--max-tree-size", type=int, default=8)
    parser.add_argument("--vanilla-only", action="store_true", help="Run only vanilla (block_size=1) for comparison")
    parser.add_argument("--verify-lossless", action="store_true",
                        help="Verify losslessness: compare vanilla vs speculative (with tree) at temp=0.")
    parser.add_argument("--verify-lossless-no-tree", action="store_true",
                        help="Verify losslessness: compare vanilla vs DFlash without tree (sequential draft path) at temp=0.")
    parser.add_argument("--debug-mismatch", action="store_true",
                        help="When mismatch: print token IDs and decoded text around first divergence.")
    args = parser.parse_args()

    args.verify_lossless = args.verify_lossless or args.verify_lossless_no_tree
    if args.verify_lossless:
        args.temperature = 0.0

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    device = torch.device("cuda:0")

    def has_flash_attn():
        try:
            import flash_attn
            return True
        except ImportError:
            return False

    attn_impl = "flash_attention_2" if has_flash_attn() else "sdpa"
    print(f"Loading target model: {args.model_name_or_path} (attn={attn_impl})")
    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
    ).to(device).eval()

    print(f"Loading draft model: {args.draft_name_or_path}")
    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
    ).to(device).eval()

    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_and_process_dataset(args.dataset)

    if len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    print(f"\n{'='*70}")
    if args.verify_lossless:
        mode = "no-tree" if args.verify_lossless_no_tree else "tree"
        print(f"Losslessness verification ({mode}): {args.dataset}, {len(dataset)} samples, temp=0 (deterministic)")
        if args.verify_lossless_no_tree:
            print("Vanilla (block_size=1) vs DFlash without tree (block_size=16, sequential draft) must produce identical token sequences.")
        else:
            print("Vanilla (block_size=1) vs Speculative (with tree) must produce identical token sequences.")
    else:
        print(f"Sanity check: {args.dataset}, {len(dataset)} samples, block_size={block_size}")
    if args.verify_lossless_no_tree:
        print("Mode: DFlash sequential draft (no tree)")
    else:
        print(f"Dynamic branching: {args.dynamic_branching}, max_tree_size={args.max_tree_size}")
    print(f"{'='*70}\n")

    matches = 0
    mismatches = []

    for idx, instance in enumerate(dataset):
        messages = [{"role": "user", "content": c} for c in instance["turns"]]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        prompt_text = instance["turns"][0]
        if not args.verify_lossless:
            print(f"\n--- Sample {idx + 1} ---")
            print(f"PROMPT ({len(prompt_text)} chars):")
            print("-" * 40)
            print(prompt_text[:500] + ("..." if len(prompt_text) > 500 else ""))
            print("-" * 40)

        # Vanilla (block_size=1) baseline — always run when comparing
        if args.vanilla_only or block_size > 1 or args.verify_lossless:
            if not args.verify_lossless:
                print("\n[VANILLA block_size=1]")
            vanilla = dflash_generate(
                model=draft_model,
                target=target,
                input_ids=input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=args.max_new_tokens,
                block_size=1,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
                chain_attention=False,
                dynamic_branching=False,
            )
            vanilla_text = tokenizer.decode(
                vanilla.output_ids[0, vanilla.num_input_tokens:],
                skip_special_tokens=True,
            )
            if not args.verify_lossless:
                print(f"OUTPUT ({vanilla.num_output_tokens} tokens):")
                print(vanilla_text[:800] + ("..." if len(vanilla_text) > 800 else ""))
                print(f"  (time/token: {vanilla.time_per_output_token*1000:.2f} ms)")

        # Speculative: with tree or without (sequential draft path)
        if not args.vanilla_only or args.verify_lossless:
            use_tree = args.dynamic_branching and not args.verify_lossless_no_tree
            if not args.verify_lossless:
                label = "SPECULATIVE (no tree)" if args.verify_lossless_no_tree else "SPECULATIVE (with tree)"
                print(f"\n[{label} block_size={block_size}]")
            spec = dflash_generate(
                model=draft_model,
                target=target,
                input_ids=input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=args.max_new_tokens,
                block_size=block_size,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
                chain_attention=use_tree,
                dynamic_branching=use_tree,
                theta_uni=args.theta_uni,
                theta_bi=args.theta_bi,
                theta_tri=args.theta_tri,
                max_tree_size=args.max_tree_size,
            )
            spec_text = tokenizer.decode(
                spec.output_ids[0, spec.num_input_tokens:],
                skip_special_tokens=True,
            )
            avg_acc = sum(spec.acceptance_lengths) / len(spec.acceptance_lengths) if spec.acceptance_lengths else 0

            # Losslessness check: compare token sequences
            if args.verify_lossless:
                v_ids = vanilla.output_ids[0, vanilla.num_input_tokens:]
                s_ids = spec.output_ids[0, spec.num_input_tokens:]
                v_len, s_len = v_ids.shape[0], s_ids.shape[0]
                min_len = min(v_len, s_len)
                if v_len == s_len and (v_ids == s_ids).all():
                    matches += 1
                    print(f"  Sample {idx + 1}: MATCH ({vanilla.num_output_tokens} tokens)")
                else:
                    diff_mask = (v_ids[:min_len] != s_ids[:min_len])
                    diff_pos = diff_mask.nonzero(as_tuple=True)[0]
                    first_diff = int(diff_pos[0]) if len(diff_pos) > 0 else min_len
                    mismatches.append((idx + 1, first_diff, v_len, s_len))
                    print(f"  Sample {idx + 1}: MISMATCH at token {first_diff} | vanilla={v_len} spec={s_len}")
                    if args.debug_mismatch:
                        ctx = 5
                        lo, hi = max(0, first_diff - ctx), min(min_len, first_diff + ctx + 1)
                        print(f"    Tokens [{lo}:{hi}] vanilla: {v_ids[lo:hi].tolist()}")
                        print(f"    Tokens [{lo}:{hi}] spec:    {s_ids[lo:hi].tolist()}")
                        print(f"    Decoded vanilla: {repr(tokenizer.decode(v_ids[lo:hi]))}")
                        print(f"    Decoded spec:    {repr(tokenizer.decode(s_ids[lo:hi]))}")
            else:
                print(f"OUTPUT ({spec.num_output_tokens} tokens):")
                print(spec_text[:800] + ("..." if len(spec_text) > 800 else ""))
                print(f"  avg acceptance: {avg_acc:.2f}, time/token: {spec.time_per_output_token*1000:.2f} ms")
                print(f"  acceptance lengths: {spec.acceptance_lengths[:20]}{'...' if len(spec.acceptance_lengths) > 20 else ''}")

    if args.verify_lossless:
        total = len(dataset)
        mode = "DFlash no-tree" if args.verify_lossless_no_tree else "Speculative (tree)"
        print(f"\n{'='*70}")
        print(f"Losslessness result: {matches}/{total} samples match")
        if matches == total:
            print(f"PASS: {mode} is lossless (identical to vanilla at temp=0).")
        else:
            print(f"FAIL: {total - matches} mismatch(es). First divergence:")
            for midx, pos, vlen, slen in mismatches[:5]:
                print(f"  Sample {midx}: first diff at token {pos} (vanilla len={vlen}, spec len={slen})")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'='*70}")
        print("Done. If outputs look coherent, the tree mechanism is working correctly.")
        print("Use --verify-lossless (tree) or --verify-lossless-no-tree to confirm losslessness at temp=0.")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
