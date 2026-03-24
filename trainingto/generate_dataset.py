"""
Generate on-distribution training data for GTO by regenerating assistant
responses using Qwen3-4B with greedy decoding and thinking disabled.

Uses vLLM for fast batched inference. Multi-turn conversations are processed
depth-by-depth: all turn-1 prompts are batched, then turn-2 (using generated
turn-1 responses as context), etc.

Usage (vllm_gen env):
    python generate_dataset.py \
        --model Qwen/Qwen3-4B \
        --max-conversations 5000 \
        --output-dir /homes/videetm/GTO/data \
        --tensor-parallel-size 4
"""

import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import argparse
import json
import random
import re

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
SPECIAL_TOKEN_RE = re.compile(
    r"<\|(?:start_of_text|end_of_text|eot_id|start_header_id|end_header_id|"
    r"im_start|im_end|endoftext|begin_of_text|pad_id)\|>"
)


def clean_response(text: str) -> str:
    text = THINK_BLOCK_RE.sub("", text)
    text = SPECIAL_TOKEN_RE.sub("", text)
    text = text.strip()
    return text


def load_sharegpt(path: str, max_conversations: int | None = None, seed: int = 42):
    if path.startswith("http"):
        from datasets import load_dataset
        ds = load_dataset("json", data_files=path)["train"]
        raw = list(ds)
    else:
        with open(path) as f:
            raw = json.load(f)

    conversations = []
    for item in raw:
        conv_id = item.get("id", f"conv_{len(conversations)}")
        turns = item.get("conversations", [])
        human_turns = [t["value"] for t in turns if t.get("from") in ("human", "user")]
        if human_turns:
            conversations.append({"id": conv_id, "human_turns": human_turns})

    random.seed(seed)
    random.shuffle(conversations)
    if max_conversations is not None:
        conversations = conversations[:max_conversations]
    return conversations


def regenerate_depth_by_depth(llm, tokenizer, conversations,
                              max_new_tokens=2048, max_prompt_tokens=3072):
    """
    Process multi-turn conversations depth-by-depth.
    At each depth d, batch all conversations that have a d-th human turn.
    """
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
    )

    max_depth = max(len(c["human_turns"]) for c in conversations)

    states = []
    for conv in conversations:
        states.append({
            "id": conv["id"],
            "human_turns": conv["human_turns"],
            "messages": [],
            "output_conversations": [],
        })

    for depth in range(max_depth):
        active_indices = []
        prompts = []

        for i, s in enumerate(states):
            if depth >= len(s["human_turns"]):
                continue

            human_text = s["human_turns"][depth]
            s["messages"].append({"role": "user", "content": human_text})
            s["output_conversations"].append({"from": "human", "value": human_text})

            prompt = tokenizer.apply_chat_template(
                s["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            tok_len = len(tokenizer.encode(prompt, add_special_tokens=False))
            if tok_len > max_prompt_tokens:
                s["messages"].append({"role": "assistant", "content": ""})
                s["output_conversations"].append({"from": "assistant", "value": ""})
                continue

            active_indices.append(i)
            prompts.append(prompt)

        if not prompts:
            continue

        print(f"  Depth {depth}: generating {len(prompts)} responses...")
        outputs = llm.generate(prompts, sampling_params)

        for idx, output in zip(active_indices, outputs):
            generated = clean_response(output.outputs[0].text)
            states[idx]["messages"].append({"role": "assistant", "content": generated})
            states[idx]["output_conversations"].append({"from": "assistant", "value": generated})

    results = []
    for s in states:
        conv = s["output_conversations"]
        if len(conv) >= 2 and any(t["from"] == "assistant" and t["value"] for t in conv):
            results.append({"id": s["id"], "conversations": conv})
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate on-distribution GTO training data")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--data-path", type=str,
                        default="https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V4.3_unfiltered_cleaned_split.json")
    parser.add_argument("--max-conversations", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="/homes/videetm/GTO/data")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=None,
                        help="Process only this shard (for multi-GPU parallelism)")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Total number of GPU shards")
    parser.add_argument("--merge-only", action="store_true",
                        help="Only merge existing shard files into train/test")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.merge_only:
        all_results = []
        for gid in range(args.num_gpus):
            p = os.path.join(args.output_dir, f"shard_{gid}.json")
            if os.path.exists(p):
                with open(p) as f:
                    all_results.extend(json.load(f))
        print(f"Merged {len(all_results)} conversations from {args.num_gpus} shards")
        random.shuffle(all_results)
        split_idx = int(len(all_results) * (1 - args.test_ratio))
        train_path = os.path.join(args.output_dir, "train_qwen3.json")
        test_path = os.path.join(args.output_dir, "test_qwen3.json")
        with open(train_path, "w") as f:
            json.dump(all_results[:split_idx], f, ensure_ascii=False)
        with open(test_path, "w") as f:
            json.dump(all_results[split_idx:], f, ensure_ascii=False)
        print(f"Saved {split_idx} train -> {train_path}")
        print(f"Saved {len(all_results) - split_idx} test -> {test_path}")
        return

    print("Loading ShareGPT conversations...")
    all_conversations = load_sharegpt(args.data_path, args.max_conversations, args.seed)

    if args.gpu_id is not None:
        conversations = [c for i, c in enumerate(all_conversations) if i % args.num_gpus == args.gpu_id]
        print(f"[GPU {args.gpu_id}] Shard: {len(conversations)} / {len(all_conversations)} conversations")
    else:
        conversations = all_conversations
        print(f"Loaded {len(conversations)} conversations")

    print(f"Loading {args.model} with tensor_parallel_size={args.tensor_parallel_size}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        trust_remote_code=True,
    )
    print("Model loaded.")

    print("Regenerating assistant responses (depth-by-depth)...")
    results = regenerate_depth_by_depth(
        llm, tokenizer, conversations,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Valid conversations: {len(results)}")

    if args.gpu_id is not None:
        shard_path = os.path.join(args.output_dir, f"shard_{args.gpu_id}.json")
        with open(shard_path, "w") as f:
            json.dump(results, f, ensure_ascii=False)
        print(f"[GPU {args.gpu_id}] Saved {len(results)} -> {shard_path}")
    else:
        random.seed(args.seed)
        random.shuffle(results)
        split_idx = int(len(results) * (1 - args.test_ratio))
        train_path = os.path.join(args.output_dir, "train_qwen3.json")
        test_path = os.path.join(args.output_dir, "test_qwen3.json")
        with open(train_path, "w") as f:
            json.dump(results[:split_idx], f, ensure_ascii=False)
        with open(test_path, "w") as f:
            json.dump(results[split_idx:], f, ensure_ascii=False)
        print(f"Saved {split_idx} train -> {train_path}")
        print(f"Saved {len(results) - split_idx} test -> {test_path}")


if __name__ == "__main__":
    main()
