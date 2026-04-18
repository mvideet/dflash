"""
Regenerate Nemotron-v2 chat/stem/code/math prompts with Qwen3-4B (greedy,
thinking-disabled) and write JSONL in DFlash/ShareGPT training format.

This replaces the two-step SpecForge arrow regen → nemotron_arrow_to_json
path with a single script that goes directly from the HF dataset cache to
DFlash training JSONL.

Output layout:
    <output-dir>/train_nemotron_<split>.jsonl
    <output-dir>/test_nemotron_<split>.jsonl  (if test-ratio > 0)

Example (vllm_gen env with vLLM available, 8×A100):

    python trainingto/regen_nemotron_broad.py \\
        --split chat --max-samples 30000 \\
        --output-dir trainingto/data/nemotron_chat_30k \\
        --tensor-parallel-size 1 --num-gpus 8

Parallelism model: we shard the prompt set across `--num-gpus`, spawn
that many vLLM processes with disjoint CUDA_VISIBLE_DEVICES, and merge
their shard JSONL outputs at the end.
"""

import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path

# Heavy imports are deferred until actually needed so the driver process can
# fork shard-workers without loading vLLM/torch.


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
SPECIAL_TOKEN_RE = re.compile(
    r"<\|(?:start_of_text|end_of_text|eot_id|start_header_id|end_header_id|"
    r"im_start|im_end|endoftext|begin_of_text|pad_id)\|>"
)


def clean_response(text: str) -> str:
    text = THINK_BLOCK_RE.sub("", text)
    text = SPECIAL_TOKEN_RE.sub("", text)
    return text.strip()


def nemotron_to_human_only(example):
    """Extract user turns from a Nemotron-v2 row.

    Returns dict with {id, human_turns} if valid, else None.  Drops the
    Nemotron 235B assistant turns (to be regenerated) and any system
    turns (empty in chat/stem/code/math splits).
    """
    messages = example.get("messages") or []
    human_turns = []
    for m in messages:
        role = (m.get("role") or "").strip().lower()
        content = m.get("content")
        if content is None:
            continue
        content = str(content)
        if role in ("user", "human") and content.strip():
            human_turns.append(content)
    if not human_turns:
        return None
    eid = example.get("uuid") or example.get("id") or ""
    return {"id": str(eid), "human_turns": human_turns}


def load_nemotron_prompts(split, max_samples, seed):
    """Load Nemotron-v2 split from HF cache and extract human-only prompts."""
    from datasets import load_dataset
    ds = load_dataset("nvidia/Nemotron-Post-Training-Dataset-v2", split=split)
    if max_samples is not None and len(ds) > max_samples:
        ds = ds.shuffle(seed=seed).select(range(max_samples))
    out = []
    for i, ex in enumerate(ds):
        row = nemotron_to_human_only(ex)
        if row is not None:
            if not row["id"]:
                row["id"] = f"{split}_{i}"
            out.append(row)
    return out


def regenerate_depth_by_depth(llm, tokenizer, conversations,
                              max_new_tokens=2048, max_prompt_tokens=3072):
    """Batched depth-by-depth regeneration with vLLM.

    At each depth d, all conversations with a d-th user turn are batched.
    Greedy (T=0) + thinking disabled.  Invalid / over-length prompts are
    emitted as empty assistant messages so downstream filtering drops them.
    """
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
    )

    if not conversations:
        return []
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
                s["output_conversations"].append({"from": "gpt", "value": ""})
                continue

            active_indices.append(i)
            prompts.append(prompt)

        if not prompts:
            continue

        print(f"  Depth {depth}: generating {len(prompts)} responses...", flush=True)
        outputs = llm.generate(prompts, sampling_params)
        for idx, output in zip(active_indices, outputs):
            generated = clean_response(output.outputs[0].text)
            states[idx]["messages"].append({"role": "assistant", "content": generated})
            states[idx]["output_conversations"].append({"from": "gpt", "value": generated})

    results = []
    for s in states:
        conv = s["output_conversations"]
        if len(conv) >= 2 and any(
            t["from"] == "gpt" and t["value"].strip() for t in conv
        ):
            results.append({"id": s["id"], "conversations": conv})
    return results


def shard_worker(args):
    """Worker process: loads model, regenerates its shard, writes shard_N.jsonl."""
    from transformers import AutoTokenizer
    from vllm import LLM

    all_prompts = load_nemotron_prompts(args.split, args.max_samples, args.seed)
    prompts = [
        c for i, c in enumerate(all_prompts)
        if i % args.num_gpus == args.gpu_id
    ]
    print(
        f"[shard {args.gpu_id}/{args.num_gpus}] split={args.split} "
        f"prompts={len(prompts)}/{len(all_prompts)}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )

    results = regenerate_depth_by_depth(
        llm, tokenizer, prompts,
        max_new_tokens=args.max_new_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
    )

    shard_path = os.path.join(args.output_dir, f"shard_{args.gpu_id}.jsonl")
    with open(shard_path, "w") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"[shard {args.gpu_id}] wrote {len(results)} records -> {shard_path}",
        flush=True,
    )


def driver(args):
    """Spawn one worker per GPU shard, then merge + train/test split."""
    os.makedirs(args.output_dir, exist_ok=True)

    if args.merge_only:
        _merge_shards(args)
        return

    procs = []
    for gid in range(args.num_gpus):
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--split", args.split,
            "--output-dir", args.output_dir,
            "--model", args.model,
            "--tensor-parallel-size", str(args.tensor_parallel_size),
            "--num-gpus", str(args.num_gpus),
            "--gpu-id", str(gid),
            "--max-new-tokens", str(args.max_new_tokens),
            "--max-prompt-tokens", str(args.max_prompt_tokens),
            "--max-model-len", str(args.max_model_len),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--seed", str(args.seed),
        ]
        if args.max_samples is not None:
            cmd += ["--max-samples", str(args.max_samples)]

        env = os.environ.copy()
        # pin each shard worker to its own GPU block
        start = gid * args.tensor_parallel_size
        end = start + args.tensor_parallel_size
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(start, end))
        print(
            f"[driver] launching shard {gid} on GPUs "
            f"{env['CUDA_VISIBLE_DEVICES']}",
            flush=True,
        )
        procs.append(subprocess.Popen(cmd, env=env))

    ret = [p.wait() for p in procs]
    if any(r != 0 for r in ret):
        print(f"[driver] shard returns: {ret}", flush=True)
        raise SystemExit(1)

    _merge_shards(args)


def _merge_shards(args):
    all_rows = []
    for gid in range(args.num_gpus):
        shard_path = os.path.join(args.output_dir, f"shard_{gid}.jsonl")
        if not os.path.exists(shard_path):
            print(f"[merge] WARNING: missing {shard_path}")
            continue
        with open(shard_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                all_rows.append(json.loads(line))
    random.seed(args.seed)
    random.shuffle(all_rows)
    if args.test_ratio > 0 and len(all_rows) > 1:
        n_test = max(1, int(len(all_rows) * args.test_ratio))
        test_rows, train_rows = all_rows[:n_test], all_rows[n_test:]
    else:
        train_rows, test_rows = all_rows, []

    train_path = os.path.join(args.output_dir, f"train_nemotron_{args.split}.jsonl")
    with open(train_path, "w") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[merge] wrote {len(train_rows)} train -> {train_path}")
    if test_rows:
        test_path = os.path.join(args.output_dir, f"test_nemotron_{args.split}.jsonl")
        with open(test_path, "w") as f:
            for r in test_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[merge] wrote {len(test_rows)} test -> {test_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=str, required=True,
                   choices=["chat", "code", "math", "stem"])
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--num-gpus", type=int, default=8)
    p.add_argument("--gpu-id", type=int, default=None,
                   help="Internal flag: shard id when run as worker")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-prompt-tokens", type=int, default=3072)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                   help="vLLM GPU mem fraction. Lower if sharing GPU with "
                        "other jobs (e.g. 0.25 leaves ~60 GB for other user).")
    p.add_argument("--test-ratio", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--merge-only", action="store_true")
    args = p.parse_args()

    if args.gpu_id is not None:
        shard_worker(args)
    else:
        driver(args)


if __name__ == "__main__":
    main()
