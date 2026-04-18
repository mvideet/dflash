"""Merge per-split JSONLs into a single broad-mix train/test.

Example:
    python trainingto/mix_split_jsonls.py \\
        --inputs \\
            trainingto/data/nemotron_math_10k/train.jsonl:1.0 \\
            trainingto/data/nemotron_chat_30k/train_nemotron_chat.jsonl:1.0 \\
            trainingto/data/nemotron_stem_30k/train_nemotron_stem.jsonl:1.0 \\
        --output-dir trainingto/data/nemotron_broad
"""

import argparse
import json
import os
import random


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inputs", nargs="+", required=True,
        help="List of path[:weight]. Weight = fraction of that split to keep "
             "(default 1.0). Weights <1 subsample via random shuffle."
    )
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--output-name", type=str, default="train")
    p.add_argument("--test-ratio", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    all_rows = []
    for spec in args.inputs:
        if ":" in spec:
            path, w = spec.rsplit(":", 1)
            weight = float(w)
        else:
            path, weight = spec, 1.0
        rows = read_jsonl(path)
        if weight < 1.0:
            random.shuffle(rows)
            k = int(len(rows) * weight)
            rows = rows[:k]
        print(f"{path}: {len(rows)} rows (weight={weight})")
        all_rows.extend(rows)

    random.shuffle(all_rows)
    n_test = max(1, int(len(all_rows) * args.test_ratio)) if args.test_ratio > 0 else 0
    test_rows, train_rows = all_rows[:n_test], all_rows[n_test:]

    train_path = os.path.join(args.output_dir, f"{args.output_name}.jsonl")
    with open(train_path, "w") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(train_rows)} train -> {train_path}")
    if test_rows:
        test_path = os.path.join(args.output_dir, f"test.jsonl")
        with open(test_path, "w") as f:
            for r in test_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(test_rows)} test -> {test_path}")


if __name__ == "__main__":
    main()
