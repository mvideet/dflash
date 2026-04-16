"""
Convert a Hugging Face Dataset on disk (Arrow shards, e.g. SpecForge nemotron_regen/math)
into JSON / JSONL files compatible with DFlash training scripts (main_lk.py, main_gto.py).

Those scripts use ``load_dataset("json", data_files=...)`` and expect each example to have:
  - ``id``: string
  - ``conversations``: list of ``{"from": ..., "value": ...}`` where ``from`` is one of
    ``human``, ``gpt``, ``assistant``, ``user``, ``system`` (see ``build_dataset`` in main_lk.py).

Nemotron-style ``messages`` (``role`` / ``content``) are mapped to ShareGPT-style turns:
  user -> human, assistant -> gpt, system -> system.

Default output is JSONL (one JSON object per line), which ``datasets`` loads the same way as
``--trainpath`` / ``--testpath`` and scales to large corpora without holding everything in RAM.

Example (conda env that has ``datasets`` installed, e.g. ``dflash``):

    python nemotron_arrow_to_json.py \\
        --input /homes/videetm/SpecForge/nemotron_regen/math \\
        --output-dir /homes/videetm/dflash/trainingto/data/nemotron_math \\
        --test-ratio 0.01 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from datasets import Dataset, load_from_disk


def messages_to_conversations(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Map chat messages to ShareGPT-style turns for DFlash preprocessing."""
    conv: list[dict[str, str]] = []
    for m in messages:
        role = (m.get("role") or "").strip().lower()
        content = m.get("content")
        if content is None:
            continue
        content = str(content)
        if role in ("user", "human"):
            conv.append({"from": "human", "value": content})
        elif role in ("assistant", "model"):
            conv.append({"from": "gpt", "value": content})
        elif role == "system":
            conv.append({"from": "system", "value": content})
        # skip unknown roles (e.g. tool) for math splits
    return conv


def convert_example(example: dict[str, Any], fallback_id: str) -> dict[str, Any] | None:
    messages = example.get("messages") or []
    conv = messages_to_conversations(messages)
    if len(conv) < 2:
        return None
    if not any(t["from"] == "gpt" and t["value"].strip() for t in conv):
        return None
    eid = example.get("uuid") or example.get("id") or fallback_id
    return {"id": str(eid), "conversations": conv}


def train_test_split_ds(ds: Dataset, test_ratio: float, seed: int) -> tuple[Dataset, Dataset]:
    if test_ratio <= 0.0:
        return ds, ds.select([])
    if test_ratio >= 1.0:
        return ds.select([]), ds
    shuffled = ds.shuffle(seed=seed)
    n = len(shuffled)
    if n == 0:
        return shuffled, shuffled
    n_test = int(n * test_ratio)
    if n_test >= n:
        n_test = n - 1
    if n_test < 1:
        n_test = 1 if n > 1 else 0
    if n_test <= 0:
        return shuffled, shuffled.select([])
    return shuffled.select(range(n_test, n)), shuffled.select(range(n_test))


def _filter_keep_row(example: dict[str, Any]) -> bool:
    return bool(example["_keep"])


def write_json_array(path: str, ds: Dataset) -> None:
    """Write a single JSON array file (fine for small exports)."""
    rows = [{"id": r["id"], "conversations": r["conversations"]} for r in ds]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Nemotron Arrow dataset -> DFlash JSON/JSONL")
    p.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path from load_from_disk (directory with .arrow shards + dataset_info.json)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="./data/nemotron_math",
        help="Directory for train/test output files",
    )
    p.add_argument(
        "--test-ratio",
        type=float,
        default=0.01,
        help="Fraction held out for test (0 disables test file)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--format",
        choices=("jsonl", "json"),
        default="jsonl",
        help="jsonl (recommended): one record per line. json: single array (memory-heavy)",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="If set, only convert the first N rows after shuffle (for smoke tests)",
    )
    p.add_argument(
        "--no-filter",
        action="store_true",
        help="Keep rows even if conversion yields <2 turns or empty assistant (not recommended)",
    )
    p.add_argument("--num-proc", type=int, default=8, help="Workers for datasets.map / filter")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ds = load_from_disk(args.input)
    if args.max_samples is not None:
        ds = ds.shuffle(seed=args.seed).select(range(min(args.max_samples, len(ds))))

    def map_row(example: dict[str, Any], idx: int) -> dict[str, Any]:
        if args.no_filter:
            messages = example.get("messages") or []
            conv = messages_to_conversations(messages)
            eid = example.get("uuid") or example.get("id") or f"row_{idx}"
            return {"id": str(eid), "conversations": conv, "_keep": True}
        out = convert_example(example, f"row_{idx}")
        if out is None:
            return {"id": "", "conversations": [], "_keep": False}
        return {"id": out["id"], "conversations": out["conversations"], "_keep": True}

    flat = ds.map(
        map_row,
        with_indices=True,
        remove_columns=ds.column_names,
        load_from_cache_file=False,
        num_proc=args.num_proc,
    )
    if not args.no_filter:
        flat = flat.filter(_filter_keep_row, num_proc=args.num_proc)
    flat = flat.remove_columns(["_keep"])

    train_ds, test_ds = train_test_split_ds(flat, args.test_ratio, args.seed)

    train_path = os.path.join(args.output_dir, f"train_nemotron_math.{args.format}")
    test_path = os.path.join(args.output_dir, f"test_nemotron_math.{args.format}")

    if args.format == "jsonl":
        train_ds.to_json(train_path, lines=True)
        if len(test_ds) > 0:
            test_ds.to_json(test_path, lines=True)
    else:
        write_json_array(train_path, train_ds)
        if len(test_ds) > 0:
            write_json_array(test_path, test_ds)

    print(f"Wrote {len(train_ds)} train -> {train_path}")
    if len(test_ds) > 0:
        print(f"Wrote {len(test_ds)} test -> {test_path}")
    else:
        print("No test split written (test_ratio==0 or empty train).")


if __name__ == "__main__":
    main()
