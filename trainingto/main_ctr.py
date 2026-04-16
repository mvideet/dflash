"""
DFlash CTR (Conditional Tree Refinement) training script (DeepSpeed).

Two-pass training: standard bidirectional CE + tree-attention conditional CE.
Teaches the draft model to produce good predictions under both attention patterns.

Usage:
    deepspeed --num_gpus=N main_ctr.py \
        --basepath Qwen/Qwen3-4B \
        --draftpath z-lab/Qwen3-4B-DFlash-b16 \
        --trainpath data/nemotron_math/train_nemotron_math.jsonl \
        --testpath  data/nemotron_math/test_nemotron_math.jsonl \
        --savedir   dflash_ctr_checkpoints \
        --deepspeed_config ds_config.json
"""

import argparse
import json
import os
import re

import datetime
import deepspeed
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

# ---------- CLI ----------
parser = argparse.ArgumentParser(description="DFlash CTR training")
parser.add_argument("--basepath", type=str, required=True, help="Target model path")
parser.add_argument("--draftpath", type=str, required=True, help="Draft model path")
parser.add_argument("--trainpath", type=str, required=True, help="Training data path")
parser.add_argument("--testpath", type=str, required=True, help="Test data path")
parser.add_argument("--savedir", type=str, default="./dflash_ctr_checkpoints")
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--save-every", type=int, default=200, help="Save checkpoint every N batches")
parser.add_argument("--ctr-weight", type=float, default=0.5, help="Weight for CTR tree-attention loss")
parser.add_argument("--max-tree-size", type=int, default=16, help="Max tree nodes for CTR trees")
parser.add_argument("--tree-expand-k", type=int, default=5, help="Top-K branching for tree building")
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

with open(args.deepspeed_config) as f:
    ds_config = json.load(f)

BATCH_SIZE = ds_config["train_micro_batch_size_per_gpu"]
MAX_LEN = 2048


# ---------- Dataset ----------

def build_dataset(tokenizer, datapath, max_length=MAX_LEN, num_proc=8):
    ds = load_dataset("json", data_files=datapath)["train"]
    ds = ds.shuffle(seed=42)
    original_columns = ds.column_names

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            getattr(tokenizer, "eos_token_id", None) or tokenizer.unk_token_id
        )

    role_map = {
        "human": "user", "gpt": "assistant",
        "assistant": "assistant", "user": "user", "system": "system",
    }

    def preprocess_function(examples):
        out = {"attention_mask": [], "input_ids": [], "loss_mask": []}
        for i in range(len(examples["id"])):
            source = examples["conversations"][i]
            if not source:
                continue
            turns = []
            for msg in source:
                role = role_map.get(msg["from"])
                if role is None:
                    continue
                turns.append({"role": role, "content": msg["value"]})
            if not turns or turns[0]["role"] != "user":
                idx = 0
                while idx < len(turns) and turns[idx]["role"] != "user":
                    idx += 1
                turns = turns[idx:]
            if not turns:
                continue

            messages = turns
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            ids = tokenizer(
                text, return_tensors="pt", max_length=max_length,
                truncation=True, add_special_tokens=False,
            ).input_ids[0]
            total_len = ids.shape[0]
            if total_len > max_length:
                continue

            loss_mask = torch.zeros_like(ids)
            messages_so_far = []
            prev_len = 0
            for msg in turns:
                messages_so_far.append(msg)
                cur_text = tokenizer.apply_chat_template(
                    messages_so_far, tokenize=False, add_generation_prompt=False,
                )
                cur_ids = tokenizer(
                    cur_text, add_special_tokens=False,
                ).input_ids
                cur_len = len(cur_ids)
                if msg["role"] == "assistant":
                    s = min(prev_len, total_len)
                    e = min(cur_len, total_len)
                    if e > s:
                        loss_mask[s:e] = 1
                prev_len = cur_len
                if prev_len >= total_len:
                    break

            attention_mask = torch.ones_like(ids)
            if loss_mask.sum() == 0:
                continue
            out["input_ids"].append(ids[None, :])
            out["loss_mask"].append(loss_mask[None, :])
            out["attention_mask"].append(attention_mask[None, :])
        return out

    ds = ds.map(
        preprocess_function, batched=True, num_proc=num_proc,
        remove_columns=original_columns, load_from_cache_file=False,
    )
    ds.set_format(type="torch")
    return ds


class DataCollatorWithPadding:
    def _pad2d(self, t, N):
        B, n = t.shape
        return torch.cat([t, torch.zeros(B, N - n, dtype=t.dtype)], dim=1)

    def __call__(self, features):
        max_len = max(item["input_ids"].shape[1] for item in features)
        return {
            "input_ids": torch.cat([self._pad2d(f["input_ids"], max_len) for f in features]),
            "attention_mask": torch.cat([self._pad2d(f["attention_mask"], max_len) for f in features]),
            "loss_mask": torch.cat([self._pad2d(f["loss_mask"], max_len) for f in features]),
        }


# ---------- Checkpoint resume ----------

def find_latest_checkpoint(directory):
    max_epoch = -1
    for subdir in os.listdir(directory) if os.path.isdir(directory) else []:
        m = re.match(r"state_(\d+)", subdir)
        if m:
            max_epoch = max(max_epoch, int(m.group(1)))
    if max_epoch == -1:
        return None, 0
    return os.path.join(directory, f"state_{max_epoch}"), max_epoch + 1


# ---------- Main ----------

tokenizer = AutoTokenizer.from_pretrained(args.basepath)
traindataset = build_dataset(tokenizer, args.trainpath)
testdataset = build_dataset(tokenizer, args.testpath)

class _Cfg:
    pass

cfg = _Cfg()
cfg.ctr_weight = args.ctr_weight
cfg.max_tree_size = args.max_tree_size
cfg.tree_expand_k = args.tree_expand_k

from dflash_ctr_model import DFlashCTRModel
model = DFlashCTRModel(cfg, args.basepath, args.draftpath)

trainable = [
    p for n, p in model.named_parameters()
    if not n.startswith("target_model.") and p.requires_grad
]

_torch_optimizer = torch.optim.AdamW(
    trainable,
    lr=ds_config.get("optimizer", {}).get("params", {}).get("lr", 2e-5),
    betas=tuple(ds_config.get("optimizer", {}).get("params", {}).get("betas", [0.9, 0.95])),
    weight_decay=ds_config.get("optimizer", {}).get("params", {}).get("weight_decay", 0.0),
)

deepspeed.init_distributed(timeout=datetime.timedelta(minutes=60))

model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args, model=model, model_parameters=trainable,
    optimizer=_torch_optimizer,
)

opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
bad = [
    n for n, p in model.named_parameters()
    if (n.startswith("target_model.") or n.startswith("ref_draft_model."))
    and id(p) in opt_ids
]
assert len(bad) == 0, f"frozen params leaked into optimizer: {bad}"

global_rank = deepspeed.comm.get_rank()
rank = deepspeed.comm.get_local_rank()
world_size = deepspeed.comm.get_world_size()

if global_rank == 0:
    try:
        import wandb
        wandb.init(project="dflash-ctr", config={
            "ctr_weight": args.ctr_weight,
            "max_tree_size": args.max_tree_size,
            "tree_expand_k": args.tree_expand_k,
            "draft": args.draftpath,
            "target": args.basepath,
            **ds_config,
        })
        USE_WANDB = True
    except Exception:
        USE_WANDB = False
else:
    USE_WANDB = False

os.makedirs(args.savedir, exist_ok=True)

train_sampler = DistributedSampler(traindataset, num_replicas=world_size, rank=global_rank, shuffle=True)
train_loader = DataLoader(
    traindataset, batch_size=BATCH_SIZE, sampler=train_sampler,
    num_workers=4, pin_memory=True, collate_fn=DataCollatorWithPadding(),
)
test_sampler = DistributedSampler(testdataset, num_replicas=world_size, rank=global_rank, shuffle=False)
test_loader = DataLoader(
    testdataset, batch_size=BATCH_SIZE, sampler=test_sampler,
    num_workers=4, pin_memory=True, collate_fn=DataCollatorWithPadding(),
)

checkpoint_path, start_epoch = find_latest_checkpoint(args.savedir)
if checkpoint_path:
    if global_rank == 0:
        print(f"Resuming from {checkpoint_path}")
    model_engine.load_checkpoint(checkpoint_path)

num_epochs = args.num_epochs
global_batch_idx = 0

for epoch in range(start_epoch, num_epochs):
    train_sampler.set_epoch(epoch)
    model.train()

    epoch_losses, epoch_ploss, epoch_gto, epoch_reward = [], [], [], []

    for batch_idx, data in enumerate(tqdm(train_loader, disable=(global_rank != 0))):
        model.zero_grad()

        total_loss, marginal_loss, ctr_loss, ctr_acc = model_engine(
            input_ids=data["input_ids"].to(rank),
            attention_mask=data["attention_mask"].to(rank),
            loss_mask=data["loss_mask"].to(rank),
        )

        model_engine.backward(total_loss)
        model_engine.step()

        epoch_losses.append(total_loss.item())
        epoch_ploss.append(marginal_loss.item())
        epoch_gto.append(ctr_loss.item())
        epoch_reward.append(ctr_acc.item())

        if global_rank == 0 and USE_WANDB:
            wandb.log({
                "train/loss": total_loss.item(),
                "train/marginal_loss": marginal_loss.item(),
                "train/ctr_loss": ctr_loss.item(),
                "train/ctr_acc": ctr_acc.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
            })

        # -- Save checkpoint every N batches --
        save_every = getattr(args, "save_every", 50)
        if save_every > 0 and (global_batch_idx + 1) % save_every == 0:
            save_path = os.path.join(args.savedir, f"step_{global_batch_idx + 1}")
            model_engine.save_16bit_model(save_path, exclude_frozen_parameters=True)
            if global_rank == 0:
                print(f"  [batch {global_batch_idx + 1}] Saved checkpoint: {save_path}")

        global_batch_idx += 1

    if global_rank == 0 and epoch_losses:
        avg = lambda xs: sum(xs) / len(xs)
        print(
            f"Epoch {epoch}: loss={avg(epoch_losses):.4f}, "
            f"ploss={avg(epoch_ploss):.4f}, "
            f"gto={avg(epoch_gto):.4f}, "
            f"reward={avg(epoch_reward):.4f}"
        )
        if USE_WANDB:
            wandb.log({
                "train/epoch_loss": avg(epoch_losses),
                "train/epoch_ploss": avg(epoch_ploss),
                "train/epoch_gto": avg(epoch_gto),
                "train/epoch_reward": avg(epoch_reward),
            })

    # -- Eval --
    eval_losses, eval_reward = [], []
    for batch_idx, data in enumerate(tqdm(test_loader, disable=(global_rank != 0))):
        if batch_idx > 40:
            break
        with torch.no_grad():
            total_loss, ploss, gto_loss, group_reward = model_engine(
                input_ids=data["input_ids"].to(rank),
                attention_mask=data["attention_mask"].to(rank),
                loss_mask=data["loss_mask"].to(rank),
            )
            eval_losses.append(total_loss.item())
            eval_reward.append(group_reward.item())

    if global_rank == 0 and eval_losses:
        avg_eval = sum(eval_losses) / len(eval_losses)
        avg_eval_rwd = sum(eval_reward) / len(eval_reward)
        print(f"  Test epoch {epoch}: avg_loss={avg_eval:.4f}  avg_reward={avg_eval_rwd:.4f}")
        if USE_WANDB:
            wandb.log({"test/epoch_loss": avg_eval, "test/epoch_reward": avg_eval_rwd})

    model_engine.save_16bit_model(
        os.path.join(args.savedir, f"state_{epoch}"),
        exclude_frozen_parameters=True,
    )
    if global_rank == 0:
        print(f"  Saved checkpoint: state_{epoch}")
