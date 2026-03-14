"""
DFlash-GTO training script (DeepSpeed).

Usage:
    deepspeed --num_gpus=N main_gto.py \
        --basepath <target_model_path> \
        --draftpath <draft_model_path> \
        --trainpath <train_data.json> \
        --testpath  <test_data.json> \
        --savedir   <output_dir> \
        --deepspeed_config ds_config.json
"""

import argparse
import json
import os
import re

import deepspeed
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

# ---------- CLI ----------
parser = argparse.ArgumentParser(description="DFlash-GTO training")
parser.add_argument("--basepath", type=str, required=True, help="Target model path")
parser.add_argument("--draftpath", type=str, required=True, help="Draft model path")
parser.add_argument("--trainpath", type=str, required=True, help="Training data path")
parser.add_argument("--testpath", type=str, required=True, help="Test data path")
parser.add_argument("--savedir", type=str, default="./dflash_gto_checkpoints")
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--local_rank", type=int, default=-1)
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

from dflash_gto_model import DFlashGTOModel

class _Cfg:
    pass

cfg = _Cfg()

model = DFlashGTOModel(cfg, args.basepath, args.draftpath)

def is_ref_param(name):
    return "ref_draft_model" in name

trainable = [
    p for n, p in model.named_parameters()
    if (not is_ref_param(n))
    and (not n.startswith("target_model."))
    and p.requires_grad
]

model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args, model=model, model_parameters=trainable,
)

opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
bad = [
    n for n, p in model.named_parameters()
    if (is_ref_param(n) or n.startswith("target_model."))
    and id(p) in opt_ids
]
assert len(bad) == 0, f"frozen params leaked into optimizer: {bad}"

global_rank = deepspeed.comm.get_rank()
rank = deepspeed.comm.get_local_rank()
world_size = deepspeed.comm.get_world_size()

if global_rank == 0:
    try:
        import wandb
        wandb.init(project="dflash-gto", config=ds_config)
        USE_WANDB = True
    except ImportError:
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

for epoch in range(start_epoch, num_epochs):
    train_sampler.set_epoch(epoch)
    model.train()

    epoch_losses, epoch_plosses, epoch_gto_losses, epoch_rewards = [], [], [], []

    for batch_idx, data in enumerate(tqdm(train_loader, disable=(global_rank != 0))):
        model.zero_grad()

        total_loss, ploss, gto_loss, rewards_mean = model_engine(
            input_ids=data["input_ids"].to(rank),
            attention_mask=data["attention_mask"].to(rank),
            loss_mask=data["loss_mask"].to(rank),
        )

        model_engine.backward(total_loss)

        if global_rank == 0 and os.environ.get("DEBUG_RL") and batch_idx < 10:
            print(f"[GRAD batch {batch_idx}] lr={optimizer.param_groups[0]['lr']:.2e}")
            print(f"[GRAD batch {batch_idx}] total_loss.requires_grad={total_loss.requires_grad}, grad_fn={total_loss.grad_fn}")

        model_engine.step()

        if global_rank == 0 and os.environ.get("DEBUG_RL") and batch_idx < 10:
            for n, p in model.named_parameters():
                if n == "draft_model.fc.weight":
                    print(f"[WEIGHT batch {batch_idx}] fc.weight hash={p.data.float().sum().item():.8f} norm={p.data.float().norm().item():.8f}")
                    break

        epoch_losses.append(total_loss.item())
        epoch_plosses.append(ploss.item())
        epoch_gto_losses.append(gto_loss.item())
        epoch_rewards.append(rewards_mean.item())

        if global_rank == 0 and os.environ.get("DEBUG_RL"):
            print(f"[RL batch {batch_idx}] ploss={ploss.item():.4f} gto_loss={gto_loss.item():.4f} reward_mean={rewards_mean.item():.4f} total_loss={total_loss.item():.4f}")

        if global_rank == 0 and USE_WANDB:
            wandb.log({
                "train/loss": total_loss.item(),
                "train/ploss": ploss.item(),
                "train/gto_loss": gto_loss.item(),
                "train/reward_mean": rewards_mean.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
            })

    if global_rank == 0 and epoch_losses:
        avg = lambda xs: sum(xs) / len(xs)
        print(
            f"Epoch {epoch}: loss={avg(epoch_losses):.4f}, "
            f"ploss={avg(epoch_plosses):.4f}, "
            f"gto={avg(epoch_gto_losses):.4f}, "
            f"reward={avg(epoch_rewards):.4f}"
        )
        if USE_WANDB:
            wandb.log({
                "train/epoch_loss": avg(epoch_losses),
                "train/epoch_ploss": avg(epoch_plosses),
                "train/epoch_gto_loss": avg(epoch_gto_losses),
                "train/epoch_reward": avg(epoch_rewards),
            })

    # -- Eval --
    eval_losses = []
    for batch_idx, data in enumerate(tqdm(test_loader, disable=(global_rank != 0))):
        if batch_idx > 40:
            break
        with torch.no_grad():
            total_loss, ploss, gto_loss, rewards_mean = model_engine(
                input_ids=data["input_ids"].to(rank),
                attention_mask=data["attention_mask"].to(rank),
                loss_mask=data["loss_mask"].to(rank),
            )
            eval_losses.append(total_loss.item())

    if global_rank == 0 and eval_losses:
        avg_eval = sum(eval_losses) / len(eval_losses)
        print(f"  Test epoch {epoch}: avg_loss={avg_eval:.4f}")
        if USE_WANDB:
            wandb.log({"test/epoch_loss": avg_eval})

    model_engine.save_16bit_model(
        os.path.join(args.savedir, f"state_{epoch}"),
        exclude_frozen_parameters=True,
    )
    if global_rank == 0:
        print(f"  Saved checkpoint: state_{epoch}")
