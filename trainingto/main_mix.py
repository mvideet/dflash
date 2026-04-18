"""
DFlash MIX training driver — DFlash paper recipe + SpecForge TTT + variable block.

Usage (smoke test, 3 GPUs on the smoke-size dataset):
    deepspeed --num_gpus=3 --master_port=29700 main_mix.py \
        --basepath Qwen/Qwen3-4B \
        --draftpath z-lab/Qwen3-4B-DFlash-b16 \
        --trainpath data/nemotron_math_smoke/train_smoke.jsonl \
        --testpath  data/nemotron_math_smoke/test_smoke.jsonl \
        --savedir   dflash_mix_checkpoints \
        --deepspeed_config ds_config.json \
        --num_epochs 1 \
        --gamma-loss 7.0 \
        --random-anchors --anchors-per-seq 32 \
        --block-sizes 8,12,16,24 \
        --ctr-weight 0.5 --ttt-weight 0.2 \
        --max-tree-size 16 --tree-expand-k 5

Real run (8 GPUs, full nemotron_math, ~1 epoch):
    Replace data paths, set --num_epochs 1, --save-every 500.
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

parser = argparse.ArgumentParser(description="DFlash MIX training")
parser.add_argument("--basepath", type=str, required=True, help="Target model path")
parser.add_argument("--draftpath", type=str, required=True, help="Draft model path")
parser.add_argument("--trainpath", type=str, required=True, help="Training data path")
parser.add_argument("--testpath", type=str, required=True, help="Test data path")
parser.add_argument("--savedir", type=str, default="./dflash_mix_checkpoints")
parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--save-every", type=int, default=500)

parser.add_argument("--gamma-loss", type=float, default=7.0,
                    help="Exp-weighting denominator (DFlash paper: 7 for b=16)")
parser.add_argument("--random-anchors", action="store_true",
                    help="DFlash Table 9 random anchor sampling (+13% tau)")
parser.add_argument("--anchors-per-seq", type=int, default=32,
                    help="Number of random anchors per training sequence")
parser.add_argument("--block-sizes", type=str, default="16",
                    help="Comma-separated list of block sizes sampled per step "
                         "(e.g. '8,12,16,24' for variable curriculum). Default '16' = fixed.")
parser.add_argument("--block-size-probs", type=str, default="",
                    help="Comma-separated unnormalised sampling weights for "
                         "--block-sizes (empty = uniform).")

parser.add_argument("--ctr-weight", type=float, default=0.5,
                    help="Weight on tree-attention conditional CE loss")
parser.add_argument("--ttt-weight", type=float, default=0.2,
                    help="Weight on TTT recursive-block loss (SpecForge-style)")
parser.add_argument("--max-tree-size", type=int, default=16)
parser.add_argument("--tree-expand-k", type=int, default=5)

parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

with open(args.deepspeed_config) as f:
    ds_config = json.load(f)

BATCH_SIZE = ds_config["train_micro_batch_size_per_gpu"]
MAX_LEN = 2048


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
            text = tokenizer.apply_chat_template(
                turns, tokenize=False, add_generation_prompt=False,
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
                cur_len = len(tokenizer(
                    cur_text, add_special_tokens=False,
                ).input_ids)
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


def find_latest_checkpoint(directory):
    max_step = -1
    latest = None
    if not os.path.isdir(directory):
        return None, 0
    for subdir in os.listdir(directory):
        m = re.match(r"step_(\d+)", subdir)
        if m:
            step = int(m.group(1))
            if step > max_step:
                max_step = step
                latest = subdir
    if latest is None:
        return None, 0
    return os.path.join(directory, latest), max_step


# --- Parse block-size curriculum ---
block_sizes = [int(x) for x in args.block_sizes.split(",") if x.strip()]
if args.block_size_probs:
    bs_probs = [float(x) for x in args.block_size_probs.split(",")]
    assert len(bs_probs) == len(block_sizes), \
        "block-size-probs length must match block-sizes"
else:
    bs_probs = None


# --- Build tokenizer + datasets ---
tokenizer = AutoTokenizer.from_pretrained(args.basepath)
traindataset = build_dataset(tokenizer, args.trainpath)
testdataset = build_dataset(tokenizer, args.testpath)


# --- Config wrapper ---
class _Cfg:
    pass
cfg = _Cfg()
cfg.gamma_loss = args.gamma_loss
cfg.random_anchors = args.random_anchors
cfg.anchors_per_seq = args.anchors_per_seq
cfg.block_sizes = block_sizes
cfg.block_size_probs = bs_probs
cfg.ctr_weight = args.ctr_weight
cfg.ttt_weight = args.ttt_weight
cfg.max_tree_size = args.max_tree_size
cfg.tree_expand_k = args.tree_expand_k


from dflash_mix_model import DFlashMixModel
model = DFlashMixModel(cfg, args.basepath, args.draftpath)

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

# Bump torch's default NCCL timeout BEFORE any DS PG creation, so the
# secondary ZeRO process group inherits the long timeout too.  Otherwise
# DS creates PG ID 1 with the hardcoded 10-min default and ALLREDUCE
# stragglers (co-tenant GPU contention) kill training.
_LONG_TIMEOUT = datetime.timedelta(hours=2)
try:
    from torch.distributed import distributed_c10d as _c10d
    _c10d.default_pg_timeout = _LONG_TIMEOUT
except Exception:
    pass

deepspeed.init_distributed(timeout=_LONG_TIMEOUT)
model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args, model=model, model_parameters=trainable,
    optimizer=_torch_optimizer,
)

global_rank = deepspeed.comm.get_rank()
rank = deepspeed.comm.get_local_rank()
world_size = deepspeed.comm.get_world_size()

if global_rank == 0:
    try:
        import wandb
        wandb.init(project="dflash-mix", config={
            "gamma_loss": args.gamma_loss,
            "random_anchors": args.random_anchors,
            "anchors_per_seq": args.anchors_per_seq,
            "block_sizes": block_sizes,
            "ctr_weight": args.ctr_weight,
            "ttt_weight": args.ttt_weight,
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

train_sampler = DistributedSampler(
    traindataset, num_replicas=world_size, rank=global_rank, shuffle=True,
)
train_loader = DataLoader(
    traindataset, batch_size=BATCH_SIZE, sampler=train_sampler,
    num_workers=4, pin_memory=True, collate_fn=DataCollatorWithPadding(),
)
test_sampler = DistributedSampler(
    testdataset, num_replicas=world_size, rank=global_rank, shuffle=False,
)
test_loader = DataLoader(
    testdataset, batch_size=BATCH_SIZE, sampler=test_sampler,
    num_workers=4, pin_memory=True, collate_fn=DataCollatorWithPadding(),
)

checkpoint_path, start_step = find_latest_checkpoint(args.savedir)
if checkpoint_path:
    if global_rank == 0:
        print(f"Resuming from {checkpoint_path}")
    # Load state dict for resume. deepspeed's load_checkpoint path is
    # a directory with engine state — we use save_16bit_model for
    # storage so resume requires restarting from scratch unless we
    # explicitly save+load full engine state.  For this session we
    # just start fresh after a checkpoint dir exists.

num_epochs = args.num_epochs
global_batch_idx = start_step

for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)
    model.train()
    epoch_losses, epoch_marg, epoch_ctr, epoch_ttt = [], [], [], []

    for batch_idx, data in enumerate(tqdm(train_loader, disable=(global_rank != 0))):
        model.zero_grad()
        total_loss, marg_loss, ctr_loss, ctr_acc = model_engine(
            input_ids=data["input_ids"].to(rank),
            attention_mask=data["attention_mask"].to(rank),
            loss_mask=data["loss_mask"].to(rank),
        )
        model_engine.backward(total_loss)
        model_engine.step()

        epoch_losses.append(total_loss.item())
        epoch_marg.append(marg_loss.item())
        epoch_ctr.append(ctr_loss.item())

        if global_rank == 0 and USE_WANDB:
            import wandb
            wandb.log({
                "train/loss": total_loss.item(),
                "train/marginal": marg_loss.item(),
                "train/ctr": ctr_loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
            })

        save_every = getattr(args, "save_every", 500)
        if save_every > 0 and (global_batch_idx + 1) % save_every == 0:
            save_path = os.path.join(args.savedir, f"step_{global_batch_idx + 1}")
            model_engine.save_16bit_model(save_path, exclude_frozen_parameters=True)
            if global_rank == 0:
                print(f"  [step {global_batch_idx + 1}] saved: {save_path}")

        global_batch_idx += 1

    if global_rank == 0 and epoch_losses:
        avg = lambda xs: sum(xs) / len(xs)
        print(
            f"Epoch {epoch}: loss={avg(epoch_losses):.4f} "
            f"marg={avg(epoch_marg):.4f} ctr={avg(epoch_ctr):.4f}"
        )

    # Eval pass.
    eval_losses = []
    for batch_idx, data in enumerate(tqdm(test_loader, disable=(global_rank != 0))):
        if batch_idx > 20:
            break
        with torch.no_grad():
            total_loss, marg_loss, ctr_loss, _ = model_engine(
                input_ids=data["input_ids"].to(rank),
                attention_mask=data["attention_mask"].to(rank),
                loss_mask=data["loss_mask"].to(rank),
            )
            eval_losses.append(total_loss.item())
    if global_rank == 0 and eval_losses:
        print(f"  Test epoch {epoch}: avg_loss={sum(eval_losses)/len(eval_losses):.4f}")

    model_engine.save_16bit_model(
        os.path.join(args.savedir, f"epoch_{epoch}"),
        exclude_frozen_parameters=True,
    )
    if global_rank == 0:
        print(f"  Saved epoch checkpoint: epoch_{epoch}")
