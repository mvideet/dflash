"""
DFlash LK fine-tuning script (DeepSpeed).

Fine-tunes an existing draft checkpoint using LK losses that directly
optimize acceptance rate instead of KL divergence.

Usage:
    deepspeed --num_gpus=N main_lk.py \
        --basepath <target_model_path> \
        --draftpath <draft_model_path> \
        --trainpath <train_data.json> \
        --testpath  <test_data.json> \
        --savedir   <output_dir> \
        --lk-mode hybrid --lk-eta 3.0 \
        --deepspeed_config ds_config.json
"""

import argparse
import json
import os
import re
import sys
import datetime

import numpy as np
import deepspeed
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, DynamicCache
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmark import dflash_generate
from model.utils import load_and_process_dataset

# ---------- CLI ----------
parser = argparse.ArgumentParser(description="DFlash LK fine-tuning")
parser.add_argument("--basepath", type=str, required=True, help="Target model path")
parser.add_argument("--draftpath", type=str, required=True, help="Draft model path")
parser.add_argument("--trainpath", type=str, required=True, help="Training data path")
parser.add_argument("--testpath", type=str, required=True, help="Test data path")
parser.add_argument("--savedir", type=str, default="./dflash_lk_checkpoints")
parser.add_argument("--num_epochs", type=int, default=6)
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--save-every", type=int, default=200,
                    help="Save checkpoint every N batches")
parser.add_argument("--lk-mode", type=str, default="hybrid",
                    choices=["hybrid", "alpha", "kl"],
                    help="Loss mode: hybrid (recommended), alpha, or kl (baseline)")
parser.add_argument("--lk-eta", type=float, default=3.0,
                    help="Adaptive schedule decay rate for hybrid mode")
parser.add_argument("--pos-decay", type=float, default=1.0,
                    help="Per-position exponential weight decay (1.0=uniform, 0.8=prioritize early)")
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

with open(args.deepspeed_config) as f:
    ds_config = json.load(f)

BATCH_SIZE = ds_config["train_micro_batch_size_per_gpu"]
MAX_LEN = 2048


# ---------- Dataset (reused from main_gto.py) ----------

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
                cur_ids = tokenizer(cur_text, add_special_tokens=False).input_ids
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
cfg.lk_eta = args.lk_eta
cfg.lk_mode = args.lk_mode
cfg.pos_decay = args.pos_decay

from dflash_lk_model import DFlashLKModel
model = DFlashLKModel(cfg, args.basepath, args.draftpath)

trainable = [
    p for n, p in model.named_parameters()
    if not n.startswith("target_model.") and p.requires_grad
]

_torch_optimizer = torch.optim.AdamW(
    trainable,
    lr=ds_config.get("optimizer", {}).get("params", {}).get("lr", 3e-5),
    betas=(0.9, 0.95),
    weight_decay=0.0,
)

deepspeed.init_distributed(timeout=datetime.timedelta(minutes=60))

model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args, model=model, model_parameters=trainable,
    optimizer=_torch_optimizer,
)

opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
bad = [
    n for n, p in model.named_parameters()
    if n.startswith("target_model.") and id(p) in opt_ids
]
assert len(bad) == 0, f"frozen params leaked into optimizer: {bad}"

global_rank = deepspeed.comm.get_rank()
rank = deepspeed.comm.get_local_rank()
world_size = deepspeed.comm.get_world_size()

if global_rank == 0:
    try:
        import wandb
        wandb.init(project="dflash-lk", config={
            "lk_mode": args.lk_mode,
            "lk_eta": args.lk_eta,
            "pos_decay": args.pos_decay,
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

if global_rank == 0:
    mt_bench = load_and_process_dataset("mt-bench")
    print(f"Loaded MT-Bench: {len(mt_bench)} prompts for eval")


@torch.inference_mode()
def mtbench_eval(target_model, draft_model, tok, dataset, device,
                 max_new_tokens=512, max_tree_size=32, expand_k=7):
    """Run full MT-Bench speculative decoding eval (tree v2)."""
    block_size = draft_model.block_size
    mask_token_id = draft_model.mask_token_id
    baseline_tpots, spec_tpots = [], []
    all_acc_lens = []

    for instance in tqdm(dataset, desc="MT-Bench eval"):
        messages = []
        for user_content in instance["turns"]:
            messages.append({"role": "user", "content": user_content})
            input_text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = tok.encode(input_text, return_tensors="pt").to(device)

            baseline_resp = dflash_generate(
                model=draft_model, target=target_model, input_ids=input_ids,
                mask_token_id=mask_token_id, max_new_tokens=max_new_tokens,
                block_size=1, stop_token_ids=[tok.eos_token_id], temperature=0.0,
            )
            baseline_tpots.append(baseline_resp.time_per_output_token)

            spec_resp = dflash_generate(
                model=draft_model, target=target_model, input_ids=input_ids,
                mask_token_id=mask_token_id, max_new_tokens=max_new_tokens,
                block_size=block_size, stop_token_ids=[tok.eos_token_id],
                temperature=0.0, tree_version=2,
                max_tree_size=max_tree_size, expand_k=expand_k,
            )
            spec_tpots.append(spec_resp.time_per_output_token)
            all_acc_lens.extend(spec_resp.acceptance_lengths)

            gen_ids = spec_resp.output_ids[0, spec_resp.num_input_tokens:]
            messages.append({
                "role": "assistant",
                "content": tok.decode(gen_ids, skip_special_tokens=True),
            })

    avg_acc = np.mean(all_acc_lens) if all_acc_lens else 0.0
    avg_base = np.mean(baseline_tpots) if baseline_tpots else 1.0
    avg_spec = np.mean(spec_tpots) if spec_tpots else 1.0
    speedup = avg_base / avg_spec if avg_spec > 0 else 0.0
    return avg_acc, speedup


for epoch in range(start_epoch, num_epochs):
    train_sampler.set_epoch(epoch)
    model.train()

    epoch_losses, epoch_alphas = [], []

    for batch_idx, data in enumerate(tqdm(train_loader, disable=(global_rank != 0))):
        model.zero_grad()

        total_loss, avg_alpha = model_engine(
            input_ids=data["input_ids"].to(rank),
            attention_mask=data["attention_mask"].to(rank),
            loss_mask=data["loss_mask"].to(rank),
        )

        model_engine.backward(total_loss)
        model_engine.step()

        epoch_losses.append(total_loss.item())
        epoch_alphas.append(avg_alpha.item())

        if global_rank == 0 and USE_WANDB:
            wandb.log({
                "train/loss": total_loss.item(),
                "train/avg_alpha": avg_alpha.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
            })

        save_every = args.save_every
        if save_every > 0 and (global_batch_idx + 1) % save_every == 0:
            save_path = os.path.join(args.savedir, f"step_{global_batch_idx + 1}")
            model_engine.save_16bit_model(save_path, exclude_frozen_parameters=True)
            if global_rank == 0:
                print(f"  [batch {global_batch_idx + 1}] Saved checkpoint: {save_path}")

        global_batch_idx += 1

    if global_rank == 0 and epoch_losses:
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_a = sum(epoch_alphas) / len(epoch_alphas)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, avg_alpha={avg_a:.4f}")
        if USE_WANDB:
            wandb.log({
                "train/epoch_loss": avg_loss,
                "train/epoch_alpha": avg_a,
            })

    # -- Eval: full MT-Bench with speedup & acceptance length (rank 0 only) --
    if global_rank == 0:
        model.eval()
        avg_acc_len, speedup = mtbench_eval(
            model.target_model, model.draft_model, tokenizer, mt_bench, rank,
        )
        print(f"  MT-Bench epoch {epoch}: avg_acceptance_len={avg_acc_len:.2f}  speedup={speedup:.2f}x")
        if USE_WANDB:
            wandb.log({
                "eval/avg_acceptance_len": avg_acc_len,
                "eval/speedup": speedup,
            })
        model.train()
    deepspeed.comm.barrier()

    model_engine.save_16bit_model(
        os.path.join(args.savedir, f"state_{epoch}"),
        exclude_frozen_parameters=True,
    )
    if global_rank == 0:
        print(f"  Saved checkpoint: state_{epoch}")
