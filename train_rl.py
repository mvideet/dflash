"""RL training for DFlash drafter — GRPO-style on heap-verified rollouts.

Setup
-----
- Drafter (trainable): z-lab/Qwen3-4B-DFlash-b16 (block-diffusion).
- Target (frozen): Qwen/Qwen3-4B.
- One block per training step. Per prompt, M heap rollouts with Gumbel-noised
  heap scores → group-relative advantages over τ.
- PPO-clip(ε=0.2) policy gradient + entropy bonus α. No separate ref-policy KL
  (PPO clip provides the trust region).
- Reward: τ (number of accepted tokens in the block).

Loop per step
-------------
1. Pick batch_size prompts from data.
2. For each prompt:
   a. Drafter forward on [prefix, anchor, mask*15] — logits with grad.
   b. M times: build heap with Gumbel noise → target forward → τ_i, accepted path.
3. Per-prompt advantages: A_i = (τ_i - mean(τ_prompt)) / (std(τ_prompt) + ε).
4. Loss = PPO-clip(A, π_new/π_old, ε) + α · entropy_bonus  (clamp by sign).
5. Optimizer step.

Diagnostics (logged every step)
-------------------------------
τ_mean / τ_std / τ_min / τ_max
advantage_mean / advantage_std
clipped_frac
entropy_per_position[0..b-1]
grad_norm
loss components
"""
import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DynamicCache

try:
    import wandb
    _HAVE_WANDB = True
except ImportError:
    _HAVE_WANDB = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DFlashDraftModel, load_and_process_dataset
from model.dflash_tree import (
    build_node_budget_tree,
    create_tree_attention_mask_dynamic,
    select_best_dynamic_leaf,
)
from model.utils import extract_context_feature


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-4B")
    ap.add_argument("--drafter", default="z-lab/Qwen3-4B-DFlash-b16")
    ap.add_argument("--out-dir", default="trainingto/dflash_rl_v1")
    ap.add_argument("--dataset", default="math500",
                    help="prompt source; math500 is small but fine for smoke")
    ap.add_argument("--num-steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="prompts per step")
    ap.add_argument("--rollouts-per-prompt", type=int, default=2,
                    help="M: rollouts per prompt for GRPO advantage")
    ap.add_argument("--max-prompt-tokens", type=int, default=256)
    ap.add_argument("--max-tree-size", type=int, default=160)
    ap.add_argument("--expand-k", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--gumbel-scale", type=float, default=0.5,
                    help="heap-score Gumbel noise scale for rollout variance")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--ppo-eps", type=float, default=0.2)
    ap.add_argument("--entropy-coef", type=float, default=0.01,
                    help="DEPRECATED in v3: kept for backward compat; "
                         "use --entropy-sharp-coef / --entropy-flat-coef instead.")
    ap.add_argument("--entropy-sharp-coef", type=float, default=0.0,
                    help="v3 asymmetric entropy: at depths where drafter top-1 == "
                         "target argmax, MINIMIZE H(q_d) (sharpen). Positive value.")
    ap.add_argument("--entropy-flat-coef", type=float, default=0.0,
                    help="v3 asymmetric entropy: at depths where drafter top-1 != "
                         "target argmax, MAXIMIZE H(q_d) (flatten). Positive value.")
    ap.add_argument("--aux-ce-coef", type=float, default=0.0,
                    help="v3 auxiliary CE: weight of -log q_d(target_argmax_d) loss "
                         "on the accepted path. Free signal from target's verify.")
    ap.add_argument("--aux-tv-coef", type=float, default=0.0,
                    help="v4 auxiliary TV loss: weight of (1/2)Σ|p_d - q_d| over "
                         "target's top-K. Direct optimization of acceptance "
                         "(α = 1 - TV). DistillSpec ICLR'24 showed TV > KL.")
    ap.add_argument("--loo-baseline", action="store_true",
                    help="Use leave-one-out baseline (RLOO, Ahmadian et al. 2024) "
                         "instead of GRPO same-mean. For rollout i in a group of "
                         "size M, baseline_i = mean(tau_j for j != i). Lower "
                         "variance than the group-mean baseline.")
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="v5 DVI-style warmup: for the first N steps, PG and "
                         "asym-ent coefs are 0 (pure aux-TV distillation). "
                         "After N, RL kicks in at full strength. 0 disables.")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="target sampling temperature (0 = argmax)")
    ap.add_argument("--wandb-project", type=str, default="dflash-rl",
                    help="wandb project name; empty disables wandb")
    ap.add_argument("--wandb-run-name", type=str, default="",
                    help="wandb run name; defaults to a timestamp")
    return ap.parse_args()


def _target_prefill_and_extract_hidden(target, drafter, prefix_ids: torch.Tensor):
    """Run target on the prefix; return (past_key_values, target_hidden_full).

    target_hidden_full covers ALL prefix positions; drafter uses it as cross-
    attention context (NOT just the anchor's hidden state).
    """
    out = target.model(
        prefix_ids[:, :-1], use_cache=True, output_hidden_states=True, return_dict=True,
    )
    # Note: we drop the last token from past_kv since the anchor is fed as the
    # block's first input. target_hidden covers up to prefix_len-1.
    target_hidden_full = extract_context_feature(out.hidden_states, drafter.target_layer_ids)
    return out.past_key_values, target_hidden_full


@torch.no_grad()
def one_rollout(
    drafter: DFlashDraftModel,
    target,
    tokenizer,
    prefix_ids: torch.Tensor,
    target_hidden_full: torch.Tensor,
    past_kv,
    args,
    gumbel_seed: int,
):
    """Run one heap-and-verify rollout from a prefix. Returns (τ, accepted_token_ids,
    accepted_depth_indices, drafter_topk_idx, drafter_topk_logp).

    target_hidden_full: [1, 1, hidden_concat] from extract_context_feature on prefix.
    past_kv: target's KV cache after prefix prefill (so we don't redo it M times).
    """
    device = prefix_ids.device
    block_size = args.block_size
    mask_id = drafter.mask_token_id

    # Build drafter input: [anchor, mask, mask, ..., mask] (b tokens).
    # DFlash drafter takes a NOISE EMBEDDING (target.embed_tokens applied to the
    # block_output_ids), NOT raw input_ids, and is conditioned on target_hidden.
    anchor = prefix_ids[:, -1:]
    masks = torch.full(
        (1, block_size - 1), mask_id, dtype=prefix_ids.dtype, device=device,
    )
    block_output_ids = torch.cat([anchor, masks], dim=1)  # [1, b]
    noise_embedding = target.model.embed_tokens(block_output_ids)  # [1, b, H]
    # Anchor at position prefix_len-1, masks at prefix_len, ..., prefix_len+b-2
    P = prefix_ids.shape[1]
    pos = torch.arange(P - 1, P - 1 + block_size, device=device).unsqueeze(0)

    # Drafter forward (no_grad — gradient will be re-computed in compute_loss).
    # is_causal=False; drafter starts with a fresh KV cache for this block.
    past_kv_draft = DynamicCache()
    draft_hidden = drafter(
        target_hidden=target_hidden_full,
        noise_embedding=noise_embedding,
        position_ids=pos,
        past_key_values=past_kv_draft,
        use_cache=True,
        is_causal=False,
    )
    # draft_hidden: [1, b, H] — take last b-1 (the mask positions)
    draft_hidden_block = draft_hidden[:, -(block_size - 1):, :]
    draft_logits = target.lm_head(draft_hidden_block)  # [1, b-1, V]

    # Top-K
    K = args.expand_k
    topk_logp, topk_idx = torch.log_softmax(draft_logits[0].float(), dim=-1).topk(K, dim=-1)
    # Build heap with Gumbel noise (the source of within-prompt variance for GRPO)
    torch.manual_seed(gumbel_seed)
    random.seed(gumbel_seed)
    np.random.seed(gumbel_seed)
    (packed_ids, packed_pos_relative, parent_idx,
     leaf_paths, leaf_tokens) = build_node_budget_tree(
        draft_logits=draft_logits,
        anchor_token_ids=anchor,
        max_tree_size=args.max_tree_size,
        expand_k=K,
        sample_gumbel_scale=args.gumbel_scale,
        sample_gumbel_scope="all",
    )

    # Target verify — re-prefill prefix[:, :-1] (the last token is the anchor
    # of the new block and must NOT be in past_kv; it's added at position
    # prefix_len-1 as the root of the tree). Eliminates state-bleed across
    # rollouts at the cost of one re-prefill per rollout.
    saved_attn = target.config._attn_implementation
    target.config._attn_implementation = "sdpa"
    fresh_past_kv = DynamicCache()
    _ = target.model(
        prefix_ids[:, :-1], use_cache=True, past_key_values=fresh_past_kv, return_dict=True,
    )
    prefix_len = prefix_ids.shape[1]
    # Block starts at position prefix_len-1 (anchor sits there); packed_pos_relative
    # gives [0, 1, 2, ...] within the block, so absolute = relative + (prefix_len-1).
    packed_pos = packed_pos_relative + (prefix_len - 1)
    attn_mask = create_tree_attention_mask_dynamic(
        packed_pos_relative, parent_idx, prefix_len - 1,
    )
    mod_out = target.model(
        packed_ids,
        position_ids=packed_pos,
        past_key_values=fresh_past_kv,
        use_cache=True,           # need past_kv usage; mutation is OK (we recreate each rollout)
        attention_mask=attn_mask,
        return_dict=True,
    )
    logits = target.lm_head(mod_out.last_hidden_state)
    target.config._attn_implementation = saved_attn

    best_leaf, n = select_best_dynamic_leaf(
        logits=logits, leaf_paths=leaf_paths, leaf_tokens=leaf_tokens,
        temperature=args.temperature,
    )
    best_path = leaf_paths[best_leaf]
    best_tokens = leaf_tokens[best_leaf]
    accepted_tokens = best_tokens[:n].cpu().tolist()

    # v3: extract target's argmax token at each accepted depth, for aux CE and
    # asymmetric entropy. logits[0, best_path[d], :].argmax() is target's
    # prediction at the parent node of the d-th accepted token.
    # v4: also extract target's top-K vocab indices and probs for TV loss.
    target_argmax_per_depth: list = []
    target_topk_idx_at_acc = None
    target_topk_probs_at_acc = None
    if n > 0:
        # best_path has length block_size; best_path[d] is the PARENT node index
        # that predicts the (d+1)-th token. So for accepted depth d in [0, n-1],
        # the target's argmax at depth d is argmax of logits at best_path[d].
        parent_indices = best_path[:n].to(logits.device, dtype=torch.long)
        parent_logits = logits[0].index_select(0, parent_indices).float()  # [n, V]
        target_argmax_per_depth = parent_logits.argmax(dim=-1).cpu().tolist()
        # v4: target's top-K distribution at each accepted depth (no grad).
        parent_probs = parent_logits.softmax(dim=-1)  # [n, V]
        topk = parent_probs.topk(args.expand_k, dim=-1)
        target_topk_idx_at_acc = topk.indices.detach().cpu()    # [n, K]
        target_topk_probs_at_acc = topk.values.detach().cpu()    # [n, K]


    # For gradient computation: we need (depth_in_block, token_id) of each accepted token.
    # accepted token at position i in block_size is best_tokens[i-1] for i ∈ 1..n.
    # The depth in topk arrays is i-1 ∈ 0..b-2.
    tau = n  # accepted length
    accepted_depths = list(range(n))  # 0..n-1 in drafter logits

    return {
        "tau": tau,
        "accepted_tokens": accepted_tokens,
        "accepted_depths": accepted_depths,
        "topk_idx": topk_idx,  # [b-1, K]
        "topk_logp": topk_logp,  # [b-1, K]
        "target_argmax_per_depth": target_argmax_per_depth,  # length n
        "target_topk_idx_at_acc": target_topk_idx_at_acc,    # [n, K] or None
        "target_topk_probs_at_acc": target_topk_probs_at_acc,  # [n, K] or None
    }


def compute_loss(
    drafter: DFlashDraftModel,
    target,
    prefix_ids: torch.Tensor,
    target_hidden_full: torch.Tensor,
    rollouts: list,  # list of dicts from one_rollout
    advantages: list,  # list of float, parallel to rollouts
    args,
    block_size: int,
    rl_coef_scale: float = 1.0,
):
    """Compute PPO-clip policy gradient + entropy bonus.

    Reruns drafter WITH grad to get fresh log-probs at the accepted tokens, and
    uses the topk_logp snapshot from the rollout as the "old" policy (PPO ratio).
    """
    device = prefix_ids.device
    mask_id = drafter.mask_token_id
    anchor = prefix_ids[:, -1:]
    masks = torch.full(
        (1, block_size - 1), mask_id, dtype=prefix_ids.dtype, device=device,
    )
    block_output_ids = torch.cat([anchor, masks], dim=1)
    noise_embedding = target.model.embed_tokens(block_output_ids).detach()
    P = prefix_ids.shape[1]
    pos = torch.arange(P - 1, P - 1 + block_size, device=device).unsqueeze(0)

    past_kv_draft = DynamicCache()
    draft_hidden = drafter(
        target_hidden=target_hidden_full.detach(),
        noise_embedding=noise_embedding,
        position_ids=pos,
        past_key_values=past_kv_draft,
        use_cache=True,
        is_causal=False,
    )
    draft_hidden_block = draft_hidden[:, -(block_size - 1):, :]
    logits = target.lm_head(draft_hidden_block)  # [1, b-1, V] WITH grad through drafter
    log_probs = torch.log_softmax(logits[0].float(), dim=-1)  # [b-1, V]

    # Entropy per position (used for entropy bonus)
    probs = log_probs.exp()
    entropy_per_pos = -(probs * log_probs).sum(dim=-1)  # [b-1]
    mean_entropy = entropy_per_pos.mean()

    pg_loss_total = torch.zeros((), device=device, dtype=torch.float32)
    clipped_count = 0
    n_terms = 0
    for r, A in zip(rollouts, advantages):
        if r["tau"] == 0:
            continue
        tokens = torch.tensor(r["accepted_tokens"], device=device, dtype=torch.long)
        depths = torch.tensor(r["accepted_depths"], device=device, dtype=torch.long)
        # New log-prob at accepted tokens
        new_lp = log_probs[depths, tokens]  # [n]
        # Old log-prob (from rollout snapshot) — we need to find each token's rank within topk
        # topk_idx[d, :] has the K tokens; find rank where topk_idx[d, k] == tokens[i]
        # If token isn't in top-K, its old_lp is unavailable — fall back to log_softmax from
        # the SAME drafter call that produced topk_logp (which had no grad).
        old_topk_idx = r["topk_idx"].to(device)  # [b-1, K]
        old_topk_lp = r["topk_logp"].to(device)  # [b-1, K]
        # Find old_lp[d] = topk_logp at the rank of tokens[d]
        # Vectorize: for each d, check which k makes topk_idx[d,k]==tokens[d]
        old_lp = torch.zeros(len(depths), device=device, dtype=torch.float32)
        for i, (d, t) in enumerate(zip(r["accepted_depths"], r["accepted_tokens"])):
            mask_rank = (old_topk_idx[d] == t)
            if mask_rank.any():
                k = mask_rank.nonzero(as_tuple=False)[0, 0].item()
                old_lp[i] = old_topk_lp[d, k]
            else:
                # token outside top-K (rare); use log-prob from the new logits as fallback,
                # detached (so ratio ≈ 1 for that step).
                old_lp[i] = new_lp[i].detach()

        ratio = (new_lp - old_lp).exp()  # [n]
        ratio_clipped = ratio.clamp(1 - args.ppo_eps, 1 + args.ppo_eps)
        # PPO loss with advantage A (scalar) broadcast to all tokens
        pg = -torch.minimum(ratio * A, ratio_clipped * A).sum()
        pg_loss_total = pg_loss_total + pg
        clipped_count += int(((ratio < 1 - args.ppo_eps) | (ratio > 1 + args.ppo_eps)).sum().item())
        n_terms += len(depths)

    pg_loss_mean = pg_loss_total / max(n_terms, 1)

    # ------------------------------------------------------------------
    # v3 additions: auxiliary CE + asymmetric entropy
    # v4 addition: TV-distance loss (DistillSpec ICLR'24)
    # ------------------------------------------------------------------
    # Drafter's top-1 at each block position (no grad — used for match check).
    drafter_top1_per_pos = log_probs.argmax(dim=-1).detach()  # [b-1]

    # Accumulate match-mask across rollouts: which (rollout, depth) had a match?
    # We compute aux CE and asymmetric entropy per rollout per accepted depth.
    aux_ce_total = torch.zeros((), device=device, dtype=torch.float32)
    aux_ce_terms = 0
    aux_tv_total = torch.zeros((), device=device, dtype=torch.float32)
    aux_tv_terms = 0
    ent_loss_total = torch.zeros((), device=device, dtype=torch.float32)
    ent_terms = 0
    n_match = 0
    n_mismatch = 0
    for r, _A in zip(rollouts, advantages):
        if r["tau"] == 0:
            continue
        tgt_argmax = r.get("target_argmax_per_depth") or []
        depths = r["accepted_depths"]
        for i, d in enumerate(depths):
            if i >= len(tgt_argmax):
                break
            t_star = int(tgt_argmax[i])
            # Aux CE: -log q_d(t_star)  (target's argmax is the "label")
            aux_ce_total = aux_ce_total + (-log_probs[d, t_star])
            aux_ce_terms += 1
            # Asymmetric entropy at this depth
            is_match = int(drafter_top1_per_pos[d].item()) == t_star
            if is_match:
                # Drafter is correct — SHARPEN: minimize H(q_d)
                ent_loss_total = ent_loss_total + args.entropy_sharp_coef * entropy_per_pos[d]
                n_match += 1
            else:
                # Drafter is wrong — FLATTEN: maximize H(q_d)
                ent_loss_total = ent_loss_total - args.entropy_flat_coef * entropy_per_pos[d]
                n_mismatch += 1
            ent_terms += 1

        # v4: TV loss over target's top-K at each accepted depth.
        # TV(p, q) approximated as (1/2) Σ_{v in topK(p)} |p_v - q_v|
        # (truncated to top-K — DistillSpec showed this works well).
        # Note 1 - TV(p, q) is a lower bound on acceptance probability under
        # standard speculative-decoding rejection sampling.
        if args.aux_tv_coef > 0 and r.get("target_topk_idx_at_acc") is not None:
            tgt_idx = r["target_topk_idx_at_acc"].to(device, dtype=torch.long)   # [n, K]
            tgt_p = r["target_topk_probs_at_acc"].to(device, dtype=torch.float32)  # [n, K]
            depths_t = torch.tensor(depths, device=device, dtype=torch.long)
            # q_at_tgt[i, k] = q_d=depths[i](v=tgt_idx[i,k])
            q_d = probs[depths_t]  # [n, V]  WITH grad
            q_at_tgt = q_d.gather(1, tgt_idx)  # [n, K]
            tv_per_depth = 0.5 * (tgt_p - q_at_tgt).abs().sum(dim=-1)  # [n]
            aux_tv_total = aux_tv_total + tv_per_depth.sum()
            aux_tv_terms += tv_per_depth.numel()

    aux_ce_loss = aux_ce_total / max(aux_ce_terms, 1)
    aux_tv_loss = aux_tv_total / max(aux_tv_terms, 1)
    asym_ent_loss = ent_loss_total / max(ent_terms, 1)

    # Legacy symmetric entropy bonus (only if v3 coefs are both 0, i.e. running v2 config)
    legacy_ent_active = (args.entropy_sharp_coef == 0.0 and args.entropy_flat_coef == 0.0 and
                         args.entropy_coef > 0.0)
    if legacy_ent_active:
        legacy_ent_loss = -args.entropy_coef * mean_entropy
    else:
        legacy_ent_loss = torch.zeros((), device=device, dtype=torch.float32)

    # v5: scale RL terms (PG, asym ent, aux CE) by rl_coef_scale. Aux TV always
    # active. legacy_ent is also scaled (it's an RL exploration term).
    total_loss = (
        rl_coef_scale * pg_loss_mean
        + rl_coef_scale * args.aux_ce_coef * aux_ce_loss
        + args.aux_tv_coef * aux_tv_loss
        + rl_coef_scale * asym_ent_loss
        + rl_coef_scale * legacy_ent_loss
    )

    return {
        "total_loss": total_loss,
        "pg_loss": pg_loss_mean.detach(),
        "aux_ce_loss": aux_ce_loss.detach(),
        "aux_tv_loss": aux_tv_loss.detach(),
        "asym_ent_loss": asym_ent_loss.detach(),
        "legacy_ent_loss": legacy_ent_loss.detach(),
        "mean_entropy": mean_entropy.detach(),
        "entropy_per_pos": entropy_per_pos.detach(),
        "clipped_frac": clipped_count / max(n_terms, 1),
        "n_terms": n_terms,
        "n_match": n_match,
        "n_mismatch": n_mismatch,
        "match_frac": n_match / max(n_match + n_mismatch, 1),
        "pos0_top1_prob": float(log_probs[0].exp().max().item()),
    }


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading target {args.target}...", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, attn_implementation="sdpa", dtype=torch.bfloat16,
    ).to(device).eval()
    for p in target.parameters():
        p.requires_grad_(False)

    print(f"Loading drafter {args.drafter}...", flush=True)
    drafter = DFlashDraftModel.from_pretrained(
        args.drafter, attn_implementation="sdpa", dtype=torch.bfloat16,
    ).to(device)
    drafter.train()

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset of prompts (use math500 for v1)
    print(f"Loading dataset {args.dataset}...", flush=True)
    raw_ds = load_and_process_dataset(args.dataset).shuffle(seed=args.seed)
    prompts = [raw_ds[i]["turns"][0] for i in range(min(len(raw_ds), 500))]

    def tokenize_prompt(text: str):
        # Use the same chat template as benchmark.py (enable_thinking=False emits
        # an EMPTY think block: <think>\n\n</think>\n\n), then SAMPLE one token
        # from target to serve as the response-domain anchor — mirroring the
        # AR-warmup that dflash_generate does at the start of each sequence.
        input_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        if ids.shape[1] > args.max_prompt_tokens - 1:
            ids = ids[:, -(args.max_prompt_tokens - 1):]
        with torch.no_grad():
            out = target(ids, use_cache=False, return_dict=True)
            first_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, first_tok], dim=1)
        return ids

    opt = torch.optim.AdamW(
        [p for p in drafter.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0,
    )
    print(f"trainable params: {sum(p.numel() for p in drafter.parameters() if p.requires_grad):,}",
          flush=True)

    # wandb (optional)
    _use_wandb = bool(args.wandb_project) and _HAVE_WANDB
    if _use_wandb:
        run_name = args.wandb_run_name or f"rl-{int(time.time())}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            dir=args.out_dir,
        )
        print(f"[wandb] project={args.wandb_project} run={run_name}", flush=True)
    elif args.wandb_project and not _HAVE_WANDB:
        print("[wandb] wandb not installed; skipping", flush=True)

    # ---- log file
    diag_path = os.path.join(args.out_dir, "rl_diagnostics.jsonl")
    diag_f = open(diag_path, "a")

    step_idx = 0
    t0 = time.time()
    while step_idx < args.num_steps:
        # v5: DVI-style warmup — RL coefs = 0 during warmup, full afterwards.
        rl_coef_scale = 0.0 if step_idx < args.warmup_steps else 1.0

        # Pick batch of prompts
        batch_prompts = random.sample(prompts, args.batch_size)

        # Phase 1: rollouts (no grad needed for sampling τ).
        # Per prompt we prefill target ONCE (extract anchor hidden + cache), then
        # do M rollouts reusing both.
        all_rollouts = []  # list of (prompt_idx, rollout_dict)
        prompt_ids_list = []
        target_hidden_per_prompt = []
        past_kv_per_prompt = []
        with torch.no_grad():
            for prompt_i, p_text in enumerate(batch_prompts):
                prefix_ids = tokenize_prompt(p_text)
                prompt_ids_list.append(prefix_ids)
                past_kv, target_hidden_full = _target_prefill_and_extract_hidden(
                    target, drafter, prefix_ids,
                )
                target_hidden_per_prompt.append(target_hidden_full)
                past_kv_per_prompt.append(past_kv)
                for m in range(args.rollouts_per_prompt):
                    seed = step_idx * 10_000 + prompt_i * 100 + m
                    r = one_rollout(
                        drafter, target, tokenizer, prefix_ids,
                        target_hidden_full, past_kv, args, seed,
                    )
                    all_rollouts.append((prompt_i, r))

        # Phase 2: per-prompt advantages — GRPO (same-mean) or RLOO (leave-one-out).
        # LOO: for rollout i in a group of size M, baseline_i = mean of the other
        # M-1 rollouts. This is the standard RLOO baseline (Ahmadian et al. 2024)
        # and has strictly lower variance than the same-mean baseline when M>=2.
        by_prompt = defaultdict(list)
        for p_i, r in all_rollouts:
            by_prompt[p_i].append(r["tau"])
        prompt_sum = {p_i: float(sum(taus)) for p_i, taus in by_prompt.items()}
        prompt_M = {p_i: len(taus) for p_i, taus in by_prompt.items()}
        prompt_std = {p_i: float(np.std(taus)) + 1e-3 for p_i, taus in by_prompt.items()}
        prompt_mean = {p_i: float(np.mean(taus)) for p_i, taus in by_prompt.items()}
        advantages = []
        for p_i, r in all_rollouts:
            if args.loo_baseline and prompt_M[p_i] >= 2:
                loo_mean = (prompt_sum[p_i] - r["tau"]) / (prompt_M[p_i] - 1)
                advantages.append((r["tau"] - loo_mean) / prompt_std[p_i])
            else:
                advantages.append((r["tau"] - prompt_mean[p_i]) / prompt_std[p_i])

        # Phase 3: per-prompt loss (gradients only flow through this drafter call)
        opt.zero_grad()
        total_loss_scalar = 0.0
        pg_loss_scalar = 0.0
        clipped_frac_total = 0.0
        mean_entropy_scalar = 0.0
        ent_per_pos_accum = torch.zeros(args.block_size - 1, device=device)
        aux_ce_scalar = 0.0
        aux_tv_scalar = 0.0
        asym_ent_scalar = 0.0
        n_match_total = 0
        n_mismatch_total = 0
        pos0_top1_prob_accum = 0.0
        n_prompt_losses = 0
        for prompt_i in range(args.batch_size):
            prompt_rollouts = [r for (p_i, r) in all_rollouts if p_i == prompt_i]
            prompt_advs = [
                advantages[i] for i, (p_i, _) in enumerate(all_rollouts) if p_i == prompt_i
            ]
            if all(r["tau"] == 0 for r in prompt_rollouts):
                continue
            loss_dict = compute_loss(
                drafter, target, prompt_ids_list[prompt_i],
                target_hidden_per_prompt[prompt_i],
                prompt_rollouts, prompt_advs, args, args.block_size,
                rl_coef_scale=rl_coef_scale,
            )
            loss_dict["total_loss"].backward()
            total_loss_scalar += float(loss_dict["total_loss"].detach().item())
            pg_loss_scalar += float(loss_dict["pg_loss"].item())
            mean_entropy_scalar += float(loss_dict["mean_entropy"].item())
            ent_per_pos_accum += loss_dict["entropy_per_pos"]
            clipped_frac_total += float(loss_dict["clipped_frac"])
            aux_ce_scalar += float(loss_dict["aux_ce_loss"].item())
            aux_tv_scalar += float(loss_dict["aux_tv_loss"].item())
            asym_ent_scalar += float(loss_dict["asym_ent_loss"].item())
            n_match_total += int(loss_dict["n_match"])
            n_mismatch_total += int(loss_dict["n_mismatch"])
            pos0_top1_prob_accum += float(loss_dict["pos0_top1_prob"])
            n_prompt_losses += 1

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in drafter.parameters() if p.requires_grad],
            args.max_grad_norm,
        ).item()
        opt.step()

        # ---- Diagnostics
        all_taus = [r["tau"] for _, r in all_rollouts]
        diag = {
            "step": step_idx,
            "tau_mean": float(np.mean(all_taus)),
            "tau_std": float(np.std(all_taus)),
            "tau_min": int(np.min(all_taus)),
            "tau_max": int(np.max(all_taus)),
            "adv_mean": float(np.mean(advantages)),
            "adv_std": float(np.std(advantages)),
            "adv_nonzero_frac": float(np.mean([abs(a) > 1e-6 for a in advantages])),
            "total_loss": total_loss_scalar / max(n_prompt_losses, 1),
            "pg_loss": pg_loss_scalar / max(n_prompt_losses, 1),
            "aux_ce_loss": aux_ce_scalar / max(n_prompt_losses, 1),
            "aux_tv_loss": aux_tv_scalar / max(n_prompt_losses, 1),
            "asym_ent_loss": asym_ent_scalar / max(n_prompt_losses, 1),
            "mean_entropy": mean_entropy_scalar / max(n_prompt_losses, 1),
            "entropy_per_pos": (ent_per_pos_accum / max(n_prompt_losses, 1)).cpu().tolist(),
            "match_frac": n_match_total / max(n_match_total + n_mismatch_total, 1),
            "n_match": n_match_total,
            "n_mismatch": n_mismatch_total,
            "pos0_top1_prob": pos0_top1_prob_accum / max(n_prompt_losses, 1),
            "clipped_frac": clipped_frac_total / max(n_prompt_losses, 1),
            "grad_norm": float(grad_norm),
            "rl_coef_scale": float(rl_coef_scale),
            "elapsed_s": time.time() - t0,
        }
        diag_f.write(json.dumps(diag) + "\n"); diag_f.flush()
        if _use_wandb:
            wandb_payload = {k: v for k, v in diag.items() if k != "entropy_per_pos"}
            for i, h_i in enumerate(diag["entropy_per_pos"]):
                wandb_payload[f"entropy/pos_{i:02d}"] = h_i
            wandb.log(wandb_payload, step=step_idx)
        print(
            f"[step {step_idx:>4}] τ={diag['tau_mean']:.2f}±{diag['tau_std']:.2f} "
            f"[{diag['tau_min']}..{diag['tau_max']}] "
            f"loss={diag['total_loss']:+.4f} (pg={diag['pg_loss']:+.4f}) "
            f"H={diag['mean_entropy']:.3f} clip={diag['clipped_frac']:.2f} "
            f"|g|={diag['grad_norm']:.3f}",
            flush=True,
        )

        if (step_idx + 1) % args.save_every == 0:
            save_dir = os.path.join(args.out_dir, f"step_{step_idx+1}_hf")
            drafter.save_pretrained(save_dir)
            print(f"  saved {save_dir}", flush=True)

        step_idx += 1

    # Final save
    drafter.save_pretrained(os.path.join(args.out_dir, "final_hf"))
    diag_f.close()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
