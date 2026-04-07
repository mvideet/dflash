import copy
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.dflash_tree import (
    build_dynamic_tree,
    create_tree_attention_mask_dynamic,
    select_best_dynamic_leaf,
)
from base_model import DFlashTrainBase


def segments_overlap(seg1, seg2):
    return not (seg1[1] < seg2[0] or seg2[1] < seg1[0])


class DFlashGTOModel(DFlashTrainBase):
    def __init__(self, config, target_model_path, draft_model_path):
        super().__init__(config, target_model_path, draft_model_path)

        if getattr(config, "gto_weight", 0.5) > 0:
            self.ref_draft_model = copy.deepcopy(self.draft_model)
            self.ref_draft_model.eval()
            for param in self.ref_draft_model.parameters():
                param.requires_grad = False
        else:
            self.ref_draft_model = None

    def draft_forward(self, target_hidden, anchor_ids, position_start, ref=False):
        if not ref:
            return super().draft_forward(target_hidden, anchor_ids, position_start)

        device = anchor_ids.device
        mask_tokens = torch.full(
            (1, self.block_size - 1), self.mask_token_id,
            dtype=torch.long, device=device,
        )
        block_ids = torch.cat([anchor_ids, mask_tokens], dim=1)
        noise_embedding = self.target_model.model.embed_tokens(block_ids)
        position_ids = torch.arange(
            position_start, position_start + self.block_size,
            device=device,
        ).unsqueeze(0)

        draft_hidden = self.ref_draft_model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids,
        )[:, -self.block_size + 1:, :]

        draft_logits = self.target_model.lm_head(draft_hidden)
        return draft_logits

    # ------------------------------------------------------------------
    # tree_verify
    # ------------------------------------------------------------------

    @torch.no_grad()
    def tree_verify(self, draft_logits, anchor_ids, input_ids_prefix,
                    position_start, max_tree_size=8, top_k=5,
                    theta_uni=0.9, theta_bi=0.3, theta_tri=0.1):
        packed_ids, packed_pos, parent_idx, leaf_paths, leaf_tokens = \
            build_dynamic_tree(
                draft_logits.detach(),
                anchor_ids,
                theta_uni=theta_uni,
                theta_bi=theta_bi,
                theta_tri=theta_tri,
                max_tree_size=max_tree_size,
                top_k=top_k,
            )

        tree_mask = create_tree_attention_mask_dynamic(
            packed_pos, parent_idx, prefix_len=input_ids_prefix.shape[1],
        )

        packed_pos_abs = packed_pos + position_start

        saved_attn = self.target_model.config._attn_implementation
        self.target_model.config._attn_implementation = "sdpa"

        prefix_cache = DynamicCache()
        prefix_pos = torch.arange(
            input_ids_prefix.shape[1], device=input_ids_prefix.device,
        ).unsqueeze(0)
        self.target_model(
            input_ids_prefix,
            position_ids=prefix_pos,
            past_key_values=prefix_cache,
            use_cache=True,
        )

        tree_output = self.target_model(
            packed_ids,
            position_ids=packed_pos_abs,
            past_key_values=prefix_cache,
            use_cache=False,
            attention_mask=tree_mask,
        )
        self.target_model.config._attn_implementation = saved_attn

        tree_logits = tree_output.logits
        best_leaf, n = select_best_dynamic_leaf(
            tree_logits, leaf_paths, leaf_tokens, temperature=0.0,
        )
        return tree_logits, leaf_paths, leaf_tokens, best_leaf, n

    # ------------------------------------------------------------------
    # compute_tree_reward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_tree_reward(self, logits, leaf_paths, leaf_tokens, eta=1.0):
        prev_nodes = leaf_paths[:, :-1]
        realized = leaf_tokens
        n_leaves, depth = prev_nodes.shape
        V = logits.size(-1)

        gathered = logits[0].index_select(
            0, prev_nodes.reshape(-1),
        ).view(n_leaves, depth, V)
        pred = gathered.argmax(dim=-1)
        matches = (pred == realized)
        per_leaf_lengths = matches.cumprod(dim=1).sum(dim=1).float()

        max_len = per_leaf_lengths.max()
        aggregated = max_len + (1.0 / eta) * torch.log(
            torch.sum(torch.exp(eta * (per_leaf_lengths - max_len))) + 1e-12
        )
        best_leaf = int(per_leaf_lengths.argmax().item())
        best_n = int(per_leaf_lengths[best_leaf].item())
        return aggregated.item(), best_leaf, best_n

    # ------------------------------------------------------------------
    # compute_path_logprob
    # ------------------------------------------------------------------

    def compute_path_logprob(self, draft_logits, accepted_tokens):
        n = accepted_tokens.shape[0]
        log_probs = F.log_softmax(draft_logits[0, :n, :], dim=-1)
        return log_probs.gather(1, accepted_tokens.unsqueeze(1)).squeeze(1).sum()

    # ------------------------------------------------------------------
    # get_gto_loss
    # ------------------------------------------------------------------

    def get_gto_loss(self, group_rewards, group_probs, group_ref_probs,
                     epsilon=0.1, delta=1e-6):
        device = group_probs[0].device if isinstance(group_probs, list) else group_probs.device

        if isinstance(group_rewards, list):
            group_rewards = torch.tensor(group_rewards, device=device, dtype=torch.float)
        if isinstance(group_probs, list):
            group_probs = torch.stack(group_probs).to(device)
        if isinstance(group_ref_probs, list):
            group_ref_probs = torch.stack(group_ref_probs).to(device)

        group_mean = group_rewards.mean()
        group_std = group_rewards.std()
        advantages = (group_rewards - group_mean) / (group_std + delta)
        advantages = advantages.clamp(-3.0, 3.0)

        if os.environ.get("DEBUG_RL", ""):
            try:
                rank = torch.distributed.get_rank()
            except Exception:
                rank = 0
            if rank == 0:
                print(f"[RL] advantages={advantages.tolist()}")

        log_ratio = group_probs - group_ref_probs
        log_ratio = log_ratio.clamp(-10.0, 10.0)
        ratio = torch.exp(log_ratio).clamp(0.1, 10.0)

        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
        gto_loss = -torch.min(unclipped, clipped).mean()
        if os.environ.get("DEBUG_RL", ""):
            try:
                rank = torch.distributed.get_rank()
            except Exception:
                rank = 0
            if rank == 0:
                print(f"[RL] gto_loss={gto_loss.item():.4f} ratio_mean={ratio.mean().item():.4f}")
        return gto_loss

    # ------------------------------------------------------------------
    # split_input_ids_to_groups
    # ------------------------------------------------------------------

    def split_input_ids_to_groups(self, input_ids, loss_mask, K=2, m=8):
        lm = self.normalize_loss_mask(loss_mask)

        seq_len = lm.shape[0]
        valid_segments = []
        for i in range(seq_len - m + 1):
            if torch.all(lm[i:i + m] != 0):
                valid_segments.append((i, i + m))

        selected = []
        available = valid_segments.copy()
        for _ in range(K):
            if not available:
                break
            chosen = random.choice(available)
            selected.append(chosen)
            available = [s for s in available if not segments_overlap(s, chosen)]
        return selected

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, input_ids, attention_mask, loss_mask):
        device = input_ids.device
        all_target_hidden, target_logits = self.dataprepare(input_ids, attention_mask)

        gto_weight = getattr(self.config, "gto_weight", 0.5)

        gto_loss_all = []
        group_rewards_all = []

        if gto_weight > 0:
            K = getattr(self.config, "dtk_k", 2)
            seq_hash = int(input_ids.sum().item()) % (2**31)
            random.seed(seq_hash)
            m = getattr(self.config, "group_m", 8)
            groups = self.split_input_ids_to_groups(input_ids, loss_mask, K=K, m=m)

            for (start, end) in groups:
                group_rewards = []
                group_probs = []
                group_ref_probs = []

                for p in range(start, end):
                    target_hidden_p = all_target_hidden[:, :p + 1, :]
                    anchor_ids = input_ids[:, p:p + 1]

                    draft_logits = self.draft_forward(
                        target_hidden_p, anchor_ids, p, ref=False,
                    )

                    max_tree_size = getattr(self.config, "max_tree_size", 8)
                    top_k = getattr(self.config, "tree_top_k", 5)
                    with torch.no_grad():
                        tree_logits, leaf_paths, leaf_tokens, best_leaf, n = \
                            self.tree_verify(
                                draft_logits, anchor_ids,
                                input_ids[:, :p + 1], p,
                                max_tree_size=max_tree_size,
                                top_k=top_k,
                            )

                    if n < 1:
                        continue

                    reward, _, _ = self.compute_tree_reward(
                        tree_logits, leaf_paths, leaf_tokens,
                    )

                    with torch.no_grad():
                        ref_logits = self.draft_forward(
                            target_hidden_p, anchor_ids, p, ref=True,
                        )
                        ref_tree_logits, ref_leaf_paths, ref_leaf_tokens, _, _ = \
                            self.tree_verify(
                                ref_logits, anchor_ids,
                                input_ids[:, :p + 1], p,
                                max_tree_size=max_tree_size,
                                top_k=top_k,
                            )
                        ref_reward, _, _ = self.compute_tree_reward(
                            ref_tree_logits, ref_leaf_paths, ref_leaf_tokens,
                        )

                    accepted_tokens = leaf_tokens[best_leaf, :n]
                    log_prob = self.compute_path_logprob(draft_logits, accepted_tokens)
                    with torch.no_grad():
                        ref_log_prob = self.compute_path_logprob(ref_logits, accepted_tokens)

                    r = reward - ref_reward
                    if np.isnan(r):
                        continue

                    if os.environ.get("DEBUG_RL", ""):
                        try:
                            rank = torch.distributed.get_rank()
                        except Exception:
                            rank = 0
                        if rank == 0:
                            logit_diff = (draft_logits - ref_logits).abs().max().item()
                            trees_same = torch.equal(leaf_tokens, ref_leaf_tokens)
                            print(f"[RL] p={p} reward={reward:.4f} ref_reward={ref_reward:.4f} advantage={r:.4f} best_n={n} logit_maxdiff={logit_diff:.6f} trees_same={trees_same}")

                    group_rewards.append(r)
                    group_probs.append(log_prob)
                    group_ref_probs.append(ref_log_prob)

                if len(group_rewards) < 2:
                    continue

                if os.environ.get("DEBUG_RL", ""):
                    try:
                        rank = torch.distributed.get_rank()
                    except Exception:
                        rank = 0
                    if rank == 0:
                        print(f"[RL] group [{start},{end}) rewards={[f'{x:.4f}' for x in group_rewards]} mean={np.mean(group_rewards):.4f}")

                gto_loss = self.get_gto_loss(group_rewards, group_probs, group_ref_probs)
                gto_loss_all.append(gto_loss)
                group_rewards_all.append(torch.tensor(group_rewards).mean().item())

        ploss_positions = self.iter_block_positions(input_ids, loss_mask)

        if ploss_positions:
            ploss_terms = []
            for p in ploss_positions:
                target_hidden_p = all_target_hidden[:, :p + 1, :]
                anchor_ids = input_ids[:, p:p + 1]
                dl = self.draft_forward(target_hidden_p, anchor_ids, p, ref=False)
                tl = target_logits[:, p:p + self.block_size - 1, :]
                actual_len = min(dl.shape[1], tl.shape[1])
                if actual_len == 0:
                    continue
                dl = dl[:, :actual_len, :]
                tl = tl[:, :actual_len, :]
                target_probs = F.softmax(tl.float(), dim=-1).detach()
                draft_log_probs = F.log_softmax(dl.float(), dim=-1)
                kl = -(target_probs * draft_log_probs).sum(-1).mean()
                ploss_terms.append(kl)

            if ploss_terms:
                ploss = torch.stack(ploss_terms).mean()
            else:
                ploss = None
        else:
            ploss = None

        if gto_loss_all:
            gto_loss = torch.stack(gto_loss_all).mean()
            group_rewards_mean = torch.tensor(group_rewards_all).mean()
        else:
            gto_loss = None
            group_rewards_mean = torch.tensor(0.0)

        if ploss is not None and gto_loss is not None:
            total_loss = ploss + gto_weight * gto_loss
        elif ploss is not None:
            total_loss = ploss
        elif gto_loss is not None:
            total_loss = gto_weight * gto_loss
        else:
            total_loss = sum(p.float().sum() * 0.0 for p in self.draft_model.parameters()
                            if p.requires_grad)

        if ploss is None:
            ploss = torch.tensor(0.0, device=device)
        if gto_loss is None:
            gto_loss = torch.tensor(0.0, device=device)

        return total_loss, ploss, gto_loss, group_rewards_mean
