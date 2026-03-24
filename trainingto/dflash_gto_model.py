import copy
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, DynamicCache

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.dflash import DFlashDraftModel
from model.dflash_tree import (
    build_dynamic_tree,
    create_tree_attention_mask_dynamic,
    select_best_dynamic_leaf,
)
from model.utils import extract_context_feature


def segments_overlap(seg1, seg2):
    return not (seg1[1] < seg2[0] or seg2[1] < seg1[0])


class DFlashGTOModel(nn.Module):
    def __init__(self, config, target_model_path, draft_model_path):
        super().__init__()
        self.config = config

        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_path,
            torch_dtype=torch.bfloat16,
            output_hidden_states=True,
            attn_implementation="flash_attention_2",
        )
        self.target_model.eval()
        for param in self.target_model.parameters():
            param.requires_grad = False

        self.draft_model = DFlashDraftModel.from_pretrained(
            draft_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.draft_model.train()

        if getattr(config, "gto_weight", 0.5) > 0:
            self.ref_draft_model = copy.deepcopy(self.draft_model)
            self.ref_draft_model.eval()
            for param in self.ref_draft_model.parameters():
                param.requires_grad = False
        else:
            self.ref_draft_model = None

        self.block_size = self.draft_model.block_size
        self.mask_token_id = self.draft_model.mask_token_id
        self.target_layer_ids = self.draft_model.target_layer_ids

    # ------------------------------------------------------------------
    # Step 2: dataprepare
    # ------------------------------------------------------------------

    @torch.no_grad()
    def dataprepare(self, input_ids, attention_mask):
        output = self.target_model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        target_hidden = extract_context_feature(
            output.hidden_states, self.target_layer_ids,
        )
        target_logits = output.logits
        return target_hidden, target_logits

    # ------------------------------------------------------------------
    # Step 3: draft_forward
    # ------------------------------------------------------------------

    def draft_forward(self, target_hidden, anchor_ids, position_start, ref=False):
        device = anchor_ids.device
        mask_tokens = torch.full(
            (1, self.block_size - 1), self.mask_token_id,
            dtype=torch.long, device=device,
        ) # [1, 15]
        block_ids = torch.cat([anchor_ids, mask_tokens], dim=1) # [1,16] get the block size
        noise_embedding = self.target_model.model.embed_tokens(block_ids) # [1,16,768] get the noise embedding
        position_ids = torch.arange(
            position_start, position_start + self.block_size,
            device=device,
        ).unsqueeze(0)
        # [1,16]
        model = self.ref_draft_model if ref else self.draft_model
        draft_hidden = model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids,
        )[:, -self.block_size + 1:, :] # [1,15,768] get the draft hidden

        draft_logits = self.target_model.lm_head(draft_hidden)
        return draft_logits

    # ------------------------------------------------------------------
    # Step 4: tree_verify
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
        ) #create the attnetion tree mask based on the positions and the parent indicies

        packed_pos_abs = packed_pos + position_start # [1,16] get the absolute positions

        saved_attn = self.target_model.config._attn_implementation
        self.target_model.config._attn_implementation = "sdpa"

        prefix_cache = DynamicCache() #create the prefix cache
        prefix_pos = torch.arange(
            input_ids_prefix.shape[1], device=input_ids_prefix.device,
        ).unsqueeze(0) #get the prefix positions
        self.target_model(
            input_ids_prefix,
            position_ids=prefix_pos,
            past_key_values=prefix_cache,
            use_cache=True,
        ) #forward the prefix through the target model

        tree_output = self.target_model(
            packed_ids,
            position_ids=packed_pos_abs,
            past_key_values=prefix_cache,
            use_cache=False,
            attention_mask=tree_mask,
        ) #forward the packed ids through the target model
        self.target_model.config._attn_implementation = saved_attn

        tree_logits = tree_output.logits
        best_leaf, n = select_best_dynamic_leaf(
            tree_logits, leaf_paths, leaf_tokens, temperature=0.0,
        ) #select the best leaf based ont he tree lgoits 
        return tree_logits, leaf_paths, leaf_tokens, best_leaf, n

    # ------------------------------------------------------------------
    # Step 5: compute_tree_reward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_tree_reward(self, logits, leaf_paths, leaf_tokens, eta=1.0):
        prev_nodes = leaf_paths[:, :-1] # get all the previous nodes
        realized = leaf_tokens # get all the realized tokens
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
    # Step 6: compute_path_logprob
    # ------------------------------------------------------------------

    def compute_path_logprob(self, draft_logits, accepted_tokens):
        n = accepted_tokens.shape[0]
        log_probs = F.log_softmax(draft_logits[0, :n, :], dim=-1)
        return log_probs.gather(1, accepted_tokens.unsqueeze(1)).squeeze(1).sum()

    # ------------------------------------------------------------------
    # Step 7: get_gto_loss
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
    # Step 8: split_input_ids_to_groups
    # ------------------------------------------------------------------

    def split_input_ids_to_groups(self, input_ids, loss_mask, K=2, m=8):
        if loss_mask.dim() == 3:
            loss_mask = loss_mask.squeeze(-1)
        if loss_mask.dim() == 2 and loss_mask.shape[0] == 1:
            loss_mask = loss_mask.squeeze(0)

        seq_len = loss_mask.shape[0]
        valid_segments = []
        for i in range(seq_len - m + 1):
            if torch.all(loss_mask[i:i + m] != 0):
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
    # Step 9: forward
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
            groups = self.split_input_ids_to_groups(input_ids, loss_mask, K=K)

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

        # -- Distillation loss (ploss) across block-aligned positions --
        ploss_positions = []
        lm = loss_mask
        if lm.dim() == 3:
            lm = lm.squeeze(-1)
        if lm.dim() == 2:
            lm = lm[0]

        stride = max(1, self.block_size // 2)
        for p in range(0, input_ids.shape[1] - self.block_size, stride):
            if lm[p].item() != 0:
                ploss_positions.append(p)

        if ploss_positions:
            ploss_terms = []
            for p in ploss_positions:
                target_hidden_p = all_target_hidden[:, :p + 1, :]
                anchor_ids = input_ids[:, p:p + 1]
                dl = self.draft_forward(target_hidden_p, anchor_ids, p, ref=False)
                tl = target_logits[:, p + 1:p + self.block_size, :]
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
                ploss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            ploss = torch.tensor(0.0, device=device, requires_grad=True)

        if gto_loss_all:
            gto_loss = torch.stack(gto_loss_all).mean()
            group_rewards_mean = torch.tensor(group_rewards_all).mean()
        else:
            gto_loss = torch.tensor(0.0, device=device)
            group_rewards_mean = torch.tensor(0.0)

        total_loss = ploss + gto_weight * gto_loss
        return total_loss, ploss, gto_loss, group_rewards_mean
