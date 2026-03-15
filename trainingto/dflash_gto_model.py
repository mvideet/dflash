import copy
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
        )
        self.target_model.eval()
        for param in self.target_model.parameters():
            param.requires_grad = False

        self.draft_model = DFlashDraftModel.from_pretrained(
            draft_model_path,
            torch_dtype=torch.bfloat16,
        )
        self.draft_model.train()

        self.ref_draft_model = copy.deepcopy(self.draft_model)
        self.ref_draft_model.eval()
        for param in self.ref_draft_model.parameters():
            param.requires_grad = False

        self.block_size = self.draft_model.block_size
        self.mask_token_id = self.draft_model.mask_token_id
        self.target_layer_ids = self.draft_model.target_layer_ids

        self.dtk_k = getattr(config, "dtk_k", 3)
        self.dtk_tau = getattr(config, "dtk_tau", 2.0)
        self.gto_weight = getattr(config, "gto_weight", 0.5)

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
                    position_start, max_tree_size=8,
                    theta_uni=0.9, theta_bi=0.3, theta_tri=0.1):
        packed_ids, packed_pos, parent_idx, leaf_paths, leaf_tokens = \
            build_dynamic_tree(
                draft_logits.detach(),
                anchor_ids,
                theta_uni=theta_uni,
                theta_bi=theta_bi,
                theta_tri=theta_tri,
                max_tree_size=max_tree_size,
            ) #build the dynamic tree 

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
    # Step 5b: compute_dtk_loss (Decoupled Tempered Top-K distillation)
    # ------------------------------------------------------------------

    def compute_dtk_loss(self, draft_logits, target_logits, K=None, tau=None):
        """
        Decoupled Tempered Top-K distillation loss.
        draft_logits:  [B, L, V] -- draft model output (has grad)
        target_logits: [B, L, V] -- target model output (detached)
        Returns: scalar loss
        """
        K = K if K is not None else self.dtk_k
        tau = tau if tau is not None else self.dtk_tau

        eps = 1e-6
        target_logits_f = target_logits.float()
        draft_logits_f = draft_logits.float()

        log_p_t = F.log_softmax(target_logits_f, dim=-1).detach()
        log_p_s = F.log_softmax(draft_logits_f, dim=-1)

        _, topk_idx = torch.topk(target_logits_f, K, dim=-1)

        log_p_t_topk = log_p_t.gather(-1, topk_idx)
        log_p_s_topk = log_p_s.gather(-1, topk_idx)

        log_pt_T = torch.logsumexp(log_p_t_topk, dim=-1)
        log_ps_T = torch.logsumexp(log_p_s_topk, dim=-1)
        pt_T = log_pt_T.exp()
        ps_T = log_ps_T.exp()

        pt_T_c = pt_T.clamp(min=eps, max=1 - eps)
        ps_T_c = ps_T.clamp(min=eps, max=1 - eps)
        L_B = (
            pt_T_c * (torch.log(pt_T_c) - torch.log(ps_T_c))
            + (1 - pt_T_c) * (torch.log1p(-pt_T_c) - torch.log1p(-ps_T_c))
        )

        log_p_t_cond = log_p_t_topk - log_pt_T.unsqueeze(-1)
        log_p_s_cond = log_p_s_topk - log_ps_T.unsqueeze(-1)

        log_p_t_tau = F.log_softmax(log_p_t_cond / tau, dim=-1)
        log_p_s_tau = F.log_softmax(log_p_s_cond / tau, dim=-1)
        p_t_tau = log_p_t_tau.exp()

        kl_cond = (p_t_tau * (log_p_t_tau - log_p_s_tau)).sum(dim=-1)
        L_C = tau * tau * kl_cond

        L_DTK = L_B + pt_T.detach() * L_C
        return L_DTK.mean()

    def compute_topk_recall(self, draft_logits, target_logits, K=None):
        """Fraction of positions where argmax(target) is in top-K(draft)."""
        K = K if K is not None else self.dtk_k
        target_argmax = target_logits.argmax(dim=-1)
        _, draft_topk_idx = torch.topk(draft_logits, K, dim=-1)
        in_topk = (target_argmax.unsqueeze(-1) == draft_topk_idx).any(dim=-1)
        return in_topk.float().mean().item()

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
                     epsilon=0.2, delta=1e-6):
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
        advantages = advantages.clamp(-5.0, 5.0)

        log_ratio = group_probs - group_ref_probs
        log_ratio = log_ratio.clamp(-10.0, 10.0)
        ratio = torch.exp(log_ratio).clamp(0.1, 10.0)

        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
        gto_loss = -torch.min(unclipped, clipped).mean()
        return gto_loss

    # ------------------------------------------------------------------
    # Step 8: split_input_ids_to_groups
    # ------------------------------------------------------------------

    def split_input_ids_to_groups(self, input_ids, loss_mask, K=2, m=4):
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

        groups = self.split_input_ids_to_groups(input_ids, loss_mask)

        gto_loss_all = []
        group_rewards_all = []

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

                with torch.no_grad():
                    tree_logits, leaf_paths, leaf_tokens, best_leaf, n = \
                        self.tree_verify(
                            draft_logits, anchor_ids,
                            input_ids[:, :p + 1], p,
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

                group_rewards.append(r)
                group_probs.append(log_prob)
                group_ref_probs.append(ref_log_prob)

            if len(group_rewards) < 2:
                continue

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

        topk_recall_sum = 0.0
        topk_recall_count = 0
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
                dtk = self.compute_dtk_loss(dl, tl, K=self.dtk_k, tau=self.dtk_tau)
                ploss_terms.append(dtk)
                with torch.no_grad():
                    topk_recall_sum += self.compute_topk_recall(dl, tl, K=self.dtk_k)
                    topk_recall_count += 1

            if ploss_terms:
                ploss = torch.stack(ploss_terms).mean()
                topk_recall = topk_recall_sum / topk_recall_count if topk_recall_count else 0.0
            else:
                ploss = torch.tensor(0.0, device=device, requires_grad=True)
                topk_recall = 0.0
        else:
            ploss = torch.tensor(0.0, device=device, requires_grad=True)
            topk_recall = 0.0

        if gto_loss_all:
            gto_loss = torch.stack(gto_loss_all).mean()
            group_rewards_mean = torch.tensor(group_rewards_all).mean()
        else:
            gto_loss = torch.tensor(0.0, device=device)
            group_rewards_mean = torch.tensor(0.0)

        total_loss = ploss + self.gto_weight * gto_loss
        return total_loss, ploss, gto_loss, group_rewards_mean, topk_recall
