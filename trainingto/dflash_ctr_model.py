"""
DFlash CTR (Conditional Tree Refinement) training model.

Two-pass training for block diffusion drafters:
  Pass 1 (bidirectional): standard CE/KL on marginal logits (existing)
  Pass 2 (tree attention): CE/KL on CONDITIONAL logits at tree nodes

The tree attention pass teaches the draft to produce good predictions when
it can only see ancestor nodes (as in EAGLE-2 tree verification), not all
block positions. This closes the gap between marginal and conditional
prediction quality.

At inference, the trained model supports a second draft pass with tree
attention that refines the tree with branch-specific conditional logits.
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.dflash_tree import (
    build_dynamic_tree_v2,
    create_tree_attention_mask_dynamic,
)
from base_model import DFlashTrainBase

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


def _rank0():
    try:
        return torch.distributed.get_rank() == 0
    except Exception:
        return True


class DFlashCTRModel(DFlashTrainBase):
    def __init__(self, config, target_model_path, draft_model_path):
        super().__init__(config, target_model_path, draft_model_path)
        self._step_count = 0

    def draft_forward_tree(self, target_hidden, packed_ids, packed_pos_relative,
                           parent_idx, position_start):
        """
        Second draft pass with tree attention.

        Instead of bidirectional attention over a linear block, uses the tree
        attention mask so each node only sees its ancestors. This produces
        CONDITIONAL logits at each tree node.

        Returns logits [1, L, V] for all L tree nodes.
        """
        device = packed_ids.device
        ctx_len = target_hidden.shape[1]
        L = packed_ids.shape[1]

        tree_noise = self.target_model.model.embed_tokens(packed_ids)  # [1, L, H]
        tree_pos = (packed_pos_relative + position_start).clone()       # [1, L]

        tree_mask = create_tree_attention_mask_dynamic(
            packed_pos_relative, parent_idx, prefix_len=ctx_len,
        )  # [1, 1, L, ctx_len + L]

        saved_attn = self.draft_model.config._attn_implementation
        self.draft_model.config._attn_implementation = "sdpa"

        tree_hidden = self.draft_model(
            target_hidden=target_hidden,
            noise_embedding=tree_noise,
            position_ids=tree_pos,
            attention_mask=tree_mask,
            use_cache=False,
            is_causal=False,
        )  # [1, L, H]

        self.draft_model.config._attn_implementation = saved_attn

        tree_logits = self.target_model.lm_head(tree_hidden)  # [1, L, V]
        return tree_logits

    def forward(self, input_ids, attention_mask, loss_mask):
        device = input_ids.device
        all_target_hidden, target_logits = self.dataprepare(input_ids, attention_mask)

        ctr_weight = getattr(self.config, "ctr_weight", 0.5)
        tree_expand_k = getattr(self.config, "tree_expand_k", 5)
        max_tree_size = getattr(self.config, "max_tree_size", 16)

        positions = self.iter_block_positions(input_ids, loss_mask)

        marginal_loss_terms = []
        ctr_loss_terms = []
        marginal_accs = []
        ctr_accs = []

        for p in positions:
            target_hidden_p = all_target_hidden[:, :p + 1, :]
            anchor_ids = input_ids[:, p:p + 1]

            # === Pass 1: Standard bidirectional (marginal logits) ===
            dl = self.draft_forward(target_hidden_p, anchor_ids, p)
            tl = target_logits[:, p:p + self.block_size - 1, :]

            actual_len = min(dl.shape[1], tl.shape[1])
            if actual_len == 0:
                continue
            dl_block = dl[:, :actual_len, :]
            tl_block = tl[:, :actual_len, :]

            target_probs = F.softmax(tl_block.float(), dim=-1).detach()
            draft_log_probs = F.log_softmax(dl_block.float(), dim=-1)
            marginal_kl = -(target_probs * draft_log_probs).sum(-1).mean()
            marginal_loss_terms.append(marginal_kl)

            with torch.no_grad():
                m_acc = (dl_block.argmax(-1) == tl_block.argmax(-1)).float().mean().item()
                marginal_accs.append(m_acc)

            # === Pass 2: Tree attention (conditional logits) ===
            with torch.no_grad():
                packed_ids, packed_pos_rel, parent_idx, leaf_paths, leaf_tokens = \
                    build_dynamic_tree_v2(
                        dl_block.detach(),
                        anchor_ids,
                        max_tree_size=max_tree_size,
                        expand_k=tree_expand_k,
                    )

            tree_logits = self.draft_forward_tree(
                target_hidden_p, packed_ids, packed_pos_rel, parent_idx, p,
            )  # [1, L, V]

            # Compute CE loss on tree nodes (skip root which is the anchor).
            # Each node at depth d predicts the token at position p+d.
            # Target labels: input_ids at those positions.
            L = packed_ids.shape[1]
            depths = packed_pos_rel[0]  # [L]

            tree_loss_sum = torch.tensor(0.0, device=device)
            tree_count = 0
            tree_correct = 0

            for ni in range(1, L):
                d = depths[ni].item()
                gt_pos = p + d
                if gt_pos >= input_ids.shape[1]:
                    continue

                # Target: what the target model predicts at this position
                if p + d - 1 < target_logits.shape[1]:
                    node_target_logits = target_logits[:, p + d - 1, :]  # predicts token at p+d
                    node_target_probs = F.softmax(node_target_logits.float(), dim=-1).detach()

                    node_draft_log_probs = F.log_softmax(tree_logits[:, ni, :].float(), dim=-1)
                    node_kl = -(node_target_probs * node_draft_log_probs).sum(-1)
                    tree_loss_sum = tree_loss_sum + node_kl.squeeze()
                    tree_count += 1

                    with torch.no_grad():
                        if tree_logits[0, ni, :].argmax() == node_target_logits[0].argmax():
                            tree_correct += 1

            if tree_count > 0:
                ctr_loss = tree_loss_sum / tree_count
                ctr_loss_terms.append(ctr_loss)
                ctr_accs.append(tree_correct / tree_count)

        # --- Aggregate ---
        if marginal_loss_terms:
            marginal_loss = torch.stack(marginal_loss_terms).mean()
        else:
            marginal_loss = torch.tensor(0.0, device=device, requires_grad=True)

        if ctr_loss_terms:
            ctr_loss = torch.stack(ctr_loss_terms).mean()
        else:
            ctr_loss = torch.tensor(0.0, device=device, requires_grad=True)

        total_loss = marginal_loss + ctr_weight * ctr_loss

        avg_marginal_acc = sum(marginal_accs) / len(marginal_accs) if marginal_accs else 0
        avg_ctr_acc = sum(ctr_accs) / len(ctr_accs) if ctr_accs else 0

        if _rank0() and self._step_count % 10 == 0:
            print(f"[CTR] step={self._step_count} "
                  f"marginal_loss={marginal_loss.item():.4f} "
                  f"ctr_loss={ctr_loss.item():.4f} "
                  f"total={total_loss.item():.4f} "
                  f"marginal_acc={avg_marginal_acc:.3f} "
                  f"ctr_acc={avg_ctr_acc:.3f}")

        if _HAS_WANDB and wandb.run is not None and _rank0():
            wandb.log({
                "train/step": self._step_count,
                "loss/total": total_loss.item(),
                "loss/marginal": marginal_loss.item(),
                "loss/ctr": ctr_loss.item(),
                "acc/marginal": avg_marginal_acc,
                "acc/ctr": avg_ctr_acc,
                "acc/gap": avg_marginal_acc - avg_ctr_acc,
            })

        self._step_count += 1
        return total_loss, marginal_loss, ctr_loss, torch.tensor(avg_ctr_acc)
