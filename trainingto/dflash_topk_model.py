"""
DFlash training model with Top-K Recall loss for tree-aware drafting.

The draft model's job during v4 inference is to produce logits where the
target's greedy token falls within the draft's top-K at every position.
Standard KL distillation wastes capacity on the long tail of the vocabulary.
This module focuses training on what matters: top-K recall.

Two loss components:
  L_distill: standard forward KL (target || draft) — baseline distillation
  L_topk:    weighted cross-entropy that upweights positions where the
             target token is near or outside the draft's top-K boundary

The weighting scheme:
  - target token is draft rank #1       → weight = w_min (already fine)
  - target token is draft rank #K       → weight = 1.0 (boundary — fragile)
  - target token is draft rank > K      → weight = w_max (failing — needs work)

This directly optimizes the bottleneck for v4: if the target token isn't in
the draft's top-K at position d, no branch in the tree can match at depth d.
"""

import os

import torch
import torch.nn.functional as F

from base_model import DFlashTrainBase


class DFlashTopKModel(DFlashTrainBase):
    def __init__(self, config, target_model_path, draft_model_path):
        super().__init__(config, target_model_path, draft_model_path)

        self.top_k = getattr(config, "top_k", 3)
        self.topk_weight = getattr(config, "topk_weight", 1.0)
        self.w_min = getattr(config, "w_min", 0.1)
        self.w_max = getattr(config, "w_max", 2.0)

    def compute_topk_weight(self, draft_logits, target_argmax):
        """
        Compute per-position weights based on where the target token
        ranks in the draft's sorted logits.

        rank 1..K-1   → linearly interpolate w_min to 1.0
        rank K        → 1.0 (boundary)
        rank K+1..V   → w_max (outside top-K, tree can't cover)
        """
        K = self.top_k
        B_minus_1 = draft_logits.shape[0]

        _, sorted_indices = draft_logits.float().sort(dim=-1, descending=True)
        ranks = torch.zeros(B_minus_1, device=draft_logits.device, dtype=torch.long)
        for j in range(B_minus_1):
            match = (sorted_indices[j] == target_argmax[j]).nonzero(as_tuple=True)[0]
            ranks[j] = match[0] if match.numel() > 0 else K + 1

        weights = torch.ones(B_minus_1, device=draft_logits.device)
        in_topk = ranks < K
        outside_topk = ranks >= K

        weights[in_topk] = self.w_min + (1.0 - self.w_min) * (ranks[in_topk].float() / K)
        weights[outside_topk] = self.w_max

        return weights, ranks

    def forward(self, input_ids, attention_mask, loss_mask):
        device = input_ids.device
        all_target_hidden, target_logits = self.dataprepare(input_ids, attention_mask)

        positions = self.iter_block_positions(input_ids, loss_mask)

        distill_terms = []
        topk_terms = []
        topk_recalls = []
        acceptance_lengths = []

        for p in positions:
            target_hidden_p = all_target_hidden[:, :p + 1, :]
            anchor_ids = input_ids[:, p:p + 1]

            dl = self.draft_forward(target_hidden_p, anchor_ids, p)
            tl = target_logits[:, p:p + self.block_size - 1, :]
            actual_len = min(dl.shape[1], tl.shape[1])
            if actual_len == 0:
                continue
            dl_block = dl[0, :actual_len]  # [L, V]
            tl_block = tl[0, :actual_len]  # [L, V]

            target_probs = F.softmax(tl_block.float(), dim=-1).detach()
            draft_log_probs = F.log_softmax(dl_block.float(), dim=-1)
            distill = -(target_probs * draft_log_probs).sum(-1).mean()
            distill_terms.append(distill)

            target_argmax = tl_block.argmax(dim=-1).detach()
            weights, ranks = self.compute_topk_weight(dl_block.detach(), target_argmax)

            ce_per_pos = F.cross_entropy(
                dl_block.float(), target_argmax, reduction="none",
            )
            topk_loss = (weights.detach() * ce_per_pos).mean()
            topk_terms.append(topk_loss)

            with torch.no_grad():
                recall = (ranks < self.top_k).float().mean().item()
                topk_recalls.append(recall)

                draft_argmax = dl_block.argmax(dim=-1)
                matches = (draft_argmax == target_argmax).float()
                acc_len = int(matches.cumprod(0).sum().item())
                acceptance_lengths.append(acc_len)

        if distill_terms:
            distill_loss = torch.stack(distill_terms).mean()
        else:
            distill_loss = None

        if topk_terms:
            topk_loss = torch.stack(topk_terms).mean()
            avg_recall = sum(topk_recalls) / len(topk_recalls)
            avg_acceptance = sum(acceptance_lengths) / len(acceptance_lengths)
        else:
            topk_loss = None
            avg_recall = 0.0
            avg_acceptance = 0.0

        if distill_loss is not None and topk_loss is not None:
            total_loss = distill_loss + self.topk_weight * topk_loss
        elif distill_loss is not None:
            total_loss = distill_loss
        elif topk_loss is not None:
            total_loss = self.topk_weight * topk_loss
        else:
            total_loss = sum(p.float().sum() * 0.0 for p in self.draft_model.parameters()
                            if p.requires_grad)

        if distill_loss is None:
            distill_loss = torch.tensor(0.0, device=device)
        if topk_loss is None:
            topk_loss = torch.tensor(0.0, device=device)

        if os.environ.get("DEBUG_TOPK", ""):
            try:
                rank = torch.distributed.get_rank()
            except Exception:
                rank = 0
            if rank == 0:
                print(
                    f"[TOPK] distill={distill_loss.item():.4f} "
                    f"topk={topk_loss.item():.4f} "
                    f"recall@{self.top_k}={avg_recall:.3f} "
                    f"avg_accept={avg_acceptance:.2f} "
                    f"total={total_loss.item():.4f}"
                )

        return total_loss, distill_loss, topk_loss, torch.tensor(avg_acceptance)
