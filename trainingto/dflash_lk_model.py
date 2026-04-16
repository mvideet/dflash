"""
DFlash training model with LK losses for direct acceptance rate optimization.

Implements two loss variants from "LK Losses: Direct Acceptance Rate
Optimization for Speculative Decoding" (Samarin et al., 2026):

  L_LK^alpha  = -log(sum(min(p, q)))       [negative log-acceptance]
  L_LK^lambda = lambda*KL + (1-lambda)*TV  [hybrid with adaptive schedule]

where lambda = exp(-eta * stop_grad(alpha)) transitions from KL-dominated
(stable early training) to TV-dominated (direct acceptance optimization)
as the draft model improves.
"""

import torch
import torch.nn.functional as F

from base_model import DFlashTrainBase


class DFlashLKModel(DFlashTrainBase):

    def __init__(self, config, target_model_path, draft_model_path):
        super().__init__(config, target_model_path, draft_model_path)
        self.eta = getattr(config, "lk_eta", 3.0)
        self.loss_mode = getattr(config, "lk_mode", "hybrid")
        self.pos_decay = getattr(config, "pos_decay", 1.0)

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    @staticmethod
    def acceptance_rate(target_probs, draft_probs):
        """Per-position acceptance rate: alpha = sum_v min(p(v), q(v))."""
        return torch.sum(torch.min(target_probs, draft_probs), dim=-1)

    @staticmethod
    def lk_alpha_loss(target_probs, draft_probs):
        """Negative log-acceptance: -log(alpha). Gradient = (1/alpha)*grad_TV."""
        alpha = torch.sum(torch.min(target_probs, draft_probs), dim=-1)
        return -torch.log(alpha.clamp(min=1e-10))

    @staticmethod
    def tv_distance(target_probs, draft_probs):
        """Total variation: 0.5 * sum|p - q|. Equivalently 1 - alpha."""
        return 0.5 * torch.sum(torch.abs(target_probs - draft_probs), dim=-1)

    @staticmethod
    def forward_kl(target_probs, draft_log_probs):
        """Forward KL(p||q) = -sum(p * log q) + const."""
        return -(target_probs * draft_log_probs).sum(dim=-1)

    def lk_hybrid_loss(self, target_probs, draft_probs, draft_log_probs):
        """Hybrid LK^lambda: adaptive blend of KL and TV.

        lambda = exp(-eta * sg[alpha]).  When alpha is low (poor draft),
        lambda -> 1 and KL dominates (stable gradients).  As alpha improves,
        lambda -> 0 and TV dominates (direct acceptance optimization).
        """
        alpha = self.acceptance_rate(target_probs, draft_probs)
        lam = torch.exp(-self.eta * alpha.detach())

        kl = self.forward_kl(target_probs, draft_log_probs)
        tv = self.tv_distance(target_probs, draft_probs)

        return lam * kl + (1.0 - lam) * tv, alpha

    # ------------------------------------------------------------------
    # Position weighting
    # ------------------------------------------------------------------

    def position_weights(self, seq_len, device):
        """Exponential decay: position d gets weight gamma^d.

        gamma=1.0 means uniform (same as existing KL training).
        gamma=0.8 prioritizes earlier positions (higher E[tau] contribution).
        """
        if self.pos_decay >= 1.0:
            return None
        w = torch.pow(torch.tensor(self.pos_decay, device=device),
                      torch.arange(seq_len, device=device, dtype=torch.float))
        return w / w.sum()

    # ------------------------------------------------------------------
    # Compute loss for a single block
    # ------------------------------------------------------------------

    def compute_block_loss(self, draft_logits, target_logits):
        """Compute LK loss for one (anchor, block) pair.

        Args:
            draft_logits:  [1, L, V] from draft model
            target_logits: [1, L, V] from frozen target

        Returns:
            loss: scalar, alpha_mean: scalar (for logging)
        """
        actual_len = min(draft_logits.shape[1], target_logits.shape[1])
        if actual_len == 0:
            return None, None

        dl = draft_logits[:, :actual_len, :].float()
        tl = target_logits[:, :actual_len, :].float()

        target_probs = F.softmax(tl, dim=-1).detach()
        draft_probs = F.softmax(dl, dim=-1)
        draft_log_probs = F.log_softmax(dl, dim=-1)

        if self.loss_mode == "alpha":
            per_pos = self.lk_alpha_loss(target_probs, draft_probs)
            alpha = self.acceptance_rate(target_probs, draft_probs)
        elif self.loss_mode == "hybrid":
            per_pos, alpha = self.lk_hybrid_loss(
                target_probs, draft_probs, draft_log_probs,
            )
        elif self.loss_mode == "kl":
            per_pos = self.forward_kl(target_probs, draft_log_probs)
            alpha = self.acceptance_rate(target_probs, draft_probs)
        else:
            raise ValueError(f"Unknown lk_mode: {self.loss_mode}")

        per_pos = per_pos.squeeze(0)
        alpha = alpha.squeeze(0)

        weights = self.position_weights(per_pos.shape[0], per_pos.device)
        if weights is not None:
            loss = (per_pos * weights).sum()
        else:
            loss = per_pos.mean()

        return loss, alpha.mean().item()

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, input_ids, attention_mask, loss_mask):
        device = input_ids.device
        all_target_hidden, target_logits = self.dataprepare(input_ids, attention_mask)

        positions = self.iter_block_positions(input_ids, loss_mask)

        loss_terms = []
        alpha_sum = 0.0
        alpha_count = 0

        for p in positions:
            target_hidden_p = all_target_hidden[:, :p + 1, :]
            anchor_ids = input_ids[:, p:p + 1]

            dl = self.draft_forward(target_hidden_p, anchor_ids, p)
            tl = target_logits[:, p:p + self.block_size - 1, :]

            loss, alpha_mean = self.compute_block_loss(dl, tl)
            if loss is not None:
                loss_terms.append(loss)
                if alpha_mean is not None:
                    alpha_sum += alpha_mean
                    alpha_count += 1

        if loss_terms:
            total_loss = torch.stack(loss_terms).mean()
        else:
            total_loss = sum(
                p.float().sum() * 0.0
                for p in self.draft_model.parameters() if p.requires_grad
            )

        avg_alpha = alpha_sum / max(alpha_count, 1)

        return total_loss, torch.tensor(avg_alpha, device=device)
