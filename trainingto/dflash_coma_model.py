"""
DFlash training model with COMA-based tree-aware loss.

Combines:
  - L_ploss: standard distillation (cross-entropy against target distribution)
  - L_coma:  closed-form counterfactual policy gradient for acceptance length

The COMA loss is derived from the counterfactual multi-agent advantage
specialized to the acceptance-length reward R = sum_i prod_{j<=i} 1[a_j=t*_j].

For a parallel block drafter, each position j is an independent "agent."
The exact COMA advantage for reached position j is:
    A_j = (1[a_j=t*_j] - p_j) * F_j
where p_j = P_theta(t*_j) and F_j = future acceptance from j+1 onward.

The deterministic zero-variance objective with the same expected gradient is:
    L_coma = -sum_j reached_j * F_j * p_j
since E[A_j * grad log pi(a_j)] = F_j * grad p_j.
"""

import os

import torch
import torch.nn.functional as F

from base_model import DFlashTrainBase


class DFlashCOMAModel(DFlashTrainBase):
    def __init__(self, config, target_model_path, draft_model_path):
        super().__init__(config, target_model_path, draft_model_path)

    def compute_coma_loss(self, draft_logits, target_logits_block):
        """
        Deterministic COMA loss for parallel block drafter.

        L_det = -sum_j reached_j * F_j * p_j

        where:
          p_j = P_theta(target_argmax_j)  [differentiable]
          reached_j = prod_{i<j} 1[draft_argmax_i == target_argmax_i]  [detached]
          F_j = 1 + sum_{i>j} prod_{k=j+1..i} 1[a_k == t*_k]  [detached]

        The gradient is: -sum_j sg(reached_j * F_j) * grad p_j
        which equals the expected COMA policy gradient E[A_j * grad log pi(a_j)].
        """
        B_minus_1 = min(draft_logits.shape[1], target_logits_block.shape[1])
        if B_minus_1 == 0:
            return torch.tensor(0.0, device=draft_logits.device, requires_grad=True), 0.0

        dl = draft_logits[0, :B_minus_1]
        tl = target_logits_block[0, :B_minus_1]

        target_argmax = tl.argmax(dim=-1)
        draft_argmax = dl.argmax(dim=-1)
        matches = (draft_argmax == target_argmax).float()

        draft_probs = F.softmax(dl.float(), dim=-1)
        p_j = draft_probs.gather(1, target_argmax.unsqueeze(1)).squeeze(1)

        reached = torch.ones(B_minus_1, device=dl.device)
        for j in range(1, B_minus_1):
            reached[j] = reached[j - 1] * matches[j - 1]

        F_vals = torch.ones(B_minus_1, device=dl.device)
        for j in range(B_minus_1 - 2, -1, -1):
            F_vals[j] = 1.0 + matches[j + 1].item() * F_vals[j + 1].item()

        reached = reached.detach()
        F_vals = F_vals.detach()

        coma_loss = -(reached * F_vals * p_j).sum() / max(B_minus_1, 1)

        with torch.no_grad():
            acceptance_len = int(matches.cumprod(0).sum().item())

        return coma_loss, acceptance_len

    def forward(self, input_ids, attention_mask, loss_mask):
        device = input_ids.device
        all_target_hidden, target_logits = self.dataprepare(input_ids, attention_mask)

        positions = self.iter_block_positions(input_ids, loss_mask)

        ploss_terms = []
        coma_terms = []
        acceptance_lengths = []

        for p in positions:
            target_hidden_p = all_target_hidden[:, :p + 1, :]
            anchor_ids = input_ids[:, p:p + 1]

            dl = self.draft_forward(target_hidden_p, anchor_ids, p)
            tl = target_logits[:, p + 1:p + self.block_size, :]
            actual_len = min(dl.shape[1], tl.shape[1])
            if actual_len == 0:
                continue
            dl_block = dl[:, :actual_len, :]
            tl_block = tl[:, :actual_len, :]

            target_probs = F.softmax(tl_block.float(), dim=-1).detach()
            draft_log_probs = F.log_softmax(dl_block.float(), dim=-1)
            kl = -(target_probs * draft_log_probs).sum(-1).mean()
            ploss_terms.append(kl)

            coma_loss, acc_len = self.compute_coma_loss(dl_block, tl_block)
            coma_terms.append(coma_loss)
            acceptance_lengths.append(acc_len)

        if ploss_terms:
            ploss = torch.stack(ploss_terms).mean()
        else:
            ploss = torch.tensor(0.0, device=device, requires_grad=True)

        if coma_terms:
            coma_loss = torch.stack(coma_terms).mean()
            avg_acceptance = sum(acceptance_lengths) / len(acceptance_lengths)
        else:
            coma_loss = torch.tensor(0.0, device=device, requires_grad=True)
            avg_acceptance = 0.0

        gto_weight = getattr(self.config, "gto_weight", 0.5)
        total_loss = ploss + gto_weight * coma_loss

        if os.environ.get("DEBUG_RL", ""):
            try:
                rank = torch.distributed.get_rank()
            except Exception:
                rank = 0
            if rank == 0:
                print(f"[COMA] ploss={ploss.item():.4f} coma={coma_loss.item():.4f} "
                      f"avg_accept={avg_acceptance:.2f} total={total_loss.item():.4f}")

        return total_loss, ploss, coma_loss, torch.tensor(avg_acceptance)
