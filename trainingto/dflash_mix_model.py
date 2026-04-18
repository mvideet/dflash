"""
DFlash MIX (Mixed-Block + Exp-Weighted + Tree-Aware) fine-tuning model.

Implements the union of training improvements from the DFlash paper and
SpecForge + the variable-block-size extension:

  1. Exponential-weighted CE loss  w_k = exp(-(k-1)/gamma_loss)
     (DFlash paper, Fig 5; gamma_loss=7 for b=16).
  2. Random anchor sampling — block starts drawn at random assistant-mask
     positions each step (DFlash paper, Table 9, +13-14% tau).
  3. Tree-aware conditional CE (second pass with tree attention) — carried
     over from main_ctr.py / dflash_ctr_model.py.
  4. Variable block-size curriculum — each sequence trained with a block
     size sampled from {8, 12, 16, 24} with configurable probabilities.
  5. TTT-style recursive pass (SpecForge): one extra forward where the
     draft's own argmax output becomes the anchor for a second block;
     loss at block-2 teaches the draft to handle self-generated prefixes.

All five are layered on the existing CTR scaffold.  Each component has an
explicit weight so ablations are easy.
"""

import math
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


class DFlashMixModel(DFlashTrainBase):
    """Fine-tune a DFlash draft with every proven trick layered on.

    Config knobs (attached by main_mix.py from CLI):
      gamma_loss:       exp-weighting denominator (7 = DFlash b16)
      random_anchors:   if True, sample N random valid positions per seq
      anchors_per_seq:  N (0 = use fixed-stride as before)
      block_sizes:      list of ints sampled uniformly (e.g. [8, 12, 16, 24])
      ctr_weight:       weight on tree-attention conditional CE loss
      ttt_weight:       weight on TTT recursive-block loss
      max_tree_size:    tree budget for CTR pass
      tree_expand_k:    top-K branching for CTR tree build
    """

    def __init__(self, config, target_model_path, draft_model_path):
        super().__init__(config, target_model_path, draft_model_path)
        self._step_count = 0
        self._cached_pos_weights = {}  # depth -> tensor

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def pos_weights(self, depth: int, gamma_loss: float, device) -> torch.Tensor:
        """Return per-position normalised weights w_k = exp(-(k-1)/gamma_loss).
        Cached by depth so we don't reallocate every step."""
        key = (depth, float(gamma_loss), str(device))
        if key in self._cached_pos_weights:
            return self._cached_pos_weights[key]
        if gamma_loss <= 0:
            w = torch.ones(depth, device=device) / depth
        else:
            k = torch.arange(depth, device=device, dtype=torch.float32)
            w = torch.exp(-k / gamma_loss)
            w = w / w.sum()
        self._cached_pos_weights[key] = w
        return w

    def draft_forward_variable(self, target_hidden, anchor_ids, position_start,
                               block_size: int):
        """Variant of base draft_forward that uses a caller-supplied block size."""
        device = anchor_ids.device
        mask_tokens = torch.full(
            (1, block_size - 1), self.mask_token_id,
            dtype=torch.long, device=device,
        )
        block_ids = torch.cat([anchor_ids, mask_tokens], dim=1)
        noise_embedding = self.target_model.model.embed_tokens(block_ids)
        position_ids = torch.arange(
            position_start, position_start + block_size,
            device=device,
        ).unsqueeze(0)

        draft_hidden = self.draft_model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids,
        )[:, -block_size + 1:, :]

        draft_logits = self.target_model.lm_head(draft_hidden)
        return draft_logits, draft_hidden

    def draft_forward_tree(self, target_hidden, packed_ids, packed_pos_relative,
                           parent_idx, position_start):
        """Second draft pass with tree attention — identical to CTR model."""
        ctx_len = target_hidden.shape[1]
        tree_noise = self.target_model.model.embed_tokens(packed_ids)
        tree_pos = (packed_pos_relative + position_start).clone()
        tree_mask = create_tree_attention_mask_dynamic(
            packed_pos_relative, parent_idx, prefix_len=ctx_len,
        )
        saved_attn = self.draft_model.config._attn_implementation
        self.draft_model.config._attn_implementation = "sdpa"
        tree_hidden = self.draft_model(
            target_hidden=target_hidden,
            noise_embedding=tree_noise,
            position_ids=tree_pos,
            attention_mask=tree_mask,
            use_cache=False,
            is_causal=False,
        )
        self.draft_model.config._attn_implementation = saved_attn
        return self.target_model.lm_head(tree_hidden)

    def select_positions(self, input_ids, loss_mask, block_size: int):
        """Random-anchor or fixed-stride block-start selection."""
        lm = self.normalize_loss_mask(loss_mask)
        valid = (lm == 1).nonzero(as_tuple=True)[0]
        max_start = input_ids.shape[1] - block_size
        valid = valid[valid < max_start]
        if valid.numel() == 0:
            return []

        anchors_per_seq = getattr(self.config, "anchors_per_seq", 0)
        if getattr(self.config, "random_anchors", False) and anchors_per_seq > 0:
            # DFlash paper Table 9 — random anchor positions per epoch.
            n = min(anchors_per_seq, valid.numel())
            idx = torch.randperm(valid.numel(), device=valid.device)[:n]
            return valid[idx].tolist()
        # Fallback: fixed stride (legacy behaviour).
        stride = max(1, block_size // 2)
        out = [int(p.item()) for p in valid[::stride]]
        return out

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(self, input_ids, attention_mask, loss_mask):
        device = input_ids.device

        # --- Sample block size for this step (variable block curriculum) ---
        block_sizes = getattr(self.config, "block_sizes", [16])
        bs_probs = getattr(self.config, "block_size_probs", None)
        if len(block_sizes) > 1:
            if bs_probs is None:
                idx = int(torch.randint(0, len(block_sizes), (1,)).item())
            else:
                idx = int(torch.multinomial(
                    torch.tensor(bs_probs), 1,
                ).item())
            block_size = block_sizes[idx]
        else:
            block_size = block_sizes[0]

        gamma_loss = float(getattr(self.config, "gamma_loss", 7.0))
        ctr_weight = float(getattr(self.config, "ctr_weight", 0.0))
        ttt_weight = float(getattr(self.config, "ttt_weight", 0.0))
        tree_expand_k = int(getattr(self.config, "tree_expand_k", 5))
        max_tree_size = int(getattr(self.config, "max_tree_size", 16))

        # Need target hidden states and logits once per sequence.
        all_target_hidden, target_logits = self.dataprepare(input_ids, attention_mask)

        positions = self.select_positions(input_ids, loss_mask, block_size)

        marginal_loss_terms = []
        ctr_loss_terms = []
        ttt_loss_terms = []
        marginal_accs = []
        ctr_accs = []

        for p in positions:
            target_hidden_p = all_target_hidden[:, :p + 1, :]
            anchor_ids = input_ids[:, p:p + 1]

            # --- Pass 1: bidirectional, variable block size ---
            dl, draft_hidden = self.draft_forward_variable(
                target_hidden_p, anchor_ids, p, block_size,
            )
            tl = target_logits[:, p:p + block_size - 1, :]

            actual_len = min(dl.shape[1], tl.shape[1])
            if actual_len == 0:
                continue
            dl_block = dl[:, :actual_len, :]
            tl_block = tl[:, :actual_len, :]

            target_probs = F.softmax(tl_block.float(), dim=-1).detach()
            draft_log_probs = F.log_softmax(dl_block.float(), dim=-1)
            per_pos = -(target_probs * draft_log_probs).sum(-1).squeeze(0)  # [L]
            w = self.pos_weights(per_pos.shape[0], gamma_loss, device)
            marginal_loss = (per_pos * w).sum()
            marginal_loss_terms.append(marginal_loss)

            with torch.no_grad():
                m_acc = (dl_block.argmax(-1) == tl_block.argmax(-1)).float().mean().item()
                marginal_accs.append(m_acc)

            # --- Pass 2: Tree-aware conditional CE (CTR) ---
            if ctr_weight > 0.0:
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
                )
                L = packed_ids.shape[1]
                depths = packed_pos_rel[0]

                tree_loss_sum = torch.tensor(0.0, device=device)
                tree_count = 0
                tree_correct = 0
                for ni in range(1, L):
                    d = depths[ni].item()
                    gt_pos = p + d
                    if gt_pos >= input_ids.shape[1]:
                        continue
                    if p + d - 1 < target_logits.shape[1]:
                        node_target_logits = target_logits[:, p + d - 1, :]
                        node_target_probs = F.softmax(
                            node_target_logits.float(), dim=-1,
                        ).detach()
                        node_draft_log_probs = F.log_softmax(
                            tree_logits[:, ni, :].float(), dim=-1,
                        )
                        # Exp-weight by depth d so deep tree nodes get less weight
                        # (parallel to the marginal exp-weighted loss).
                        depth_w = math.exp(-(d - 1) / gamma_loss) if gamma_loss > 0 else 1.0
                        node_ce = -(node_target_probs * node_draft_log_probs).sum(-1)
                        tree_loss_sum = tree_loss_sum + depth_w * node_ce.squeeze()
                        tree_count += 1
                        with torch.no_grad():
                            if tree_logits[0, ni, :].argmax() == node_target_logits[0].argmax():
                                tree_correct += 1
                if tree_count > 0:
                    ctr_loss_terms.append(tree_loss_sum / tree_count)
                    ctr_accs.append(tree_correct / tree_count)

            # --- Pass 3: TTT recursive — block_2 anchored on draft's own argmax ---
            # Only runs if (a) there's room for block_2 and (b) ttt_weight > 0.
            if (
                ttt_weight > 0.0
                and p + 2 * block_size - 1 < input_ids.shape[1]
                and p + actual_len < input_ids.shape[1]
            ):
                # Draft's argmax at end of block_1 becomes block_2 anchor.
                block1_argmax = dl_block.argmax(-1)[0]                   # [L1]
                block2_start = p + block_size  # next block starts one block later
                block2_anchor = block1_argmax[-1].view(1, 1)

                dl2, _ = self.draft_forward_variable(
                    target_hidden_p, block2_anchor, block2_start, block_size,
                )
                # Ground-truth for block_2 is the REAL target tokens at positions
                # block2_start+1 .. block2_start + block_size - 1
                # but the draft_2 was anchored on draft's own block_1 argmax,
                # not on real tokens.  That's EXACTLY the training signal we
                # want — teach the draft to recover when its own argmax is wrong.
                tl2 = target_logits[:, block2_start:block2_start + block_size - 1, :]
                L2 = min(dl2.shape[1], tl2.shape[1])
                if L2 > 0:
                    dl2_b = dl2[:, :L2, :]
                    tl2_b = tl2[:, :L2, :]
                    tp2 = F.softmax(tl2_b.float(), dim=-1).detach()
                    dlp2 = F.log_softmax(dl2_b.float(), dim=-1)
                    per_pos2 = -(tp2 * dlp2).sum(-1).squeeze(0)
                    w2 = self.pos_weights(L2, gamma_loss, device)
                    ttt_loss_terms.append((per_pos2 * w2).sum())

        # --- Aggregate ---
        if marginal_loss_terms:
            marginal_loss = torch.stack(marginal_loss_terms).mean()
        else:
            marginal_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if ctr_loss_terms:
            ctr_loss = torch.stack(ctr_loss_terms).mean()
        else:
            ctr_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if ttt_loss_terms:
            ttt_loss = torch.stack(ttt_loss_terms).mean()
        else:
            ttt_loss = torch.tensor(0.0, device=device, requires_grad=True)

        total_loss = marginal_loss + ctr_weight * ctr_loss + ttt_weight * ttt_loss

        avg_m_acc = sum(marginal_accs) / len(marginal_accs) if marginal_accs else 0.0
        avg_ctr_acc = sum(ctr_accs) / len(ctr_accs) if ctr_accs else 0.0

        if _rank0() and self._step_count % 10 == 0:
            print(
                f"[MIX] step={self._step_count} bs={block_size} "
                f"n_pos={len(marginal_loss_terms)} "
                f"marg={marginal_loss.item():.4f} "
                f"ctr={ctr_loss.item():.4f} "
                f"ttt={ttt_loss.item():.4f} "
                f"total={total_loss.item():.4f} "
                f"m_acc={avg_m_acc:.3f} c_acc={avg_ctr_acc:.3f}"
            )
        if _HAS_WANDB and wandb.run is not None and _rank0():
            wandb.log({
                "train/step": self._step_count,
                "train/block_size": block_size,
                "train/n_positions": len(marginal_loss_terms),
                "loss/total": total_loss.item(),
                "loss/marginal": marginal_loss.item(),
                "loss/ctr": ctr_loss.item(),
                "loss/ttt": ttt_loss.item(),
                "acc/marginal": avg_m_acc,
                "acc/ctr": avg_ctr_acc,
            })
        self._step_count += 1
        return total_loss, marginal_loss, ctr_loss, torch.tensor(avg_ctr_acc)
