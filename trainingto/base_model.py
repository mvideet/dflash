"""
Shared base class for DFlash training models (TopK, COMA, GTO).

Encapsulates the common init (target model frozen, draft model in train mode),
data preparation, draft forward pass, loss-mask normalization, and the
block-position iteration pattern used by all three training variants.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.dflash import DFlashDraftModel
from model.utils import extract_context_feature


class DFlashTrainBase(nn.Module):
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

        self.block_size = self.draft_model.block_size
        self.mask_token_id = self.draft_model.mask_token_id
        self.target_layer_ids = self.draft_model.target_layer_ids

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

    def draft_forward(self, target_hidden, anchor_ids, position_start):
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

        draft_hidden = self.draft_model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids,
        )[:, -self.block_size + 1:, :]

        draft_logits = self.target_model.lm_head(draft_hidden)
        return draft_logits

    def normalize_loss_mask(self, loss_mask):
        """Squeeze loss_mask from (1, S, 1) or (1, S) down to a 1-D tensor."""
        lm = loss_mask
        if lm.dim() == 3:
            lm = lm.squeeze(-1)
        if lm.dim() == 2:
            lm = lm[0]
        return lm

    def iter_block_positions(self, input_ids, loss_mask):
        """Return valid block-start positions using the half-block stride."""
        lm = self.normalize_loss_mask(loss_mask)
        stride = max(1, self.block_size // 2)
        positions = []
        for p in range(0, input_ids.shape[1] - self.block_size, stride):
            if lm[p].item() != 0:
                positions.append(p)
        return positions
