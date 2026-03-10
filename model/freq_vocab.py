"""Frequency-based vocabulary for FR-Spec style draft pruning.

Loads freq_{r}.pt from generate_freq.py. Provides reduced lm_head computation
to save compute: only compute logits for top-r frequent tokens.
"""

from typing import Optional, Tuple
import torch


def load_freq_mapping(path: str) -> tuple[torch.Tensor, list[int]]:
    """
    Load frequency mapping from freq_{r}.pt.

    Returns:
        used_tokens: list of top-r target token IDs (sorted)
        used_tokens is derived from d2t: used_tokens[i] = d2t[i] + i
    """
    cache = torch.load(path, map_location="cpu", weights_only=True)
    d2t = cache["d2t"]  # shape [r]
    r = d2t.shape[0]
    used_tokens = [int(d2t[i].item()) + i for i in range(r)]
    return d2t, used_tokens


def get_reduced_lm_head(
    lm_head: torch.nn.Linear,
    used_tokens: list[int],
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Extract rows of target's lm_head for DRAFT candidate selection only.
    The target model itself always uses its full lm_head for verification.
    This reduced projection is only used to compute draft logits (hidden @ reduced_weight.T).

    Returns:
        reduced_weight: [r, hidden_size]
        reduced_bias: [r] or None
    """
    used_t = torch.tensor(used_tokens, device=lm_head.weight.device, dtype=torch.long)
    reduced_weight = lm_head.weight.index_select(0, used_t)
    reduced_bias = None
    if lm_head.bias is not None:
        reduced_bias = lm_head.bias.index_select(0, used_t)
    return reduced_weight.to(device), reduced_bias.to(device) if reduced_bias is not None else None


def compute_reduced_draft_logits(
    hidden: torch.Tensor,
    reduced_weight: torch.Tensor,
    reduced_bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Compute draft logits over reduced vocabulary only. Saves compute vs full lm_head.

    Args:
        hidden: [B, seq_len, hidden_size]
        reduced_weight: [r, hidden_size]
        reduced_bias: [r] or None

    Returns:
        logits: [B, seq_len, r]
    """
    # hidden @ reduced_weight.T = [B, seq, hidden] @ [hidden, r] = [B, seq, r]
    logits = torch.matmul(hidden, reduced_weight.T)
    if reduced_bias is not None:
        logits = logits + reduced_bias
    return logits
