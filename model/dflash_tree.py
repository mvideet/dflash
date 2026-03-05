"""Tree utilities for DFlash speculative decoding."""

from typing import Dict, List, Optional, Tuple
import itertools
import torch


# ---------------------------------------------------------------------------
# Tree token packing
# ---------------------------------------------------------------------------

def sample_first(
    logits: torch.Tensor,
    anchor_token_ids: torch.LongTensor,
    temperature: float = 0.0,
    top_k: int = 1,
    used_tokens: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Build packed tree tokens (branching at position 1):
    idx 0        : anchor (given)
    idx 1..K     : top-K candidates for position 1 (from logits[:,0,:])
    idx 1+K..    : positions 2.. duplicated K times each (argmax)
    Output shape: [B, 1 + K*seq_len]

    When used_tokens is provided, logits are [B, seq, r] (reduced vocab).
    topk/argmax return indices 0..r-1; we map to token IDs via used_tokens[i].

    NOTE: Always uses deterministic top-K / argmax for tree construction
    regardless of temperature. The tree should cover the most probable
    tokens to maximise acceptance. Temperature only affects the bonus
    token sampled after verification.
    """
    bsz, seq_len, r_or_vocab = logits.shape
    K = top_k
    out = torch.empty((bsz, 1 + K * seq_len), device=logits.device, dtype=torch.long)

    if anchor_token_ids.dim() == 2:
        anchor_token_ids = anchor_token_ids.squeeze(-1)
    out[:, 0] = anchor_token_ids

    # Position 1: deterministic top-K (most probable candidates)
    _, topk_idx = torch.topk(logits[:, 0, :], K, dim=-1)  # [B, K] indices
    if used_tokens is not None:
        used_t = torch.tensor(used_tokens, device=logits.device, dtype=torch.long)
        topk_token_ids = used_t[topk_idx]
        out[:, 1:1+K] = topk_token_ids
    else:
        out[:, 1:1+K] = topk_idx

    # Positions 2+: argmax, duplicated across K branches
    if seq_len > 1:
        rest_idx = torch.argmax(logits[:, 1:, :], dim=-1)  # [B, seq_len-1] indices
        if used_tokens is not None:
            used_t = torch.tensor(used_tokens, device=logits.device, dtype=torch.long)
            rest_token_ids = used_t[rest_idx]
            for pos_idx in range(seq_len - 1):
                s = 1 + K + pos_idx * K
                out[:, s:s+K] = rest_token_ids[:, pos_idx].unsqueeze(-1).expand(-1, K)
        else:
            for pos_idx in range(seq_len - 1):
                s = 1 + K + pos_idx * K
                out[:, s:s+K] = rest_idx[:, pos_idx].unsqueeze(-1).expand(-1, K)

    return out


# ---------------------------------------------------------------------------
# Position IDs
# ---------------------------------------------------------------------------

def get_position_ids(top_k_indices: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Position IDs for first tree (branching at position 1).
    Format: [0, 1, 1, 1, ..., 2, 2, 2, ..., ...]
    """
    bsz, extended_seq_len = top_k_indices.shape
    device = top_k_indices.device

    seq_len = 1 + (extended_seq_len - 1) // top_k
    position_ids = torch.zeros((bsz, extended_seq_len), dtype=torch.long, device=device)

    position_ids[:, 0] = 0
    position_ids[:, 1 : 1 + top_k] = 1

    if seq_len > 2:
        for pos in range(2, seq_len):
            start_idx = 1 + top_k + (pos - 2) * top_k
            end_idx = start_idx + top_k
            position_ids[:, start_idx : end_idx] = pos

    return position_ids


# ---------------------------------------------------------------------------
# Branch IDs
# ---------------------------------------------------------------------------

def make_branch_ids(L: int, top_k: int, device) -> torch.LongTensor:
    """Branch IDs for first tree: -1 for anchor, 0..K-1 starting at position 1."""
    branch = torch.full((L,), -1, device=device, dtype=torch.long)
    if L <= 1:
        return branch

    end1 = min(1 + top_k, L)
    branch[1:end1] = torch.arange(end1 - 1, device=device, dtype=torch.long)

    if L > 1 + top_k:
        idx = torch.arange(1 + top_k, L, device=device)
        branch[idx] = (idx - (1 + top_k)) % top_k

    return branch


# ---------------------------------------------------------------------------
# Tree attention mask
# ---------------------------------------------------------------------------

def create_tree_attention_mask(
    position_ids: torch.LongTensor,
    top_k: int,
    prefix_len: int = 0,
) -> torch.Tensor:
    """
    Build additive attention mask for tree verification.
    position_ids: [B, L]
    returns: [B, 1, L, prefix_len + L]
    """
    B, L = position_ids.shape
    device = position_ids.device

    branch_ids_1d = make_branch_ids(L, top_k=top_k, device=device)
    branch_ids = branch_ids_1d.unsqueeze(0).expand(B, L)

    q_pos = position_ids.unsqueeze(-1)   # [B, L, 1]
    k_pos = position_ids.unsqueeze(-2)   # [B, 1, L]
    q_branch = branch_ids.unsqueeze(-1)  # [B, L, 1]
    k_branch = branch_ids.unsqueeze(-2)  # [B, 1, L]

    same_branch = (q_branch == k_branch) & (q_branch != -1)
    k_is_shared_prefix = (k_branch == -1)  # anchor (pos 0) for first tree

    strictly_past = (k_pos < q_pos) & (k_is_shared_prefix | same_branch)

    self_only = torch.eye(L, device=device, dtype=torch.bool).unsqueeze(0)
    self_only = self_only & (k_pos == q_pos)

    allow = strictly_past | self_only

    mask = torch.zeros((B, 1, L, L), device=device, dtype=torch.bfloat16)
    mask.masked_fill_(~allow.unsqueeze(1), torch.finfo(torch.bfloat16).min)

    if prefix_len > 0:
        prefix = torch.zeros((B, 1, L, prefix_len), device=device, dtype=torch.bfloat16)
        mask = torch.cat([prefix, mask], dim=-1)

    return mask


# ---------------------------------------------------------------------------
# Path packed indices
# ---------------------------------------------------------------------------

def compute_path_packed_indices(
    branch: torch.Tensor, seq_len: int, top_k: int
) -> torch.Tensor:
    """
    Packed indices for first tree (branching at position 1).
    Format: [0, 1+branch, 1+K+branch, 1+2K+branch, ...]
    """
    bsz = branch.shape[0]
    device = branch.device

    path = torch.empty((bsz, seq_len), device=device, dtype=torch.long)
    path[:, 0] = 0
    path[:, 1] = 1 + branch
    for p in range(2, seq_len):
        path[:, p] = 1 + top_k + (p - 2) * top_k + branch
    return path


# ---------------------------------------------------------------------------
# Dynamic branching tree helpers
# ---------------------------------------------------------------------------

def classify_distribution(
    logits: torch.Tensor,
    theta_uni: float,
    theta_bi: float,
    theta_tri: float,
) -> int:
    """
    Classify one distribution as unimodal/bimodal/trimodal.
    Returns K in {1, 2, 3}.
    """
    probs = torch.softmax(logits, dim=-1)
    top_probs, _ = torch.topk(probs, k=3, dim=-1)
    p1, p2, p3 = top_probs[0].item(), top_probs[1].item(), top_probs[2].item()

    if p1 > theta_uni:
        return 1
    if (p2 / max(p1, 1e-12)) > theta_bi:
        return 2
    if (p3 / max(p1, 1e-12)) > theta_tri:
        return 3
    return 1


def _cap_branch_counts(counts: List[int], max_leaves: int) -> List[int]:
    """Cap per-position branching counts so product(counts) <= max_leaves."""
    capped = []
    leaves = 1
    for k in counts:
        k_eff = max(1, min(3, int(k)))
        if leaves * k_eff > max_leaves:
            k_eff = max(1, max_leaves // max(leaves, 1))
        capped.append(k_eff)
        leaves *= k_eff
    return capped


def build_dynamic_tree(
    draft_logits: torch.Tensor,
    anchor_token_ids: torch.LongTensor,
    theta_uni: float,
    theta_bi: float,
    theta_tri: float,
    max_tree_size: int = 8,
    used_tokens: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a variable-branching tree from draft logits (B must be 1).

    When used_tokens is provided, logits are [1, seq, r] (reduced vocab).
    top3_idx are indices 0..r-1; we map to token IDs via used_tokens[i].

    Returns:
      packed_ids:      [1, L]
      packed_pos:      [1, L]
      parent_idx:      [L]   (root parent = -1)
      leaf_paths:      [N, block_size] packed indices from root to leaf
      leaf_tokens:     [N, block_size - 1] tokens for positions 1..block_size-1
    """
    if draft_logits.size(0) != 1:
        raise ValueError("Dynamic tree currently supports batch size 1.")

    device = draft_logits.device
    _, seq_len, _ = draft_logits.shape  # seq_len = block_size - 1

    # Extract per-position top-3 tokens and probabilities in one shot on GPU,
    # then move compact results to CPU for Python-side tree building.
    logits_pos = draft_logits[0]  # [seq_len, r or vocab]
    top3_vals, top3_idx = torch.topk(logits_pos, k=3, dim=-1)  # [seq_len, 3]
    log_denom = torch.logsumexp(logits_pos, dim=-1, keepdim=True)  # [seq_len, 1]
    top3_probs = torch.exp(top3_vals - log_denom)  # [seq_len, 3]

    if used_tokens is not None:
        used_t = torch.tensor(used_tokens, device=device, dtype=torch.long)
        top3_token_ids = used_t[top3_idx]  # [seq_len, 3]
        top3_tokens_per_pos: List[List[int]] = top3_token_ids.detach().cpu().tolist()
    else:
        top3_tokens_per_pos = top3_idx.detach().cpu().tolist()
    top3_probs_cpu: List[List[float]] = top3_probs.detach().cpu().tolist()

    raw_counts: List[int] = []
    for p1, p2, p3 in top3_probs_cpu:
        if p1 > theta_uni:
            raw_counts.append(1)
        elif (p2 / max(p1, 1e-12)) > theta_bi:
            raw_counts.append(2)
        elif (p3 / max(p1, 1e-12)) > theta_tri:
            raw_counts.append(3)
        else:
            raw_counts.append(1)

    counts = _cap_branch_counts(raw_counts, max_leaves=max_tree_size)

    # Enumerate leaves as cartesian product of local branch choices.
    ranges = [range(k) for k in counts]
    combos = list(itertools.product(*ranges)) if ranges else [()]
    num_leaves = len(combos)

    leaf_tokens_list: List[List[int]] = []
    for combo in combos:
        row = []
        for pos, local_choice in enumerate(combo):
            row.append(top3_tokens_per_pos[pos][local_choice])
        leaf_tokens_list.append(row)

    leaf_tokens = torch.tensor(leaf_tokens_list, device=device, dtype=torch.long)

    if anchor_token_ids.dim() == 2:
        anchor_token = int(anchor_token_ids[0, 0].detach().cpu().item())
    else:
        anchor_token = int(anchor_token_ids[0].detach().cpu().item())

    # Build trie to get packed nodes and per-leaf packed paths.
    node_tokens: List[int] = [anchor_token]
    node_pos: List[int] = [0]
    node_parent: List[int] = [-1]
    children_maps: List[Dict[int, int]] = [dict()]  # token -> child packed idx
    leaf_paths_list: List[List[int]] = []

    for i in range(num_leaves):
        cur = 0
        path = [0]
        for pos in range(seq_len):
            tok = leaf_tokens_list[i][pos]
            child = children_maps[cur].get(tok)
            if child is None:
                child = len(node_tokens)
                children_maps[cur][tok] = child
                node_tokens.append(tok)
                node_pos.append(pos + 1)
                node_parent.append(cur)
                children_maps.append(dict())
            cur = child
            path.append(cur)
        leaf_paths_list.append(path)

    packed_ids = torch.tensor(node_tokens, device=device, dtype=torch.long).unsqueeze(0)
    packed_pos = torch.tensor(node_pos, device=device, dtype=torch.long).unsqueeze(0)
    parent_idx = torch.tensor(node_parent, device=device, dtype=torch.long)
    leaf_paths = torch.tensor(leaf_paths_list, device=device, dtype=torch.long)

    return packed_ids, packed_pos, parent_idx, leaf_paths, leaf_tokens


def create_tree_attention_mask_dynamic(
    position_ids: torch.LongTensor,
    parent_idx: torch.LongTensor,
    prefix_len: int = 0,
) -> torch.Tensor:
    """
    Build additive attention mask for a variable-branching tree.
    A query can attend to its ancestors and itself.
    """
    B, L = position_ids.shape
    device = position_ids.device
    if B != 1:
        raise ValueError("Dynamic tree mask currently supports batch size 1.")

    # Build ancestor-allow matrix on CPU to avoid repeated CUDA syncs from
    # scalar tensor access in Python loops.
    parent = parent_idx.detach().cpu().tolist()
    allow = torch.zeros((L, L), dtype=torch.bool)
    for q in range(L):
        # Self
        allow[q, q] = True
        # Ancestors (including root)
        p = parent[q]
        while p >= 0:
            allow[q, p] = True
            p = parent[p]

    pos_cpu = position_ids[0].detach().cpu()
    q_pos = pos_cpu.unsqueeze(-1)  # [L,1]
    k_pos = pos_cpu.unsqueeze(0)   # [1,L]
    causal_allow = allow & ((k_pos < q_pos) | torch.eye(L, dtype=torch.bool))
    causal_allow = causal_allow.to(device=device)

    min_val = torch.finfo(torch.bfloat16).min
    mask = torch.full((1, 1, L, prefix_len + L), min_val, device=device, dtype=torch.bfloat16)
    if prefix_len > 0:
        mask[:, :, :, :prefix_len] = 0
    mask[:, :, :, prefix_len:].masked_fill_(causal_allow.unsqueeze(0).unsqueeze(0), 0)
    return mask


def select_best_dynamic_leaf(
    logits: torch.Tensor,
    leaf_paths: torch.Tensor,
    leaf_tokens: torch.Tensor,
    temperature: float = 0.0,
) -> Tuple[int, int]:
    """
    Select leaf with maximum consecutive acceptance length.
    Returns:
      best_leaf_idx
      n  (# accepted positions after anchor)
    """
    # B is 1. Vectorize across leaves to avoid Python loops.
    prev_nodes = leaf_paths[:, :-1]  # [N, block_size-1]
    realized = leaf_tokens  # [N, block_size-1]
    n_leaves, depth = prev_nodes.shape
    vocab = logits.size(-1)

    gathered = logits[0].index_select(0, prev_nodes.reshape(-1)).view(n_leaves, depth, vocab)
    if temperature < 1e-5:
        pred = gathered.argmax(dim=-1)  # [N, depth]
    else:
        probs = torch.softmax(gathered / temperature, dim=-1)
        pred = torch.multinomial(probs.view(-1, vocab), 1).view(n_leaves, depth)

    matches = (pred == realized)
    acc = matches.cumprod(dim=1).sum(dim=1)  # [N]
    best_n, best_leaf = torch.max(acc, dim=0)
    return int(best_leaf.item()), int(best_n.item())
