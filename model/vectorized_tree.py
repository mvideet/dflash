"""Vectorized fixed-width tree builder — kills the per-element Python heap.

Trade-off vs DDTree's heap:
  - GAIN: tree-build time drops from O(B·M·K log(B·M)) Python-heap to a few
    GPU torch.topk calls. At B=8, M=16: ~3-6 ms → <0.5 ms.
  - LOSS: each element gets the SAME tree shape (width per depth chosen
    globally). Per-element best-paths are not individually optimized.
    Expected tau loss: 0-15% depending on how heterogeneous the prompts are.

Algorithm (Sequoia-shape BFS, layer by layer):

   layer 0:    anchor (1 node)
   layer d>0:  per-element, take top-(width(d)) candidates from
               (parents at layer d-1) × (top-K tokens at draft depth d-1),
               sorted by parent_score + draft log-prob.

Output: packed_ids [B, M], packed_pos_rel [B, M], parent_idx [B, M],
         node_valid [B, M], leaf_paths [B, N, P], leaf_tokens [B, N, P-1],
         leaf_valid [B, N] — same interface as build_node_budget_tree_batched.
"""
from typing import List, Optional, Tuple

import torch


@torch.no_grad()
def vectorized_tree_build(
    draft_logits: torch.Tensor,            # [B, seq_len, V]
    anchor_token_ids: torch.LongTensor,    # [B] or [B, *]
    width_vec: List[int],                  # [seq_len+1] width per depth (depth 0 = 1)
    K: int = 8,                            # max candidate fanout per parent
    max_M: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a fixed-width tree per layer using vectorized torch ops only."""
    B, seq_len, _ = draft_logits.shape
    device = draft_logits.device

    if anchor_token_ids.dim() > 1:
        anchors = anchor_token_ids[:, 0]
    else:
        anchors = anchor_token_ids                              # [B]

    # Per-position top-K tokens and log-probs.
    log_denom = torch.logsumexp(draft_logits, dim=-1, keepdim=True)
    log_probs = draft_logits - log_denom
    topk_lp, topk_idx = log_probs.topk(K, dim=-1)               # [B, seq_len, K]

    # Total budget. Width vec includes depth 0 (=1, the anchor).
    max_depth = min(len(width_vec), seq_len + 1)
    if max_M is None:
        max_M = sum(width_vec[:max_depth])

    # Layer-wise build. We accumulate into per-layer lists then concat.
    # Each layer entry: tokens [B, n_d], parent_in_prev_layer [B, n_d], score [B, n_d].
    # Layer 0 = anchor: tokens = anchors[:, None], parent = -1 (we use 0 as sentinel),
    # score = 0.
    layer_tokens: List[torch.Tensor] = [anchors.unsqueeze(1)]   # [B, 1]
    layer_parents_in_prev: List[torch.Tensor] = [
        torch.full((B, 1), -1, dtype=torch.long, device=device)
    ]
    layer_scores: List[torch.Tensor] = [
        torch.zeros(B, 1, dtype=torch.float32, device=device)
    ]
    layer_sizes: List[int] = [1]

    for d in range(1, max_depth):
        prev_n = layer_sizes[d - 1]
        if prev_n == 0:
            break
        w_d = width_vec[d] if d < len(width_vec) else 0
        if w_d <= 0:
            break

        # Each parent at layer d-1 generates K candidates from draft pos d-1.
        # candidate_score = parent_score + topk_lp[:, d-1, :]
        cand_score = (
            layer_scores[d - 1].unsqueeze(-1)                   # [B, prev_n, 1]
            + topk_lp[:, d - 1, :].unsqueeze(1)                  # [B, 1, K]
        )                                                         # [B, prev_n, K]
        cand_token = topk_idx[:, d - 1, :].unsqueeze(1).expand(-1, prev_n, -1)  # [B, prev_n, K]

        cand_score_flat = cand_score.reshape(B, prev_n * K)      # [B, prev_n*K]
        cand_token_flat = cand_token.reshape(B, prev_n * K)

        keep = min(w_d, prev_n * K)
        top_score, top_idx = cand_score_flat.topk(keep, dim=-1)  # [B, keep]
        top_token = cand_token_flat.gather(1, top_idx)
        parent_in_prev = top_idx // K                             # [B, keep]

        layer_tokens.append(top_token)
        layer_parents_in_prev.append(parent_in_prev)
        layer_scores.append(top_score)
        layer_sizes.append(keep)

    # Pack into a single [B, M] tensor of tree nodes, ordered layer-by-layer.
    # Compute global node index for each (layer, slot): cum_offsets[d] + slot.
    cum_offsets: List[int] = [0]
    for s in layer_sizes:
        cum_offsets.append(cum_offsets[-1] + s)
    M_total = cum_offsets[-1]
    if max_M is not None and M_total > max_M:
        # Trim deepest layers until we fit.
        while M_total > max_M and len(layer_sizes) > 1:
            layer_sizes.pop()
            layer_tokens.pop()
            layer_parents_in_prev.pop()
            layer_scores.pop()
            cum_offsets.pop()
            M_total = cum_offsets[-1]

    # Stack across layers.
    packed_ids = torch.cat(layer_tokens, dim=1)                  # [B, M_total]
    packed_pos = torch.cat([
        torch.full((B, layer_sizes[d]), d, dtype=torch.long, device=device)
        for d in range(len(layer_sizes))
    ], dim=1)                                                    # [B, M_total]

    # parent_idx: global index of parent node in packed_ids.
    parent_idx_layers = []
    for d in range(len(layer_sizes)):
        if d == 0:
            parent_idx_layers.append(
                torch.zeros(B, 1, dtype=torch.long, device=device)  # root self-parents (sentinel)
            )
        else:
            par_local = layer_parents_in_prev[d]                 # [B, n_d] index in prev layer
            par_global = par_local + cum_offsets[d - 1]          # offset into prev layer in packed
            parent_idx_layers.append(par_global)
    parent_idx = torch.cat(parent_idx_layers, dim=1)             # [B, M_total]
    node_valid = torch.ones(B, M_total, dtype=torch.bool, device=device)

    # Leaf paths and tokens.
    # A leaf is a node whose layer is the deepest, OR a node with no children.
    # In fixed-width tree, all nodes at the deepest layer are leaves; all earlier
    # layer nodes have at least one child UNLESS they didn't get picked as a parent
    # by any depth-d+1 candidate. We handle this by treating ALL nodes at the
    # final layer as leaves.
    final_d = len(layer_sizes) - 1
    n_leaves = layer_sizes[final_d]
    path_len = final_d + 1
    leaf_paths = torch.zeros(B, n_leaves, path_len, dtype=torch.long, device=device)
    leaf_tokens = torch.zeros(B, n_leaves, path_len - 1, dtype=torch.long, device=device)
    # For each leaf at final layer, walk parent chain back to root.
    cur_global = (
        torch.arange(n_leaves, device=device).unsqueeze(0).expand(B, -1)
        + cum_offsets[final_d]
    )
    leaf_paths[:, :, final_d] = cur_global
    if path_len - 1 >= 1:
        leaf_tokens[:, :, final_d - 1] = packed_ids.gather(1, cur_global)
    for d in range(final_d - 1, -1, -1):
        cur_global = parent_idx.gather(1, cur_global)
        leaf_paths[:, :, d] = cur_global
        if d - 1 >= 0:
            leaf_tokens[:, :, d - 1] = packed_ids.gather(1, cur_global)
    leaf_valid = torch.ones(B, n_leaves, dtype=torch.bool, device=device)

    return packed_ids, packed_pos, parent_idx, node_valid, leaf_paths, leaf_tokens, leaf_valid


def default_width_vec(M: int, expand_k: int = 8) -> List[int]:
    """Heuristic default width vector for a budget of M nodes — front-heavy."""
    widths = [1]                          # depth 0: anchor
    remaining = M
    d = 1
    parents = 1
    while remaining > 0:
        cap = parents * expand_k
        # Front-heavy: at depth 1, take expand_k. At depth d, halve.
        target = max(1, expand_k // (2 ** (d - 1))) * parents
        w = min(target, cap, remaining)
        if w <= 0:
            break
        widths.append(w)
        parents = w
        remaining -= w
        d += 1
        if d > 16:
            break
    return widths
