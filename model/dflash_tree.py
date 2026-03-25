"""Tree construction and verification utilities for DFlash speculative decoding.

Three tree-building strategies are provided:
  v1  build_dynamic_tree      — per-position threshold branching (cartesian product)
  v2  build_dynamic_tree_v2   — EAGLE-2 expand + rerank (Li et al., 2024)
  v3  build_bestfirst_tree    — best-first search by cumulative log-probability
"""

from typing import Dict, List, Optional, Tuple
import heapq
import itertools
import torch


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
# v1: Threshold + cartesian-product tree
# ---------------------------------------------------------------------------

def _cap_branch_counts(counts: List[int], max_leaves: int, top_k: int = 3) -> List[int]:
    """Cap per-position branching counts so product(counts) <= max_leaves."""
    capped = []
    leaves = 1
    for k in counts:
        k_eff = max(1, min(top_k, int(k)))
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
    top_k: int = 3,
    used_tokens: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a variable-branching tree from draft logits (B must be 1).

    When used_tokens is provided, logits are [1, seq, r] (reduced vocab).
    topk_idx are indices 0..r-1; we map to token IDs via used_tokens[i].

    top_k controls the maximum per-node expansion width (default 3).
    Adaptive branching decides how many of the top_k to actually use
    based on probability thresholds.

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

    logits_pos = draft_logits[0]  # [seq_len, r or vocab]
    topk_vals, topk_idx = torch.topk(logits_pos, k=top_k, dim=-1)  # [seq_len, top_k]
    log_denom = torch.logsumexp(logits_pos, dim=-1, keepdim=True)  # [seq_len, 1]
    topk_probs = torch.exp(topk_vals - log_denom)  # [seq_len, top_k]

    if used_tokens is not None:
        used_t = torch.tensor(used_tokens, device=device, dtype=torch.long)
        topk_token_ids = used_t[topk_idx]
        topk_tokens_per_pos: List[List[int]] = topk_token_ids.detach().cpu().tolist()
    else:
        topk_tokens_per_pos = topk_idx.detach().cpu().tolist()
    topk_probs_cpu: List[List[float]] = topk_probs.detach().cpu().tolist()

    ratio_thresholds = [theta_bi, theta_tri] + [theta_tri * 0.5] * (top_k - 3)

    raw_counts: List[int] = []
    for probs in topk_probs_cpu:
        p1 = probs[0]
        if p1 > theta_uni:
            raw_counts.append(1)
            continue
        count = 1
        for i in range(1, top_k):
            thresh = ratio_thresholds[i - 1] if (i - 1) < len(ratio_thresholds) else theta_tri * 0.25
            if (probs[i] / max(p1, 1e-12)) > thresh:
                count = i + 1
            else:
                break
        raw_counts.append(count)

    counts = _cap_branch_counts(raw_counts, max_leaves=max_tree_size, top_k=top_k)

    # Enumerate leaves as cartesian product of local branch choices.
    ranges = [range(k) for k in counts]
    combos = list(itertools.product(*ranges)) if ranges else [()]
    num_leaves = len(combos)

    leaf_tokens_list: List[List[int]] = []
    for combo in combos:
        row = []
        for pos, local_choice in enumerate(combo):
            row.append(topk_tokens_per_pos[pos][local_choice])
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


# ---------------------------------------------------------------------------
# v2: EAGLE-2 expand + rerank
# ---------------------------------------------------------------------------

def build_dynamic_tree_v2(
    draft_logits: torch.Tensor,
    anchor_token_ids: torch.LongTensor,
    max_tree_size: int = 8,
    expand_k: int = 3,
    used_tokens: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    EAGLE-2 faithful implementation adapted for DFlash's parallel draft.

    Two phases following the original paper (Li et al., 2024):

    **Phase 1 — Expansion:** Build the draft tree layer by layer.  At each
    depth d, select the top-`expand_k` nodes from the *current layer* ranked
    by their global value V_i (cumulative confidence = product of top-1 probs
    along the path).  Only those nodes are expanded with `expand_k` children
    at depth d+1.  Since DFlash produces all position logits in parallel, we
    read from the pre-computed logits instead of running the draft model again.

    **Phase 2 — Reranking:** Collect *all* nodes across *all* layers, rank
    them globally by value V_i, and select the top `max_tree_size` nodes.
    Shallow high-value nodes can beat deep low-value ones, producing a
    variable-depth tree.  Ties are broken in favour of shallower nodes (as
    specified in the paper).

    Returns same 5-tuple as build_dynamic_tree:
      packed_ids, packed_pos, parent_idx, leaf_paths, leaf_tokens
    """
    if draft_logits.size(0) != 1:
        raise ValueError("Dynamic tree currently supports batch size 1.")

    device = draft_logits.device
    _, seq_len, _ = draft_logits.shape

    logits_pos = draft_logits[0]  # [seq_len, vocab_or_r]
    log_denom = torch.logsumexp(logits_pos, dim=-1, keepdim=True)
    log_probs_all = logits_pos - log_denom  # [seq_len, vocab_or_r]

    topk_logprobs, topk_indices = torch.topk(log_probs_all, k=expand_k, dim=-1)
    topk_logprobs_cpu = topk_logprobs.detach().cpu().tolist()

    if used_tokens is not None:
        used_t = torch.tensor(used_tokens, device=device, dtype=torch.long)
        topk_token_ids = used_t[topk_indices]
        topk_tokens_cpu: List[List[int]] = topk_token_ids.detach().cpu().tolist()
    else:
        topk_tokens_cpu = topk_indices.detach().cpu().tolist()

    # -- Phase 1: Expansion (layer-by-layer, top-k_expand per layer) --------
    # Each node: (token_id, depth, parent_index_in_all_nodes, value=cum_logprob)
    # all_nodes stores every node created during expansion (across all layers).
    # Index 0 is the anchor (root).

    if anchor_token_ids.dim() == 2:
        anchor_token = int(anchor_token_ids[0, 0].detach().cpu().item())
    else:
        anchor_token = int(anchor_token_ids[0].detach().cpu().item())

    # (token_id, depth, parent_idx_in_all_nodes, cum_logprob)
    all_nodes: List[Tuple[int, int, int, float]] = [(anchor_token, 0, -1, 0.0)]
    children_of: List[List[int]] = [[]]  # children_of[i] = list of child indices

    # current_layer holds indices into all_nodes for the latest layer.
    current_layer = [0]

    for d in range(seq_len):
        # Select top-expand_k nodes from current layer by value (cum_logprob).
        layer_with_val = [(idx, all_nodes[idx][3]) for idx in current_layer]
        layer_with_val.sort(key=lambda x: x[1], reverse=True)
        nodes_to_expand = [idx for idx, _ in layer_with_val[:expand_k]]

        next_layer: List[int] = []
        for parent_idx_in_all in nodes_to_expand:
            parent_clp = all_nodes[parent_idx_in_all][3]
            for j in range(expand_k):
                child_clp = parent_clp + topk_logprobs_cpu[d][j]
                child_idx = len(all_nodes)
                all_nodes.append((topk_tokens_cpu[d][j], d + 1, parent_idx_in_all, child_clp))
                children_of.append([])
                children_of[parent_idx_in_all].append(child_idx)
                next_layer.append(child_idx)

        if not next_layer:
            break
        current_layer = next_layer

    # -- Phase 2: Reranking (global top-m by value, prefer shallow) ----------
    # Sort all non-root nodes by (value DESC, depth ASC) — ties broken shallow.
    node_indices = list(range(1, len(all_nodes)))  # exclude root
    node_indices.sort(key=lambda i: (-all_nodes[i][3], all_nodes[i][1]))

    # Greedily select top-m nodes ensuring connectivity (ancestors included).
    selected: set = {0}  # root always included
    for idx in node_indices:
        if len(selected) - 1 >= max_tree_size:  # -1 for root
            break
        # Include this node and all its ancestors.
        chain = []
        cur = idx
        while cur not in selected and cur >= 0:
            chain.append(cur)
            cur = all_nodes[cur][2]  # parent
        for n in reversed(chain):
            selected.add(n)
        if len(selected) - 1 >= max_tree_size:
            break

    # Build the output trie from selected nodes.
    # Map old indices -> new packed indices.
    # Traverse in BFS order to preserve parent-before-child ordering.
    old_to_new: Dict[int, int] = {}
    queue = [0]
    packed_node_tokens: List[int] = []
    packed_node_pos: List[int] = []
    packed_node_parent: List[int] = []

    while queue:
        old_idx = queue.pop(0)
        new_idx = len(packed_node_tokens)
        old_to_new[old_idx] = new_idx
        tok, depth, par_old, _ = all_nodes[old_idx]
        packed_node_tokens.append(tok)
        packed_node_pos.append(depth)
        packed_node_parent.append(old_to_new[par_old] if par_old >= 0 else -1)
        for child_old in children_of[old_idx]:
            if child_old in selected:
                queue.append(child_old)

    # Identify leaves (nodes with no selected children).
    has_selected_child: set = set()
    for idx in selected:
        par_old = all_nodes[idx][2]
        if par_old >= 0 and par_old in selected:
            has_selected_child.add(old_to_new[par_old])

    leaf_new_indices = [old_to_new[idx] for idx in selected
                        if old_to_new[idx] not in has_selected_child and idx != 0]

    if not leaf_new_indices:
        leaf_new_indices = [old_to_new[idx] for idx in selected if idx != 0]

    # Build leaf_paths and leaf_tokens by tracing from each leaf to root.
    leaf_paths_list: List[List[int]] = []
    leaf_tokens_list: List[List[int]] = []
    max_depth = 0

    for leaf_new in leaf_new_indices:
        path = []
        cur = leaf_new
        while cur >= 0:
            path.append(cur)
            cur = packed_node_parent[cur]
        path.reverse()  # root -> ... -> leaf
        tokens = [packed_node_tokens[n] for n in path[1:]]  # exclude anchor
        leaf_paths_list.append(path)
        leaf_tokens_list.append(tokens)
        max_depth = max(max_depth, len(tokens))

    # Pad to uniform width.
    PAD_TOKEN = -1
    for i in range(len(leaf_paths_list)):
        pad_len = max_depth - len(leaf_tokens_list[i])
        if pad_len > 0:
            last_node = leaf_paths_list[i][-1]
            leaf_paths_list[i] += [last_node] * pad_len
            leaf_tokens_list[i] += [PAD_TOKEN] * pad_len

    packed_ids = torch.tensor(packed_node_tokens, device=device, dtype=torch.long).unsqueeze(0)
    packed_pos = torch.tensor(packed_node_pos, device=device, dtype=torch.long).unsqueeze(0)
    parent_idx_t = torch.tensor(packed_node_parent, device=device, dtype=torch.long)
    leaf_paths = torch.tensor(leaf_paths_list, device=device, dtype=torch.long)
    leaf_tokens = torch.tensor(leaf_tokens_list, device=device, dtype=torch.long)

    return packed_ids, packed_pos, parent_idx_t, leaf_paths, leaf_tokens


# ---------------------------------------------------------------------------
# v3: Best-first (priority queue) tree
# ---------------------------------------------------------------------------

def build_bestfirst_tree(
    draft_logits: torch.Tensor,
    anchor_token_ids: torch.LongTensor,
    max_tree_size: int = 8,
    expand_k: int = 3,
    used_tokens: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Best-first (probability-budget) tree builder.

    Uses a priority queue ordered by cumulative log-probability.  At each step
    the highest-probability frontier node is expanded with up to *expand_k*
    children.  Expansion stops when the number of **leaf** nodes (frontier +
    finalized) reaches *max_tree_size*, or when the frontier is empty.

    This naturally allocates width where uncertainty is high and depth where
    confidence is high — no theta thresholds, cartesian products, or separate
    adaptive-depth gating required.

    Returns the same 5-tuple as build_dynamic_tree / build_dynamic_tree_v2:
      packed_ids, packed_pos, parent_idx, leaf_paths, leaf_tokens
    """
    if draft_logits.size(0) != 1:
        raise ValueError("Best-first tree currently supports batch size 1.")

    device = draft_logits.device
    _, seq_len, _ = draft_logits.shape

    logits_pos = draft_logits[0]  # [seq_len, vocab_or_r]
    log_denom = torch.logsumexp(logits_pos, dim=-1, keepdim=True)
    log_probs_all = logits_pos - log_denom  # [seq_len, vocab_or_r]

    topk_logprobs, topk_indices = torch.topk(log_probs_all, k=expand_k, dim=-1)
    topk_logprobs_cpu = topk_logprobs.detach().cpu().tolist()
    topk_indices_cpu = topk_indices.detach().cpu().tolist()

    if used_tokens is not None:
        used_t = torch.tensor(used_tokens, device=device, dtype=torch.long)
        topk_token_ids = used_t[topk_indices]
        topk_tokens_cpu: List[List[int]] = topk_token_ids.detach().cpu().tolist()
    else:
        topk_tokens_cpu = topk_indices_cpu

    # Each heap entry: (neg_cum_logprob, counter, token_list, depth)
    # counter breaks ties deterministically.
    counter = 0
    frontier: List[Tuple[float, int, List[int], int]] = []

    for j in range(expand_k):
        heapq.heappush(frontier, (
            -topk_logprobs_cpu[0][j], counter, [topk_tokens_cpu[0][j]], 1
        ))
        counter += 1

    finalized: List[Tuple[List[int], float]] = []

    while frontier and (len(finalized) + len(frontier)) < max_tree_size:
        neg_clp, _, toks, depth = heapq.heappop(frontier)

        if depth >= seq_len:
            finalized.append((toks, -neg_clp))
            continue

        for j in range(expand_k):
            child_clp = -neg_clp + topk_logprobs_cpu[depth][j]
            heapq.heappush(frontier, (
                -child_clp, counter, toks + [topk_tokens_cpu[depth][j]], depth + 1
            ))
            counter += 1

    # Remaining frontier entries become leaves.
    for neg_clp, _, toks, _ in frontier:
        finalized.append((toks, -neg_clp))

    if not finalized:
        finalized = [([topk_tokens_cpu[0][0]], topk_logprobs_cpu[0][0])]

    # Determine max leaf depth for padding.
    max_depth = max(len(toks) for toks, _ in finalized)

    if anchor_token_ids.dim() == 2:
        anchor_token = int(anchor_token_ids[0, 0].detach().cpu().item())
    else:
        anchor_token = int(anchor_token_ids[0].detach().cpu().item())

    # Build trie from finalized leaves (same structure as v2).
    node_tokens: List[int] = [anchor_token]
    node_pos: List[int] = [0]
    node_parent: List[int] = [-1]
    children_maps: List[Dict[int, int]] = [dict()]
    leaf_paths_list: List[List[int]] = []
    leaf_tokens_list: List[List[int]] = []

    PAD_TOKEN = -1

    for toks, _ in finalized:
        cur = 0
        path = [0]
        for pos, tok in enumerate(toks):
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

        # Pad shorter leaves so all have the same tensor width.
        pad_len = max_depth - len(toks)
        padded_toks = toks + [PAD_TOKEN] * pad_len
        padded_path = path + [cur] * pad_len  # repeat last node index
        leaf_tokens_list.append(padded_toks)
        leaf_paths_list.append(padded_path)

    packed_ids = torch.tensor(node_tokens, device=device, dtype=torch.long).unsqueeze(0)
    packed_pos = torch.tensor(node_pos, device=device, dtype=torch.long).unsqueeze(0)
    parent_idx = torch.tensor(node_parent, device=device, dtype=torch.long)
    leaf_paths = torch.tensor(leaf_paths_list, device=device, dtype=torch.long)
    leaf_tokens = torch.tensor(leaf_tokens_list, device=device, dtype=torch.long)

    return packed_ids, packed_pos, parent_idx, leaf_paths, leaf_tokens


# ---------------------------------------------------------------------------
# Shared: dynamic tree attention mask, leaf selection, adaptive depth
# ---------------------------------------------------------------------------

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

def optimal_tree_depth(
    draft_logits: torch.Tensor,
    threshold: float = 0.0,
) -> int:
    """
    Cumulative-probability depth gating (v1 only).

    Computes the running product of draft top-1 probabilities and returns the
    first depth where it drops below *threshold*.  threshold=0.0 disables gating.
    """
    if threshold <= 0.0:
        return draft_logits.shape[1]

    top1_prob = draft_logits[0].softmax(-1).max(-1).values  # [seq_len]
    seq_len = top1_prob.shape[0]

    cumulative = 1.0
    for d in range(seq_len):
        cumulative *= top1_prob[d].item()
        if cumulative < threshold:
            return max(1, d)

    return seq_len