"""Tree construction and verification utilities for DFlash speculative decoding.

Four tree-building strategies are provided:
  v1  build_dynamic_tree      — per-position threshold branching (cartesian product)
  v2  build_dynamic_tree_v2   — EAGLE-2 expand + rerank (Li et al., 2024)
  v3  build_bestfirst_tree    — best-first search by cumulative log-probability
  v4  build_prefixaware_tree  — prefix-aware greedy (submodular E[tau] maximization)
"""

from typing import Dict, List, Optional, Tuple
import heapq
import itertools
import math
import torch


# ---------------------------------------------------------------------------
# Shared helpers (used by v2/v3/v4 builders)
# ---------------------------------------------------------------------------

def _get_anchor_token(anchor_token_ids: torch.LongTensor) -> int:
    """Extract the scalar anchor token id, handling both 1-D and 2-D inputs."""
    if anchor_token_ids.dim() == 2:
        return int(anchor_token_ids[0, 0].detach().cpu().item())
    return int(anchor_token_ids[0].detach().cpu().item())


def _prepare_topk_logprobs(
    draft_logits: torch.Tensor,
    expand_k: int,
    used_tokens: Optional[List[int]] = None,
) -> Tuple[List[List[float]], List[List[int]], torch.device, int]:
    """Compute per-position top-k log-probabilities and token ids.

    Common setup shared by v2, v3, and v4 builders.

    Returns:
        topk_logprobs_cpu: per-position top-k log-probs as nested Python list
        topk_tokens_cpu:   per-position top-k token ids as nested Python list
        device:            device of the input tensor
        seq_len:           number of draft positions
    """
    device = draft_logits.device
    _, seq_len, _ = draft_logits.shape

    logits_pos = draft_logits[0]  # [seq_len, vocab_or_r]
    log_denom = torch.logsumexp(logits_pos, dim=-1, keepdim=True)
    log_probs_all = logits_pos - log_denom

    topk_logprobs, topk_indices = torch.topk(log_probs_all, k=expand_k, dim=-1)
    topk_logprobs_cpu = topk_logprobs.detach().cpu().tolist()

    if used_tokens is not None:
        used_t = torch.tensor(used_tokens, device=device, dtype=torch.long)
        topk_token_ids = used_t[topk_indices]
        topk_tokens_cpu: List[List[int]] = topk_token_ids.detach().cpu().tolist()
    else:
        topk_tokens_cpu = topk_indices.detach().cpu().tolist()

    return topk_logprobs_cpu, topk_tokens_cpu, device, seq_len


def _pack_trie_from_leaves(
    anchor_token: int,
    finalized: List[Tuple[List[int], ...]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack finalized leaf token sequences into a trie.

    Each entry in *finalized* is a tuple whose first element is the token list.
    Shorter leaves are padded with PAD_TOKEN=-1 so all leaf tensors have
    uniform width.

    Returns: (packed_ids, packed_pos, parent_idx, leaf_paths, leaf_tokens)
    """
    PAD_TOKEN = -1
    max_depth = max(len(entry[0]) for entry in finalized)

    node_tokens: List[int] = [anchor_token]
    node_pos: List[int] = [0]
    node_parent: List[int] = [-1]
    children_maps: List[Dict[int, int]] = [dict()]
    leaf_paths_list: List[List[int]] = []
    leaf_tokens_list: List[List[int]] = []

    for entry in finalized:
        toks = entry[0]
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

        pad_len = max_depth - len(toks)
        leaf_tokens_list.append(toks + [PAD_TOKEN] * pad_len)
        leaf_paths_list.append(path + [cur] * pad_len)

    packed_ids = torch.tensor(node_tokens, device=device, dtype=torch.long).unsqueeze(0)
    packed_pos = torch.tensor(node_pos, device=device, dtype=torch.long).unsqueeze(0)
    parent_idx = torch.tensor(node_parent, device=device, dtype=torch.long)
    leaf_paths = torch.tensor(leaf_paths_list, device=device, dtype=torch.long)
    leaf_tokens = torch.tensor(leaf_tokens_list, device=device, dtype=torch.long)

    return packed_ids, packed_pos, parent_idx, leaf_paths, leaf_tokens


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

    anchor_token = _get_anchor_token(anchor_token_ids)

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

    topk_logprobs_cpu, topk_tokens_cpu, device, seq_len = _prepare_topk_logprobs(
        draft_logits, expand_k, used_tokens,
    )

    # -- Phase 1: Expansion (layer-by-layer, top-k_expand per layer) --------
    anchor_token = _get_anchor_token(anchor_token_ids)

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

    topk_logprobs_cpu, topk_tokens_cpu, device, seq_len = _prepare_topk_logprobs(
        draft_logits, expand_k, used_tokens,
    )

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

    anchor_token = _get_anchor_token(anchor_token_ids)
    return _pack_trie_from_leaves(anchor_token, finalized, device)


def build_efficiency_tree(
    draft_logits: torch.Tensor,
    anchor_token_ids: torch.LongTensor,
    max_tree_size: int = 64,
    expand_k: int = 3,
    alpha: float = 15.0,
    used_tokens: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Efficiency-optimal tree builder (v6).

    Two-phase algorithm:
      Phase 1 — **Expand**: identical to v4 (best-first by cumulative
        log-probability) producing a large candidate pool.
      Phase 2 — **Select**: density-greedy selection that picks candidates
        by the ratio Δf/Δg (marginal E[τ] gain per new trie node).  Stops
        when adding more leaves would decrease the throughput efficiency
        η = E[τ] / (α + |trie|).

    The density-greedy + singleton guarantee achieves (1-1/e) approximation
    to OPT_node(B) via Budgeted Maximum Coverage (Khuller, Moss, Naor 1999).
    This strictly dominates v4's leaf-cardinality guarantee because
    OPT_node(B) ≥ OPT_leaf(m) when B = |trie(v4_m)|.

    Args:
        alpha: Fixed per-step overhead in trie-node-equivalent units.
            Controls self-sizing: larger α → bigger trees (fixed cost
            dominates, so adding nodes is cheap relative to per-step cost).
            Set to 0 to disable self-sizing (uses max_tree_size cap only).
        max_tree_size: Safety cap on number of leaves selected.
    """
    if draft_logits.size(0) != 1:
        raise ValueError("Efficiency tree currently supports batch size 1.")

    topk_logprobs_cpu, topk_tokens_cpu, device, seq_len = _prepare_topk_logprobs(
        draft_logits, expand_k, used_tokens,
    )

    # ================================================================
    # Phase 1: candidate generation (identical to v4 best-first)
    # ================================================================
    pool_budget = max(max_tree_size * 3, max_tree_size + expand_k * seq_len * 2)

    counter = 0
    frontier: List[Tuple[float, int, List[int], int, List[float]]] = []
    candidates: List[Tuple[List[int], List[float]]] = []

    for j in range(expand_k):
        toks = [topk_tokens_cpu[0][j]]
        clp = [topk_logprobs_cpu[0][j]]
        heapq.heappush(frontier, (-clp[-1], counter, toks, 1, clp))
        counter += 1

    while frontier and (len(candidates) + len(frontier)) < pool_budget:
        neg_clp, _, toks, depth, clp_prefix = heapq.heappop(frontier)

        if depth >= seq_len:
            candidates.append((toks, clp_prefix))
            continue

        for j in range(expand_k):
            child_toks = toks + [topk_tokens_cpu[depth][j]]
            child_clp = clp_prefix + [
                clp_prefix[-1] + topk_logprobs_cpu[depth][j]
            ]
            heapq.heappush(
                frontier, (-child_clp[-1], counter, child_toks, depth + 1, child_clp)
            )
            counter += 1

    for _, _, toks, _, clp_prefix in frontier:
        candidates.append((toks, clp_prefix))

    if not candidates:
        candidates = [
            ([topk_tokens_cpu[0][0]], [topk_logprobs_cpu[0][0]])
        ]

    # ================================================================
    # Phase 2: density-greedy selection (Δf/Δg ratio ordering)
    # ================================================================
    covered_prefixes: List[set] = [set() for _ in range(seq_len)]

    cand_prefix_tuples: List[List[tuple]] = []
    cand_prefix_probs: List[List[float]] = []
    for toks, clp_prefix in candidates:
        cand_prefix_tuples.append([tuple(toks[: k + 1]) for k in range(len(toks))])
        cand_prefix_probs.append([math.exp(c) for c in clp_prefix])

    def _marginal_gain_and_cost(idx: int) -> Tuple[float, int]:
        """Δf = probability mass of uncovered prefixes, Δg = count of new trie nodes."""
        gain = 0.0
        cost = 0
        ptups = cand_prefix_tuples[idx]
        pprobs = cand_prefix_probs[idx]
        for k in range(len(ptups)):
            if ptups[k] not in covered_prefixes[k]:
                gain += pprobs[k]
                cost += 1
        return gain, cost

    def _cover(idx: int) -> None:
        for k in range(len(cand_prefix_tuples[idx])):
            covered_prefixes[k].add(cand_prefix_tuples[idx][k])

    f_val = 0.0
    g_val = 1      # root node
    best_eta = 0.0
    finalized: List[Tuple[List[int], float]] = []
    alive = set(range(len(candidates)))

    while alive and len(finalized) < max_tree_size:
        top_ratio = -1.0
        top_ci = -1
        top_df = 0.0
        top_dg = 0
        dead: List[int] = []

        for ci in alive:
            df, dg = _marginal_gain_and_cost(ci)
            if df <= 1e-12 or dg == 0:
                dead.append(ci)
                continue
            ratio = df / dg
            if ratio > top_ratio:
                top_ratio = ratio
                top_ci = ci
                top_df = df
                top_dg = dg

        for ci in dead:
            alive.discard(ci)

        if top_ci < 0:
            break

        if alpha > 0:
            f_new = f_val + top_df
            g_new = g_val + top_dg
            eta_new = f_new / (alpha + g_new)
            if eta_new <= best_eta and finalized:
                break
            best_eta = eta_new

        toks, clp_prefix = candidates[top_ci]
        finalized.append((toks, clp_prefix[-1]))
        _cover(top_ci)
        f_val += top_df
        g_val += top_dg
        alive.discard(top_ci)

    if not finalized:
        toks, clp_prefix = candidates[0]
        finalized = [(toks, clp_prefix[-1])]

    anchor_token = _get_anchor_token(anchor_token_ids)
    return _pack_trie_from_leaves(anchor_token, finalized, device)


def build_node_budget_tree(
    draft_logits: torch.Tensor,
    anchor_token_ids: torch.LongTensor,
    max_tree_size: int = 64,
    expand_k: int = 3,
    score_alpha: float = 1.0,
    score_beta: float = 0.0,
    used_tokens: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Node-budget tree builder (v7) with power-scaled / deviation-penalized scoring.

    Base DDTree: keeps the B highest-probability prefixes under the product
    distribution. Optimal-in-expectation when product ≈ joint.

    Fix A (Q1): replace log q(u) with
        score(u) = Σ_i α^{i-1} log q_i(u_i) - β · #{i : u_i != rank-1}

    α ∈ (0, 1] : depth discount — shallower positions more reliable.
    β ≥ 0      : deviation penalty — flat rank-1 continuations preferred
                  over mixed-rank deep paths where product >> joint.

    α=1, β=0 exactly recovers DDTree. Prefix closure is preserved: extending
    adds α^d · log q ≤ 0 and monotone-non-increasing β · devcount.
    """
    if draft_logits.size(0) != 1:
        raise ValueError("Node-budget tree currently supports batch size 1.")

    topk_logprobs_cpu, topk_tokens_cpu, device, seq_len = _prepare_topk_logprobs(
        draft_logits, expand_k, used_tokens,
    )

    anchor_token = _get_anchor_token(anchor_token_ids)

    counter = 0
    # heap entry: (neg_composite, counter, toks, depth, score, devcount)
    frontier: List[Tuple[float, int, List[int], int, float, int]] = []
    selected: List[Tuple[List[int], float]] = []

    for j in range(expand_k):
        lp = topk_logprobs_cpu[0][j]
        score = lp  # α^0 = 1
        devcount = 1 if j > 0 else 0
        composite = score - score_beta * devcount
        heapq.heappush(
            frontier,
            (-composite, counter, [topk_tokens_cpu[0][j]], 1, score, devcount),
        )
        counter += 1

    while frontier and len(selected) < max_tree_size:
        _, _, toks, depth, score, devcount = heapq.heappop(frontier)
        selected.append((toks, score))

        if depth >= seq_len:
            continue

        alpha_weight = score_alpha ** depth
        for j in range(expand_k):
            new_score = score + alpha_weight * topk_logprobs_cpu[depth][j]
            new_dev = devcount + (1 if j > 0 else 0)
            new_comp = new_score - score_beta * new_dev
            heapq.heappush(
                frontier,
                (
                    -new_comp,
                    counter,
                    toks + [topk_tokens_cpu[depth][j]],
                    depth + 1,
                    new_score,
                    new_dev,
                ),
            )
            counter += 1

    if not selected:
        selected = [([topk_tokens_cpu[0][0]], topk_logprobs_cpu[0][0])]

    # Identify leaves: selected nodes whose children are all outside the
    # selected set.  Under heap ordering, a child can only be selected if
    # its parent was popped first (generates the child), so checking direct
    # children at the next depth suffices.
    selected_set = {tuple(t) for t, _ in selected}
    finalized: List[Tuple[List[int], float]] = []
    for toks, clp in selected:
        t = tuple(toks)
        if len(toks) >= seq_len or all(
            t + (topk_tokens_cpu[len(toks)][j],) not in selected_set
            for j in range(expand_k)
        ):
            finalized.append((toks, clp))

    if not finalized:
        finalized = [selected[0]]

    return _pack_trie_from_leaves(anchor_token, finalized, device)


def build_chained_tree(
    draft_logits: torch.Tensor,
    draft_logits_2: torch.Tensor,
    anchor_token_ids: torch.LongTensor,
    max_tree_size: int = 128,
    expand_k: int = 8,
    score_alpha: float = 1.0,
    score_beta: float = 0.0,
    chain_depth: int = 0,
    used_tokens: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q2: v7 with linear block_2 argmax-chain extension.

    1. Run v7 on block_1 (draft_logits) with max_tree_size budget.
    2. Identify the rank-1 argmax-chain leaf (always present when seq_len <= budget).
    3. From draft_logits_2 (anchored at block_1's argmax-end-token), extract
       chain_depth argmax tokens and append them linearly to the argmax-chain leaf.
    4. Target forward (memory-bandwidth-bound) pays nearly zero extra cost for
       +chain_depth nodes; benefit is up to +chain_depth tau on argmax-accepting steps.

    chain_depth=0 is identity with v7.
    chain_depth=seq_len takes the full second block.
    """
    if draft_logits.size(0) != 1 or draft_logits_2.size(0) != 1:
        raise ValueError("Chained tree currently supports batch size 1.")

    topk_logprobs_cpu, topk_tokens_cpu, device, seq_len = _prepare_topk_logprobs(
        draft_logits, expand_k, used_tokens,
    )
    anchor_token = _get_anchor_token(anchor_token_ids)

    # ---- v7 heap expansion (identical to build_node_budget_tree) ----
    counter = 0
    frontier: List[Tuple[float, int, List[int], int, float, int]] = []
    selected: List[Tuple[List[int], float]] = []

    for j in range(expand_k):
        lp = topk_logprobs_cpu[0][j]
        score = lp
        devcount = 1 if j > 0 else 0
        composite = score - score_beta * devcount
        heapq.heappush(
            frontier,
            (-composite, counter, [topk_tokens_cpu[0][j]], 1, score, devcount),
        )
        counter += 1

    while frontier and len(selected) < max_tree_size:
        _, _, toks, depth, score, devcount = heapq.heappop(frontier)
        selected.append((toks, score))
        if depth >= seq_len:
            continue
        alpha_weight = score_alpha ** depth
        for j in range(expand_k):
            new_score = score + alpha_weight * topk_logprobs_cpu[depth][j]
            new_dev = devcount + (1 if j > 0 else 0)
            new_comp = new_score - score_beta * new_dev
            heapq.heappush(
                frontier,
                (
                    -new_comp,
                    counter,
                    toks + [topk_tokens_cpu[depth][j]],
                    depth + 1,
                    new_score,
                    new_dev,
                ),
            )
            counter += 1

    if not selected:
        selected = [([topk_tokens_cpu[0][0]], topk_logprobs_cpu[0][0])]

    # Identify leaves (v7 convention)
    selected_set = {tuple(t) for t, _ in selected}
    finalized: List[Tuple[List[int], float]] = []
    for toks, clp in selected:
        t = tuple(toks)
        if len(toks) >= seq_len or all(
            t + (topk_tokens_cpu[len(toks)][j],) not in selected_set
            for j in range(expand_k)
        ):
            finalized.append((toks, clp))
    if not finalized:
        finalized = [selected[0]]

    # ---- Q2 extension: append block_2 argmax chain to argmax-chain leaf ----
    if chain_depth > 0:
        # block_1 argmax tokens per position (rank-1 at each of seq_len positions)
        argmax_chain_1 = [topk_tokens_cpu[d][0] for d in range(seq_len)]
        argmax_leaf_key = tuple(argmax_chain_1)

        # Check if the full rank-1 chain is present as a leaf (it almost always is,
        # since v7 pops it as the highest-score path first).
        argmax_leaf_idx = -1
        for i, (toks, _) in enumerate(finalized):
            if tuple(toks) == argmax_leaf_key:
                argmax_leaf_idx = i
                break

        if argmax_leaf_idx >= 0:
            # Block_2 argmax chain (top-1 at each position).
            logits_2 = draft_logits_2[0]
            log_denom_2 = torch.logsumexp(logits_2, dim=-1, keepdim=True)
            _, argmax_idx_2 = torch.max(logits_2, dim=-1)
            if used_tokens is not None:
                used_t = torch.tensor(used_tokens, device=device, dtype=torch.long)
                argmax_chain_2 = used_t[argmax_idx_2].detach().cpu().tolist()
            else:
                argmax_chain_2 = argmax_idx_2.detach().cpu().tolist()

            seq_len_2 = draft_logits_2.shape[1]
            take = min(chain_depth, seq_len_2)
            extension = argmax_chain_2[:take]
            old_toks, old_score = finalized[argmax_leaf_idx]
            finalized[argmax_leaf_idx] = (list(old_toks) + extension, old_score)

    return _pack_trie_from_leaves(anchor_token, finalized, device)


def build_prefixaware_tree(
    draft_logits: torch.Tensor,
    anchor_token_ids: torch.LongTensor,
    max_tree_size: int = 8,
    expand_k: int = 3,
    used_tokens: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prefix-aware greedy tree builder (v4).

    Two-phase algorithm:
      Phase 1 — **Expand**: identical to v3 (best-first by cumulative
        log-probability) but with a larger candidate pool (~3× budget).
        This generates deep, high-probability candidate leaves.
      Phase 2 — **Select**: greedy submodular selection of *max_tree_size*
        leaves from the candidate pool, ordered by marginal E[tau] gain
        (prefix-coverage aware).  Yields a (1-1/e) approximation guarantee
        via Nemhauser-Wolsey-Fisher (1978).

    Returns the same 5-tuple as the other builders:
      packed_ids, packed_pos, parent_idx, leaf_paths, leaf_tokens
    """
    if draft_logits.size(0) != 1:
        raise ValueError("Prefix-aware tree currently supports batch size 1.")

    topk_logprobs_cpu, topk_tokens_cpu, device, seq_len = _prepare_topk_logprobs(
        draft_logits, expand_k, used_tokens,
    )

    # ================================================================
    # Phase 1: candidate generation (v3-style best-first expansion)
    # ================================================================
    pool_budget = max(max_tree_size * 2, max_tree_size + expand_k * seq_len)

    counter = 0
    frontier: List[Tuple[float, int, List[int], int, List[float]]] = []
    candidates: List[Tuple[List[int], List[float]]] = []

    for j in range(expand_k):
        toks = [topk_tokens_cpu[0][j]]
        clp = [topk_logprobs_cpu[0][j]]
        heapq.heappush(frontier, (-clp[-1], counter, toks, 1, clp))
        counter += 1

    while frontier and (len(candidates) + len(frontier)) < pool_budget:
        neg_clp, _, toks, depth, clp_prefix = heapq.heappop(frontier)

        if depth >= seq_len:
            candidates.append((toks, clp_prefix))
            continue

        for j in range(expand_k):
            child_toks = toks + [topk_tokens_cpu[depth][j]]
            child_clp = clp_prefix + [
                clp_prefix[-1] + topk_logprobs_cpu[depth][j]
            ]
            heapq.heappush(
                frontier, (-child_clp[-1], counter, child_toks, depth + 1, child_clp)
            )
            counter += 1

    for _, _, toks, _, clp_prefix in frontier:
        candidates.append((toks, clp_prefix))

    if not candidates:
        candidates = [
            ([topk_tokens_cpu[0][0]], [topk_logprobs_cpu[0][0]])
        ]

    # ================================================================
    # Phase 2: greedy submodular leaf selection (lazy-greedy heap)
    # ================================================================
    covered_prefixes: List[set] = [set() for _ in range(seq_len)]

    # Pre-compute prefix tuples once per candidate (avoids repeated
    # tuple(toks[:k+1]) allocation inside the hot loop).
    cand_prefix_tuples: List[List[tuple]] = []
    cand_prefix_probs: List[List[float]] = []
    for toks, clp_prefix in candidates:
        cand_prefix_tuples.append([tuple(toks[: k + 1]) for k in range(len(toks))])
        cand_prefix_probs.append([math.exp(c) for c in clp_prefix])

    def _marginal_gain_fast(idx: int) -> float:
        gain = 0.0
        ptups = cand_prefix_tuples[idx]
        pprobs = cand_prefix_probs[idx]
        for k in range(len(ptups)):
            if ptups[k] not in covered_prefixes[k]:
                gain += pprobs[k]
        return gain

    def _cover(toks: List[int], idx: int) -> None:
        for k in range(len(cand_prefix_tuples[idx])):
            covered_prefixes[k].add(cand_prefix_tuples[idx][k])

    # Lazy greedy: heap keyed by (stale) upper-bound gain.
    # On pop, recompute; if still top, accept; else re-push.
    sel_counter = 0
    sel_heap: List[Tuple[float, int, int]] = []
    for ci in range(len(candidates)):
        g = _marginal_gain_fast(ci)
        if g > 0:
            heapq.heappush(sel_heap, (-g, sel_counter, ci))
            sel_counter += 1

    finalized: List[Tuple[List[int], float]] = []

    while sel_heap and len(finalized) < max_tree_size:
        _, _, ci = heapq.heappop(sel_heap)
        actual = _marginal_gain_fast(ci)
        if actual <= 0:
            continue
        if sel_heap and actual < -sel_heap[0][0]:
            heapq.heappush(sel_heap, (-actual, sel_counter, ci))
            sel_counter += 1
            continue
        toks, clp_prefix = candidates[ci]
        finalized.append((toks, clp_prefix[-1]))
        _cover(toks, ci)

    if not finalized:
        toks, clp_prefix = candidates[0]
        finalized = [(toks, clp_prefix[-1])]

    anchor_token = _get_anchor_token(anchor_token_ids)
    return _pack_trie_from_leaves(anchor_token, finalized, device)


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

    GPU-vectorized: parent-jumping closure on-device. Avoids the CPU sync +
    Python loop that previously cost ~3 ms/step.
    """
    B, L = position_ids.shape
    device = position_ids.device
    if B != 1:
        raise ValueError("Dynamic tree mask currently supports batch size 1.")

    # GPU parent-jumping. One CPU sync to get the tree depth (small); zero
    # syncs inside the loop. Trims rounds to exactly what's needed.
    max_depth = int(position_ids.max().item())

    allow = torch.eye(L, device=device, dtype=torch.bool)
    current = torch.arange(L, device=device, dtype=torch.long)
    i_idx = current.clone()

    for _ in range(max_depth):
        next_anc = parent_idx[current]
        valid = next_anc >= 0
        safe_next = next_anc.clamp(min=0)
        allow[i_idx, safe_next] = allow[i_idx, safe_next] | valid
        current = torch.where(valid, next_anc, current)

    pos = position_ids[0]                                     # [L]
    q_pos = pos.unsqueeze(-1)
    k_pos = pos.unsqueeze(0)
    self_mask = torch.eye(L, device=device, dtype=torch.bool)
    causal_allow = allow & ((k_pos < q_pos) | self_mask)

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