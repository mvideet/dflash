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
    score_gamma: float = 0.0,
    score_min_penalty: float = 0.0,
    rank_logprobs: Optional[torch.Tensor] = None,
    rank_logprobs_by_dev: Optional[torch.Tensor] = None,
    narrow_after_dev: int = 0,
    per_pos_expand_k: Optional[List[int]] = None,
    used_tokens: Optional[List[int]] = None,
    cgdb_shallow_depth: int = 0,
    cgdb_high_thresh: float = 0.0,
    cgdb_low_thresh: float = 0.0,
    cgdb_mid_k: int = 0,
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

    rank_logprobs: Optional [seq_len, expand_k] — Q4 legacy depth-indexed
      calibration. Applied uniformly regardless of deviation history.

    rank_logprobs_by_dev: Optional [seq_len, expand_k, 2] — Q4b deviation-
      conditional calibration.  Indexed by (depth, rank, dev_bucket) where
      dev_bucket=0 when path has 0 prior deviations, 1 when ≥1.  Directly
      attacks phantom mixed-rank-deep-paths: the rank-1 continuation AFTER
      a deviation gets a different (empirically much smaller) weight than
      rank-1 on the argmax chain, even though draft's marginal is identical.
      Takes precedence over rank_logprobs when both are provided.
    """
    if draft_logits.size(0) != 1:
        raise ValueError("Node-budget tree currently supports batch size 1.")

    topk_logprobs_cpu, topk_tokens_cpu, device, seq_len = _prepare_topk_logprobs(
        draft_logits, expand_k, used_tokens,
    )

    # Q4b: 3D [seq_len, K, 2] table takes precedence.
    score_table_by_dev = None
    if rank_logprobs_by_dev is not None:
        score_table_by_dev = rank_logprobs_by_dev.detach().cpu().tolist()
        score_table = None
    elif rank_logprobs is not None:
        score_table = rank_logprobs.detach().cpu().tolist()
    else:
        score_table = topk_logprobs_cpu

    def _lp(depth: int, j: int, devcount: int) -> float:
        if score_table_by_dev is not None:
            return score_table_by_dev[depth][j][1 if devcount >= 1 else 0]
        return score_table[depth][j]

    anchor_token = _get_anchor_token(anchor_token_ids)

    counter = 0
    # heap entry: (neg_composite, counter, toks, depth, score, devcount, min_lp)
    # min_lp tracks the smallest log-prob along the prefix; used by the
    # phantom-aware "min-log-prob penalty" (score_min_penalty > 0).
    # Rationale: under additive scoring, a prefix with one low-prob position
    # and many high-prob positions can have moderate sum but near-zero true
    # joint probability (a phantom path).  Adding mu * min_lp (mu > 0)
    # punishes any single weak position more than the sum does — preferring
    # consistent-quality prefixes over mostly-good-one-weak ones.
    frontier: List[Tuple[float, int, List[int], int, float, int, float]] = []
    selected: List[Tuple[List[int], float]] = []

    def _local_k(depth: int, devcount: int) -> int:
        # Per-position entropy-adaptive width (optional): widen at uncertain
        # positions, narrow at confident ones, with the same TOTAL budget.
        # Confident-position rank-2..K contribute ε to E[tau] (draft's top-1
        # already has ~0.9 mass) so spending heap slots on them is wasteful;
        # reallocating to widen at uncertain positions lets us COVER draft's
        # likely target-disagreement ranks (rank-3..8).  `per_pos_expand_k`
        # is a Python list of length seq_len or None.
        k = expand_k
        if per_pos_expand_k is not None:
            k = min(per_pos_expand_k[depth], expand_k)
        if narrow_after_dev > 0 and devcount >= 1:
            k = min(narrow_after_dev, k)
        return k

    def _lookahead(next_pos_idx: int) -> float:
        # Q3: downstream-aware bonus. Reward prefixes whose next position
        # (d+1) has high draft top-1 confidence — a proxy for "the bonus
        # token handoff to step N+1 will land on a high-quality continuation."
        # Zero at/past block end (no in-block lookahead available).
        if next_pos_idx >= seq_len:
            return 0.0
        return topk_logprobs_cpu[next_pos_idx][0]

    for j in range(_local_k(0, 0)):
        lp = _lp(0, j, 0)
        score = lp  # α^0 = 1
        devcount = 1 if j > 0 else 0
        min_lp = lp
        composite = (
            score
            - score_beta * devcount
            + score_gamma * _lookahead(1)
            + score_min_penalty * min_lp
        )
        heapq.heappush(
            frontier,
            (-composite, counter, [topk_tokens_cpu[0][j]], 1, score, devcount, min_lp),
        )
        counter += 1

    while frontier and len(selected) < max_tree_size:
        _, _, toks, depth, score, devcount, min_lp = heapq.heappop(frontier)
        selected.append((toks, score))

        if depth >= seq_len:
            continue

        alpha_weight = score_alpha ** depth
        local_k = _local_k(depth, devcount)
        # Iter 9 — CGDB: at deep depths, gate expand_k by parent path prob.
        # Path-prob gating is structural (not entropy-based) so it composes
        # cleanly with score_min_penalty. Score is cumulative log-prob with
        # score_alpha=1, so exp(score) is the path probability.
        if (cgdb_high_thresh > 0.0 or cgdb_low_thresh > 0.0) and (
            depth + 1 > cgdb_shallow_depth
        ):
            path_prob = math.exp(score) if score > -700 else 0.0
            if path_prob < cgdb_low_thresh:
                local_k = 1  # argmax-only tail
            elif path_prob < cgdb_high_thresh and cgdb_mid_k > 0:
                local_k = min(local_k, cgdb_mid_k)
            # else: full local_k (high confidence, full branching)
        for j in range(local_k):
            child_lp = _lp(depth, j, devcount)
            new_score = score + alpha_weight * child_lp
            new_dev = devcount + (1 if j > 0 else 0)
            new_min_lp = min(min_lp, child_lp)
            new_comp = (
                new_score
                - score_beta * new_dev
                + score_gamma * _lookahead(depth + 1)
                + score_min_penalty * new_min_lp
            )
            heapq.heappush(
                frontier,
                (
                    -new_comp,
                    counter,
                    toks + [topk_tokens_cpu[depth][j]],
                    depth + 1,
                    new_score,
                    new_dev,
                    new_min_lp,
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


def build_v8_tree(
    draft_logits: torch.Tensor,
    anchor_token_ids: torch.LongTensor,
    max_tree_size: int = 128,
    expand_k: int = 8,
    v8_entropy_beta: float = 0.0,
    v8_leaf_gamma: float = 0.0,
    v8_overlap_lambda: float = 0.0,
    v8_pool_multiplier: int = 2,
    v8_dev_depth_cost: int = 0,
    v8_postdev_beta: float = 0.0,
    v8_fdrp_beta: float = 0.0,
    v8_fdrp_exp: float = 2.0,
    v8_fdrc_cap: int = 0,
    v8_spb_alpha: float = 0.0,
    v8_sps_lambda: float = 0.0,
    v8_dae_shallow_k: int = 0,
    v8_dae_shallow_depth: int = 3,
    v8_dae_deep_k: int = 0,
    v8_pdw_k: int = 0,
    # Iter 9 — CGDB: confidence-gated deep branching.
    v8_cgdb_shallow_depth: int = 3,
    v8_cgdb_high_thresh: float = 0.0,
    v8_cgdb_low_thresh: float = 0.0,
    v8_cgdb_mid_k: int = 0,
    # Iter 10 — TT-CGDB: past tail_depth, only argmax extensions.
    v8_tt_depth: int = 0,  # 0=disabled; >0 = depth past which argmax-only.
    # Iter 11 — 3-Tier CGDB: 4 thresholds → 4 expand_k regimes.
    v8_tier3_t_hi: float = 0.0,    # path_prob >= t_hi → expand_k=tier_hi_k
    v8_tier3_t_um: float = 0.0,    # t_um <= prob < t_hi → expand_k=tier_um_k
    v8_tier3_t_lm: float = 0.0,    # t_lm <= prob < t_um → expand_k=tier_lm_k
    v8_tier3_t_lo: float = 0.0,    # prob < t_lo → expand_k=1 (argmax-only)
    v8_tier3_um_k: int = 6,
    v8_tier3_lm_k: int = 3,
    # Iter 12 — MAG: margin-aware gating. Per-depth expand_k by rank-0 vs rank-1 margin.
    v8_mag_high: float = 0.0,   # margin >= mag_high → expand_k=1 (argmax-only)
    v8_mag_low: float = 0.0,    # mag_low <= margin < mag_high → expand_k=mag_mid_k
    v8_mag_mid_k: int = 4,
    v8_mag_shallow_depth: int = 4,  # apply only past this depth
    # Iter 13 — PLDG: smooth power-law deep gating expand_k = max(1, round(K * prob^p)).
    v8_pldg_p: float = 0.0,
    v8_pldg_shallow_depth: int = 4,
    # Iter 14 — VPPS: subtract β · Var(log q_i) from heap priority.
    v8_vpps_beta: float = 0.0,
    # Iter 16 — CMG: gate deep expand_k by Σ margin_i.
    v8_cmg_high: float = 0.0,    # Σ margin >= cmg_high → expand_k=1
    v8_cmg_low: float = 0.0,     # cmg_low <= cum_margin < cmg_high → mid_k
    v8_cmg_mid_k: int = 4,
    v8_cmg_shallow_depth: int = 4,
    # Iter 17 — Adaptive shallow_depth from first-position entropy.
    v8_adapt_sd: bool = False,   # if True, override shallow_depth dynamically
    # Iter 18 — ECS: replace per-step log q(u_i) with offline log P_emp[d, j].
    v8_ecs: bool = False,
    # Iter 19 — CDS: corrective bonus log(P_emp[d,j]/P_marg[j]) added to score.
    v8_cds_lambda: float = 0.0,
    # Iter 21 — SCM: per-(d, j) bucket-coverage bonus.
    v8_scm_alpha: float = 0.0,  # bonus added to nodes covering uncovered bucket
    used_tokens: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Node-budget tree builder v8 — Joint-Conditional Scoring w/ Lazy-Greedy.

    Attacks v7 DDTree's dominant flaw (Flaw 1/2: product distribution
    overestimates joint probability of mixed-rank-deep-paths) via two
    orthogonal signals, both using the draft's own logits (no extra forward
    pass):

    1. **Entropy-gated deviation penalty**. A rank-2+ choice at a LOW-entropy
       position (draft was confident) is a severe shift from the trained
       distribution — subsequent marginals q_{d+1}, q_{d+2} were computed as
       if the argmax continued, so they over-estimate the joint under the
       chosen deviation. Penalise proportionally to that confidence:

           score_core(u) = Σ_{i=1}^{depth} [ log q_i(u_i)
                                           - β_e · 1{u_i ≠ rank-1}
                                                  · (1 - H(q_i)/log K) ]

       At H(q_i) = 0 (perfectly confident), penalty = β_e per deviation. At
       H(q_i) = log K (uniform), penalty = 0 (deviation is free).

       Compared to v7's flat β (Q1, Finding 12), this reallocates the penalty
       to the positions where it actually matters (phantom-prone confident
       positions) rather than spending it uniformly.

       score_core is additive and monotone-non-increasing in depth (each
       extension subtracts log q + β_e-term ≥ 0), so heap enumeration is
       correct and sibling-monotone — parents pop before children, giving a
       prefix-closed candidate pool for free.

    2. **Downstream-aware leaf bonus** (Q3-as-submodular). For every leaf
       u ∈ T, add γ · log q_{depth(u)+1}(argmax). Rewards leaves whose
       bonus-token handoff will land on a confident continuation (better
       chance of matching target, compounding into next step's tau). This
       is NON-ADDITIVE: adding a child to u removes u's leaf bonus and
       adds the child's. That non-additivity is exactly what v4's lazy-
       greedy selector (but not v7's top-B) can exploit.

    Two-stage merge (per docs/v8_merge_plan.md §4):
      - Stage 1: v7-style heap enumeration with score_core, yielding
        pool_size = v8_pool_multiplier · max_tree_size candidates. Pool is
        prefix-closed (heap pops parents first under monotone additive score).
      - Stage 2: select max_tree_size nodes from pool via local-search
        swaps on f(T) = Σ s_core(u) + γ Σ_{u leaf in T} leaf_bonus(u)
                      - λ · Σ_{u ∈ T} (siblings_of_u_in_T),
        subject to prefix closure. λ penalises picking many children of the
        same parent (which share the same phantom base-rate of their common
        deviation prefix).

    At v8_entropy_beta = 0, v8_leaf_gamma = 0, v8_overlap_lambda = 0 AND
    v8_pool_multiplier = 1, this reduces to v7 top-B: use as sanity check.
    """
    if draft_logits.size(0) != 1:
        raise ValueError("v8 tree currently supports batch size 1.")

    # Iter 8 — if DAE active, bump top-K pass to the widest needed width.
    effective_topk = expand_k
    if v8_dae_shallow_k > 0:
        effective_topk = max(expand_k, v8_dae_shallow_k, v8_dae_deep_k)
    topk_logprobs_cpu, topk_tokens_cpu, device, seq_len = _prepare_topk_logprobs(
        draft_logits, effective_topk, used_tokens,
    )
    anchor_token = _get_anchor_token(anchor_token_ids)

    # Iter 7 — Smoothed-Probability Score (SPS). Replace log q_i(token) with
    # log[(1-λ) q_i + λ/K]. Over-weights low-probability alternatives at each
    # depth so more rank-2/3 branches enter top-B. Target greedy deviations
    # often hit these rank-2/3 tokens; v7 under-represents them by scoring
    # strictly by log q (which is very negative for small q). λ=0 restores v7.
    if v8_sps_lambda > 0.0:
        uniform_mass = v8_sps_lambda / max(expand_k, 1)
        w = 1.0 - v8_sps_lambda
        topk_logprobs_cpu = [
            [math.log(w * math.exp(lp) + uniform_mass) for lp in row]
            for row in topk_logprobs_cpu
        ]

    # ---- Per-position confidence from top-K distribution entropy ----
    # H(q) computed over the top-K mass only (normalised). Cheap, accurate
    # proxy for full-vocab entropy when top-K covers most of the mass.
    if v8_entropy_beta != 0.0:
        logits_pos = draft_logits[0]
        log_denom = torch.logsumexp(logits_pos, dim=-1, keepdim=True)
        log_probs_all = logits_pos - log_denom
        topk_lp_gpu, _ = torch.topk(log_probs_all, k=expand_k, dim=-1)
        topk_p = topk_lp_gpu.exp()
        topk_p_norm = topk_p / topk_p.sum(-1, keepdim=True).clamp(min=1e-12)
        entropy = -(topk_p_norm * topk_p_norm.clamp(min=1e-12).log()).sum(-1)
        max_entropy = math.log(max(expand_k, 2))
        conf = (1.0 - entropy / max_entropy).clamp(0.0, 1.0)
        conf_cpu = conf.detach().cpu().tolist()
    else:
        conf_cpu = [0.0] * seq_len

    # Argmax log-prob per depth (for leaf bonus at depth+1).
    argmax_lp_cpu = [row[0] for row in topk_logprobs_cpu]
    # Iter 19 — CDS: load DELTA_LOG correction.
    if v8_cds_lambda != 0.0:
        from model.p_emp_table import DELTA_LOG, NUM_DEPTHS, NUM_RANKS
        _delta = DELTA_LOG  # [d][j]
    # Iter 18 — ECS: load empirical log-prob table once per call.
    if v8_ecs:
        from model.p_emp_table import P_EMP_LOG, NUM_DEPTHS, NUM_RANKS
        # Override topk_logprobs_cpu with empirical priors at corresponding (d, j).
        # Top-K at draft position d-1 corresponds to choices at TREE depth d.
        # P_EMP_LOG[d_pos][j] is calibrated for d_pos = TREE depth d (1-indexed
        # in the trace, so we map draft position p -> tree depth p+1; clamp to
        # NUM_DEPTHS-1 for ranges past the trace).
        ecs_topk_logprobs = []
        for p in range(seq_len):
            tree_d = min(p, NUM_DEPTHS - 1)
            ecs_topk_logprobs.append(P_EMP_LOG[tree_d][:expand_k]
                                     if expand_k <= NUM_RANKS
                                     else P_EMP_LOG[tree_d] +
                                     [P_EMP_LOG[tree_d][-1]] * (expand_k - NUM_RANKS))
        topk_logprobs_cpu = ecs_topk_logprobs  # NOTE: tokens still come from draft top-K
    # Iter 12 — MAG margin per draft-position d: log q_d(rank-0) - log q_d(rank-1).
    # Larger margin = rank-1 dominates more strongly; smaller = closer ties.
    margin_cpu = [
        (row[0] - row[1]) if len(row) > 1 else float("inf")
        for row in topk_logprobs_cpu
    ]
    # Iter 17 — Adaptive shallow_depth from first-position confidence.
    # Easy step (top-1 high): cgdb kicks in earlier (sd small); hard step (top-1
    # low): widen shallow phase (sd larger). Overrides v8_cgdb_shallow_depth.
    _eff_sd = v8_cgdb_shallow_depth
    if v8_adapt_sd:
        top1_prob_d1 = math.exp(topk_logprobs_cpu[0][0])
        if top1_prob_d1 > 0.7:
            _eff_sd = 2
        elif top1_prob_d1 < 0.3:
            _eff_sd = 6
        else:
            _eff_sd = 4

    def _leaf_bonus_for(depth: int) -> float:
        # Bonus if we cut off the tree at a node of this depth (its "next"
        # position is depth, 0-indexed: depth 0 == root, children at depth 1).
        return argmax_lp_cpu[depth] if depth < seq_len else 0.0

    # ---- Stage 1: enumerate candidate pool via heap on score_core ----
    pool_size = max(max_tree_size, v8_pool_multiplier * max_tree_size)
    counter = 0
    # heap/pool entries now carry first_dev_depth (INT_MAX if never deviated),
    # used by the Post-Deviation Depth Penalty. State transition:
    #   rank-0 extension of u: first_dev unchanged.
    #   rank>0 extension of u: first_dev = min(first_dev(u), depth+1).
    # score_core includes the cumulative PDDP penalty (additive, monotone in
    # depth so heap sibling-monotone stays valid).
    NO_DEV = seq_len + 1  # sentinel for "no deviation"
    # Iter 21 — SCM: covered (depth, rank-of-step) buckets. Updated on pop.
    _scm_covered = set() if v8_scm_alpha != 0.0 else None
    # Iter 6 — SPB (Sibling-Probability Budget): per-step α · log(rank+1)
    # penalty at every rank>0 extension. Models greedy-target mutual-
    # exclusivity — siblings are mutually exclusive for acceptance so their
    # coverage contribution decays sub-linearly with rank.
    # 10th field: sum_sq_lp (iter-14 VPPS). 11th field: cum_margin (iter-16 CMG).
    frontier: List[Tuple[float, int, Tuple[int, ...], int, float, int, int, int, int, float, float]] = []
    pool: List[Tuple[Tuple[int, ...], float, int, int, int, int, int]] = []

    # Iter 8 — DAE: per-depth expand_k. When dae_shallow_k > 0, use
    # ek_shallow at depths 1..shallow_depth, ek_deep at deeper. Broader
    # shallow pool catches target deviations at low depths.
    def _ek_at_depth(d: int) -> int:
        """Return expand_k to use for extending to depth d."""
        if v8_dae_shallow_k <= 0:
            return expand_k
        return v8_dae_shallow_k if d <= v8_dae_shallow_depth else v8_dae_deep_k

    for j in range(_ek_at_depth(1)):
        lp = topk_logprobs_cpu[0][j]
        step_dev = 1 if j > 0 else 0
        first_dev = 1 if step_dev else NO_DEV
        first_dev_rank = j if step_dev else 0
        # FDRC hard cap: skip if this deviation rank is above threshold.
        if v8_fdrc_cap > 0 and first_dev_rank > v8_fdrc_cap:
            continue
        score_core = lp - v8_entropy_beta * step_dev * conf_cpu[0]
        if v8_postdev_beta != 0.0 and first_dev <= 1:
            score_core -= v8_postdev_beta * max(1 - first_dev, 0)
        if v8_fdrp_beta != 0.0 and first_dev_rank > 0:
            score_core -= v8_fdrp_beta * (first_dev_rank ** v8_fdrp_exp)
        # Iter 6 — SPB: α · log(rank+1) penalty at each step.
        if v8_spb_alpha != 0.0 and j > 0:
            score_core -= v8_spb_alpha * math.log(j + 1)
        # Iter 19 — CDS: add λ · DELTA_LOG[d=0][j] for the depth-0 (first) step.
        if v8_cds_lambda != 0.0 and j < 8:
            score_core += v8_cds_lambda * _delta[0][j]
        sum_sq_lp = lp * lp
        cum_margin = margin_cpu[0]  # margin at depth 1's underlying position.
        # Iter 14 — VPPS: priority = score_core - β · Var(log q_i along path)
        prio = score_core
        # Variance with depth=1 is 0 → no adjustment at root.
        heapq.heappush(
            frontier,
            (
                -prio, counter,
                (topk_tokens_cpu[0][j],), 1,
                score_core, step_dev, -1, first_dev, first_dev_rank, sum_sq_lp,
                cum_margin,
            ),
        )
        counter += 1

    while frontier and len(pool) < pool_size:
        (_, _, toks, depth, score_core, devcount, parent_idx,
         first_dev, first_dev_rank, sum_sq_lp, cum_margin) = heapq.heappop(frontier)
        cur_idx = len(pool)
        pool.append((toks, score_core, depth, devcount, parent_idx,
                     first_dev, first_dev_rank))
        # Iter 21 — SCM: mark this node's (depth, last-step-rank) bucket covered.
        if _scm_covered is not None:
            # The last step's rank is the rank of the LAST token in toks. Look up.
            if depth >= 1:
                last_tok = toks[-1]
                # Find rank j: it must be in topk_tokens_cpu[depth-1].
                tk = topk_tokens_cpu[depth - 1]
                last_rank = tk.index(last_tok) if last_tok in tk else -1
                _scm_covered.add((depth, last_rank))
        if depth >= seq_len:
            continue
        if v8_dev_depth_cost > 0 and depth + 1 > seq_len - devcount * v8_dev_depth_cost:
            continue
        # Iter 9 — CGDB: at deep depths, gate expand_k by parent path prob.
        _cgdb_ek = _ek_at_depth(depth + 1)
        if (v8_cgdb_high_thresh > 0.0 or v8_cgdb_low_thresh > 0.0) and (
            depth + 1 > _eff_sd
        ):
            path_prob = math.exp(score_core)
            if path_prob < v8_cgdb_low_thresh:
                _cgdb_ek = 1  # argmax-only tail
            elif path_prob < v8_cgdb_high_thresh and v8_cgdb_mid_k > 0:
                _cgdb_ek = min(_cgdb_ek, v8_cgdb_mid_k)
            # else: full _ek_at_depth (high confidence, full branching)
        # Iter 10 — TT-CGDB: past tail_depth, argmax-only extensions.
        if v8_tt_depth > 0 and depth + 1 > v8_tt_depth:
            _cgdb_ek = 1
        # Iter 12 — MAG: per-depth margin gating (only past shallow phase).
        # Margin at draft position `depth` (the position whose top-K we're
        # about to enumerate as children at tree depth `depth+1`).
        if (v8_mag_high > 0.0
                and depth + 1 > v8_mag_shallow_depth
                and depth < seq_len):
            margin_d = margin_cpu[depth]
            if margin_d >= v8_mag_high:
                _cgdb_ek = 1
            elif margin_d >= v8_mag_low and v8_mag_mid_k > 0:
                _cgdb_ek = min(_cgdb_ek, v8_mag_mid_k)
        # Iter 16 — CMG: gate by cumulative margin Σ margin_i along path.
        if v8_cmg_high > 0.0 and depth + 1 > v8_cmg_shallow_depth:
            if cum_margin >= v8_cmg_high:
                _cgdb_ek = 1
            elif cum_margin >= v8_cmg_low and v8_cmg_mid_k > 0:
                _cgdb_ek = min(_cgdb_ek, v8_cmg_mid_k)
        # Iter 13 — PLDG: smooth power-law gating by parent path prob.
        # expand_k(d, prob) = max(1, round(K * prob^p)). Replaces step-tier.
        if v8_pldg_p > 0.0 and depth + 1 > v8_pldg_shallow_depth:
            path_prob_p = math.exp(score_core)
            scaled = max(1, min(_cgdb_ek, round(expand_k * (path_prob_p ** v8_pldg_p))))
            _cgdb_ek = scaled
        # Iter 11 — 3-Tier CGDB: smoother gating with 4 path-prob bands.
        # Active when v8_tier3_t_hi > 0; supersedes iter-9 CGDB at deep depths.
        if v8_tier3_t_hi > 0.0 and (depth + 1 > v8_cgdb_shallow_depth):
            path_prob_t = math.exp(score_core)
            if path_prob_t >= v8_tier3_t_hi:
                _cgdb_ek = _ek_at_depth(depth + 1)
            elif path_prob_t >= v8_tier3_t_um:
                _cgdb_ek = min(_cgdb_ek, v8_tier3_um_k)
            elif path_prob_t >= v8_tier3_t_lm:
                _cgdb_ek = min(_cgdb_ek, v8_tier3_lm_k)
            elif path_prob_t >= v8_tier3_t_lo:
                _cgdb_ek = min(_cgdb_ek, 2)
            else:
                _cgdb_ek = 1
        for j in range(_cgdb_ek):
            lp = topk_logprobs_cpu[depth][j]
            step_dev = 1 if j > 0 else 0
            new_first_dev = min(first_dev, depth + 1) if step_dev else first_dev
            new_first_dev_rank = (
                j if (step_dev and first_dev_rank == 0) else first_dev_rank
            )
            # FDRC hard cap: skip if this extension would be a first deviation
            # at rank j > cap.
            if (v8_fdrc_cap > 0 and step_dev and first_dev_rank == 0
                    and j > v8_fdrc_cap):
                continue
            new_sc = score_core + lp - v8_entropy_beta * step_dev * conf_cpu[depth]
            if v8_postdev_beta != 0.0 and first_dev < NO_DEV and first_dev <= depth:
                new_sc -= v8_postdev_beta
            # FDRP: applied once, at the transition from zero-dev to first-dev.
            if (v8_fdrp_beta != 0.0 and step_dev and first_dev_rank == 0):
                new_sc -= v8_fdrp_beta * (j ** v8_fdrp_exp)
            # Iter 6 — SPB: penalty at every rank>0 step along the path.
            if v8_spb_alpha != 0.0 and j > 0:
                new_sc -= v8_spb_alpha * math.log(j + 1)
            # Iter 19 — CDS: per-step bonus from delta table.
            if v8_cds_lambda != 0.0 and j < 8:
                d_idx = min(depth, 15)  # depth here is parent's depth; child at depth+1
                new_sc += v8_cds_lambda * _delta[d_idx][j]
            # Iter 21 — SCM: bonus for novel (depth+1, j) bucket.
            if _scm_covered is not None and (depth + 1, j) not in _scm_covered:
                new_sc += v8_scm_alpha
            # Iter 8b — PDW: reward children of "just-deviated" parent at
            # exactly one step past the deviation. Targets the joint-vs-product
            # shift: v7's argmax-tail-after-deviation assumes marginal q_{d+1}
            # is correct under the deviation; PDW boosts ALL children's scores
            # at post-dev step so multiple rank-j alternatives enter top-B.
            # Specifically fires when parent first_dev == parent's depth.
            if v8_pdw_k != 0 and first_dev != NO_DEV and first_dev == depth:
                # j up to v8_pdw_k gets a +α boost. j=0..v8_pdw_k-1 get a
                # reward scaled by (v8_pdw_k - j)/v8_pdw_k, so rank-0 gets
                # the largest boost, rank-(pdw_k-1) gets the smallest.
                # v8_pdw_k acts as both a WIDTH limit and a strength scale.
                if j < abs(v8_pdw_k):
                    new_sc += 0.5 * (abs(v8_pdw_k) - j) / abs(v8_pdw_k)
            new_dev = devcount + step_dev
            new_toks = toks + (topk_tokens_cpu[depth][j],)
            new_sum_sq = sum_sq_lp + lp * lp
            # margin at the position we just expanded (draft position `depth`).
            new_cum_margin = cum_margin + (margin_cpu[depth] if depth < seq_len else 0.0)
            # Iter 14 VPPS: priority = new_sc - β · variance(log q along path)
            new_prio = new_sc
            if v8_vpps_beta > 0.0:
                # depth + 1 is the new depth.
                d_new = depth + 1
                # Use raw log q sum: treat new_sc as approx. (small bias from
                # other penalties is acceptable since this is heap priority).
                mean_lp = new_sc / d_new
                var_lp = (new_sum_sq / d_new) - (mean_lp * mean_lp)
                if var_lp > 0:
                    new_prio = new_sc - v8_vpps_beta * var_lp
            heapq.heappush(
                frontier,
                (
                    -new_prio, counter,
                    new_toks, depth + 1,
                    new_sc, new_dev, cur_idx, new_first_dev, new_first_dev_rank,
                    new_sum_sq, new_cum_margin,
                ),
            )
            counter += 1

    if not pool:
        fallback = [(topk_tokens_cpu[0][0],), topk_logprobs_cpu[0][0]]
        finalized = [(list(fallback[0]), fallback[1])]
        return _pack_trie_from_leaves(anchor_token, finalized, device)

    P = len(pool)
    pool_parent = [e[4] for e in pool]         # parent index in pool (-1 = root)
    pool_depth = [e[2] for e in pool]
    pool_score = [e[1] for e in pool]
    pool_leaf_b = [
        _leaf_bonus_for(e[2]) for e in pool    # leaf bonus if node i is a leaf
    ]
    # Children list per pool entry (for leaf detection after selection).
    pool_children: List[List[int]] = [[] for _ in range(P)]
    for i in range(P):
        p = pool_parent[i]
        if p >= 0:
            pool_children[p].append(i)

    # ---- Stage 2: selection ----
    # Shortcut: if pool_size == B and γ == 0 and λ == 0, Stage 2 is identity
    # (heap-pop-order top-B is already optimal under additive score_core).
    no_stage2 = (
        v8_leaf_gamma == 0.0
        and v8_overlap_lambda == 0.0
        and pool_size <= max_tree_size
    )

    if no_stage2:
        selected = [False] * P
        for i in range(min(max_tree_size, P)):
            selected[i] = True
    else:
        # Fast pool-reselect: compute an "if-leaf" effective score per pool
        # entry, sort descending, and greedily include with prefix-closure.
        #
        # eff_score(u) = pool_score(u) + γ · leaf_bonus(u)
        #              - λ · (would-be-sibling-count-at-entry)
        #
        # This is an APPROXIMATION to the true non-additive lazy-greedy
        # objective — we assume every candidate would be a leaf if included.
        # In practice, when prefix-closure forces a parent's inclusion, the
        # parent stops being a leaf but is still "paid for" with eff_score;
        # this over-counts γ by γ·(leaf_bonus(parent) - leaf_bonus(child)),
        # a O(γ·|log q|) error per internal node. Acceptable in return for
        # O(P log P) instead of O(B³) runtime.
        eff_score: List[float] = [0.0] * P
        for i in range(P):
            s = pool_score[i]
            if v8_leaf_gamma != 0.0:
                s += v8_leaf_gamma * pool_leaf_b[i]
            eff_score[i] = s

        # Optionally fold a pool-structure-dependent sibling-overlap penalty
        # into eff_score. We use `sibling_rank[i]` ∈ {0, 1, 2, ...} — the
        # rank of i among its siblings in the pool, ordered by pool_score.
        # Penalty: -λ · sibling_rank[i]. Rank-0 sibling (best of its parent)
        # is free; each worse sibling pays λ. This is CHEAP to compute,
        # order-independent, and captures the "don't waste budget on many
        # siblings of the same parent" intuition without order-dependent
        # bookkeeping.
        if v8_overlap_lambda != 0.0:
            # Group pool entries by parent; assign sibling rank by pool_score.
            from collections import defaultdict
            by_parent: Dict[int, List[int]] = defaultdict(list)
            for i in range(P):
                by_parent[pool_parent[i]].append(i)
            for p, siblings in by_parent.items():
                siblings.sort(key=lambda i: -pool_score[i])
                for rank, i in enumerate(siblings):
                    eff_score[i] -= v8_overlap_lambda * rank

        # Sort pool-indices by effective score, descending.
        order = sorted(range(P), key=lambda i: -eff_score[i])

        selected = [False] * P
        n_selected = 0

        for i in order:
            if selected[i]:
                continue
            # Walk unselected ancestor chain (pool is prefix-closed, so all
            # ancestors are in pool).
            chain = []
            cur = i
            while cur != -1 and not selected[cur]:
                chain.append(cur)
                cur = pool_parent[cur]
            if not chain:
                continue
            if n_selected + len(chain) > max_tree_size:
                continue
            for c in chain:
                selected[c] = True
                n_selected += 1
            if n_selected >= max_tree_size:
                break

    # ---- Stage 3: extract leaves of the selected tree and pack ----
    # A leaf in T is any selected node with no selected children.
    selected_indices = [i for i in range(P) if selected[i]]
    finalized: List[Tuple[List[int], float]] = []
    for i in selected_indices:
        has_selected_child = any(selected[c] for c in pool_children[i])
        if not has_selected_child:
            finalized.append((list(pool[i][0]), pool[i][1]))

    if not finalized:
        # Absolute fallback: take top-1 from pool.
        finalized = [(list(pool[0][0]), pool[0][1])]

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