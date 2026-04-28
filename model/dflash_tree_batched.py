"""Batched versions of v7 (DDTree) tree-build / mask / leaf-select.

The originals in `dflash_tree.py` hard-assert B=1 because they were written
for the single-stream `dflash_generate` driver. To benchmark v7 against
vanilla AR at realistic batch sizes we need:

  - per-element tree construction (different prompts -> different trees)
  - padded packed tensors of shape [B, max_tree_size, ...] for one
    batched target.model forward
  - per-element ragged accept-path selection with pad-to-max(n)
  - per-element ragged KV-cache trim that keeps the cache aligned across B

This module is a focused re-implementation of the **core** DDTree algorithm
only (no CGDB, PDRR, narrow_after_dev, calibration, APB, etc.). Those v7
extensions are independent score adjustments and could be ported the same
way later — they don't change the batched plumbing.

Convention
----------
- All "B" axes are leading.
- Padded slots use `_PAD_NODE` (an index that points at the anchor of that
  element's own tree, so attention through it never escapes the prefix).
- The anchor at trie position 0 is always the start of every element's tree.
"""

from typing import List, Tuple, Optional
import heapq
import torch


_PAD_NODE = 0   # anchor node — safe sentinel: always points to root, valid for attention


# ---------------------------------------------------------------------------
# Per-element pure-Python core (one DDTree build for one element)
# ---------------------------------------------------------------------------

def _build_one_tree(
    topk_logprobs: List[List[float]],   # [seq_len][K]
    topk_tokens: List[List[int]],       # [seq_len][K]
    anchor_token: int,
    max_tree_size: int,
    expand_k: int,
    seq_len: int,
) -> Tuple[List[int], List[int], List[int], List[List[int]], List[List[int]]]:
    """One DDTree build — returns Python lists, no torch ops.

    Returns:
        node_tokens : list[int]            length M (1 ≤ M ≤ max_tree_size)
        node_pos    : list[int]            length M
        node_parent : list[int]            length M (parent index in node_tokens; -1 for root)
        leaf_paths  : list[list[int]]      [N_leaves, max_path_len]   (rectangular, padded with last node)
        leaf_tokens : list[list[int]]      [N_leaves, max_path_len-1] (rectangular, padded with -1)
    """
    counter = 0
    frontier: List[Tuple[float, int, List[int], int, float]] = []
    selected: List[Tuple[List[int], float]] = []

    # Seed with rank-0..K-1 at depth 1.
    for j in range(min(expand_k, len(topk_tokens[0]))):
        lp = topk_logprobs[0][j]
        heapq.heappush(frontier, (-lp, counter, [topk_tokens[0][j]], 1, lp))
        counter += 1

    while frontier and len(selected) < max_tree_size:
        _, _, toks, depth, score = heapq.heappop(frontier)
        selected.append((toks, score))

        if depth >= seq_len:
            continue

        for j in range(min(expand_k, len(topk_tokens[depth]))):
            child_lp = topk_logprobs[depth][j]
            new_score = score + child_lp
            heapq.heappush(
                frontier,
                (-new_score, counter, toks + [topk_tokens[depth][j]], depth + 1, new_score),
            )
            counter += 1

    if not selected:
        selected = [([topk_tokens[0][0]], topk_logprobs[0][0])]

    # Identify leaves under the selected-set closure.
    selected_set = {tuple(t) for t, _ in selected}
    finalized: List[Tuple[List[int], float]] = []
    for toks, clp in selected:
        t = tuple(toks)
        if len(toks) >= seq_len or all(
            t + (topk_tokens[len(toks)][j],) not in selected_set
            for j in range(min(expand_k, len(topk_tokens[len(toks)])))
        ):
            finalized.append((toks, clp))
    if not finalized:
        finalized = [selected[0]]

    # Build trie.
    node_tokens: List[int] = [anchor_token]
    node_pos: List[int] = [0]
    node_parent: List[int] = [-1]
    children_maps: List[dict] = [dict()]
    leaf_paths_list: List[List[int]] = []
    leaf_tokens_list: List[List[int]] = []
    max_depth = max(len(entry[0]) for entry in finalized)

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

        pad = max_depth - len(toks)
        leaf_tokens_list.append(toks + [-1] * pad)
        leaf_paths_list.append(path + [cur] * pad)

    return node_tokens, node_pos, node_parent, leaf_paths_list, leaf_tokens_list


# ---------------------------------------------------------------------------
# Batched API
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_node_budget_tree_batched(
    draft_logits: torch.Tensor,            # [B, seq_len, V]
    anchor_token_ids: torch.LongTensor,    # [B] or [B, *]
    max_tree_size: int,
    expand_k: int,
    used_tokens: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched core DDTree builder.

    Returns
    -------
    packed_ids        : [B, M_max]            padded with PAD_NODE positions cloning anchor
    packed_pos        : [B, M_max]            padded with 0
    parent_idx        : [B, M_max]            padded with 0 (so attention is well-defined)
    node_valid        : [B, M_max]            True where node is real
    leaf_paths        : [B, N_max, P_max]     padded by repeating last node
    leaf_tokens       : [B, N_max, P_max-1]   padded with -1
    leaf_valid        : [B, N_max]            True where leaf is real
    """
    B, seq_len, _ = draft_logits.shape
    device = draft_logits.device

    # Top-k once, on GPU, then move to CPU for the heap loop.
    log_denom = torch.logsumexp(draft_logits, dim=-1, keepdim=True)
    log_probs = draft_logits - log_denom
    topk_lp, topk_idx = torch.topk(log_probs, k=expand_k, dim=-1)  # [B, seq_len, K]

    if used_tokens is not None:
        used_t = torch.tensor(used_tokens, device=device, dtype=torch.long)
        topk_tok = used_t[topk_idx]
    else:
        topk_tok = topk_idx
    topk_lp_cpu = topk_lp.detach().cpu().tolist()
    topk_tok_cpu = topk_tok.detach().cpu().tolist()

    if anchor_token_ids.dim() == 1:
        anchors = anchor_token_ids.detach().cpu().tolist()
    else:
        anchors = anchor_token_ids[:, 0].detach().cpu().tolist()

    # Build all elements' Python lists first (heap+trie ops are GIL-bound),
    # then batch-convert to tensors with ONE host→device transfer per output.
    per_elem = []
    M_max = 0
    N_max = 0
    P_max = 0
    for b in range(B):
        node_tokens, node_pos, node_parent, leaf_paths_b, leaf_tokens_b = _build_one_tree(
            topk_lp_cpu[b], topk_tok_cpu[b],
            anchor_token=int(anchors[b]),
            max_tree_size=max_tree_size, expand_k=expand_k, seq_len=seq_len,
        )
        per_elem.append((node_tokens, node_pos, node_parent, leaf_paths_b, leaf_tokens_b))
        M_max = max(M_max, len(node_tokens))
        N_max = max(N_max, len(leaf_paths_b))
        P_max = max(P_max, len(leaf_paths_b[0]) if leaf_paths_b else 1)

    # CPU-side padded nested lists (cheap Python; no GPU sync per element).
    pad_packed_ids = []
    pad_packed_pos = []
    pad_parent_idx = []
    pad_node_valid = []
    pad_leaf_paths = []
    pad_leaf_tokens = []
    pad_leaf_valid = []
    for n_toks, n_pos, n_par, l_paths, l_toks in per_elem:
        M = len(n_toks)
        pad_M = M_max - M
        pad_packed_ids.append(n_toks + [0] * pad_M)
        pad_packed_pos.append(n_pos + [0] * pad_M)
        pad_parent_idx.append([(p if p >= 0 else 0) for p in n_par] + [0] * pad_M)
        pad_node_valid.append([True] * M + [False] * pad_M)

        N = len(l_paths)
        P = len(l_paths[0]) if l_paths else 1
        # Pad each leaf path to P_max by repeating the last node, leaf_tokens with -1.
        padded_paths = [list(p) + [p[-1]] * (P_max - P) for p in l_paths]
        padded_toks = [list(t) + [-1] * (P_max - 1 - len(t)) for t in l_toks]
        # Pad rows up to N_max with all-zero / all-(-1).
        while len(padded_paths) < N_max:
            padded_paths.append([0] * P_max)
            padded_toks.append([-1] * (P_max - 1))
        pad_leaf_paths.append(padded_paths)
        pad_leaf_tokens.append(padded_toks)
        pad_leaf_valid.append([True] * N + [False] * (N_max - N))

    # Single host→device transfer per output (much faster than per-elem torch.tensor).
    packed_ids = torch.tensor(pad_packed_ids, dtype=torch.long, device=device)
    packed_pos = torch.tensor(pad_packed_pos, dtype=torch.long, device=device)
    parent_idx = torch.tensor(pad_parent_idx, dtype=torch.long, device=device)
    node_valid = torch.tensor(pad_node_valid, dtype=torch.bool, device=device)
    leaf_paths = torch.tensor(pad_leaf_paths, dtype=torch.long, device=device)
    leaf_tokens = torch.tensor(pad_leaf_tokens, dtype=torch.long, device=device)
    leaf_valid = torch.tensor(pad_leaf_valid, dtype=torch.bool, device=device)

    return packed_ids, packed_pos, parent_idx, node_valid, leaf_paths, leaf_tokens, leaf_valid


@torch.no_grad()
def create_tree_attention_mask_batched(
    position_ids: torch.LongTensor,   # [B, L]   tree positions (relative)
    parent_idx: torch.LongTensor,     # [B, L]
    node_valid: torch.Tensor,         # [B, L] bool — pad nodes use parent=0, allow=trivial
    prefix_len: int,
) -> torch.Tensor:
    """Vectorized parent-jumping closure across the batch dim.

    Returns mask of shape [B, 1, L, prefix_len + L] with bf16 -inf at blocked
    positions and 0 at allowed. Prefix columns are always 0.
    """
    B, L = position_ids.shape
    device = position_ids.device

    eye_BL = torch.eye(L, device=device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    allow = eye_BL.clone()                                       # [B, L, L] start with self-attention

    current = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1).clone()  # [B, L]
    i_idx = current.clone()                                       # row index, persistent
    b_arange = torch.arange(B, device=device).unsqueeze(1).expand(-1, L)

    max_depth = int(position_ids.max().item())
    for _ in range(max_depth):
        next_anc = parent_idx.gather(1, current)                  # [B, L]
        valid = next_anc >= 0                                     # always true here (we set -1 → 0)
        # additionally: a pad-node at column c has parent_idx=0 already; it
        # can't reach beyond the anchor. That's fine.
        safe_next = next_anc.clamp(min=0)
        cur_allow = allow[b_arange, i_idx, safe_next]
        allow[b_arange, i_idx, safe_next] = cur_allow | valid
        current = torch.where(valid, next_anc, current)

    pos = position_ids                                            # [B, L]
    q_pos = pos.unsqueeze(-1)                                     # [B, L, 1]
    k_pos = pos.unsqueeze(-2)                                     # [B, 1, L]
    self_mask = eye_BL
    causal_allow = allow & ((k_pos < q_pos) | self_mask)

    # Mask out invalid (padded) keys completely (not even self-attention).
    nv = node_valid.unsqueeze(1).expand(-1, L, -1)                # [B, L, L] — column validity
    causal_allow = causal_allow & nv

    min_val = torch.finfo(torch.bfloat16).min
    mask = torch.full((B, 1, L, prefix_len + L), min_val, device=device, dtype=torch.bfloat16)
    if prefix_len > 0:
        mask[:, :, :, :prefix_len] = 0
    mask[:, :, :, prefix_len:].masked_fill_(causal_allow.unsqueeze(1), 0)
    return mask


@torch.no_grad()
def select_best_dynamic_leaf_batched(
    logits: torch.Tensor,             # [B, L, V]
    leaf_paths: torch.Tensor,         # [B, N, P]
    leaf_tokens: torch.Tensor,        # [B, N, P-1]
    leaf_valid: torch.Tensor,         # [B, N]
    temperature: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pick the leaf with the longest accept-prefix per batch element.

    Returns (best_leaf [B], n [B]) where n is the number of accepted positions
    AFTER the anchor (i.e. real tokens accepted besides the anchor).
    """
    B, N, P = leaf_paths.shape
    depth = P - 1
    V = logits.shape[-1]

    prev_nodes = leaf_paths[:, :, :-1]                            # [B, N, depth]
    realized = leaf_tokens                                        # [B, N, depth]

    # Gather logits[b, prev_nodes[b, n, d], :] -> [B, N, depth, V]
    pn = prev_nodes.reshape(B, N * depth)                         # [B, N*depth]
    gathered = torch.gather(
        logits, 1, pn.unsqueeze(-1).expand(-1, -1, V)
    ).view(B, N, depth, V)

    if temperature < 1e-5:
        pred = gathered.argmax(dim=-1)                            # [B, N, depth]
    else:
        probs = torch.softmax(gathered.float() / temperature, dim=-1).view(-1, V)
        pred = torch.multinomial(probs, 1).view(B, N, depth)

    matches = (pred == realized) & (realized >= 0)                # padding (=-1) never matches
    acc = matches.cumprod(dim=-1).sum(dim=-1)                     # [B, N]
    # Mask out invalid leaves.
    acc = torch.where(leaf_valid, acc, torch.full_like(acc, -1))
    n, best_leaf = acc.max(dim=-1)                                # [B], [B]
    n = n.clamp(min=0)
    return best_leaf, n


# ---------------------------------------------------------------------------
# Per-element ragged KV-cache trim (pad-to-max-n strategy)
# ---------------------------------------------------------------------------

@torch.no_grad()
def trim_target_kv_cache_batched(
    past_key_values,
    prefix_lens: torch.Tensor,          # [B] — per-element prefix length BEFORE this step
    accepted_paths: torch.Tensor,       # [B, max_n_plus_1] tree-node indices to keep
    n_per_elem: torch.Tensor,           # [B] — accepted lengths (n+1 = real keep count)
    device: torch.device,
) -> torch.Tensor:
    """Per-element trim: keep prefix[b] + accepted_paths[b, :n_per_elem[b]] for each b.

    Strategy: pad to max_n_plus_1 per row (right-pad with the last accepted
    index, repeated). Cache becomes shape [B, H, max_prefix + max_n_plus_1, D].

    Returns: new_prefix_lens [B] = prefix_lens + n_per_elem (per-element).
    """
    B = prefix_lens.shape[0]
    max_prefix = int(prefix_lens.max().item())
    max_n_plus_1 = int(n_per_elem.max().item())
    new_seq_len = max_prefix + max_n_plus_1

    # Build per-element keep-index list of length new_seq_len.
    # prefix part: positions 0..prefix_lens[b]-1 (rest pad with idx 0 — anchor of prefix)
    # accepted part: prefix_lens[b] + accepted_paths[b, :n+1]
    # tail pad: repeat last accepted-cache-index

    # For each element, need indices: [0,1,...,prefix-1, prefix+ap_0, prefix+ap_1, ..., prefix+ap_{n}, repeat last]
    # Pad prefix to max_prefix by repeating index 0 in unused slots (those slots
    # are read by attention as pad-positions → masked downstream by attention
    # mask in subsequent steps).
    keep = torch.zeros(B, new_seq_len, dtype=torch.long, device=device)
    for b in range(B):
        pl = int(prefix_lens[b].item())
        n1 = int(n_per_elem[b].item())
        keep[b, :pl] = torch.arange(pl, device=device)
        # Pad prefix slots [pl, max_prefix) with 0 (these positions still exist
        # in the cache; will be masked out via attention_mask.).
        if pl < max_prefix:
            keep[b, pl:max_prefix] = 0
        ap = max_prefix + accepted_paths[b, :n1]
        keep[b, max_prefix:max_prefix + n1] = ap
        if n1 < max_n_plus_1:
            keep[b, max_prefix + n1:] = ap[-1] if n1 > 0 else 0

    # Apply gather to each layer's keys/values.
    for layer in past_key_values.layers:
        # keys: [B, H, S_old, D]
        H = layer.keys.shape[1]
        D = layer.keys.shape[3]
        idx = keep.view(B, 1, new_seq_len, 1).expand(-1, H, -1, D)
        layer.keys = layer.keys.gather(2, idx).contiguous()
        layer.values = layer.values.gather(2, idx).contiguous()

    return prefix_lens + n_per_elem
