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
    pdrr_k1: float = 0.0,
    pdrr_k2: float = 0.0,
    pdrr_k3: float = 0.0,
    cgdb_shallow_depth: int = 0,
    cgdb_high_thresh: float = 0.0,
    cgdb_low_thresh: float = 0.0,
    cgdb_mid_k: int = 0,
    bjc_score_fn=None,                # callable(child_depth, child_rank, parent_dev_d, parent_dev_r, draft_q) -> float
    per_pos_ek=None,                  # Optional[List[int]] of length seq_len: per-position expand_k cap
    optree_threshold: float = 0.0,    # Item 3: OPT-Tree termination — stop heap when marginal E[ΔAAL] < threshold
) -> Tuple[List[int], List[int], List[int], List[List[int]], List[List[int]]]:
    """One DDTree build — returns Python lists, no torch ops.

    Optional v8 score adjustments (all zero = plain v7 DDTree):
      - PDRR (Post-Deviation Rank Reweighting): boost non-rank-0 children at
        depth = parent's dev_depth + {1, 2, 3}. Empirically corrects drafter's
        full-mask q-distribution toward target's true post-deviation conditional.
      - CGDB (Confidence-Gated Deep Branching): at depth > cgdb_shallow_depth,
        gate `expand_k` by parent path probability:
            p_path < cgdb_low_thresh   → expand_k = 1 (argmax-only tail)
            p_path < cgdb_high_thresh  → expand_k = min(expand_k, cgdb_mid_k)
            otherwise                  → expand_k unchanged
    """
    import math
    pdrr_active = (pdrr_k1 != 0.0) or (pdrr_k2 != 0.0) or (pdrr_k3 != 0.0)
    cgdb_active = cgdb_shallow_depth > 0 and (cgdb_high_thresh > 0.0 or cgdb_low_thresh > 0.0)

    def _pdrr(child_depth: int, child_j: int, parent_dev_depth: int) -> float:
        if child_j == 0 or parent_dev_depth < 0:
            return 0.0
        k = child_depth - parent_dev_depth
        if k == 1:
            return pdrr_k1
        if k == 2:
            return pdrr_k2
        if k == 3:
            return pdrr_k3
        return 0.0

    def _local_k(depth: int, score: float) -> int:
        # Start from CGDB's path-prob gating (if active) then cap by per-position
        # entropy schedule (if active). Both reduce; we take the min.
        if cgdb_active and (depth + 1) > cgdb_shallow_depth:
            path_prob = math.exp(score) if score > -700 else 0.0
            if cgdb_low_thresh > 0.0 and path_prob < cgdb_low_thresh:
                base = 1
            elif cgdb_high_thresh > 0.0 and path_prob < cgdb_high_thresh and cgdb_mid_k > 0:
                base = min(expand_k, cgdb_mid_k)
            else:
                base = expand_k
        else:
            base = expand_k
        if per_pos_ek is not None and 0 <= depth < len(per_pos_ek):
            base = min(base, max(1, per_pos_ek[depth]))
        return base

    counter = 0
    # heap entry: (neg_composite, counter, toks, depth, score, dev_depth, dev_rank)
    # dev_rank = rank-j at which the path FIRST deviated from rank-0 (-1 if no deviation yet).
    frontier: List[Tuple[float, int, List[int], int, float, int, int]] = []
    selected: List[Tuple[List[int], float]] = []

    # Seed with rank-0..K-1 at depth 1. (Depth-0 expansion has no PDRR/CGDB.)
    for j in range(min(expand_k, len(topk_tokens[0]))):
        lp = topk_logprobs[0][j]
        dev_depth = 0 if j > 0 else -1
        dev_rank = j if j > 0 else -1
        heapq.heappush(frontier, (-lp, counter, [topk_tokens[0][j]], 1, lp, dev_depth, dev_rank))
        counter += 1

    while frontier and len(selected) < max_tree_size:
        # OPT-Tree variable-depth termination: stop if the next pop's
        # path probability (= exp(score)) falls below threshold of marginal
        # contribution.
        if optree_threshold > 0.0:
            top_neg = frontier[0][0]  # smallest = most negative composite
            top_score = -top_neg
            top_path_prob = math.exp(top_score) if top_score > -700 else 0.0
            if top_path_prob < optree_threshold:
                break
        _, _, toks, depth, score, dev_depth, dev_rank = heapq.heappop(frontier)
        selected.append((toks, score))

        if depth >= seq_len:
            continue

        local_k = _local_k(depth, score) if (cgdb_active or per_pos_ek is not None) else expand_k
        for j in range(min(local_k, len(topk_tokens[depth]))):
            child_lp = topk_logprobs[depth][j]
            new_score = score + child_lp
            if dev_depth >= 0:
                new_dev_d = dev_depth
                new_dev_r = dev_rank
            elif j > 0:
                new_dev_d = depth     # parent itself just deviated at this depth
                new_dev_r = j
            else:
                new_dev_d = -1
                new_dev_r = -1
            new_composite = new_score
            if pdrr_active:
                new_composite += _pdrr(depth, j, dev_depth)
            if bjc_score_fn is not None:
                draft_q = math.exp(child_lp) if child_lp > -700 else 0.0
                bjc_corr = bjc_score_fn(depth + 1, j, dev_depth, dev_rank, draft_q)
                new_composite += bjc_corr
            heapq.heappush(
                frontier,
                (-new_composite, counter, toks + [topk_tokens[depth][j]], depth + 1, new_score, new_dev_d, new_dev_r),
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
    pdrr_k1: float = 0.0,
    pdrr_k2: float = 0.0,
    pdrr_k3: float = 0.0,
    cgdb_shallow_depth: int = 0,
    cgdb_high_thresh: float = 0.0,
    cgdb_low_thresh: float = 0.0,
    cgdb_mid_k: int = 0,
    bjc_calib=None,                                          # BJC JointCalib instance
    bjc_anchor_ent: Optional[List[float]] = None,            # [B] anchor entropy norm
    per_pos_ek_per_elem: Optional[List[List[int]]] = None,   # [B][seq_len] entropy-guided ek
    optree_threshold: float = 0.0,                           # Item 3: OPT-Tree termination
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
        # Build a per-element BJC scoring callable that the heap can call.
        bjc_score_fn = None
        if bjc_calib is not None and bjc_calib.warmup_done():
            from model.joint_dist_calib import _bucket_dev_depth, _bucket_dev_rank, _bucket_entropy
            ent_b_b = _bucket_entropy(bjc_anchor_ent[b] if bjc_anchor_ent else 0.5)
            calib = bjc_calib  # closure captures
            def bjc_score_fn(child_depth, child_rank, parent_dev_depth, parent_dev_rank, draft_q):
                return calib.get_log_correction(
                    depth=child_depth,
                    rank=child_rank,
                    dev_d_bucket=_bucket_dev_depth(parent_dev_depth),
                    dev_r_bucket=_bucket_dev_rank(parent_dev_rank),
                    ent_bucket=ent_b_b,
                    draft_q=draft_q,
                )

        per_pos_ek_b = per_pos_ek_per_elem[b] if per_pos_ek_per_elem else None
        node_tokens, node_pos, node_parent, leaf_paths_b, leaf_tokens_b = _build_one_tree(
            topk_lp_cpu[b], topk_tok_cpu[b],
            anchor_token=int(anchors[b]),
            max_tree_size=max_tree_size, expand_k=expand_k, seq_len=seq_len,
            pdrr_k1=pdrr_k1, pdrr_k2=pdrr_k2, pdrr_k3=pdrr_k3,
            cgdb_shallow_depth=cgdb_shallow_depth,
            cgdb_high_thresh=cgdb_high_thresh,
            cgdb_low_thresh=cgdb_low_thresh,
            cgdb_mid_k=cgdb_mid_k,
            bjc_score_fn=bjc_score_fn,
            per_pos_ek=per_pos_ek_b,
            optree_threshold=optree_threshold,
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
    pwls: bool = False,
    bqat_threshold: float = 0.0,
    cppr: bool = False,
    cppr_lambda: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pick the leaf with the longest accept-prefix per batch element.

    Returns (best_leaf [B], n [B]) where n is the number of accepted positions
    AFTER the anchor (i.e. real tokens accepted besides the anchor).

    Optional inference-time enhancements (no training, no extra forwards):
      - pwls (Posterior-Weighted Leaf Selection): among leaves with max accept
        length, pick the leaf whose BONUS position has the highest target top-1
        probability. Strengthens next step's anchor.
      - bqat_threshold (Bonus-Quality-Aware Truncation): if target top-1 prob
        at the chosen bonus is below threshold AND n > 0, truncate by 1 to
        promote a higher-confidence prior position to bonus. Trades 1 token
        of accept length for anchor quality. 0.0 disables.
      - cppr (Cross-Path Posterior Re-Ranking): score each leaf by accept length
        + cppr_lambda * mean target log-prob along its accepted prefix. Captures
        per-leaf JOINT probability under target — picks the most-confident
        accepted path among ties.
    """
    B, N, P = leaf_paths.shape
    depth = P - 1
    V = logits.shape[-1]

    prev_nodes = leaf_paths[:, :, :-1]                            # [B, N, depth]
    realized = leaf_tokens                                        # [B, N, depth]
    pn = prev_nodes.reshape(B, N * depth)                         # [B, N*depth]

    # Memory-efficient acceptance check: at temp=0 we only need each tree node's
    # argmax token (not its full V-vocab logit row). argmax over logits is
    # [B, M] — much smaller than gathered [B, N*depth, V] which OOMs at B≥32.
    if temperature < 1e-5:
        node_argmax = logits.argmax(dim=-1)                       # [B, M]
        pred = node_argmax.gather(1, pn).view(B, N, depth)        # [B, N, depth]
    else:
        gathered = torch.gather(
            logits, 1, pn.unsqueeze(-1).expand(-1, -1, V)
        ).view(B, N, depth, V)
        probs = torch.softmax(gathered.float() / temperature, dim=-1).view(-1, V)
        pred = torch.multinomial(probs, 1).view(B, N, depth)

    matches = (pred == realized) & (realized >= 0)                # padding (=-1) never matches
    acc = matches.cumprod(dim=-1).sum(dim=-1)                     # [B, N]
    # Mask out invalid leaves.
    acc = torch.where(leaf_valid, acc, torch.full_like(acc, -1))

    if not (pwls or cppr) and bqat_threshold <= 0.0:
        n, best_leaf = acc.max(dim=-1)                            # [B], [B]
        n = n.clamp(min=0)
        return best_leaf, n

    b_arange = torch.arange(B, device=acc.device)

    # ---- Compute per-leaf bonus confidence (used by PWLS and BQAT) ----
    acc_clamped = acc.clamp(min=0)                                # [B, N]
    bonus_node_idx = leaf_paths.gather(
        -1, acc_clamped.unsqueeze(-1)
    ).squeeze(-1)                                                  # [B, N]
    bonus_logits = logits.gather(
        1, bonus_node_idx.unsqueeze(-1).expand(-1, -1, V)
    )                                                              # [B, N, V]
    bonus_top1_prob = torch.softmax(bonus_logits.float(), dim=-1).max(dim=-1).values

    # ---- CPPR: per-leaf mean target log-prob along accepted prefix ----
    cppr_score = None
    if cppr:
        # Memory-efficient: compute log P_target(realized_token | parent_node)
        # without materializing the full V dim. Use logsumexp(per-node logits)
        # as the normaliser, gather the realized-token logit via flat indexing.
        lse_per_node = torch.logsumexp(logits.float(), dim=-1)                 # [B, M]
        lse_at_leaf_d = lse_per_node.gather(1, pn).view(B, N, depth)           # [B, N, depth]
        realized_safe = realized.clamp(min=0)                                  # [B, N, depth]
        # Flat-index gather: logits[b, pn[b,k], realized[b,k]] for each (b, k)
        # idx = pn * V + realized_safe, viewed against logits.view(B, M*V).
        flat_idx = pn * V + realized_safe.reshape(B, N * depth)                # [B, N*depth]
        logits_at = logits.view(B, M * V).gather(1, flat_idx).view(B, N, depth)
        lp_realized = logits_at - lse_at_leaf_d                                # [B, N, depth]
        cum = matches.cumprod(dim=-1)                                          # [B, N, depth] bool
        mean_lp = (lp_realized * cum.float()).sum(dim=-1) / cum.float().sum(dim=-1).clamp(min=1)
        cppr_score = acc.float() + cppr_lambda * mean_lp                       # [B, N]
        cppr_score = torch.where(leaf_valid, cppr_score,
                                  torch.full_like(cppr_score, float("-inf")))

    # ---- Pick best leaf ----
    if cppr:
        best_leaf = cppr_score.argmax(dim=-1)                                  # [B]
    elif pwls:
        max_acc, _ = acc.max(dim=-1, keepdim=True)                             # [B, 1]
        is_max = (acc == max_acc) & leaf_valid
        score = torch.where(is_max, bonus_top1_prob,
                            torch.full_like(bonus_top1_prob, -1.0))
        best_leaf = score.argmax(dim=-1)
    else:
        _, best_leaf = acc.max(dim=-1)

    n = acc[b_arange, best_leaf]                                                # [B]
    n = n.clamp(min=0)

    if bqat_threshold > 0.0:
        chosen_bonus_conf = bonus_top1_prob[b_arange, best_leaf]                # [B]
        truncate = (chosen_bonus_conf < bqat_threshold) & (n > 0)
        n = torch.where(truncate, n - 1, n)

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
