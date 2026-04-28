"""Sequoia-style DP for optimal tree layer widths under positional acceptance.

Adapted to factorized marginal Q from a one-pass diffusion drafter.

Standard Sequoia (Chen et al, 2024) solves:

   f(n, h, j) = optimal expected accepted tokens for a tree of n nodes,
                height ≤ h, root has at least j children.

The recurrence under POSITIONAL ACCEPTANCE p_d (probability that target
accepts the rank-0 candidate at depth d, averaged over realizations):

   f(n, h, j) = max over (k1, k2, ..., kj) summing to n
                of  sum_i p_d(layer of child i) * f(...)

For the diffusion / DDTree case, q_i are MARGINALS not conditionals, so
expected acceptance compounds DIFFERENTLY: factorized E[τ] = sum over
prefixes of Π q_i. The optimal shape is more aggressively front-heavy
because compounding error grows faster than under AR conditionals.

Outputs a layer-width vector (n_1, n_2, ..., n_D) telling the heap-driven
tree-build how to allocate the M budget per depth.

Inputs:
   p_d : [max_depth] float — empirical P(target argmax = draft argmax | depth d).
        Estimated online from verify outputs (cross-batch pooled).
   M   : total node budget.
   K   : max per-position fanout (= expand_k).

Output:
   width_vec : [max_depth] int — width(d) ∈ [0, K] s.t. sum = M+1 (incl. root).
"""
from typing import List, Optional

import numpy as np


def solve_sequoia_widths(
    p_d: np.ndarray,        # [max_depth] positional acceptance
    M: int,                 # node budget
    K: int = 8,             # max fanout per parent (expand_k)
    cumulative_q: Optional[np.ndarray] = None,  # [max_depth] cumulative Π q_i (per-position)
) -> List[int]:
    """Greedy DP for layer widths.

    The optimization: given budget M and positional acceptance p_d, choose
    width(d) ∈ [0, K] such that:
        E[τ] = sum_d (Π_{i ≤ d} p_i) * width(d) / sum width(0..d-1)
    is maximized.

    Approximation (matches Sequoia's recurrence): treat each layer's
    contribution as p^d * width(d) / parents_at_d-1, and greedily fill
    the budget from depth 0 outward.
    """
    D = len(p_d)
    width = [0] * D
    # Cumulative chain probability (the "trunk" survival probability).
    chain_prob = np.cumprod(p_d)  # Π p_i
    if cumulative_q is not None:
        # Diffusion variant: weight by drafter's own chain probability too.
        chain_prob = np.minimum(chain_prob, cumulative_q)

    # Marginal value of one more node at depth d:
    #   value ≈ (chain_prob[d] / parents_at_d-1) * (1 - already_filled / K)
    # Greedy: at each iteration pick the depth where adding one node has
    # max marginal value. The "parents_at_d-1" enforces that we can't
    # overfill a depth without first having parents at the previous depth.
    parents_at = [1] + [0] * (D - 1)  # depth 0 has the anchor as parent

    remaining = M
    while remaining > 0:
        best_d, best_val = -1, -1.0
        for d in range(D):
            # Cap fanout: width(d) can't exceed parents_at[d-1] * K.
            cap = parents_at[d - 1] * K if d > 0 else K
            if width[d] >= cap:
                continue
            # Marginal value: chain_prob to here, divided by current width
            # (saturating returns).
            val = chain_prob[d] / max(width[d] + 1, 1)
            if val > best_val:
                best_val = val
                best_d = d
        if best_d < 0:
            break
        width[best_d] += 1
        # If this is the first node at depth d, depth d+1's parent count grows.
        if best_d + 1 < D:
            # Each new node at depth d adds K potential children at depth d+1.
            parents_at[best_d] += 1 if width[best_d] == 1 else 0  # actually +1 always
            # Correction:
            parents_at[best_d] = width[best_d]
        remaining -= 1
    return width


def width_vec_to_per_pos_ek(
    width_vec: List[int], parents_at: List[int], K: int = 8,
) -> List[int]:
    """Convert a Sequoia layer-width vector into a per-position expand_k list
    that the heap-based tree-build can consume.

    expand_k(d) = ceil(width(d) / parents_at(d-1)).
    """
    eks = []
    for d in range(len(width_vec)):
        par = parents_at[d - 1] if d > 0 else 1
        if par == 0:
            eks.append(1)
        else:
            eks.append(min(K, max(1, (width_vec[d] + par - 1) // par)))
    return eks


# ---------------------------------------------------------------------------
# Online positional-acceptance estimator (cross-batch pooled).
# ---------------------------------------------------------------------------

class PosAcceptanceTracker:
    """Maintains an EMA estimate of p_d = P(target argmax = draft argmax at depth d).

    Updates from verify outputs: for each parent node at depth d, observe
    target's argmax and check if it equals draft's rank-0 token at depth d.
    """

    def __init__(self, max_depth: int, alpha: float = 0.95, prior: float = 0.85):
        self.max_depth = max_depth
        self.alpha = alpha  # EMA decay (closer to 1 = slower update)
        self.p_d = np.full(max_depth, prior, dtype=np.float32)
        self.counts = np.zeros(max_depth, dtype=np.float32)

    def update_from_verify(
        self,
        target_argmax_at_node: np.ndarray,    # [B, M] int
        draft_top1_at_depth: np.ndarray,      # [B, max_depth] int — draft's rank-0 token per depth
        node_depth: np.ndarray,               # [B, M] int — depth of each tree node
        node_valid: np.ndarray,               # [B, M] bool
        seq_len: int,
    ) -> int:
        """Returns number of (parent, child-rank) observations added."""
        B, M = node_depth.shape
        added = 0
        for b in range(B):
            for n in range(M):
                if not node_valid[b, n]:
                    continue
                d = int(node_depth[b, n])
                if d >= self.max_depth - 1 or d >= seq_len - 1:
                    continue
                # Parent at depth d predicts what comes at depth d+1.
                target_pred = int(target_argmax_at_node[b, n])
                draft_top1 = int(draft_top1_at_depth[b, d])  # draft's rank-0 at depth d (predicts position d+1)
                # Observation: target's prediction matches draft's argmax at depth d+1?
                # Hmm — node n at depth d has logits predicting d+1's token. Compare to draft_top1[b, d].
                hit = 1.0 if target_pred == draft_top1 else 0.0
                # EMA update.
                self.p_d[d + 1] = self.alpha * self.p_d[d + 1] + (1 - self.alpha) * hit
                self.counts[d + 1] += 1
                added += 1
        return added

    def get_p_d(self) -> np.ndarray:
        return self.p_d.copy()

    def warm(self, min_count: float = 50.0) -> bool:
        return float(self.counts.min()) >= min_count
