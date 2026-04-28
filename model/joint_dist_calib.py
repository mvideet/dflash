"""Bayesian Joint-Distribution Correction (BJC) for tree-build heap scoring.

DDTree's optimality theorem assumes a product distribution
   q_joint(u_1, ..., u_d) = Π q_i(u_i)
which is provably violated by DFlash's bidirectional draft. Empirically the
post-deviation rank-0 fraction collapses from 88% (marginal) to 51% (joint
conditional on prior deviation) — see docs/v8_final_synthesis.md path-trace
study.

v8's PDRR is a hand-tuned table for math500. BJC estimates the same correction
ONLINE, pooled across every batch element and every step, so it adapts to
whatever workload is actually running.

State
-----
We track sparse counts of (rank j observed as target argmax | conditioning):

   counts[depth, rank, dev_d_bucket, dev_r_bucket, ent_bucket]

with separate `total[depth, dev_d_bucket, dev_r_bucket, ent_bucket]` so the
empirical posterior is `(counts + α·draft_q) / (total + α)` (Dirichlet smoothed).

Conditioning is bucketed coarsely so cells fill quickly:
  - dev_d_bucket: {0=no_dev, 1=dev_at_d≤2, 2=dev_at_d>2}
  - dev_r_bucket: {0=no_dev, 1=dev_rank=1, 2=dev_rank=2, 3=dev_rank≥3}
  - ent_bucket:   {0=high_anchor_conf, 1=med, 2=low_anchor_conf}

This gives 3 × 4 × 3 = 36 contexts × max_depth=15 × K=8 = 4320 cells.
At B=32, ~30 obs per step → fills in 5 steps.

API
---
- update(packed_pos, parent_idx, accepted_path, target_logits, draft_topk_idx,
         draft_topk_lp, anchor_ent_per_elem)
    Called after every step's verify. Pools counts across the whole batch.
- score_correction(depth, rank, dev_d, dev_r, ent_bucket, draft_q)
    Returns log-correction to add to heap composite. 0.0 when bucket is below
    `min_count` (cold-start gate).
- log_prob(depth, rank, dev_d, dev_r, ent_bucket, draft_q)
    Returns the SMOOTHED empirical log-probability for the heap (replaces
    raw log_q in the score).

Notes on bucketing
------------------
The "conditioning" of a candidate child being expanded depends on its PARENT's
trajectory, not its own. Specifically:
  - parent_first_dev_depth: the depth at which parent's path first deviated
    from rank-0 (-1 if no deviation yet).
  - parent_first_dev_rank: rank at that deviation (if any), else 0.
The candidate's own (depth, rank) is the (key, value) we're estimating
P(target argmax = candidate's rank | parent's pattern, ...).

Updates after verify only have data on REAL parent nodes that target verified.
We pool ALL such parents (not just accepted-path) — every verified node gives
us one observation of "given this parent's pattern, what was target's argmax?"
"""
from typing import Optional, Tuple

import math
import torch


_NUM_DEV_D = 3   # 0=no_dev, 1=dev_at_depth≤2, 2=dev_at_depth>2
_NUM_DEV_R = 4   # 0=no_dev, 1=dev_rank=1, 2=dev_rank=2, 3=dev_rank≥3
_NUM_ENT = 3     # 0=high_conf (low entropy), 1=med, 2=low_conf
_MISS_RANK = -1  # target argmax not in draft's top-K → record but don't credit any rank


def _bucket_dev_depth(dev_d: int) -> int:
    if dev_d < 0:
        return 0
    if dev_d <= 2:
        return 1
    return 2


def _bucket_dev_rank(dev_r: int) -> int:
    if dev_r <= 0:
        return 0
    if dev_r == 1:
        return 1
    if dev_r == 2:
        return 2
    return 3


def _bucket_entropy(anchor_ent_norm: float) -> int:
    """anchor_ent_norm in [0, 1] (normalized by log(V))."""
    if anchor_ent_norm < 0.10:
        return 0
    if anchor_ent_norm < 0.30:
        return 1
    return 2


class JointCalib:
    """Online Bayesian rank-conditional acceptance posterior.

    Total state size: max_depth × K × 3 × 4 × 3 floats. For max_depth=16, K=8:
    16 × 8 × 36 = 4608 cells. Negligible memory.
    """

    def __init__(
        self,
        max_depth: int,
        K: int,
        alpha_prior: float = 5.0,
        min_count: float = 30.0,
        decay: float = 1.0,                          # 1.0 = no decay; 0.99 = slow forget
        device: torch.device = torch.device("cpu"),
    ):
        self.max_depth = max_depth
        self.K = K
        self.alpha_prior = alpha_prior
        self.min_count = min_count
        self.decay = decay

        # counts[d, j, dev_d_b, dev_r_b, ent_b]
        self.counts = torch.zeros(
            max_depth, K, _NUM_DEV_D, _NUM_DEV_R, _NUM_ENT,
            dtype=torch.float32, device=device,
        )
        self.total = torch.zeros(
            max_depth, _NUM_DEV_D, _NUM_DEV_R, _NUM_ENT,
            dtype=torch.float32, device=device,
        )
        self.miss_count = torch.zeros(
            max_depth, _NUM_DEV_D, _NUM_DEV_R, _NUM_ENT,
            dtype=torch.float32, device=device,
        )

    # ------------------------------------------------------------
    # Update from a verified step
    # ------------------------------------------------------------

    @torch.no_grad()
    def update_from_verify(
        self,
        packed_pos: torch.Tensor,        # [B, M] tree positions (relative depth)
        parent_idx: torch.Tensor,        # [B, M] parent index in tree
        node_valid: torch.Tensor,        # [B, M] bool — real nodes
        target_logits: torch.Tensor,     # [B, M, V] target's logits at each tree node
        draft_topk_idx: torch.Tensor,    # [B, seq_len, K] top-k token IDs per draft pos
        node_token_ids: torch.Tensor,    # [B, M] tree node tokens (for dev pattern check)
        anchor_ent_per_elem: torch.Tensor,  # [B] anchor entropy normalised to [0,1]
    ) -> int:
        """Accumulate counts from one step's verify outputs.

        Each VERIFIED PARENT gives us one observation: at this parent (with its
        dev pattern, ent bucket), we ask "what rank in the draft's top-K of
        depth+1 did target's argmax fall on?" — and increment that bucket.
        """
        if self.decay < 1.0:
            self.counts.mul_(self.decay)
            self.total.mul_(self.decay)
            self.miss_count.mul_(self.decay)

        B, M = packed_pos.shape
        device = self.counts.device

        # Compute target's argmax at each parent (used as the realised next-token).
        target_argmax = target_logits.argmax(dim=-1)        # [B, M]

        # Determine each node's "deviation history". For each tree node we need:
        #   first_dev_depth(node), first_dev_rank(node)  — based on path from root.
        # Walk up parent chain. node_token_ids[b, n] = token at node n.
        # node_token is rank-0 iff token equals draft_topk_idx[b, depth-1, 0].
        depth = packed_pos                                  # [B, M]
        depth_minus1 = (depth - 1).clamp(min=0)
        # draft_topk_idx is [B, seq_len, K]. For position d-1, top-1 is index 0.
        b_arange = torch.arange(B, device=device).unsqueeze(1)
        rank0_token_at_depth = draft_topk_idx[b_arange, depth_minus1, 0]   # [B, M]
        is_argmax = (node_token_ids == rank0_token_at_depth) & (depth > 0)
        is_argmax[:, 0] = True  # anchor (depth=0) considered on argmax chain

        # Compute each node's rank w.r.t. draft's top-K at its depth.
        # rank = j s.t. node_token == draft_topk_idx[b, depth-1, j] ; -1 if none.
        node_rank = torch.full((B, M), -1, dtype=torch.long, device=device)
        for j in range(self.K):
            tok_j = draft_topk_idx[b_arange, depth_minus1, j]      # [B, M]
            match = (node_token_ids == tok_j) & (depth > 0)
            node_rank = torch.where(match & (node_rank < 0),
                                    torch.full_like(node_rank, j),
                                    node_rank)
        node_rank[:, 0] = 0    # anchor has rank 0 by convention

        # Walk parent chain to compute first_dev_depth and first_dev_rank per node.
        # iterate at most max_depth steps.
        first_dev_depth = torch.full((B, M), -1, dtype=torch.long, device=device)
        first_dev_rank = torch.zeros((B, M), dtype=torch.long, device=device)
        # We compute by DFS-style propagation from parent: a node's dev pattern
        # = parent's dev pattern (if already deviated) else this node's own dev.
        # Sort by depth so parents are processed before children.
        sorted_idx = depth.argsort(dim=-1, stable=True)             # [B, M]
        for slot in range(M):
            cur = sorted_idx[:, slot]                               # [B]
            # node info
            cur_b = torch.arange(B, device=device)
            par = parent_idx[cur_b, cur]                            # [B]
            par_dev_d = first_dev_depth[cur_b, par]
            par_dev_r = first_dev_rank[cur_b, par]
            # Determine deviation at THIS node.
            this_dev = (~is_argmax[cur_b, cur])                     # [B] bool
            this_d = depth[cur_b, cur]
            this_r = node_rank[cur_b, cur]
            # Inherit parent's dev pattern; if not yet deviated and this node is dev,
            # set first_dev to this node.
            inherit_d = par_dev_d
            inherit_r = par_dev_r
            new_d = torch.where((par_dev_d < 0) & this_dev, this_d, inherit_d)
            new_r = torch.where((par_dev_d < 0) & this_dev, this_r.clamp(min=0), inherit_r)
            # Don't overwrite root (depth=0) — keep its sentinel.
            is_root = (this_d == 0)
            new_d = torch.where(is_root, torch.full_like(new_d, -1), new_d)
            new_r = torch.where(is_root, torch.zeros_like(new_r), new_r)
            first_dev_depth[cur_b, cur] = new_d
            first_dev_rank[cur_b, cur] = new_r

        # Now for each VALID parent, find what rank target's argmax matches in
        # draft's top-K at depth d+1. But we only have draft_topk_idx for depths
        # 0..seq_len-1, which corresponds to PARENT depths 0..seq_len-2 (so
        # children at depth 1..seq_len-1).
        seq_len = draft_topk_idx.shape[1]
        # parent_d ranges 0..seq_len-1 (d+1 must be ≤ seq_len-1, so parent_d ≤ seq_len-2... actually we have draft logits for positions 0..seq_len-1 — those correspond to children at depths 1..seq_len. Parent at depth d's child is at depth d+1, predicted at draft pos d. So parent_d in [0, seq_len-1].
        valid_parent = node_valid & (depth >= 0) & (depth < seq_len)
        # For each valid parent at depth pd, target's argmax = target_argmax[b, n].
        # We want to find which rank j in draft_topk_idx[b, pd, :] equals target_argmax[b, n].
        # Vectorized: equality across K then argmax.
        pd = depth[valid_parent.nonzero(as_tuple=True)]                       # [P]
        bidx, nidx = valid_parent.nonzero(as_tuple=True)                       # [P], [P]
        if pd.numel() == 0:
            return 0
        ta = target_argmax[bidx, nidx]                                         # [P]
        topk_at_pd = draft_topk_idx[bidx, pd, :]                               # [P, K]
        match = (topk_at_pd == ta.unsqueeze(-1))                               # [P, K]
        any_match = match.any(dim=-1)
        rank_j = match.float().argmax(dim=-1).long()                           # [P] (0 if no match)
        rank_j = torch.where(any_match, rank_j, torch.full_like(rank_j, -1))   # mark misses

        # Conditioning buckets per parent.
        dev_d_b = self._dev_d_bucket(first_dev_depth[bidx, nidx])              # [P]
        dev_r_b = self._dev_r_bucket(first_dev_rank[bidx, nidx])               # [P]
        ent_b = self._ent_bucket(anchor_ent_per_elem[bidx])                    # [P]

        # Increment counts.
        d_idx = (pd + 1).clamp(max=self.max_depth - 1)                         # child's depth
        valid_rank = rank_j >= 0
        # counts[d_idx, rank_j, dev_d_b, dev_r_b, ent_b] += 1 for matched
        if valid_rank.any():
            sel = valid_rank
            self.counts.index_put_(
                (d_idx[sel], rank_j[sel], dev_d_b[sel], dev_r_b[sel], ent_b[sel]),
                torch.ones(int(sel.sum().item()), device=device),
                accumulate=True,
            )
        # total++  for ALL parents (including misses) — denominator
        self.total.index_put_(
            (d_idx, dev_d_b, dev_r_b, ent_b),
            torch.ones(d_idx.numel(), device=device),
            accumulate=True,
        )
        # miss count (target argmax not in top-K): track for completeness
        if (~valid_rank).any():
            sel = ~valid_rank
            self.miss_count.index_put_(
                (d_idx[sel], dev_d_b[sel], dev_r_b[sel], ent_b[sel]),
                torch.ones(int(sel.sum().item()), device=device),
                accumulate=True,
            )

        return int(d_idx.numel())  # how many obs added

    @staticmethod
    def _dev_d_bucket(dev_d: torch.Tensor) -> torch.Tensor:
        # 0=no_dev, 1=dev_at_d≤2, 2=dev_at_d>2.
        b = torch.where(dev_d < 0, torch.zeros_like(dev_d),
                        torch.where(dev_d <= 2, torch.ones_like(dev_d),
                                    torch.full_like(dev_d, 2)))
        return b.long()

    @staticmethod
    def _dev_r_bucket(dev_r: torch.Tensor) -> torch.Tensor:
        # 0=no_dev, 1=r=1, 2=r=2, 3=r≥3.
        b = torch.where(dev_r <= 0, torch.zeros_like(dev_r),
                        torch.where(dev_r == 1, torch.ones_like(dev_r),
                                    torch.where(dev_r == 2, torch.full_like(dev_r, 2),
                                                torch.full_like(dev_r, 3))))
        return b.long()

    @staticmethod
    def _ent_bucket(ent_norm: torch.Tensor) -> torch.Tensor:
        # ent_norm in [0, 1].
        b = torch.where(ent_norm < 0.10, torch.zeros_like(ent_norm),
                        torch.where(ent_norm < 0.30, torch.ones_like(ent_norm),
                                    torch.full_like(ent_norm, 2)))
        return b.long()

    # ------------------------------------------------------------
    # Score correction at heap-build time
    # ------------------------------------------------------------

    def get_log_correction(
        self,
        depth: int,
        rank: int,
        dev_d_bucket: int,
        dev_r_bucket: int,
        ent_bucket: int,
        draft_q: float,
    ) -> float:
        """Returns log( P_emp / draft_q ) — to be ADDED to the heap composite.

        When the bucket has < min_count observations, returns 0.0 (no correction).
        """
        if depth >= self.max_depth or rank >= self.K:
            return 0.0
        n_total = float(self.total[depth, dev_d_bucket, dev_r_bucket, ent_bucket].item())
        if n_total < self.min_count:
            return 0.0
        n_rank = float(self.counts[depth, rank, dev_d_bucket, dev_r_bucket, ent_bucket].item())
        # Bayesian smoothed empirical:
        # P_emp = (n_rank + α·q) / (n_total + α).  Subtract log(q) to keep the
        # original q-component unchanged when n_rank ≈ q · n_total.
        p_emp = (n_rank + self.alpha_prior * draft_q) / (n_total + self.alpha_prior)
        if p_emp <= 0.0 or draft_q <= 0.0:
            return 0.0
        return math.log(p_emp) - math.log(draft_q)

    def num_observations(self) -> float:
        return float(self.total.sum().item())

    def warmup_done(self) -> bool:
        return float(self.total.max().item()) >= self.min_count
