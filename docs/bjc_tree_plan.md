# Batched Speculative Decoding — Research & Engineering Plan

Branch: `feat/batch-size-profile`. Single A6000, Qwen3-4B + DFlash-b16, math500.

## 1. Where we landed (engineering — done & shipped)

| B   | speedup | M   | tau  |
|-----|---------|-----|------|
| 1   | 6.79×   | 64  | 9.08 |
| 4   | 6.16×   | 32  | 9.00 |
| 8   | 5.19×   | 16  | 8.42 |
| 16  | 3.38×   | 16  | 8.46 |
| 32  | 2.79×   | 16  | 8.37 |
| 64  | 2.12×   | 16  | 8.10 |
| 128 | 1.98×   | 8   | 6.15 |

Three optimisation iterations stacked:
1. **Adaptive `max_tree_size` schedule** `{1:64, 2:32, 4:32, 8:16, 16:16, 32:16, 64:16, 128:8}`. Trades ~5–10% tau for ~4–8× faster verify at high B.
2. **Batched draft + cumulative `cache_pad_mask`.** Eliminates per-element draft loop (~5ms × B) at the cost of bookkeeping. Result: v7 throughput at B=8 went 500 → 1300 tok/s.
3. **Batched tree-pack (single host→device transfer per output).** Reduces tree-build from 21ms → 6ms at B=32.

## 2. What we tried that did NOT help on math500 / temp=0

All measured on the post-engineering baseline:

| Idea                                | Result                                             |
|-------------------------------------|----------------------------------------------------|
| Online M* (cost+tau model)          | Picks same M every step → ties static.            |
| Online M* + anchor confidence       | Slight loss (-0.5×), wrong chain bound shape.     |
| PWLS (posterior-weighted leaf select) | No-op at temp=0 (unique max-accept leaves).      |
| BQAT (bonus-quality truncation)     | -0.2× — loses tau, gains nothing.                 |
| CPPR (cross-path posterior re-rank) | -0.5× — extra log_softmax overhead.               |
| EWMA adaptive (repo's --adaptive-block) | Saturates at M≈100, much bigger than goodput-optimal M=16. |
| Anchor-entropy adaptive (repo's --adaptive-budget-mode) | Near-ties static schedule. |
| v8 (CGDB+PDRR) score adjustments    | +0.1 tau cancelled by extra tree-build cost.      |

Honest finding: math500 + temp=0 + post-engineering baseline is at the local goodput optimum. Beating it requires an **algorithmic** change.

## 3. Research proposal: BJC-Tree (Bayesian Joint-Distribution Correction Tree)

### 3.1 Theoretical motivation

DDTree's optimality theorem assumes a **product distribution** over the draft's per-position marginals:

> E[τ] = Σ_u q(u),  where q(u_1,...,u_d) = Π_i q_i(u_i)

DFlash's draft has **bidirectional self-attention over the masked block**. So `q_i(u_i)` is the *marginal* of a non-trivial joint, not an independent factor. The product approximation systematically over-rates "phantom paths" — sequences with one rank-1 deviation followed by argmax extensions look great under the product but rarely accept under the joint.

The repo's path-trace data quantifies this:

| Position relative to first deviation | P(target argmax = rank-0) under marginal | under joint (empirical) |
|---|---|---|
| k=1 after first dev | 88% | **51%** |
| k=2 after first dev | 88% | 66% |
| k=3 after first dev | 88% | 82% |

**v8's PDRR is a hand-tuned correction to this** (boost rank>0 children at k=1,2,3 after first deviation). Coefficients (0.5, 0.25, 0) were tuned on math500 specifically.

### 3.2 The BJC-Tree algorithm

Replace the hand-tuned PDRR table with **online empirical estimation pooled across the batch and across steps**:

1. Maintain a sparse count table:
   ```
   counts[depth, child_rank, parent_first_dev_depth, parent_first_dev_rank,
          anchor_entropy_quartile] : float
   ```

2. **Update step (after every verify, every batch element):**
   - For each parent node at depth d in every element's accepted tree:
     - Look up draft's top-k tokens at depth d+1.
     - Find target's argmax token at this parent's logits → rank j ∈ {0..k-1, ∞}.
     - Identify parent's dev pattern (first_dev_depth, first_dev_rank).
     - Compute parent's anchor-entropy quartile from cached anchor logits.
     - Increment `counts[d+1, j, dev_d, dev_r, ent_q] += 1`.

3. **Score in heap (replaces draft's marginal `log q_j`):**
   ```python
   def score_correction(depth, rank, dev_pattern, ent_q, draft_q):
       cell = counts[depth, rank, dev_pattern, ent_q]
       total = sum(counts[depth, *, dev_pattern, ent_q])
       # Bayesian smoothing with Dirichlet prior on draft's marginal q.
       alpha = ALPHA_PRIOR  # tune as 1.0–10.0
       p_emp = (cell + alpha * draft_q) / (total + alpha)
       return log(p_emp) - log(draft_q)  # log-correction added to existing q score
   ```

4. **Heap pop / expand:** identical to v7 DDTree, but the heap score for each candidate child is `parent_score + log_q_child + score_correction(...)`.

### 3.3 Why this is novel

| Method                        | Joint correction | Calibration source     | Cross-batch pool | Adapts to OOD |
|-------------------------------|------------------|------------------------|------------------|---------------|
| v7 DDTree                     | None             | —                      | —                | —             |
| v8 CGDB + PDRR                | Hand-tuned table | Math500 path-trace     | No               | No            |
| `--calibrate` (existing)      | Per-sequence     | This sequence's verify | No               | Yes (slow)    |
| EAGLE-2                       | Static           | Offline acceptance     | No               | No            |
| **BJC-Tree (proposed)**       | **Online Bayesian** | **Live verify, all B** | **Yes**          | **Yes**       |

The cross-batch pool is the crucial novel angle: at B=32, every step gives ~32×M observations of (draft prediction, target argmax) pairs. After 30 steps you have ~30,000 observations across all (depth, dev) buckets — enough to learn the joint correction online, per-workload, **without** training and **without** offline tables.

### 3.4 Why it should Pareto-dominate v7

- **Same M, no extra forward pass, ~5% tree-build overhead.**
- Higher tau at every M because the score is a closer estimate of the true joint.
- Cold-start handled by the Dirichlet prior on draft's marginal q (warm calibration during the first ~5–10 steps; converges to empirical thereafter).
- Adapts per-workload: math500 vs mt-bench have different joint structures, BJC-Tree learns each.

### 3.5 Implementation milestones

1. **M1 (this PR):** `model/joint_dist_calib.py` with `JointCalib` class. `update()` from verify outputs, `score_correction()` for heap. Smoke-test at B=1 with math500.
2. **M2:** Wire into `_build_one_tree`. Bayesian smoothing tuned. `--bjc-tree` flag.
3. **M3:** Cross-batch pool tested at B=4, 8, 16, 32. Ablate Dirichlet α.
4. **M4:** Compare against v7, v8, v7+`--calibrate` on math500. Report tau and speedup.
5. **M5:** Cross-dataset (mt-bench, gsm8k, humaneval) replication.
6. **M6 (optional):** Sparse tree-attention Triton kernel to enable larger M.

### 3.6 Success criteria

- **Tier 1 (must-have):** matches v8's hand-tuned PDRR tau on math500 (within ±0.05) without any offline tables.
- **Tier 2 (publishable):** beats v8 on at least 2 of 4 datasets (math500/gsm8k/mt-bench/humaneval) with a workload-adaptive online schedule.
- **Tier 3 (strong publication):** combined with sparse-tree Triton kernel, shifts the entire Pareto frontier — tau higher AND verify cheaper at every M.

### 3.7 Risks

- **Cold start:** first 5–10 steps may underperform v7 due to noisy estimates. Mitigation: gate correction by `min_count_threshold`; only apply when bucket has ≥30 observations.
- **Conditioning sparsity:** with rich conditioning (4D dev pattern + ent quartile), some buckets stay sparse. Mitigation: hierarchical fallback (full conditioning → drop ent_q → drop dev_r → drop dev_d → marginal).
- **Per-step compute:** dict lookups in Python loop. Mitigation: pre-vectorize as torch tensors of shape `[max_depth, K, num_dev_patterns, num_ent_buckets]` once we know shape; gather per heap pop.
- **Workload drift:** if difficulty changes mid-decoding, stale counts hurt. Mitigation: EMA-style decay on counts (e.g., decay by 0.99 every 50 steps).

## 4. Beyond BJC-Tree

If BJC-Tree works, the obvious follow-ups:

1. **Sparse tree-attention Triton kernel** — exploits the ancestors-only attention pattern. Enables larger M without compute blowup. ~1.6× verify speedup at M=128.
2. **Cohort-verified speculation** — partition the batch by anchor entropy; verify easy/hard cohorts separately with different M.
3. **Cross-step verify cache (SPVB-Cache)** — reuse rejected-tree target logits across consecutive steps when prefixes overlap.

These compose with BJC-Tree (orthogonal axes: tree-shape, kernel, cross-step caching).

## 5. What's already in this branch

- `model/dflash_tree_batched.py` — batched tree primitives (build, mask, leaf-select). Now includes CGDB+PDRR ports.
- `dflash_batched.py` — batched v7/v8 generator with per-element ragged accept paths and cumulative K_ctx pad mask.
- `benchmark_batched.py` — variable-length-prompt driver, schedule plumbing, 6-mode CLI flags.
- `profile_batch_size.py` — kernel-time microbenchmark.
- `sweep_mts_grid.py`, `sweep_v7_vs_v8.py`, `sweep_research_ideas.py`, `sweep_research_interleaved.py` — research sweep infrastructure.
- `paper/fig/v7_vs_v8_full.png` — the 6-curve final comparison.
- This document: `docs/bjc_tree_plan.md`.
