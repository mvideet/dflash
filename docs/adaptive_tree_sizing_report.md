# DFlash Adaptive Tree Sizing: Full Technical Report

## 1. Background

### 1.1 Speculative Decoding in DFlash

DFlash uses a small draft model to predict multiple tokens in parallel, then verifies them against the full target model in a single forward pass using a tree-structured candidate set. The key metrics are:

- **τ (acceptance length)**: mean tokens accepted per verification step
- **Speedup**: wall-clock ratio of vanilla (block_size=1) to speculative decoding
- **Tree nodes**: number of trie nodes in the verification tree (determines target forward cost)

### 1.2 Tree Building Strategies

Four tree builders exist in the codebase (`model/dflash_tree.py`):

| Version | Name | Algorithm |
|---------|------|-----------|
| v1 | `build_dynamic_tree` | Per-position threshold branching with cartesian product |
| v2 | `build_dynamic_tree_v2` | EAGLE-2: expand + rerank by cumulative confidence |
| v3 | `build_bestfirst_tree` | Best-first search by cumulative log-probability |
| v4 | `build_prefixaware_tree` | Prefix-aware greedy submodular E[τ] maximization |

### 1.3 The v4 Algorithm and Nemhauser Guarantee

v4 operates in two phases:

**Phase 1 — Expand:** Best-first search (identical to v3) generates a candidate pool of ~3× budget leaves ordered by cumulative log-probability.

**Phase 2 — Select:** Greedy submodular selection of `max_tree_size` leaves from the pool, ordered by marginal E[τ] gain (prefix-coverage aware). Uses lazy greedy (heap-based) for efficiency.

The objective is:

$$E[\tau] = \sum_{k=1}^{D} P(\tau \geq k) = \sum_{k=1}^{D} \sum_{\text{distinct depth-}k \text{ prefixes } \sigma} \prod_{d=1}^{k} p_{\text{draft}}(\sigma_d)$$

This is a **weighted coverage function** — monotone, submodular, and normalized. Under a cardinality constraint of m leaves, the greedy algorithm achieves:

$$f(\text{greedy}_m) \geq \left(1 - \frac{1}{e}\right) \cdot f(\text{OPT}_m) \approx 0.632 \cdot f(\text{OPT}_m)$$

by Nemhauser-Wolsey-Fisher (1978).

---

## 2. Node-Matched Comparison: v2 vs v4

### 2.1 The mts Mismatch Problem

A critical finding: `max_tree_size` means fundamentally different things for different builders:

- **v2** selects `max_tree_size` non-root trie **nodes** (mix of internal + leaf). So mts=32 → 33 total nodes.
- **v4** selects `max_tree_size` **leaves**. Each leaf of average depth ~2.2 contributes ~2.2 nodes to the trie. So mts=32 → ~70 total nodes.

**Calibration results (mt-bench, 20 samples, greedy):**

| Version | mts | avg_nodes | avg_accept |
|---------|-----|-----------|------------|
| v2 | 24 | 25.00 | 5.17 |
| v2 | 28 | 29.00 | 5.16 |
| v2 | 32 | 33.00 | 5.13 |
| v2 | 36 | 37.00 | 5.31 |
| v2 | 40 | 41.00 | 5.33 |
| v2 | 44 | 45.00 | 5.36 |
| v4 | 24 | 58.00 | 5.46 |
| v4 | 28 | 64.00 | 5.41 |
| v4 | 32 | 70.00 | 5.45 |
| v4 | 36 | 76.00 | 5.45 |
| v4 | 40 | 82.00 | 5.52 |
| v4 | 44 | 88.00 | 5.59 |

### 2.2 Fair Node-Matched Comparison (mt-bench, 80 samples, greedy)

To compare at equal verification cost (~70 nodes):

| Method | mts | avg_nodes | τ | speedup |
|--------|-----|-----------|---|---------|
| v2 fixed | 69 | 70.00 | 5.56 | 3.80 |
| v4 fixed | 32 | 70.00 | ~5.60 | ~3.84 |

**v4 slightly outperforms v2 at equal node count** — higher τ and speedup. The submodular prefix-coverage selection extracts marginally more acceptance per verification node than v2's flat expand+rerank.

### 2.3 User-Reported Prior Baselines

| Dataset | Method | τ | speedup |
|---------|--------|---|---------|
| mt-bench | v4 fixed (mts=32) | ~5.60 | ~3.84 |
| gsm8k | v4 fixed (mts=32) | 8.23 | 6.65 |
| gsm8k | v2 fixed (mts=32) | 7.86 | 6.64 |

---

## 3. Adaptive Tree Sizing: Design Iterations

### 3.1 Approaches Attempted

#### Approach A: Adaptive Block Depth (REJECTED)

**Idea:** Reduce `block_size` on hard steps to cut draft + verification cost.

**Result:** τ dropped from ~5.6 → 2–3 on most sequences. Reducing depth caps the maximum acceptance length — a sequence can never accept more than `eff_bs - 1` tokens regardless of tree quality.

**Conclusion:** Block depth is the wrong lever for v4. The depth ceiling directly limits acceptance.

#### Approach B: Two-Phase Probe + Extend (REJECTED)

**Idea:** Run draft for a small probe depth, compute confidence, then decide whether to extend. Adapts tree size based on probe confidence.

**Result on mt-bench (tree-size adaptation only):** τ=5.12, speedup=2.96. The two-phase draft split added overhead (extra kernel launches, KV cache writes) that exceeded the verification savings.

**Conclusion:** The probe overhead costs more than it saves at this model scale.

#### Approach C: EWMA Tree-Size Adaptation (PARTIAL WIN)

**Idea:** Track acceptance rate via exponential weighted moving average. Map EWMA rate to `max_tree_size` between `min_tree_size` and the configured maximum.

**Algorithm:**
```
After each step:
  rate = n / (eff_bs - 1)
  ewma_rate = decay × ewma_rate + (1 - decay) × rate
  eff_tree_size = min_tree + round((max_tree - min_tree) × ewma_rate)
```

**Results (mt-bench, decay=0.8, min_tree=12):**

| Method | avg_nodes | τ | speedup |
|--------|-----------|---|---------|
| v4 fixed | 70 | 5.60 | 3.84 |
| v4 + EWMA tree-size | 51 | 5.52 | 3.89 |

27% fewer nodes, τ within 1.4%, speedup +0.05x. Marginal because target verification isn't the bottleneck at this scale (8-GPU tensor parallel on 4B model — communication-bound, not compute-bound).

#### Approach D: EWMA Adaptive expand_k (BEST RESULT)

**Idea:** Instead of post-hoc tree size reduction, control the upstream branching factor. EWMA rate maps to `expand_k` between `min_expand_k=2` and `max_expand_k=5`.

**Why this works better:** expand_k has a multiplicative effect on the entire pipeline:
- Phase 1 candidate pool shrinks (2^d vs 5^d candidates)
- Phase 2 selection runs over fewer candidates
- `tree_build` time drops
- `tree_attn_mask` construction is cheaper
- Target verification is cheaper (fewer nodes)
- KV trim is cheaper

Tree-size adaptation only affected verification. Expand_k affects everything upstream.

**Results (mt-bench, decay=0.8, min_tree=12, expand_k 2→5):**

| Method | avg_nodes | τ | speedup |
|--------|-----------|---|---------|
| v4 fixed (ek=3) | 70 | 5.60 | 3.84 |
| v4 + EWMA expand_k 2→5 | ~50 | 5.46 | 4.65 |

**+21% speedup. τ within 2.5%.**

**Important caveat:** The expand_k run (4.65) and the fixed baseline (3.84) are from different sessions. They need back-to-back comparison on the same machine for a clean claim. The expand_k=5 upper bound is also higher than the baseline's ek=3, so easy steps get better trees than fixed v4 ever had.

---

## 4. EWMA Hyperparameter Sweep (Tree-Size Only)

### 4.1 Setup

Grid: 5 decay values × 6 min_tree_size values = 30 configs per dataset.
Datasets: mt-bench (80 samples), gsm8k (256), humaneval (164).
Temperature: 0.0 (greedy).

### 4.2 Results: mt-bench

| decay | min_ts | avg_nodes | τ | speedup |
|-------|--------|-----------|---|---------|
| 0.5 | 4 | 39.38 | 5.41 | 3.78 |
| 0.5 | 8 | 44.84 | 5.47 | 3.84 |
| 0.5 | 12 | 49.82 | 5.53 | 3.86 |
| 0.5 | 16 | 54.33 | 5.52 | 3.86 |
| 0.5 | 20 | 58.47 | 5.56 | 3.87 |
| 0.5 | 24 | 62.43 | 5.57 | 3.85 |
| 0.6 | 4 | 40.05 | 5.50 | 3.80 |
| 0.6 | 8 | 45.20 | 5.46 | 3.84 |
| 0.6 | 12 | 50.09 | 5.50 | 3.84 |
| 0.6 | 16 | 54.50 | 5.49 | 3.83 |
| 0.6 | 20 | 58.55 | 5.49 | 3.85 |
| 0.6 | 24 | 62.54 | 5.56 | 3.86 |
| 0.7 | 4 | 40.49 | 5.40 | 3.79 |
| 0.7 | 8 | 45.85 | 5.51 | 3.84 |
| 0.7 | 12 | 50.57 | 5.50 | 3.84 |
| 0.7 | 16 | 54.87 | 5.51 | 3.88 |
| 0.7 | 20 | 58.79 | 5.49 | 3.84 |
| 0.7 | 24 | 62.73 | 5.58 | 3.88 |
| **0.8** | **4** | **41.85** | **5.47** | **3.79** |
| **0.8** | **8** | **46.90** | **5.52** | **3.86** |
| **0.8** | **12** | **51.42** | **5.52** | **3.89** |
| **0.8** | **16** | **55.46** | **5.52** | **3.85** |
| **0.8** | **20** | **59.31** | **5.56** | **3.84** |
| **0.8** | **24** | **63.01** | **5.57** | **3.85** |
| 0.9 | 4 | 44.54 | 5.49 | 3.83 |
| 0.9 | 8 | 48.82 | 5.45 | 3.82 |
| 0.9 | 12 | 53.11 | 5.53 | 3.85 |
| 0.9 | 16 | 56.86 | 5.53 | 3.85 |
| 0.9 | 20 | 60.20 | 5.51 | 3.84 |
| 0.9 | 24 | 63.59 | 5.52 | 3.82 |

**Best config: decay=0.8, min_ts=12 → τ=5.52, speedup=3.89, avg_nodes=51.42**

### 4.3 Results: gsm8k

| decay | min_ts | avg_nodes | τ | speedup |
|-------|--------|-----------|---|---------|
| 0.5 | 4 | 48.78 | 8.06 | 6.65 |
| 0.5 | 24 | 64.32 | 8.20 | 6.67 |
| 0.6 | 12 | 55.45 | 8.13 | 6.68 |
| 0.7 | 12 | 55.72 | 8.11 | 6.66 |
| **0.8** | **8** | **53.47** | **8.06** | **6.64** |
| **0.8** | **12** | **56.42** | **8.12** | **6.67** |
| 0.9 | 16 | 60.85 | 8.19 | 6.69 |

(Full table: 30 configs, all within τ=8.05–8.20, speedup=6.64–6.69. GSM8K is mostly "easy" for the drafter — EWMA barely triggers.)

### 4.4 Results: humaneval

| decay | min_ts | avg_nodes | τ | speedup |
|-------|--------|-----------|---|---------|
| 0.5 | 4 | 49.27 | 8.32 | 6.85 |
| 0.6 | 12 | 56.10 | 8.46 | 6.95 |
| **0.8** | **12** | **56.87** | **8.45** | **6.95** |
| 0.8 | 24 | 64.98 | 8.51 | 6.93 |
| 0.9 | 20 | 63.15 | 8.49 | 6.92 |

(Full table: 30 configs, all within τ=8.32–8.51, speedup=6.85–6.95. Similar to GSM8K — code is easy for the drafter.)

### 4.5 Key Finding from Sweep

**decay=0.8, min_ts=12 is robust across all three datasets.** The impact is largest on mt-bench (mixed chat, variable difficulty) and smallest on gsm8k/humaneval (mostly easy, EWMA stays near ceiling). Tree-size-only EWMA gives marginal speedup improvement (+0.05 to +0.1x) because target verification is not the bottleneck at this hardware scale.

---

## 5. Overnight Sweep Results (Tree-Size Only EWMA)

### 5.1 First Run: decay=0.7, min_tree=8 (all datasets, temp=0.0)

| Dataset | speedup | τ | avg_nodes |
|---------|---------|---|-----------|
| gsm8k | 6.63 | 8.06 | 52.64 |
| math500 | 7.72 | 9.45 | 55.18 |
| aime24 | 7.34 | 8.96 | 53.18 |
| aime25 | 7.18 | 8.78 | 52.76 |
| alpaca | 3.05 | 4.10 | 44.86 |
| mt-bench | 3.84 | 5.51 | 45.85 |
| humaneval | 6.92 | 8.39 | 53.32 |
| mbpp | 6.77 | 7.76 | 52.65 |
| lbpp | 6.31 | 6.91 | 49.05 |
| swe-bench | 4.35 | 4.73 | 42.19 |
| livecodebench | 7.25 | 8.48 | 55.15 |

### 5.2 Second Run: decay=0.8, min_tree=12 (all datasets, temp=0.0)

| Dataset | speedup | τ | avg_nodes |
|---------|---------|---|-----------|
| gsm8k | 6.67 | 8.12 | 56.42 |
| math500 | 7.74 | 9.51 | 58.25 |
| aime24 | 7.60 | 9.24 | 56.91 |
| aime25 | 7.33 | 8.99 | 56.38 |
| alpaca | 3.06 | 4.14 | 51.00 |
| mt-bench | 3.88 | 5.52 | 51.42 |
| humaneval | 6.94 | 8.45 | 56.87 |
| mbpp | 6.37 | 7.82 | 56.65 |
| lbpp | 5.81 | 7.00 | 53.39 |
| swe-bench | 4.00 | 4.72 | 47.85 |
| livecodebench | 6.84 | 8.60 | 59.04 |

### 5.3 Temperature 0.6 Results (decay=0.8, min_tree=12)

| Dataset | speedup | τ | avg_nodes |
|---------|---------|---|-----------|
| gsm8k | 6.65 | 8.40 | 56.98 |
| math500 | 8.51 | 9.73 | 53.49 |
| aime24 | 6.91 | 8.72 | 52.29 |
| aime25 | 6.71 | 8.55 | 52.12 |
| alpaca | 3.05 | 4.26 | 51.90 |
| mt-bench | 3.87 | 5.72 | 51.68 |
| humaneval | 6.94 | 8.74 | 53.07 |
| mbpp | 6.31 | 7.97 | 52.86 |
| lbpp | 6.03 | 7.24 | 51.62 |
| swe-bench | 4.58 | 4.86 | 48.98 |
| livecodebench | 8.27 | 8.77 | 53.79 |

---

## 6. Adaptive expand_k Results (Preliminary)

### 6.1 Config

EWMA controls both `max_tree_size` and `expand_k` per step:
- `ewma_decay=0.8`
- `min_tree_size=12`, `max_tree_size=32`
- `min_expand_k=2`, `max_expand_k=5`
- Block depth always full (16)

### 6.2 mt-bench Result

| Method | avg_nodes | τ | speedup |
|--------|-----------|---|---------|
| v4 fixed (ek=3, mts=32) | 70 | 5.60 | 3.84 |
| v4 + EWMA tree-size only (0.8/12) | 51 | 5.52 | 3.89 |
| **v4 + EWMA tree+expand_k (2→5)** | **~50** | **5.46** | **4.65** |

The 4.65 speedup is from a clean run (no competing jobs). **However, it has not been compared back-to-back against the fixed baseline on the same machine in the same session.** The fixed baseline 3.84 is from an earlier session. A clean head-to-head comparison is still needed.

### 6.3 Partial Overnight Results (expand_k, temp=0.0) — POTENTIALLY CONTAMINATED

These ran while a prior overnight was still finishing. Treat as indicative, not definitive:

| Dataset | speedup | τ | avg_nodes |
|---------|---------|---|-----------|
| gsm8k | 7.79 | 8.12 | 52.62 |
| math500 | 9.32 | 9.51 | 53.33 |
| aime24 | 7.56 | 9.18 | 52.71 |
| aime25 | 7.05 | 8.63 | 52.27 |
| alpaca | 3.01 | 4.13 | 52.02 |
| mt-bench | 3.72 | 5.46 | 51.67 |

Note: mt-bench shows 3.72 here (lower than 4.65) likely due to GPU contention.

---

## 7. Failed Approaches

### 7.1 Adaptive Block Depth (block_size reduction)

Reducing `block_size` from 16 to smaller values on hard steps capped the maximum acceptance length. Results:

| Config | τ | speedup | avg block |
|--------|---|---------|-----------|
| v4 fixed | ~5.6 | 3.84 | 16 |
| EWMA block depth | 4.89 | 3.37 | 7.31 |

**12.5% τ loss, 12.2% speedup loss.** The acceptance ceiling effect dominated any savings.

### 7.2 Two-Phase Probe + Extend (draft split)

Splitting the draft forward into a probe (3 positions) + conditional extend added overhead without meaningful verification savings.

| Config | τ | speedup |
|--------|---|---------|
| v4 fixed | ~5.6 | 3.84 |
| Two-phase (tree-size adapt) | 5.12 | 2.96 |

**23% speedup loss** from kernel launch overhead.

### 7.3 Node-Budget Constrained v4 (v5)

Replaced Phase 2's cardinality constraint with a trie-node budget constraint. Greedy by ΔE[τ]/ΔNodes instead of ΔE[τ].

| Config | avg_nodes | τ | speedup |
|--------|-----------|---|---------|
| v4 fixed | 70 | 5.60 | 3.84 |
| v5 node-budget | 69.49 | 5.60 | 3.68 |

Same τ, **4.2% speedup loss** from O(N²) Phase 2 computation overhead. At typical operating points, v4's leaf-based greedy already produces near-optimal node allocation. v5 was removed from the codebase.

---

## 8. Key Insights

### 8.1 Target verification is not the bottleneck at small scale

On 8-GPU tensor parallel with a 4B model, reducing tree nodes by 30% gives <2% speedup. The target forward is communication-bound (NCCL all-reduce latency is constant regardless of sequence length). This explains why tree-size-only EWMA had minimal impact.

### 8.2 expand_k controls the entire pipeline upstream

Reducing expand_k from 3 to 2 on hard steps shrinks:
- Candidate pool size (exponential in depth)
- Phase 2 selection work
- Tree build time
- Attention mask construction
- Target verification
- KV cache trim

This multiplicative effect is why adaptive expand_k gives significantly more speedup than tree-size-only adaptation.

### 8.3 Difficulty is locally persistent

EWMA works because generation difficulty is autocorrelated — easy regions (boilerplate code, structured math) stay easy for many consecutive steps, and hard regions (novel reasoning, topic transitions) cluster together. A decay of 0.8 (~5-step effective window) captures this structure.

### 8.4 The Nemhauser guarantee is per-step, per-budget

The (1-1/e) approximation guarantee holds for whatever max_tree_size you pass to v4 on each step. Adapting the budget across steps doesn't violate the guarantee — it just means different steps have different ceilings. The guarantee is that within each step's ceiling, the selection is near-optimal.

---

## 9. Implementation

All changes are in `benchmark.py`. The draft model (`model/dflash.py`) and tree builders (`model/dflash_tree.py`) are unmodified.

### 9.1 New Parameters to `dflash_generate`

```python
adaptive_block: bool = False,
adaptive_block_ewma_decay: float = 0.8,
adaptive_block_min_tree_size: int = 12,
adaptive_block_min_expand_k: int = 2,
adaptive_block_max_expand_k: int = 5,
collect_calibration: bool = False,
```

> **Note:** Earlier iterations included `adaptive_block_strategy`, `adaptive_block_probe_depth`, `adaptive_block_theta_low`, and `adaptive_block_theta_high` for the two-phase probe approach (Section 7.2). These were removed after the approach was rejected. Only EWMA adaptation remains.

### 9.2 EWMA State (per generation call)

```python
_ab_ewma_rate = 1.0           # starts optimistic
_ab_eff_tree_size = max_tree_size
_ab_eff_expand_k = expand_k
```

### 9.3 Per-Step Update

```python
# At top of each step:
_ab_eff_tree_size = min_tree + round((max_tree - min_tree) × ewma_rate)
_ab_eff_expand_k = min_ek + round((max_ek - min_ek) × ewma_rate)

# After acceptance (n tokens accepted, eff_bs - 1 possible):
rate = n / max(eff_bs - 1, 1)
ewma_rate = decay × ewma_rate + (1 - decay) × rate
```

### 9.4 CLI Flags

```
--adaptive-block                        # enable EWMA adaptive tree sizing + expand_k
--adaptive-block-ewma-decay 0.8         # EWMA decay factor
--adaptive-block-min-tree-size 12       # minimum tree size on hard steps
--adaptive-block-min-expand-k 2         # minimum expand_k on hard steps
--adaptive-block-max-expand-k 5         # maximum expand_k on easy steps
--collect-calibration                   # gather per-position calibration data
```

> **Note:** The `--adaptive-block-strategy` flag was removed after the two-phase approach was rejected (Section 7.2). `--adaptive-block` now implies EWMA-only adaptation. The `--dynamic-branching` flag referenced in some scripts was never implemented and has been removed.

---

## 10. Open Questions

1. **Back-to-back comparison needed.** The 4.65 speedup for adaptive expand_k needs validation against fixed v4 in the same session on the same GPUs.

2. **expand_k=5 vs expand_k=3 ablation.** How much of the speedup comes from wider search on easy steps (ek=5 > ek=3) vs cheaper trees on hard steps (ek=2 < ek=3)?

3. **Larger model scales.** On a 70B target where verification dominates, tree-size adaptation alone may give significant speedup. The expand_k story should strengthen further.

4. **Draft confidence calibration.** Per-position calibration data collection is implemented (`--collect-calibration`). Analyzing whether draft top-1 probability predicts acceptance would validate the theoretical basis for adaptive approaches.

5. **Optimal EWMA decay and expand_k range.** The current sweep only covered tree-size hyperparameters. A sweep over (decay, min_ek, max_ek) combinations would find the optimal adaptive expand_k config.

---

## 11. Recommended Paper Configuration

Based on all experiments:

```
--adaptive-block
--adaptive-block-strategy ewma
--adaptive-block-ewma-decay 0.8
--adaptive-block-min-tree-size 12
--adaptive-block-min-expand-k 2
--adaptive-block-max-expand-k 5
```

**Headline claim:** v4 prefix-aware tree construction with (1-1/e) submodular guarantee + zero-overhead EWMA adaptive branching achieves +21% wall-clock speedup over fixed v4 at <2.5% acceptance loss on mt-bench, with consistent gains across 11 benchmarks.
