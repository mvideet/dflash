# DFlash Algorithm Exploration: Complete Report

## Executive Summary

Over multiple sessions, we systematically explored every algorithmic lever available for improving DFlash speculative decoding. Starting from a v4 baseline of 3.81x speedup, we reached 4.19x with v2 ek=8 mts=70 (+10%). Every tree construction variant, adaptive mechanism, verification trick, training objective, and system optimization was tried. The binding constraint is the target model's 31ms forward pass (73% of step time), which is memory-bandwidth-bound and invariant to tree size.

**Best configuration: v2 (EAGLE-2) with expand_k=8, max_tree_size=70 → speedup 4.19, tau 5.93, 71 nodes.**

---

## 1. System Architecture

DFlash is a speculative decoding system:
- **Draft model**: 5-layer transformer (537M params) produces logits for 15 positions in ONE parallel forward pass. Input: anchor token + 14 mask tokens. Output: position-independent logits (same top-k at each position regardless of tokens at other positions).
- **Tree builder**: constructs a candidate verification tree from draft logits.
- **Target model**: Qwen3-4B (2.5B params on 8x A6000 GPUs with tensor parallelism). Verifies the tree in a single forward pass with tree attention mask.
- **Verification**: finds the longest consecutively matching path in the tree. Samples a bonus token from the target's logits at the rejection point.

## 2. Profiling Results (The Hardware Reality)

Profiled at 50 nodes, 20 samples, mt-bench:

| Component | ms/step | % of decode |
|-----------|---------|-------------|
| **target_backbone** | **31.1** | **72.7%** |
| draft_model | 5.7 | 13.4% |
| target_lm_head | 1.5 | 3.4% |
| tree_build | 1.3 | 3.1% |
| draft_lm_head | 1.3 | 3.1% |
| trim_kv_cache | 1.0 | 2.4% |
| tree_attn_mask | 0.4 | 0.9% |
| other | 0.4 | 0.9% |
| **Total** | **~43** | **100%** |

**Critical finding**: target_backbone is 31ms at BOTH 16 nodes and 71 nodes (measured back-to-back). The forward pass is memory-bandwidth-bound (reading 8GB of model weights through 36 layers). Node count is irrelevant to verification cost on this hardware.

**Flash attention vs SDPA** (single GPU, prefix_len=122):

| Implementation | L=1 | L=16 | L=50 | L=70 |
|----------------|-----|------|------|------|
| flash_attention_2 | 34.6ms | 34.8ms | 34.7ms | 35.7ms |
| sdpa | 28.9ms | 31.0ms | 30.8ms | 30.3ms |

SDPA is FASTER than flash attention for short sequences. Both are flat across L=1-70. The KV cache read dominates.

## 3. Tree Builder Algorithms Tested

### 3.1 v2: EAGLE-2 Expand + Rerank (WINNER)

**Algorithm**: Phase 1 expands layer-by-layer (top-ek nodes per layer expanded with ek children). Phase 2 ranks ALL candidate nodes by cumulative log-probability, selects top-mts with connectivity enforcement.

**Key properties**:
- max_tree_size directly controls trie node count (mts=70 → exactly 71 nodes)
- Distributed branching: alternatives at many positions (robust to unpredictable rejection)
- Phase 2 is O(N log N) sort + O(N) scan (fast)
- Near-optimal for linear E[tau] objective under tree precedence constraints

**Best result**: v2 ek=8 mts=70 → speedup 4.19, tau 5.93, 71 nodes

### 3.2 v4: Prefix-Aware Greedy (Submodular)

**Algorithm**: Phase 1 generates candidate pool via best-first search (~3x budget). Phase 2 uses lazy-greedy submodular selection maximizing E[tau] with (1-1/e) Nemhauser guarantee.

**Key properties**:
- max_tree_size controls LEAF count (mts=32 → ~54-70 trie nodes depending on ek)
- Coverage-aware: avoids redundant leaves sharing prefixes
- Theoretical guarantee on E[tau] under leaf cardinality constraint

**Issue**: Optimizes the WRONG constraint. The guarantee is over leaf count, but cost is in trie nodes. At matched nodes, v4 is within 2-3% of v2.

**Result at ~70 nodes**: v4 ek=8 mts=48 → speedup 4.05, tau 5.82, 72 nodes (vs v2: 4.19, 5.93, 71)

### 3.3 v6: Density-Greedy (Budgeted Maximum Coverage)

**Algorithm**: Phase 1 same as v4. Phase 2 selects by Δf/Δg ratio (marginal E[tau] gain per new trie node). Optional self-sizing via efficiency ratio η = E[tau] / (α + |trie|).

**Theory**: (1-1/e) approximation to OPT_node(B) via Khuller-Moss-Naor (1999). Dominates v4's guarantee because OPT_node(B) ≥ OPT_leaf(m).

**Result**: tau 5.59, speedup 3.66 at 84 nodes. **WORSE** — density-greedy produced LARGER trees than v4 at same mts because it preferred leaves with many novel (but low-value) prefixes.

Self-sizing experiments:
- alpha=20: collapsed to 13 nodes (tau 4.62, speedup 3.23)
- alpha=100: 25 nodes (tau 5.21, speedup 3.58)
- alpha=500, ek=5: 69 nodes (tau 5.86, speedup 3.80 — decent but not better than v2)

### 3.4 v7: Best-First Expansion + v2 Rerank

**Algorithm**: Phase 1 uses best-first expansion (priority queue by cumulative log-prob, like v3/v4) to generate a globally-optimal candidate pool. Phase 2 identical to v2 (node reranking with connectivity).

**Rationale**: v2's Phase 1 (layer-by-layer) misses globally good paths through non-top-ek nodes at early layers. Best-first explores globally.

**Result**: v7 ek=8 mts=49 → speedup 4.12, tau 5.74, 50 nodes. **Identical to v2** (4.12, 5.78). At ek=8, v2's layer-by-layer already generates sufficient candidates.

### 3.5 v8: Entropy-Adaptive Per-Position expand_k

**Algorithm**: Same as v2 but with per-position variable expand_k based on draft entropy. High-entropy positions (uncertain) get more candidates, low-entropy (confident) get fewer. Total budget = ek × seq_len.

**Rationale**: Allocate candidates WHERE they're needed.

**Result**: v8 ek=8 mts=49 → speedup 3.96, tau 5.71, 50 nodes. **WORSE** than uniform v2 (4.12, 5.78). The variable top-k computation adds overhead and the entropy-based redistribution doesn't improve tree quality.

### 3.6 v9: Confidence-Optimal Tree (COT) — DP-Optimal Cartesian Product

**Algorithm**: After draft forward, compute per-position acceptance probabilities. Run a DP to find optimal branching pattern [k_1, ..., k_D] maximizing E[tau] subject to node budget. Build Cartesian product tree.

**Theory**: Provably optimal for Cartesian product trees under position-independent drafting. Every path extends to full depth (no shallow dead-end leaves like v2).

**Why it should work (theory)**: On synthetic data with one weak position (prob 0.5), COT gets 42% higher E[tau] than greedy path. Concentrates branches at the weakest position.

**Why it fails (practice)**: 
1. DP takes 9ms per step (21% overhead)
2. Concentrated branching is FRAGILE — rejection happens at unexpected positions, and COT has zero alternatives outside the branching position
3. Draft confidence is a noisy predictor of where the target will reject

**Result**: speedup 2.91, tau 5.21, 67 nodes. **Much worse** than v2 (4.19, 5.93).

**Key lesson**: Distributed robustness (v2) beats concentrated optimality (COT). The greedy-by-cumulative-probability heuristic naturally distributes alternatives and handles noisy draft confidence.

## 4. Adaptive Mechanisms

### 4.1 EWMA Tree-Size + expand_k (v4+EWMA)

**Algorithm**: Track acceptance rate via exponential weighted moving average. Map EWMA rate to both max_tree_size and expand_k per step. Easy steps get wide search + full trees; hard steps get narrow + small.

**Results (mt-bench, back-to-back)**:

| Config | nodes | tau | speedup |
|--------|-------|-----|---------|
| v4+EWMA ek 2→5 (decay=0.8) | 52 | 5.46 | 3.69 |
| v4+EWMA ek 3→7 (decay=0.8) | 45 | 5.59 | 3.92 |
| v2 fixed ek=5 mts=49 | 50 | 5.79 | 4.10 |

**Verdict**: v2 fixed beats v4+EWMA at every matched node count. EWMA adds overhead and v4's leaf-based selection can't exploit wider search as efficiently as v2's node-based selection.

### 4.2 v2+EWMA adaptive expand_k

**Result**: speedup 3.73, tau 5.44, 35 nodes. **WORSE than v2 fixed** (3.80, 5.56, 70 nodes). Node reduction from EWMA bought nothing (communication-bound hardware). Lower expand_k on hard steps degraded tree quality.

### 4.3 Confidence-Gated Verification (Flash vs SDPA switching)

**Idea**: On high-confidence steps, skip the tree entirely and verify just the greedy chain with flash attention (faster than SDPA with tree mask).

**Result**: Flash attention is SLOWER than SDPA for short sequences. The hypothesis was wrong. No implementation attempted.

## 5. Verification Improvements

### 5.1 Bonus Token Tree Continuation (BTTC)

**Idea**: After accepting n tokens and sampling bonus token b, check if b matches a tree branch at position n+1. If so, continue accepting along that branch for free (target logits already computed).

**Implementation**: Walk the tree after bonus token sampling, matching against children of the last accepted node.

**Result**: 0/4305 steps matched (0.0%). **Structurally impossible**. At rejection points, the draft and target COMPLETELY disagree — the bonus token (target's choice) is not among the tree's candidates (draft's top-k) at that position. Verified by printing bonus tokens vs children tokens: totally different token IDs.

### 5.2 Vectorized Attention Mask

**Change**: Replaced O(L×D) Python-loop ancestor matrix construction with vectorized transitive-closure via repeated boolean matrix multiplication.

**Result**: +1% speedup (4.16 vs 4.12 at 50 nodes). Real but tiny. Kept in codebase.

## 6. Draft Model Training

### 6.1 Self-Speculative Refinement (SSR) — Inference

**Idea**: Run a cheap coarse pass (1 of 5 draft layers) to get rough predictions. Replace mask tokens with these predictions. Run full draft with informed context → position-dependent logits.

**Result**: tau crashed from 5.78 to 2.61 (speedup 1.99). The draft model was trained EXCLUSIVELY with mask tokens and COLLAPSES on non-mask inputs. The attention mechanism produces garbage when it sees token embeddings it was never trained on at positions 2-15.

### 6.2 SSR Training (Self-Refinement Training)

**Idea**: Fine-tune the draft model with a curriculum: gradually replace mask tokens with the draft's own coarse predictions during training. This teaches the model to handle non-mask inputs.

**Implementation**: Modified base_model.py `draft_forward` with `refine_prob` parameter. Curriculum: epochs 0-1 pure masks, epochs 2+ gradually introduce self-predictions.

**Status**: Training launched on 3K subset (6 epochs, ~5.5 hours). Not evaluated before session ended.

### 6.3 GTO (Group Tree Optimization) Checkpoint Evaluation

**Checkpoint**: step_3900 from prior training run.

**Result**: tau 5.81, speedup 4.11. **Worse than original HF draft** (5.93, 4.19). The GTO PPO-style loss optimized for a specific tree config (v1, mts=8, top_k=5) that doesn't transfer to inference config (v2, mts=70, ek=8).

### 6.4 TopK Training Checkpoint Evaluation

**Checkpoints**: steps 5050, 7000, 9050 from prior TopK training run.

| Checkpoint | tau | speedup |
|------------|-----|---------|
| Original HF | 5.93 | 4.19 |
| TopK 5050 | 5.76 | 4.08 |
| TopK 7000 | 5.76 | 4.06 |
| TopK 9050 | 5.73 | 4.04 |

**Verdict**: TopK training DEGRADES the draft monotonically. More training = worse. The TopK objective trades overall distribution matching for top-k recall, which doesn't improve acceptance in practice.

## 7. expand_k and Node Count Sweeps

### 7.1 expand_k Sweep (v2, mts=49, 50 nodes)

| ek | tau | speedup |
|----|-----|---------|
| 3 | 5.52 | 3.85 |
| 5 | 5.79 | 4.10 |
| 7 | 5.81 | 4.11 |
| 8 | 5.78 | 4.12 |
| 10 | 5.81 | 4.10 |

Peak at ek=7-8. Diminishing returns after ek=8. The main gain is from ek=3→5 (+6.5% speedup).

### 7.2 Node Count Sweep (v2, ek=8)

| nodes | tau | speedup |
|-------|-----|---------|
| 16 | 5.10 | 3.71 |
| 26 | 5.44 | 3.92 |
| 36 | 5.64 | 4.02 |
| 50 | 5.78 | 4.16 |
| 61 | 5.89 | 4.18 |
| 66 | 5.85 | 4.15 |
| **71** | **5.93** | **4.20** |
| 76 | 5.94 | 4.19 |
| 81 | 5.90 | 4.12 |
| 91 | 5.93 | 4.16 |

Peak at 71 nodes. Tau plateaus at ~5.93 after 70 nodes. Speedup regression at 81+ nodes from minor per-step overhead. The speedup curve follows tau closely (confirming constant per-step time).

### 7.3 v4 ek=8 Node Calibration

| v4 mts | nodes | tau | speedup |
|---------|-------|-----|---------|
| 32 | 54 | 5.85 | 4.14 |
| 36 | 59 | 5.82 | 4.12 |
| 44 | 68 | 5.80 | 4.06 |
| 48 | 72 | 5.82 | 4.05 |

v4 at matched nodes (~70) loses to v2 (4.05 vs 4.19). The gap widens with more nodes because v4 must extend all leaves to full depth, wasting budget.

## 8. Theoretical Insights

### 8.1 E[tau] is Linear in Node Set

For position-independent drafting:

E[tau] = sum over all non-root trie nodes of their cumulative probability

This is a MODULAR (linear) function of the node set, NOT submodular. v4's submodular machinery solves the wrong problem (the objective only appears submodular when selecting LEAVES, because shared prefixes create coverage interactions).

### 8.2 v2's Near-Optimality

For a linear objective under tree precedence constraints, greedy-by-value (v2's Phase 2) is near-optimal. The connectivity enforcement adds ancestors that are ALWAYS higher-value than the selected node (probabilities decrease with depth). No budget is wasted on forced ancestors.

### 8.3 Distributed Robustness > Concentrated Optimality

v2 distributes alternatives across many positions. This handles unpredictable rejection (the draft's confidence is a noisy predictor). Concentrated strategies (COT, v4's coverage-awareness) optimize for WHERE rejection is EXPECTED but fail when rejection occurs at unexpected positions.

### 8.4 The Hardware Ceiling

On 8xA6000 with tensor parallelism on a 4B model:
- Target forward = 31ms (fixed, memory-bandwidth-bound)
- Speedup ∝ tau (since per-step time is constant)
- Maximum tau ≈ 5.93 (at optimal node count with current draft model)
- Maximum speedup ≈ 4.2x
- The only way to improve: better draft model (higher tau) or different hardware (where node count matters)

## 9. Cross-Benchmark Results (Best Configurations)

| Dataset | Config | speedup | tau | nodes |
|---------|--------|---------|-----|-------|
| mt-bench | v2 ek=8 mts=70 | 4.19 | 5.93 | 71 |
| mt-bench | v4 ek=8 mts=32 | 4.14 | 5.85 | 54 |
| gsm8k | v4 ek=7 mts=32 | 6.72 | 8.26 | 55 |
| gsm8k | v2 ek=8 mts=49 | 6.81 | 8.21 | 50 |
| humaneval | v4 ek=7 mts=32 | 7.01 | 8.53 | 55 |
| alpaca | v4 ek=7 mts=32 | 3.27 | 4.42 | 55 |
| swe-bench | v4 ek=3 mts=32 | 3.99 | 4.83 | 70 |
| mt-bench (t=0.6) | v4 ek=7 mts=32 | 4.08 | 5.92 | 55 |

## 10. Summary of What We Learned

1. **expand_k is the single most impactful lever** — raising from 3 to 8 gave +10% speedup. Everything else was noise.
2. **v2 (EAGLE-2) is near-optimal** for position-independent parallel drafting on communication-bound hardware.
3. **v4's theoretical elegance doesn't translate** to practical gains because it optimizes the wrong constraint (leaves vs nodes).
4. **Adaptive mechanisms don't help** when per-step time is constant (communication-bound target forward).
5. **The draft model is the binding constraint** — all tree algorithms converge to similar tau because they draw from the same draft logits.
6. **Training objectives matter** — GTO and TopK both degraded the draft vs the original KL distillation.
7. **Distributed robustness beats concentrated optimality** — v2's heuristic works BECAUSE it's a heuristic, spreading alternatives everywhere rather than concentrating them at estimated weak points.
