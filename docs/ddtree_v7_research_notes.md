# DDTree / v7 Research Notes

Comprehensive context for speculative decoding research on DFlash.
Captures algorithm analysis, flaw identification, and proposed improvements.

---

## 1. Repo overview

**DFlash** = block diffusion draft model for speculative decoding.

- **Draft**: Small model produces `block_size=16` token logits in parallel (one forward pass, bidirectional attention over mask tokens).
- **Tree builder**: Constructs a candidate verification tree from draft logits.
- **Target**: Full model verifies the tree in one forward pass.
- **Accepted path**: longest leaf match; bonus token sampled from target.

**Key files**:
- `benchmark.py` — main eval driver with `dflash_generate()`.
- `model/dflash_tree.py` — tree builders (v1-v7).
- `model/dflash.py` — draft model (Qwen3-based). "noise_embedding" parameter = embedding of `[anchor, mask, mask, ...]` via target's embed_tokens. NOT stochastic.
- `program.md`, `docs/adaptive_tree_sizing_report.md` — experiment logs.

**Metrics** (priority order):
1. Speedup (wall-clock ratio vanilla AR / speculative) — headline.
2. Tau (acceptance length, mean tokens accepted per step).
3. Avg trie nodes (verification cost proxy).

**Standard test**: mt-bench 80 samples, Qwen3-4B target + z-lab/Qwen3-4B-DFlash-b16 draft, 8 GPUs tensor-parallel.

**Prior best**: v2 (EAGLE-2 style) mts=70 ek=8 on mt-bench → speedup 4.19, tau 5.93, nodes 71.

---

## 2. Tree builder zoo

| Version | Function | Algorithm | Budget unit |
|---------|----------|-----------|-------------|
| v1 | `build_dynamic_tree` | Threshold + cartesian product | — |
| v2 | `build_dynamic_tree_v2` | EAGLE-2 expand + rerank by cumulative conf | Non-root trie nodes |
| v3 | `build_bestfirst_tree` | Priority-queue best-first | Leaves |
| v4 | `build_prefixaware_tree` | Best-first expand + lazy-greedy submodular select | **Leaves** (→ ~2-2.5x node blowup) |
| v6 | `build_efficiency_tree` | Density-greedy (Δf/Δg ratio) | Leaves + self-sizing |
| **v7** | `build_node_budget_tree` | **DDTree — single-phase heap, keep every popped node** | **Trie nodes directly** |

v7 was added during this research session as an implementation of the DDTree algorithm.

---

## 3. The DDTree / v7 algorithm

**Setup**: draft model gives `q_1, ..., q_L` — independent marginal distributions at each position.

**Objective**: maximize E[tau] subject to |T| ≤ B trie nodes.

**Key identity** (under product distribution assumption):
\[
E[\tau] = \sum_{u \in T} q(u) \quad \text{where } q(u) = \prod_{i=1}^{d(u)} q_i(u_i)
\]

**Algorithm** (child+sibling enumeration):

```
heap = [(rank-1 at pos 1, log q_1(rank-1))]
for B iterations:
    pop (prefix, score) ← heap  # max-score prefix
    add prefix to tree
    # child: extend with rank-1 at next position
    push (prefix + rank-1, score + log q_{d+1}(rank-1))
    # sibling: increment last position's rank
    push (prefix[:-1] + (last_rank+1), adjusted_score)
```

**Complexity**: O(B log B). Each pop produces exactly 2 pushes.

**Optimality theorem**: Under product distribution + trie-node budget, the B highest-probability prefixes maximize ∑ q(u). Prefix-closure is automatic because q(parent) ≥ q(child) under product distribution (each extension multiplies by q_d(u_d) ≤ 1).

**Why this is strictly better than v4**:
- v4 budgets on leaves → 2-2.5x node blowup, can't target node count precisely.
- v4 uses submodular greedy → only (1-1/e) approximation to OPT_leaf, which is the wrong optimum anyway (OPT_node ≥ OPT_leaf).
- v7 budgets on nodes directly → exact.
- v7 is single-phase (no Phase 2) because under product distribution, each node contributes independently to ∑ q(u). No submodular interaction.

**Our implementation** uses v4's `expand_k` fan-out (rather than true child+sibling) for the heap expansion, capped at B pops. Under `expand_k` large enough (8), this behaves nearly identically to child+sibling, though child+sibling is cleaner and eliminates the artificial rank cutoff.

---

## 4. DDTree paper reference (MATH-500, Qwen3-8B, temp=0.0)

Budget tradeoff: speedup peaks at B=256-512, tau keeps climbing to 1024. Speedup around 11x at peak. Paper's takeaway: front-heavy tree (lots of high-probability nodes) beats flat block of 16.

The repo was stuck at 32-70 nodes — massively under-budgeted. The v7 sweep goes to 1024 to find the peak on this hardware.

---

## 5. Deep analysis: what's wrong with DDTree

### Flaw 1: Product distribution ≠ true joint

DFlash's draft has bidirectional attention — every q_i attends to every q_j through hidden states. The q_i's are marginals of a joint, not samples from independent distributions.

\[
q^{\text{true joint}}(u_1, ..., u_d) \neq \prod_i q_i(u_i)
\]

Product = mean-field approximation of a correlated joint. Systematically wrong in one direction: **product distribution OVERESTIMATES mixed-rank paths.**

### Flaw 2: Depth-deviation paths over-valued (the dominant failure)

Let d* = first position where prefix deviates from rank-1. For all i > d*, unconditional q_i overestimates conditional q_i(u_i | u_<i).

- Argmax chain (1,1,1,...,1): joint ≈ product (small error)
- Single late deviation (1,1,...,1,2): joint ≈ product (small error)
- Early deviation + long continuation (1,3,1,1,...,1): **joint << product (LARGE error)**

DDTree at B=16 picks mostly argmax chain + immediate siblings — correct.
DDTree at B=1024 picks 80%+ mixed-rank deep paths — mostly phantom value.

**This explains the paper's speedup plateau at 256-512**: extra nodes past 256 DO contribute to ∑ q(u), but they DON'T contribute to actual accepted tokens.

### Flaw 3: Block-size ceiling

Tau ≤ block_size = 16. Easy sequences (math500 argmax chain) saturate. Draft could continue but we stop at position 15.

### Flaw 4: Single-step greedy

Bonus token at step N becomes anchor for step N+1. Different leaves → different bonus tokens → different step N+1 quality. DDTree optimizes each step in isolation; blind to downstream effects.

### Flaw 5: Rank representation is context-free

Rank-2 at a position could be a near-synonym of rank-1 (same acceptance behavior) or a divergent alternative (different). Product probability is identical; DDTree can't distinguish.

### Flaw 6: Heap is CPU, item-by-item

At B=1024, heap operations take real time (~5-10ms). No GPU vectorization. Wall-clock issue, not quality.

### Flaw 7: Fixed budget per step

Easy steps over-provisioned, hard steps under-provisioned. Waste.

---

## 6. Proposed fixes (training-free)

### Fix A: Power-scaled / deviation-penalized scoring (attacks Flaw 2)

Replace DDTree's score with:

\[
\text{score}(u) = \sum_{i=1}^{d} \alpha^{i-1} \log q_i(u_i) \;-\; \beta \cdot \#\{i : u_i \neq \text{rank-1}\}
\]

- α ∈ (0, 1]: depth discount (reliability decay)
- β > 0: deviation penalty (mixed-path correction)
- α = 1, β = 0 recovers DDTree

**Why principled**: joint-vs-product error grows with depth AND number of deviations. Two knobs = first-order correction to mean-field misspecification.

**Expected behavior**:
- B < 128: minimal effect (DDTree picks argmax-heavy tree anyway)
- B > 256: **breaks the plateau** — marginal budget flows to wider shallow paths or longer argmax chains instead of phantom mixed-deep-paths

**Impl cost**: 5 lines. Two hyperparameters to sweep.

### Fix B: Chained speculation (attacks Flaw 3)

When current step's tau ≥ 0.8 × block_size:
1. Don't return to main loop
2. Immediately run another draft forward anchored at bonus token
3. Build new tree, verify, accept
4. Chain until tau drops

**Why it works**: KV cache stays warm, saves a full target forward (30ms) per chained extension.

**Wins on**: math500, humaneval, code — any domain with long deterministic runs.

**Impl cost**: 20 lines. One threshold.

### Fix C: Downstream-aware leaf scoring (attacks Flaw 4)

Augment score:

\[
\text{score}(u) = \log q(u) + \gamma \cdot \log \max_v q_{d+1}(v)
\]

Rewards paths ending at positions with confident next-position prediction → bonus token leads to "locked in" continuation → better step N+1.

**Impl cost**: 3 lines. One knob. Free signal (q_{d+1} already computed).

### Fix D: Per-position uncertainty-aware branching (original noise-ensemble idea)

**Originally proposed**: run draft twice with different noise, use KL divergence as per-position uncertainty signal, scale expand_k by it.

**CORRECTION**: DFlash is DETERMINISTIC. "noise_embedding" in the code is just the embedding of `[anchor, mask, mask, ...]`. Two draft forwards give identical outputs under greedy decoding.

**Salvageable alternatives**:
- **MC dropout**: enable dropout at inference (training/inference mismatch risk)
- **Anchor perturbation**: run with target top-2 as alternate anchor
- **Input perturbation**: Gaussian noise on `target_hidden` (artificial)
- **Layer ensemble**: apply lm_head to hidden states at layers L and L-2; KL between them as uncertainty. **Zero extra forward pass cost** — uses only already-computed information. This is the clean version.

Layer ensemble signal: positions where last-layer and second-to-last-layer distributions disagree = still-settling = uncertain → widen branching.

---

## 7. Other inference-time techniques (non-DDTree)

### A. Target-logit calibration (cross-step)

Target computes logits at EVERY tree node during verification. Currently only accepted-path + bonus logits used. Rest discarded.

Build online calibration `C[d, rank]` = ratio of target probability to draft probability, updated from observed tree-node pairs. In next step, multiply draft score by `C[d, rank(token)]`.

**Subtlety**: miscalibration has two components:
1. Marginal draft-target mismatch
2. Ancestry-dependent conditioning

C[d, rank] only corrects (1), averages over (2). Component (2) is larger in practice.

Refinement: online Platt scaling `log P_T(v) ≈ a_d + b_d · log q_d(v)` — one regression per depth.

**Expected gain**: +0.05 to +0.15 tau. Not transformative but real.

### B. Block-carryover as latency optimization

Stale draft logits from step N are strictly LESS informative than fresh logits at step N+1 (step N+1 sees additional real tokens as context that were masks for step N). So don't use for quality.

BUT: on EASY steps (high EWMA), skip the fresh draft forward entirely, use stale logits. Saves ~15ms. Tradeoff small tau loss for wall-clock gain.

**Expected gain**: 0 tau, +3-5% wall-clock.

### C. Cache-based verification skip

In long decoding runs, target produces stereotyped patterns (boilerplate, formatting). If a sequence of tokens appears identical to a previously-verified sequence at a similar context, skip the target forward entirely — reuse the prior verification's result.

Training-free cache of (context hash, token sequence) → verified. Potentially 2-3x on top of DDTree for code/math workloads.

Not explored in the repo. Bigger project.

### D. Variance-aware objective (DEBUNKED for throughput)

My original claim: max E[tau] - λ·Var[tau] beats max E[tau] for wall-clock. **Wrong** for steady-state throughput — total time = N_tokens · T_step / mean(tau), depends only on mean.

Relevant for latency SLAs / batch scheduling. NOT for the benchmark setting.

### E. MCTS / UCB over tree

Doesn't fit: no simulator, no repeated visits. The one useful reframe is **robust optimization**: `max_T min_{q' ∈ KL-ball(q)} ∑_u q'(u)`. Closed form: down-weight path by e^{-ε H_path(u)}. But this is basically entropy-discounted DDTree with a single knob.

---

## 8. Ranked research directions

| Direction | Expected gain | Cost | Confidence |
|-----------|---------------|------|------------|
| **Power-scaled scoring (Fix A)** | Large at B > 256, breaks plateau | 5 lines | High |
| **Chained speculation (Fix B)** | Large on easy sequences | 20 lines | High |
| Target-logit calibration (C7.A) | +0.05–0.15 tau | Moderate | Medium |
| Downstream-aware leaves (Fix C) | +0.05 tau, compounds | 3 lines | Medium |
| Layer-ensemble uncertainty (Fix D) | +0.05–0.15 tau | 0 extra forward | Medium |
| Block-carryover as latency (C7.B) | +3-5% wall-clock | Low | Medium |
| Cache-based verification skip (C7.C) | **Big on code/math** | High (new system) | Speculative |

---

## 9. Current experimental status

**Running**: v7 node-budget sweep on math500 (Qwen3-4B + DFlash-b16, 8 GPUs, temp=0.0, 256 samples).

Budgets: 16, 32, 64, 128, 256, 512, 1024.

**Completed**:
- v7 mts=16: **speedup 7.26, tau 8.58, 17 nodes** ✓

**Running**: v7 mts=32 (in startup/early samples)

**Queue**: 64, 128, 256, 512, 1024

**In parallel on other node**: v4 sweep with leaf budgets chosen to match v7's node counts (mts=1, 14, 30, 58, 125, 250, 500). First point gave ~26-27 nodes at mts=8 → 3.3x blowup (higher than predicted 2.25x at small budgets).

**Scripts**:
- `run_v7_sweep.sh`
- `run_v4_sweep.sh`
- `paper/fig_pareto_v4_vs_v7.py` — parses logs, generates Pareto comparison

---

## 10. Key empirical questions to answer

1. Does v7 peak and plateau at 256-512 nodes (like paper's MATH-500), or keep climbing to 1024?
   - If plateau: Fix A (power-scaled scoring) directly attacks the cause
   - If no plateau: target cost is flat on 8 GPUs; different bottleneck

2. Does v7 beat v4 at matched node counts?
   - Theoretical prediction: YES, by construction (exact-optimal vs approximate)
   - Need empirical confirmation

3. Where does tau vs speedup gap widen?
   - Plateau point in speedup while tau keeps climbing = exactly the Flaw 2 signature

4. Do any sequences saturate block_size=16?
   - Histogram of tau values per step
   - If substantial mass at tau=15, Fix B (chained speculation) has runway

---

## 11. Key config reference

```bash
# Current v7 (baseline)
torchrun --nproc_per_node=8 --master_port=29501 benchmark.py \
  --dataset math500 --max-samples 256 \
  --model-name-or-path Qwen/Qwen3-4B \
  --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 \
  --tree-version 7 --max-tree-size <MTS> --expand-k 8 \
  --temperature 0.0
```

---

## 12. Prior experimental baselines (mt-bench, Qwen3-4B, temp=0)

| Method | speedup | tau | nodes |
|--------|---------|-----|-------|
| v2 ek=8 mts=70 | 4.19 | 5.93 | 71 |
| v4 ek=8 mts=32 | 4.14 | 5.85 | 54 |
| v2 ek=8 mts=49 | 4.12 | 5.78 | 50 |
| v4 ek=8 mts=48 | 4.05 | 5.82 | 72 |
| v2 ek=3 mts=70 | 3.85 | 5.52 | 71 |
| v4 ek=3 mts=32 | 3.81 | 5.60 | 70 |

Note: everything here is under-budgeted. DDTree paper suggests 256-512 is the sweet spot.

---

## 13. Architecture nuance (gotcha)

- `noise_embedding` parameter name is misleading. It's `target.model.embed_tokens(block_output_ids)` where `block_output_ids` = `[anchor, mask, mask, ...]`. Fully deterministic given anchor.
- Draft model block_size = 16. Can't exceed 16 accepted tokens per step without chained speculation (Fix B).
- Target uses `sdpa` during tree verification (flash-attn doesn't support arbitrary attn masks).
- Finding 5 from prior experiments: target_backbone = 31ms/step regardless of node count (memory-bandwidth-bound on 8-GPU Qwen3-4B). Implies adaptive budget has limited upside on this setup.

---

## 14. Open problems worth the reviewer's time

1. **The plateau is real — why?** Power-scaled scoring hypothesis: phantom mixed-rank deep paths. Alternative hypotheses: (a) target cost actually nonlinear past 512 nodes, (b) KV-trim overhead becomes dominant, (c) attn-mask quadratic scaling.

2. **Is there a MORE principled correction than power-scaling?** Could we estimate the joint-vs-product error empirically by comparing observed acceptance rates to predicted q(u)? This would give a calibration table indexed by (depth, #deviations) instead of just (depth, rank).

3. **Can chained speculation push speedup past the DDTree paper's ~11x?** If tau ceiling is the constraint on easy sequences, chaining could hit tau=30+ in single "virtual step."

4. **Does layer-ensemble uncertainty correlate with target-rejection?** Empirical question — needs instrumentation but is cheap to check.
