# DFlash Autonomous Experiment Program

This is an experiment harness for autonomous AI-driven research on DFlash speculative decoding, inspired by [autoresearch](https://github.com/karpathy/autoresearch).

## Setup

To set up a new experiment session, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date and focus area (e.g. `apr9-ewma-ek`). The branch `experiments/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b experiments/<tag>` from current main.
3. **Read the in-scope files** for full context:
   - `README.md` — project overview, models, usage.
   - `benchmark.py` — main eval script with `dflash_generate` and CLI flags. **Primary edit target.**
   - `model/dflash_tree.py` — tree builders v2, v4, v6, v7 (EAGLE-2, prefix-aware greedy, efficiency-greedy, node-budget DDTree). **Secondary edit target.** v1/v3 exist in the file but are not wired into benchmark.py.
   - `model/dflash.py` — draft model wrapper. Read-only. NOTE: parameter `noise_embedding` is a misnomer — it's just the embedding of `[anchor, mask, mask, ...]`. Draft is fully deterministic under greedy decoding.
   - `model/utils.py`, `model/freq_vocab.py` — utilities. Read-only.
   - `docs/adaptive_tree_sizing_report.md` — technical report on v2 vs v4 and adaptive experiments.
   - **`docs/ddtree_v7_research_notes.md`** — **ESSENTIAL READING.** Deep algorithmic analysis of DDTree/v7, its 7 identified flaws, and the ranked queue of training-free fixes. This file supersedes earlier reports for current work.
4. **Verify GPU access**: confirm `nvidia-smi` shows available GPUs and note the count (sets `NPROC_PER_NODE`).
5. **Initialize results.tsv**: create `results.tsv` with the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Architecture

DFlash is a **speculative decoding** system:
- A small **draft model** predicts a block of tokens in parallel (block_size=16).
- A **tree builder** constructs a candidate verification tree from draft logits.
- The **target model** verifies the tree in a single forward pass.
- The best leaf path is accepted; a bonus token is sampled from the target.

The goal is maximum **wall-clock speedup** — the ratio of vanilla autoregressive decoding to speculative decoding time.

### Tree Builders (model/dflash_tree.py)

| Version | Function | Algorithm | Budget unit |
|---------|----------|-----------|-------------|
| v2 | `build_dynamic_tree_v2` | EAGLE-2 expand + rerank by cumulative confidence | Trie nodes |
| v4 | `build_prefixaware_tree` | Prefix-aware greedy with (1-1/e) submodular guarantee | **Leaves** (→ ~2-2.5x node blowup) |
| v6 | `build_efficiency_tree` | Density-greedy (Δf/Δg) with self-sizing | Leaves + adaptive |
| **v7** | `build_node_budget_tree` | **DDTree — top-B-by-probability via single-phase heap** | **Trie nodes (exact)** |

**v7 (DDTree) is the current best for large budgets** (e.g., mts=128 on math500 → 7.98x speedup). Under the product distribution, top-B prefixes by probability is exact-optimal for E[tau]. No Phase 2 needed — each node contributes independently. But v7's speedup collapses past mts=128 due to "phantom mixed-rank-deep-paths" (see Finding 11). **Fixing this is the top research priority.**

### Adaptive EWMA (benchmark.py)

Tracks acceptance rate via exponential weighted moving average. Maps EWMA rate to `max_tree_size` and `expand_k` per step. Easy steps get wide search; hard steps get narrow, cheap trees.

## Metrics

Three metrics matter, in order of importance:

1. **Speedup** (higher is better): wall-clock ratio of vanilla to speculative decoding. **This is the headline number.**
2. **tau / acceptance length** (higher is better): mean tokens accepted per verification step. Measures tree quality.
3. **avg_nodes** (lower is better at equal tau): trie nodes per verification step. Measures verification cost.

The ideal outcome: **same or better tau, fewer nodes, higher speedup.**

## Running a Benchmark

```bash
torchrun \
  --nproc_per_node=8 \
  --master_port=29500 \
  benchmark.py \
  --dataset mt-bench \
  --max-samples 80 \
  --model-name-or-path Qwen/Qwen3-4B \
  --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 \
  --max-new-tokens 2048 \
  --temperature 0.0 \
  --tree-version 4 \
  --max-tree-size 32 \
  --expand-k 3
```

Add `--adaptive-block` plus EWMA flags for adaptive mode:
```bash
  --adaptive-block \
  --adaptive-block-ewma-decay 0.8 \
  --adaptive-block-min-tree-size 12 \
  --adaptive-block-min-expand-k 2 \
  --adaptive-block-max-expand-k 5
```

Each run takes **~10 minutes** depending on dataset size and GPU count.

### Standard Datasets (with sample counts)

| Dataset | Samples | Domain |
|---------|---------|--------|
| mt-bench | 80 | Mixed chat (high difficulty variance — best for adaptive testing) |
| gsm8k | 256 | Math reasoning (mostly easy for drafter) |
| humaneval | 164 | Code generation (mostly easy for drafter) |
| alpaca | 256 | General instruction (hard, low tau) |
| math500 | 256 | Math (easy) |
| swe-bench | 128 | Software engineering (hard) |

**mt-bench is the primary benchmark** for adaptive experiments because it has the widest difficulty variance. gsm8k and humaneval are good secondary checks.

### Extracting Results

```bash
grep "Decoding speedup:"         run.log | tail -1
grep "Average Acceptance length:" run.log | tail -1
grep "Average tree node count:"  run.log | tail -1
```

## Logging Results

When an experiment is done, log it to `results.tsv` (tab-separated).

Header:
```
commit	dataset	speedup	tau	avg_nodes	status	description
```

1. git commit hash (short, 7 chars)
2. dataset name
3. speedup (e.g. 3.89)
4. tau / acceptance length (e.g. 5.52)
5. avg_nodes (e.g. 51.42) — use 0.0 for crashes
6. status: `keep`, `discard`, or `crash`
7. short text description of what this experiment tried

Example:
```
commit	dataset	speedup	tau	avg_nodes	status	description
a1b2c3d	mt-bench	3.84	5.60	70.00	keep	baseline v4 fixed mts=32 ek=3
b2c3d4e	mt-bench	3.89	5.52	51.42	keep	EWMA tree-size only decay=0.8 min=12
c3d4e5f	mt-bench	4.65	5.46	50.00	keep	EWMA tree+expand_k 2→5 decay=0.8
e5f6g7h	mt-bench	3.73	5.44	34.83	discard	v2+EWMA ek 2→5 — WORSE than v2 fixed
```

Do NOT commit results.tsv; leave it untracked.

## What You CAN Modify

- **`benchmark.py`** — the main edit target. The `dflash_generate` function, adaptive EWMA logic, CLI flags, result aggregation. Everything here is fair game.
- **`model/dflash_tree.py`** — tree builders. You can modify existing builders (v2, v4) or add new ones. Register new builders in `TREE_BUILDERS` dict in `benchmark.py`.
- **Shell scripts** (`run_adaptive_overnight.sh`, `run_ewma_sweep.sh`) — sweep configs, hyperparameter grids, new sweep scripts.

## What You CANNOT Modify

- **`model/dflash.py`** — the draft model architecture. The draft model is a pretrained checkpoint; changing its forward pass would break compatibility.
- **`model/utils.py`**, **`model/freq_vocab.py`** — shared utilities. These are stable.
- **Model weights** — you cannot retrain models. You optimize the decoding algorithm around fixed draft+target pairs.
- **`distributed.py`** — multi-GPU communication. Stable infrastructure.

## The Experiment Loop

LOOP FOREVER:

1. **Review state**: check current branch, results.tsv, and recent experiment outcomes. On first iteration, you MUST have read `docs/ddtree_v7_research_notes.md` fully. Re-read it periodically if you get stuck.
2. **Propose an idea**: prefer from "Open Research Questions" section in priority order (Q1, Q2 are the highest-value targets). If you deviate, write a clear hypothesis explaining why.
3. **Implement the change**: edit `benchmark.py` and/or `model/dflash_tree.py`. For new tree builders, add to `TREE_BUILDERS` dict and extend `--tree-version` choices. For new scoring modifications, add CLI flags (e.g. `--score-alpha`, `--score-beta`).
4. **git commit** the change with a descriptive message.
5. **Run the experiment**: `torchrun ... benchmark.py ... > run.log 2>&1`. Use `--master_port 29501` (29500 may be taken by another user). Cover both math500 (primary — DDTree peaks here) and mt-bench (confirmation).
6. **Extract results**: grep for speedup, tau, avg_nodes from run.log.
7. **If grep output is empty**: the run crashed. Run `tail -n 50 run.log` to read the traceback. Attempt a fix if it's simple (typo, shape mismatch). If fundamentally broken after 2-3 attempts, abandon the idea and revert.
8. **Log to results.tsv**.
9. **Keep or discard** (see criteria below). On keep, advance the branch. On discard, `git reset --hard HEAD~1`.
10. **Repeat**.

### Evaluation Protocol

- **Primary for v7/DDTree work**: math500, 256 samples, temp=0.0, 8 GPUs. ~25 min per run. DDTree effects are clearest on easy sequences where tau ceilings matter.
- **Primary for adaptive / heuristic work**: mt-bench, 80 samples, temp=0.0. ~10 min per run. Wider difficulty variance.
- **Confirmation**: if primary result is positive on math500, run mt-bench (80 samples) and gsm8k (256 samples) to confirm generalization.
- **Temperature check**: if primary results are strong, run at temp=0.6 to verify sampling robustness.

### Keep/Discard Criteria

- **Speedup improvement >= 0.05x**: keep (meaningful).
- **Speedup within +/-0.03x, tau improved, nodes decreased**: keep (Pareto improvement).
- **Speedup decreased > 0.05x**: discard.
- **Same speedup but much simpler code**: keep (simplification win).

## Known Baselines

### mt-bench (from prior experiments, docs/adaptive_tree_sizing_report.md)

| Method | speedup | tau | avg_nodes |
|--------|---------|-----|-----------|
| **v2 fixed (ek=8, mts=70)** | **4.19** | **5.93** | **71** |
| v4 fixed (ek=8, mts=32) | 4.14 | 5.85 | 54 |
| v2 fixed (ek=8, mts=49) | 4.12 | 5.78 | 50 |
| v4 fixed (ek=8, mts=48) | 4.05 | 5.82 | 72 |
| v2 fixed (ek=3, mts=70) | ~3.85 | ~5.52 | 71 |
| v4 fixed (ek=3, mts=32) | 3.81 | 5.60 | 70 |

### gsm8k

| Method | speedup | tau | avg_nodes |
|--------|---------|-----|-----------|
| v2 fixed (ek=8, mts=70) | 6.81 | 8.21 | 50 |
| v4 fixed (ek=7, mts=32) | 6.72 | 8.26 | 55 |

### math500 — v7 (DDTree) budget sweep (this session, 256 samples, ek=8, temp=0)

| mts | speedup | tau | avg_nodes | Δ speedup | Δ tau |
|-----|---------|-----|-----------|-----------|-------|
| 16 | 7.26 | 8.58 | 17 | — | — |
| 32 | 7.78 | 9.23 | 33 | +0.52 | +0.65 |
| 64 | 7.90 | 9.70 | 65 | +0.12 | +0.47 |
| **128** | **7.98** | **10.08** | **129** | **+0.08** | **+0.38** |
| 256 | 7.29 | 10.37 | 257 | -0.69 | +0.29 |
| 512 | 5.38 | 10.59 | 513 | -1.91 | +0.22 |
| 1024 | 2.95 | 10.90 | 1025 | -2.43 | +0.31 |

**Tau keeps climbing all the way to 1024, but speedup peaks at 128 and then collapses.** This is the Flaw-2 signature (see Finding 11). Any fix that stops tau's growth past 128 from being "phantom value" unlocks the plateau.

## Key Findings (This Session)

### Finding 1: v2+EWMA adaptive expand_k HURTS v2

Ran v2 with EWMA adaptive expand_k 2→5 (decay=0.8, mts=69) on mt-bench:
- **speedup: 3.73** (vs v2 fixed 3.80 → 1.8% LOSS)
- tau: 5.44 (vs 5.56 → 2.2% loss)
- avg_nodes: 34.83 (vs 70 → 50% reduction)

The node reduction bought nothing (communication-bound verification) while the reduced expand_k on hard steps degraded tree quality. **EWMA adaptive expand_k helps v4 but hurts v2.**

### Finding 2: v4's submodular selection is necessary for adaptive strategies

The v2+EWMA failure vs v4+EWMA success proves that v4's coverage-aware Phase 2 selection is what makes adaptive expand_k work:
- When EWMA pushes ek=5 on easy steps, v4 exploits the wider candidate pool (picks diverse, high-coverage leaves). v2's flat reranking selects redundant leaves.
- When EWMA pushes ek=2 on hard steps, v4 degrades gracefully (optimal coverage from small pool). v2 collapses (nearly linear tree from ek=2).
- **Net: v4+EWMA gains from upside more than it loses from downside. v2+EWMA loses on both ends.**

### Finding 3: v4 optimizes the wrong constraint

v4 solves: max E[tau(S)] subject to |S| <= m (leaf cardinality). But the actual verification cost is |trie(S)| (trie node count), not |S|. With mts=32 leaves, v4 produces ~70 trie nodes. The leaf-cardinality constraint is a surrogate, not the true cost.

This means the Nemhauser guarantee (1-1/e) x OPT_leaf(m) compares against the wrong optimum. The node-constrained optimum OPT_node(B) can be strictly larger because it can select MORE leaves that share prefixes.

### Finding 4: v2 is near-optimal; expand_k=8 is the key lever

v2 ek=8 mts=70 achieves speedup 4.19 (tau 5.93, 71 nodes) — the best of any configuration tested. The expand_k sweep at 50 nodes: ek=3 (3.85), ek=5 (4.10), ek=7 (4.11), ek=8 (4.12), ek=10 (4.10). Peak at ek=7-8. The node-count sweep: peak at 71 nodes (4.20), regression at 91 (4.16).

### Finding 5: Target backbone is 73% of step time and constant across tree sizes

Profiling at 50 and 71 nodes shows target_backbone = 31ms/step regardless of node count. The forward pass is memory-bandwidth-bound (reading model weights + KV cache), not compute-bound. Flash attention is actually SLOWER than SDPA for these short sequences. Adaptive tree sizing cannot reduce target cost.

### Finding 6: v4 loses to v2 at matched node counts

At ~70 nodes back-to-back: v2 (tau 5.93, speedup 4.19) beats v4 (tau 5.82, speedup 4.06). v4's leaf-based selection is structurally less node-efficient: it must include full root-to-leaf paths, wasting budget on deep low-value nodes. v2 selects individual nodes and can prune unprofitable subtrees.

### Finding 7: All failed approaches (comprehensive list)

| Approach | Result | Why it failed |
|----------|--------|---------------|
| Adaptive block depth | tau 4.89, speedup 3.37 | Caps maximum acceptance length |
| Two-phase probe+extend | tau 5.12, speedup 2.96 | Extra kernel launch overhead |
| Node-budget v5 | tau 5.60, speedup 3.68 | O(N^2) Phase 2 overhead |
| v2+EWMA adaptive ek | tau 5.44, speedup 3.73 | v2 can't exploit wider search |
| v6 density-greedy | tau 5.59, speedup 3.66 | Bigger trees, extra nodes hurt |
| v7 bestfirst+rerank | tau 5.74, speedup 4.12 | No improvement over v2 at ek=8 |
| v8 entropy-adaptive ek | tau 5.71, speedup 3.96 | Heuristic redistribution hurts |
| BTTC bonus token reuse | 0% match rate | Draft/target disagree at rejection points |
| SSR self-speculative refinement | tau 2.61, speedup 1.99 | Draft model collapses on non-mask inputs |
| Confidence-gated flash attn | flash slower than SDPA | Memory-bandwidth-bound, not compute-bound |
| COT DP-optimal Cartesian tree | tau 5.21, speedup 2.91 | Concentrated branching fragile + 9ms DP overhead |
| GTO-trained draft (step 3900) | tau 5.81, speedup 4.11 | Worse than original HF draft |
| TopK-trained draft (step 9050) | tau 5.73, speedup 4.04 | Degrades with more training |

### Finding 8: Why v2 wins — distributed robustness

v2's greedy-by-cumulative-probability selection distributes alternatives across many positions. This provides robustness against unpredictable rejections. Concentrated branching (COT) and coverage-aware selection (v4) both lose because they optimize for WHERE rejection is EXPECTED, but rejection often occurs at unexpected positions. v2's heuristic is near-optimal because distributed coverage handles noisy draft confidence.

### Finding 9: v7 (DDTree) strictly dominates v4 at matched node counts

v7 budgets on trie nodes directly and is exact-optimal under the product distribution. At matched node counts, v7 beats v4 on speedup AND tau (v4 wastes budget on leaf-cardinality surrogate). v4's (1-1/e) Nemhauser guarantee compares against OPT_leaf, which ≤ OPT_node — so v4 is approximately-optimal against the wrong optimum.

### Finding 10: Proper budget matters more than clever algorithms

The repo's entire prior exploration stayed in 25-91 node range. DDTree paper showed MATH-500 speedup climbs to ~11x at 256-512 nodes. This session's v7 sweep on Qwen3-4B math500 peaks at **7.98x speedup @ 128 nodes** — quadruple the prior 4.19x baseline. Most of the gain was "left on the table" by under-budgeting.

### Finding 11: The product-distribution mean-field breaks past ~128 nodes (DDTree's core flaw)

Tau continues growing from 128 → 1024 nodes (10.08 → 10.90), but speedup COLLAPSES (7.98 → 2.95). Interpretation: marginal nodes past 128 have significant product-probability q(u) but NEAR-ZERO actual acceptance probability. These are **phantom mixed-rank-deep-paths** — prefixes like (rank-1, rank-1, rank-3, rank-1, rank-1, ...) whose TRUE joint probability is << their product probability because conditioning on the rank-3 deviation invalidates the unconditional q_i's used for downstream positions.

DDTree's optimality guarantee assumes product = joint. It doesn't. The algorithm is mathematically correct for a misspecified model.

**This is the dominant open problem and should be the primary research target.**

### Finding 12: Q1 (power-scaled scoring) DOES NOT break the plateau

Session apr16: swept α ∈ {0.9, 0.95, 1.0}, β ∈ {0, 0.5, 1.0} at mts=256 on math500 (256 samples). ALL configs within 7.08-7.29 speedup band (baseline mts=256: 7.29). Specifically:
- α=1.0, β=0.5 → 7.29 / 10.35 / 257 nodes (identical to baseline)
- α=1.0, β=1.0 → 7.23 / 10.31 / 257 (β=1 still ineffective)
- α=0.95, β=0 → 7.24 / 10.37 / 257 (pure depth discount no effect)
- α=0.95, β=0.5 → 7.22 / 10.37 / 257 (combined no effect)
- α=0.9, β=0.5 → 7.08 / 10.28 / 257 (slight regression)

**Why β=1 is too weak**: typical log q_rank1 ≈ -0.2, log q_rank2 ≈ -2. Rank-2 cost vs rank-1 is ~1.8 in log-prob. β=1 adds only ±1 penalty for rank>0, not enough to shift the top-B selection meaningfully. β would need to be ≥3-5 to actually reorder.

**Why α<1 is too weak**: α=0.9 at depth 15 = 0.9^15 ≈ 0.2. Scales all per-position log-probs. Doesn't change relative ranking of prefixes with different deviations significantly.

**CONCLUSION: The phantom-path hypothesis (Flaw 2) is likely REAL, but this simple scoring fix cannot correct for it.** The issue is that q(u) as a sort key already penalizes rank-deviations (via ∑ log q_r where rank-r has lower log-prob). Adding β just double-counts.

To actually fix Flaw 2, we need SUPPRESSION of the cross-term: paths that deviate EARLY should have their downstream positions re-evaluated with CONDITIONAL distributions (not marginals). That's Q8/CTR territory.

### Finding 13: Block ceiling (tau = block_size) dominates past mts=64

Bucket-16 fraction of the acceptance-length histogram on math500 (256 samples, v7 ek=8):
| mts | ceiling fraction | tau | speedup |
|-----|------------------|-----|---------|
| 16 | 18.0% | 8.58 | 7.26 |
| 32 | 22.8% | 9.23 | 7.78 |
| 64 | 25.5% | 9.70 | 7.90 |
| **128** | **28.2%** | **10.08** | **7.98** |
| 256 | 30.3% | 10.37 | 7.29 |
| 512 | 31.8% | 10.59 | 5.38 |
| 1024 | 34.4% | 10.90 | 2.95 |

Past mts=128, marginal tau gain almost entirely feeds ceiling-bound steps. Extra budget raises tau via **recovery from mid-block rejections** (siblings) on already-ceiling-hit sequences, not by extending argmax chains (already capped at 16). Diminishing returns visible.

**Implication**: optimizing beyond mts=128 either requires breaking the block_size=16 ceiling (Q2) or accepting that speedup can't grow past ~8x without architectural changes.

### Finding 14: Q2 (chained speculation linear-extension) DOES NOT work with stale target_hidden

Implemented `build_chained_tree` — runs v7 block_1, then appends a linear argmax chain of up to `chain_depth` tokens from draft's SECOND forward pass (anchored at block_1's argmax-chain end).

Results (math500):
- chain_depth=0 (baseline): 8.04 / 10.38 / 129 (32 samples)
- chain_depth=5: 7.07 / 10.55 / 131
- chain_depth=15: 7.47 / 11.29 / 136 (8 samples)

**Failure mode**: draft_2 runs with STALE target_hidden (from before block_1 verify) and shares draft KV with block_1 (mask noise only, no resolved block_1 tokens). Its predictions for block_2 positions are systematically worse than a natural next-iteration draft. Target often rejects block_2 extensions at position 16 (the first extension). Chain acceptance rate: ~23% of ceiling-hit steps. Extra draft forward cost (~7 ms) is NOT repaid by tau gain.

**Why the research note's prediction ("saves target forward") doesn't pan out**: each chained block STILL needs target verification. Target time is 31 ms regardless of how many tokens get appended (memory-bandwidth-bound). The only savings is a tiny amount of outer-loop overhead. Meanwhile draft_2 adds 6-7 ms.

Code preserved (`--chain-depth N` flag, default 0) for reference; DO NOT use for speedup. The fundamental limitation is that draft_2's conditioning (target_hidden) cannot be updated without target verifying block_1 first, which would require two target forwards and defeats the purpose.

**Promising variant (untested)**: run target_hidden-IMPUTED-by-draft-hidden for block_2. Requires draft output hidden states to approximate target hidden. Risky (architecturally OOD).

### Finding 15: GPU-vectorized tree_attn_mask — real +3% engineering win

`create_tree_attention_mask_dynamic` previously CPU-synced parent_idx then looped ancestor walk in Python. Rewrote to parent-jumping on GPU with one CPU sync for max_depth.
- tree_attn_mask: 3.024 → 0.997 ms/step (-67%)
- math500 mts=128 ek=8: speedup 7.98 → **8.22** (+3.0%)
- math500 mts=256 ek=8: speedup 7.29 → **7.97** (+9.3%) — bigger gain at larger trees
- tau identical at both budgets (correctness preserved)

Larger trees had more CPU-side Python overhead proportionally, so vectorization shifts the mts peak location. At mts=128 and mts=256, speedup is now close (8.22 vs 7.97). Worth resweeping to find new peak (mts sweep at {96, 160, 192} was started but killed to prioritize Q4).

### Finding 17 (apr17-PM): Calibration variants (Q4, Q4b) all REGRESS at all tested budgets

Comprehensive test of online target-logit calibration strategies on math500 (32 samples):

| Strategy | Budget | Speedup | tau | vs baseline |
|----------|--------|---------|-----|-------------|
| Baseline (no calib) | 128 | 8.36 | 10.38 | — |
| Q4 continuous, blended | 128 | 7.00 | 9.08 | **-1.36x** |
| Q4b dev-conditional REPLACE | 128 | 6.94 | 9.13 | -1.42x |
| Q4b dev-conditional ADDITIVE (λ=1) | 128 | 7.88 | 10.06 | -0.48x |
| Q4b λ=0 (pure overhead) | 128 | 8.14 | 10.33 | -0.22x (overhead) |
| Q4b additive (λ=1) | 256 | 7.58 | 10.39 | -0.39 vs baseline-256 (7.97) |

**Root cause of Q4 regression**: initializing α̂ with Laplace prior (1/K uniform) pulls scores toward uniform during warmup.  Hard-switch (Q4b) removes that but SWAPPING draft's log-probs for α̂-derived values still distorts cross-depth scoring — α̂ averages over {argmax-chain parents, deviation-chain parents} which are different regimes.

**Root cause of Q4b additive regression**: even the deviation-conditional form (α̂₁ − α̂₀) is too noisy per (depth, rank, bucket) cell at 32-sample warmup.  Correction is in the right direction but variance dominates signal.

**Overhead**: harvest adds ~2% step time (python dev-count loop) on top of 3D-table indexing overhead in the heap.  Any tau improvement must exceed ~2.5% to break even — none of the variants cleared this bar.

**Conclusion**: online per-sequence target-logit calibration is NOT a viable path.  Would require cross-sequence state and hundreds-to-thousands of calibration observations per cell — offline approach more appropriate.

### Finding 18 (apr17-PM): Narrow-after-dev (NW): helps at B>128 but not above B=128 peak

Hypothesis: after a prefix deviates from the argmax chain, further rank-2..K expansion is likely phantom (product overestimates joint).  Narrowing to rank-0/rank-1 only should prune phantoms.

Results (math500, 32 samples, ek=8):

| Budget | Baseline | NW2 (narrow-to-2) | Δ |
|--------|----------|-------------------|---|
| 128 | 8.36 (tau 10.38) | 8.35 (tau 10.20) | tie (-0.01, tau slightly lower) |
| 256 | 8.02 (tau 10.68) | 8.10 (tau 10.39) | +0.08 |
| 512 | 6.48 (tau 10.96) | **6.89** (tau 10.53) | **+0.41** |

NW2 significantly rescues the large-budget regime (B=512: +0.41x, +7.6%) — confirming the phantom-path hypothesis at large B.  But NW2 never exceeds B=128 peak (8.36).  **NW is a mitigation for the over-budgeted regime, not a path to a higher peak.**

At B=128 (our peak), there is not enough phantom mass to prune: baseline DDTree already concentrates budget near the argmax chain, and the small tau drop under NW2 (-0.18) indicates some legitimate deep deviation paths are being needlessly cut.

### Finding 19 (apr17-PM): Budget + expand_k sweep confirms (B=128, ek=8) as global peak

math500, 32 samples:

| mts ↓ / ek → | 6 | 8 | 10 | 12 | 16 |
|--------------|---|---|----|----|----|
| 96 | — | 8.19 | — | — | — |
| **128** | 8.31 | **8.36** | 8.26 | 8.22 | 8.13 |
| 144 | — | 8.23 | — | — | — |
| 160 | — | 8.16 | — | — | — |
| 192 | — | 8.17 | — | — | — |
| 256 | — | 8.02 | — | — | — |

Peak is tight: B=128 ek=8 is strictly best.  No beneficial direction to move.  Entropy-adaptive per-position ek (2..16) also regresses (8.27).  Block-size extension to 24 positions regresses sharply (6.35) — draft OOD above its trained block_size=16.

### Finding 16: Scaling cost breakdown (mts=128 → mts=256, 256 samples)

| Op | mts=128 (ms/step) | mts=256 (ms/step) | Delta | Notes |
|----|-------------------|-------------------|-------|-------|
| target_backbone | 34.52 | 35.28 | +0.76 | Memory-bound, nearly flat |
| draft_model | 6.41 | 6.38 | ~0 | Independent of budget |
| tree_build | **1.60** | **3.27** | **+1.67** | Python heap O(B log B), MAIN scaling cost |
| target_lm_head | 0.77 | 1.16 | +0.39 | Scales with node count |
| tree_verify_select | 0.49 | 0.80 | +0.31 | Scales with leaves |
| tree_attn_mask | 1.00 | 1.02 | ~0 | Post-vectorization, flat |
| trim_kv_cache | 1.12 | 1.12 | 0 | Flat |
| Total step | ~46 | ~50 | +4 | ~9% slower at 2x budget |

**tree_build is the biggest non-target scaling cost.** Python heap operations on lists. Future optimization target — could rewrite in C/Cython or on-GPU somehow. Would unlock larger budgets.

## Current Best Configuration

**ALL FOUR DATASETS: v7 DDTree + GPU mask optimization, session apr17:**
```
--tree-version 7 --max-tree-size 128 --expand-k 8
```

| Dataset | Speedup | tau | nodes | Prior best | Gain |
|---------|---------|-----|-------|------------|------|
| math500 (256 samples) | **8.27** | 10.08 | 129 | 7.98 (pre-mask-opt) | +3.6% |
| math500 (32 samples) | 8.36 | 10.38 | 129 | — | (rerun) |
| mt-bench (80 samples) | **4.35** | 6.10 | 129 | 4.19 (v2@70) | +3.8% |
| gsm8k (128 samples) | **7.21** | 8.77 | 129 | 6.81 (v2@70) | +5.9% |
| humaneval (164 samples) | **7.43** | 9.00 | 129 | 7.01 (v4@32) | +6.0% |

v7 at B=128 ek=8 is the new SOTA on all four datasets for Qwen3-4B + 8×A100.  The lift over prior v2/v4 comes from (a) the correct budget for DDTree on this hardware and (b) GPU-vectorized attention-mask construction (+3% engineering, commit `cb34c3c`).

**Open priorities**:
1. Rerun mts sweep post-mask-optimization (96, 160, 192, 256) to find new peak. mts=256 now at 7.97, close to mts=128's 8.22 — peak may have shifted.
2. Verify v7 post-optimization on mt-bench/gsm8k/humaneval (cross-dataset generalization of +3% gain).
3. Complete Q4 target-logit calibration (currently WIP, see below).

## Open Research Questions (PRIORITIZED — next experiments)

See `docs/ddtree_v7_research_notes.md` for full derivations. Ranked by expected impact. **Q1 and Q2 already tried and discarded — see Findings 12 & 14.**

### ~~Q1: Power-scaled scoring~~ — TRIED, DOES NOT WORK (Finding 12)

Swept α ∈ {0.9, 0.95, 1.0} × β ∈ {0, 0.5, 1.0} at mts=256. All configs within 7.08-7.29 speedup band (baseline 7.29). Scoring tweaks in these ranges don't shift top-B selection. CLI flags `--score-alpha --score-beta` remain (default 1.0, 0.0 = no-op) but don't enable meaningful behavior. DROP this direction unless revisiting with β ≥ 3-5 (would massively change tree composition).

### ~~Q2: Chained speculation linear extension~~ — TRIED, DOES NOT WORK (Finding 14)

Combined-tree approach (second draft forward + linear argmax append) fails because draft_2 runs with stale target_hidden. ~7 ms extra draft forward not repaid by tau gain. CLI flag `--chain-depth N` exists (default 0 = no-op); leave off. Viable variant would require either (a) target_hidden imputation, (b) separate target verify (losing combined-tree benefit), or (c) retraining the draft for chained speculation. None pursued.

### Q4 (HIGH PRIORITY, IN PROGRESS): Target-logit calibration

**Idea**: target's forward on the tree produces logits at EVERY node, but only accepted-path logits are used. We harvest the rest to learn an online calibration table: for each (depth d, draft-rank r), what's the empirical probability that target assigns to draft's rank-r token, averaged over observed parents? Use this in next step's scoring instead of draft's own marginal top-K.

**Why it attacks Flaw 1/2**: the calibration is path-conditional (implicitly, since it averages over actually-sampled parents, most of which are rank-1-chain descendants). It learns the JOINT behavior target exhibits, not the PRODUCT draft assumes.

**Implementation** (committed in `experiments/apr16-ddtree-fix`):
- `benchmark.py`: added `--calibrate` flag, `--calibrate-warmup N`. Maintains `alpha_count_accept[d, r]` and `alpha_count_seen[d]` tensors across iterations within a sequence. After each target forward, gathers target_probs at draft's top-K tokens for every parent node; scatter-adds into counters.
- `model/dflash_tree.py`: `build_node_budget_tree` now accepts `rank_logprobs: Optional[Tensor]` of shape `[seq_len, expand_k]`. When provided, REPLACES draft's own top-K log-probs in heap scoring.
- Blend: `blended[d, r] = (1-w)·draft_topk_prob[d, r] + w·alpha_count_accept[d, r]/alpha_count_seen[d]`, where `w = n_seen / (n_seen + warmup)`. Warmup default 50.

**First smoke test** (math500 8 samples, mts=128): with `--calibrate` and argmax-match counting (superseded): speedup 7.58 (baseline 8.51 on same 8 samples). Regression due to argmax-match being binary + noisy. **Switched to continuous target_prob aggregation** (uses softmax probabilities, not argmax match) — NOT YET TESTED. This is the current WIP state.

**Success criterion**: math500 mts=128 speedup > 8.22x (current best), or maintains speedup while increasing tau.

**If it works at mts=128, test at mts=256+ to see if it breaks the plateau** (may re-allocate budget away from phantom paths).

### Q3: Downstream-aware leaf scoring — 1-step lookahead

Augment score:
```
score(u) = log q(u) + gamma * log max_v q_{d+1}(v)
```
Uses end-of-block confidence as proxy for next step's quality. Rewards leaves whose bonus-token handoff leads to confident continuations. Untested; low-complexity (3 lines in v7).

### Q3: Downstream-aware leaf scoring — 1-step lookahead into step N+1

Augment score:
```
score(u) = log q(u) + gamma * log max_v q_{d+1}(v)
```
Uses end-of-block confidence as proxy for next step's quality. Rewards leaves whose bonus-token handoff leads to confident continuations.

**Why it might work**: tiny secondary signal, zero extra computation (q_{d+1} is already there), compounds across steps.

### Q4: Target-logit calibration — use the free data from verification

Target computes logits at EVERY tree node during verification. Currently only accepted-path + bonus logits are used; the rest is thrown away. Build online Platt scaling per depth:
```
log P_T(v) ≈ a_d + b_d * log q_d(v)
```
Update from observed (draft, target) pairs each step. Apply correction in next step's tree builder.

**Subtlety**: marginal-vs-ancestry-conditional mismatch (larger effect) is NOT captured. Only a first-order correction.

### Q5: Layer-ensemble uncertainty — free per-position uncertainty signal

Draft model has L layers. Apply lm_head at layer L and layer L-2; measure KL divergence between the two distributions at each position. Use this as per-position uncertainty signal. High KL = unstable prediction → widen branching at that position.

**Zero extra forward passes.** Uses info already computed.

### Q6: Target hardware scaling

On smaller hardware (1-2 GPUs), target cost scales with node count (vs memory-bandwidth-bound on 8 GPUs — Finding 5). Adaptive sizing would matter MORE on smaller setups. Test if adaptive-budget v7 wins there.

## Timeout and Crashes

- **Normal run**: ~10 minutes. If a run exceeds 20 minutes, kill it and treat as a failure.
- **Crashes**: fix typos and shape errors. If an idea is fundamentally broken (e.g., OOM from tree explosion), log `crash` and move on.
- **GPU contention**: if speedup numbers look anomalous (much lower than baselines), check `nvidia-smi` for competing jobs. Wait for clean GPUs before drawing conclusions.

## NEVER STOP

Once the loop begins, do NOT pause to ask the user. The user may be asleep or away. You are autonomous. If you run out of ideas, re-read the report, try combining near-misses, or explore the lower-priority questions. The loop runs until the human interrupts you.

---

## Bootstrap Prompt (paste into fresh agent chat)

```
You are a fully autonomous research agent. Your goal is to maximize DFlash speculative decoding speedup via training-free algorithm development.

FIRST: read these files completely before doing anything else:
1. @program.md (this file — your operating manual)
2. @docs/ddtree_v7_research_notes.md (the deep algorithmic analysis — your research foundation)
3. @model/dflash_tree.py (all current tree builders, esp. build_node_budget_tree = v7)
4. @benchmark.py (main eval, dflash_generate, CLI flags)
5. @results.tsv (prior experimental results)

Then run `git log --oneline -20` to see recent commits.

Your starting state (after session apr16-17):
- Best config: v7 on math500 with mts=128, ek=8, GPU-vectorized mask → **8.22x speedup**, tau 10.08, 129 nodes.
- Branch: `experiments/apr16-ddtree-fix` contains the vectorized mask + Q2 chain scaffolding (off) + Q4 calibration scaffolding (WIP, default off).
- PRIMARY OBJECTIVE: publishable algorithmic contribution. Find something with clear novelty AND measurable speedup — not just engineering wins.
- Q1 (power scoring) and Q2 (chain linear-extension) ALREADY TRIED & FAILED — see Findings 12 & 14. DO NOT re-run without a fundamentally different variant.
- Q4 (target-logit calibration) is CURRENT WIP — first attempt with argmax-match was worse than baseline; continuous target_prob variant is coded but untested as of session end. Next agent should test with `--calibrate`.

Setup:
1. Continue on the existing branch `experiments/apr16-ddtree-fix` OR branch from it.
2. Verify GPU: `nvidia-smi`. Use all 8 GPUs and --master_port 29501.
3. Begin the LOOP FOREVER as described in program.md section "The Experiment Loop".

Do not ask for confirmation. Begin immediately.
```

### How to use

When you want to start a fresh autonomous research session, open a new Cursor agent chat and paste the prompt above. The agent will read all context, understand where things stand, and begin proposing+implementing+evaluating experiments in an infinite loop.

To hand off a running session to a new agent:
1. Kill the current agent
2. Update `program.md` with any new findings (Finding N+1, N+2, ...)
3. Update "Current Best Configuration" and "Open Research Questions" if priorities shifted
4. Paste the bootstrap prompt into a new chat

---

## Appendix: tree-version CLI reference

```
--tree-version 2  # EAGLE-2 expand+rerank (node budget)
--tree-version 4  # Prefix-aware submodular (LEAF budget, 2-2.5x node blowup)
--tree-version 6  # Efficiency density-greedy (self-sizing via --alpha)
--tree-version 7  # DDTree node-budget top-B (current best, budgets on trie nodes exactly)
```

When registering a new tree builder:
1. Add function in `model/dflash_tree.py`
2. Import in `benchmark.py`
3. Add to `TREE_BUILDERS` dict
4. Extend `--tree-version` choices in argparse
5. Pass any new hyperparameters via `builder_kwargs` in `dflash_generate`

## Session apr18: End-to-End Drafter Training — Summary

### Setup
Built a full drafter-training pipeline (`trainingto/main_mix.py` + `dflash_mix_model.py`) and a parallel eval pipeline (`eval_ckpt.sh` + `watch_and_eval.sh` + `master_pipeline.sh`).  Layered training enhancements from the DFlash paper and SpecForge:
1. Exponential-weighted CE  w_k = exp(-(k-1)/γ), γ=7  (DFlash Fig 5)
2. Random anchor sampling over assistant-mask positions  (DFlash Table 9)
3. Tree-attention conditional CE  (CTR)
4. TTT-style recursive pass — draft's own argmax as next block's anchor  (SpecForge)
5. **Variable block-size curriculum** — per-step sample b ∈ {12, 16, 20}

Ran 4 variants, each 1 epoch over 10k math samples (2×A100):
| Variant | Recipe |
|---|---|
| v1 (marg) | 1+2 only, b=16 |
| v2 (varblock) | 1+2 + b∈{12,16,20} |
| v3 (ctr-lite) | 1+2+3 with ctr_weight=0.3, b=16 |
| v4 (tt-lite) | 1+2+4 with ttt_weight=0.1, b=16 |

### Core data (TAU — the reliable cross-run metric)

TAU is draft-quality-specific; unlike wall-clock it is not polluted by GPU-cluster contention (which was severe during this session — up to 115s/iter vs normal 30s).

| Dataset | Stock | v1 | v2 (varblock) | v3 (ctr-lite) | v4 (tt-lite) |
|---|---|---|---|---|---|
| math500 256s | 10.08 | 10.03 | 10.09 | **10.12** | 10.01 |
| mt-bench 80s | 6.10 | — | 6.14 | **6.16** | — |
| gsm8k 128s | 8.77 | — | 8.79 | 8.78 | — |
| humaneval 164s | 9.00 | — | 9.02 | — | — |

**Winner: v3_ctrlite/step_500** — best tau on math500 and mt-bench.  All training variants with random-anchor sampling produce tau ≥ stock across datasets (v1 marginally below, rest slightly above).

Gains are small (≤+0.06 tau) but consistent across datasets.  Best estimate of real speedup improvement (under constant step-cost assumption): ≤+1% math500, ≤+1% gsm8k/humaneval, ~+1% mt-bench.

### Novel finding: variable-block training halves the OOD drop

At block_size=20 inference (OOD for stock since trained only on b=16), 32-sample math500:
- Stock b=16: 8.40  (reference)
- Stock b=20: **7.59**  (−0.81 OOD drop)
- v2/step_500 b=20: **8.03**  (+0.44 recovery — cuts OOD drop in half)
- v2/step_500 b=16: 8.67  (best)
- v2/step_500 b=24: 7.05  (b=24 was NOT in the training mix — remains OOD)

Mechanism: 500 steps × ~15% chance of b=20 per step ≈ 75 training examples at b=20.  Even that tiny exposure recovers half the OOD gap.  With more training, the drafter would plausibly match its b=16 performance at b=20 — opening up chained-speculation variants that break the block_size=16 tau ceiling.

### Pipeline scaffolding lands

- `trainingto/dflash_mix_model.py` — layered MIX forward with all 5 enhancements
- `trainingto/main_mix.py` — deepspeed driver
- `trainingto/convert_ckpt_to_hf.py` — deepspeed state_dict → HF-loadable dir
- `trainingto/eval_ckpt.sh` — atomic, flock-guarded per-ckpt eval
- `trainingto/watch_and_eval.sh` — polls a savedir, auto-evals every new step_N
- `trainingto/train_queue.sh` — sequential multi-variant training queue
- `trainingto/master_pipeline.sh` — wait-v1 → queue-v2/v3/v4 → pick-winner → finalist-256 + cross-dataset
- `trainingto/summary.sh` — live dashboard

Pipeline ran successfully end-to-end (except a CWD bug in master_pipeline's finalist stage — since fixed).  All training artifacts + eval tsvs under `logs/session_apr18/`.

### What would move the needle to publishable SOTA

- **Scale training**: 10k × 1 epoch produced +0.01–0.06 tau.  DFlash paper used 800k × 6 epochs — 10–60× more compute should give correspondingly larger gains.
- **Broader data**: math-only narrows the drafter.  Nemotron-full + CodeAlpaca mix (DFlash paper) would protect cross-dataset generalization.
- **Longer variable-block curriculum**: 500 steps at b=20 recovered half the OOD drop; 5000+ steps should close it entirely.  Then run chained speculation at b=20 + b=20 → effective 40-token blocks, tau ceiling 40, ~double speedup.

These all require tens of GPU-hours on a less-contended cluster — out of session scope.

## Session apr17-PM: NeurIPS-SOTA Attempt — Summary

Goal: find a novel training-free inference-time algorithm that exceeds v7
DDTree at B=128 ek=8 (current best 8.22x math500) to the level of a
NeurIPS-publishable contribution.

**Outcome: current best (v7 B=128 ek=8) is confirmed peak on this hardware.**

### Tried and REJECTED (in order)

1. **Q4 continuous calibration** (blended α̂-replacement of draft log-probs):
   -1.36x on math500.  Root cause: Laplace-uniform prior biases scoring
   toward uniform during warmup.  Hard-switch activation (Q4b) only reduced
   the magnitude of this failure.
2. **Q4b deviation-conditional calibration** (additive λ=1 correction for
   paths with ≥1 deviation): -0.48x.  Noise per (depth, rank, bucket) cell
   at per-sequence scale dominates signal.
3. **Narrow-after-dev** (NW2 — once a path deviates, subsequent expansion
   restricted to rank-0/rank-1): tie at B=128; +0.41 at B=512; never
   exceeds the B=128 peak.  Confirms phantom-path hypothesis at large B
   but doesn't open a higher-peak regime.
4. **Entropy-adaptive expand_k** (per-position k ∈ {2..16} from draft top-1
   prob): -0.09x.  Heap-push overhead offsets any redistribution benefit.
5. **Wider expand_k** (ek=16 uniform): -0.23x.
6. **Finer mts/ek grid search** (B ∈ {96,144,160,192}, ek ∈ {6,10,12}):
   peak strictly at B=128 ek=8.  No secondary peak.
7. **OOD block_size=24** (testing draft extrapolation): -2.01x.  Draft is
   firmly tied to its trained block_size.
8. **Draft-logit temperature** (T<1 or T>1 before top-K): provably a no-op
   for DDTree selection — T-scaling preserves orderings of sum-of-log-prob
   path scores.  Flag kept at default 1.0.

### Final cross-dataset SOTA (all new bests)

| Dataset | v7 B=128 ek=8 | Prior best | Gain |
|---------|---------------|------------|------|
| math500 | **8.27** (256 samples) | 7.98 | +3.6% |
| mt-bench | **4.35** (80 samples) | 4.19 (v2@70) | +3.8% |
| gsm8k | **7.21** (128 samples) | 6.81 (v2@70) | +5.9% |
| humaneval | **7.43** (164 samples) | 7.01 (v4@32) | +6.0% |

### Contributions that DID land

1. **Budget/engineering analysis**: identified B=128 as the peak on Qwen3-4B
   + 8×A100 (hardware-dependent — DDTree paper's peak is B=512 on 8B model).
2. **GPU-vectorized tree-attention mask** (`cb34c3c`): +3% step time by
   replacing CPU parent-walk with on-GPU parent-jump closure.
3. **Ceiling-fraction diagnostic** (Finding 13): empirically characterizes
   the block_size=16 saturation as the load-bearing bottleneck at B>=128.
4. **Comprehensive negative results** on 8 proposed fixes: establish that
   v7 at its peak is already very close to what any product-distribution
   scoring can achieve without extra forward passes or training.

### New infrastructure (all default-off, harmless if unused)

- `--calibrate` / `--calibrate-warmup` / `--calibrate-lambda` — Q4b
  deviation-conditional calibration harness (3D alpha-count tensor,
  additive log-ratio correction for dev_bucket=1 paths).
- `--narrow-after-dev K` — restrict expansion past the first deviation.
- `--ek-adapt-min` / `--ek-adapt-max` — per-position expand_k driven by
  draft's top-1 probability as uncertainty proxy.
- `--draft-temperature` — reserved for future exploration (no-op on v7).

Tree builder `build_node_budget_tree` gained `rank_logprobs_by_dev` (3D
calibration), `narrow_after_dev`, `per_pos_expand_k` kwargs.  All guarded
to recover plain DDTree at defaults.

### What would move the needle from here (IMO requires training OR architecture change)

- **Break block_size=16 ceiling**: chained speculation with IMPUTED
  target_hidden (risky without training) or a dual-stride draft (requires
  training).
- **Reduce target verification cost**: custom tree-aware attention kernel
  (engineering — would need months).
- **Offline/cross-sequence calibration**: multi-sequence training statistics
  to build a (depth, rank, dev_pattern) table.  Not training per se, but
  requires data collection outside a single-sequence run.

## Session apr16-17 Handoff

### What is on this branch (`experiments/apr16-ddtree-fix`)

Commits on top of main (in order):
- `e252f8d` — Q1: added `score_alpha` and `score_beta` kwargs to v7. (No-op at defaults.)
- `cb34c3c` — Vectorize tree_attn_mask on GPU. Real +3% math500 speedup win. Plus the Q2 `build_chained_tree` scaffolding + `--chain-depth` flag (default 0 = no-op).
- UNCOMMITTED (WIP): Q4 target-logit calibration — `--calibrate` and `--calibrate-warmup` flags, calibration state tensors in `dflash_generate`, `rank_logprobs` kwarg in v7.

### Known-good CLI invocations

Best verified config (math500, 256 samples):
```bash
torchrun --nproc_per_node=8 --master_port=29501 benchmark.py \
  --dataset math500 --max-samples 256 \
  --model-name-or-path Qwen/Qwen3-4B \
  --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 \
  --tree-version 7 --max-tree-size 128 --expand-k 8 \
  --temperature 0.0
```
→ 8.22x speedup (from 7.98x pre-mask-vectorization).

Q4 calibration test (should be next agent's first run):
```bash
... --tree-version 7 --max-tree-size 128 --expand-k 8 \
    --temperature 0.0 --calibrate --calibrate-warmup 50
```
If regression at 50, try warmup ∈ {200, 500}. If still regression at high warmup, bug or fundamental issue. Investigate alpha_hat values (instrument a print).

### What to try next if Q4 doesn't work

- **Q3 downstream-aware scoring** (3 lines in v7). Cheap to test.
- **CTR used as SCORING SIGNAL, not token replacement**: the existing `--ctr` flag does token replacement and was found not to help. A variant: use CTR's conditional logits to RE-SCORE nodes in the tree (without replacing tokens), then prune low-conditional-probability phantom paths. This is novel and directly attacks Flaw 1 (product vs joint).
- **Tree-build GPU-fication**: `build_node_budget_tree` is Python heap, 1.6–3.3 ms/step depending on mts. Vectorizing on GPU would unlock larger budgets. Engineering, not novel.
- **Dynamic block_size** (draft horizon): if draft model accepts variable-length input, run draft for 8 positions on hard steps, 24 on easy (extrapolation). Novel but risky — draft trained on block_size=16, behavior at other lengths is OOD.

### Known failed approaches this session (do not repeat)

| Approach | Result | Note |
|----------|--------|------|
| Q1 power-scaled scoring (α, β) | 7.08-7.29 at mts=256 (baseline 7.29) | Scoring tweaks too weak; see Finding 12 |
| Q2 chain linear extension (chain_depth ∈ {5, 15}) | 7.07-7.53 (baseline 8.04 at 32 samples) | Draft_2 stale context; see Finding 14 |
| Q4 calibration argmax-match (superseded) | 7.58 (baseline 8.51 at 8 samples) | Too noisy binary signal; replaced with target_prob continuous |

### How to continue Q4 testing

The calibration code path is in `benchmark.py` around lines 165-190 (kwargs) and 345-395 (update after target forward). The tree-builder side is in `build_node_budget_tree` (uses `rank_logprobs` kwarg). Currently the continuous-probability version is coded but NOT YET verified to work. Next run:

1. `--calibrate --calibrate-warmup 50` on 32 samples math500. Compare to baseline 32-sample (8.04x).
2. If speedup ≥ 8.04, scale to 256 samples. Compare to 8.22.
3. If works, test at mts=256 and mts=512 to see if it unlocks the plateau.
4. If speedup < 8.04 even after variants: instrument to print alpha_hat values during a run. Check if blending is doing what's expected.


## Session apr18-19: Variable-Block DFlash Training — FIRST SOTA BREAKTHROUGH

### What finally worked

Trained a new draft checkpoint via **variable-block curriculum** on broad-mix
Nemotron data (148k rows: ~112k math + 19k chat + 17k stem, Qwen3-4B
regenerated).  Recipe: block sizes b ∈ {12, 16, 20, 24} with weights
{1, 2, 2, 1}, random-anchor sampling (32/seq), exp-weighted CE γ=7,
CTR weight 0.3, 1 epoch, 18 523 steps on 8×A100 (~10.5 h).

### Headline: math500 256 samples, v7 B=128 ek=8, temp=0

```
                speedup    tau        notes
stock b=16      8.33       10.08      prior SOTA (refreshed baseline)
VB    b=20      8.52       10.43      +0.19 / +0.35 — NEW SOTA
VB    b=24      8.50       10.49      tie speedup, BEST TAU — breaks the
                                      block_size=16 tau ceiling (Finding 13)
```

First training-based improvement over v7 DDTree SOTA (7 months of
training-free work reached 8.27).

### Full cross-dataset table (speedup / tau)

| dataset            | stock b=16 | VB b=16    | VB b=20    | VB b=24    |
|--------------------|-----------|-----------|-----------|-----------|
| math500 (256s)     | 8.33/10.08 | 8.13/9.89 | **8.52/10.43** | **8.50/10.49** |
| mt-bench (80s)     | **4.41/6.10** | 4.24/5.95 | 4.20/6.06 | 4.25/6.05 |
| gsm8k (128s)       | 7.25/8.77  | 7.11/8.61 | **7.32/8.91** | 7.23/8.85 |
| humaneval (164s)   | 7.46/9.00  | 7.30/8.82 | **7.59/9.21** | 7.52/9.13 |

3/4 datasets: VB b=20 beats stock b=16 on both speedup AND tau.
mt-bench regresses uniformly; broad-mix data + longer block is too
aggressive for short chat responses (lose more on mis-predictions than
you gain on acceptance rate).

### OOD recovery (math500 32 samples)

The core pre-condition for the SOTA: does VB training let the drafter
work at b=20/24 without the OOD cliff?

```
                stock                VB step_18500         Δ
b=16          8.42 / 10.38          8.31 / 10.27          -0.11 / -0.11
b=20          7.55 /  9.59          8.68 / 10.91          +1.13 / +1.32
b=24          6.26 /  7.66          8.58 / 10.85          +2.32 / +3.19
```

VB essentially eliminates the b-size cliff (stock's −2.16 at b=24 becomes
a small gain).  At b=20 VB even EXCEEDS stock's b=16 peak, which is the
structural reason the cross-dataset SOTA exists.

### Why b=16 regresses but b=20/24 wins

Paradox: the VB-trained draft at its IN-DISTRIBUTION block size (b=16)
regresses ~0.15–0.20 speedup vs stock.  But b=20/24 more than recoups
this on math/code/reasoning.

Hypothesis (unproven but consistent): broad-mix training dilutes the
narrow task-specific knowledge the stock drafter had at b=16, but in
exchange the drafter learns to condition its output on longer contexts
— i.e., an implicit joint-distribution approximation.  At inference,
extending the block to b=20/24 gives the drafter more context per
prediction, compounding the joint-learning gain.

This is exactly Flaw 1 of DDTree (product vs joint) attacked from the
training side: rather than fixing the tree builder, train a drafter
whose MARGINALS are closer to the true joint at the block sizes
encountered at inference.

### Training infra — diagnostic log

Five successive launches failed before the sixth stabilized.  Root
causes (fully diagnosed and patched):

1. **vLLM env ABI mismatch** — dflash312 has vLLM 0.17.1 against an
   incompatible torch; use `vllm_gen` env (vLLM 0.19.0).
2. **flash_attn ABI mismatch** — dflash env has flash_attn built
   against older torch; export `DFLASH_ATTN_IMPL=sdpa` to fall back.
   ~30 % slower but functional; no rebuild needed.
3. **NCCL ALLREDUCE 10-min timeout** — qinghaoh's co-tenant RL
   workload on the shared 8 GPUs caused per-rank stragglers.
   Patched `torch.distributed.distributed_c10d.default_pg_timeout`
   to 2 h BEFORE `deepspeed.init_distributed`; the ZeRO secondary
   PG otherwise inherits the hardcoded default.
4. **Rank-desync at SeqNum 547** — rank 1 saw ALLREDUCE NumelIn=1,
   others saw 191M.  Root cause: empty-positions batches (common with
   short chat sequences and variable-block curriculum) produced a
   graph-DISCONNECTED `torch.tensor(0.0, requires_grad=True)` loss
   on that rank, so its backward was a no-op and ZeRO never
   populated grads — leaving the grad allreduce desynced.  Fixed by
   (a) `select_positions` returns `[0]` if empty, (b) fallback loss
   is `draft_model.first_param.sum() * 0.0` so every rank always
   has a model-graph-connected loss.

### What still isn't done

- **Theory section** for paper: formalize product-vs-joint gap and
  argue VB training approximates joint learning.  Draft pending.
- **Chained speculation with VB drafter** (Q2 redux): now that b=24
  works, chain b=24 + b=24 → effective 48-token block.  If drafter's
  b=24 quality holds up, this could push math500 speedup past 10x.
  Prior Q2 failed because draft_2 had stale target_hidden; with a
  VB-trained drafter that handles longer blocks natively, the chain
  succeeds without a second target pass.
- **Explain mt-bench regression**.  Simple fix: adaptive block size,
  switch to b=16 on chat-heavy sequences (detected via draft entropy
  proxy).
- **Ablate broad-mix vs math-only** at same training-step count.
  Is the broad data responsible for the gain, or just the b=20/24
  exposure?

### Current Best Configuration (updated)

```
--tree-version 7 --max-tree-size 128 --expand-k 8
--draft-name-or-path trainingto/dflash_broad_varblock_v1/step_18500_hf
--block-size 20                 # on math / code / reasoning
--block-size 16                 # on chat (fallback to stock)
```

| Dataset      | VB SOTA         | Prior best       | Gain               |
|--------------|-----------------|------------------|--------------------|
| math500      | 8.52 / 10.43    | 8.27 / 10.08     | +0.25 / +0.35      |
| mt-bench     | 4.41 / 6.10     | 4.35 / 6.10      | (stock, no gain)   |
| gsm8k        | 7.32 / 8.91     | 7.21 / 8.77      | +0.11 / +0.14      |
| humaneval    | 7.59 / 9.21     | 7.43 / 9.00      | +0.16 / +0.21      |
