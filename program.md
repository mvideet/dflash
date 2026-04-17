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

**math500** (v7 DDTree + GPU mask optimization, session apr17):
```
--tree-version 7 --max-tree-size 128 --expand-k 8
```
Speedup: **8.22**, tau: 10.08, nodes: 129 (math500, 256 samples, temp=0.0)

Improvement comes from vectorized `create_tree_attention_mask_dynamic` committed in `cb34c3c`. Baseline v7 pre-optimization was 7.98x.

**mt-bench** (prior best — v7 post-mask-optimization untested):
```
--tree-version 2 --max-tree-size 70 --expand-k 8
```
Speedup: 4.19, tau: 5.93, nodes: 71 (mt-bench, 80 samples, temp=0.0)

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

