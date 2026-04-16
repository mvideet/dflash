# DFlash Autonomous Experiment Program

This is an experiment harness for autonomous AI-driven research on DFlash speculative decoding, inspired by [autoresearch](https://github.com/karpathy/autoresearch).

## Setup

To set up a new experiment session, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date and focus area (e.g. `apr9-ewma-ek`). The branch `experiments/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b experiments/<tag>` from current main.
3. **Read the in-scope files** for full context:
   - `README.md` — project overview, models, usage.
   - `benchmark.py` — main eval script with `dflash_generate` and CLI flags. **Primary edit target.**
   - `model/dflash_tree.py` — tree builders v2, v4 (EAGLE-2, prefix-aware greedy). **Secondary edit target.** v1/v3 exist in the file but are not wired into benchmark.py.
   - `model/dflash.py` — draft model wrapper. Read-only unless experimenting with draft utilization.
   - `model/utils.py`, `model/freq_vocab.py` — utilities. Read-only.
   - `docs/adaptive_tree_sizing_report.md` — full technical report on prior adaptive experiments. **Essential reading before proposing ideas.**
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

| Version | Function | Algorithm |
|---------|----------|-----------|
| v2 | `build_dynamic_tree_v2` | EAGLE-2 expand + rerank by cumulative confidence |
| v4 | `build_prefixaware_tree` | Prefix-aware greedy with (1-1/e) submodular guarantee |

v4 is the current default and best performer. It operates in two phases:
- **Phase 1 — Expand**: best-first search generates a candidate pool (~3x budget)
- **Phase 2 — Select**: lazy-greedy submodular selection maximizing E[tau]

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

1. **Review state**: check current branch, results.tsv, and recent experiment outcomes.
2. **Propose an idea**: based on the open questions, prior results, and your understanding of the codebase. Write a brief hypothesis.
3. **Implement the change**: edit `benchmark.py` and/or `model/dflash_tree.py`.
4. **git commit** the change.
5. **Run the experiment**: `torchrun ... benchmark.py ... > run.log 2>&1`
6. **Extract results**: grep for speedup, tau, avg_nodes from run.log.
7. **If grep output is empty**: the run crashed. Run `tail -n 50 run.log` to read the traceback. Attempt a fix if it's simple (typo, shape mismatch). If fundamentally broken after 2-3 attempts, abandon the idea.
8. **Log to results.tsv**.
9. **Keep or discard**:
   - If speedup improved (or tau improved without speedup regression): **keep**. Advance the branch.
   - If speedup regressed or no meaningful change: **discard**. `git reset --hard HEAD~1`.
10. **Repeat**.

### Evaluation Protocol

- **Primary**: mt-bench, 80 samples, temp=0.0. ~10 min per run.
- **Confirmation**: if primary result is positive, run gsm8k (256 samples) and humaneval (164) to confirm generalization.
- **Temperature check**: if primary results are strong, run mt-bench at temp=0.6 to verify sampling robustness.

### Keep/Discard Criteria

- **Speedup improvement >= 0.05x**: keep (meaningful).
- **Speedup within +/-0.03x, tau improved, nodes decreased**: keep (Pareto improvement).
- **Speedup decreased > 0.05x**: discard.
- **Same speedup but much simpler code**: keep (simplification win).

## Known Baselines

From prior experiments (docs/adaptive_tree_sizing_report.md):

| Method | Dataset | speedup | tau | avg_nodes |
|--------|---------|---------|-----|-----------|
| **v2 fixed (ek=8, mts=70)** | **mt-bench** | **4.19** | **5.93** | **71** |
| v4 fixed (ek=8, mts=32) | mt-bench | 4.14 | 5.85 | 54 |
| v2 fixed (ek=8, mts=49) | mt-bench | 4.12 | 5.78 | 50 |
| v4 fixed (ek=8, mts=48) | mt-bench | 4.05 | 5.82 | 72 |
| v2 fixed (ek=3, mts=70) | mt-bench | ~3.85 | ~5.52 | 71 |
| v4 fixed (ek=3, mts=32) | mt-bench | 3.81 | 5.60 | 70 |
| v2 fixed (ek=8, mts=70) | gsm8k | 6.81 | 8.21 | 50 |
| v4 fixed (ek=7, mts=32) | gsm8k | 6.72 | 8.26 | 55 |

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

## Current Best Configuration

```
--tree-version 2 --max-tree-size 70 --expand-k 8
```
Speedup: 4.19, tau: 5.93, nodes: 71 (mt-bench, 80 samples, temp=0.0)

## Open Research Questions

1. **Larger model scales.** On a 70B target where verification IS compute-bound, tree-size optimization and adaptive sizing would have real impact.
2. **Different hardware.** On 1-2 GPUs, the target forward scales with node count — adaptive sizing and node-efficient builders (v4, COT) would help.
3. **Draft model improvements.** The binding constraint is draft quality. Better training objectives, position-dependent drafting, or larger draft models could improve tau.

## Timeout and Crashes

- **Normal run**: ~10 minutes. If a run exceeds 20 minutes, kill it and treat as a failure.
- **Crashes**: fix typos and shape errors. If an idea is fundamentally broken (e.g., OOM from tree explosion), log `crash` and move on.
- **GPU contention**: if speedup numbers look anomalous (much lower than baselines), check `nvidia-smi` for competing jobs. Wait for clean GPUs before drawing conclusions.

## NEVER STOP

Once the loop begins, do NOT pause to ask the user. The user may be asleep or away. You are autonomous. If you run out of ideas, re-read the report, try combining near-misses, or explore the lower-priority questions. The loop runs until the human interrupts you.
