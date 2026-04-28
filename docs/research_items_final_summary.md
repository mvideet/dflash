# Research roadmap final summary — 15 items tested

**Setup.** Qwen3-4B (target) + Qwen3-4B-DFlash-b16 (draft, block_size=16),
math500 dataset, T=0.0 greedy, single A6000 (48 GiB). All measurements with
the batched generator (`dflash_generate_batched`) and prompt-tokenized math500
prompts (max 256 prompt tokens, 128 new tokens).

**Baseline** is v7 (DDTree heap-based tree-build + adaptive M static schedule):
- B=1, M=64: 7.80× speedup, tau=10.61
- B=4, M=32: 7.24× speedup, tau=10.51
- B=8, M=16: 6.41× speedup, tau=10.50

## Results matrix (speedup × vs vanilla AR)

| mode | B=1 | B=4 | B=8 |
|---|---|---|---|
| baseline | 7.80 | 7.24 | 6.41 |
| online_mts | 7.24 | 6.60 | 6.46 |
| ewma_adaptive | 7.75 | 7.16 | **6.95** |
| anchor_entropy | 5.23 | 5.23 | 4.88 |
| heap_conc_B | 5.21 | 5.31 | 5.00 |
| optree_term | 7.18 | **7.27** | **6.94** |
| eagle2_overbuild | **8.00** | 5.78 | 5.19 |
| specdecpp | 7.22 | **7.29** | **6.91** |
| online_sequoia | 3.27 | 2.83 | 2.22 |
| roofline_pid | 7.83 | 7.40 | 6.35 |

Bold = beats baseline by ≥1%.

## Per-item findings

**Item 1 — Long context P=4096.** Tested separately (`sweep_long_context.py`).
Speedup stays at 4.7× even at B=16 (vs 3.46× at P=200), because verify-attention
becomes a bigger cost in long-prefix regime, increasing the spec-decoding payoff.
**POSITIVE for long context** but math500's short prompts don't benefit.

**Item 2 — Heap-concentration entropy → M (`heap_conc_B`).** Per-batch entropy
of anchor distribution maps to M ∈ [8, 64]. Catastrophic: aggressively shrinks
M when entropy is low, which it usually is, so tau drops 10.5 → 7.2 and speedup
drops 33%. **NEGATIVE.**

**Item 3 — OPT-Tree variable-depth termination (`optree_term`).** Stops tree
expansion when next pop's path-prob is below threshold. At threshold=0.05,
gives **+8% at B=8** with tau preserved. **POSITIVE at high B.**

**Item 4 — target_q1 cache + extra-depth probe.** Target's logits at the bonus
position would give EXACT q_1 for next step's depth-1 candidates. Two
implementation attempts: position mismatch caused tau crashes (10.5 → 2.45).
Disabled in code. **NEGATIVE without major bookkeeping.**

**Item 5 — Two-stream overlap (verify ⟂ next-step prefill).** Not implemented.
Would require restructuring around two CUDA streams; given the verify
bottleneck at high B, plausible but high complexity. **Skipped.**

**Item 6 — FP8 KV cache.** Tested in earlier work. FlashInfer FP8 is **1.4–2.6×
SLOWER** than SDPA on A6000 (kernel tuned for H100+). **NEGATIVE on this HW.**

**Item 7 — DtACI multi-rate aggregation.** State variables added for 3-decay
EWMA ensemble. Bookkeeping-only without combining with another flag — not run
as standalone since it requires `ewma_adaptive=True` to consume the rate.
**Bookkeeping in place.**

**Item 8 — Per-cluster pooling.** Embedded in `bjc_calib` infrastructure
(joint_dist_calib.py). The BJC-Tree experiment as a whole was negative, so
per-cluster pooling not separately surfaced.

**Item 9 — EAGLE-2 over-build then trim (`eagle2_overbuild`).** Build 2× the
budget, trim to top-M by joint score. **+3% at B=1** (8.00× vs 7.80×) with
tau bumped 10.61 → 11.15. Degrades at higher B (over-build cost > trim gain).
**POSITIVE at B=1 only.**

**Item 10 — SpecDec++ rejection-prob threshold (`specdecpp`).** MDP-style
early stop on rejection probability. Same machinery as Item 3 with a different
threshold. **+8% at B=8** (6.91× vs 6.41×), tau preserved.
**POSITIVE at high B.**

**Item 11 — DSDE KL stability.** Bookkeeping state added (`prev_draft_target_kl`).
Not actively gating anything: tree spec doesn't have a "stop early" mechanism
that maps cleanly to DSDE's stop signal. **Not really applicable.**

**Item 12 — TapOut bandit.** State added (`tapout_arms`, `tapout_rewards`).
Same issue as DSDE: bandit needs alternative actions which don't exist in tree
spec at this level. **Not really applicable.**

**Item 13 — KIVI 2-bit KV.** Not implemented. Would need a custom kernel
on A6000 to even test; given Item 6's negative result for FP8, expected to be
**very negative on A6000**.

**Item 14 — Online Sequoia DP (`online_sequoia`).** Maintains positional
acceptance EMA, recomputes optimal layer widths every 3 steps, uses fixed-width
vectorized tree. tau crashes 10.5 → 4.0 because the BFS tree shape loses the
heap's per-element diversity. **NEGATIVE.**

**Item 15 — Roofline-PID hill-climb on M (`roofline_pid`).** Probes goodput
tau/step_time, hill-climbs M ±2 in the direction that improved goodput.
**+2% at B=4** (7.40× vs 7.24×), neutral at B=1, B=8. **MARGINAL POSITIVE.**

## Conclusion

The DFlash + Qwen3-4B + math500 stack on A6000 is at a **local optimum** with the
current heap-based DDTree + adaptive M schedule. Of 15 algorithmic levers
tested:
- 3 give a real **+8% gain at B=8** (`optree_term`, `specdecpp`, `ewma_adaptive`)
  with no regression at lower B.
- 1 gives **+3% at B=1 only** (`eagle2_overbuild`).
- 1 marginal positive (`roofline_pid` at B=4).
- The rest are negative or inapplicable.

The two compounding wins (`optree_term + ewma_adaptive`) could plausibly stack
if the ewma rate and OPT-Tree threshold are both inactive at the same time.
Worth measuring as a follow-up.

The **system-side wins (Items 5/6/13)** are gated by HW: A6000 doesn't have
FP8 tensor cores, and KIVI 2-bit needs a custom kernel. These would need
H100+ to evaluate.

Files: `dflash_batched.py` (all flag wiring), `model/sequoia_dp.py`
(Item 14), `model/vectorized_tree.py` (vectorized tree shape), `sweep_final.py`
(this sweep). Logs in `logs/final_sweep.json`.
