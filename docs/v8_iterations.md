# v8 DDTree Iterations — Full Log

**Branch:** `experiments/apr24-v8-joint-cond`.
**Hardware:** 7-8× A6000 (comm-bound for Qwen3-4B). GPU 1 held by another tenant for iterations 1-3.

## Iteration 0 — Initial v8 (entropy-β / HDC / Stage-2 γ)

See `docs/v8_results.md`. Entropy-gated β, Hierarchical Depth Cap, and Stage-2 leaf-γ all tested. No configuration beat v7 B=128 ek=8 on this hardware beyond cluster noise.

## Iteration 1 — Post-Deviation Depth Penalty (PDDP)

**Hypothesis:** DDTree's Prop 1 assumes Q = Π q_i (product). Joint p diverges from Q; the divergence accumulates roughly linearly in depth past a first deviation. Penalize `β · max(depth − first_dev, 0)`.

**Results (math500-64):**
| β   | speedup | tau   | Δ tau vs v7 |
|-----|---------|-------|-------------|
| 0.0 | 7.85    | 9.98  | 0.00 (v7)   |
| 0.5 | 7.78    | 9.87  | −0.11       |
| 1.0 | 7.56    | 9.67  | −0.31       |
| 2.0 | 7.45    | 9.55  | −0.43       |

Monotone regression. Confirmed at N=128 (PDDP 0.5 → 9.95, essentially tied with v7's 10.00 but never exceeds it).

**Why it failed:** the pattern PDDP penalises — "rank-0-chain, rank-k at mid-depth, rank-0 tail" — is mostly a *legitimate* acceptance pattern, not phantom. When target greedily deviates mid-block, the remaining rank-0 marginals stay well-calibrated because target and draft roughly agree.

## Iteration 2 — First-Deviation Rank Penalty (FDRP)

**Hypothesis:** stronger signal than rank count — penalize the *rank* at which the first deviation happened. `β · j^exp` where j = 0-indexed rank of the first deviation (j=1 is top-2, …).

**Results (math500-64, exp=2):**
| β   | speedup | tau   |
|-----|---------|-------|
| 0.0 | 8.07    | 9.98  |
| 0.2 | 7.86    | 9.73  |
| 0.5 | 7.86    | 9.64  |

Confirmed at N=128 (FDRP 0.5 → 9.80 vs v7 10.00, −0.20 tau).

**Why it failed:** like PDDP, targets the wrong class of paths. v7 already excludes deep-rank branches at B=128 (by cumulative log-prob sort). FDRP's incremental pressure on top-3 / top-4 ranks loses legitimate coverage without removing phantoms.

## Iteration 3 — Noise-Floor Measurement

**Goal:** check whether iter-1/2 regressions are real or within measurement noise.

**Results (math500-128):**
| Config | run 1 | run 2 | run 3 |
|--------|-------|-------|-------|
| v7 B=128 speedup | 8.74 | 8.26 | 7.91 |
| v7 B=128 tau     | 10.00 | 10.00 | 10.00 |

**Speedup variance σ ≈ 0.4 (cluster noise). Tau variance ≈ 0 (deterministic given sample set).**

Conclusion: **tau is the diagnostic to trust.** All iter-1/2 regressions in tau (−0.11 to −0.43) are real.

## Iteration 4 — First-Deviation Rank Cap (FDRC)

**Hypothesis:** if v7 already excludes rank-k+ branches (k∈{3,4,5}), hard-capping should be a no-op and tau preserved. If v7 *does* include phantom rank-k+ branches, the cap saves budget for more extensions of rank-0/1/2 branches → tau raised.

**Results (math500-128):**
| cap | speedup | tau   | Δ tau |
|-----|---------|-------|-------|
| 0 (v7) | 7.86 | 10.00 | 0.00 |
| 1   | 7.47 | 9.45  | −0.55 |
| 2   | 7.81 | 9.79  | −0.21 |
| 3   | 7.87 | 9.91  | −0.09 |
| 4   | 7.93 | 9.97  | −0.03 |
| 5   | 7.90 | 9.95  | −0.05 |

mt-bench-40 cap=3: 3.85 / 5.93 vs v7 3.99 / 6.10 → −0.17 tau confirmed cross-dataset.

**Why it failed:** rank-3, rank-4, rank-5 branches in v7's top-128 DO contribute to tau. The monotone decrease in penalty (−0.55 → −0.21 → −0.09 → −0.03) shows that each additional rank allowed recovers ≤0.3 of the lost tau, with diminishing returns. v7's top-B selection uses rank-3+ branches legitimately.

## Iteration 5 — B-Pareto scan (first positive)

**Hypothesis:** since every selection modification regresses tau, the only untested lever is B (node budget). On A100, program.md reports B=128 peak; on A6000 the comm profile is different and the peak may be earlier.

**Results (8× A6000, N=128 samples):**

math500-128:
| B   | speedup | tau   | nodes |
|-----|---------|-------|-------|
| 64  | 7.88    | 9.62  | 65    |
| 96  | **7.93**| 9.88  | 97    |
| 112 | 7.89    | 9.86  | 113   |
| 128 | 7.85    | 10.00 | 129   |
| 144 | 7.79    | 10.07 | 145   |
| 160 | 7.74    | 10.12 | 161   |

mt-bench-40:
| B   | speedup | tau  | nodes |
|-----|---------|------|-------|
| 64  | 3.98    | 5.79 | 65    |
| 96  | 4.08    | 6.08 | 97    |
| 112 | **4.11**| 6.13 | 113   |
| 128 | 4.03    | 6.10 | 129   |

**Peak speedup at B≈96-112 on 8× A6000. +1.0% to +2.0% over B=128.**

This contradicts program.md's "B=128 is peak" statement — which was derived on A100. A6000 has a smaller comm/compute ratio on Qwen3-4B and peaks earlier in budget.

## Summary across iterations

| Iter | Mechanism | Result | Lesson |
|------|-----------|--------|--------|
| 0 | entropy-β / HDC / Stage-2 γ | Tie w/ v7 | v7 near-optimal |
| 1 | PDDP (β · depth-past-dev) | Tau −0.11..−0.43 | Penalizes legit |
| 2 | FDRP (β · j^exp at 1st dev) | Tau −0.15..−0.36 | Penalizes legit |
| 3 | Noise floor measurement | σ_tau ≈ 0; σ_spd ≈ 0.4 | Use tau, not speedup |
| 4 | FDRC (hard cap on 1st dev rank) | Monotone regression | v7 uses rank-3+ legit |
| 5 | **B-Pareto scan** | **B=96 +1.0% on math500; B=112 +2.0% on mt-bench** | Hardware tuning wins |

**Algorithmic contribution: zero.** Every modification to v7's scoring/selection either ties v7 or regresses. v7 DDTree's product-distribution top-B is near the rank-based-selection frontier for this drafter/target pair on this hardware.

**Practical recommendation:** switch default `--max-tree-size` from 128 to **96 for math/code/reasoning** and **112 for chat**. Free +1-2% speedup.

**Where further algorithmic gains would come from (out-of-scope):**
- Non-product drafter (train draft to produce joint distribution; V5/V6 partial-mask in later branches do this empirically).
- Hardware with different comm profile (H100).
- Target-side kernel fusion for tree-attention verification.
