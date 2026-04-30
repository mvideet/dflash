# CDDT (Calibrated DDTree) — synthesis from autonomous research loop

**Setup throughout:** Qwen3-4B target + Qwen3-4B-DFlash-b16 draft, math500
benchmark, T=0.0 greedy, A6000. All measurements in `logs/`. The autonomous
loop ran 7 algorithm experiments (decay sweep, fine sweep, comparative,
universality, UST, LDB, RCD, truncated) plus 3 verification diagnostics
(failure-mode verification, comparative tree shapes, U-shape universality).

## L-CDDT (online learned-curve) — final adaptive attempt, NEGATIVE

Implemented per-depth EWMA of P(rank-0 chain accept | alive at d) and used the
learned curve directly as heap-score weights. Sanity check confirmed the
curves are learned correctly per dataset (`logs/lcddt_sanity.json`):

- math500: U-shape ✓ (mid 0.55, deep 0.62)
- gsm8k: monotone-decay ✓ (mid 0.54, deep 0.37)
- humaneval: weak U-shape ✓

**Sanity correct, performance NEGATIVE** (`logs/lcddt_cross.json` at N=24, B=4):

| dataset | v7 | static CDDT | L-CDDT p1 | L-CDDT p2 | L-CDDT p3 |
|---|---|---|---|---|---|
| math500 | 6.65× | **+8.3%** | +2.2% | +0.8% | -1.1% |
| gsm8k | 4.54× | -2.5% | -2.3% | -1.0% | -2.7% |
| humaneval | 4.81× | -0.3% | -1.2% | -2.0% | -4.6% |

L-CDDT under-performs static CDDT everywhere. **Curve-matching is not the
mechanism.** Geometric decay's gain on math500 comes from its *aggressive*
deep-shrinkage (weight 0.087 at d=15), much stronger than even gsm8k's
natural learned curve (0.29 at d=15). The aggressive shrinkage is a
strategic budget reallocation, not a calibration to the data — and no
learned curve replicates this without data-specific over-amplification.

This is a definitive negative for the "learn the curve" approach. **No
adaptive variant we tested (A-CDDT entropy, W-CDDT EWMA, L-CDDT curve)
matches static CDDT on its native dataset, AND none rescues cross-dataset
failure.**

## Definitive cross-dataset (B=4, N=24 prompts, 240-431 rounds)

The variance question from earlier N=12 runs is now resolved:

| dataset | baseline | CDDT static | gain | rounds |
|---|---|---|---|---|
| math500 | 6.65× | 7.14× | **+7.4%** | 240 |
| gsm8k | 4.58× | 4.43× | **-3.3%** | 431 |
| humaneval | 4.79× | 4.80× | +0.2% | 328 |

Across 4 independent runs at varying N, CDDT's math500 gain is reliably
positive (+5.2%, +5.4%, +7.4%, +12.4%); gsm8k is reliably non-positive
(+1.4%, -0.5%, -3.3%); humaneval is essentially zero on average.

**Final shipping recommendation:** CDDT static γ=0.85 is workload-specific. Do
NOT ship as default. Enable per-workload after running the per-depth accuracy
diagnostic (`diag_curve_cross.py`) — apply CDDT iff the rank-0 chain accept
curve has a U-shape recovery (P(accept) at depth 15 > 0.5 + P(accept) at
depth 8). Otherwise leave disabled.

## Final cross-dataset comparison (B=4, N=12 prompts) — superseded by N=24 above

| variant | math500 | gsm8k | humaneval | safe to ship? |
|---|---|---|---|---|
| v7 baseline | 6.32× | 4.75× | 4.40× | (reference) |
| **CDDT static γ=0.85** | **+12.4%** | +1.4% | **+2.3%** | math500-recommended; small positive elsewhere with high variance |
| A-CDDT (entropy gate) | +10.5% | +1.0% | +2.8% | similar to static |
| W-CDDT (EWMA gate) | +6.7% | +0.9% | -0.9% | warmup cost loses too much |

(Earlier cross-dataset run at same N showed CDDT = -0.5% to -9.7% on
non-math datasets. The variance between runs at N=12 is large — both
positive and negative observations on gsm8k. Larger N would be needed for
a confident generalization claim.)

**Final algorithm decision:** ship CDDT (static γ=0.85) as opt-in flag, default
disabled. A-CDDT and W-CDDT explored but don't reliably improve on static.

## TWO CRITICAL CORRECTIONS — multi-seed reveals headline claims were seed-0 artifacts

### Correction 1: math500 CDDT gain at B=4

| dataset | seed 0 | seed 1 | seed 2 | mean | std | sig? |
|---|---|---|---|---|---|---|
| math500 B=4 | +5.6% | -0.8% | -7.0% | -0.7% | 5.1% | ✗ |
| aime24 B=4 | +4.4% | +4.5% | +2.0% | **+3.6%** | **1.1%** | ✓ |
| gsm8k B=4 | -3.6% | -2.8% | -1.4% | -2.6% | 0.9% | ✓ neg |

### Correction 2: chain_mode at B=8

| dataset | seed 0 | seed 1 | seed 2 | mean | std | sig? |
|---|---|---|---|---|---|---|
| math500 B=8 chain | +9.3% | -5.3% | -0.0% | +1.3% | 6.0% | ✗ |
| aime24 B=8 chain | -2.1% | +3.1% | -1.8% | -0.3% | 2.4% | ✗ |
| gsm8k B=8 chain | -6.8% | -6.7% | -8.1% | -7.2% | 0.7% | ✓ neg |

### The autonomous loop's only verified positive contribution

**CDDT γ=0.85 reliably helps AIME at B=4: +3.6% ± 1.1%.**

Everything else is either significantly negative or noise-dominated at N=20 × 3
seeds. The single-seed runs throughout the loop (all using seed=0 as the
default `shuffle(seed=0)` in tokenize_prompts) were systematically biased.

### Methodology lesson

**Multi-seed evaluation should have been step 0**, not step 22 of the
autonomous loop. The entire intermediate research narrative — CDDT works on
math500, generalizes to formal math, chain_mode helps at B=8 — was based on
single-seed measurements that did not survive multi-seed validation. Only the
AIME finding (which used a smaller more uniform dataset) replicated.

## CRITICAL CORRECTION: multi-seed reveals math500 gain is sample bias (single-seed artifact, see corrections above)

Single-seed measurements throughout the loop showed math500 +5-12% across 7+
runs. With proper multi-seed evaluation (`logs/multiseed.json`, N=20 × 3
shuffle seeds × 2 variants):

| dataset | seed 0 | seed 1 | seed 2 | mean | std | significant? |
|---|---|---|---|---|---|---|
| math500 | +5.6% | -0.8% | -7.0% | **-0.7%** | **5.1%** | ✗ NO |
| aime24 | +4.4% | +4.5% | +2.0% | **+3.6%** | **1.1%** | ✓ YES |
| gsm8k | -3.6% | -2.8% | -1.4% | **-2.6%** | **0.9%** | ✓ YES |

**The math500 win was sample-specific.** Different shuffles produce wildly
different gains (+5.6% to -7.0%). The 7+ prior single-seed measurements that
agreed within +5-12% were all on seed=0 (default) — they reflected the same
problem-shuffle bias. With seeds 1 and 2, math500 is essentially neutral
(-0.8%) or actively negative (-7.0%).

Math500 has high heterogeneity (algebra/geometry/probability/combinatorics/
calculus). Different seeds sample different problem-type compositions. CDDT
likely helps some problem types and hurts others; the "average" is noise.

**The true robust signals are:**
1. aime24 (30 Olympiad problems, more uniform): **+3.6% ± 1.1%**, significant
2. gsm8k: **-2.6% ± 0.9%**, significant negative
3. math500: noise-dominated; no robust signal at this sample size

**Major revision to shipping recommendation:** AIME is the canonical workload
where CDDT robustly helps, not math500. The earlier "math500 +5-8%" claim is
unreliable.

## Math-domain generalization — AIME confirms formal-math-reasoning hypothesis

CDDT tested on aime24 (math olympiad, similar structure to math500):

| dataset | γ=1.0 | γ=0.95 | γ=0.85 |
|---|---|---|---|
| math500 | 6.66× | +3.5% | **+8.2%** |
| aime24 | 5.48× | -0.1% | **+4.6%** |
| gsm8k | 4.53× | -2.1% | -2.9% |

**The gain transfers to AIME** — CDDT is not math500-specific. The right
characterization is **formal multi-step math reasoning**:

- math500, aime24: long deterministic algebraic continuations create U-shape
  recovery in per-depth accuracy → CDDT works
- gsm8k: shorter natural-language word problems → no U-shape → CDDT doesn't help

The deep-recovery is a property of "what statistically survives to depth 12-15
along the rank-0 chain." For formal math, survivors are working through
deterministic notation (equations, simplifications) where the drafter is
reliable. For natural-language word problems, survivors are mid-sentence
where language entropy is moderate-high. The dataset's "deep token
predictability" is the load-bearing property.

## Final N=24 headline — the honest reversal

After running γ=0.95 production config (CDDT + chain_mode at B=8) across all
3 datasets at N=24:

| dataset | B=1 | B=4 | B=8 |
|---|---|---|---|
| math500 | +2.5% | +1.9% | **+9.0%** |
| gsm8k | -2.5% | -2.4% | -3.0% |
| humaneval | -1.6% | +2.9% | -4.5% |

**The "γ=0.95 generalizes" claim from the γ-sweep was sampling noise.** With
larger N=24, gsm8k flips back to negative.

Aggregating ALL cross-dataset runs in the loop:

- **math500 γ=0.85**: +5.4%, +5.2%, +12.4%, +7.4%, +7.5%, +8.3%, +1.9% — robust positive
- **math500 γ=0.95**: +2.5% (this run only)
- **gsm8k γ=0.85**: +1.4%, -0.5%, -3.3% — mostly negative
- **gsm8k γ=0.95**: +1.0%, +0.9%, -2.4% — variance straddles zero, mean ~-0.2%
- **humaneval γ=0.85**: +2.3%, -0.3%, -0.9% — neutral
- **humaneval γ=0.95**: +3.3%, +2.9%, -1.6% — neutral, high variance

**No γ value tested reliably helps across all 3 datasets.** CDDT is genuinely
math500-specific. Any "safe default γ" claim cannot be supported with the
sample sizes we ran.

## γ-sweep cross-dataset (B=4, N=16) — earlier optimistic interpretation, superseded

| dataset | γ=1.0 | γ=0.95 | γ=0.85 | γ=0.75 | γ=0.65 | γ=0.55 |
|---|---|---|---|---|---|---|
| math500 | (ref) | +4.6% | **+7.5%** | +5.2% | +1.3% | +2.9% |
| gsm8k | (ref) | **+1.0%** | +0.1% | -1.9% | -3.6% | -7.2% |
| humaneval | (ref) | **+3.3%** | -0.9% | -0.4% | -5.2% | -10.3% |

**γ=0.95 is positive on all three datasets simultaneously.** The previous
"CDDT doesn't generalize" claim was using math500-tuned γ=0.85, which is
too aggressive for gsm8k/humaneval. With the right γ, CDDT generalizes:

- math500: +4.6% (vs +7.5% peak; ~60% of headline gain captured)
- gsm8k: +1.0% (vs -2.5% at γ=0.85; flipped sign)
- humaneval: +3.3% (vs -0.9% at γ=0.85)

**Two ship-able configs:**

1. **γ=0.85 — math500-optimal**: +7.5% on math500, -2.5% on gsm8k. Opt-in
   for known workloads with U-shape recovery.
2. **γ=0.95 — cross-dataset safe default**: +1-5% across all 3 datasets,
   no negative regressions. **Recommended deployment default.**

The pattern is monotone in dataset hardness. math500 (easy U-shape) tolerates
aggressive deep-shrinkage; gsm8k/humaneval (harder, monotone curves) need
gentle shrinkage. The γ optimum tracks the slope of the per-depth accuracy
curve, which explains why no single static γ is universally best — but γ=0.95
is the "low-risk" choice that wins everywhere.

## TL;DR

After exhaustively testing the v7 (DDTree heap) failure modes and 5
algorithm candidates derived from each, **the only orthogonal-axis algorithmic
win is geometric depth-decay scoring** (`score_decay = 0.85`). Every more
complex schedule we derived from the failure-mode analysis fails to match it:

| algorithm | mechanism | result at B=4 |
|---|---|---|
| baseline (v7) | `Σ log q_d` | 6.65-6.77× |
| **CDDT (= v7 + decay)** | `Σ γ^d log q_d`, γ=0.85 | **+5-7%** (7.02-7.13×) |
| UST_w (U-shape weights) | weights match measured P(accept) curve | +1-3% |
| UST_K (depth-conditional K) | wide K at middle depths, narrow at edges | 0% |
| LDB (penalize early deviation) | bonus on deviation depth | -5 to -15% |
| RCD (decay only post-deviation) | rank-0 chain uniform; siblings decay | +2-4% |
| trunc_D (cut weights past d=D) | hard cutoff | -25 to -50% |

The clean, single-parameter geometric decay wins. **CDDT is not a redesign;
it's a 1-line score function modification.** It reliably beats v7 by 5-7% at
B=4 and is approximately neutral at B=1 and B=8.

## Verified failure-mode profile

(See `docs/v7_failure_mode_verification.md` for the full cross-tab.)

| FM | Original claim | Verified status |
|---|---|---|
| FM1: Rank-0 rejections (22-35%) | "heap walking deep-rank-2 chains crowds out rank-0 chain; decay would let rank-0 win" | **FALSIFIED**: rank-0 chain never beats best leaf in 0/551 rounds. The statistic is real but the causal explanation was wrong. |
| FM2: Late-depth heavy tails | "deeper depths have heavier rejection-rank tails" | **REFINED**: depth 10-15 has 2-3× tail rate vs 6-9, but shallow also has tails. Not a clean monotone story. |
| FM3: Out-of-K rate (rank ≥ 8) | 6-15% of rejections | **CONFIRMED exactly**: 14.6% / 8.6% / 6.2% across B=1/4/8 |
| FM4: Bimodal accept distribution | "~30% n=15 + flat 5%/bucket" | **REFINED**: full-block 23-26% (not 30%); middle bucket (n=4-7) heavily weighted at 30-38% (not flat) |
| FM5: B=8 tree → chain | tree degenerates at M=block_size | **CONFIRMED**: 96% of rounds have 14-17 nodes, modal 16 |
| FM6 (NEW): U-shaped per-depth draft accuracy | predicted by training-time noise schedule reasoning | **CONFIRMED**: P(target.argmax = draft top-1 \| rank-0 chain alive at d) = 0.94 at d=1, dips to 0.60-0.66 at d=5-8, recovers to 0.68-0.77 at d=12-15 |
| FM7 (NEW): score_decay's mechanism is sibling-shape redistribution | from comparative diagnostic | **CONFIRMED**: deviation depths shift from middle (dev@5: 9.7%→4.2%) to late (dev@7: 3.2%→9.2%); full-block rate rises 24%→30%; pure-rank-0 best leaves slightly decrease (42.7%→40.0%) |

## What we learned by failing

The 4 negative algorithm experiments each isolate a piece of the mechanism:

1. **UST_w (U-shape weights) preserves too much deep weight.** UST_w used
   measurement-derived `w(d) = p(d)` curve which assigns ~0.77 weight at
   d=15. Score_decay 0.85 assigns 0.087 at d=15 — 9× lower. UST_w's gain at
   B=4 was +1.3%; decay's was +5-7%. **Lesson:** the win comes from
   AGGRESSIVELY shrinking deep weights, not from matching the U-shape.

2. **LDB (deviation-depth bonus) actively hurts.** Penalizing early-deviation
   children kills sibling diversity at shallow depths. The heap loses options
   for rounds where target diverges early. **Lesson:** the empirical "shift
   from dev@5 to dev@7" is a SYMPTOM of decay's mechanism, not the cause —
   forcing it directly via deviation-depth penalty is wrong.

3. **RCD (decay only deviated paths) gains less than full decay.** If rank-0
   chain truly were "ground truth" we'd want to NOT decay it. But RCD's gain
   peaks at +4.7% (vs decay's +9.9%) at B=4. **Lesson:** the rank-0 chain's
   deep tokens are imperfect too (per FM6's U-shape) — decay applies usefully
   to all paths, not just deviated ones.

4. **Truncated decay (hard zero past d=D) collapses τ.** trunc_3 gives only
   3.90× at B=4 (vs baseline 6.77×). **Lesson:** the heap needs deep
   contributions for ORDERING (telling apart paths with similar shallow
   prefix); decay reduces magnitude without zeroing the signal.

The picture: **decay does ONE thing well — soft deep-contribution
attenuation that preserves ordering**. Every variation either over-emphasizes
deep weights, under-emphasizes them, or destroys the ordering structure.

## CDDT specification

### The change

In the heap-build, replace the path scoring:
```
old: score(P) = Σ_d log q_d(token_d)
new: score(P) = Σ_d γ^d · log q_d(token_d)
```
with `γ = 0.85` (configurable). All other v7 mechanics — heap order, expansion
strategy, leaf selection, verify, accept-path — unchanged.

### Why this works (the verified mechanism)

The geometric decay shifts heap budget from "deep extensions of moderate paths"
toward "shallow paths with high q at every position." This causes two
empirically-measured downstream effects (from `logs/decay_compare.json`):

1. **Deviation-depth distribution shifts later.** Best-leaf deviations move
   from depth 5 (9.7% → 4.2%) to depth 7 (3.2% → 9.2%). Late-deviation
   leaves have longer accept length on average because they share more rank-0
   prefix with the eventual target trajectory.
2. **Full-block accept rate rises.** P(n=15) goes 24% → 30% (+6 pts), driving
   a +0.42 mean τ at B=4 even though best-leaf-is-pure-rank-0 rate
   actually slightly DROPS (42.7% → 40.0%).

The mechanism is **sibling-shape redistribution at the round level**, NOT
"the heap picks the rank-0 chain more often."

### Recommended γ values per B

From sweeps:

| B | Optimal γ | Speedup gain over v7 |
|---|---|---|
| 1 | 0.95 | +1-2% (approximately neutral) |
| 2 | 0.88 | +6.2% |
| 4 | 0.85 | **+5.4-9.9%** (most reliable) |
| 8 | 1.00 (no decay) | M=block_size, decay is no-op; +0% |

At B=8 M=block_size, the tree degenerates to chain (96% modal at 16 nodes per FM5).
Decay has nothing to apply to (single chain, no siblings) — graceful no-op.

### Implementation effort

3-line patch in `_build_one_tree`:
- One new arg `score_decay: float = 1.0`
- One factor `decay**(d+1)` multiplied into each child contribution
- One additional state field in heap entries to track decayed score

Already implemented in current `dflash_batched.py` and `model/dflash_tree_batched.py`.

### Caveats

- **+5-7% is at B=4 specifically.** B=1 and B=8 see no meaningful gain.
- **All measurements at math500 + Qwen3-4B + DFlash-b16 + bf16 + greedy.**
  Generalization to other datasets/models/temperatures untested.
- **Single-prompt-set noise:** observed gains range from +2.3% to +9.9% across
  different sample sizes. Conservative production estimate: **+4-6% at B=4**.

## Where this leaves the open research questions

- **Why does score_decay's mechanism work?** Sibling-shape redistribution is
  the OBSERVED effect, but the deeper question — why deep extensions of
  off-track paths beat the shallow-prefix winner under uniform weighting —
  remains open. Likely connected to the diffusion drafter's training-time
  noise schedule, but verifying this would require examining the drafter's
  training code.
- **Is there a non-geometric schedule that beats 0.85^d?** Tested 4 schedule
  families (UST, LDB, RCD, truncated) — all worse. Possibly an exhaustive
  search of weight schedules would find marginally better, but at this point
  the variance dominates.
- **Can we make CDDT B-conditional cleanly?** The optimal γ shifts with B
  (0.95 at B=1 → 0.85 at B=4 → 1.00 at B=8). A self-adaptive γ from per-round
  signals (anchor entropy, prior accept rate) is a future direction.

## Recommendation

1. **Ship CDDT as the v7 default with γ=0.85.** Rename `score_decay` parameter
   to `gamma` for clarity in API.
2. **Use γ=0.95 at B=1, γ=0.88 at B=2, γ=0.85 at B=4, chain_mode at B=8** via
   B-conditional default rule.
3. **Stop further hyperparameter search on schedule shape.** The autonomous
   loop has explored 4 families with sufficient evidence that the simple
   geometric form is the local optimum.
4. **The "score_decay's actual mechanism" remains a research lead.** A
   follow-up understanding might suggest a related improvement (perhaps
   targeting the sibling-shape redistribution directly), but this is research
   not engineering.

## The mechanism, fully explained

The per-depth rank-0 accept curve differs by dataset (`logs/curve_cross.json`):

| depth | math500 | gsm8k | humaneval |
|---|---|---|---|
| 1 | 0.94 | 0.87 | 0.88 |
| 5 | 0.64 | 0.46 | 0.39 |
| 10 | 0.65 | 0.40 | 0.53 |
| 15 | **0.80** | **0.31** | **0.40** |

**math500 alone has an end-recovery U-shape.** gsm8k monotonically decays;
humaneval has weak recovery. This is the LOAD-BEARING property of CDDT:

- **math500's depth-15 accept rate is 0.80** because rounds whose rank-0
  chain reaches depth 15 are SURVIVORSHIP-BIASED toward easy prompts. The
  drafter is genuinely calibrated on those prompts at deep depths. Decay's
  "shallow-focus" budget reallocation captures more of this end-mass → more
  full-block accepts → tau gain.
- **gsm8k drops to 0.31 at depth 15** — the drafter never recovers. Decay
  removes useful information without exposing additional "easy" mass. No gain.
- **humaneval rises slightly mid-block then declines** — limited recovery,
  limited gain.

**The right per-workload signal is the deep-depth accept rate**, NOT the
anchor entropy that A-CDDT uses. A truly dataset-robust algorithm would
estimate P(rank-0 chain accept at depth ≥ 12) online and gate decay on it.

This explains why anchor-entropy-based A-CDDT only partially generalizes:
anchor entropy correlates with shallow drafter confidence but doesn't predict
deep-depth recovery, which is the actual driver.

## Adaptive-CDDT (A-CDDT): the dataset-robust extension

A-CDDT addresses CDDT's generalization failure at B=4 by selecting γ
**per-round** based on anchor entropy:
- Mean batch anchor entropy < threshold → γ=0.85 (round is "easy", apply decay)
- Else → γ=1.0 (round is "hard", no decay)

Cross-dataset results at B=4 with threshold=0.4:

| dataset | v7 baseline | static CDDT (γ=0.85) | A-CDDT (t=0.4) |
|---|---|---|---|
| math500 | 6.66× | +5.2% | **+6.6%** |
| gsm8k | 4.81× | -0.5% | **-0.1%** |
| humaneval | 4.40× | +3.0% | +1.8% |

**A-CDDT at B=4:** preserves math500 win (better than static CDDT), neutralizes
gsm8k hurt, modest positive on humaneval. **Genuinely dataset-robust at B=4.**

A-CDDT does NOT solve B=1 — all variants hurt at B=1 on gsm8k (-4 to -5%) and
humaneval (-1 to -4%). The B=1 regime has M=64 (large budget); the heap's
budget allocation is fine without intervention. **Recommendation: don't apply
A-CDDT at B=1.**

Mechanism: A-CDDT's threshold gating defaults to γ=1.0 when rounds are hard,
which is most of gsm8k. So A-CDDT degrades to baseline on hard datasets. On
math500 where rounds are mostly "easy" (low anchor entropy), it fires γ=0.85
nearly always — and slightly outperforms static because it correctly disables
on the rare hard rounds.

## Cross-dataset generalization — KEY CAVEAT

**CDDT and chain_mode do NOT generalize beyond math500.** Tested on gsm8k
and humaneval (`logs/cddt_cross_dataset.json`):

| dataset | B=1 | B=4 | B=8 |
|---|---|---|---|
| math500 | +1.9% | **+7.6%** | **+9.4%** |
| gsm8k | -1.7% | ≈0% | **-9.7%** |
| humaneval | -2.2% | +4.2% | **-7.2%** |

- **gsm8k**: CDDT decay is ~neutral; chain_mode at B=8 actively HURTS
  (-9.7%). The dataset has lower mean τ (6.58 vs math500's 9.37 at B=4) — the
  "easy round" mass that decay's mechanism exploits is largely absent.
- **humaneval**: only B=4 + decay survives marginally (+4.2%); chain_mode
  at B=8 hurts (-7.2%); B=1 hurts (-2.2%).

**Implication:** the U-shape per-depth accuracy curve and the bimodal accept
distribution that drove the algorithm's design are **dataset-specific**. Fixed
γ values are unsafe defaults. A robust deployment needs either:
1. **Per-dataset tuning** (calibrate γ on a held-out set per workload), or
2. **Adaptive γ** (use a per-round signal like anchor entropy to back off to
   γ=1.0 when the round looks "hard"), or
3. **Just don't ship CDDT as default** and treat it as an opt-in flag for
   workloads where you've validated it.

**chain_mode at B=8 is even less safe to ship as default.** Outside math500
it hurts τ substantially. Only enable when validated on the target workload.

## Final headline (consolidated production benchmark, N=24 prompts × 192 tokens)

| B | M | v7 baseline | CDDT production config | gain |
|---|---|---|---|---|
| 1 | 64 | 7.12× | 7.17× | +0.7% |
| 2 | 32 | 6.25× | 6.54× | **+4.5%** |
| 4 | 32 | 6.36× | 6.51× | +2.3% |
| 8 | 16 | 5.35× | 5.78× | **+8.0%** |

**Production config per B:**
- B=1: CDDT γ=0.95
- B=2: ewma_adaptive + CDDT γ=0.88
- B=4: ewma_adaptive + CDDT γ=0.85
- B=8: specdecpp + chain_mode

The B=8 chain_mode win was understated in earlier small-N tests (+0.7% at
N=8) but materializes at +8.0% with larger N. CDDT itself is no-op at B=8
(M=block_size leaves no room for decay to redistribute). The two pieces
together make a clean B-conditional production policy.

## Files

- Sweeps: `logs/score_decay.json`, `logs/score_decay_fine.json`,
  `logs/decay_compare.json`, `logs/ust_sweep.json`, `logs/ldb_sweep.json`,
  `logs/rcd_sweep.json`, `logs/trunc_sweep.json`
- Diagnostics: `logs/failure_mode_verify.json`, `logs/u_shape_universality.json`
- Code: `model/dflash_tree_batched.py:38-220` (`_build_one_tree`),
  `dflash_batched.py:418` (signature with `score_decay`)
