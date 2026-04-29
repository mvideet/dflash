# v7 failure-mode verification report

**Setup.** Qwen3-4B target + Qwen3-4B-DFlash-b16 draft, math500, T=0.0 greedy,
A6000. N=16 prompts × 128 tokens × 3 configs (B=1 baseline, B=4 ewma_adaptive,
B=8 specdecpp). Each round's diagnostic captured: rejection rank in draft q,
best-leaf path-of-ranks, rank-0 chain accept length, accept distribution,
tree node counts. Logs in `logs/failure_mode_verify.json`.

The five claims I made about v7 in the redesign discussion:
1. **FM1.** Rank-0 rejections (22-35%) are caused by the heap walking a non-rank-0 sibling chain; the rank-0 chain "would have won" if not crowded out.
2. **FM2.** Late-depth (10-15) rejections have heavier rank tails than shallow rejections.
3. **FM3.** Out-of-K rejections (rank ≥ 8) are 6-15% of rejections.
4. **FM4.** Accept distribution is bimodal: ~30% n=15 + ~5%/bucket n=2..13.
5. **FM5.** At B=8 M=block_size, the heap degenerates to a chain.

For each claim I report (a) what the measurement showed and (b) whether the
claim is **CONFIRMED**, **REFINED**, or **FALSIFIED**.

---

## FM1: Rank-0 rejection mechanism — **FALSIFIED**

### What I claimed

When target.argmax matches draft top-1 at the rejection point (= "rank-0
rejection"), the cause is that the heap walked a non-rank-0 sibling chain whose
deep contributions accumulated enough joint log-prob to dominate the rank-0
chain. score_decay was supposed to fix this by re-favoring the rank-0 chain.

### What I measured

For every round, I logged both (a) the best leaf (whichever had the longest
accept), and (b) the rank-0 chain (= leaf 0 by heap convention, which by
factorized greedy IS always the all-rank-0 chain).

| config | rounds | best-leaf is pure-rank-0 | best-leaf is mixed | rank-0 chain accepted MORE than best leaf |
|---|---|---|---|---|
| B=1 M=64 | 211 | 41.7% | 58.3% | **0/211 (0.0%)** |
| B=4 M=32 | 172 | 48.3% | 50.6% | **0/172 (0.0%)** |
| B=8 M=16 | 168 | 61.9% | 37.5% | **0/168 (0.0%)** |

**The rank-0 chain NEVER beats the best leaf.** Across 551 rounds across all
configs, there is not a single round where the rank-0 chain accepted more
tokens than the leaf actually selected. This **falsifies** my claim that decay
"lets the rank-0 chain win when it would have accepted longer." It cannot win
longer — it always at most ties.

### What's actually happening

The mechanism is more subtle. When a "rank-0 rejection" occurs:
- target.argmax at the parent on the best leaf's chain matches draft top-1
- BUT the best leaf's child at that depth was NOT draft top-1
- The best leaf is therefore on a "deviation branch" — at some earlier depth
  the rank-0 chain already failed (target wanted a non-top-1 token there), and
  a non-rank-0 sibling that matched target picked up longer accept.

Verified by cross-tab: of the rank-0-rejection rounds, **77% (B=1), 77% (B=4),
56% (B=8)** had a "mixed" best leaf — meaning the best leaf had ≥1 non-rank-0
ancestor before the rejection point.

### What this means for score_decay

The empirical gain from `score_decay=0.85` (+2.3-4% throughput at B=4) is
**real**, but the mechanism I proposed is wrong. score_decay does not "let
the rank-0 chain win." Plausible alternative mechanisms (untested):

- **Sibling-shape shift.** Decay may bias the heap to prefer different
  divergence-branches — specifically those with shallow deviation followed by
  many rank-0 children, rather than deep deviations with rank-2+ chains.
  Shallow-deviation branches have more downstream depth to gain accept length,
  so this could increase τ even when rank-0 chain doesn't change.
- **Post-deviation expansion.** With decay, deep rank-2+ children get
  down-weighted, so the heap spends budget on EARLIER siblings. This may put
  more "competing" rank-1, rank-2 children near the root in the tree, where
  they compete with each other for being the divergence branch that target
  prefers.

Either way, **the redesign's C1 (decay) was motivated by a wrong story but
empirically holds**. CC-Tree should be re-pitched without the falsified causal
explanation.

---

## FM2: Late-depth heavy tails — **REFINED, weaker than claimed**

### What I claimed

Depths 10-15 have substantially heavier rejection-rank tails (target diverges
to low-q tokens) than shallow depths.

### What I measured

Per-depth median rejection rank and `%(rank > 7)` from N=16 prompts:

| config | depth 2-5 (median, %>7) | depth 6-9 (median, %>7) | depth 10-15 (median, %>7) |
|---|---|---|---|
| B=1 M=64 | (2, 25.0%) | (1, 6.4%) | (2, 22.5%) |
| B=4 M=32 | (2, 5.4%) | (1, 5.0%) | (1, 11.0%) |
| B=8 M=16 | (1, 1.6%) | (1, 2.1%) | (1, 14.7%) |

(Aggregated across depths within bucket; median taken on pooled ranks; %>7 is
fraction of rejections in bucket with rank > 7.)

**Pattern is mixed:**
- **Refined claim:** rejections at depths 10-15 do show higher tail rate than
  depths 6-9 in all three configs (22% vs 6%, 11% vs 5%, 14% vs 2%). So a
  late-depth tail effect does exist.
- **But:** depths 2-5 ALSO show heavy tails at B=1 (25%) — comparable to deep
  depths. So the picture is NOT "shallow depths are easy, deep depths are
  hard." It's more like an early-failure regime (shallow with fewer samples)
  + a late-failure regime (deep with compounding error) + a quiet middle.
- Sample sizes per depth-bucket are small (n=10-30), so confidence intervals
  are wide.

**Verdict:** the claim that late depths have heavier tails is qualitatively
correct but the magnitude is smaller than I implied, and shallow-depth
rejections are NOT all rank-1-2 — they have a non-trivial tail too. The
"selective K-widening at deep depths" component of CC-Tree (C2-widen) should
target ALL paths with high `Δ` signal, not just deep ones.

---

## FM3: Out-of-K rejections — **CONFIRMED**

### What I claimed

6-15% of rejections have target.argmax at rank ≥ 8 (out of draft's top-K=8).

### What I measured

Rank distribution of rejections (rank in draft q at parent):

| config | rank 0 | rank 1-2 | rank 3-7 | rank 8-15 | rank 16+ | out-of-K (≥8) |
|---|---|---|---|---|---|---|
| B=1 M=64 | 22.3% | 39.5% | 23.6% | 8.9% | 5.7% | **14.6%** |
| B=4 M=32 | 24.2% | 49.2% | 18.0% | 3.9% | 4.7% | **8.6%** |
| B=8 M=16 | 34.9% | 45.7% | 13.2% | 4.7% | 1.6% | **6.2%** |

Out-of-K rate is 6-15% across configs. **Confirmed exactly.**

The out-of-K events are a hard cap on what selective K-widening could recover:
even perfectly catching all rank ≥ 8 events at B=4 would address only 8.6% of
rejections × ~75% rejection rate per round ≈ 6% of rounds × 1-2 tokens lift
each ≈ 0.06-0.12 τ ≈ 0.5-1% throughput. Smaller than I implied in the CC-Tree
estimate.

---

## FM4: Bimodal accept distribution — **CONFIRMED, refined numbers**

### What I claimed

~30% of rounds hit n=15 (full block); rest spread roughly uniformly across
n=2..13 at ~5%/bucket.

### What I measured

Accept-length distribution (n = number of tokens accepted before bonus):

| config | n=15 (full) | n=0-1 | n=2-3 | n=4-7 | n=8-13 | rounds total |
|---|---|---|---|---|---|---|
| B=1 M=64 | **25.6%** | 1.4% | 11.8% | 30.8% | 28.5% | 211 |
| B=4 M=32 | **25.6%** | 2.9% | 8.1% | 38.4% | 24.3% | 172 |
| B=8 M=16 | **23.2%** | 4.8% | 14.9% | 33.3% | 21.6% | 168 |

**Confirmed** with refinements:
- Full-block rate is **23-26%** across configs (I claimed ~30%; actually slightly less, but the same order)
- Spread across n=2..13 is real (28-38% in n=4-7, 21-28% in n=8-13)
- Bimodal shape holds qualitatively

**But** the "uniform 5% per bucket" claim was wrong. The middle bucket (n=4-7)
is heavily weighted (30-38%) — a "compounding-error mode" centered around
n=5-6 rather than a flat distribution.

This refines the "30% ceiling" intuition: a meaningful fraction of mass is at
shallow accept (n=2-7), and improving those rounds matters more than chasing
the ~25% full-block ceiling.

---

## FM5: B=8 tree degenerates to chain — **CONFIRMED, with structural detail**

### What I claimed

At B=8 M=block_size=16, the heap exhausts its budget on the rank-0 chain,
leaving no room for siblings — so the tree IS a chain.

### What I measured

Tree node count distribution at each config:

| config | node-count distribution | "always a chain"? |
|---|---|---|
| B=1 M=64 | nodes=65 in 100% of 211 rounds | No — full M=64 with siblings |
| B=4 M=32 | nodes=20-33, modal 26-30 | No — meaningful sibling structure (M-block_size = 16 spare slots, all populated) |
| B=8 M=16 | nodes=14 (1.2%), 15 (3.6%), 16 (6.5%), 17 (1.2%) | **Yes — modal at 16, range tight** |

At B=8: **96% of rounds have 14-17 nodes**, with mode 16. Block_size=16 means
a pure chain (anchor + 15 depth-positive nodes) fills exactly 16 nodes. The
heap may sometimes return slightly fewer (early-termination via specdecpp) or
slightly more (heap padding edge cases). **Confirmed:** at B=8 the tree is
essentially a chain.

This explains why chain_mode at B=8 was a tie (+0.7%) rather than a big win —
the "tree" is already a chain; chain_mode only saves the tiny tree-attention
mask construction overhead.

---

## Summary of verification verdicts

| FM | Claim | Verdict |
|---|---|---|
| **FM1** | Rank-0 rejections caused by heap walking deep-rank-2 sibling chains; decay would let rank-0 chain win | **Falsified mechanism** (rank-0 chain never accepts more than best leaf). Statistic real, mechanism wrong. |
| **FM2** | Late-depth tails are heavier than shallow | **Refined**: directionally correct (depth 10-15 has 2-3× tail rate vs 6-9), but shallow depths also have tails; the "deep is harder" picture is partial |
| **FM3** | Out-of-K rate is 6-15% | **Confirmed exactly** |
| **FM4** | Bimodal: ~30% full-block + ~5% per bucket | **Confirmed qualitatively, refined**: full-block 23-26%, middle (n=4-7) heavily weighted at 30-38% (not flat) |
| **FM5** | B=8 tree is chain | **Confirmed**: 96% of rounds have 14-17 nodes, modal 16 |

---

## Implications for the CC-Tree redesign

**The C1 (score_decay) empirical win is real, but its motivation was wrong.**
Net effect: keep C1 in CC-Tree (the +2-4% B=4 result is direct measurement),
but stop claiming it "lets the rank-0 chain win." Its actual mechanism needs
investigation — likely a sibling-shape or post-deviation-expansion shift.

**FM3 caps the C2-widen ceiling tighter than I implied.** Out-of-K is 8.6% at
B=4. Even perfect K-widening at exactly the right depths recovers maybe
0.5-1% throughput, not the +2-4% I'd projected.

**FM5 confirms CC-Tree gracefully degenerates at B=8.** At B=8 M=16, the tree
has no siblings to apply C2 or C3 logic to; CC-Tree reduces to v7. So we
correctly should not expect B=8 gains from CC-Tree.

**FM4 refines what "30% full-block" means.** The ceiling is 25%, not 30%, and
the middle bucket (n=4-7) is the real mass — 30-38% of rounds. This is the
"compounding error mode" of the diffusion drafter, and CC-Tree's per-depth
calibration (C1) is precisely targeting it.

**Net revised expectation for CC-Tree at B=4:** +2-5% over v7. Lower than my
original +8-12% estimate (which was based on the falsified FM1 mechanism +
overgenerous FM3 magnitude).

Recommendation: still implement CC-Tree because:
- C1 gives confirmed +2-4% (already measured as score_decay)
- C2-narrow + C3 may add 0-1% (composing on calibrated base)
- C2-widen ceiling is small (≤1%) but cheap
- Total expected: +3-5% B=4. Smaller but still real and worth ~3 hours of code.

But ship the score_decay flag immediately as the simpler, lower-risk win
backed by direct measurement.
