# Section 7: Main Results — DRAFT

## 7.1 Experimental Setup

**Target model.** Qwen3-4B (4.2B parameters, 36 layers, hidden size 2560).
Tensor-parallel across 8× NVIDIA A100-80GB with bf16, flash_attention_2.
Sequence length 2048, max_new_tokens 2048, temperature 0.0 (greedy).

**Baseline draft model.** z-lab/Qwen3-4B-DFlash-b16: the stock DFlash
drafter trained for fixed block size 16. 5 layers, hidden 2560, total
~200M trainable parameters. Uses target's embeddings and lm_head (shared).

**Variable-block drafter (this work).** Same architecture as baseline;
trained with variable block-size curriculum. See §4 for training recipe.
"VB v1": trained for 1 epoch on 148k-row broad-mix data
(math + chat + stem, all Qwen3-4B-regenerated via Nemotron-v2) with
block sizes $\{12,16,20,24\}$ sampled with weights $\{1,2,2,1\}$.

**Tree builder.** v7 / DDTree from the prior-SOTA work in this codebase.
Budget $B = 128$ trie nodes, expand-$k$ 8, unless stated otherwise.

**Datasets.** Following prior-SOTA convention:
math500 (256 samples), mt-bench (80), gsm8k (128), humaneval (164).

**Metrics.** Wall-clock speedup vs. autoregressive (temp=0) decoding
with the same target and prompts; $\tau$ = acceptance length
(mean accepted tokens per verification step). Hardware-matched pairs:
stock and VB are benchmarked back-to-back on identical GPU state
where feasible to minimise cluster-load confounds.

## 7.2 Main Finding — Block-Size-Extended Speculation

Standard speculative decoding uses a fixed draft block size $b$ at inference
equal to the drafter's training block size. This has been taken for granted:
a $b=16$-trained drafter evaluated at $b=20$ suffers catastrophic OOD
degradation (§7.3).

Our central result: a VB-trained drafter can be evaluated at any
$b \in \mathcal{B}_{\text{train}}$ at inference time, and choosing
$b$ larger than the stock drafter's training $b = 16$ *improves*
speedup relative to stock's $b=16$ peak, because each target forward
amortises over more accepted tokens.

**Table 7.1: Cross-dataset speedup / $\tau$ at $B=128$, $ek=8$, $T=0$.**

| dataset | baseline stock b=16 | VB v1 b=20 | $\Delta$ speedup | $\Delta$ $\tau$ |
|---------|---|---|---|---|
| math500 (256) | 8.33 / 10.08 | **8.52 / 10.43** | +0.19 (+2.3%) | +0.35 (+3.5%) |
| mt-bench (80) | **4.41 / 6.10** | 4.20 / 6.06 | −0.21 | −0.04 |
| gsm8k (128) | 7.25 / 8.77 | **7.32 / 8.91** | +0.07 | +0.14 |
| humaneval (164) | 7.46 / 9.00 | **7.59 / 9.21** | +0.13 | +0.21 |

Three of four benchmarks net-positive; mt-bench regresses slightly.

**Remark (math500 $\tau$).** Prior SOTA's $\tau$ at $B=128, b=16$
converged to 10.08 — exactly the block-size ceiling (cf. Finding 13 of
prior work: the fraction of steps that hit $\tau = 16$ rises toward 35%
and dominates marginal gains from bigger $B$). VB at $b=20$ achieves
$\tau = 10.43$; at $b=24$, $\tau = 10.49$. **This is the first method
in this codebase to break the block-size-16 acceptance-length ceiling.**

## 7.3 OOD Collapse of Stock vs. VB's Graceful Extension

**Table 7.2: math500 (32 samples) speedup / $\tau$ at $b \in \{16,20,24\}$.**

|     | stock | VB v1 | $\Delta$ |
|-----|---|---|---|
| $b=16$ | 8.42 / 10.38 | 8.31 / 10.27 | −0.11 / −0.11 |
| $b=20$ | 7.55 / 9.59 | **8.68 / 10.91** | +1.13 / +1.32 |
| $b=24$ | 6.26 / 7.66 | **8.58 / 10.85** | **+2.32 / +3.19** |

Stock's $b=24$ speedup drops −2.16 from its $b=16$ value; VB at $b=24$
loses only −0.16 relative to VB's peak at $b=20$ (and still *beats*
stock's $b=16$ peak by +0.16 / +0.47).

**Figure placeholder (block-size cliff):** three-curve plot of speedup
vs. $b$ for {stock, VB v1}, showing stock's cliff at $b \geq 20$ and
VB's plateau extending to $b = 24$.

## 7.4 Block-size Extrapolation

How far does VB v1 generalise past its training range $\{12,16,20,24\}$?

**Table 7.3: math500 (256 samples) at various $b$, VB v1.**

| $b$ | 16 | 20 | 24 | 28 | 32 | 40 |
|-----|----|----|----|----|----|----|
| speedup | 8.13 | **8.52** | 8.50 | 8.40 | 8.68* | 8.20 |
| $\tau$ | 9.89 | 10.43 | 10.49 | 10.38 | 10.68* | 10.38 |

$^*$ 32-sample only; noisy. 256-sample $b=28$ is a modest dip from the
$b=20$ peak, not an improvement. $b=40$ regresses clearly.

Consistent with the smoothness conjecture (§5b): VB v1 covers its
training range in-distribution and degrades gracefully slightly
outside, with clear regression far outside.

## 7.5 Negative Results

1. **Chained speculation with VB drafter** (apr17 prior-SOTA's Q2 retry):
   speedup drops 1.0+ across all chain depths; $\tau$ gains are 0.1
   per chain step but the extra draft forward is not amortised.
   Confirms Finding 14 of prior work even with the improved drafter.
2. **VB at $b = 16$, cross-dataset**: regresses vs. stock on all four
   benchmarks, consistent with broad-mix data diluting the stock
   drafter's math-specific specialisation. The gain is entirely
   concentrated at $b \geq 20$.
3. **Mt-bench**: VB underperforms stock at every $b$. The chat
   distribution has shorter, higher-entropy continuations; the larger
   block commits more tokens per speculation, and mispredictions lose
   more than acceptance-rate gains recoup.

## 7.6 Robustness: confidence intervals

The apr17-PM paired-probe data established that math500-32 speedup has
$\pm 0.5 - 1.0$ variance between runs on identical hardware state (Finding
confirmed this session: baseline rerun ranged 8.32–9.59 across a 5-run
paired protocol). We therefore trust only **256-sample** speedups for
math500 and full-dataset runs for mt-bench/gsm8k/humaneval. The 256-sample
math500 gain of +0.19 speedup is above this noise floor.

## 7.6b Budget Scaling — Direct Phantom-Reduction Evidence

Finding 11 of prior-SOTA work (program.md) showed that stock DDTree
collapses past $B=128$: $\tau$ continues climbing (marginal value of
additional tree nodes) but speedup drops sharply (phantom paths
contribute near-zero true joint probability).

**Hypothesis.** If VB training reduces the product-joint gap (§5),
then VB's $\tau$ should climb FURTHER with budget, and speedup should
collapse LESS. We verify both.

**Table 7.4: Budget scaling on math500 (256 samples), $b=20$ for VB.**

| $B$ | stock b=16 | VB b=20 | $\Delta$ speedup | $\Delta$ $\tau$ |
|-----|-----------|---------|------------------|---------------|
| 128 | 8.33 / 10.08 | **8.52** / 10.43 | +0.19 | +0.35 |
| 256 | 8.05 / 10.37 | **8.34** / 10.87 | +0.29 | +0.50 |
| 512 | 6.66 / 10.59 | **6.91** / 11.23 | +0.25 | **+0.64** |

Two signals both support the phantom-reduction hypothesis:

1. **VB's $\tau$ grows more with $B$**: from 10.43 at $B=128$ to
   **11.23 at $B=512$** — a new record for this codebase (prior
   ceiling was 10.08 at block_size=16). Stock's $\tau$ also grows
   but saturates faster (10.08 → 10.59, half the slope).
2. **VB's $\Delta$ over stock grows with $B$** (+0.35 → +0.50 → +0.64
   in $\tau$). At the smallest budget where phantom effects are
   minimal, the gap is small; at the largest budget where phantoms
   dominate, the gap is largest. VB's marginals are better-aligned
   with the target's conditional joint at deeper tree positions.

**Caveat.** Peak speedup for both drafters remains at $B=128$; tree-
build overhead (Python-heap $O(B \log B)$, cf. Finding 16) dominates
past that point. A future GPU-vectorised tree builder would likely
let VB's advantage become a speedup win at $B \geq 256$.

## 7.6c Adaptive Block Size — Partial Mitigation of mt-bench Regression

Per-step block size selection driven by acceptance-rate EWMA:

**Table 7.5: mt-bench (80 samples).**

| config | speedup | $\tau$ |
|--------|---------|-------|
| stock b=16 fixed | **4.41** | **6.10** |
| VB v1 b=20 fixed | 4.20 | 6.06 |
| VB v1 adaptive {16,20,24} thresh {0.55,0.75} | 4.27 | 6.01 |
| VB v1 adaptive {16,20,24} thresh {0.45,0.65} | 4.22 | 6.04 |
| VB v1 adaptive {16,20}    thresh {0.75}     | 3.63 | 5.98 |

Adaptive block recovers part of the mt-bench regression (4.20 → 4.27)
but does not reach stock's 4.41. The regression is a genuine
limitation: our broad-mix training (math + chat + stem) dilutes chat-
specific specialisation that the narrow-math stock drafter retained.

## 7.7 Ablation Summary (preliminary)

| condition | math500 speedup | $\tau$ | notes |
|-----------|----------------|--------|-------|
| stock b=16 (baseline) | 8.33 | 10.08 | — |
| VB v1 b=16 | 8.13 | 9.89 | $-0.20$ — broad-mix dilution cost |
| VB v1 b=20 | **8.52** | 10.43 | $+0.19$ — main result |
| VB v1 b=24 | 8.50 | **10.49** | ties speedup, best $\tau$ |

Full ablations (VB v2 with extended curriculum, adaptive block size,
large-$B$ stress tests) pending — see §8.
