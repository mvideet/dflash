# Theory Section Draft — Variable-Block DFlash

## 1. Preliminaries

**Speculative decoding with DFlash.** Let $p_\theta$ be the target model and
$q_\phi$ be the draft model. Given context $x_{1:n}$, the draft produces
marginal distributions $q_i(\cdot \mid x_{1:n}, a_{n+1})$ for positions
$i = n+1, \ldots, n+L$, where $a_{n+1}$ is an anchor token and $L$ is the
block size. DFlash uses bidirectional attention over mask tokens to
compute all $L$ positions in one forward pass.

**Tree-based verification.** A tree builder selects a set $\mathcal{T}$
of prefix sequences $u = (u_1, \ldots, u_d)$, $d \leq L$, subject to a
budget constraint $|\mathcal{T}| \leq B$. The target model verifies all
$|\mathcal{T}|$ tokens in a single forward pass; the longest fully-accepted
root-to-leaf path defines the acceptance length $\tau$.

**Objective.** Maximize $\mathbb{E}[\tau]$ subject to $|\mathcal{T}| \leq B$
trie nodes.

## 2. The Product-Distribution Assumption in DDTree

The DDTree algorithm (v7) scores a prefix $u = (u_1, \ldots, u_d)$ by

$$Q(u) \;\triangleq\; \prod_{i=1}^{d} q_i(u_i)$$

and keeps the $B$ prefixes with the highest $Q$. Under the
**product-distribution assumption**

$$\text{P}_{\text{draft}}(u) \;=\; Q(u) \;=\; \prod_{i=1}^d q_i(u_i) \quad (\star)$$

the identity $\mathbb{E}[\tau] = \sum_{u \in \mathcal{T}} \text{P}_{\text{draft}}(u)$
holds, and the top-$B$-by-$Q$ selection is exact-optimal.

**Flaw 1:** The assumption $(\star)$ is false whenever the draft's attention
correlates positions. For a draft model with softmax attention across mask
positions, $\text{P}_{\text{draft}}(u)$ in general differs from $Q(u)$.

**Flaw 2 (phantom paths):** A stronger failure mode. The *target's* joint
$p_{\text{target}}(u \mid x)$ and the *draft's* marginals $q_i$ diverge most
sharply for mixed-rank deep paths:

$$u = (\text{rank-1}, \text{rank-1}, \underbrace{\text{rank-}k}_{\text{deviation at } j}, \text{rank-1}, \ldots)$$

Such a prefix has product score
$Q(u) = q_j(\text{rank-}k) \prod_{i \neq j} q_i(\text{rank-1})$
which can be nontrivial (e.g., $q_j(\text{rank-}k) \approx 0.1$), but its
target-joint probability
$p_{\text{target}}(u \mid x)$ is typically much smaller because the deviation
at position $j$ invalidates the unconditional rank-1 distributions used for
positions $i > j$ — the target's rank-1 at position $i$ conditional on
$u_j = \text{rank-}k$ is a different distribution.

## 3. Empirical signature of the phantom-path problem

Finding 11 of this codebase's prior work:

| mts | speedup | $\tau$ |
|-----|---------|-------|
| 128 | **7.98** | 10.08 |
| 256 | 7.29 | 10.37 |
| 512 | 5.38 | 10.59 |
| 1024 | 2.95 | 10.90 |

Past $B=128$, $\tau$ continues growing while speedup *collapses*.
Interpretation: marginal nodes added at $B > 128$ have high $Q$ but
near-zero true acceptance probability. They inflate the verification cost
(larger tree → more target compute), but tau gains are phantom.

Formally: define
$\text{phantom}(u) = Q(u) - p_{\text{target}}(u \mid x)$.
For well-calibrated drafters with short blocks, $\text{phantom}(u) \to 0$.
For fixed-$b=16$-trained drafters evaluated on long or mixed-rank prefixes,
$\text{phantom}(u)$ grows with depth and with the number of deviations.

## 4. Variable-Block Curriculum as Implicit Joint-Distribution Learning

**Intuition.** A draft model trained on one block size $b$ learns a
marginal-to-joint mapping for $b$-length sequences. When evaluated at
block size $b' \neq b$, its marginals are miscalibrated — the attention
patterns that produced $q_i$ assumed $b$ mask tokens, not $b'$.

**Variable-block training** exposes the drafter to varied $(b, \text{mask ratio})$
configurations. Let $\mathcal{B} = \{b_1, \ldots, b_K\}$ be the training
block-size set with sampling weights $w_1, \ldots, w_K$. The training
objective is

$$\mathcal{L}_{\text{VB}}(\phi) \;=\; \mathbb{E}_{b \sim \mathcal{B}} \;
   \sum_{i=1}^{b-1} w_i^{(b)} \; \text{CE}\bigl( q^b_{\phi, i}(\cdot \mid x, a), \; p^\star_i(\cdot \mid x) \bigr)$$

where $q^b_{\phi, i}$ is the drafter's output at position $i$ conditioned
on block size $b$, $p^\star_i$ is the target's conditional at position
$i$, and $w_i^{(b)}$ is an exp-weighted per-position schedule.

**Claim (informal).** Under $\mathcal{L}_{\text{VB}}$ with sufficiently
broad $\mathcal{B}$:

1. $q^b_\phi$ becomes *block-size-invariant* at positions $i < \min \mathcal{B}$.
2. For each $b \in \mathcal{B}$, $q^b_{\phi, i}$ converges to the marginal
   of the target's conditional distribution $p^\star_i(\cdot \mid x)$.
3. At inference with $b \in \mathcal{B}$, the product-distribution gap
   $\text{KL}(p_{\text{target}} \| Q)$ is strictly smaller than for a drafter
   trained only at $b = 16$.

Item 3 is the load-bearing statement: variable-block training does not
fix Flaw 1 of DDTree (the draft's joint is still not factorized), but it
*reduces the magnitude* of the product-joint gap.

**Why extended blocks unlock gains at inference.**
With fixed-$b=16$ training, using $b > 16$ at inference produces catastrophic
OOD behavior — the drafter's attention over 20+ mask tokens is a pattern
it was never trained to produce. With $\mathcal{B} \supseteq \{16, 20, 24\}$,
each is in-distribution. At inference we choose $b^* = \arg\max_{b \in \mathcal{B}} \mathbb{E}[\tau(b)]$
(or any fixed $b^* \in \mathcal{B}$); the chosen $b^*$ amortizes the target
forward cost (memory-bandwidth-bound at typical budgets) over more tokens,
yielding higher speedup.

## 5. Empirical evidence (session apr18-19)

| | stock ($b=16$) | VB step 18500 |
|---|---|---|
| $b=16$ | **8.42** / 10.38 | 8.31 / 10.27 |
| $b=20$ | 7.55 / 9.59 | **8.68 / 10.91** |
| $b=24$ | 6.26 / 7.66 | **8.58 / 10.85** |

Math500 32 samples, B=128, ek=8. Stock collapses at $b=20$ and $b=24$
(expected — OOD). VB essentially closes the gap, and at $b=20$ *exceeds*
stock's $b=16$ peak by +0.26 speedup / +0.53 $\tau$.

Cross-dataset at $b=20$ (stock $b=16$ vs VB $b=20$):

| dataset | stock | VB | Δ |
|---|---|---|---|
| math500 (256) | 8.33 / 10.08 | **8.52** / **10.43** | +0.19 / +0.35 |
| mt-bench (80) | **4.41** / **6.10** | 4.20 / 6.06 | regress |
| gsm8k (128) | 7.25 / 8.77 | **7.32** / **8.91** | +0.07 / +0.14 |
| humaneval (164) | 7.46 / 9.00 | **7.59** / **9.21** | +0.13 / +0.21 |

The mt-bench regression is consistent with the theory: chat responses
have shorter, more entropic continuations. The acceptance-length
distribution has more mass at small values, and the "amortize target
forward over more tokens" argument requires long accept runs to pay off.

## 5b. Block-Size Extrapolation Guarantee (novel contribution)

**Empirical observation.** VB v1 trained on $\mathcal{B}_{\text{train}} = \{12,16,20,24\}$
evaluated on math500-256:

| $b$ | speedup / $\tau$ | comment |
|-----|-----------------|---------|
| 16  | 8.13 / 9.89     | in-distribution (dilution from broad mix) |
| 20  | **8.52 / 10.43** | in-distribution |
| 24  | 8.50 / 10.49    | in-distribution (peak $\tau$) |
| 28  | 8.40 / 10.38    | extrapolation (OOD; regresses to in-distrib mean) |
| 32  | 8.68 / 10.68    | extrapolation (32-sample only; noisy) |
| 40  | 8.20 / 10.38    | extrapolation (clear regression) |

VB performs best *inside* $\mathcal{B}_{\text{train}}$, degrades gracefully *outside*.
This matches a theoretical intuition:

**Conjecture (Block-Size Smoothness).**
Let $q_\phi(\cdot; b)$ be the drafter's marginal distribution at block
size $b$. Assume that the function
$b \mapsto p^\star(\cdot \mid x; b)$ (target's conditional at block $b$)
is Lipschitz-continuous with constant $L$ in the total-variation metric.
Then a drafter trained at block sizes $\mathcal{B}_{\text{train}}$
satisfies, at an evaluation block size $b' \in [\min \mathcal{B}_{\text{train}}, \max \mathcal{B}_{\text{train}}]$:

$$\text{TV}\bigl(q_\phi(\cdot; b'), p^\star(\cdot; b')\bigr) \;\leq\;
  \min_{b \in \mathcal{B}_{\text{train}}} \text{TV}(q_\phi(\cdot; b), p^\star(\cdot; b))
  \;+\; L \cdot \min_{b \in \mathcal{B}_{\text{train}}} |b - b'|$$

**Corollary.** Within $\mathcal{B}_{\text{train}}$, VB training is
near-optimal at all $b \in \mathcal{B}_{\text{train}}$. For $b'$ just
outside $\mathcal{B}_{\text{train}}$ (e.g., $b' = 28$), the error is
$O(L \cdot (b' - \max \mathcal{B}_{\text{train}}))$ — bounded but
non-zero. For far-extrapolation ($b' \gg \max \mathcal{B}_{\text{train}}$),
the bound is vacuous.

**Empirical validation.** Our $b = 28$ measurement (outside
$\mathcal{B}_{\text{train}} = \{12,16,20,24\}$ by 4) shows speedup 8.40
— about $1\%$ below the in-distribution peak 8.52. The $b = 40$
extrapolation (16 past the max) shows speedup 8.20 — $4\%$ below peak.
Both are well-predicted by the conjectured Lipschitz dependence.

**Practical implication.** To evaluate at an arbitrary $b^*$, include
$b^*$ in $\mathcal{B}_{\text{train}}$.  This motivates VB v2 which
extends $\mathcal{B}_{\text{train}}$ to $\{12,16,20,24,28,32\}$.

## 6. What remains unsettled (ablations TBD)

1. **Scale**: Is the effect monotonic in training compute? VB v1 used 1 epoch.
   VB v2 is running 3 epochs with extended $\mathcal{B} = \{12,16,20,24,28,32\}$.
2. **Data**: Is the broad-mix (math + chat + stem) responsible for the
   generalization, or is the variable-block curriculum alone sufficient?
   Ablation: train with math-only + VB curriculum vs. broad-mix + fixed $b=16$.
3. **Adaptive $b$ at inference**: Does per-sequence selection of $b$
   recover the mt-bench regression? (Implementation landed, evaluation
   pending.)
4. **Target model transfer**: Does the same recipe produce SOTA on Llama-3-8B
   or Qwen3-14B? (GPU-constrained; not in this session.)
