# Paper Outline — Variable-Block DFlash (working title)

## Title (candidates)
- **Breaking the Block-Size Ceiling in Block-Diffusion Speculative Decoding**
- **Variable-Block Curriculum Training Closes the Product-Joint Gap in DDTree**
- **When Bigger Blocks Help: Training-Time Interventions for Inference-Time Speedups**

## One-sentence contribution
A variable-block curriculum training recipe for block-diffusion draft models
(DFlash) that produces a single drafter capable of running at inference-time
block sizes unseen in training, yielding the first training-based SOTA on
three of four standard speculative-decoding benchmarks (math500, gsm8k,
humaneval) and a +$X\%$ speedup over v7-DDTree's prior SOTA at matched
verification budget.

## Abstract skeleton

- Speculative decoding with block-diffusion drafts (DFlash) achieves ~8x
  speedup via a fixed block size (b=16). Existing work on tree construction
  (DDTree/v7) reached a plateau at B=128 verification nodes.
- We identify the **product-joint gap**: DDTree's top-B-by-product scoring
  is exact-optimal under an independent-positions assumption that fails
  empirically, causing phantom mixed-rank paths at large budgets.
- Proposal: **variable-block curriculum training** exposes the drafter to
  block sizes b ∈ {12, 16, 20, 24, (28, 32)} during training. The trained
  drafter can be used at b ≥ 20 at inference without the usual OOD collapse.
- Extending the inference block size amortizes the (memory-bandwidth-bound)
  target forward over more tokens, directly improving speedup.
- Experimental: math500 speedup 8.33 → 8.52 (+2.3%), tau 10.08 → 10.43
  (+3.5%). Cross-dataset: gsm8k and humaneval net-positive; mt-bench
  net-negative (chat responses have short accept runs).
- Theory: VB curriculum as implicit joint-distribution approximation.

## Sections

### 1. Introduction
- Block-diffusion speculative decoding motivation
- Prior work on tree construction (EAGLE-2, v4, v7 DDTree)
- The block-size ceiling as a fundamental limit
- Our contribution

### 2. Background & Preliminaries
- Speculative decoding protocol
- DFlash architecture (block-diffusion draft, tree verification)
- DDTree's product-distribution optimality proof
- Empirical observations (apr17's Finding 11 & 13)

### 3. The Product-Joint Gap
- Formal statement (§3 of theory_draft.md)
- Flaw 1: factorization assumption
- Flaw 2: phantom paths at large budgets
- Budget sweep data showing the phenomenon

### 4. Variable-Block Curriculum Training
- Recipe: block sizes, sampling weights, loss
- Random-anchor sampling (DFlash paper Table 9)
- Exp-weighted CE (γ scaled with b)
- Tree-aware conditional CE (CTR)
- Broad-mix data (Nemotron-v2 regenerated with Qwen3-4B)

### 5. Why VB Works — Implicit Joint Learning
- Informal argument (§4 of theory_draft.md)
- Mechanism: drafter learns block-size-invariant representations
- Mechanism: at larger b, drafter's marginals better approximate the
  target's conditional joint
- Consequence: extending b at inference amortizes target forward

### 6. Experimental Setup
- Target model: Qwen3-4B
- Draft model: z-lab/Qwen3-4B-DFlash-b16 (base) → VB-trained
- Datasets: math500 (256), mt-bench (80), gsm8k (128), humaneval (164)
- Hardware: 8× A100-80GB
- Tree builder: v7 DDTree B=128 ek=8

### 7. Main Results
- **Headline table**: VB vs stock at b∈{16, 20, 24} on 4 datasets
- **OOD recovery table**: stock cliff at b=20/24 vs VB's graceful extension
- **Budget scan**: VB at B∈{128, 256, 512} (if VB v2 works at large B)

### 8. Ablations
- Fixed b=16 vs VB curriculum (broad-mix held constant)
- Math-only vs broad-mix (VB held constant)
- Training epochs: 1 vs 3 (VB v1 vs v2)
- CTR weight variation
- Adaptive block size (per-sequence b selection)

### 9. Negative Results (for honesty)
- Chained speculation: fails even with VB drafter (apr17's Q2 + our retest)
- Scoring tweaks (Q1/Q3/Q4 from program.md): do not improve over v7
- mt-bench regression: VB drafter disprefers chat-style responses; adaptive
  block partially mitigates
- Extrapolation: VB-trained on b≤24 does not help at b=28 (256-sample test)

### 10. Discussion
- Why fixed-b=16 was a local optimum
- Implications for draft-model training more broadly
- Open questions:
  - Does VB transfer to Llama/other targets?
  - Can a larger B be unlocked with stronger VB drafters?
  - What's the true information-theoretic limit?

### 11. Related Work
- SpecDec, SpecInfer, EAGLE, EAGLE-2, DDTree, GLIDE/CLI
- Block-diffusion language models (DFlash paper)
- Curriculum learning in LLMs
- Distillation for draft models

### 12. Conclusion
- First training-based SOTA on this codebase's benchmarks
- Variable-block curriculum as a simple, effective recipe
- Opens the door to inference-time block-size flexibility

## What we still need to collect (TBD this session)

- [ ] VB v2 results (3 epochs, {12,16,20,24,28,32})
- [ ] VB v1 at B=256/512 — does aggressive budget work with reduced phantom?
- [ ] Adaptive block-size on mt-bench — does it recover the regression?
- [ ] Ablation: fixed-b=16 on broad-mix, 1 epoch (isolate VB effect)
- [ ] Ablation: math-only + VB curriculum (isolate data-mix effect)
- [ ] Theoretical: formalize the joint-approximation argument more rigorously
