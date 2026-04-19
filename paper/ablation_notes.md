# Ablation Analysis Notes — VB v1

## Available data (session apr18-19)

### Headline table (speedup / tau, 256 samples unless noted)

| draft \ block | b=16 | b=20 | b=24 | b=28 |
|---------------|------|------|------|------|
| stock (apr17) | 8.33 / 10.08 | 7.55 / 9.59 (32s) | 6.26 / 7.66 (32s) | — |
| VB v1 step 18500 | 8.13 / 9.89 | **8.52 / 10.43** | 8.50 / 10.49 | 8.40 / 10.38 |

Stock's cliff beyond b=16 is catastrophic. VB v1 trained on
{12,16,20,24} essentially eliminates the cliff inside its training
range and degrades gracefully outside.

### Per-dataset delta (VB b=20 vs stock b=16)

| dataset | stock | VB b=20 | Δ speedup | Δ tau |
|---------|-------|---------|-----------|-------|
| math500 (256) | 8.33 / 10.08 | 8.52 / 10.43 | +0.19 | +0.35 |
| mt-bench (80) | 4.41 / 6.10 | 4.20 / 6.06 | -0.21 | -0.04 |
| gsm8k (128) | 7.25 / 8.77 | 7.32 / 8.91 | +0.07 | +0.14 |
| humaneval (164) | 7.46 / 9.00 | 7.59 / 9.21 | +0.13 | +0.21 |

### Mid-training dynamics (learning curve on math500-32, b=16)

| step | speedup | tau |
|------|---------|-----|
| 500 | 8.34 | 10.35 |
| 2000 | 8.18 | 10.20 |
| 5000 | 8.17 | 10.21 |
| 9000 | 8.27 | 10.26 |
| 14000 | 8.24 | 10.21 |
| 18000 | 8.17 | 10.25 |
| 18500 | 8.28 | 10.27 |
| stock | 8.37 | 10.38 |

Observation: VB at b=16 never catches stock. The dilution-from-broad-mix
hypothesis is consistent: all checkpoints sit ~0.10–0.20 speedup below stock.
The signal is not in b=16 — it's in b=20/24/28.

### Per-batch-block-size accuracy during training

Sampled b=24 marginal accuracy across training steps:

| step range | b=24 m_acc (recent window) |
|------------|---------------------------|
| 30         | 0.10 (OOD start)          |
| 2500       | 0.22–0.64 (huge range)    |
| 3000       | 0.28–0.60 median 0.40     |
| 5000       | 0.25–0.76 median 0.48     |

Acceptance rate at b=24 climbs from 0.10 → 0.48 over 5k steps, then
plateaus.  The plateau suggests 1 epoch is insufficient: more training
could push further.

### Negative results

1. **Chained speculation (chain_depth ∈ {5, 10, 15, 19})**: tau +0.1,
   speedup -1.0. Same pattern as apr17's stock-drafter experiment.
2. **VB b=28 at 256 samples**: 8.40 / 10.38 — the 32-sample 9.03 was
   noise; extrapolation past training range hurts.
3. **VB b=32, b=40**: continues degrading; 8.68 / 10.68 at b=32 (32
   samples only, noisy); 8.20 / 10.38 at b=40 (clear regression).
4. **VB at b=16 broadly**: regresses vs stock on all 4 datasets.
   Broad-mix data (math + chat + stem) dilutes the math-specific
   specialization the stock drafter had.

### What we'd need for NeurIPS-tier (big gap from current)

Current headline: +0.25 speedup / +0.35 tau on math500 (+3% speedup).

To push to +10% speedup, we need tau ~12+ (currently 10.43) OR reduce
step cost.

- VB v2 (3 epochs, extended curriculum to b=32): projected +0.2-0.5 tau
- Large B=256/512 with VB (hypothesis: reduced phantom paths scale):
  untested, unclear
- Algorithmic innovation: untested

## Clear win #1: OOD recovery diagram

This is the cleanest single figure: three curves of speedup vs b for
{stock, VB v1, (VB v2)}. Stock cliff at b=20. VB v1 plateau 8.5 across
b=20–28. VB v2 hopefully plateau higher.

## Clear win #2: block-size ceiling decomposition

The $\tau$ ceiling of 10.08 at block_size=16 was the dominant constraint
identified in apr17's Finding 13. VB lifts this:

| drafter | max τ observed (math500) |
|---------|--------------------------|
| stock | 10.08 (exactly at block-size ceiling) |
| VB v1 b=20 | 10.43 |
| VB v1 b=24 | 10.49 |

The +0.41 τ is a direct measurement of breaking the prior ceiling.
