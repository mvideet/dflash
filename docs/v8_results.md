# v8 Experiments — Session Apr 24

## Branch & Code

Branch `experiments/apr24-v8-joint-cond` off `experiments/apr18-neurips-vb`.

New code (against that base):

- `model/dflash_tree.py::build_v8_tree` (~260 lines) — two-stage node-budget builder:
  - Stage 1: v7-style heap enumeration over `score_core(u) = Σ log q_i(u_i) − β_e · Σ 1{dev} · conf_i`, producing a prefix-closed pool of `v8_pool_multiplier × B` candidates.
  - Stage 2 (fast): pool-reselect sort by `pool_score + γ · leaf_bonus − λ · sibling_rank`, greedy include with prefix closure. Preserves budget; runs in <2 ms for B=128.
  - Hierarchical Depth Cap: `v8_dev_depth_cost > 0` forbids expansion past `seq_len − devcount · cost`, directly blocking deep mixed-rank-deep-paths.

- `benchmark.py` CLI flags: `--tree-version 8`, `--v8-entropy-beta`, `--v8-leaf-gamma`, `--v8-overlap-lambda`, `--v8-pool-multiplier`, `--v8-dev-depth-cost`.

## Hardware

8× NVIDIA A6000; GPU 1 in use by another tenant (34 GB @ ~19% util), so all runs are 7× A6000. A6000 is comm-bound for Qwen3-4B (see `project_a6000_comm_bound.md`), so absolute numbers differ from the 8× A100 baselines in `program.md`.

## v7 baseline on this hardware

math500-64, temp=0, ek=8:

| B   | speedup | tau   | nodes |
|-----|---------|-------|-------|
| 128 | **7.84** | **9.98** | 129 |
| 256 | 5.72    | 10.17 | 257 |
| 512 | 3.67    | 10.48 | 513 |

Same phantom-path-collapse pattern as program.md Finding 11: tau keeps climbing, speedup craters past B=128.

mt-bench-40:

| B   | speedup | tau  | nodes |
|-----|---------|------|-------|
| 128 | **3.99** | **6.10** | 129 |
| 256 | 2.96    | 6.27 | 257 |

## Full v8 sweep (math500-64)

All speedups at temp=0, ek=8, 7× A6000.

### Phase A — entropy-gated β alone (pool=1, γ=0, λ=0)

| Config | B | β_e | speedup | tau | vs v7-B |
|--------|---|-----|---------|-----|---------|
| identity (sanity) | 128 | 0 | 7.93 | 9.98 | +0.09 / ±0 (noise) |
| A_beta2 | 128 | 2 | 7.77 | 9.81 | -0.07 / -0.17 |
| v8_b256_beta4 | 256 | 4 | 5.81 | 10.13 | +0.09 / -0.04 |
| v8_b256_beta8 | 256 | 8 | 5.76 | 10.00 | +0.04 / -0.17 |
| v8_b256_beta16 | 256 | 16 | 5.44 | 9.64 | -0.28 / -0.53 (too strong) |
| v8_b512_beta8 | 512 | 8 | 3.75 | 10.16 | +0.08 / -0.32 |
| v8_b512_beta16 | 512 | 16 | 3.59 | 9.83 | -0.08 / -0.65 |

### Phase B — Hierarchical Depth Cap (HDC)

| Config | B | dev-cost | speedup | tau | vs v7-B |
|--------|---|----------|---------|-----|---------|
| hdc_b128_d1 | 128 | 1 | 7.77 | 9.86 | -0.07 / -0.12 |
| hdc_b128_d2 | 128 | 2 | 7.75 | 9.80 | -0.09 / -0.18 |
| hdc_b128_d4 | 128 | 4 | 7.62 | 9.64 | -0.22 / -0.34 |
| hdc_b256_d1 | 256 | 1 | **5.87** | 10.20 | +0.15 / +0.03 ← best B=256 speedup |
| hdc_b256_d2 | 256 | 2 | 5.82 | 10.12 | +0.10 / -0.05 |
| hdc_b256_d4 | 256 | 4 | 5.60 | 9.74 | -0.12 / -0.43 |
| hdc_b256_d2+β=4 | 256 | 2 | 5.63 | 10.00 | -0.09 / -0.17 |

### Phase C — Stage-2 reselection (γ-leaf bonus + pool oversampling)

| Config | B | γ | pool×B | speedup | tau | vs v7-B |
|--------|---|---|--------|---------|-----|---------|
| s2_b128_g0p5_pm2 | 128 | 0.5 | 256 | 7.72 | 9.89 | -0.12 / -0.09 |
| s2_b128_g1_pm2   | 128 | 1.0 | 256 | 7.67 | 9.85 | -0.17 / -0.13 |
| s2_b128_g2_pm2   | 128 | 2.0 | 256 | 7.46 | 9.70 | -0.38 / -0.28 |
| s2_b128_g1_pm3   | 128 | 1.0 | 384 | 7.51 | 9.85 | -0.33 / -0.13 |
| s2_b128_g1_pm4   | 128 | 1.0 | 512 | 7.30 | 9.85 | -0.54 / -0.13 |
| s2_b128_g2_pm4   | 128 | 2.0 | 512 | 7.27 | 9.72 | -0.57 / -0.26 |
| s2_b256_g1_pm2   | 256 | 1.0 | 512 | 5.77 | **10.29** | +0.05 / +0.12 ← best B=256 tau |
| s2_b256_b4g1_pm2 | 256 | 1.0 | 512 (β=4) | 5.70 | 10.09 | -0.02 / -0.08 |
| s2_b256_b4g1_pm3 | 256 | 1.0 | 768 (β=4) | 5.48 | 10.09 | -0.24 / -0.08 |

### mt-bench-40 cross-check

| Config | B | speedup | tau | nodes |
|--------|---|---------|-----|-------|
| v7 | 128 | 3.99 | 6.10 | 129 |
| v8 identity | 128 | 4.03 | 6.10 | 129 |
| v8 HDC d=1 | 128 | 4.07 | 6.10 | 129 |
| v7 | 256 | 2.96 | 6.27 | 257 |
| v8 HDC d=1 | 256 | 3.01 | 6.26 | 257 |

## Summary and interpretation

**No v8 configuration beats v7 B=128 on either dataset beyond measurement noise.** The most promising candidate (HDC d=1) gave ±0.08 speedup on mt-bench-40 but −0.07 on math500-64, staying within single-rank sampling variance (~±0.1 with 40–64 samples × 7 ranks). Tau differences are similarly within noise.

### Why the phantom-path attack didn't unlock a new peak

1. **At B=128, v7 already concentrates budget on the argmax chain.** Finding 18 in program.md noted this. Entropy-gated β and HDC can only prune *marginal* candidates at B=128, not wholesale restructure selection.

2. **At B≥256, phantom-path pruning recovers tau but the doubled node count's comm cost dominates.** HDC d=1 at B=256 did trim phantom expansion (+0.15 speedup vs v7 B=256), and Stage-2 γ-reselection raised tau to 10.29 (+0.12), but even the best B=256 result (5.87) is 25% below v7 B=128 (7.84). On A6000, comm cost of 257 nodes is the hard ceiling.

3. **Entropy-gated β is additive and so does not escape the same top-B sort as v7**, just with a shifted rank order. Phase-A data show the shift is too small to reorder the first 128 candidates meaningfully; at larger β_e, legitimate high-scoring paths get over-penalised (β_e=16 costs 0.4 tau and 0.4 speedup).

4. **Stage-2 γ-reselection with pool>B changes selection but the non-argmax-chain leaves it promotes are mostly shallow**; at tight budgets (B=128) this trades 0.1 tau for nothing.

### What would actually beat v7 B=128 on this hardware

- **Reduce comm cost per node.** A tree-aware attention kernel, or fewer ranks per node (bucketed verification). This is engineering, not an algorithmic contribution.
- **Break the block_size=16 tau ceiling.** Draft retraining to handle b≥20 (see apr18 variable-block findings) raises tau directly. Already done in this repo's later branches (V5 / V6 partial-mask) and yields the session's real SOTA gains.
- **Target-side changes.** Smaller / distilled target; different speculative decoding primitive (e.g., cache-based verify skip). Out of scope for this session.

## What to do with this code

- `build_v8_tree` is kept as reference implementation of the submodular-lazy-greedy node-budget framework; useful if someone revisits the v8 merge plan on harder-to-collapse hardware (e.g., H100 where tau gains at B≥256 could actually pay back the comm cost).
- All flags default to a v7-identity path; enabling any of them is explicit opt-in.
- Sweep scripts (`run_v8_sweep.sh`, `run_v8_b256_now.sh`, `run_v8_hdc.sh`, `run_v8_s2_b128.sh`, `run_mtbench_final.sh`) and results aggregator (`summarize_v8.sh`) remain in the tree for reproducibility.
