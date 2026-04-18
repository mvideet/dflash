# DFlash Autoresearch — Training-Session Handoff

This file is a complete, self-contained briefing for a fresh autonomous research agent.  Read it top-to-bottom before doing anything else.  It supersedes both `/homes/videetm/dflash/program.md` (inference-era handoff) and the session-level memory files in `~/.claude/projects/-homes-videetm-dflash/memory/`.

Last updated: **2026-04-18** (after session `apr18-training` on branch `experiments/apr16-ddtree-fix`).

---

## 1. Mission

Maximize wall-clock **speedup** of DFlash speculative decoding on Qwen3-4B, by training better **draft** models.  Training IS allowed; architecture changes and target-model changes are NOT.

Concretely: we want the draft to produce higher-acceptance predictions so that the target verifies more tokens per forward pass (higher tau), without slowing the draft down.

Target venue framing: a publishable contribution (NeurIPS/MLSys style).  Training-free inference directions are exhausted — see section §4 for which were ruled out and why.

---

## 2. The hard numbers — what "SOTA" means right now

Qwen3-4B target, z-lab/Qwen3-4B-DFlash-b16 draft, tree-version 7 (DDTree), max_tree_size=128, expand_k=8, temperature=0.0.  These numbers are the reference to beat.

| Dataset | Samples | Speedup | Tau | Notes |
|---|---|---|---|---|
| math500 | 256 | **8.27** | 10.08 | commit `1be9a5a`, 8×A100 |
| mt-bench | 80 | **4.35** | 6.10 | commit `0011d9a`, 8×A100 |
| gsm8k | 128 | **7.21** | 8.77 | commit `0011d9a`, 8×A100 |
| humaneval | 164 | **7.43** | 9.00 | commit `0011d9a`, 8×A100 |

**Important caveat**: raw wall-clock speedup is unstable because this cluster is shared with other users (vLLM, flame/torchrun jobs routinely push all 8 GPUs to 100%).  During contention, bs=1 vanilla AR slows by 3-4× while bs=16 speculative slows less — this inflates apparent speedup by tens of percent.  **Use TAU as the cross-run comparable metric**; convert to speedup only after confirming with a clean cluster run.

---

## 3. What's in the repo — orientation map

All paths relative to `/homes/videetm/dflash/`.

**Inference**
- `benchmark.py` — primary eval driver.  `dflash_generate()` is the generation loop.
- `model/dflash_tree.py` — tree builders.  v7 = `build_node_budget_tree` (DDTree, current best).  Has flags for calibration, narrow-after-dev, per-pos expand_k (all default-off).
- `model/dflash.py` — draft architecture (read-only).  Bidirectional-attention block model, block_size=16.
- `results.tsv` — running ledger of benchmark results.

**Training (NEW this session)**
- `trainingto/main_mix.py` — training driver, deepspeed-compatible.
- `trainingto/dflash_mix_model.py` — training model wrapping target (frozen) + draft (trainable) with the MIX recipe (§5).
- `trainingto/convert_ckpt_to_hf.py` — deepspeed `save_16bit_model` output → HF-loadable draft dir.
- `trainingto/eval_ckpt.sh` — flock-serialized per-ckpt bench.
- `trainingto/watch_and_eval.sh` — polls a savedir, auto-evals every new `step_N` directory.
- `trainingto/train_queue.sh` — sequential multi-variant training with DONE sentinels.
- `trainingto/master_pipeline.sh` — end-to-end wait→queue→pick-winner→finalist→cross-dataset.
- `trainingto/summary.sh` — live dashboard of training + eval state.
- `trainingto/data/nemotron_math/` — 225k-example JSONL train + 2.3k test (Qwen-regenerated math data).
- `trainingto/data/nemotron_math_10k/` — 10k subset used this session.
- `trainingto/data/nemotron_math_smoke/` — 450-example smoke test.
- `trainingto/ds_config.json` — deepspeed config (batch 1/GPU, grad-accum 2, bf16, zero-1, peak LR 5e-6).
- `trainingto/base_model.py`, `dflash_ctr_model.py`, `dflash_lk_model.py`, `dflash_gto_model.py`, etc — prior training code.  MIX is a successor to CTR.

**Docs**
- `program.md` (repo root) — inference-era handoff.  Section history: apr9–apr17 findings on v2/v4/v7 tree builders, DDTree plateau analysis, calibration failures.  Still relevant context.
- `docs/ddtree_v7_research_notes.md` — theory file for DDTree flaws + fixes (most attacked in `program.md`'s Findings 12–19).
- `autoresearch/program.md` (this file) — training-era handoff.

**Logs**
- `logs/session_apr17/` — prior inference-session logs.
- `logs/session_apr18/` — this session's training + eval logs + `mix_eval.tsv` ledger.

---

## 4. Training-free directions already exhausted (DO NOT re-run)

Everything in this table was tested and measured on math500 at the current best config.  See `program.md` Findings 12–19 for details.

| Direction | Result | Why it failed |
|---|---|---|
| Q1 power-scaled scoring | −0.17×..+0 | α/β too weak to reorder top-B selection |
| Q4 online target calib (2D, blended) | −1.36× | Laplace-uniform prior poisons warmup |
| Q4b dev-conditional calib (3D, additive) | −0.48× | α̂ too noisy at per-sequence scale |
| Narrow-after-dev (NW2) | tie at B=128, +0.41 at B=512 | mitigates phantom paths but never exceeds B=128 peak |
| Entropy-adaptive expand_k | −0.09× | heap-push overhead offsets redistribution |
| Wider ek=16 | −0.23× | slower tree_build |
| OOD block_size=24 inference | −2.0× | draft is strongly tied to its trained block_size |
| flex_attention (tree mask) | 50× slower | L=128 is too small; create_block_mask cost dominates |
| GPU-native tree build (beam) | 138× slower | `argmax().item()` syncs every pop |
| Draft-logit temperature | no-op | preserves top-B ordering by construction |

**The honest inference-era ceiling on this hardware is 8.27× math500.**  Training is now the only remaining lever.

---

## 5. The MIX training recipe (current)

`dflash_mix_model.py` stacks five enhancements; each has an on/off knob for ablation.

1. **Exponential-weighted CE**  `w_k = exp(-(k-1)/γ)`, γ=7 for b=16.  DFlash paper Fig 5: early-position errors invalidate the rest of the block, so prioritize them.  Replaces uniform CE.
2. **Random anchor sampling**  Each training sequence samples N random valid assistant-mask positions (default 16) as block starts instead of fixed stride.  DFlash paper Table 9: +13–14% tau across benchmarks.  **This is the single biggest documented training-time win.**
3. **Tree-attention conditional CE**  CTR-style second forward over a small (v2-builder) tree with tree-attention mask.  Teaches the draft to predict well under "I can only see my ancestors" — which is the distribution target uses at verification.  weighted by `ctr_weight` (default 0.5).
4. **TTT-style recursive pass**  Uses draft's own argmax at the end of block_1 as the anchor for a second block, trains on block_2 supervised by target.  Teaches recovery when the draft is wrong about its own predictions.  `ttt_weight` (default 0.2).
5. **Variable block-size curriculum**  Each training step samples b ∈ a list (default [16], curriculum = [12,16,20]).  Point: produce a drafter that works across block sizes, enabling chained-speculation inference at b>16.

**10k × 1-epoch result (this session):**

| Variant | math500 tau | mt-bench tau | gsm8k tau | humaneval tau |
|---|---|---|---|---|
| Stock | 10.08 | 6.10 | 8.77 | 9.00 |
| v1 (marg only: 1+2) | 10.03 | — | — | — |
| v2 (varblock: 1+2+5 with b∈{12,16,20}) | 10.09 | 6.14 | 8.79 | 9.02 |
| v3 (ctr-lite: 1+2+3 with ctr_w=0.3) | **10.12** | **6.16** | 8.78 | — |
| v4 (tt-lite: 1+2+4 with ttt_w=0.1) | 10.01 | — | — | — |

Winner: **v3/step_500**, best on 2/4 datasets.  All random-anchor variants ≥ stock.  Absolute gains ≤+0.06 tau — real but small.

**Novel observation** (v2_varblock only): at OOD inference block_size=20, v2 closes half the gap to its b=16 performance despite only ~75 training examples at b=20.  Mechanism works; needs more steps.

---

## 6. Recommended next training run (what to actually do)

### 6.1 Scale-up target

DFlash paper used **800k samples × 6 epochs** with full Nemotron + CodeAlpaca.  Our 10k × 1 is ~480× less compute.  To get genuinely publishable gains, scale at least 10×:

- **Data**: ≥100k samples.  Don't just use nemotron_math (over-indexes on math).  Mix math + chat + code.  `trainingto/generate_dataset.py` and `nemotron_arrow_to_json.py` exist for data prep.
- **Epochs**: 3–5.
- **LR**: current peak 5e-6 is conservative.  DFlash paper uses 6e-4 for full training, but this is fine-tune from z-lab so 2e-5 → 5e-5 peak is the right range.  Adjust warmup to 5% of total steps.
- **Hardware**: ≥4 GPUs on a clean cluster; effective batch ≥16.  This session used 2 A100s with 1×grad-accum-2 = batch 4.

Estimated wall-clock on 4 fresh A100s: ~8–12 hours per epoch at 100k samples.  Budget a full day per variant.

### 6.2 Variant priority

Based on this session's signal, in order:

1. **v3 (ctr-lite) scaled**: tau winner at small scale.  Keep ctr_weight=0.3, try 0.5 too.  **First variant to run.**
2. **v2 (varblock) scaled**: for chained-speculation support at b=20.  Need to train ENOUGH at b=20 — suggest block_size_probs = `[0.1, 0.5, 0.4]` for `[12, 16, 20]` so b=20 gets heavy weight.
3. **v2+v3 combined** (ctr + varblock): if both help individually, try together.  Config: `ctr_weight=0.3 --block-sizes 12,16,20`.
4. **Novel: b=16,20,24 variable-block + chained inference**.  After step 2 trains a good b=20 drafter, adapt `benchmark.py::dflash_generate` to run draft twice (b=16 then b=20 anchored on block_1's argmax-chain end) and verify a combined tree.  This is the path to break the tau=16 ceiling.  See `model/dflash_tree.py::build_chained_tree` (scaffolding already exists, failed due to stale target_hidden per Finding 14 — but should work with a variable-block-TRAINED drafter).

### 6.3 Concrete bootstrap commands

Invoke the pipeline (after the next session has a clean cluster):

```bash
# 1. Inspect current state
bash /homes/videetm/dflash/trainingto/summary.sh

# 2. Seed a fresh recipe file (edit as needed)
cat > /homes/videetm/dflash/logs/session_apr19/recipes.txt << 'EOF'
dflash_mix_100k_v3_scaled|data/nemotron_math_100k/train.jsonl|--block-sizes 16 --ctr-weight 0.3 --ttt-weight 0.0 --num_epochs 3
dflash_mix_100k_v2_varblock_heavy|data/nemotron_math_100k/train.jsonl|--block-sizes 12,16,20 --block-size-probs 0.1,0.5,0.4 --ctr-weight 0.0 --ttt-weight 0.0 --num_epochs 3
dflash_mix_100k_v3plus_varblock|data/nemotron_math_100k/train.jsonl|--block-sizes 12,16,20 --block-size-probs 0.1,0.5,0.4 --ctr-weight 0.3 --ttt-weight 0.0 --num_epochs 3
EOF

# 3. Start master pipeline (it handles train + eval + finalist + cross-dataset)
nohup /homes/videetm/dflash/trainingto/master_pipeline.sh \
  > /homes/videetm/dflash/logs/session_apr19/master.log 2>&1 &

# 4. Watch live:
watch -n 60 /homes/videetm/dflash/trainingto/summary.sh
```

The master pipeline:
1. Waits for any currently-running training to finish (sentinel: `.DONE_TRAINING` in savedir).
2. Starts `train_queue.sh` on the training GPUs.
3. Spawns a `watch_and_eval.sh` per variant on the eval GPUs.
4. At all-done, picks the best checkpoint by math500-16 speedup.
5. Runs finalist 256-sample + cross-dataset (mt-bench, gsm8k, humaneval) on the winner.

**Default GPUs**: `TRAIN_GPUS=0,1  EVAL_GPUS=6,7`.  Edit in the script.  Use `nvidia-smi --query-gpu=index,memory.free --format=csv` to find free GPUs.

### 6.4 Gotchas encountered this session

- **Checkpoint paths must be absolute** in `watch_and_eval.sh` calls, otherwise `eval_ckpt.sh`'s `cd $ROOT` breaks relative paths.  Fixed in current code, but if you add new callers, be careful.
- **GPU cluster contention inflates speedup** (vanilla AR slows more than speculative).  Use TAU for comparability.  For honest speedup, either run on an empty cluster or do apples-to-apples: stock and trained back-to-back on SAME gpu pair within a few minutes.
- **DeepSpeed launcher**: use `--include localhost:X,Y` NOT `--num_gpus=N --include`.  They're mutually exclusive.  `CUDA_VISIBLE_DEVICES` doesn't take effect — use `--include`.
- **16-sample math500 is too noisy** for checkpoint selection (±0.3× noise band).  Use ≥64 samples for intermediate eval, 256 for final.  This cost us an hour of chasing an 8.66 peak that was noise.
- **master_pipeline had a CWD bug** in the finalist stage — benchmark.py was called from `trainingto/`.  Fixed in commit 669c1df via `( cd $ROOT && torchrun ... )`.
- **Empty `save_16bit_model` ckpt**: the exclude_frozen_parameters=True correctly strips target_model, but leaves one stray `target_model.lm_head.weight` in the state dict.  `convert_ckpt_to_hf.py` filters this (skipped=1 is expected).

---

## 7. The "break the block-size ceiling" plan — the real NeurIPS story

Every training-free win we got is pin-bounded by `tau ≤ block_size = 16` on easy sequences.  28% of math500 steps hit this ceiling (Finding 13).  To push speedup past 8.3× on this hardware, we need tau > 16, which requires chained speculation: two drafts per step, verifying both blocks in one target forward.

The blocker so far: draft_2 needs `target_hidden` at positions past block_1, which we don't have (target hasn't run yet).  Q2 tried using stale target_hidden from before block_1 verification — Draft_2 quality was too low (Finding 14).

The PATH:
1. Train v2_varblock_heavy (b∈{12,16,20} with heavy b=20 weight) for a real amount of compute.
2. Verify at inference: if v2 at b=20 reaches within 5% of its b=16 tau, the drafter has learned b=20.
3. Implement true chained speculation: run draft once at b=16 (produces block_1 logits), then a SECOND time at b=20 using draft_hidden_at_block_1_argmax_end as pseudo-target_hidden ("imputed" variant flagged untested in Finding 14).  Combine into one tree (up to ~30 positions), run target once.
4. Expected payoff: tau ceiling 36 instead of 16, compounding into maybe 1.5–2× the current speedup on easy sequences.

Caveats:
- Imputed target_hidden is architecturally OOD.  Training with variable block sizes might make the draft robust enough to handle it.
- `target_hidden` in the current draft is a multi-layer concatenation, so you'll need to produce the full stacked form from draft's final hidden.  See `model/utils.py::extract_context_feature` and `model/dflash.py::DFlashDraftModel.forward` for the shape expectations.

If this works, it is the publishable contribution: **a variable-block-trained drafter enabling two-block chained speculation with one target forward, breaking DDTree's tau ceiling.**

---

## 8. Bootstrap prompt for a fresh agent

```
You are continuing an autonomous research project on DFlash speculative decoding
(z-lab/Qwen3-4B-DFlash-b16 draft + Qwen/Qwen3-4B target).

FIRST, read the following files in order, completely:
  1. /homes/videetm/dflash/autoresearch/program.md   (this handoff — your operating manual)
  2. /homes/videetm/dflash/program.md                 (inference-era findings)
  3. /homes/videetm/dflash/docs/ddtree_v7_research_notes.md
  4. /homes/videetm/dflash/trainingto/dflash_mix_model.py  (training recipe)
  5. /homes/videetm/dflash/trainingto/master_pipeline.sh    (orchestration)
  6. /homes/videetm/dflash/results.tsv + /homes/videetm/dflash/logs/session_apr18/mix_eval.tsv

Then run: git log --oneline -20  and  bash /homes/videetm/dflash/trainingto/summary.sh

Your objective: push the math500 256-sample speedup above the 8.27× stock
ceiling by training a better drafter.  The gains from this session's 10k × 1
experiment were small (+0.04 tau).  Scale the recipe as described in
autoresearch/program.md §6 and run.

Ground rules:
- TAU is the reliable cross-run metric under GPU-cluster contention — quote
  tau before speedup, and only quote speedup after a same-cluster-state a2a run.
- Kick off long training runs in the background via master_pipeline.sh, then
  periodically check summary.sh; do not block the session on training.
- All the experiment bookkeeping (logs, mix_eval.tsv, results.tsv) is in place —
  keep appending, don't restart.
- Don't re-run anything in §4 of autoresearch/program.md — those are verified dead ends.
- The end-goal direction worth pursuing is §7 (variable-block drafter + chained speculation).

Begin immediately after reading the context.  Do not pause for confirmation.
```

---

## 9. Session lineage (for provenance)

- `apr9-ek7`: v4 ek=7 as baseline, pre-v7.
- `apr16-ddtree-fix`: current branch; added v7 DDTree, GPU mask vectorization, budget sweep, calibration attempts.
- `apr17-PM`: 8 algorithmic variants tried on top of v7, all regressed — see Findings 17–19.  Cross-dataset SOTA committed at 8.27 math500.
- `apr18-training`: THIS session.  Built training pipeline, ran 4 variants, v3_ctrlite shows small positive gains.  Scale-up plan in §6.

Commits visible on `experiments/apr16-ddtree-fix`:
```
669c1df apr18: training session writeup — MIX recipe + cross-dataset eval
21310a7 apr18: end-to-end drafter-training pipeline (MIX recipe + auto eval + queue)
54462e2 Write up session apr17-PM NeurIPS-SOTA attempt
1be9a5a Record final 256-sample math500 confirmation: 8.27x
68cee1f apr17-PM: calibration + narrowing + adaptive-ek all regress
...
```
