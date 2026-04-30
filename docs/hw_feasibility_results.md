# Hardware ideas — feasibility loop results

Autonomous loop running 7 feasibility studies, one per HW idea from
`docs/hw_research_ideas.md`. Each measures a proxy or upper bound that
informs whether the full custom-kernel project is worth months of
engineering.

## Summary table

| # | Idea | Result | Effort | Shippable now? |
|---|------|--------|--------|----------------|
| 1 | Compressed-tree masks (parent-list kernel) | **NEGATIVE** — mask construction is 0.05ms (<1% of round) | months CUDA | NO — solves a non-problem |
| 2 | SDPA backend variants | **NEGATIVE** — default == efficient; flash needs no mask | trivial | NO — no faster backend exists |
| 3 | Persistent kernel for verify | **STRONG POSITIVE — 1.81× from torch.compile alone** | hours (compile) or months (custom) | **YES** — ship torch.compile today |
| 4 | Tile-aligned tree shapes | small positive (+2-4%, single-seed) | days | maybe — needs multi-seed validation |
| 5 | Prefix-shared KV (paged) | **NEGATIVE on math benchmarks** (LCP=0); positive for production chat | months | depends on workload |
| 6 | Multi-GPU stream overlap | NEGATIVE single-GPU (0.8% savings); needs NVLink | weeks (multi-GPU) | NO on single-card A6000 |
| 7 | INT4 KV quantization | **HUGE ceiling — 4× larger feasible B** (74% memory saving) | months (custom kernel) | NO — needs CUDA work |

## Detail per idea

### Idea 1 — Compressed-tree masks: NEGATIVE

`logs/hw_mask_cost.json`. Across all measured shapes:
- mask construction time: **0.05ms** regardless of B/M
- mask memory: 0.08-1.98 MB
- as fraction of full round time: <0.5%

The mask "construction overhead" isn't real — PyTorch's `torch.zeros` +
`masked_fill_` is microseconds. The compressed-mask kernel project would
optimize a non-bottleneck.

### Idea 2 — SDPA backend variants: NEGATIVE

`logs/hw_sdpa_backends.json`. Across all measured shapes:
- `default` ≡ `efficient` (xformers-style mem-efficient)
- `math` is 6-10× slower
- **`flash` cannot accept custom attn_mask** — fundamental Flash Attention limitation
- cuDNN attention: runtime-disabled

There's no free 2× speedup hiding in a backend switch. The current default
already uses the best available.

### Idea 3 — Persistent kernel for verify: STRONG POSITIVE

`logs/hw_compile.json`. At B=4 M=32 prefix=200:
- eager: 34.4ms / verify call
- torch.compile (`mode="reduce-overhead"`): **19.0ms / verify call**
- **1.81× speedup** (45% per-call time reduction)

**This is the single biggest win in the entire research loop.** A custom
persistent kernel could plausibly go further (~3-5ms/call, ~7×). But
torch.compile alone — zero new code beyond `torch.compile(target.model,
mode="reduce-overhead")` — gives almost half the benefit immediately.

Caveats:
- 40-second one-time compile warmup
- May break on dynamic shapes; need to test in full pipeline

### Idea 4 — Tile-aligned tree: small positive

`logs/hw_tile_aligned.json`. Per-depth K=4 instead of K=8:
- B=1: 7.69× → 7.98× (+3.8%)
- B=4: 7.46× → 7.62× (+2.1%)

Mechanism plausible (less Python heap overhead + better attention tile
alignment) but **single-seed numbers**. Multi-seed lessons say ±5% noise is
typical. Probably real but small.

### Idea 5 — Prefix-shared KV: NEGATIVE on math benchmarks

`logs/hw_prefix_sharing.json`. Longest common prefix across math500 prompts:
- math500: **LCP = 0 tokens**
- aime24: LCP = 0
- gsm8k: LCP = 0

Math benchmarks don't share a prefix because each problem statement is
unique content followed by a fixed suffix template. **The prefix-sharing
memory unlock as designed in Idea 5 doesn't apply to these workloads.**

It WOULD apply to production chat serving with shared system prompts (~30
tokens for Qwen3 chat header), reaching meaningful savings only for
workloads with long shared system prompts / tool descriptions / RAG context.

### Idea 6 — Multi-GPU stream overlap: NEGATIVE single-GPU

`logs/hw_streams.json`. At B=4 M=32 verify+draft:
- sequential: 66.7 ms/iter
- parallel via streams: 66.2 ms/iter
- **overlap savings: 0.8%**

The GPU is fully saturated by either pass alone. Single-GPU stream overlap
provides essentially nothing. Idea 6 truly requires multi-GPU NVLink to
provide value. Dead on A6000 single-card.

### Idea 7 — INT4 KV quantization: HUGE ceiling

`logs/hw_kv_quant.json`. INT4 prefix saves 74% of total KV memory:

| config | bf16 total | INT4 saving | INT4 % |
|---|---|---|---|
| B=4 M=32 prefix=600 | 1264 MB | 900 MB | 71% |
| B=64 M=16 prefix=1200 | 38912 MB | 28800 MB | 74% |
| B=128 M=8 prefix=1500 | 96512 MB | 72000 MB | 75% |

**At B=128 prefix=1500: bf16 baseline 96GB (impossible on A6000); INT4
drops to 24GB (fits cleanly).** Could unlock 4× larger feasible batch.

But A6000 lacks INT4 tensor cores. Requires KIVI-style custom kernels
(1-3 month engineering project).

## Compositional summary

```
torch.compile (Idea 3, ~1 day)
  +1.81× per-call verify speedup
  = SHIP TODAY

INT4 KV (Idea 7, 1-3 months CUDA)
  +4× larger feasible batch
  = HIGH-VALUE CAPEX

Tile-aligned tree (Idea 4, days)
  +2-4% (single-seed, needs validation)

Everything else: negative or non-applicable on A6000 / math benchmarks
```

## Recommended path forward

1. **Immediate (1 day):** add `torch.compile(target.model, mode="reduce-overhead")`
   to the inference pipeline. Verify it works with the full DDTree loop
   (handle compile-cache warmup, dynamic shape issues). Ship the +45% per-call
   speedup.

2. **Short-term (weeks):** validate Idea 4 (tile-aligned tree) with multi-seed
   evaluation. If it survives, ship.

3. **Medium-term (1-3 months):** INT4 KV cache as the main capex. Lifts
   memory-imposed B ceiling 4×, enabling B=128/256 cleanly on A6000.

4. **Pass on:** Ideas 1, 2, 5 (math), 6 (single-GPU). All gave negative
   feasibility results.

## What this means for the broader research arc

The cumulative finding from the entire autonomous loop (~50 sweeps,
diagnostics, multi-seed corrections, survivorship test, HW feasibility):

- **Algorithm side has been thoroughly explored.** Static γ=0.85 helps AIME
  reliably (+3.6%). All adaptive variants tested don't beat static. Math500
  gain was sample bias. U-shape was survivorship bias. The drafter is
  monotonically less reliable at deep depths and no clever heap tricks
  recover much beyond what geometric decay already does.

- **Hardware side has clear next steps.** torch.compile is shippable today
  for ~1.8× verify speedup. INT4 KV unlocks 4× batch headroom but requires
  real CUDA work. These are bigger leverage than any algorithm change.

- **Ideas with positive feasibility (3, 4, 7) are all about reducing
  per-call cost or memory pressure** — they don't increase τ. Algorithm
  side appears to have hit a hard ceiling on this stack.
