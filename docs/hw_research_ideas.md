# Hardware-aware research ideas for diffusion-tree attention

Research-level ideas for making DDTree-style speculative decoding faster
through custom CUDA kernels and hardware-aware tree-mask handling.

## Bottleneck context (measured on this stack)

- **Verify is 78-82% of round time** (per phase profile)
- **Per-call kernel-launch + mask-construction overhead is ~30ms** regardless
  of q_len — fixed-cost dominates at small q_len
- The tree mask is `[B, 1, M, prefix+M]` of bf16 — **~80MB** of mask passed
  per layer at B=64 M=16 prefix=600
- Tree attention is causal-tree (each node attends to ancestors only) —
  the dense mask is **mostly -inf**, wasted memory bandwidth
- A6000 is the binding constraint: OOMs at B=128 with 1024-token KV; B=64 is
  the practical ceiling for math500 standard config
- Per-depth accuracy curve has a U-shape on formal-math datasets (math500,
  AIME) — drafter is reliable at depth 1-2 and 12-15, less so in the middle

## Idea 1 — Compressed-tree attention masks via bit-packed parent-list

**Problem:** mask is dense `[B, M, M+prefix]` of bf16. Each attention call
materializes 2 bytes × M × (M+prefix) per head per layer.

**Idea:** for the tree portion, store as `parent_idx[B, M]` (tree topology)
plus per-batch prefix-len. The kernel computes
`mask[q, k] = (k is ancestor of q) OR (k in prefix and k < prefix_len[b])`
inline from the parent walk. Never materialize the dense mask.

**Specific design:**
- At most 4-5 ancestors per node since block_size=16
- Store as `[B, M, 5]` int8 ancestor-list instead of `[B, M, M]` bf16 mask
- **96% mask-memory reduction**
- Kernel reads ancestor list once per query, AND/ORs the visibility into a
  small register-resident bitmask

**Why research-level:** standard SDPA / FlashAttention-2 accepts a dense
`attention_mask` argument. A custom kernel that takes the topology directly
and does visibility check inline is a non-trivial CUDA project but maps to
FlashAttention-3's work-decomposition pattern.

**Expected payoff:** per-call mask construction is ~5% of round time (small
direct gain), but the bigger benefit is **fitting more B in memory** since
mask materialization is non-trivial at high B.

---

## Idea 2 — Block-sparse FlashAttention restricted to ancestor sets

**Problem:** target attends each query to all `M+prefix` keys. Most of the
M-portion is masked out. Wasted FLOPs on masked entries.

**Idea:** block-sparse FlashAttention with tree-structured sparsity. Each
query block reads only its ancestor key-blocks.

**Specific design:**
- Tile the keys into blocks of 4 nodes
- For each query node q at depth d, its ancestor set spans depths 0..d-1 —
  at most ⌈d/4⌉ key tiles
- With block_size=16, depth-15 query touches ~4 tiles
- Average over the tree: ~2-3 tiles per query vs 4 tiles full → **30-50%
  FLOP reduction** on the M-portion of attention

**Why research-level:** standard block-sparse attention assumes 2D
sliding-window or random patterns. Tree-structured sparsity has a different
pattern (depth-dependent, ancestor-only). Composing this with prefix
attention (full attend to prefix) requires a hybrid mask: full attention for
prefix-keys, block-sparse for tree-keys, in the same kernel.

**Expected payoff:** verify FFN dominates over attention at small q_len, but
at high B (verify q_len × B is large) attention scales quadratically. Could
save 10-20% verify time at high B.

---

## Idea 3 — Persistent-kernel verify that fuses tree-mask + attention + FFN

**Problem:** verify is one transformer forward pass. Each layer launches
separate K/Q/V projection, attention, output projection, FFN-up, FFN-gate,
FFN-down kernels (~10 kernels per layer × 32 layers = 320 kernel launches
per forward).

**Idea:** persistent CUDA kernel that loads layer weights once into shared
memory and processes all `M` tree-positions through all 32 layers in a
single kernel invocation. Tree-mask is a register-resident bitfield
re-checked each layer. Eliminates ~95% of kernel-launch overhead.

**Specific design:**
- SM-resident kernel pinned to specific tree subtrees
- Each SM owns ~M/N_SMs queries, processes them through entire model
- Layer-boundary activations communicated via L2 cache
- Hopper-style "persistent kernel" pattern extended to tree topology

**Why research-level:** persistent kernels exist for inference (vLLM prefill,
FA-3) but not for tree-structured masks. Tree topology must be encoded in
shared memory for fast lookup.

**Expected payoff:** per-call overhead measured at ~30ms. Persistent kernel
could reduce to ~3-5ms (kernel-launch only once instead of 320 times). At
our q_len, this is the **single biggest possible verify speedup** — back-of-
envelope **4-6× verify time reduction** at small q_len.

---

## Idea 4 — Hardware-aware tree shape selection

**Problem:** tree shape is decided algorithmically (heap, fixed-width)
without considering kernel efficiency. A tree with M=16, depths [1..15] has
irregular sequence lengths per query that's hard for tensor cores.

**Idea:** constrain tree shape so each query has the **same number of
ancestor key-blocks** (fixed sparsity). Choose tree shapes where each depth
has `2^d` width up to the kernel's tile boundary, then truncate. Trades some
tau for kernel efficiency.

**Specific design:**
- Heap respects "tile-aligned" budget: per-depth widths multiples of 4 (FA
  tile size)
- Each query attends to exactly ⌈max_depth/4⌉ key tiles
- Kernel runs at full tensor-core efficiency, no bubble cycles from
  variable sparsity

**Why research-level:** algorithm-kernel co-design. Sequoia's DP for tree
shape didn't model kernel-tile-alignment costs; doing so changes the optimum.

**Expected payoff:** modest, maybe 5-10% verify speedup. Composes with
Idea 2 above.

---

## Idea 5 — Streaming-paged tree-aware KV cache (★ recommended priority)

**Problem:** target KV cache is monolithic `[B, H, prefix_len, D]`. At high
B, the prefix is the same/similar across batch elements (math500 system
prompts identical). Wasted memory. Currently A6000 OOMs at B=128 with
1024-token KV.

**Idea:** **paged + prefix-shared KV cache for tree spec**. Prefix is one
shared page across batch (when prompts share a prefix); per-batch unique
tail in private pages. Tree nodes append to private pages and get trimmed
on accept. Same paged-attention idea as vLLM but extended with **per-tree
cache forking** for hypothetical extensions.

**Specific design:**
- Shared system-prompt page (read-only, ref-counted)
- Per-sequence private page chain
- **Tree pages**: each tree node's K/V written into a "speculative" page
  that gets either committed (accept path) or discarded (reject path)
- Accept-path commit is a pointer-swap; no memory copy

**Why research-level:** vLLM's paged-attention doesn't handle speculative
branches that get discarded. Adding "tree pages" with O(1) commit/rollback
is non-trivial — must coordinate with the attention kernel's KV indexing.

**Expected payoff:** **the binding constraint at B≥128 is KV cache memory**.
Prefix sharing across batch could halve memory at high B for math500. That
single change might enable B=256 or B=512 instead of OOMing at B=128 — a
much bigger lift than any attention-kernel optimization. Absolute throughput
could 3-4× simply from larger feasible B.

**This is the single highest-priority idea on this list.**

---

## Idea 6 — Verify-while-drafting via NVLink/streams

**Problem:** drafter and verifier serialized — one runs, then the other.
Per the profile, drafter is 12% of round time; serializing wastes that.

**Idea:** **run target on GPU 0, draft on GPU 1**, communicate via NVLink.
Target verifies round R while draft generates round R+1's tree on a guess
of round R's bonus. ~99% of draft cost is hidden under verify.

**Specific design:**
- Small bonus-token prediction model (could just be "use rank-0 of last
  block's last position") provides speculative anchor
- Draft-on-GPU-1 produces tree for round R+1 conditional on this guess
- When round R commits, actual bonus is compared to guess
  - Same: R+1's tree reused (paid: zero)
  - Different: discard and redo (paid: extra draft, but on separate GPU
    so wall-clock free)

**Why research-level on A6000:** A6000 doesn't have NVLink between cards
typically (PCIe-only). Needs careful KV-state sync over PCIe, higher
latency. Could be done with prefetched activations.

**Expected payoff:** if bonus-prediction right ~70% of the time (rank-0 of
last masked position), draft cost is hidden 70% of rounds. Effectively
turns the 12% drafter cost into 4% — saves ~8% of round time. Bigger
upside on multi-GPU systems with NVLink.

---

## Idea 7 — FP8/INT4 KV-cache quantization with tree-aware re-quantization

**Problem:** FP8 KV failed on A6000 because A6000 lacks FP8 tensor cores.
INT4 KV (KIVI-style) requires custom kernels.

**Idea (specific to tree spec):** quantize **only the prefix** KV (reused
many times) at FP8/INT4, leave the tree-node KV in bf16 (used briefly,
discarded on accept/reject). The "tree-page" approach from Idea 5 makes
this clean — different memory regions, different quant levels.

**Specific design:**
- Prefix: FP8 tensor-quantized (per-head scale)
- Tree pages: bf16
- Attention kernel handles dual-precision: dequantizes prefix on-the-fly
  from FP8 to bf16 for the attention math
- Per-call dequant cost amortizes across the M queries attending to that
  prefix that round

**Why research-level:** existing FP8 KV assumes uniform precision.
Mixed-precision attention (FP8 prefix + bf16 tree) is not standard. The
dequant cost must amortize across M queries per round.

**Expected payoff on A6000:** native FP8 isn't supported, so dequant runs
in bf16 emulation — slow. **INT4 KV with bit-packed dequant** could work:
~50% prefix memory reduction → higher B before OOM → addresses binding
constraint. Composes naturally with Idea 5.

---

## Priority ranking

If you can pursue only one:

1. **Idea 5 — paged + prefix-shared KV** is the highest-priority because
   it directly attacks the binding constraint (memory limits B at ~128).
   Every other idea optimizes time per call but doesn't unlock larger
   batch. Pushing B from 128 to 512 could 3-4× absolute throughput at
   the same per-token speed.

2. **Idea 3 — persistent kernel for verify** is the most contained CUDA
   project (1-2 month research scope) with clear payoff (4-6× verify
   speedup at our q_len) on any GPU including A6000.

3. **Idea 7 — INT4 prefix KV** as a short-term complement to Idea 5:
   stacks for ~4× B headroom together.

The remaining ideas (1, 2, 4, 6) are smaller wins individually but all
compose with the top 3.

## Composition

Best stack:
- **Idea 5** unlocks 2-4× larger feasible batch
- **Idea 3** speeds up verify per-call by 4-6×
- **Idea 1+2** save attention-mask memory and FLOPs at the new larger B
- **Idea 7** stacks on Idea 5 for further memory headroom
- **Idea 6** (multi-GPU) hides remaining drafter cost on multi-card systems

Net potential improvement vs current baseline: **8-15× absolute throughput**
on a memory-rich GPU (H100/H200). On A6000, primarily a memory unlock from
Idea 5+7 → larger B → ~3-4× absolute throughput at same per-token speed.

## Open research questions per idea

- Idea 1: how does inline mask computation compare to a precomputed
  bit-packed mask buffer at the kernel level?
- Idea 2: what's the optimal block size for tree-sparsity given that
  ancestor-set size varies with depth?
- Idea 3: can the persistent kernel handle variable q_len per round
  (when accepts vary), or must M be fixed?
- Idea 5: how to handle the "tree-page rollback on partial accept"
  efficiently — naive gather costs ~M × hidden bytes per layer
- Idea 6: what's the right speculative-bonus-prediction model? Is rank-0 of
  last masked position sufficient, or do we need a tiny learned head?
- Idea 7: at what quant level (INT4 vs INT8) does the dequant cost equal
  the memory-bandwidth saving?

These are all 1-3 month research projects individually; the full stack is
genuinely a 6-12 month effort but with cleanly separable milestones.
