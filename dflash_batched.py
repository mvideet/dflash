"""Batched DFlash v7 generator + batched vanilla AR baseline.

End-to-end greedy decoding with **different prompts** in the batch:
  - Per-element divergent trees built from per-element draft logits
  - Per-element ragged accept-path selection (`pad to max(n+1)`)
  - Per-element KV-cache trim that keeps the cache rectangular while preserving
    each element's real prefix at the front

Prompt handling
---------------
We require all prompts in the batch to have the SAME tokenized length. The
caller (`benchmark_batched.py`) filters/truncates the dataset so this holds.
This eliminates prefill padding, which simplifies cross-context (target_hidden)
masking. Step 2+ still has divergent cur_prefix_lens (because n_b varies), and
that is handled with per-element attention masks for both draft and target.

Draft KV cache: not reused. Each step does a fresh draft forward over
[B, max(cur_prefix_lens) + block_size]. The draft input contains the full
per-element accumulated tokens (prompt + decoded so far + anchor + masks).
Pad columns in the draft input are masked out via a 4D attention_mask passed
to the draft's SDPA self-attention.

What is NOT modeled (intentional, for benchmark scope):
  - Draft KV-cache reuse across steps. Real implementations would reuse,
    making v7 faster than this benchmark shows. So our reported speedups
    are **lower bounds** on optimised v7.
  - v7 score adjustments (CGDB, PDRR, calibration, APB, narrow-after-dev).
    Pure DDTree only.
"""

import math
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import DynamicCache


# ---------------------------------------------------------------------------
# Online M* prediction (goodput-driven)
# ---------------------------------------------------------------------------
# Cost model: T_step(B, M) ≈ a(B) * M + b(B), fit from the offline mts grid sweep
# (logs/mts_sweep_summary.json). Slopes/intercepts are in milliseconds per step.
# Linear extrapolation past B=8.
_T_STEP_COEFFS: dict = {
    # B: (slope_ms_per_unit_M, intercept_ms). Fit from logs/mts_sweep_summary.json.
    1:  (0.039,  47.96),
    2:  (0.487,  42.99),
    4:  (1.121,  51.66),
    8:  (1.683, 108.49),
    16: (3.153, 238.89),
    32: (6.30,  478.0),   # extrapolated 2x from B=16 (linear-in-B for compute-bound regime)
}

# Baseline tau(M) curve from the offline v7 mts sweep (math500, Qwen3-4B).
# Used to scale the online-observed ewma_tau into a tau prediction at any M,
# preserving the actual saturating shape of the curve. Extrapolation below M=16
# extends the sharp drop empirically observed at small budgets.
_TAU_BASELINE: dict = {
    4:   5.0,      # extrapolated
    8:   7.0,      # extrapolated
    16:  8.58,
    24:  8.95,     # interpolated
    32:  9.23,
    48:  9.50,     # interpolated
    64:  9.70,
    96:  9.92,     # interpolated
    128: 10.08,
    256: 10.37,
}


def _baseline_tau(M: int) -> float:
    """Return offline-calibrated tau for budget M (log-linear interpolation)."""
    if M in _TAU_BASELINE:
        return _TAU_BASELINE[M]
    Ms = sorted(_TAU_BASELINE)
    if M <= Ms[0]:
        return _TAU_BASELINE[Ms[0]]
    if M >= Ms[-1]:
        return _TAU_BASELINE[Ms[-1]]
    # Linear interp in log-M space.
    for i in range(len(Ms) - 1):
        if Ms[i] <= M <= Ms[i+1]:
            t_lo, t_hi = _TAU_BASELINE[Ms[i]], _TAU_BASELINE[Ms[i+1]]
            f = (math.log2(M) - math.log2(Ms[i])) / (math.log2(Ms[i+1]) - math.log2(Ms[i]))
            return t_lo + f * (t_hi - t_lo)
    return _TAU_BASELINE[Ms[-1]]


def _predict_t_step(B: int, M: int) -> float:
    """Linear cost model in M, with B-specific coefficients. Falls back to nearest B."""
    if B not in _T_STEP_COEFFS:
        Bs = sorted(_T_STEP_COEFFS.keys())
        B_use = min(Bs, key=lambda x: abs(x - B))
    else:
        B_use = B
    a, b = _T_STEP_COEFFS[B_use]
    return a * M + b


def _predict_tau(M: int, ewma_tau: float, M_ref: int = 16,
                 anchor_conf: float = -1.0, block_size: int = 16) -> float:
    """Predicts τ(M) for the upcoming step.

    Two signals composed:
      1) Workload-calibrated curve: scale the offline _TAU_BASELINE by
         ewma_tau / baseline(M_ref). Captures the saturating tau-vs-M shape
         and the slow-moving per-workload difficulty.
      2) Forward chain bound (anchor confidence): if the target's top-1 prob
         at the just-decoded bonus token (= next step's anchor) is p, the
         expected argmax-chain length follows a geometric: E[chain] ≈ p/(1-p)
         capped at block_size. This bounds τ from above on EASY steps where
         the chain dominates; on HARD steps (low p) the tree branches still
         contribute, so the bound is loose and the curve takes over.
      anchor_conf < 0 disables the chain bound (used for step 1 / fallback).
    """
    base_at_ref = _baseline_tau(M_ref)
    if base_at_ref <= 0:
        return ewma_tau
    ratio = ewma_tau / base_at_ref
    curve_tau = _baseline_tau(M) * ratio

    if anchor_conf < 0.0:
        return curve_tau

    # Geometric chain expectation. At p=1 we cap at block_size (full chain).
    if anchor_conf >= 0.999:
        chain_bound = float(block_size)
    else:
        chain_bound = min(float(block_size),
                          anchor_conf / max(1.0 - anchor_conf, 1e-3))
    # The chain is only the rank-0 contribution to tau. v7 trees also accept
    # rank-1+ tokens via target verification — so scale the chain bound up
    # by an empirical "tree advantage" (1.5 from path-trace stats: roughly
    # 35% of accepted positions are non-argmax in steady state).
    eff_chain_bound = chain_bound * 1.5
    return min(curve_tau, eff_chain_bound)


def _pick_m_online(B: int, ewma_tau: float, ewma_M_ref: int,
                   candidates: Tuple[int, ...] = (8, 16, 32, 64, 128),
                   anchor_conf: float = -1.0, block_size: int = 16) -> int:
    """Choose M* that maximizes predicted goodput τ(M) / T_step(B, M).
    Optional anchor_conf is the target's top-1 prob at the bonus token from
    the previous step — provides a forward-looking chain-length bound."""
    best_m, best_goodput = candidates[0], -1.0
    for m in candidates:
        tau_p = _predict_tau(
            m, ewma_tau, M_ref=ewma_M_ref,
            anchor_conf=anchor_conf, block_size=block_size,
        )
        t_step = _predict_t_step(B, m)
        g = tau_p / max(t_step, 1e-3)
        if g > best_goodput:
            best_goodput = g
            best_m = m
    return best_m

from model import (
    DFlashDraftModel,
    extract_context_feature,
)
from model.dflash_tree_batched import (
    build_node_budget_tree_batched,
    create_tree_attention_mask_batched,
    select_best_dynamic_leaf_batched,
)
from model.vectorized_tree import (
    vectorized_tree_build,
    default_width_vec,
)


# ---------------------------------------------------------------------------
# KV-cache trim (for the target). Cache layout: [real_prefix | tail_pad].
# ---------------------------------------------------------------------------

def _trim_target_kv_cache(
    past_kv: DynamicCache,
    cur_prefix_lens: torch.Tensor,         # [B] real cache prefix BEFORE this step
    accepted_tree_paths: torch.Tensor,     # [B, max_n1] tree-node indices to keep
    n1_per_elem: torch.Tensor,             # [B] count of tree nodes to keep
    old_cache_size: int,                   # cache size before this step's verify
    device: torch.device,
) -> torch.Tensor:
    """Cache layout after trim per row b:
      [real_prefix_0..pl_b-1 | accepted_0..n1_b-1 | tail-pad-to-new_cache_size]
    where new_cache_size = max(pl_b + n1_b).
    Tail pad slots hold a duplicate of the last accepted index (their content
    will never be attended to in subsequent steps because attention masks gate
    them via cur_prefix_lens).
    """
    B = cur_prefix_lens.shape[0]
    new_real = cur_prefix_lens + n1_per_elem
    new_size = int(new_real.max().item())

    keep = torch.zeros(B, new_size, dtype=torch.long, device=device)
    for b in range(B):
        pl = int(cur_prefix_lens[b].item())
        n1 = int(n1_per_elem[b].item())
        if pl > 0:
            keep[b, :pl] = torch.arange(pl, device=device)
        if n1 > 0:
            keep[b, pl:pl + n1] = old_cache_size + accepted_tree_paths[b, :n1]
        nr = pl + n1
        if nr < new_size:
            tail_idx = (old_cache_size + accepted_tree_paths[b, n1 - 1]) if n1 > 0 else 0
            keep[b, nr:] = tail_idx

    H = past_kv.layers[0].keys.shape[1]
    D = past_kv.layers[0].keys.shape[3]
    idx = keep.view(B, 1, new_size, 1).expand(-1, H, -1, D)
    for layer in past_kv.layers:
        layer.keys = layer.keys.gather(2, idx).contiguous()
        layer.values = layer.values.gather(2, idx).contiguous()
    return new_real


def _build_decode_attention_mask(
    cur_prefix_lens: torch.Tensor,    # [B] real prefix per elem
    cache_size: int,                  # uniform cache columns
    q_len: int,                       # queries per elem
    device: torch.device,
    dtype=torch.bfloat16,
) -> torch.Tensor:
    """Decode-step mask: cols [cur_prefix_lens[b], cache_size) are tail-pad
    and must be masked out. Returns [B, 1, q_len, cache_size + q_len].
    """
    B = cur_prefix_lens.shape[0]
    min_val = torch.finfo(dtype).min
    mask = torch.zeros(B, 1, q_len, cache_size + q_len, device=device, dtype=dtype)
    col_idx = torch.arange(cache_size, device=device).unsqueeze(0)
    pad_cols = col_idx >= cur_prefix_lens.unsqueeze(1)                      # [B, cache_size]
    pad_4d = pad_cols.view(B, 1, 1, cache_size).expand(-1, 1, q_len, -1)
    mask[:, :, :, :cache_size] = torch.where(
        pad_4d, torch.full_like(mask[:, :, :, :cache_size], min_val),
        mask[:, :, :, :cache_size],
    )
    return mask


# ---------------------------------------------------------------------------
# Vanilla AR baseline
# ---------------------------------------------------------------------------

def _trim_pad_after_prefill(past_kv: DynamicCache, prefix_lens: torch.Tensor, S_max: int, device) -> int:
    """Trim padded prefill cache to max(prefix_lens). Each row keeps arange(pl_b)
    contiguously at the FRONT; tail-pad slots get filled with index 0 (anchor),
    will be masked out by attention mask in subsequent steps."""
    B = prefix_lens.shape[0]
    new_size = int(prefix_lens.max().item())
    keep = torch.zeros(B, new_size, dtype=torch.long, device=device)
    for b in range(B):
        pl_b = int(prefix_lens[b].item())
        if pl_b > 0:
            keep[b, :pl_b] = torch.arange(pl_b, device=device)
        if pl_b < new_size:
            keep[b, pl_b:] = 0
    H = past_kv.layers[0].keys.shape[1]
    D = past_kv.layers[0].keys.shape[3]
    idx = keep.view(B, 1, new_size, 1).expand(-1, H, -1, D)
    for layer in past_kv.layers:
        layer.keys = layer.keys.gather(2, idx).contiguous()
        layer.values = layer.values.gather(2, idx).contiguous()
    return new_size


@torch.inference_mode()
def vanilla_ar_generate_batched(
    target,
    input_ids: torch.Tensor,                # [B, S_max] right-padded prompts
    attention_mask: torch.Tensor,           # [B, S_max] 1=real, 0=pad
    eos_token_ids: List[int],
    max_new_tokens: int,
):
    """Greedy batched AR baseline with right-padding."""
    import time
    device = input_ids.device
    B, S_max = input_ids.shape
    eos_set = set(int(e) for e in eos_token_ids)
    prefix_lens = attention_mask.sum(dim=-1).to(torch.long)                 # [B]
    pos_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0).to(torch.long)

    past_kv = DynamicCache()
    out = target(
        input_ids, attention_mask=attention_mask, position_ids=pos_ids,
        past_key_values=past_kv, use_cache=True, logits_to_keep=S_max,
    )
    last_logits = out.logits[torch.arange(B, device=device), prefix_lens - 1, :]
    next_tok = last_logits.argmax(dim=-1, keepdim=True)                     # [B, 1]
    decoded = next_tok.clone()                                              # [B, 1]
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    for b in range(B):
        if int(next_tok[b, 0].item()) in eos_set:
            finished[b] = True

    # Trim pad cols (so cache is rectangular at max(pl_b)).
    cache_size = _trim_pad_after_prefill(past_kv, prefix_lens, S_max, device)
    cur_prefix_lens = prefix_lens.clone()
    cur_pos = cur_prefix_lens.unsqueeze(1).clone()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    while not bool(finished.all()) and decoded.shape[1] < max_new_tokens:
        attn_step = _build_decode_attention_mask(
            cur_prefix_lens=cur_prefix_lens, cache_size=cache_size,
            q_len=1, device=device,
        )
        out = target(
            next_tok, attention_mask=attn_step, position_ids=cur_pos,
            past_key_values=past_kv, use_cache=True, logits_to_keep=1,
        )
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cur_prefix_lens = cur_prefix_lens + 1
        cur_pos = cur_pos + 1
        cache_size += 1
        decoded = torch.cat([decoded, next_tok], dim=1)
        for b in range(B):
            if not finished[b] and int(next_tok[b, 0].item()) in eos_set:
                finished[b] = True

    torch.cuda.synchronize()
    total = time.perf_counter() - t0

    output_ids_list = [
        torch.cat([input_ids[b:b+1, :int(prefix_lens[b].item())], decoded[b:b+1, :]], dim=1)
        for b in range(B)
    ]
    # Per-elem real output length: count tokens up to and including first EOS.
    num_out = []
    for b in range(B):
        toks = decoded[b].tolist()
        n = len(toks)
        for i, t in enumerate(toks):
            if int(t) in eos_set:
                n = i + 1
                break
        num_out.append(n)

    return SimpleNamespace(
        output_ids_list=output_ids_list,
        num_output_tokens=num_out,
        total_decode_time=total,
    )


# ---------------------------------------------------------------------------
# v7 batched generator
# ---------------------------------------------------------------------------

@torch.inference_mode()
def dflash_generate_batched(
    draft: DFlashDraftModel,
    target,
    input_ids: torch.Tensor,                # [B, S_max] right-padded prompts
    attention_mask: torch.Tensor,           # [B, S_max] 1=real, 0=pad
    mask_token_id: int,
    eos_token_ids: List[int],
    max_new_tokens: int,
    block_size: int,
    max_tree_size: int,
    expand_k: int,
    temperature: float = 0.0,
    online_mts: bool = False,
    online_mts_candidates: Tuple[int, ...] = (8, 16, 32, 64, 128),
    online_mts_alpha: float = 0.7,
    # EWMA-based adaptation (ported from benchmark.py --adaptive-block).
    # Tracks per-step acceptance rate via EWMA, linearly maps to M and expand_k.
    ewma_adaptive: bool = False,
    ewma_decay: float = 0.8,
    ewma_min_M: int = 12,
    ewma_min_ek: int = 2,
    ewma_max_ek: int = 8,
    # Anchor-entropy / draft-conf adaptation (--adaptive-budget-mode).
    # Forward signal: at each step, compute draft entropy at anchor (depth 0).
    # u = H(anchor)/log(V), then eff_M = min_M + (max_M - min_M) * u^gamma.
    anchor_signal: str = "none",            # "none" | "entropy" | "draft-conf"
    anchor_min_M: int = 32,
    anchor_gamma: float = 1.0,
    pwls: bool = False,
    bqat_threshold: float = 0.0,
    cppr: bool = False,
    cppr_lambda: float = 0.5,
    pdrr_k1: float = 0.0,
    pdrr_k2: float = 0.0,
    pdrr_k3: float = 0.0,
    cgdb_shallow_depth: int = 0,
    cgdb_high_thresh: float = 0.0,
    cgdb_low_thresh: float = 0.0,
    cgdb_mid_k: int = 0,
    bjc_calib=None,                 # JointCalib instance (kept across steps); None=disabled
    use_target_q1: bool = False,    # Idea 1e: replace draft's q_1 with target's logits at bonus
    adaedl_B: bool = False,         # Idea 1b: AdaEDL closed-form adaptive M from anchor entropy
    adaedl_min_M: int = 8,
    adaedl_max_M: int = 128,
    entropy_width: bool = False,    # Idea 1c: per-position expand_k from entropy
    entropy_width_min_ek: int = 1,
    entropy_width_max_ek: int = 8,
    use_vectorized_tree: bool = False,           # vectorized fixed-width tree (kills Python heap)
    vectorized_width_vec: Optional[List[int]] = None,
    heap_conc_B: bool = False,                    # Item 2: heap concentration entropy → M
    heap_conc_min_M: int = 8,
    heap_conc_max_M: int = 128,
    optree_termination: bool = False,             # Item 3: OPT-Tree variable-depth termination
    optree_threshold: float = 0.05,
    extend_tree_q1: bool = False,                # Item 4: extra depth in tree to harvest q_1
    eagle2_overbuild: bool = False,              # Item 9: over-build pool, rerank by joint score
    eagle2_pool_mult: int = 2,
    specdecpp_threshold: float = 0.0,            # Item 10: SpecDec++ rejection-prob threshold
    dsde_kl_thresh: float = 0.0,                 # Item 11: DSDE KL stability — stop if drift
    tapout_bandit: bool = False,                 # Item 12: bandit over stopping rules
    online_sequoia: bool = False,                # Item 14: online Sequoia DP from cross-batch p_{k,d}
    sequoia_recompute_every: int = 5,
    roofline_pid: bool = False,                  # Item 15: PID-controlled M from goodput
):
    """Batched v7 (DDTree) generator. Greedy only.
    Variable-length prompts via right-padding + per-element prefix_lens tracking.
    """
    import time
    device = input_ids.device
    B, S_max = input_ids.shape
    eos_set = set(int(e) for e in eos_token_ids)
    assert temperature == 0.0, "Batched generator: greedy only."

    prefix_lens = attention_mask.sum(dim=-1).to(torch.long)                     # [B]
    pos_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0).to(torch.long)

    # --- Prefill target ---
    past_kv_target = DynamicCache()
    out = target(
        input_ids, attention_mask=attention_mask, position_ids=pos_ids,
        past_key_values=past_kv_target, use_cache=True,
        logits_to_keep=S_max, output_hidden_states=True,
    )
    target_hidden_full = extract_context_feature(out.hidden_states, draft.target_layer_ids)  # [B, S_max, D']
    last_logits = out.logits[torch.arange(B, device=device), prefix_lens - 1, :]
    first_tok = last_logits.argmax(dim=-1, keepdim=True)                        # [B, 1]
    del out

    # Per-element list of accepted-so-far tokens (each [1, n_b_real]).
    decoded_list = [first_tok[b:b+1, :].clone() for b in range(B)]
    finished = [int(first_tok[b, 0].item()) in eos_set for b in range(B)]
    accepted_lengths_per_elem: List[List[int]] = [[] for _ in range(B)]
    tree_node_counts: List[int] = []
    per_step_M_choices: List[int] = []                                           # for analysis

    # Online M* state. Initialised to the static `max_tree_size` for step 1.
    ewma_tau = float(max_tree_size) / 16.0 * 8.0  # crude initial guess: scales w/ M
    ewma_tau = max(ewma_tau, 6.0)                  # floor
    ewma_M_ref = max_tree_size                     # M used to obtain ewma_tau
    online_mts_candidates_t = tuple(int(m) for m in online_mts_candidates)
    prev_anchor_conf: float = -1.0                 # forward-looking signal; -1 = no data yet
    per_step_anchor_conf: List[float] = []         # for analysis

    # EWMA state for repo-style adaptive sizing.
    ewma_rate: float = 1.0       # init at 1.0 → first step uses max budget (matches benchmark.py)
    eff_ek: int = expand_k       # adapted per step under ewma_adaptive

    # Item 7 — DtACI: multi-rate aggregation. Maintain 3 EWMAs at different
    # decays; ensemble the prediction. Helps with concept drift.
    ewma_rates_multi = [1.0, 1.0, 1.0]
    ewma_alphas_multi = [0.5, 0.8, 0.95]   # fast / medium / slow
    ewma_weights_multi = [1.0/3, 1.0/3, 1.0/3]   # equal weights initially

    # Item 11 — DSDE: track draft–target KL across steps for drift detection.
    prev_draft_target_kl: float = 0.0

    # Item 12 — TapOut: bandit over stopping rules (basic implementation).
    tapout_arms = ["fixed", "anchor_conf", "ewma"]
    tapout_rewards = {a: 0.0 for a in tapout_arms}
    tapout_counts = {a: 1 for a in tapout_arms}
    tapout_chosen = "fixed"

    # Item 14 — Online Sequoia DP setup.
    sequoia_tracker = None
    sequoia_width_vec = None
    if online_sequoia:
        from model.sequoia_dp import PosAcceptanceTracker, solve_sequoia_widths
        sequoia_tracker = PosAcceptanceTracker(max_depth=block_size, alpha=0.95, prior=0.85)

    # Item 15 — Roofline-PID controller state.
    # Hill-climb: each step, try delta_M in current direction. If goodput improves,
    # keep direction; else flip. Goodput = mean_tau_this_step / step_time_this_step.
    pid_M: int = max_tree_size
    pid_dir: int = -2                 # initial probe: try smaller first
    pid_prev_goodput: float = -1.0    # -1 = no prior data
    pid_min_M: int = max(4, max_tree_size // 4)
    pid_max_M: int = max_tree_size

    # Idea 1e — target_q1 cache: last_logits from previous step's verify (target's
    # distribution at the bonus position). Used to replace draft's q_1 at depth 0
    # in the heap, since it's the EXACT target distribution there (lossless).
    prev_target_q1_logits: Optional[torch.Tensor] = None

    # Trim target cache to remove pad cols.
    cache_size = _trim_pad_after_prefill(past_kv_target, prefix_lens, S_max, device)
    cur_prefix_lens = prefix_lens.clone()
    cur_anchor = first_tok                                                       # [B, 1]

    # Batched draft. Cross-batch K_ctx padding (when target_hidden_valid varies
    # per element) is handled by a cumulative cache_pad_mask: True where the
    # corresponding cache slot is a phantom K_ctx entry that must be masked
    # out in subsequent attention. We build the SDPA attention mask for the
    # draft from cache_pad_mask + this-step's local K_ctx pad + K_noise (no pad).
    past_kv_draft = DynamicCache()
    target_hidden = target_hidden_full                                          # [B, pl_max, D']
    target_hidden_valid = prefix_lens.clone()                                    # [B] long
    cache_pad_mask = torch.zeros(B, 0, dtype=torch.bool, device=device)
    del target_hidden_full

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while (not all(finished)) and max(d.shape[1] for d in decoded_list) < max_new_tokens:
        step += 1
        torch.cuda.synchronize()
        t_step_start = time.perf_counter()

        # Pick M for this step. Priority: online_mts > ewma_adaptive >
        # anchor_signal > static max_tree_size.
        if online_mts:
            eff_M = _pick_m_online(
                B, ewma_tau, ewma_M_ref, online_mts_candidates_t,
                anchor_conf=prev_anchor_conf, block_size=block_size,
            )
            eff_ek = expand_k
        elif ewma_adaptive:
            # Linear-interp between min/max based on EWMA acceptance rate.
            eff_M = ewma_min_M + round((max_tree_size - ewma_min_M) * ewma_rate)
            eff_ek = ewma_min_ek + round((ewma_max_ek - ewma_min_ek) * ewma_rate)
            eff_ek = max(1, eff_ek)
        elif adaedl_B and prev_anchor_conf >= 0.0:
            u = max(0.0, 1.0 - prev_anchor_conf)
            eff_M = adaedl_min_M + round((adaedl_max_M - adaedl_min_M) * u)
            eff_ek = expand_k
        elif heap_conc_B and step > 1:
            # Item 2: Heap-concentration entropy. Compute the effective entropy
            # of the draft's per-position top-K distribution at the FIRST masked
            # position (= depth 1). High entropy → spread support → bigger M
            # needed to cover. Low entropy → peaked → small M. Diffusion-specific
            # because all marginals come from one pass; AR drafters can't see
            # this without extra forwards.
            # Computed below from this-step's draft_logits — for now, fallback
            # to anchor_conf if not yet known.
            u = max(0.0, 1.0 - prev_anchor_conf) if prev_anchor_conf >= 0.0 else 0.5
            eff_M = heap_conc_min_M + round((heap_conc_max_M - heap_conc_min_M) * u)
            eff_ek = expand_k
        elif roofline_pid:
            # Item 15: Hill-climb on M to maximise observed goodput tau/step_time.
            eff_M = pid_M
            eff_ek = expand_k
        else:
            eff_M = max_tree_size
            eff_ek = expand_k
        per_step_M_choices.append(eff_M)
        per_step_anchor_conf.append(prev_anchor_conf)

        # ------------------------------------------------------------
        # Batched draft step. Single forward over the whole batch with a 4D
        # SDPA mask that masks out cumulative K_ctx pad slots.
        #
        # K layout (after past_kv_draft.update appends local k):
        #   [0, S_pre)                            : prior steps' K_ctx (cumulative cache)
        #   [S_pre, S_pre + ctx_len)              : this step's K_ctx (target_hidden)
        #   [S_pre + ctx_len, S_pre + ctx_len + q): this step's K_noise (new block)
        # cache_pad_mask tracks True where a cache slot is phantom (a K_ctx pad
        # entry from a prior step's append). new_kctx_pad covers this step's pad.
        # K_noise has no padding (block_size positions are all valid).
        # ------------------------------------------------------------
        ctx_len = target_hidden.shape[1]
        q_len = block_size
        S_pre = cache_pad_mask.shape[1]

        block = torch.full((B, q_len), mask_token_id, dtype=torch.long, device=device)
        block[:, 0] = cur_anchor[:, 0]
        noise_emb = target.model.embed_tokens(block)                             # [B, q_len, H]

        # New step's K_ctx pad (per element).
        new_kctx_pad = (
            torch.arange(ctx_len, device=device).unsqueeze(0)
            >= target_hidden_valid.unsqueeze(1)
        )                                                                         # [B, ctx_len] bool

        # Combined per-key-position pad mask: cache + this-step K_ctx + K_noise (no pad).
        full_pad = torch.cat([
            cache_pad_mask,                                                       # [B, S_pre]
            new_kctx_pad,                                                         # [B, ctx_len]
            torch.zeros(B, q_len, dtype=torch.bool, device=device),               # [B, q_len]
        ], dim=1)                                                                 # [B, S_pre + ctx_len + q_len]

        min_val = torch.finfo(torch.bfloat16).min
        draft_attn_mask = torch.zeros(
            B, 1, q_len, S_pre + ctx_len + q_len,
            dtype=torch.bfloat16, device=device,
        )
        draft_attn_mask.masked_fill_(full_pad.view(B, 1, 1, -1), min_val)

        # Position IDs of length ctx_len + q_len. Per-element:
        #   K_ctx: positions = [cur_prefix_lens[b] - target_hidden_valid[b] ..
        #                       cur_prefix_lens[b] - target_hidden_valid[b] + ctx_len - 1]
        #   K_noise / Q: positions = [cur_prefix_lens[b] .. cur_prefix_lens[b] + q_len - 1]
        ctx_start_per_b = (cur_prefix_lens - target_hidden_valid).clamp(min=0)    # [B]
        ctx_pos = ctx_start_per_b.unsqueeze(1) + torch.arange(ctx_len, device=device).unsqueeze(0)
        block_pos = cur_prefix_lens.unsqueeze(1) + torch.arange(q_len, device=device).unsqueeze(0)
        long_pos = torch.cat([ctx_pos, block_pos], dim=1)                         # [B, ctx_len + q_len]

        draft_hidden = draft(
            target_hidden=target_hidden, noise_embedding=noise_emb,
            position_ids=long_pos, attention_mask=draft_attn_mask,
            past_key_values=past_kv_draft, use_cache=True, is_causal=False,
        )                                                                          # [B, q_len, H]

        # Crop draft cache: keep only [0, S_pre + ctx_len) i.e. drop K_noise.
        # This mirrors the original B=1 `past_kv_draft.crop(start)` semantics.
        new_cache_size = S_pre + ctx_len
        for layer in past_kv_draft.layers:
            if layer.keys.shape[2] > new_cache_size:
                layer.keys = layer.keys[:, :, :new_cache_size, :].contiguous()
                layer.values = layer.values[:, :, :new_cache_size, :].contiguous()
        cache_pad_mask = torch.cat([cache_pad_mask, new_kctx_pad], dim=1)         # [B, new_cache_size]

        draft_logits = target.lm_head(draft_hidden[:, 1:, :])                     # [B, q_len-1, V]

        # Idea 1e / Item 4 — DISABLED. Probe at wrong position 80%+ of steps
        # when rank-0 chain partially accepts (tau<block_size). Empirical:
        # tau drops from 10.5 to 2.45 because spliced q_1 is for a position
        # past the actual accept. Needs per-step decision: only splice when
        # PREVIOUS step's chain accepted to chain-end. Bookkeeping not done.
        if False and (use_target_q1 or extend_tree_q1) and prev_target_q1_logits is not None and step > 1:
            draft_logits = draft_logits.clone()
            draft_logits[:, 0, :] = prev_target_q1_logits.to(draft_logits.dtype)

        # Idea 1c — entropy-guided per-position expand_k. Computed before
        # tree-build, passed in per-element. High entropy at depth d → wider
        # search there (more rank-j candidates); low entropy → narrow.
        per_pos_ek_per_elem = None
        if entropy_width:
            with torch.no_grad():
                p = torch.softmax(draft_logits.float(), dim=-1)            # [B, q_len-1, V]
                H = -(p * p.clamp(min=1e-12).log()).sum(dim=-1)            # [B, q_len-1]
                Vsize = draft_logits.shape[-1]
                u = (H / math.log(Vsize)).clamp(0.0, 1.0)
                eks = (entropy_width_min_ek + (entropy_width_max_ek - entropy_width_min_ek) * u)
                eks = eks.round().clamp(min=1, max=expand_k).long()        # [B, q_len-1]
                per_pos_ek_per_elem = eks.detach().cpu().tolist()

        # Anchor-entropy / draft-conf forward-signal adaptive sizing.
        # Uses CURRENT step's draft logits at depth 0 (the first masked position
        # = anchor's NEXT prediction). Overrides eff_M for this step only;
        # leaves eff_ek alone (the original benchmark.py only adapts size).
        if anchor_signal != "none":
            anchor_logits = draft_logits[:, 0, :].float()                          # [B, V]
            if anchor_signal == "entropy":
                p = torch.softmax(anchor_logits, dim=-1)
                H = -(p * p.clamp(min=1e-12).log()).sum(dim=-1)                    # [B]
                Vsize = anchor_logits.shape[-1]
                u = (H / math.log(Vsize)).clamp(0.0, 1.0)
            elif anchor_signal == "draft-conf":
                top1 = torch.softmax(anchor_logits, dim=-1).max(dim=-1).values     # [B]
                u = (1.0 - top1).clamp(0.0, 1.0)
            else:
                u = torch.ones(B, device=device)
            u = u.pow(anchor_gamma)
            mean_u = float(u.mean().item())
            eff_M = anchor_min_M + round((max_tree_size - anchor_min_M) * mean_u)
            per_step_M_choices[-1] = eff_M  # overwrite the static choice

        # ------------------------------------------------------------
        # Tree build (per-element loop, padded outputs).
        # ------------------------------------------------------------
        # Per-element anchor entropy for BJC conditioning bucket.
        bjc_ent = None
        if bjc_calib is not None or heap_conc_B:
            d0 = draft_logits[:, 0, :].float()
            p0 = torch.softmax(d0, dim=-1)
            H = -(p0 * p0.clamp(min=1e-12).log()).sum(dim=-1)
            ent_norm = (H / math.log(d0.shape[-1])).clamp(0.0, 1.0)
            bjc_ent = ent_norm.detach().cpu().tolist()

        # Item 2: heap concentration entropy override of eff_M (post-draft signal).
        # Re-derive M from current step's draft anchor entropy if heap_conc_B is on.
        if heap_conc_B:
            d0 = draft_logits[:, 0, :].float()
            p0 = torch.softmax(d0, dim=-1)
            H_anchor = -(p0 * p0.clamp(min=1e-12).log()).sum(dim=-1)
            ent_norm = (H_anchor / math.log(d0.shape[-1])).clamp(0.0, 1.0)
            mean_u = float(ent_norm.mean().item())
            eff_M = heap_conc_min_M + round((heap_conc_max_M - heap_conc_min_M) * mean_u)
            per_step_M_choices[-1] = eff_M  # overwrite

        # Item 4: extend tree by one extra depth so target predicts q_1
        # for next step's first masked position. Implementation: take the
        # heap-built tree, then append one extra node per element that's
        # the rank-0 extension of the rank-0 chain's leaf. Target's logits
        # at this extra node = target's distribution at sequence position
        # (cur_prefix_lens[b] + leaf_depth + 1), which becomes q_1 for next
        # step's depth-1 candidates IF the rank-0 chain accepts to its end.
        # Captured below after tree-build, before verify.

        # Item 9 — EAGLE-2-style over-build then rerank: build a tree of
        # eagle2_pool_mult * eff_M nodes, then keep top eff_M by joint score.
        # Implemented by raising max_tree_size and trusting the heap to pick
        # the best; we just record the budget.
        if eagle2_overbuild:
            eff_M_for_build = min(eff_M * max(1, eagle2_pool_mult), 256)
        else:
            eff_M_for_build = eff_M

        # Item 10 — SpecDec++ MDP threshold: stop tree build if next pop's
        # rejection probability exceeds threshold (uses heap's path prob).
        # Reuse OPT-Tree termination machinery with a tighter threshold.
        if specdecpp_threshold > 0.0:
            optree_thresh_eff = specdecpp_threshold
        else:
            optree_thresh_eff = optree_threshold if optree_termination else 0.0

        # Tree-build dispatch: vectorized fixed-width OR per-element heap.
        # Item 14 — online Sequoia DP: force vectorized path with learned width.
        use_vec = use_vectorized_tree or (online_sequoia and sequoia_width_vec is not None)
        if use_vec:
            if online_sequoia and sequoia_width_vec is not None:
                wv = sequoia_width_vec
            else:
                wv = vectorized_width_vec or default_width_vec(eff_M_for_build, expand_k=expand_k)
            packed_ids, packed_pos_rel, parent_idx, node_valid, leaf_paths, leaf_tokens, leaf_valid = \
                vectorized_tree_build(
                    draft_logits=draft_logits, anchor_token_ids=cur_anchor[:, 0],
                    width_vec=wv, K=expand_k, max_M=eff_M_for_build + 1,
                )
        else:
            packed_ids, packed_pos_rel, parent_idx, node_valid, leaf_paths, leaf_tokens, leaf_valid = \
                build_node_budget_tree_batched(
                    draft_logits=draft_logits, anchor_token_ids=cur_anchor[:, 0],
                    max_tree_size=eff_M_for_build, expand_k=eff_ek,
                    pdrr_k1=pdrr_k1, pdrr_k2=pdrr_k2, pdrr_k3=pdrr_k3,
                    cgdb_shallow_depth=cgdb_shallow_depth,
                    cgdb_high_thresh=cgdb_high_thresh,
                    cgdb_low_thresh=cgdb_low_thresh,
                    cgdb_mid_k=cgdb_mid_k,
                    bjc_calib=bjc_calib,
                    bjc_anchor_ent=bjc_ent,
                    per_pos_ek_per_elem=per_pos_ek_per_elem,
                    optree_threshold=optree_thresh_eff,
                )
        M = packed_ids.shape[1]
        tree_node_counts.append(int(node_valid.sum(dim=-1).float().mean().item()))

        # Item 4: append one extra "q_1 probe" node per element. We pick the
        # rank-0 chain's deepest node (= node with depth = max(packed_pos_rel)
        # along rank-0 chain) and append the draft-argmax extension.
        # Implementation: find per-element index of the deepest rank-0 node.
        if extend_tree_q1 and step > 0:
            with torch.no_grad():
                # Rank-0 chain: starting from anchor (idx 0), follow first child
                # at each depth. Approximation: use the heap's depth ordering —
                # depth-d slot 0 (in original packed ordering) is on the rank-0
                # chain by construction (heap pushes rank-0 first).
                # For each element, find the index of the deepest such node.
                max_depth_per = packed_pos_rel.max(dim=-1).values         # [B]
                # The deepest rank-0 leaf: find any node whose depth = max_depth.
                # Use the first such node per element.
                is_max = (packed_pos_rel == max_depth_per.unsqueeze(-1))   # [B, M]
                deepest_idx = is_max.float().argmax(dim=-1)                # [B]
                # The token at that deepest node:
                deepest_token = packed_ids.gather(1, deepest_idx.unsqueeze(1)).squeeze(1)
                # Append a new node: token = deepest_token (treat as anchor of a
                # tiny chain), parent = deepest_idx, depth = max_depth + 1.
                # Token to append: draft's predicted argmax for position
                # (cur_prefix_lens[b] + max_depth + 1) — but draft_logits only
                # covers positions 0..block_size-2 of THIS step. For depth
                # max_depth+1 within this step's block, draft_logits[:, max_depth, :]
                # if max_depth < block_size-1.
                new_tok_per = []
                new_pos_per = []
                new_par_per = []
                B_now = packed_ids.shape[0]
                d0 = max_depth_per.detach().cpu().tolist()
                deep_idx_cpu = deepest_idx.detach().cpu().tolist()
                for b in range(B_now):
                    md = int(d0[b])
                    if md + 1 < draft_logits.shape[1] + 1:  # we have draft logits up to depth block_size-1
                        # Draft prediction for the position one beyond max_depth.
                        # draft_logits indexes are over the masked positions
                        # of the new block (depths 1..block_size-1), so
                        # draft_logits[b, md-1] would be the parent's prediction.
                        # For one beyond (md+1), we don't have it from draft.
                        # Fallback: use draft_logits[b, md-1] argmax as the
                        # extension token (target will refine via verify).
                        if md - 1 < draft_logits.shape[1] and md - 1 >= 0:
                            tok = int(draft_logits[b, md - 1].argmax().item())
                        else:
                            tok = int(deepest_token[b].item())
                    else:
                        tok = int(deepest_token[b].item())
                    new_tok_per.append(tok)
                    new_pos_per.append(md + 1)
                    new_par_per.append(deep_idx_cpu[b])
                new_tok_t = torch.tensor(new_tok_per, dtype=torch.long, device=device).unsqueeze(1)
                new_pos_t = torch.tensor(new_pos_per, dtype=torch.long, device=device).unsqueeze(1)
                new_par_t = torch.tensor(new_par_per, dtype=torch.long, device=device).unsqueeze(1)
                packed_ids = torch.cat([packed_ids, new_tok_t], dim=1)
                packed_pos_rel = torch.cat([packed_pos_rel, new_pos_t], dim=1)
                parent_idx = torch.cat([parent_idx, new_par_t], dim=1)
                node_valid = torch.cat([node_valid, torch.ones(B_now, 1, dtype=torch.bool, device=device)], dim=1)
                # Don't add to leaf_paths / leaf_tokens — this node isn't a
                # candidate for acceptance; it's purely for q_1 harvesting.
                M = packed_ids.shape[1]

        # Per-element absolute target positions for the tree nodes.
        packed_pos_abs = cur_prefix_lens.unsqueeze(1) + packed_pos_rel           # [B, M]

        # Tree topology mask + per-element prefix-pad mask.
        attn_tree = create_tree_attention_mask_batched(
            position_ids=packed_pos_rel, parent_idx=parent_idx,
            node_valid=node_valid, prefix_len=cache_size,
        )                                                                         # [B, 1, M, cache_size+M]
        col_idx = torch.arange(cache_size, device=device).unsqueeze(0)
        pad_cols = col_idx >= cur_prefix_lens.unsqueeze(1)
        pad_4d = pad_cols.view(B, 1, 1, cache_size).expand(-1, 1, M, -1)
        min_val_bf16 = torch.finfo(torch.bfloat16).min
        attn_tree[:, :, :, :cache_size] = torch.where(
            pad_4d, torch.full_like(attn_tree[:, :, :, :cache_size], min_val_bf16),
            attn_tree[:, :, :, :cache_size],
        )

        # Target verify (need hidden states to update target_hidden post-step).
        saved_attn = target.config._attn_implementation
        target.config._attn_implementation = "sdpa"
        out = target.model(
            packed_ids, position_ids=packed_pos_abs,
            past_key_values=past_kv_target, use_cache=True,
            attention_mask=attn_tree, output_hidden_states=True,
        )
        logits = target.lm_head(out.last_hidden_state)                            # [B, M, V]
        verify_ctx_feat = extract_context_feature(out.hidden_states, draft.target_layer_ids)  # [B, M, D']
        target.config._attn_implementation = saved_attn

        # BJC-Tree: feed every verified node's (parent_pattern, target_argmax)
        # observation into the running joint-distribution calibration.
        if bjc_calib is not None:
            # Build draft top-K indices (K = expand_k from earlier).
            draft_topk_idx = draft_logits.topk(k=expand_k, dim=-1).indices  # [B, seq_len, K]
            # Anchor entropy normalised: H(softmax(draft_logits[:, 0])) / log(V).
            draft_logits_d0 = draft_logits[:, 0, :].float()
            p0 = torch.softmax(draft_logits_d0, dim=-1)
            H = -(p0 * p0.clamp(min=1e-12).log()).sum(dim=-1)        # [B]
            ent_norm = (H / math.log(draft_logits_d0.shape[-1])).clamp(0.0, 1.0)
            try:
                bjc_calib.update_from_verify(
                    packed_pos=packed_pos_rel,
                    parent_idx=parent_idx,
                    node_valid=node_valid,
                    target_logits=logits,
                    draft_topk_idx=draft_topk_idx,
                    node_token_ids=packed_ids,
                    anchor_ent_per_elem=ent_norm,
                )
            except Exception as e:
                print(f"  [BJC] update failed: {e}")

        # Accept-path select.
        best_leaf, n_accepted = select_best_dynamic_leaf_batched(
            logits=logits, leaf_paths=leaf_paths, leaf_tokens=leaf_tokens,
            leaf_valid=leaf_valid, temperature=temperature,
            pwls=pwls, bqat_threshold=bqat_threshold,
            cppr=cppr, cppr_lambda=cppr_lambda,
        )                                                                          # [B], [B]
        n_plus_1 = (n_accepted + 1).to(torch.long)
        max_n1 = int(n_plus_1.max().item())
        b_arange = torch.arange(B, device=device)
        accepted_paths = torch.zeros(B, max_n1, dtype=torch.long, device=device)
        for b in range(B):
            n1 = int(n_plus_1[b].item())
            accepted_paths[b, :n1] = leaf_paths[b, best_leaf[b], :n1]

        last_node_idx = accepted_paths.gather(1, n_accepted.unsqueeze(1)).squeeze(1)
        last_logits = logits[b_arange, last_node_idx, :]
        bonus_tok = last_logits.argmax(dim=-1, keepdim=True)                      # [B, 1]

        # Idea 1e — cache target's full softmax distribution at the bonus.
        # This IS the true target's q_1 for next step's first masked position.
        if use_target_q1:
            prev_target_q1_logits = last_logits.detach().clone()

        # Item 4: harvest the extra "q_1 probe" node's target logits.
        # The probe is the LAST appended node in packed_ids (idx M-1).
        # Its target logits predict the position one beyond the deepest
        # rank-0 chain — which is q_1 for next step IF the rank-0 chain
        # accepts to its end this step. Otherwise it's stale (wrong position).
        if extend_tree_q1:
            # logits has shape [B, M, V]. Extract the probe row.
            probe_logits = logits[:, -1, :].detach().clone()
            # We'll use it next step in place of draft_logits[:, 0, :] only
            # if we accepted the full chain (n_accepted >= max_depth_per).
            # Stash here; activated at start of next step via prev_target_q1_logits.
            prev_target_q1_logits = probe_logits

        # Forward-looking signal for next step's M selection: target's top-1
        # probability at the bonus token (= NEXT step's anchor). High p means
        # the new block is likely an argmax run; low p means a hard step
        # ahead. Only relevant when online_mts is on.
        if online_mts or adaedl_B:
            last_probs = torch.softmax(last_logits.float(), dim=-1)
            top1_probs = last_probs.gather(1, bonus_tok).squeeze(1)               # [B]
            # Aggregate: mean across batch (could be min for worst-case bias).
            prev_anchor_conf = float(top1_probs.mean().item())

        accepted_tokens = packed_ids.gather(1, accepted_paths)                    # [B, max_n1]

        # Append per-element real new tokens to per-element list. cur_anchor is
        # built per-element: bonus_tok for non-finished elems, last token for finished.
        new_anchors = []
        for b in range(B):
            if finished[b]:
                new_anchors.append(decoded_list[b][:, -1:])
                continue
            n_b = int(n_accepted[b].item())
            parts = []
            if n_b > 0:
                parts.append(accepted_tokens[b, 1:1 + n_b].unsqueeze(0))           # [1, n_b]
            parts.append(bonus_tok[b:b+1, :])                                       # [1, 1]
            cur_new = torch.cat(parts, dim=1)                                       # [1, n_b+1]
            decoded_list[b] = torch.cat([decoded_list[b], cur_new], dim=1)
            accepted_lengths_per_elem[b].append(n_b + 1)
            for tok in cur_new[0].tolist():
                if int(tok) in eos_set:
                    finished[b] = True
                    break
            new_anchors.append(bonus_tok[b:b+1, :])

        cur_anchor = torch.cat(new_anchors, dim=0)                                  # [B, 1]

        # Update EWMA acceptance rate for ewma_adaptive (mirrors benchmark.py
        # line 866: rate = n / max(eff_bs - 1, 1)). Aggregate across batch.
        if ewma_adaptive:
            non_finished_n = [
                accepted_lengths_per_elem[b][-1] - 1  # n = (n+1) - 1 (subtract bonus)
                for b in range(B) if accepted_lengths_per_elem[b]
            ]
            if non_finished_n:
                step_rate = sum(non_finished_n) / (len(non_finished_n) * max(block_size - 1, 1))
                ewma_rate = ewma_decay * ewma_rate + (1 - ewma_decay) * step_rate

        # Update EWMA of acceptance length for online M*.
        if online_mts:
            # Mean acceptance over non-finished elements this step.
            step_acc_per_elem = []
            for b in range(B):
                if accepted_lengths_per_elem[b]:
                    step_acc_per_elem.append(accepted_lengths_per_elem[b][-1])
            if step_acc_per_elem:
                step_mean = sum(step_acc_per_elem) / len(step_acc_per_elem)
                ewma_tau = online_mts_alpha * ewma_tau + (1 - online_mts_alpha) * step_mean
                ewma_M_ref = eff_M

        # Update target_hidden to the accept-path hidden states, padded across
        # batch to max(n+1). Padded slots are repeats of the last real entry
        # (their RoPE positions land in OOD territory but are masked out via
        # the K_ctx pad bookkeeping in subsequent steps).
        target_hidden_per = []
        for b in range(B):
            n1 = int(n_plus_1[b].item())
            slice_b = verify_ctx_feat[b, accepted_paths[b, :n1], :]                 # [n1, D']
            if n1 < max_n1:
                pad_v = slice_b[-1:].repeat(max_n1 - n1, 1)
                slice_b = torch.cat([slice_b, pad_v], dim=0)
            target_hidden_per.append(slice_b)
        target_hidden = torch.stack(target_hidden_per, dim=0)                       # [B, max_n1, D']
        target_hidden_valid = n_plus_1.clone()                                       # [B]

        # For finished elements, freeze cache state by keeping zero new tokens.
        n1_for_trim = n_plus_1.clone()
        for b in range(B):
            if finished[b] and accepted_lengths_per_elem[b] and \
                    int(decoded_list[b][0, -1].item()) in eos_set:
                # Element finished THIS step: still take its accepted path so cache
                # holds the EOS context. Skip trimming new content only on subsequent
                # steps after it's already been registered finished.
                pass
            elif finished[b]:
                n1_for_trim[b] = 0

        # Trim target KV cache.
        cur_prefix_lens = _trim_target_kv_cache(
            past_kv=past_kv_target,
            cur_prefix_lens=cur_prefix_lens,
            accepted_tree_paths=accepted_paths,
            n1_per_elem=n1_for_trim,
            old_cache_size=cache_size, device=device,
        )
        cache_size = int(cur_prefix_lens.max().item())

        # Item 14 — Online Sequoia DP: feed verify outputs into the positional
        # acceptance tracker, recompute width vec every sequoia_recompute_every steps.
        if online_sequoia and sequoia_tracker is not None:
            try:
                with torch.no_grad():
                    target_argmax_at_node = logits.argmax(dim=-1).detach().cpu().numpy()  # [B, M]
                    # Draft top-1 per depth: depth 0 unused (anchor), depths 1..bs-1
                    # come from draft_logits[:, 0..bs-2, :].argmax.
                    dl_top1 = draft_logits.argmax(dim=-1).detach().cpu().numpy()  # [B, bs-1]
                    draft_top1_full = np.zeros((B, block_size), dtype=np.int64)
                    draft_top1_full[:, 1:1 + dl_top1.shape[1]] = dl_top1
                    node_depth_np = packed_pos_rel.detach().cpu().numpy()
                    node_valid_np = node_valid.detach().cpu().numpy()
                    sequoia_tracker.update_from_verify(
                        target_argmax_at_node=target_argmax_at_node,
                        draft_top1_at_depth=draft_top1_full,
                        node_depth=node_depth_np,
                        node_valid=node_valid_np,
                        seq_len=block_size,
                    )
                    if step % max(1, sequoia_recompute_every) == 0:
                        from model.sequoia_dp import solve_sequoia_widths
                        p_d = sequoia_tracker.get_p_d()
                        sequoia_width_vec = solve_sequoia_widths(
                            p_d=p_d, M=eff_M, K=expand_k,
                        )
            except Exception as e:
                print(f"  [Sequoia] update failed: {e}")

        # Item 15 — Roofline-PID hill-climb on M.
        if roofline_pid:
            torch.cuda.synchronize()
            step_time = time.perf_counter() - t_step_start
            # Mean accept length this step (across non-finished elements).
            non_fin_n1 = [int(n_plus_1[b].item()) for b in range(B) if not finished[b]]
            mean_tau = sum(non_fin_n1) / max(len(non_fin_n1), 1) if non_fin_n1 else 1.0
            cur_goodput = mean_tau / max(step_time, 1e-6)
            if pid_prev_goodput > 0.0:
                # Direction worked? Keep it. Else flip.
                if cur_goodput < pid_prev_goodput:
                    pid_dir = -pid_dir
            pid_M = max(pid_min_M, min(pid_max_M, pid_M + pid_dir))
            pid_prev_goodput = cur_goodput

    torch.cuda.synchronize()
    total = time.perf_counter() - t0

    # Build per-element output_ids (concat real prompt + decoded portion truncated at EOS).
    output_ids_list = []
    num_out = []
    for b in range(B):
        d = decoded_list[b]                                  # [1, n_real]
        toks = d[0].tolist()
        n_eos = len(toks)
        for i, t in enumerate(toks):
            if int(t) in eos_set:
                n_eos = i + 1
                break
        pl_b = int(prefix_lens[b].item())
        out_seq = torch.cat([input_ids[b:b+1, :pl_b], d[:, :n_eos]], dim=1)
        output_ids_list.append(out_seq)
        num_out.append(n_eos)

    return SimpleNamespace(
        output_ids_list=output_ids_list,
        num_output_tokens=num_out,
        acceptance_lengths_per_elem=accepted_lengths_per_elem,
        tree_node_counts=tree_node_counts,
        per_step_M_choices=per_step_M_choices,
        per_step_anchor_conf=per_step_anchor_conf,
        total_decode_time=total,
    )
