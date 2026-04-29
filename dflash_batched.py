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
    cache_inject: bool = False,                  # Rejection-derived cache injection
    cache_slots: int = 4,                        # extra tree slots reserved for cached candidates
    log_rejection_ranks: bool = False,           # Diagnostic: log target.argmax's rank in draft q_i at rejection points
    log_phase_timings: bool = False,             # Diagnostic: per-phase wall time per round
    chain_mode: bool = False,                    # Skip heap; build linear chain (= rank-0 chain at M=block_size)
    spine_first: bool = False,                   # Two-stage verify: spine first, side only on no-full-accept
    score_decay: float = 1.0,                    # Heap-score per-depth decay weight (training-objective alignment)
    score_weights_per_depth: Optional[List[float]] = None,  # U-Shape Tree: per-depth weight schedule (overrides score_decay)
    expand_k_per_depth: Optional[List[int]] = None,         # U-Shape Tree: per-depth K cap
    dev_depth_bonus: Optional[List[float]] = None,          # LDB: per-deviation-depth score adjustment
    rank_conditional_decay: float = 1.0,                    # RCD: post-deviation depth decay (path-conditional)
    adaptive_decay: bool = False,                           # A-CDDT: per-round γ from anchor entropy
    adaptive_decay_easy: float = 0.85,                      # A-CDDT: γ when round is "easy" (low anchor entropy)
    adaptive_decay_hard: float = 1.0,                       # A-CDDT: γ when round is "hard"
    adaptive_decay_threshold: float = 0.4,                  # A-CDDT: anchor entropy threshold (normalized)
    workload_adaptive_decay: bool = False,                  # W-CDDT: workload-level γ from EWMA of full-block rate
    workload_decay_threshold: float = 0.18,                 # W-CDDT: full-block rate threshold to apply decay
    workload_ewma_alpha: float = 0.9,                       # W-CDDT: EWMA decay rate
    learned_curve_decay: bool = False,                      # L-CDDT: online per-depth weight from P(accept|alive) EWMA
    learned_curve_alpha: float = 0.9,                       # L-CDDT: EWMA decay rate for per-depth counts
    learned_curve_warmup: int = 5,                          # L-CDDT: rounds before applying learned curve
    learned_curve_power: float = 1.0,                       # L-CDDT: exponent on (curve)^power; >1 amplifies gradient
    log_path_trajectory: bool = False,           # Diagnostic: log per-depth rank-in-q for best-leaf path
    log_full_tree_dump: bool = False,            # Diagnostic: per-round dump every leaf's path-of-ranks + accept length
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

    # W-CDDT state: EWMA of full-block accept rate (n_accepted == block_size-1).
    # Initialized at 0 (assume "hard" at start); updated each round.
    wcddt_full_block_ewma: float = 0.0
    wcddt_warmup_steps: int = 4   # before this many rounds, default to no decay

    # L-CDDT state: per-depth EWMA of (alive_count, accepted_count) along
    # rank-0 chain. After warmup, ratio = depth_acc / depth_alive becomes the
    # weight at depth d.
    lcddt_depth_alive: List[float] = [0.0] * (block_size + 1)     # index 0 unused (anchor)
    lcddt_depth_acc: List[float] = [0.0] * (block_size + 1)
    lcddt_curve: List[float] = [1.0] * (block_size + 1)           # initialized uniform
    lcddt_step_count: int = 0

    # Rejection-rank diagnostic: list of (depth, rank) tuples. rank = position of
    # target's argmax in draft q_i (sorted descending). 0 = top-1, 1 = top-2, etc.
    # Captured at the rejection point of every non-full round.
    rejection_ranks: List[Tuple[int, int]] = []

    # Path-trajectory diagnostic: for each round, log the BEST leaf's path-of-ranks
    # (= rank-in-q at each depth along the best leaf) plus rank-0 chain accept
    # length plus best-leaf accept length. Used to verify rank-0-rejection cause.
    path_trajectories: List[dict] = []
    # Full-tree dump: every leaf's (path-ranks, accept-length, deviation depth).
    full_tree_dumps: List[dict] = []

    # Per-phase wall time accumulators (ms).
    phase_t_draft: float = 0.0
    phase_t_tree_build: float = 0.0
    phase_t_verify: float = 0.0
    phase_t_select: float = 0.0
    phase_t_trim: float = 0.0
    phase_t_other: float = 0.0
    phase_step_count: int = 0
    # Spine-first sub-phase timings.
    phase_t_spine_verify: float = 0.0
    phase_t_side_verify: float = 0.0
    phase_spine_full_count: int = 0
    phase_side_runs: int = 0

    # Rejection-derived cache: for each elem, list of cached tokens at relative
    # depths [1..cache_slots]. Captured post-verify from rank-0 chain target argmax
    # at depths past the previous round's accept point. Used as additional candidates
    # in the next round's tree (appended off the anchor as a single linear chain).
    prev_cache_per_elem: List[List[int]] = [[] for _ in range(B)]
    # Diagnostics: cache hits per slot index (0..cache_slots-1), and per round.
    cache_hits_per_slot: List[int] = [0] * max(1, cache_slots)
    cache_attempts_per_slot: List[int] = [0] * max(1, cache_slots)
    cache_round_had_hit: int = 0       # rounds where cache contributed at least 1 token
    cache_round_total: int = 0         # rounds where we attempted cache injection

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

        if log_phase_timings:
            torch.cuda.synchronize(); _t0p = time.perf_counter()
        draft_hidden = draft(
            target_hidden=target_hidden, noise_embedding=noise_emb,
            position_ids=long_pos, attention_mask=draft_attn_mask,
            past_key_values=past_kv_draft, use_cache=True, is_causal=False,
        )                                                                          # [B, q_len, H]
        if log_phase_timings:
            torch.cuda.synchronize(); phase_t_draft += time.perf_counter() - _t0p

        # Crop draft cache: keep only [0, S_pre + ctx_len) i.e. drop K_noise.
        # This mirrors the original B=1 `past_kv_draft.crop(start)` semantics.
        new_cache_size = S_pre + ctx_len
        for layer in past_kv_draft.layers:
            if layer.keys.shape[2] > new_cache_size:
                layer.keys = layer.keys[:, :, :new_cache_size, :].contiguous()
                layer.values = layer.values[:, :, :new_cache_size, :].contiguous()
        cache_pad_mask = torch.cat([cache_pad_mask, new_kctx_pad], dim=1)         # [B, new_cache_size]

        draft_logits = target.lm_head(draft_hidden[:, 1:, :])                     # [B, q_len-1, V]

        # A-CDDT: per-round γ from mean anchor entropy across batch.
        # Low entropy → "easy" round → apply decay (encourage shallow-focus).
        # High entropy → "hard" round → no decay (preserve all info).
        eff_score_decay = score_decay
        if adaptive_decay:
            with torch.no_grad():
                anchor_lp = draft_logits[:, 0, :].float()
                p0 = torch.softmax(anchor_lp, dim=-1)
                H = -(p0 * p0.clamp(min=1e-12).log()).sum(dim=-1)                    # [B]
                Vsize = anchor_lp.shape[-1]
                ent_norm = (H / math.log(Vsize)).clamp(0.0, 1.0)
                mean_ent = float(ent_norm.mean().item())
                eff_score_decay = (
                    adaptive_decay_easy if mean_ent < adaptive_decay_threshold
                    else adaptive_decay_hard
                )
        # W-CDDT: workload-level γ from EWMA of full-block accept rate.
        # If past rounds frequently fully accepted (drafter does well at deep depths
        # = U-shape recovery is present), apply decay. Otherwise don't.
        if workload_adaptive_decay:
            if step <= wcddt_warmup_steps:
                eff_score_decay = adaptive_decay_hard  # 1.0 by default
            else:
                eff_score_decay = (
                    adaptive_decay_easy if wcddt_full_block_ewma >= workload_decay_threshold
                    else adaptive_decay_hard
                )

        # L-CDDT: replace heap weights with the online per-depth curve.
        eff_score_weights = score_weights_per_depth
        if learned_curve_decay and lcddt_step_count >= learned_curve_warmup:
            eff_score_weights = list(lcddt_curve)

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
        if log_phase_timings:
            torch.cuda.synchronize(); _t0p = time.perf_counter()

        # Chain mode: skip heap entirely. Build linear chain from draft argmaxes.
        # Tree-attention path still used (so verify integration unchanged), but
        # the tree IS a chain with no siblings — mask becomes effectively causal.
        if chain_mode:
            L = block_size
            B_now = draft_logits.shape[0]
            # packed_ids[b, 0] = anchor; [b, d] = draft argmax at masked pos d-1
            chain_argmax = draft_logits.argmax(dim=-1)                            # [B, q_len-1]
            packed_ids = torch.cat([
                cur_anchor[:, :1],
                chain_argmax,                                                       # [B, q_len-1]
            ], dim=1)                                                                # [B, q_len]
            # packed_pos_rel: [0, 1, 2, ..., L-1]
            packed_pos_rel = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0).expand(B_now, -1).contiguous()
            # parent_idx: [0, 0, 1, 2, ..., L-2] — node d's parent is d-1 (anchor self-parents).
            par_local = torch.cat([
                torch.zeros(1, dtype=torch.long, device=device),
                torch.arange(L - 1, dtype=torch.long, device=device),
            ])                                                                        # [L]
            parent_idx = par_local.unsqueeze(0).expand(B_now, -1).contiguous()
            node_valid = torch.ones(B_now, L, dtype=torch.bool, device=device)
            # Single leaf, path = [0, 1, ..., L-1]
            leaf_paths = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0).unsqueeze(0).expand(B_now, 1, -1).contiguous()
            leaf_tokens = chain_argmax.unsqueeze(1).contiguous()                    # [B, 1, L-1]
            leaf_valid = torch.ones(B_now, 1, dtype=torch.bool, device=device)
        elif (use_vec := (use_vectorized_tree or (online_sequoia and sequoia_width_vec is not None))):
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
                    score_decay=eff_score_decay,
                    score_weights_per_depth=eff_score_weights,
                    expand_k_per_depth=expand_k_per_depth,
                    dev_depth_bonus=dev_depth_bonus,
                    rank_conditional_decay=rank_conditional_decay,
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

        # Rejection-cache injection: append a single linear cache chain off the
        # anchor (idx 0). Each cache slot c gets depth (c+1), parent = previous cache
        # node (or anchor for c=0). Becomes its own leaf path of length up to cache_slots+1.
        # Diagnostics: track per-slot acceptance rate.
        cache_chain_start: int = -1                                              # index of cache_slot[0] in packed_ids
        cache_chain_len_per: List[int] = [0] * B
        if cache_inject and step > 1:
            max_slots = min(cache_slots, max((len(c) for c in prev_cache_per_elem), default=0))
            if max_slots > 0:
                cache_tok_t = torch.zeros(B, max_slots, dtype=torch.long, device=device)
                cache_valid_t = torch.zeros(B, max_slots, dtype=torch.bool, device=device)
                for b in range(B):
                    n_avail = min(max_slots, len(prev_cache_per_elem[b]))
                    cache_chain_len_per[b] = n_avail
                    for c in range(n_avail):
                        cache_tok_t[b, c] = int(prev_cache_per_elem[b][c])
                        cache_valid_t[b, c] = True
                cache_chain_start = M
                # Append nodes: depths [1..max_slots], chain parents.
                cache_pos_t = (torch.arange(max_slots, device=device) + 1).unsqueeze(0).expand(B, -1)
                cache_par_t = torch.zeros(B, max_slots, dtype=torch.long, device=device)
                cache_par_t[:, 0] = 0  # anchor
                for c in range(1, max_slots):
                    cache_par_t[:, c] = M + c - 1
                packed_ids = torch.cat([packed_ids, cache_tok_t], dim=1)
                packed_pos_rel = torch.cat([packed_pos_rel, cache_pos_t.to(torch.long)], dim=1)
                parent_idx = torch.cat([parent_idx, cache_par_t], dim=1)
                node_valid = torch.cat([node_valid, cache_valid_t], dim=1)
                # Add a new leaf path: [anchor, M, M+1, ..., M+max_slots-1].
                cache_path_nodes = (
                    torch.arange(max_slots, device=device).unsqueeze(0).expand(B, -1) + M
                )
                cache_leaf_path = torch.cat([
                    torch.zeros(B, 1, dtype=torch.long, device=device),
                    cache_path_nodes.to(torch.long),
                ], dim=1).unsqueeze(1)                                            # [B, 1, max_slots+1]
                cache_leaf_tokens_t = cache_tok_t.unsqueeze(1)                    # [B, 1, max_slots]
                # Mark realized slots; pad invalid to -1 so they never match.
                cache_leaf_tokens_t = torch.where(
                    cache_valid_t.unsqueeze(1), cache_leaf_tokens_t,
                    torch.full_like(cache_leaf_tokens_t, -1),
                )
                cache_leaf_valid_t = cache_valid_t.any(dim=-1, keepdim=True)      # [B, 1]
                # Pad existing leaf_paths/leaf_tokens to match new P.
                P_old = leaf_paths.shape[-1]
                P_new = max(P_old, max_slots + 1)
                if P_old < P_new:
                    pad_p = P_new - P_old
                    last_node = leaf_paths[:, :, -1:].expand(-1, -1, pad_p)
                    leaf_paths = torch.cat([leaf_paths, last_node], dim=-1)
                    leaf_tokens_pad = torch.full(
                        (B, leaf_tokens.shape[1], pad_p), -1, dtype=torch.long, device=device,
                    )
                    leaf_tokens = torch.cat([leaf_tokens, leaf_tokens_pad], dim=-1)
                if max_slots + 1 < P_new:
                    pad_q = P_new - (max_slots + 1)
                    last_cache_node = cache_leaf_path[:, :, -1:].expand(-1, -1, pad_q)
                    cache_leaf_path = torch.cat([cache_leaf_path, last_cache_node], dim=-1)
                    cache_leaf_tokens_pad = torch.full(
                        (B, 1, pad_q), -1, dtype=torch.long, device=device,
                    )
                    cache_leaf_tokens_t = torch.cat([cache_leaf_tokens_t, cache_leaf_tokens_pad], dim=-1)
                leaf_paths = torch.cat([leaf_paths, cache_leaf_path], dim=1)
                leaf_tokens = torch.cat([leaf_tokens, cache_leaf_tokens_t], dim=1)
                leaf_valid = torch.cat([leaf_valid, cache_leaf_valid_t], dim=1)
                M = packed_ids.shape[1]
                cache_round_total += 1

        if log_phase_timings:
            torch.cuda.synchronize(); phase_t_tree_build += time.perf_counter() - _t0p
            _t0p = time.perf_counter()

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
        # Spine-first verify: skip side-branch verify when the rank-0 spine fully
        # accepts. Spine = first block_size positions of packed_ids (= rank-0
        # chain by heap construction). Side = remaining M - block_size positions.
        # Two-pass attention design uses past_kv to pass spine state into side.
        # Only activate when there are SUBSTANTIAL side branches to save
        # (≥4 side nodes) — otherwise per-call overhead beats the savings.
        spine_first_active = (
            spine_first and (M >= block_size + 4) and not chain_mode
        )
        if spine_first_active:
            P = block_size
            spine_ids = packed_ids[:, :P].contiguous()
            side_ids = packed_ids[:, P:].contiguous()
            spine_pos_abs = packed_pos_abs[:, :P].contiguous()
            side_pos_abs = packed_pos_abs[:, P:].contiguous()
            spine_attn_mask = attn_tree[:, :, :P, :cache_size + P].contiguous()

            if log_phase_timings:
                torch.cuda.synchronize(); _t0_spine = time.perf_counter()
            out_spine = target.model(
                spine_ids, position_ids=spine_pos_abs,
                past_key_values=past_kv_target, use_cache=True,
                attention_mask=spine_attn_mask, output_hidden_states=True,
            )
            spine_logits = target.lm_head(out_spine.last_hidden_state)            # [B, P, V]
            spine_ctx = extract_context_feature(
                out_spine.hidden_states, draft.target_layer_ids,
            )                                                                       # [B, P, D']
            del out_spine
            if log_phase_timings:
                torch.cuda.synchronize(); phase_t_spine_verify += time.perf_counter() - _t0_spine

            # Spine accept check: rank-0 chain. spine_ids[d+1] == argmax at parent (d).
            spine_argmax = spine_logits.argmax(dim=-1)                             # [B, P]
            spine_match = (spine_argmax[:, :-1] == spine_ids[:, 1:])               # [B, P-1]
            spine_match_cum = spine_match.cumprod(dim=-1)
            spine_accept = spine_match_cum.sum(dim=-1)                              # [B]
            spine_full_all = bool((spine_accept == (P - 1)).all().item())

            V_dim = spine_logits.shape[-1]
            D_dim = spine_ctx.shape[-1]
            logits = torch.zeros(B, M, V_dim, dtype=spine_logits.dtype, device=device)
            verify_ctx_feat = torch.zeros(B, M, D_dim, dtype=spine_ctx.dtype, device=device)
            logits[:, :P, :] = spine_logits
            verify_ctx_feat[:, :P, :] = spine_ctx

            if spine_full_all:
                # Force only rank-0 leaf valid since side logits are zero (no stage 2).
                # This guarantees best_leaf=0 and n_accepted=P-1.
                leaf_valid = torch.zeros_like(leaf_valid)
                leaf_valid[:, 0] = True
                if log_phase_timings:
                    phase_spine_full_count += 1
            if not spine_full_all:
                # Stage 2: side verify. past_kv now has prefix + spine.
                if log_phase_timings:
                    torch.cuda.synchronize(); _t0_side = time.perf_counter()
                side_attn_mask = attn_tree[:, :, P:, :].contiguous()                # [B, 1, M-P, cache_size+M]
                out_side = target.model(
                    side_ids, position_ids=side_pos_abs,
                    past_key_values=past_kv_target, use_cache=True,
                    attention_mask=side_attn_mask, output_hidden_states=True,
                )
                side_logits = target.lm_head(out_side.last_hidden_state)
                side_ctx = extract_context_feature(
                    out_side.hidden_states, draft.target_layer_ids,
                )
                logits[:, P:, :] = side_logits
                verify_ctx_feat[:, P:, :] = side_ctx
                del out_side
                if log_phase_timings:
                    torch.cuda.synchronize(); phase_t_side_verify += time.perf_counter() - _t0_side
                    phase_side_runs += 1
        else:
            out = target.model(
                packed_ids, position_ids=packed_pos_abs,
                past_key_values=past_kv_target, use_cache=True,
                attention_mask=attn_tree, output_hidden_states=True,
            )
            logits = target.lm_head(out.last_hidden_state)                            # [B, M, V]
            verify_ctx_feat = extract_context_feature(out.hidden_states, draft.target_layer_ids)
            del out
        target.config._attn_implementation = saved_attn
        if log_phase_timings:
            torch.cuda.synchronize(); phase_t_verify += time.perf_counter() - _t0p
            _t0p = time.perf_counter()

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
        if log_phase_timings:
            torch.cuda.synchronize(); phase_t_select += time.perf_counter() - _t0p
            _t0p = time.perf_counter()

        # Cache diagnostics: per-slot acceptance rate (independent of which leaf
        # won). At slot c, parent = anchor (c=0) or prev cache node (c>0). Cache
        # token was accepted if target.argmax(parent_node) == cache_tok[c]
        # AND all earlier cache slots also matched (cumprod).
        if cache_inject and cache_chain_start >= 0:
            with torch.no_grad():
                target_argmax = logits.argmax(dim=-1)                              # [B, M]
                slots_in_chain = M - cache_chain_start
                cum_match = torch.ones(B, dtype=torch.bool, device=device)
                round_had_hit_b = torch.zeros(B, dtype=torch.bool, device=device)
                for c in range(slots_in_chain):
                    node_idx = cache_chain_start + c
                    if c == 0:
                        parent = 0
                    else:
                        parent = cache_chain_start + c - 1
                    realized = packed_ids[:, node_idx]                              # [B]
                    pred_at_parent = target_argmax[:, parent]                       # [B]
                    matched_this = (pred_at_parent == realized) & node_valid[:, node_idx]
                    cum_match = cum_match & matched_this
                    cache_attempts_per_slot[c] += int(node_valid[:, node_idx].sum().item())
                    cache_hits_per_slot[c] += int(cum_match.sum().item())
                    round_had_hit_b = round_had_hit_b | cum_match
                cache_round_had_hit += int(round_had_hit_b.sum().item())

        # Path-trajectory diagnostic: at every round, capture the rank-in-q
        # along the best leaf path (depth 1..block_size-1) AND the same for
        # leaf 0 (= rank-0 chain by heap convention) for comparison.
        if log_path_trajectory:
            with torch.no_grad():
                target_argmax_pt = logits.argmax(dim=-1)                            # [B, M]
                best_leaf_cpu_pt = best_leaf.detach().cpu().tolist()
                n_acc_cpu_pt = n_accepted.detach().cpu().tolist()
                target_argmax_cpu_pt = target_argmax_pt.detach().cpu().tolist()
                packed_ids_cpu_pt = packed_ids.detach().cpu().tolist()
                _, dl_sorted_pt = draft_logits.float().sort(dim=-1, descending=True)
                dl_sorted_pt_cpu = dl_sorted_pt[:, :, :64].detach().cpu().tolist()
                leaf_paths_cpu_pt = leaf_paths.detach().cpu().tolist()
                leaf_tokens_cpu_pt = leaf_tokens.detach().cpu().tolist()

                def _walk_chain_ranks(b, leaf_idx):
                    chain = leaf_paths_cpu_pt[b][leaf_idx]
                    tokens = leaf_tokens_cpu_pt[b][leaf_idx]
                    ranks = []
                    valid_depth = 0
                    for d in range(len(tokens)):
                        tok = int(tokens[d])
                        if tok < 0:
                            break
                        if d > 0 and int(chain[d + 1]) == int(chain[d]):
                            break
                        if d >= len(dl_sorted_pt_cpu[b]):
                            break
                        sorted_idx = dl_sorted_pt_cpu[b][d]
                        try:
                            r = sorted_idx.index(tok)
                        except ValueError:
                            r = 64
                        ranks.append(r)
                        valid_depth = d + 1
                    return ranks, valid_depth

                def _accept_len_along_leaf(b, leaf_idx):
                    """How many tokens at the start of leaf_idx match target.argmax(parent)."""
                    chain = leaf_paths_cpu_pt[b][leaf_idx]
                    tokens = leaf_tokens_cpu_pt[b][leaf_idx]
                    n = 0
                    for d in range(len(tokens)):
                        parent_idx_local = int(chain[d])
                        realized = int(tokens[d])
                        if realized < 0:
                            break
                        target_pred = int(target_argmax_cpu_pt[b][parent_idx_local])
                        if target_pred == realized:
                            n += 1
                        else:
                            break
                    return n

                for b in range(B):
                    if finished[b]:
                        continue
                    n_b = int(n_acc_cpu_pt[b])
                    bl_b = int(best_leaf_cpu_pt[b])
                    best_path_ranks, best_valid_d = _walk_chain_ranks(b, bl_b)
                    rank0_path_ranks, rank0_valid_d = _walk_chain_ranks(b, 0)
                    rank0_accept = _accept_len_along_leaf(b, 0)
                    # Rejection rank along best leaf
                    if n_b + 1 < block_size:
                        chain_b = leaf_paths_cpu_pt[b][bl_b]
                        if n_b < len(chain_b):
                            parent_node = int(chain_b[n_b])
                            target_tok = int(target_argmax_cpu_pt[b][parent_node])
                            sorted_idx_at = dl_sorted_pt_cpu[b][n_b] if n_b < len(dl_sorted_pt_cpu[b]) else None
                            if sorted_idx_at is not None:
                                try:
                                    rej_rank = sorted_idx_at.index(target_tok)
                                except ValueError:
                                    rej_rank = 64
                            else:
                                rej_rank = -1
                        else:
                            rej_rank = -1
                    else:
                        rej_rank = -1
                    path_trajectories.append({
                        "best_leaf_idx": bl_b,
                        "best_n_accepted": n_b,
                        "best_path_ranks": best_path_ranks,
                        "rank0_n_accepted": rank0_accept,
                        "rank0_path_ranks": rank0_path_ranks,
                        "rejection_rank_at_best": rej_rank,
                    })

        # Full-tree dump diagnostic: capture every leaf's (path-of-ranks, accept-length).
        if log_full_tree_dump:
            with torch.no_grad():
                target_argmax_ft = logits.argmax(dim=-1)                            # [B, M]
                target_argmax_cpu_ft = target_argmax_ft.detach().cpu().tolist()
                packed_ids_cpu_ft = packed_ids.detach().cpu().tolist()
                _, dl_sorted_ft = draft_logits.float().sort(dim=-1, descending=True)
                dl_sorted_ft_cpu = dl_sorted_ft[:, :, :64].detach().cpu().tolist()
                leaf_paths_cpu_ft = leaf_paths.detach().cpu().tolist()
                leaf_tokens_cpu_ft = leaf_tokens.detach().cpu().tolist()
                leaf_valid_cpu_ft = leaf_valid.detach().cpu().tolist()
                n_acc_cpu_ft = n_accepted.detach().cpu().tolist()
                best_leaf_cpu_ft = best_leaf.detach().cpu().tolist()
                N_leaves = leaf_paths.shape[1]
                for b in range(B):
                    if finished[b]:
                        continue
                    leaves_info = []
                    for l in range(N_leaves):
                        if not bool(leaf_valid_cpu_ft[b][l]):
                            continue
                        chain = leaf_paths_cpu_ft[b][l]
                        tokens = leaf_tokens_cpu_ft[b][l]
                        ranks = []
                        accept_n = 0
                        broke_at = -1
                        valid_d = 0
                        for d in range(len(tokens)):
                            tok = int(tokens[d])
                            if tok < 0:
                                break
                            if d > 0 and int(chain[d + 1]) == int(chain[d]):
                                break
                            valid_d = d + 1
                            if d < len(dl_sorted_ft_cpu[b]):
                                sorted_idx = dl_sorted_ft_cpu[b][d]
                                try:
                                    r = sorted_idx.index(tok)
                                except ValueError:
                                    r = 64
                            else:
                                r = -1
                            ranks.append(r)
                            parent_idx_local = int(chain[d])
                            tgt_pred = int(target_argmax_cpu_ft[b][parent_idx_local])
                            if tgt_pred == tok and broke_at < 0:
                                accept_n += 1
                            elif broke_at < 0:
                                broke_at = d
                        # Deviation depth of leaf: smallest d where rank > 0
                        dev_d = -1
                        for d, r in enumerate(ranks):
                            if r > 0:
                                dev_d = d
                                break
                        leaves_info.append({
                            "leaf_idx": l, "ranks": ranks, "accept_n": accept_n,
                            "broke_at": broke_at, "deviation_depth": dev_d,
                            "len": valid_d,
                        })
                    full_tree_dumps.append({
                        "round_b": b,
                        "best_leaf_idx": int(best_leaf_cpu_ft[b]),
                        "best_n_accepted": int(n_acc_cpu_ft[b]),
                        "n_valid_leaves": len(leaves_info),
                        "leaves": leaves_info,
                    })

        # Rejection-rank diagnostic: at the rejection point of round R,
        # rank of target's argmax in draft's q_i. Walk the BEST leaf (not
        # leaf 0) since n_accepted is along best_leaf.
        if log_rejection_ranks:
            with torch.no_grad():
                target_argmax = logits.argmax(dim=-1)                              # [B, M]
                n_acc_cpu = n_accepted.detach().cpu().tolist()
                target_argmax_cpu = target_argmax.detach().cpu().tolist()
                best_leaf_cpu = best_leaf.detach().cpu().tolist()
                # Build per-elem best leaf path.
                # leaf_paths is [B, N, P]; gather along best_leaf to get [B, P].
                best_paths = leaf_paths[torch.arange(B, device=device), best_leaf, :]  # [B, P]
                best_paths_cpu = best_paths.detach().cpu().tolist()
                # Sort draft_logits descending to get rank.
                _, dl_sorted_idx = draft_logits.float().sort(dim=-1, descending=True)  # [B, q_len-1, V]
                dl_sorted_idx_cpu = dl_sorted_idx[:, :, :64].detach().cpu().tolist()
                for b in range(B):
                    if finished[b]:
                        continue
                    n_b = int(n_acc_cpu[b])
                    rej_depth = n_b + 1
                    if rej_depth >= block_size:
                        continue
                    # Parent node at depth n_b along best leaf's path.
                    chain_b = best_paths_cpu[b]
                    if n_b >= len(chain_b):
                        continue
                    parent_node = int(chain_b[n_b])
                    target_tok = int(target_argmax_cpu[b][parent_node])
                    if n_b >= len(dl_sorted_idx_cpu[b]):
                        continue
                    sorted_idx_at_d = dl_sorted_idx_cpu[b][n_b]
                    try:
                        rank = sorted_idx_at_d.index(target_tok)
                    except ValueError:
                        rank = 64
                    rejection_ranks.append((rej_depth, rank))

        # Cache capture for next round: walk rank-0 chain (leaf_paths[:, 0, :])
        # and grab target.argmax at depths PAST current accept point. By heap
        # convention leaf 0 is the rank-0 chain.
        if cache_inject:
            with torch.no_grad():
                target_argmax = logits.argmax(dim=-1)                              # [B, M]
                new_prev_cache: List[List[int]] = []
                rank0_path_cpu = leaf_paths[:, 0, :].detach().cpu().tolist()       # [B][P]
                n_acc_cpu = n_accepted.detach().cpu().tolist()
                target_argmax_cpu = target_argmax.detach().cpu().tolist()           # [B][M]
                node_valid_cpu = node_valid.detach().cpu().tolist()                 # [B][M]
                for b in range(B):
                    n_b = int(n_acc_cpu[b])
                    cache_b: List[int] = []
                    chain = rank0_path_cpu[b]
                    seen_node: set = set()
                    for d in range(n_b + 2, len(chain)):
                        node_idx = int(chain[d])
                        # Heap pads leaf paths by repeating the last node — guard.
                        if node_idx in seen_node or not bool(node_valid_cpu[b][node_idx]):
                            break
                        seen_node.add(node_idx)
                        cache_b.append(int(target_argmax_cpu[b][node_idx]))
                        if len(cache_b) >= cache_slots:
                            break
                    new_prev_cache.append(cache_b)
                prev_cache_per_elem = new_prev_cache

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

        # L-CDDT: update per-depth alive/accept counts from rank-0 chain.
        # By heap convention leaf 0 = rank-0 chain. Walk it to get its accept length.
        if learned_curve_decay:
            with torch.no_grad():
                target_argmax_lc = logits.argmax(dim=-1)                            # [B, M]
                target_argmax_cpu_lc = target_argmax_lc.detach().cpu().tolist()
                leaf_paths_cpu_lc = leaf_paths[:, 0, :].detach().cpu().tolist()     # rank-0 chain per elem
                leaf_tokens_cpu_lc = leaf_tokens[:, 0, :].detach().cpu().tolist()
                # For each non-finished elem, compute rank-0 chain's accept-length.
                step_alive = [0.0] * (block_size + 1)
                step_acc = [0.0] * (block_size + 1)
                for b in range(B):
                    if finished[b]:
                        continue
                    chain_b = leaf_paths_cpu_lc[b]
                    tokens_b = leaf_tokens_cpu_lc[b]
                    accept_n_lc = 0
                    broke = False
                    for d in range(len(tokens_b)):
                        tok = int(tokens_b[d])
                        if tok < 0:
                            break
                        if d > 0 and int(chain_b[d + 1]) == int(chain_b[d]):
                            break
                        depth = d + 1
                        if depth >= len(step_alive):
                            break
                        step_alive[depth] += 1.0
                        # Check accept at this depth
                        parent_idx_local = int(chain_b[d])
                        tgt_pred = int(target_argmax_cpu_lc[b][parent_idx_local])
                        if tgt_pred == tok and not broke:
                            accept_n_lc += 1
                            step_acc[depth] += 1.0
                        else:
                            broke = True
                # EWMA update of cumulative counts.
                for d in range(1, block_size + 1):
                    lcddt_depth_alive[d] = (
                        learned_curve_alpha * lcddt_depth_alive[d] + step_alive[d]
                    )
                    lcddt_depth_acc[d] = (
                        learned_curve_alpha * lcddt_depth_acc[d] + step_acc[d]
                    )
                    if lcddt_depth_alive[d] > 0:
                        p = lcddt_depth_acc[d] / lcddt_depth_alive[d]
                        # Power transform amplifies gradient.
                        lcddt_curve[d] = max(0.001, p ** learned_curve_power)
                    else:
                        lcddt_curve[d] = 1.0
                lcddt_step_count += 1

        # W-CDDT: update EWMA of full-block accept rate (n_accepted == block_size-1).
        if workload_adaptive_decay:
            full_block_count = 0
            total_count = 0
            for b in range(B):
                if finished[b]:
                    continue
                # Last n_plus_1 entry; n_accepted = n_plus_1 - 1
                if accepted_lengths_per_elem[b]:
                    n_b = accepted_lengths_per_elem[b][-1] - 1
                    total_count += 1
                    if n_b == block_size - 1:
                        full_block_count += 1
            if total_count > 0:
                step_full_block_rate = full_block_count / total_count
                wcddt_full_block_ewma = (
                    workload_ewma_alpha * wcddt_full_block_ewma
                    + (1 - workload_ewma_alpha) * step_full_block_rate
                )

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
        if log_phase_timings:
            torch.cuda.synchronize(); phase_t_trim += time.perf_counter() - _t0p
            phase_step_count += 1

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
        cache_hits_per_slot=cache_hits_per_slot,
        cache_attempts_per_slot=cache_attempts_per_slot,
        cache_round_total=cache_round_total,
        cache_round_had_hit=cache_round_had_hit,
        rejection_ranks=rejection_ranks,
        path_trajectories=path_trajectories,
        full_tree_dumps=full_tree_dumps,
        lcddt_final_curve=list(lcddt_curve),       # for sanity-check inspection
        lcddt_step_count=lcddt_step_count,
        phase_timings_ms=dict(
            draft=round(phase_t_draft * 1000, 2),
            tree_build=round(phase_t_tree_build * 1000, 2),
            verify=round(phase_t_verify * 1000, 2),
            spine_verify=round(phase_t_spine_verify * 1000, 2),
            side_verify=round(phase_t_side_verify * 1000, 2),
            spine_full_count=phase_spine_full_count,
            side_runs=phase_side_runs,
            select=round(phase_t_select * 1000, 2),
            trim=round(phase_t_trim * 1000, 2),
            steps=phase_step_count,
        ),
    )
