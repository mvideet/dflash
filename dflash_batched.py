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
from typing import List, Tuple

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

        # Pick M for this step. Online: argmax goodput from cost+tau models,
        # using prev step's bonus-token confidence as a forward chain bound.
        if online_mts:
            eff_M = _pick_m_online(
                B, ewma_tau, ewma_M_ref, online_mts_candidates_t,
                anchor_conf=prev_anchor_conf, block_size=block_size,
            )
        else:
            eff_M = max_tree_size
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

        # ------------------------------------------------------------
        # Tree build (per-element loop, padded outputs).
        # ------------------------------------------------------------
        packed_ids, packed_pos_rel, parent_idx, node_valid, leaf_paths, leaf_tokens, leaf_valid = \
            build_node_budget_tree_batched(
                draft_logits=draft_logits, anchor_token_ids=cur_anchor[:, 0],
                max_tree_size=eff_M, expand_k=expand_k,
            )
        M = packed_ids.shape[1]
        tree_node_counts.append(int(node_valid.sum(dim=-1).float().mean().item()))

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

        # Accept-path select.
        best_leaf, n_accepted = select_best_dynamic_leaf_batched(
            logits=logits, leaf_paths=leaf_paths, leaf_tokens=leaf_tokens,
            leaf_valid=leaf_valid, temperature=temperature,
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

        # Forward-looking signal for next step's M selection: target's top-1
        # probability at the bonus token (= NEXT step's anchor). High p means
        # the new block is likely an argmax run; low p means a hard step
        # ahead. Only relevant when online_mts is on.
        if online_mts:
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
