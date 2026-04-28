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

from types import SimpleNamespace
from typing import List

import torch
from transformers import DynamicCache

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

    # Trim target cache to remove pad cols.
    cache_size = _trim_pad_after_prefill(past_kv_target, prefix_lens, S_max, device)
    cur_prefix_lens = prefix_lens.clone()
    cur_anchor = first_tok                                                       # [B, 1]

    # Per-element draft KV caches. Each draft call is per-element (looped) so
    # it sees its own un-padded target_hidden. Target verify stays batched.
    past_kv_draft_list = [DynamicCache() for _ in range(B)]
    # Per-element target_hidden = full prefill hidden states sliced to that
    # element's real prompt length.
    target_hidden_per_elem = [
        target_hidden_full[b:b+1, :int(prefix_lens[b].item()), :].clone()
        for b in range(B)
    ]
    target_hidden_pos_start = [0 for _ in range(B)]   # K_ctx RoPE start
    del target_hidden_full

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while (not all(finished)) and max(d.shape[1] for d in decoded_list) < max_new_tokens:
        step += 1

        # ------------------------------------------------------------
        # Draft step: per-element loop. Each element runs an INDEPENDENT draft
        # forward with its own (un-padded) target_hidden + own KV cache. This
        # exactly matches the B=1 single-stream semantics for every element.
        # ------------------------------------------------------------
        block = torch.full((B, block_size), mask_token_id, dtype=torch.long, device=device)
        block[:, 0] = cur_anchor[:, 0]
        noise_emb_full = target.model.embed_tokens(block)                       # [B, block, H]
        block_pos_full = (
            cur_prefix_lens.unsqueeze(1)
            + torch.arange(block_size, device=device).unsqueeze(0)
        )                                                                        # [B, block_size]

        per_elem_logits = []
        for b in range(B):
            cpl_b = int(cur_prefix_lens[b].item())
            th_b = target_hidden_per_elem[b]                                     # [1, ctx_b, D']
            ctx_b = th_b.shape[1]
            ctx_start_b = target_hidden_pos_start[b]
            # Position IDs of length ctx_b + block_size:
            #   first ctx_b entries: [ctx_start_b .. ctx_start_b + ctx_b - 1]
            #     (real positions of cross-context tokens).
            #   last block_size entries: [cpl_b .. cpl_b + block_size - 1].
            pos_b = torch.cat([
                torch.arange(ctx_start_b, ctx_start_b + ctx_b, device=device, dtype=torch.long),
                torch.arange(cpl_b, cpl_b + block_size, device=device, dtype=torch.long),
            ]).unsqueeze(0)                                                      # [1, ctx_b + block_size]
            noise_b = noise_emb_full[b:b+1]                                      # [1, block_size, H]
            draft_hidden_b = draft(
                target_hidden=th_b, noise_embedding=noise_b,
                position_ids=pos_b, attention_mask=None,
                past_key_values=past_kv_draft_list[b], use_cache=True, is_causal=False,
            )                                                                    # [1, block_size, H]

            # Crop element b's cache to pre-step real prefix (= cpl_b). Mirrors
            # the original `past_kv_draft.crop(start)` call.
            for layer in past_kv_draft_list[b].layers:
                cur_len = layer.keys.shape[2]
                if cur_len > cpl_b:
                    layer.keys = layer.keys[:, :, :cpl_b, :].contiguous()
                    layer.values = layer.values[:, :, :cpl_b, :].contiguous()

            per_elem_logits.append(target.lm_head(draft_hidden_b[:, 1:, :]))     # [1, block-1, V]

        draft_logits = torch.cat(per_elem_logits, dim=0)                         # [B, block-1, V]

        # ------------------------------------------------------------
        # Tree build (per-element loop, padded outputs).
        # ------------------------------------------------------------
        packed_ids, packed_pos_rel, parent_idx, node_valid, leaf_paths, leaf_tokens, leaf_valid = \
            build_node_budget_tree_batched(
                draft_logits=draft_logits, anchor_token_ids=cur_anchor[:, 0],
                max_tree_size=max_tree_size, expand_k=expand_k,
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

        # Update per-element target_hidden to the accepted-path hidden states
        # (no cross-batch padding — each element has its own correctly-sized
        # cross-context for the next step's draft call).
        for b in range(B):
            n1 = int(n_plus_1[b].item())
            slice_b = verify_ctx_feat[b, accepted_paths[b, :n1], :].unsqueeze(0)    # [1, n1, D']
            target_hidden_per_elem[b] = slice_b
            # The K_ctx RoPE positions for the next step start at the position
            # of the FIRST accepted-path token, which is the anchor at real
            # position cur_prefix_lens_pre_step[b] (= cur_prefix_lens[b] before trim).
            # We set this AFTER the trim updates cur_prefix_lens, so we capture
            # cur_prefix_lens BEFORE trim using cur_prefix_lens[b].item() now.
            target_hidden_pos_start[b] = int(cur_prefix_lens[b].item())

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
        total_decode_time=total,
    )
