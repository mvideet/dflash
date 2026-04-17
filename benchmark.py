import argparse
import json
import random
from collections import defaultdict
from itertools import chain
from types import SimpleNamespace
from loguru import logger
import numpy as np
import torch
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from model import (
    DFlashDraftModel,
    sample,
    load_and_process_dataset,
    extract_context_feature,
    extract_target_hidden_from_tree,
    cuda_time,
    trim_target_kv_cache,
)
from model.dflash_tree import (
    build_dynamic_tree_v2,
    build_prefixaware_tree,
    build_efficiency_tree,
    build_node_budget_tree,
    build_chained_tree,
    create_tree_attention_mask_dynamic,
    select_best_dynamic_leaf,
)
from model.freq_vocab import load_freq_mapping, get_reduced_lm_head, compute_reduced_draft_logits
import distributed as dist

TREE_BUILDERS = {
    2: build_dynamic_tree_v2,
    4: build_prefixaware_tree,
    6: build_efficiency_tree,
    7: build_node_budget_tree,
}


def _get_draft_logits(
    draft_hidden: torch.Tensor,
    target,
    freq_used_tokens: list[int] | None,
    freq_reduced_weight: torch.Tensor | None,
    freq_reduced_bias: torch.Tensor | None,
) -> torch.Tensor:
    if freq_used_tokens is not None and freq_reduced_weight is not None:
        return compute_reduced_draft_logits(draft_hidden, freq_reduced_weight, freq_reduced_bias)
    return target.lm_head(draft_hidden)


def _record_profile(profile_times: dict, name: str, t0: float | None, do_profile: bool) -> float | None:
    if do_profile and t0 is not None:
        profile_times[name] = profile_times.get(name, 0) + (cuda_time() - t0)
        return cuda_time()
    return t0


@torch.inference_mode()
def dflash_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    tree_version: int = 4,
    max_tree_size: int = 32,
    expand_k: int = 3,
    profile: bool = False,
    freq_used_tokens: list[int] | None = None,
    freq_reduced_weight: torch.Tensor | None = None,
    freq_reduced_bias: torch.Tensor | None = None,
    alpha: float = 0.0,
    score_alpha: float = 1.0,
    score_beta: float = 0.0,
    chain_depth: int = 0,
    adaptive_block: bool = False,
    adaptive_block_ewma_decay: float = 0.8,
    adaptive_block_min_tree_size: int = 12,
    adaptive_block_min_expand_k: int = 2,
    adaptive_block_max_expand_k: int = 5,
    collect_calibration: bool = False,
    ctr: bool = False,
    calibrate: bool = False,
    calibrate_warmup: float = 50.0,
) -> SimpleNamespace:
    """
    Generate tokens using DFlash speculative decoding.

    When block_size > 1, builds a candidate tree from draft logits and verifies
    it in a single target forward pass.  tree_version selects the builder:
    2 (EAGLE-2 expand+rerank) or 4 (prefix-aware greedy with (1-1/e) guarantee).

    Adaptive EWMA (--adaptive-block):
      Tracks acceptance rate via EWMA and adapts both max_tree_size and expand_k
      per step.  Easy steps get wide search (expand_k up to max_expand_k) and
      full trees.  Hard steps get narrow search (expand_k down to min_expand_k)
      and smaller trees.
    """
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    tail_margin = 2 * block_size if chain_depth > 0 else block_size
    output_ids = torch.full(
        (1, max_length + tail_margin),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()
    profile_times = {}
    _pt = cuda_time() if profile else None

    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True if block_size > 1 else False,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens + 1] = sample(output.logits, temperature)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    time_to_first_token = cuda_time() - prefill_start
    _pt = _record_profile(profile_times, "prefill_target", _pt, profile)

    decode_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    tree_node_counts = []
    calibration_data = []
    draft_prefill = True

    _ab_ewma_rate = 1.0
    _ab_eff_tree_size = max_tree_size
    _ab_eff_expand_k = expand_k

    # Q4: online target-logit calibration state (per-sequence).
    # alpha_count_accept[d, r] = pseudo-count of times target's argmax at a
    # parent node at depth d equalled draft's rank-r token (one per observed
    # parent). alpha_count_seen[d] normalises to a probability per depth.
    # Laplace-smoothed init (+1 per cell, +K per row) prevents log(0).
    calibration_seq_len = block_size - 1
    if calibrate and block_size > 1:
        alpha_count_accept = torch.ones(
            calibration_seq_len, expand_k, dtype=torch.float32, device=model.device,
        )
        alpha_count_seen = torch.full(
            (calibration_seq_len,), float(expand_k),
            dtype=torch.float32, device=model.device,
        )
    else:
        alpha_count_accept = None
        alpha_count_seen = None

    while start < max_length:
        eff_bs = block_size
        if adaptive_block and block_size > 1:
            _ab_eff_tree_size = adaptive_block_min_tree_size + round(
                (max_tree_size - adaptive_block_min_tree_size) * _ab_ewma_rate
            )
            _ab_eff_expand_k = adaptive_block_min_expand_k + round(
                (adaptive_block_max_expand_k - adaptive_block_min_expand_k) * _ab_ewma_rate
            )

        block_output_ids = output_ids[:, start:start + eff_bs].clone()
        _pt = cuda_time() if profile else None

        if eff_bs > 1:
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_hidden = model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length():start + eff_bs],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, -eff_bs + 1:, :]
            _pt = _record_profile(profile_times, "draft_model", _pt, profile)
            draft_logits = _get_draft_logits(
                draft_hidden, target, freq_used_tokens, freq_reduced_weight, freq_reduced_bias
            )
            _pt = _record_profile(profile_times, "draft_lm_head", _pt, profile)

            # --- Q2: chained speculation — 2nd draft forward BEFORE crop,
            # so it benefits from block_1's cached noise KV.
            draft_logits_2 = None
            if chain_depth > 0 and tree_version == 7:
                argmax_end_block_1 = draft_logits[0, -1].argmax().item()
                block_output_2 = torch.full(
                    (1, eff_bs), mask_token_id,
                    device=draft_logits.device, dtype=torch.long,
                )
                block_output_2[0, 0] = argmax_end_block_1
                noise_embedding_2 = target.model.embed_tokens(block_output_2)
                pos_ids_2 = position_ids[:, start + eff_bs:start + 2 * eff_bs]
                draft_hidden_2 = model(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding_2,
                    position_ids=pos_ids_2,
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                    is_causal=False,
                )[:, -eff_bs + 1:, :]
                draft_logits_2 = _get_draft_logits(
                    draft_hidden_2, target, freq_used_tokens,
                    freq_reduced_weight, freq_reduced_bias,
                )
                _pt = _record_profile(profile_times, "draft_model_2", _pt, profile)

            past_key_values_draft.crop(start)
            _pt = _record_profile(profile_times, "draft_crop", _pt, profile)

            if draft_prefill:
                draft_prefill = False
                decode_start = cuda_time()

            tree_logits = draft_logits
            tree_bs = tree_logits.shape[1] + 1

            eff_tree_size = _ab_eff_tree_size if adaptive_block else max_tree_size
            eff_expand_k = _ab_eff_expand_k if adaptive_block else expand_k

            if chain_depth > 0 and tree_version == 7 and draft_logits_2 is not None:
                (
                    packed_ids,
                    packed_pos_relative,
                    parent_idx,
                    leaf_paths,
                    leaf_tokens,
                ) = build_chained_tree(
                    draft_logits=tree_logits,
                    draft_logits_2=draft_logits_2,
                    anchor_token_ids=block_output_ids[:, :1],
                    max_tree_size=eff_tree_size,
                    expand_k=eff_expand_k,
                    score_alpha=score_alpha,
                    score_beta=score_beta,
                    chain_depth=chain_depth,
                    used_tokens=freq_used_tokens,
                )
            else:
                builder_kwargs = dict(
                    draft_logits=tree_logits,
                    anchor_token_ids=block_output_ids[:, :1],
                    max_tree_size=eff_tree_size,
                    expand_k=eff_expand_k,
                    used_tokens=freq_used_tokens,
                )
                if tree_version == 6:
                    builder_kwargs['alpha'] = alpha
                if tree_version == 7:
                    builder_kwargs['score_alpha'] = score_alpha
                    builder_kwargs['score_beta'] = score_beta
                    if calibrate and alpha_count_seen is not None:
                        # Blend draft marginal with empirical target-acceptance rate.
                        # Confidence ramps with observations: w = n / (n + warmup).
                        alpha_hat = alpha_count_accept / alpha_count_seen.unsqueeze(-1)
                        w = alpha_count_seen / (alpha_count_seen + calibrate_warmup)
                        w = w.unsqueeze(-1)
                        # Blend in probability space; convert back to log.
                        draft_p = torch.softmax(tree_logits[0], dim=-1)
                        draft_topk_vals = torch.topk(
                            draft_p, k=eff_expand_k, dim=-1,
                        ).values  # [seq_len, K]
                        blended = (1 - w) * draft_topk_vals + w * alpha_hat
                        builder_kwargs['rank_logprobs'] = blended.clamp(min=1e-9).log()
                (
                    packed_ids,
                    packed_pos_relative,
                    parent_idx,
                    leaf_paths,
                    leaf_tokens,
                ) = TREE_BUILDERS[tree_version](**builder_kwargs)
            tree_bs = leaf_tokens.shape[1] + 1
            _pt = _record_profile(profile_times, "tree_build", _pt, profile)
            tree_node_counts.append(int(packed_ids.shape[1]))

            # --- CTR: Conditional Tree Refinement ---
            # Second draft pass with tree attention to get conditional logits.
            # Marginal logits treat each position independently; conditional
            # logits at each tree node are conditioned on that node's ancestors.
            if ctr:
                ctx_len = target_hidden.shape[1]
                ctr_noise = target.model.embed_tokens(packed_ids)  # [1, L, H]
                ctr_pos = (packed_pos_relative + start).clone()    # [1, L]
                ctr_mask = create_tree_attention_mask_dynamic(
                    packed_pos_relative, parent_idx, prefix_len=ctx_len,
                )  # [1, 1, L, ctx_len + L]

                saved_draft_attn = model.config._attn_implementation
                model.config._attn_implementation = "sdpa"
                ctr_hidden = model(
                    target_hidden=target_hidden,
                    noise_embedding=ctr_noise,
                    position_ids=ctr_pos,
                    attention_mask=ctr_mask,
                    use_cache=False,
                    is_causal=False,
                )  # [1, L, H]
                model.config._attn_implementation = saved_draft_attn

                ctr_logits = _get_draft_logits(
                    ctr_hidden, target, freq_used_tokens,
                    freq_reduced_weight, freq_reduced_bias,
                )  # [1, L, V]

                # Refine tree: at each non-root node, replace the marginal
                # token with the node's own CONDITIONAL prediction.
                # DFlash uses bidirectional attention, so lm_head at node i
                # predicts the token AT position i (not the next position).
                # The conditional logits account for the ancestor path via
                # tree attention.
                L = packed_ids.shape[1]
                for ni in range(1, L):
                    packed_ids[0, ni] = ctr_logits[0, ni, :].argmax().item()

                # Rebuild leaf_tokens from updated packed_ids using leaf_paths
                N, path_len = leaf_paths.shape
                depth = path_len - 1  # leaf_tokens has one fewer col than leaf_paths
                new_leaf_tokens = torch.full_like(leaf_tokens, -1)
                for li in range(N):
                    for di in range(depth):
                        node_idx = leaf_paths[li, di + 1].item()
                        if di > 0 and node_idx == leaf_paths[li, di].item():
                            break  # padding: repeated last node
                        new_leaf_tokens[li, di] = packed_ids[0, node_idx].item()
                leaf_tokens = new_leaf_tokens

                _pt = _record_profile(profile_times, "ctr_refine", _pt, profile)

            packed_pos = packed_pos_relative + start
            prefix_len = past_key_values_target.get_seq_length()
            attn_mask = create_tree_attention_mask_dynamic(
                packed_pos_relative, parent_idx, prefix_len
            )
            _pt = _record_profile(profile_times, "tree_attn_mask", _pt, profile)

            saved_attn_impl = target.config._attn_implementation
            target.config._attn_implementation = "sdpa"
            mod_out = target.model(
                packed_ids,
                position_ids=packed_pos,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
                attention_mask=attn_mask,
                return_dict=True,
            )
            _pt = _record_profile(profile_times, "target_backbone", _pt, profile)
            logits = target.lm_head(mod_out.last_hidden_state)
            _pt = _record_profile(profile_times, "target_lm_head", _pt, profile)
            target.config._attn_implementation = saved_attn_impl
            B, Lext, V = logits.shape

            # Q4: update online calibration from target's logits at every
            # non-leaf tree node. For each parent p at depth d < seq_len:
            # target's full distribution P_target(· | path_to_p) is harvested,
            # and we accumulate target_prob of draft's rank-r token per depth.
            # Using continuous probabilities (not argmax match) gives a much
            # less noisy signal, especially early in a sequence.
            if calibrate and alpha_count_seen is not None and tree_version == 7:
                with torch.no_grad():
                    # Draft's top-K token IDs per depth, shape [seq_len, K]
                    draft_topk_idx = torch.topk(
                        tree_logits[0], k=eff_expand_k, dim=-1,
                    ).indices  # [seq_len, K]
                    if freq_used_tokens is not None:
                        used_t = torch.tensor(
                            freq_used_tokens, device=draft_topk_idx.device,
                            dtype=torch.long,
                        )
                        draft_topk_tokens = used_t[draft_topk_idx]
                    else:
                        draft_topk_tokens = draft_topk_idx                  # [seq_len, K]
                    node_depths = packed_pos_relative[0]                    # [L]
                    parent_mask = (node_depths >= 0) & (node_depths < calibration_seq_len)
                    parent_node_idxs = torch.where(parent_mask)[0]
                    if parent_node_idxs.numel() > 0:
                        pdepths = node_depths[parent_node_idxs]             # [P]
                        parent_logits = logits[0, parent_node_idxs, :]      # [P, V]
                        parent_probs = torch.softmax(parent_logits.float(), dim=-1)
                        # For each parent, extract P_target at draft's top-K
                        # tokens for the CHILD depth (pdepth).
                        rel_topk = draft_topk_tokens[pdepths]               # [P, K]
                        target_topk_probs = parent_probs.gather(1, rel_topk)  # [P, K]
                        # Scatter-add per pdepth. Accumulate expected target
                        # probability of each rank-r at each depth.
                        alpha_count_accept.index_add_(
                            0, pdepths, target_topk_probs.to(alpha_count_accept.dtype),
                        )
                        alpha_count_seen.index_add_(
                            0, pdepths,
                            torch.ones(pdepths.shape[0], device=alpha_count_seen.device),
                        )

            best_leaf, n = select_best_dynamic_leaf(
                logits=logits,
                leaf_paths=leaf_paths,
                leaf_tokens=leaf_tokens,
                temperature=temperature,
            )
            best_path = leaf_paths[best_leaf]
            best_tokens = leaf_tokens[best_leaf]
            realized = torch.empty((1, tree_bs), device=packed_ids.device, dtype=torch.long)
            realized[:, 0] = packed_ids[:, 0]
            realized[:, 1:] = best_tokens.unsqueeze(0)
            path_idx = best_path.unsqueeze(0)

            _pt = _record_profile(profile_times, "tree_verify_select", _pt, profile)

            if collect_calibration:
                draft_probs = draft_logits[0].softmax(-1)
                draft_top1 = draft_probs.max(-1).values
                for d in range(draft_top1.shape[0]):
                    calibration_data.append((d, float(draft_top1[d].item()), 1 if d < n else 0))

            output_ids[:, start:start + n + 1] = realized[:, :n + 1]

            last_node = path_idx[:, n]
            last_logits = logits.gather(1, last_node.view(B, 1, 1).expand(B, 1, V)).squeeze(1)
            if temperature < 1e-5:
                next_tok = last_logits.argmax(dim=-1)
            else:
                next_tok = sample(last_logits.unsqueeze(1), temperature).squeeze(1)
            output_ids[:, start + n + 1] = next_tok

            accepted_path = path_idx[:, :n + 1]
            trim_target_kv_cache(
                past_key_values_target, prefix_len, accepted_path, packed_ids.device
            )
            _pt = _record_profile(profile_times, "trim_kv_cache", _pt, profile)

            tree_hidden = extract_context_feature(mod_out.hidden_states, model.target_layer_ids)
            target_hidden = extract_target_hidden_from_tree(tree_hidden, path_idx[:, :n + 1])
            _pt = _record_profile(profile_times, "extract_hidden", _pt, profile)

            acceptance_lengths.append(n + 1)
            start += n + 1

            if adaptive_block and eff_bs > 1:
                rate = n / max(eff_bs - 1, 1)
                _ab_ewma_rate = adaptive_block_ewma_decay * _ab_ewma_rate + (1 - adaptive_block_ewma_decay) * rate

        else:
            # Baseline (block_size=1): single-token autoregressive
            mod_out = target.model(
                block_output_ids,
                position_ids=position_ids[:, start:start + eff_bs],
                past_key_values=past_key_values_target,
                use_cache=True,
                return_dict=True,
            )
            _pt = _record_profile(profile_times, "target_backbone", _pt, profile)
            logits_out = target.lm_head(mod_out.last_hidden_state)
            _pt = _record_profile(profile_times, "target_lm_head", _pt, profile)

            posterior = sample(logits_out, temperature)
            output_ids[:, start + 1] = posterior[:, 0]

            acceptance_lengths.append(1)
            tree_node_counts.append(1)
            start += 1
            past_key_values_target.crop(start)

        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids is not None:
        stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(
            output_ids[0][num_input_tokens:], stop_token_ids
        ).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, :num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / num_output_tokens

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
        tree_node_counts=tree_node_counts,
        calibration_data=calibration_data,
        profile_times=profile_times if profile else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tree-size", type=int, default=32)
    parser.add_argument("--tree-version", type=int, default=4, choices=[2, 4, 6, 7],
                        help="Tree building: 2=EAGLE-2, 4=prefix-aware greedy, 6=efficiency-optimal density greedy, 7=node-budget top-B")
    parser.add_argument("--expand-k", type=int, default=7,
                        help="Per-node expansion width (default: 7, empirically optimal for v4)")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="v6 self-sizing: fixed per-step cost in trie-node units. "
                             "0=disabled (use max-tree-size cap). Typical: 5-50.")
    parser.add_argument("--score-alpha", type=float, default=1.0,
                        help="v7 power-scaled scoring: depth discount α∈(0,1]. "
                             "1.0 = plain DDTree; <1 down-weights deep positions.")
    parser.add_argument("--score-beta", type=float, default=0.0,
                        help="v7 deviation penalty: β≥0. "
                             "0 = plain DDTree; >0 penalizes rank>0 tokens per prefix.")
    parser.add_argument("--chain-depth", type=int, default=0,
                        help="Q2: chained speculation linear extension depth (v7 only). "
                             "0 = disabled. 15 = append full block_2 argmax chain.")
    parser.add_argument("--calibrate", action="store_true", default=False,
                        help="Q4: online target-logit calibration — harvest target "
                             "logits at every tree node to learn empirical per-depth "
                             "rank-acceptance, blend into v7 scoring next step.")
    parser.add_argument("--calibrate-warmup", type=float, default=50.0,
                        help="Number of per-depth observations needed before "
                             "calibration fully replaces draft's marginal (w=0.5 at n=warmup).")
    parser.add_argument("--profile", action="store_true",
                        help="Print CUDA-synced per-step timing breakdown")
    parser.add_argument("--freq-path", type=str, default=None,
                        help="Path to freq_{r}.pt for FR-Spec reduced vocab")
    parser.add_argument("--adaptive-block", action="store_true", default=False,
                        help="Enable EWMA adaptive tree sizing + expand_k")
    parser.add_argument("--adaptive-block-ewma-decay", type=float, default=0.8,
                        help="EWMA decay factor (default: 0.8)")
    parser.add_argument("--adaptive-block-min-tree-size", type=int, default=12,
                        help="Minimum tree size on hard steps (default: 12)")
    parser.add_argument("--adaptive-block-min-expand-k", type=int, default=2,
                        help="Minimum expand_k on hard steps (default: 2)")
    parser.add_argument("--adaptive-block-max-expand-k", type=int, default=5,
                        help="Maximum expand_k on easy steps (default: 5)")
    parser.add_argument("--collect-calibration", action="store_true", default=False,
                        help="Collect per-position (depth, confidence, accepted) for calibration analysis")
    parser.add_argument("--ctr", action="store_true", default=False,
                        help="Enable Conditional Tree Refinement: second draft pass with tree attention")
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")

    def has_flash_attn():
        try:
            import flash_attn
            return True
        except ImportError:
            logger.warning("flash_attn is not installed. Falling back to torch.sdpa. The speedup will be lower.")
            return False

    installed_flash_attn = has_flash_attn()

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=torch.bfloat16,
    ).to(device).eval()

    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=torch.bfloat16,
    ).to(device).eval()

    block_size = args.block_size if args.block_size is not None else draft_model.block_size

    freq_used_tokens = None
    freq_reduced_weight = None
    freq_reduced_bias = None
    if args.freq_path is not None:
        _, freq_used_tokens = load_freq_mapping(args.freq_path)
        freq_reduced_weight, freq_reduced_bias = get_reduced_lm_head(
            target.lm_head, freq_used_tokens, device
        )
        logger.info(f"Loaded freq vocab from {args.freq_path} ({len(freq_used_tokens)} tokens, reduced lm_head)")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_and_process_dataset(args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    responses = []
    indices = range(dist.rank(), len(dataset), dist.size())
    for idx in tqdm(indices, disable=not dist.is_main()):
        instance = dataset[idx]
        messages = []
        for user_content in instance["turns"]:
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            response = {}
            for bs in [1, block_size]:
                response[bs] = dflash_generate(
                    model=draft_model,
                    target=target,
                    input_ids=input_ids,
                    mask_token_id=draft_model.mask_token_id,
                    max_new_tokens=args.max_new_tokens,
                    block_size=bs,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                    tree_version=args.tree_version,
                    max_tree_size=args.max_tree_size,
                    expand_k=args.expand_k,
                    alpha=args.alpha,
                    score_alpha=args.score_alpha,
                    score_beta=args.score_beta,
                    chain_depth=args.chain_depth if bs > 1 else 0,
                    calibrate=args.calibrate if bs > 1 else False,
                    calibrate_warmup=args.calibrate_warmup,
                    profile=args.profile,
                    freq_used_tokens=freq_used_tokens,
                    freq_reduced_weight=freq_reduced_weight,
                    freq_reduced_bias=freq_reduced_bias,
                    adaptive_block=args.adaptive_block if bs > 1 else False,
                    adaptive_block_ewma_decay=args.adaptive_block_ewma_decay,
                    adaptive_block_min_tree_size=args.adaptive_block_min_tree_size,
                    adaptive_block_min_expand_k=args.adaptive_block_min_expand_k,
                    adaptive_block_max_expand_k=args.adaptive_block_max_expand_k,
                    collect_calibration=args.collect_calibration if bs > 1 else False,
                    ctr=args.ctr if bs > 1 else False,
                )
            
            spec_response = response[block_size]
            avg_acc = sum(spec_response.acceptance_lengths) / len(spec_response.acceptance_lengths)
            if spec_response.tree_node_counts:
                avg_nodes = sum(spec_response.tree_node_counts) / len(spec_response.tree_node_counts)
                print(f"seq avg acceptance: {avg_acc:.2f}  avg nodes: {avg_nodes:.1f}")
            else:
                print(f"seq avg acceptance: {avg_acc:.2f}")
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))

    t1 = np.mean([r[1].time_per_output_token for r in responses])
    tb = np.mean([r[block_size].time_per_output_token for r in responses])
    print(f"Decoding speedup: {t1 / tb:.2f}")

    tau = np.mean([np.mean(r[block_size].acceptance_lengths) for r in responses])
    print(f"Average Acceptance length: {tau:.2f}")

    all_node_counts = [r[block_size].tree_node_counts for r in responses if r[block_size].tree_node_counts]
    if all_node_counts:
        avg_nodes = np.mean([np.mean(nc) for nc in all_node_counts])
        print(f"Average tree node count: {avg_nodes:.2f}")

    acceptance_lengths = list(chain(*[r[block_size].acceptance_lengths for r in responses]))
    histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")

    if args.profile:
        agg = defaultdict(float)
        total_steps = 0
        for r in responses:
            pt = r[block_size].profile_times
            if pt:
                for k, v in pt.items():
                    agg[k] += v
            al = r[block_size].acceptance_lengths
            total_steps += len(al) if al else 0
        total = sum(agg.values())
        if total > 0:
            print("\n--- Profile (CUDA-synced): aggregate over all samples ---")
            print(f"  decode_steps (sum of acceptance rounds): {total_steps}")
            decode_total = total - agg.get("prefill_target", 0.0)
            for k in sorted(agg.keys()):
                pct = 100 * agg[k] / total
                if k == "prefill_target":
                    print(f"  {k}: {agg[k]:.3f}s total ({pct:.1f}%)  |  (once per seq; not averaged /step)")
                else:
                    per_step_ms = (agg[k] / total_steps * 1000.0) if total_steps > 0 else 0.0
                    pct_dec = 100 * agg[k] / decode_total if decode_total > 0 else 0.0
                    print(f"  {k}: {agg[k]:.3f}s total ({pct_dec:.1f}% of decode)  |  avg {per_step_ms:.3f} ms/step")
            print(f"  ALL_TIMERS_SUM: {total:.3f}s")

    if args.collect_calibration:
        all_cal = list(chain(*[r[block_size].calibration_data for r in responses
                               if r[block_size].calibration_data]))
        if all_cal:
            cal_path = f"logs/calibration_{args.dataset}.json"
            with open(cal_path, "w") as f:
                json.dump(
                    {"columns": ["depth", "draft_top1_prob", "accepted"], "data": all_cal},
                    f,
                )
            print(f"\nCalibration data: {len(all_cal)} points -> {cal_path}")

if __name__ == "__main__":
    main()
