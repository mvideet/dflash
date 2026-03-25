import argparse
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
    sample_first,
    get_position_ids,
    create_tree_attention_mask,
    compute_path_packed_indices,
    build_dynamic_tree,
    build_dynamic_tree_v2,
    build_bestfirst_tree,
    create_tree_attention_mask_dynamic,
    select_best_dynamic_leaf,
    optimal_tree_depth,
)
from model.freq_vocab import load_freq_mapping, get_reduced_lm_head, compute_reduced_draft_logits
import distributed as dist


def _get_draft_logits(
    draft_hidden: torch.Tensor,
    target,
    freq_used_tokens: list[int] | None,
    freq_reduced_weight: torch.Tensor | None,
    freq_reduced_bias: torch.Tensor | None,
) -> torch.Tensor:
    """
    Draft logits for candidate selection only.
    - With freq_path: REDUCED (sliced target.lm_head rows for top-r tokens) — saves compute.
    - Without freq_path: FULL target.lm_head.
    Note: The TARGET model always uses its full lm_head for verification (prefill, tree verify,
    sequential verify). Only the draft candidate selection uses reduced when freq_path is set.
    """
    if freq_used_tokens is not None and freq_reduced_weight is not None:
        return compute_reduced_draft_logits(draft_hidden, freq_reduced_weight, freq_reduced_bias)
    return target.lm_head(draft_hidden)


def _record_profile(profile_times: dict, name: str, t0: float | None, do_profile: bool) -> float | None:
    """Record elapsed time and return new t0 for next block. No-op when not profiling."""
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
    chain_attention: bool = False,
    top_k: int = 3,
    dynamic_branching: bool = False,
    tree_version: int = 1,
    theta_uni: float = 0.9,
    theta_bi: float = 0.3,
    theta_tri: float = 0.1,
    max_tree_size: int = 8,
    expand_k: int = 3,
    adaptive_depth: bool = False,
    adaptive_depth_threshold: float = 0.0,
    profile: bool = False,
    freq_used_tokens: list[int] | None = None,
    freq_reduced_weight: torch.Tensor | None = None,
    freq_reduced_bias: torch.Tensor | None = None,
) -> SimpleNamespace:
    """
    Generate tokens using DFlash speculative decoding.

    When chain_attention or dynamic_branching is True (and block_size > 1), builds
    a candidate tree from draft logits and verifies it in a single target forward
    pass.  tree_version selects the builder: 1 (threshold+cap), 2 (EAGLE-2), or
    3 (best-first).  Otherwise falls back to sequential draft verification.
    """
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()
    profile_times = {}  # name -> total seconds (only populated when profile=True)
    _pt = cuda_time() if profile else None

    # Prefill
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

    # Decode
    decode_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    draft_prefill = True

    while start < max_length:
        block_output_ids = output_ids[:, start:start + block_size].clone()
        _pt = cuda_time() if profile else None

        if block_size > 1 and chain_attention:
            # Tree path: draft -> build tree -> single target forward -> verify
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_hidden = model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length():start + block_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, -block_size + 1:, :]
            _pt = _record_profile(profile_times, "draft_model", _pt, profile)
            draft_logits = _get_draft_logits(
                draft_hidden, target, freq_used_tokens, freq_reduced_weight, freq_reduced_bias
            )
            _pt = _record_profile(profile_times, "draft_lm_head", _pt, profile)
            past_key_values_draft.crop(start)
            _pt = _record_profile(profile_times, "draft_crop", _pt, profile)

            if draft_prefill:
                draft_prefill = False
                decode_start = cuda_time()

            # Adaptive depth gating only applies to v1; v2/v3 handle depth internally.
            if adaptive_depth and tree_version == 1:
                _ad_t0 = cuda_time() if profile else None
                depth = optimal_tree_depth(draft_logits, threshold=adaptive_depth_threshold)
                tree_logits = draft_logits[:, :depth, :]
                _pt = _record_profile(profile_times, "adaptive_depth", _ad_t0, profile)
            else:
                depth = draft_logits.shape[1]
                tree_logits = draft_logits
            tree_bs = depth + 1  # anchor + depth draft positions

            if dynamic_branching:
                if tree_version == 3:
                    (
                        packed_ids,
                        packed_pos_relative,
                        parent_idx,
                        leaf_paths,
                        leaf_tokens,
                    ) = build_bestfirst_tree(
                        draft_logits=tree_logits,
                        anchor_token_ids=block_output_ids[:, :1],
                        max_tree_size=max_tree_size,
                        expand_k=expand_k,
                        used_tokens=freq_used_tokens,
                    )
                    tree_bs = leaf_tokens.shape[1] + 1
                elif tree_version == 2:
                    (
                        packed_ids,
                        packed_pos_relative,
                        parent_idx,
                        leaf_paths,
                        leaf_tokens,
                    ) = build_dynamic_tree_v2(
                        draft_logits=tree_logits,
                        anchor_token_ids=block_output_ids[:, :1],
                        max_tree_size=max_tree_size,
                        expand_k=expand_k,
                        used_tokens=freq_used_tokens,
                    )
                    tree_bs = leaf_tokens.shape[1] + 1
                else:
                    (
                        packed_ids,
                        packed_pos_relative,
                        parent_idx,
                        leaf_paths,
                        leaf_tokens,
                    ) = build_dynamic_tree(
                        draft_logits=tree_logits,
                        anchor_token_ids=block_output_ids[:, :1],
                        theta_uni=theta_uni,
                        theta_bi=theta_bi,
                        theta_tri=theta_tri,
                        max_tree_size=max_tree_size,
                        top_k=top_k,
                        used_tokens=freq_used_tokens,
                    )
            else:
                packed_ids = sample_first(tree_logits, block_output_ids[:, :1], top_k=top_k, used_tokens=freq_used_tokens)
                packed_pos_relative = get_position_ids(packed_ids, top_k)
            _pt = _record_profile(profile_times, "tree_build", _pt, profile)

            packed_pos = packed_pos_relative + start
            prefix_len = past_key_values_target.get_seq_length()
            if dynamic_branching:
                attn_mask = create_tree_attention_mask_dynamic(
                    packed_pos_relative, parent_idx, prefix_len
                )
            else:
                attn_mask = create_tree_attention_mask(packed_pos_relative, top_k, prefix_len)
            _pt = _record_profile(profile_times, "tree_attn_mask", _pt, profile)

            # Force SDPA for tree verification (flash_attention_2 ignores 4D masks).
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

            if dynamic_branching:
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
            else:
                # Sample target token at position 1 and check K candidates
                if temperature < 1e-5:
                    t1 = logits[:, 0, :].argmax(dim=-1)
                else:
                    t1 = sample(logits[:, 0:1, :], temperature).squeeze(1)

                cands = packed_ids[:, 1:1 + top_k]
                hit = (cands == t1.unsqueeze(-1))
                has = hit.any(dim=-1)
                branch = torch.where(has, hit.float().argmax(dim=-1), torch.zeros_like(t1))

                # Reconstruct realized token sequence along chosen branch
                realized = torch.empty((B, tree_bs), device=packed_ids.device, dtype=torch.long)
                realized[:, 0] = packed_ids[:, 0]
                realized[:, 1] = torch.where(has, t1, cands.gather(-1, branch.unsqueeze(-1)).squeeze(-1))
                if tree_bs > 2:
                    for p in range(2, tree_bs):
                        base = 1 + top_k + (p - 2) * top_k
                        idx = base + branch
                        realized[:, p] = packed_ids.gather(1, idx.unsqueeze(-1)).squeeze(-1)

                path_idx = compute_path_packed_indices(branch, tree_bs, top_k=top_k)

                # Verify rest of the path
                prev_nodes = path_idx[:, :-1]
                prev_logits = logits.gather(1, prev_nodes.unsqueeze(-1).expand(-1, -1, V))
                if temperature < 1e-5:
                    pred = prev_logits.argmax(dim=-1)
                else:
                    pred = sample(prev_logits, temperature)

                matches = (pred == realized[:, 1:])
                matches[:, 0] = matches[:, 0] & has
                acc = matches.cumprod(dim=1)
                n = int(acc.sum(dim=1).item())
            
            _pt = _record_profile(profile_times, "tree_verify_select", _pt, profile)

            output_ids[:, start:start + n + 1] = realized[:, :n + 1]

            # Bonus token from the last accepted node
            last_node = path_idx[:, n]
            last_logits = logits.gather(1, last_node.view(B, 1, 1).expand(B, 1, V)).squeeze(1)
            if temperature < 1e-5:
                next_tok = last_logits.argmax(dim=-1)
            else:
                next_tok = sample(last_logits.unsqueeze(1), temperature).squeeze(1)
            output_ids[:, start + n + 1] = next_tok

            # Surgical KV-cache trim: keep only prefix + accepted path
            accepted_path = path_idx[:, :n + 1]
            trim_target_kv_cache(
                past_key_values_target, prefix_len, accepted_path, packed_ids.device
            )
            _pt = _record_profile(profile_times, "trim_kv_cache", _pt, profile)

            # Extract target_hidden for ALL accepted nodes (same as standard path)
            tree_hidden = extract_context_feature(mod_out.hidden_states, model.target_layer_ids)
            accepted_path_indices = path_idx[:, :n + 1]
            target_hidden = extract_target_hidden_from_tree(tree_hidden, accepted_path_indices)
            _pt = _record_profile(profile_times, "extract_hidden", _pt, profile)

            acceptance_lengths.append(n + 1)
            start += n + 1

        else:
            # Sequential (non-tree) path
            block_position_ids = position_ids[:, start:start + block_size]

            if block_size > 1:
                noise_embedding = target.model.embed_tokens(block_output_ids)
                draft_hidden = model(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[:, past_key_values_draft.get_seq_length():start + block_size],
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                    is_causal=False,
                )[:, -block_size + 1:, :]
                _pt = _record_profile(profile_times, "draft_model", _pt, profile)
                draft_logits = target.lm_head(draft_hidden)
                _pt = _record_profile(profile_times, "draft_lm_head", _pt, profile)
                past_key_values_draft.crop(start)
                _pt = _record_profile(profile_times, "draft_crop", _pt, profile)
                block_output_ids[:, 1:] = sample(draft_logits)
                if draft_prefill:
                    draft_prefill = False
                    decode_start = cuda_time()

            
            mod_out = target.model(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True if block_size > 1 else False,
                return_dict=True,
            )
            _pt = _record_profile(profile_times, "target_backbone", _pt, profile)
            logits_out = target.lm_head(mod_out.last_hidden_state)
            _pt = _record_profile(profile_times, "target_lm_head", _pt, profile)

            posterior = sample(logits_out, temperature)
            acceptance_length = (
                (block_output_ids[:, 1:] == posterior[:, :-1])
                .cumprod(dim=1).sum(dim=1)[0].item()
            )
            output_ids[:, start:start + acceptance_length + 1] = block_output_ids[:, :acceptance_length + 1]
            output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

            acceptance_lengths.append(acceptance_length + 1)
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            if block_size > 1:
                target_hidden = extract_context_feature(
                    mod_out.hidden_states, model.target_layer_ids
                )[:, :acceptance_length + 1, :]
                _pt = _record_profile(profile_times, "extract_hidden", _pt, profile)

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
    parser.add_argument("--chain-attention", action="store_true", default=False,
                        help="Fixed-K tree (no dynamic branching heuristics)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Branching factor K (default: 3)")
    parser.add_argument("--dynamic-branching", action="store_true", default=False,
                        help="Enable dynamic tree building (v1/v2/v3 via --tree-version)")
    parser.add_argument("--theta-uni", type=float, default=0.9)
    parser.add_argument("--theta-bi", type=float, default=0.3)
    parser.add_argument("--theta-tri", type=float, default=0.1)
    parser.add_argument("--max-tree-size", type=int, default=32)
    parser.add_argument("--tree-version", type=int, default=3, choices=[1, 2, 3],
                        help="Tree building: 1=threshold+cap, 2=EAGLE-2 expand+rerank, 3=best-first (default)")
    parser.add_argument("--expand-k", type=int, default=3,
                        help="Per-node expansion width for v2/v3 (default: 3)")
    parser.add_argument("--profile", action="store_true",
                        help="Print CUDA-synced per-step timing breakdown")
    parser.add_argument("--adaptive-depth", action="store_true", default=False,
                        help="Truncate tree depth per step based on draft confidence (v1 only)")
    parser.add_argument("--adaptive-depth-threshold", type=float, default=0.1,
                        help="Cumulative top-1 prob cutoff; 0 disables gating")
    parser.add_argument("--freq-path", type=str, default=None,
                        help="Path to freq_{r}.pt for FR-Spec reduced vocab")
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
                    chain_attention=(args.chain_attention or args.dynamic_branching) if bs > 1 else False,
                    top_k=args.top_k,
                    dynamic_branching=args.dynamic_branching if bs > 1 else False,
                    tree_version=args.tree_version,
                    theta_uni=args.theta_uni,
                    theta_bi=args.theta_bi,
                    theta_tri=args.theta_tri,
                    max_tree_size=args.max_tree_size,
                    expand_k=args.expand_k,
                    adaptive_depth=args.adaptive_depth if bs > 1 else False,
                    adaptive_depth_threshold=args.adaptive_depth_threshold,
                    profile=args.profile,
                    freq_used_tokens=freq_used_tokens,
                    freq_reduced_weight=freq_reduced_weight,
                    freq_reduced_bias=freq_reduced_bias,
                )
            
            spec_response = response[block_size]
            avg_acc = sum(spec_response.acceptance_lengths) / len(spec_response.acceptance_lengths)
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

if __name__ == "__main__":
    main()