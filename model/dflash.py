from typing import Optional, Callable
from typing_extensions import Unpack, Tuple
import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    Qwen3Config,
    Qwen3PreTrainedModel,
    Qwen3MLP,
    GradientCheckpointingLayer,
    FlashAttentionKwargs,
    rotate_half,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
)
from transformers import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache
from .utils import (
    build_target_layer_ids,
    extract_context_feature,
    extract_target_hidden_from_tree,
    sample,
    trim_target_kv_cache,
)
from .dflash_tree import (
    sample_first,
    get_position_ids,
    create_tree_attention_mask,
    compute_path_packed_indices,
    build_dynamic_tree,
    create_tree_attention_mask_dynamic,
    select_best_dynamic_leaf,
)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Qwen3DFlashAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False  
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3DFlashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.target_layer_ids = self.config.dflash_config.get("target_layer_ids", build_target_layer_ids(config.num_target_layers, config.num_hidden_layers))
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = config.block_size
        self.mask_token_id = self.config.dflash_config.get("mask_token_id", None)
        self.post_init()

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        # position_embeddings must cover full kv sequence: k_ctx (0..ctx_len-1) + k_noise (position_ids)
        ctx_len = target_hidden.shape[1]
        q_len = hidden_states.shape[1]
        full_position_ids = torch.cat([
            torch.arange(ctx_len, device=position_ids.device, dtype=position_ids.dtype)
            .unsqueeze(0).expand(position_ids.shape[0], -1),
            position_ids,
        ], dim=1)
        position_embeddings = self.rotary_emb(hidden_states, full_position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)
    
    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
        chain_attention: bool = False,
        top_k: int = 3,
        dynamic_branching: bool = False,
        theta_uni: float = 0.9,
        theta_bi: float = 0.3,
        theta_tri: float = 0.1,
        max_tree_size: int = 8,
    ):
        """
        Speculative decoding with an optional chain-attention (tree) mode.

        Args:
            chain_attention: When True, uses K-way tree branching at position 1
                and a single target forward pass per step with a tree attention
                mask and surgical KV-cache trimming. When False (default), uses
                the standard sequential draft-KV-cache path.
            top_k: Branching factor for the tree (only used when chain_attention=True).

        Returns:
            (output_ids, stats) where stats = {"acceptance_lengths": list[int],
                                               "avg_acceptance_length": float}
        """
        self.eval()
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens

        block_size = self.block_size
        output_ids = torch.full(
            (1, max_length + block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=target.device,
        )
        position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)

        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()

        # Prefill stage
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )

        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens:num_input_tokens + 1] = sample(output.logits, temperature)
        target_hidden = extract_context_feature(output.hidden_states, self.target_layer_ids)

        acceptance_lengths = []
        start = input_ids.shape[1]

        if chain_attention:
            # ------------------------------------------------------------------
            # Chain-attention path: tree branching + single target forward pass
            # ------------------------------------------------------------------
            while start < max_length:
                block_output_ids = output_ids[:, start:start + block_size].clone()
                noise_embedding = target.model.embed_tokens(block_output_ids)

                # Use absolute position IDs and draft KV cache, identical to
                # the standard path.
                draft_logits = target.lm_head(
                    self(
                        target_hidden=target_hidden,
                        noise_embedding=noise_embedding,
                        position_ids=position_ids[:, past_key_values_draft.get_seq_length():start + block_size],
                        past_key_values=past_key_values_draft,
                        use_cache=True,
                        is_causal=False,
                    )[:, -block_size + 1:, :]
                )
                past_key_values_draft.crop(start)

                # Build tree and run target
                if dynamic_branching:
                    (
                        packed_ids,
                        packed_pos_relative,
                        parent_idx,
                        leaf_paths,
                        leaf_tokens,
                    ) = build_dynamic_tree(
                        draft_logits=draft_logits,
                        anchor_token_ids=block_output_ids[:, :1],
                        theta_uni=theta_uni,
                        theta_bi=theta_bi,
                        theta_tri=theta_tri,
                        max_tree_size=max_tree_size,
                    )
                else:
                    packed_ids = sample_first(draft_logits, block_output_ids[:, :1], top_k=top_k)
                    packed_pos_relative = get_position_ids(packed_ids, top_k)
                packed_pos = packed_pos_relative + start
                prefix_len = past_key_values_target.get_seq_length()
                if dynamic_branching:
                    attn_mask = create_tree_attention_mask_dynamic(
                        packed_pos_relative, parent_idx, prefix_len
                    )
                else:
                    attn_mask = create_tree_attention_mask(packed_pos_relative, top_k, prefix_len)

                # Force SDPA for tree verification — flash_attention_2 ignores
                # custom 4D masks when is_causal=True (the target model default).
                saved_attn_impl = target.config._attn_implementation
                target.config._attn_implementation = "sdpa"
                output = target(
                    packed_ids,
                    position_ids=packed_pos,
                    past_key_values=past_key_values_target,
                    use_cache=True,
                    output_hidden_states=True,
                    attention_mask=attn_mask,
                )
                target.config._attn_implementation = saved_attn_impl
                logits = output.logits
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
                    realized = torch.empty((1, block_size), device=packed_ids.device, dtype=torch.long)
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

                    # Reconstruct the realized token sequence along chosen branch
                    realized = torch.empty((B, block_size), device=packed_ids.device, dtype=torch.long)
                    realized[:, 0] = packed_ids[:, 0]
                    realized[:, 1] = torch.where(has, t1, cands.gather(-1, branch.unsqueeze(-1)).squeeze(-1))
                    if block_size > 2:
                        for p in range(2, block_size):
                            base = 1 + top_k + (p - 2) * top_k
                            idx = base + branch
                            realized[:, p] = packed_ids.gather(1, idx.unsqueeze(-1)).squeeze(-1)

                    path_idx = compute_path_packed_indices(branch, block_size, top_k=top_k)

                    # Verify the rest of the path
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

                output_ids[:, start:start + n + 1] = realized[:, :n + 1]

                # Bonus token from the last accepted node
                last_node = path_idx[:, n]
                last_logits = logits.gather(1, last_node.view(B, 1, 1).expand(B, 1, V)).squeeze(1)
                if temperature < 1e-5:
                    next_tok = last_logits.argmax(dim=-1)
                else:
                    next_tok = sample(last_logits.unsqueeze(1), temperature).squeeze(1)
                output_ids[:, start + n + 1] = next_tok

                # Surgical KV-cache trim: keep prefix + accepted path only
                accepted_path = path_idx[:, :n + 1]
                trim_target_kv_cache(
                    past_key_values_target, prefix_len, accepted_path, packed_ids.device
                )

                # Extract target_hidden for ALL accepted nodes (same as standard path)
                tree_hidden = extract_context_feature(output.hidden_states, self.target_layer_ids)
                accepted_path_indices = path_idx[:, :n + 1]
                target_hidden = extract_target_hidden_from_tree(tree_hidden, accepted_path_indices)

                start += n + 1
                acceptance_lengths.append(n + 1)

                if stop_token_ids is not None and any(
                    sid in output_ids[:, num_input_tokens:] for sid in stop_token_ids
                ):
                    break

        else:
            # ------------------------------------------------------------------
            # Standard path: sequential draft KV-cache
            # ------------------------------------------------------------------
            while start < max_length:
                block_output_ids = output_ids[:, start:start + block_size].clone()
                block_position_ids = position_ids[:, start:start + block_size]
                noise_embedding = target.model.embed_tokens(block_output_ids)

                draft_logits = target.lm_head(
                    self(
                        target_hidden=target_hidden,
                        noise_embedding=noise_embedding,
                        position_ids=position_ids[:, past_key_values_draft.get_seq_length():start + block_size],
                        past_key_values=past_key_values_draft,
                        use_cache=True,
                        is_causal=False,
                    )[:, -block_size + 1:, :]
                )
                past_key_values_draft.crop(start)
                block_output_ids[:, 1:] = sample(draft_logits)

                output = target(
                    block_output_ids,
                    position_ids=block_position_ids,
                    past_key_values=past_key_values_target,
                    use_cache=True,
                    output_hidden_states=True,
                )

                posterior = sample(output.logits, temperature)
                acceptance_length = (
                    (block_output_ids[:, 1:] == posterior[:, :-1])
                    .cumprod(dim=1).sum(dim=1)[0].item()
                )
                output_ids[:, start:start + acceptance_length + 1] = block_output_ids[:, :acceptance_length + 1]
                output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
                start += acceptance_length + 1
                past_key_values_target.crop(start)
                target_hidden = extract_context_feature(
                    output.hidden_states, self.target_layer_ids
                )[:, :acceptance_length + 1, :]
                acceptance_lengths.append(acceptance_length + 1)

                if stop_token_ids is not None and any(
                    sid in output_ids[:, num_input_tokens:] for sid in stop_token_ids
                ):
                    break

        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        if stop_token_ids is not None:
            stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_token_indices = torch.isin(
                output_ids[0][num_input_tokens:], stop_token_ids
            ).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[:, :num_input_tokens + stop_token_indices[0] + 1]

        avg_acceptance_length = (
            sum(acceptance_lengths) / len(acceptance_lengths) if acceptance_lengths else 0.0
        )
        return output_ids, {
            "acceptance_lengths": acceptance_lengths,
            "avg_acceptance_length": avg_acceptance_length,
        }
