import copy
import math
from typing import Callable, Optional, Union

import torch
from torch import nn
from transformers import Qwen3Config
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
)
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs

from .configuration_neo_llm import NEOLLMConfig

try:
    from torch.nn.attention.flex_attention import flex_attention

    flex_attention = torch.compile(flex_attention)
except ImportError:
    print("To enable flexattention, please install torch>=2.5.0")

# Set logging level to INFO to see all messages in console
logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def create_block_causal_mask(index: torch.Tensor):
    """
    index: (L)
    return: (1, 1, L, L) block-wise causal attention mask
    """
    L = index.size(0)
    idx_i = index.unsqueeze(1).expand(L, L)
    idx_j = index.unsqueeze(0).expand(L, L)

    arange = torch.arange(L, device=index.device)
    mask = (idx_j == idx_i) | (arange.unsqueeze(0) <= arange.unsqueeze(1))

    return torch.where(
        mask[None, None, :, :] > 0, torch.tensor(0.0), torch.tensor(float("-inf"))
    )


def visualize_mask(mask: torch.Tensor, i: int = 0, j: int = 12):
    """
    mask: (1,1, L, L)
    """
    submask = torch.where(mask[0, 0, :, :] == 0, torch.tensor(1.0), torch.tensor(0.0))
    submask = mask[i:j, i:j].int().cpu().numpy()
    for row in submask:
        print(" ".join(map(str, row)))


@use_kernel_forward_from_hub("RMSNorm")
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def flex_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_attention_mask: Optional[torch.Tensor] = None,
    scaling: float = 1.0,
):
    return flex_attention(
        query,
        key,
        value,
        enable_gqa=True,
        block_mask=block_attention_mask,
        scale=scaling,
    )


def pad_sequence(tensor, pad_size):
    _, H, L, D = tensor.shape
    pad_tensor = tensor.new_zeros((_, H, pad_size, D))
    return torch.cat([tensor, pad_tensor], dim=2)


class Qwen3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"

        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.q_proj_hw = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )

        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj_hw = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )

        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.q_norm = Qwen3RMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # unlike olmo, only on the head dim!
        self.q_norm_h = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)
        self.q_norm_w = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)

        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm_h = Qwen3RMSNorm(
            self.head_dim // 2, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape
        self.k_norm_w = Qwen3RMSNorm(
            self.head_dim // 2, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape

        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

        hw_config = copy.deepcopy(config)
        hw_config.head_dim = config.head_dim // 2
        hw_config.rope_theta = config.rope_theta_hw
        hw_config.max_position_embeddings = config.max_position_embeddings_hw
        self.rotary_emb_hw = Qwen3RotaryEmbedding(config=hw_config)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        indexes: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        padding_length: int = 0,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states_t = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        query_states_h, query_states_w = (
            self.q_proj_hw(hidden_states)
            .view(hidden_shape)
            .transpose(1, 2)
            .chunk(2, dim=-1)
        )
        query_states_h, query_states_w = self.q_norm_h(query_states_h), self.q_norm_w(
            query_states_w
        )

        key_states_t = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states_h, key_states_w = (
            self.k_proj_hw(hidden_states)
            .view(hidden_shape)
            .transpose(1, 2)
            .chunk(2, dim=-1)
        )
        key_states_h, key_states_w = self.k_norm_h(key_states_h), self.k_norm_w(
            key_states_w
        )
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos_t, sin_t = self.rotary_emb(hidden_states, indexes[:, 0].unsqueeze(0))
        query_states_t, key_states_t = apply_rotary_pos_emb(
            query_states_t, key_states_t, cos_t, sin_t
        )
        cos_h, sin_h = self.rotary_emb_hw(hidden_states, indexes[:, 1].unsqueeze(0))
        query_states_h, key_states_h = apply_rotary_pos_emb(
            query_states_h, key_states_h, cos_h, sin_h
        )

        cos_w, sin_w = self.rotary_emb_hw(hidden_states, indexes[:, 2].unsqueeze(0))
        query_states_w, key_states_w = apply_rotary_pos_emb(
            query_states_w, key_states_w, cos_w, sin_w
        )

        query_states = torch.cat(
            [query_states_t, query_states_h, query_states_w], dim=-1
        )
        key_states = torch.cat([key_states_t, key_states_h, key_states_w], dim=-1)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs=None
            )

        if padding_length > 0:
            query_states = pad_sequence(query_states, padding_length)
            key_states = pad_sequence(key_states, padding_length)
            value_states = pad_sequence(value_states, padding_length)

        attn_output = flex_attention_forward(
            query_states,
            key_states,
            value_states,
            block_attention_mask=attention_mask,
            scaling=1.0 / math.sqrt(query_states.size(-1) // 2),
        )

        if padding_length > 0:
            end_index = attn_output.shape[2] - padding_length
            attn_output = attn_output[:, :, :end_index, :]

        # Permute from (batch, heads, seq_len, head_dim) to (batch, seq_len, heads, head_dim)
        # before reshaping to (batch, seq_len, hidden_size)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None


class Qwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(
        self, config: Qwen3Config, layer_idx: int, scale_attn_weights: bool = False
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attention_type = config.layer_types[layer_idx]

        self.reset_parameters()

    def reset_parameters(
        self,
        attn_wqkv_init_std: float = 0.02,
        attn_other_init_std: float = 0.02,
        ffn_uplayer_init_std: float = 0.02,
        ffn_other_init_std: float = 0.02,
        use_scaled_init: bool = True,
        init_type: str = "normal",
    ):
        """
        Reset parameters for decoder layer including attention, MLP, and layer norms.

        Args:
            attn_wqkv_init_std: Standard deviation for initializing attention Q/K/V projection weights
            attn_other_init_std: Standard deviation for initializing attention output projection weights
            ffn_uplayer_init_std: Standard deviation for initializing MLP gate_proj and up_proj weights
            ffn_other_init_std: Standard deviation for initializing MLP down_proj weights
            use_scaled_init: Whether to use scaled initialization for output projections
            init_type: Initialization type, either "normal" or "uniform"
        """
        with torch.no_grad():
            # Helper functions for initialization
            if init_type == "normal":

                def init_func(std):
                    def _init(param):
                        torch.nn.init.normal_(param, mean=0.0, std=std)

                    return _init

                def scaled_init_func(sigma, num_layers):
                    def _init(param):
                        std = sigma / math.sqrt(2 * num_layers)
                        torch.nn.init.normal_(param, mean=0.0, std=std)

                    return _init

            else:  # uniform

                def init_func(std):
                    def _init(param):
                        bound = std * math.sqrt(3.0)
                        torch.nn.init.uniform_(param, -bound, bound)

                    return _init

                def scaled_init_func(sigma, num_layers):
                    def _init(param):
                        std = sigma / math.sqrt(2 * num_layers)
                        bound = std * math.sqrt(3.0)
                        torch.nn.init.uniform_(param, -bound, bound)

                    return _init

            # 1. Initialize attention layer parameters
            for name, param in self.self_attn.named_parameters():
                if param.ndim == 1:
                    # Bias parameters: initialize to zero
                    param.data.zero_()
                elif (
                    "q_proj.weight" in name
                    or "q_proj_hw.weight" in name
                    or "k_proj.weight" in name
                    or "k_proj_hw.weight" in name
                    or "v_proj.weight" in name
                ):
                    # Q, K, V projection weights (including q_proj_hw, k_proj_hw)
                    init_func(attn_wqkv_init_std)(param.data)
                elif "o_proj.weight" in name:
                    # Output projection weight with optional scaled initialization
                    if use_scaled_init:
                        scaled_init_func(attn_other_init_std, self.layer_idx + 1)(
                            param.data
                        )
                    else:
                        init_func(attn_other_init_std)(param.data)
                # Note: RMSNorm weights are initialized to ones by default and typically don't need re-initialization

            # 2. Initialize MLP layer parameters
            for name, param in self.mlp.named_parameters():
                if param.ndim == 1:
                    # Bias parameters: initialize to zero
                    param.data.zero_()
                elif "gate_proj.weight" in name or "up_proj.weight" in name:
                    # Gate and up projection weights (SwiGLU uplayers)
                    init_func(ffn_uplayer_init_std)(param.data)
                elif "down_proj.weight" in name:
                    # Down projection weight with optional scaled initialization
                    if use_scaled_init:
                        scaled_init_func(ffn_other_init_std, self.layer_idx + 1)(
                            param.data
                        )
                    else:
                        init_func(ffn_other_init_std)(param.data)

            # 3. Layer norms are already initialized to ones by default
            # No need to re-initialize unless specifically required

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        indexes: Optional[torch.LongTensor] = None,
        padding_length: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            indexes=indexes,
            padding_length=padding_length,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class Qwen3PreTrainedModel(PreTrainedModel):
    config: Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen3DecoderLayer,
        "attentions": Qwen3Attention,
    }


@auto_docstring
class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.current_index = -1

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        indexes: Optional[torch.LongTensor] = None,
        padding_length: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        indexes (`torch.LongTensor` of shape `(3, sequence_length)`, *optional*):
            3D RoPE position indexes for temporal, height, and width dimensions.
        padding_length (`int`, *optional*, defaults to 0):
            Length of padding added to the sequence for divisibility requirements.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            if input_ids is not None:
                mask_kwargs = {
                    "config": self.config,
                    "input_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "cache_position": cache_position,
                    "past_key_values": past_key_values,
                    "position_ids": position_ids,
                }
                # Create the masks
                causal_mask_mapping = {
                    "full_attention": create_causal_mask(**mask_kwargs),
                }
                self.current_index += 1
                indexes = torch.LongTensor([[self.current_index], [0], [0]]).to(
                    input_ids.device
                )
            else:
                causal_mask_mapping = {
                    "full_attention": create_block_causal_mask(indexes[0]),
                }
                self.current_index = indexes[0].max()
        else:
            raise NotImplementedError(
                "not isinstance(causal_mask_mapping := attention_mask, dict)"
            )

            # The sliding window alternating layers are not always activated depending on the config
            # if self.has_sliding_layers:
            #     causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                indexes=indexes,
                padding_length=padding_length,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


@auto_docstring
class Qwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize loss function
        self.loss_weight_fn = nn.CrossEntropyLoss(
            ignore_index=-100,
            reduction="none",
            label_smoothing=0.0,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def loss_function(self, logits, labels, vocab_size, **kwargs):
        """
        Compute weighted cross entropy loss.

        Args:
            logits: Model logits
            labels: Target labels
            vocab_size: Vocabulary size
            **kwargs: Additional arguments (e.g., loss_weight)

        Returns:
            Weighted average loss
        """
        # Shift logits and labels for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)

        # Get loss_weight from kwargs, default to ones
        loss_weight = kwargs.get("loss_weight", None)
        if loss_weight is None:
            # Create default weight: 1.0 for valid tokens, 0.0 for ignored tokens
            loss_weight = (shift_labels != -100).float()
        else:
            # Ensure loss_weight is on the same device and dtype
            if not isinstance(loss_weight, torch.Tensor):
                loss_weight = torch.tensor(
                    loss_weight, dtype=torch.float32, device=shift_labels.device
                )
            else:
                loss_weight = loss_weight.to(
                    dtype=torch.float32, device=shift_labels.device
                )

            # Flatten loss_weight to match shift_labels
            if loss_weight.dim() > 1:
                loss_weight = loss_weight[..., 1:].contiguous().view(-1)

        # Compute loss without reduction
        loss = self.loss_weight_fn(shift_logits, shift_labels)

        # Apply weight and compute weighted average
        weight_sum = loss_weight.sum()
        loss = loss * loss_weight
        loss = loss.sum() / weight_sum.clamp(min=1e-5)  # Avoid division by zero

        return loss

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Custom from_pretrained method to load Qwen3 weights into our custom NEO model.

        This method handles two scenarios:
        1. Loading from official Qwen3 checkpoint (without q_proj_hw/k_proj_hw) - initializes HW weights
        2. Loading from pretrained NEO checkpoint (with q_proj_hw/k_proj_hw) - direct loading
        """
        import glob
        import os

        # Extract config if provided, otherwise load from checkpoint
        config = kwargs.pop("config", None)
        if config is None:
            config = NEOLLMConfig.from_pretrained(pretrained_model_name_or_path)

        # Create model instance with the config
        model = cls(config)

        # Load state dict from checkpoint
        try:
            from safetensors.torch import load_file as safe_load_file
        except ImportError:
            safe_load_file = None
            logger.warning("safetensors not available, will try pytorch format")

        state_dict = {}
        model_path = pretrained_model_name_or_path

        # Load checkpoint files
        if (
            os.path.isfile(os.path.join(model_path, "model.safetensors"))
            and safe_load_file
        ):
            logger.info(f"Loading from single safetensors file")
            state_dict = safe_load_file(os.path.join(model_path, "model.safetensors"))
        elif os.path.isfile(os.path.join(model_path, "pytorch_model.bin")):
            logger.info(f"Loading from single pytorch bin file")
            state_dict = torch.load(
                os.path.join(model_path, "pytorch_model.bin"), map_location="cpu"
            )
        else:
            # Handle sharded checkpoints
            if safe_load_file:
                safetensor_files = sorted(
                    glob.glob(os.path.join(model_path, "model-*.safetensors"))
                )
                if safetensor_files:
                    logger.info(
                        f"Loading from {len(safetensor_files)} sharded safetensors files"
                    )
                    for file_path in safetensor_files:
                        state_dict.update(safe_load_file(file_path))

            if not state_dict:
                bin_files = sorted(
                    glob.glob(os.path.join(model_path, "pytorch_model-*.bin"))
                )
                if bin_files:
                    logger.info(
                        f"Loading from {len(bin_files)} sharded pytorch bin files"
                    )
                    for file_path in bin_files:
                        state_dict.update(torch.load(file_path, map_location="cpu"))

        if not state_dict:
            raise ValueError(f"No valid checkpoint files found in {model_path}")

        # Check if this is a standard Qwen3 model or a NEO model with HW weights
        is_standard_qwen3 = (
            "model.layers.0.self_attn.q_proj_hw.weight" not in state_dict
        )

        if is_standard_qwen3:
            logger.info("Initializing NEO VLM from standard Qwen3 LLM checkpoint")
            # Need to initialize q_proj_hw and k_proj_hw from q_proj and k_proj
            neo_state_dict = {}

            # Get extra_num_layers from config (default to 0 if not specified)
            extra_num_layers = getattr(config, "extra_num_layers", 0)

            def initialize_hw_weight(weight, head_dim):
                """
                Initialize HW projection weights from standard projection weights.
                Extracts and duplicates specific dimensions for height/width encoding.
                """
                # weight shape: [num_heads * head_dim, hidden_size]
                num_heads = weight.shape[0] // head_dim
                # Reshape to [hidden_size, num_heads, head_dim]
                weight_t = weight.permute(1, 0).reshape(
                    weight.shape[1], num_heads, head_dim
                )
                # Split head_dim into 4 parts and take 1st and 3rd quarters
                w1, _, w3, _ = weight_t.chunk(4, dim=-1)
                # Concatenate and repeat to match expected shape
                weight_hw = torch.cat([w1, w3], dim=-1).repeat(1, 1, 2)
                # Reshape back to [num_heads * head_dim, hidden_size]
                return weight_hw.reshape(weight.shape[1], -1).permute(1, 0)

            # Map weights from standard Qwen3 to NEO format
            # Checkpoint layers [0, num_checkpoint_layers) map to NEO layers [extra_num_layers, num_checkpoint_layers+extra_num_layers)
            for key, value in state_dict.items():
                if key.startswith("model.layers."):
                    # Extract layer number from checkpoint
                    parts = key.split(".")
                    checkpoint_layer_idx = int(parts[2])

                    # Map to NEO layer index: checkpoint layer i -> NEO layer (i + extra_num_layers)
                    neo_layer_idx = checkpoint_layer_idx + extra_num_layers

                    # Reconstruct key with new layer index
                    parts[2] = str(neo_layer_idx)
                    new_key = ".".join(parts)

                    if ".self_attn.q_proj.weight" in key:
                        neo_state_dict[new_key] = value
                        # Initialize q_proj_hw
                        hw_key = new_key.replace("q_proj.weight", "q_proj_hw.weight")
                        neo_state_dict[hw_key] = initialize_hw_weight(
                            value, config.head_dim
                        )
                    elif ".self_attn.k_proj.weight" in key:
                        neo_state_dict[new_key] = value
                        # Initialize k_proj_hw with zeros (as in original code)
                        hw_key = new_key.replace("k_proj.weight", "k_proj_hw.weight")
                        neo_state_dict[hw_key] = torch.zeros_like(value)
                    elif (
                        ".self_attn.q_norm.weight" in key
                        or ".self_attn.k_norm.weight" in key
                    ):
                        # Handle QK norm weights
                        neo_state_dict[new_key] = value

                        # Initialize q_norm_h, q_norm_w or k_norm_h, k_norm_w
                        half_shape = value.shape[0] // 2
                        half_ones = torch.ones(
                            half_shape, device=value.device, dtype=value.dtype
                        )

                        for axis in ["h", "w"]:
                            if ".q_norm.weight" in key:
                                hw_key = new_key.replace(
                                    "q_norm.weight", f"q_norm_{axis}.weight"
                                )
                            else:  # k_norm
                                hw_key = new_key.replace(
                                    "k_norm.weight", f"k_norm_{axis}.weight"
                                )
                            neo_state_dict[hw_key] = half_ones.clone()
                    else:
                        neo_state_dict[new_key] = value
                else:
                    neo_state_dict[key] = value

            state_dict = neo_state_dict
            logger.info("Initialized HW projection weights from standard weights")

            # Initialize all weights for extra_num_layers if needed
            if extra_num_layers > 0:
                logger.info(
                    f"Initializing all weights for {extra_num_layers} extra pre-buffer layers"
                )

                # Get reference weights from the first loaded layer to determine shapes and dtypes
                ref_layer_idx = extra_num_layers

                # Initialize std values (following common practice)
                init_std = 0.02

                for j in range(extra_num_layers):
                    layer_prefix = f"model.layers.{j}"

                    # 1. Initialize attention projection weights with normal distribution
                    for proj_name in [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "q_proj_hw",
                        "k_proj_hw",
                    ]:
                        weight_key = f"{layer_prefix}.self_attn.{proj_name}.weight"
                        # Get reference shape from corresponding projection in first loaded layer
                        ref_key = (
                            f"model.layers.{ref_layer_idx}.self_attn.{proj_name}.weight"
                        )

                        if ref_key in state_dict:
                            # Get reference shape from existing layer
                            ref_weight = state_dict[ref_key]
                            shape = ref_weight.shape
                            dtype = ref_weight.dtype
                            device = ref_weight.device
                        else:
                            # Fallback: use config to determine shapes
                            if proj_name in ["q_proj", "q_proj_hw"]:
                                out_features = (
                                    config.num_attention_heads * config.head_dim
                                )
                            elif proj_name in ["k_proj", "k_proj_hw", "v_proj"]:
                                out_features = (
                                    config.num_key_value_heads * config.head_dim
                                )
                            else:  # o_proj
                                out_features = config.hidden_size
                            shape = (out_features, config.hidden_size)
                            dtype = torch.float32
                            device = "cpu"

                        # Initialize with normal distribution
                        state_dict[weight_key] = (
                            torch.randn(shape, dtype=dtype, device=device) * init_std
                        )

                    # 2. Initialize MLP weights
                    ref_mlp_key = f"model.layers.{ref_layer_idx}.mlp.gate_proj.weight"
                    if ref_mlp_key in state_dict:
                        ref_mlp_weight = state_dict[ref_mlp_key]
                        mlp_dtype = ref_mlp_weight.dtype
                        mlp_device = ref_mlp_weight.device
                        intermediate_size = ref_mlp_weight.shape[0]
                        hidden_size = ref_mlp_weight.shape[1]
                    else:
                        mlp_dtype = torch.float32
                        mlp_device = "cpu"
                        intermediate_size = int(config.hidden_size * 4)
                        hidden_size = config.hidden_size

                    # gate_proj, up_proj, down_proj
                    state_dict[f"{layer_prefix}.mlp.gate_proj.weight"] = (
                        torch.randn(
                            intermediate_size,
                            hidden_size,
                            dtype=mlp_dtype,
                            device=mlp_device,
                        )
                        * init_std
                    )
                    state_dict[f"{layer_prefix}.mlp.up_proj.weight"] = (
                        torch.randn(
                            intermediate_size,
                            hidden_size,
                            dtype=mlp_dtype,
                            device=mlp_device,
                        )
                        * init_std
                    )
                    state_dict[f"{layer_prefix}.mlp.down_proj.weight"] = (
                        torch.randn(
                            hidden_size,
                            intermediate_size,
                            dtype=mlp_dtype,
                            device=mlp_device,
                        )
                        * init_std
                    )

                    # 3. Initialize layer norms (input_layernorm, post_attention_layernorm)
                    ref_norm_key = (
                        f"model.layers.{ref_layer_idx}.input_layernorm.weight"
                    )
                    if ref_norm_key in state_dict:
                        ref_norm = state_dict[ref_norm_key]
                        norm_shape = ref_norm.shape[0]
                        norm_dtype = ref_norm.dtype
                        norm_device = ref_norm.device
                    else:
                        norm_shape = config.hidden_size
                        norm_dtype = torch.float32
                        norm_device = "cpu"

                    state_dict[f"{layer_prefix}.input_layernorm.weight"] = torch.ones(
                        norm_shape, dtype=norm_dtype, device=norm_device
                    )
                    state_dict[f"{layer_prefix}.post_attention_layernorm.weight"] = (
                        torch.ones(norm_shape, dtype=norm_dtype, device=norm_device)
                    )

                    # 4. Initialize QK norm weights
                    ref_qnorm_key = (
                        f"model.layers.{ref_layer_idx}.self_attn.q_norm.weight"
                    )
                    if ref_qnorm_key in state_dict:
                        ref_qnorm = state_dict[ref_qnorm_key]
                        qnorm_shape = ref_qnorm.shape[0]
                        qnorm_dtype = ref_qnorm.dtype
                        qnorm_device = ref_qnorm.device
                    else:
                        qnorm_shape = config.head_dim
                        qnorm_dtype = torch.float32
                        qnorm_device = "cpu"

                    for name in ["q", "k"]:
                        # Main norm weights
                        state_dict[f"{layer_prefix}.self_attn.{name}_norm.weight"] = (
                            torch.ones(
                                qnorm_shape, dtype=qnorm_dtype, device=qnorm_device
                            )
                        )

                        # H and W norm weights (half size)
                        for axis in ["h", "w"]:
                            state_dict[
                                f"{layer_prefix}.self_attn.{name}_norm_{axis}.weight"
                            ] = torch.ones(
                                qnorm_shape // 2, dtype=qnorm_dtype, device=qnorm_device
                            )

                logger.info(
                    f"Initialized all weights (attention, MLP, norms) for layers 0-{extra_num_layers-1}"
                )
        else:
            logger.info(
                "Loading NEO VLM from pretrained NEO checkpoint with HW weights"
            )

        # Handle vocab size mismatch if needed
        if hasattr(config, "vocab_size"):
            if "lm_head.weight" in state_dict:
                old_vocab_size = state_dict["lm_head.weight"].shape[0]
            elif "model.embed_tokens.weight" in state_dict:
                old_vocab_size = state_dict["model.embed_tokens.weight"].shape[0]
            else:
                old_vocab_size = None

            if old_vocab_size and old_vocab_size != config.vocab_size:
                logger.warning(
                    f"Vocab size mismatch: checkpoint has {old_vocab_size}, config has {config.vocab_size}"
                )
                # Resize embeddings
                if "model.embed_tokens.weight" in state_dict:
                    old_embed = state_dict["model.embed_tokens.weight"]
                    new_embed = torch.zeros(
                        config.vocab_size, old_embed.shape[1], dtype=old_embed.dtype
                    )
                    new_embed[: min(old_vocab_size, config.vocab_size)] = old_embed[
                        : min(old_vocab_size, config.vocab_size)
                    ]
                    state_dict["model.embed_tokens.weight"] = new_embed

                if "lm_head.weight" in state_dict:
                    old_lm_head = state_dict["lm_head.weight"]
                    new_lm_head = torch.zeros(
                        config.vocab_size, old_lm_head.shape[1], dtype=old_lm_head.dtype
                    )
                    new_lm_head[: min(old_vocab_size, config.vocab_size)] = old_lm_head[
                        : min(old_vocab_size, config.vocab_size)
                    ]
                    state_dict["lm_head.weight"] = new_lm_head

        # If lm_head.weight is missing, use embed_tokens.weight
        # IMPORTANT: Clone operation must happen on CPU state_dict before model loading
        # to ensure Zero3 compatibility (state_dict is on CPU, not partitioned)
        if (
            "lm_head.weight" not in state_dict
            and "model.embed_tokens.weight" in state_dict
        ):
            logger.info(
                "lm_head.weight not found in checkpoint, using model.embed_tokens.weight"
            )
            # Safe clone operation on CPU tensor (Zero3 compatible)
            state_dict["lm_head.weight"] = state_dict[
                "model.embed_tokens.weight"
            ].clone()

        # Load state dict into model
        # Note: With DeepSpeed Zero3, this will automatically partition parameters
        incompatible_keys = model.load_state_dict(state_dict, strict=False)

        if incompatible_keys.missing_keys:
            logger.warning(
                f"Missing keys when loading: {incompatible_keys.missing_keys}"
            )
        if incompatible_keys.unexpected_keys:
            logger.warning(
                f"Unexpected keys when loading: {incompatible_keys.unexpected_keys}"
            )

        logger.info("Successfully loaded pretrained weights")
        return model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        indexes: Optional[torch.LongTensor] = None,
        padding_length: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        indexes (`torch.LongTensor` of shape `(3, sequence_length)`, *optional*):
            3D RoPE position indexes for temporal, height, and width dimensions.
        padding_length (`int`, *optional*, defaults to 0):
            Length of padding added to the sequence for divisibility requirements.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            indexes=indexes,
            padding_length=padding_length,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["Qwen3ForCausalLM", "Qwen3PreTrainedModel", "Qwen3Model"]
