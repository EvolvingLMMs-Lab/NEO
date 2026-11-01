"""The action expert: the trainable suffix stream beside the backbone.

One narrow transformer whose layer j attends into backbone layer layer_map[j].
The suffix tokens project from the expert width into the backbone's head space,
rotate with the backbone's own phases from PrefixContext, and attend over
[prefix keys and values, then the suffix's own] in one softmax at the
backbone's scale.

layer_map is a config value; config/arch/skip28.yaml is the one shipped map. It
is not serialized: the run config stored beside every
checkpoint is its authority, and a persistent buffer would make a strict load
depend on it.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.neo_llm import RmsNorm, apply_rope_part
from .prefix import PrefixContext


class ExpertFeedForward(nn.Module):
    """Gated feed-forward at the expert width."""

    def __init__(self, width: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(width, intermediate, bias=False)
        self.up_proj = nn.Linear(width, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ExpertLayer(nn.Module):
    """One expert layer: joint attention over prefix and suffix, then a mixer.

    Its query and key norms mirror the backbone's merged split, one norm for
    the t half and one for the h and w half, so the backbone's gains copy
    across one to one. Suffix tokens have h and w indexes of zero, so only the
    t half rotates.
    """

    def __init__(
        self,
        width: int,
        intermediate: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads)
        self.head_dim = int(head_dim)
        self.input_layernorm = RmsNorm(width)
        self.q_proj = nn.Linear(width, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(width, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(width, self.n_kv_heads * self.head_dim, bias=False)
        half = self.head_dim // 2
        self.q_norm = RmsNorm(half)
        self.q_norm_hw = RmsNorm(half)
        self.k_norm = RmsNorm(half)
        self.k_norm_hw = RmsNorm(half)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, width, bias=False)
        self.post_attention_layernorm = RmsNorm(width)
        self.mlp = ExpertFeedForward(width, intermediate)

    def project_and_rotate(
        self,
        x: torch.Tensor,
        proj: nn.Linear,
        norm_t: RmsNorm,
        norm_hw: RmsNorm,
        n_heads: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """x [B, S, width] -> [B, n_heads, S, head_dim], normed and rotated."""
        batch, length, _ = x.shape
        time, space = proj(x).view(batch, length, n_heads, self.head_dim).chunk(2, dim=-1)
        time = apply_rope_part(norm_t(time).transpose(1, 2), cos, sin)
        return torch.cat([time, norm_hw(space).transpose(1, 2)], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        prefix_key: torch.Tensor,
        prefix_value: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_bias: torch.Tensor,
        scaling: float,
    ) -> torch.Tensor:
        """x [B, n_suffix, width] -> [B, n_suffix, width].

        prefix_key and prefix_value are [B, n_kv_heads, Lp, head_dim] and carry
        a graph whenever the backbone is training. cos and sin are
        [B, n_suffix, head_dim // 2]; attention_bias is additive
        [B, 1, n_suffix, Lp + n_suffix].
        """
        batch, length, _ = x.shape
        normed = self.input_layernorm(x)
        query = self.project_and_rotate(
            normed, self.q_proj, self.q_norm, self.q_norm_hw, self.n_heads, cos, sin
        )
        key = self.project_and_rotate(
            normed, self.k_proj, self.k_norm, self.k_norm_hw, self.n_kv_heads, cos, sin
        )
        value = (
            self.v_proj(normed)
            .view(batch, length, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        keys = torch.cat([prefix_key.to(query.dtype), key], dim=2)
        values = torch.cat([prefix_value.to(query.dtype), value], dim=2)
        repeat = self.n_heads // self.n_kv_heads
        if repeat > 1:
            keys = keys.repeat_interleave(repeat, dim=1)
            values = values.repeat_interleave(repeat, dim=1)
        # The bias arrives in float32 and carries -inf; clamping after the cast
        # keeps a heavily masked row finite instead of nan. It is key padding
        # only (decoder.attention_bias, suffix layout v2), so a boolean or a
        # [B, 1, 1, L] mask would say the same; measured on one H100 under
        # torch 2.6 and torch.compile they match this path's outputs, gradients and time
        # within 1%, and
        # FlexAttention's backward does not lower there, so this path stays.
        bias = attention_bias.to(query.dtype).clamp(min=torch.finfo(query.dtype).min)
        out = F.scaled_dot_product_attention(
            query, keys, values, attn_mask=bias, scale=scaling
        )
        out = out.transpose(1, 2).reshape(batch, length, -1)
        x = x + self.o_proj(out)
        return x + self.mlp(self.post_attention_layernorm(x))


class ActionExpert(nn.Module):
    """len(layer_map) expert layers over a PrefixContext.

    Embeds nothing itself: the decoder hands it suffix token embeddings and
    gets post-norm hidden states back.
    """

    def __init__(
        self,
        *,
        width: int,
        intermediate: int,
        layer_map: tuple[int, ...],
        backbone_num_layers: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        init_std: float = 0.02,
    ):
        """Raises unless layer_map is strictly increasing and inside
        [0, backbone_num_layers). Depth is the length of the map; there is no
        separate layer count to disagree with it.
        """
        super().__init__()
        mapped = tuple(int(index) for index in layer_map)
        depth = int(backbone_num_layers)
        if not mapped:
            raise ValueError(
                "layer_map is empty, expected at least one attach point; "
                "set model.layer_map from one of config/arch/*.yaml"
            )
        if any(later <= earlier for earlier, later in zip(mapped, mapped[1:], strict=False)):
            raise ValueError(
                f"layer_map {mapped} is not strictly increasing, expected "
                "ascending distinct indexes; sort it and drop the duplicates"
            )
        if mapped[0] < 0 or mapped[-1] >= depth:
            raise ValueError(
                f"layer_map spans {mapped[0]}..{mapped[-1]}, expected indexes "
                f"inside [0, {depth}); fix model.layer_map or "
                "model.backbone_num_layers"
            )
        self.layer_map = mapped
        self.backbone_num_layers = depth
        self.expert_width = int(width)
        self.layers = nn.ModuleList(
            ExpertLayer(int(width), int(intermediate), n_heads, n_kv_heads, head_dim)
            for _ in mapped
        )
        self.norm = RmsNorm(int(width))
        self.init_parameters(float(init_std))

    @property
    def width(self) -> int:
        return self.expert_width

    def init_parameters(self, init_std: float) -> None:
        """Normal init, with the residual projections scaled by depth."""
        residual_std = init_std / math.sqrt(2.0 * len(self.layers))
        for layer in self.layers:
            for linear in (
                layer.q_proj,
                layer.k_proj,
                layer.v_proj,
                layer.mlp.gate_proj,
                layer.mlp.up_proj,
            ):
                nn.init.normal_(linear.weight, mean=0.0, std=init_std)
            for linear in (layer.o_proj, layer.mlp.down_proj):
                nn.init.normal_(linear.weight, mean=0.0, std=residual_std)

    @torch.no_grad()
    def calibrate_qk_norm_gains(self, gains: list[tuple[torch.Tensor, ...]]) -> None:
        """Copy the backbone's per-layer query and key norm gains into the map.

        Takes the full per-backbone-layer list and selects through layer_map,
        so expert layer j is calibrated for the keys it actually attends.
        Raises on a list of any other length: a pre-selected list is ambiguous
        when the map is the identity.
        """
        if len(gains) != self.backbone_num_layers:
            raise ValueError(
                f"got gains for {len(gains)} layers, expected all "
                f"{self.backbone_num_layers}; pass backbone.qk_norm_gains() unsliced"
            )
        for layer, index in zip(self.layers, self.layer_map, strict=True):
            gain = gains[index]
            if len(gain) != 4:
                raise ValueError(
                    f"gain tuple {index} has {len(gain)} entries, expected 4 "
                    "(q_norm, q_norm_hw, k_norm, k_norm_hw)"
                )
            norms = (layer.q_norm, layer.q_norm_hw, layer.k_norm, layer.k_norm_hw)
            for norm, weight in zip(norms, gain, strict=True):
                norm.weight.copy_(weight.to(norm.weight.dtype))

    def forward(
        self,
        suffix: torch.Tensor,
        prefix: PrefixContext,
        attention_bias: torch.Tensor,
    ) -> torch.Tensor:
        """suffix [B, n_suffix, width] -> [B, n_suffix, width], post-norm.

        Layer j reads prefix.key_values[layer_map[j]]. Raises when the prefix
        supplies fewer than layer_map[-1] + 1 layers, which is the property
        that matters; the backbone's full depth is not it.
        """
        needed = self.layer_map[-1] + 1
        if prefix.n_layers < needed:
            raise ValueError(
                f"prefix carries {prefix.n_layers} layers of keys and values, "
                f"expected at least {needed} to read backbone layer "
                f"{self.layer_map[-1]}; raise model.backbone_num_layers"
            )
        x = suffix
        for layer, index in zip(self.layers, self.layer_map, strict=True):
            key, value = prefix.key_values[index]
            x = layer(
                x,
                key,
                value,
                prefix.suffix_cos,
                prefix.suffix_sin,
                attention_bias,
                prefix.scaling,
            )
        return self.norm(x)
