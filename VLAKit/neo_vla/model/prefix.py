"""PrefixContext: what the encoder hands the action expert for one batch.

It carries the backbone's per-layer keys and values, the mask that says which
prefix columns are real, and the suffix's rotary geometry: the anchor each
sample's suffix starts from, the position vector the suffix repeats, and the
cos and sin tables built from the two.

suffix_pos is one-dimensional on purpose. It is the layout, and a layout must
not depend on which other samples share the batch, or serving at batch size one
would not reproduce training. The anchor is what varies per sample.

Nothing here detaches. Whether the keys and values carry a graph is decided by
which parameters the mode froze.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .backbone.neo_llm import LayerKeyValue


@dataclass(frozen=True)
class PrefixContext:
    """One batch's prefix, valid for the step it was built in.

    key_values   one (key, value) per built backbone layer, each
                 [B, n_kv_heads, Lp, head_dim], post-rope.
    pad_mask     [B, Lp] bool, True where the prefix token is real.
    anchor       [B] long, the rotary position of each row's last real prefix
                 token. The suffix continues from anchor + 1.
    suffix_pos   [n_suffix] long, offsets from the anchor. Under n_slots arms
                 of block length L this is arange(L).repeat(n_slots), so the
                 first L entries repeat, and the arms are told apart by
                 content rather than by position.
    suffix_cos   [B, n_suffix, head_dim // 2] float32, built as
    suffix_sin   anchor[:, None] + 1 + suffix_pos[None, :]. Only the t half of
                 the head rotates for suffix tokens: their h and w indexes are
                 zero, so those rotations are the identity.
    scaling      the softmax scale the backbone uses, head_dim ** -0.5.
    """

    key_values: list[LayerKeyValue]
    pad_mask: torch.Tensor
    anchor: torch.Tensor
    suffix_pos: torch.Tensor
    suffix_cos: torch.Tensor
    suffix_sin: torch.Tensor
    scaling: float
    n_heads: int
    n_kv_heads: int
    head_dim: int

    @property
    def batch_size(self) -> int:
        return int(self.pad_mask.shape[0])

    @property
    def prefix_len(self) -> int:
        """Lp, the number of prefix columns, real and padded alike."""
        return int(self.pad_mask.shape[1])

    @property
    def n_layers(self) -> int:
        """How many layers of keys and values this prefix supplies.

        len(key_values), never a number carried alongside it: an expert that
        trusted a separate count would read off the end of a shallow build.
        """
        return len(self.key_values)

    @property
    def n_suffix(self) -> int:
        """The suffix length this prefix was built for, suffix_pos.numel()."""
        return int(self.suffix_pos.numel())
