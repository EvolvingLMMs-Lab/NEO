"""The NEO backbone: token embeddings, decoder layers, keys and values.

Vendored from the checkpoint's own modeling code, Qwen3 lineage. Each head's
queries and keys split [t:64 | h:32 | w:32], rope_theta 5e6 on t and 1e4 on h
and w, with a separate query and key norm per half. The forward takes
embeddings, an additive attention bias and a rope table, and returns one
(key, value) pair per layer, post-rope, which is what the action expert reads.

No cache, no generation, no language head. The head belongs to co-training and
lives in neo_vla/model/cotrain.py. The final hidden state is returned on
request rather than stashed on the module. The mask is an additive bias read by
SDPA, or, for the packed chat half only, a BlockMask read by ChatKernel.
"""
from __future__ import annotations

import json
import logging
import re
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from torch.autograd.graph import save_on_cpu
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from torch.utils.checkpoint import checkpoint as recompute

logger = logging.getLogger(__name__)

# (cos_t, sin_t, cos_h, sin_h, cos_w, sin_w); the t tables are [B, S, head_dim
# // 2] and the h and w tables [B, S, head_dim // 4].
RopeTables = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]

# One (key, value) pair, both [B, n_kv_heads, S, head_dim], post-rope.
LayerKeyValue = tuple[torch.Tensor, torch.Tensor]

# One decoder layer's checkpoint tensors, capturing the layer index.
LAYER_TENSOR = re.compile(r"^language_model\.model\.layers\.(\d+)\.")

BACKBONE_PREFIX = "language_model.model."
VISION_PREFIX = "vision_model."
LANGUAGE_HEAD_PREFIX = "language_model.lm_head"

# The checkpoint's own image tokens. An <IMG_CONTEXT> row is overwritten with a
# vision feature on every forward, so its embedding is never read. They sit
# with the checkpoint's other vocabulary facts because the robot prefix and the
# chat renderer both lay them down and have to spell the same picture.
IMG_CONTEXT_ID = 151669
IMG_START_ID = 151670
IMG_END_ID = 151671

# How many graphs the compiled chat kernel may build in a process: one, at the
# pack's fixed shape. A second is a guard that failed, and past dynamo's cache
# limit flex_attention runs eagerly and materialises every score: 16 heads x
# 36,864^2 x 4 bytes = 81 GiB a layer.
CHAT_KERNEL_FRAMES = 1
# The chat kernel's one Triton configuration a direction, the same in every
# process (ChatKernel.count): forward (BLOCK_M, BLOCK_N, num_warps,
# num_stages); backward (BLOCK1, BLOCK2, num_warps, num_stages), which torch
# 2.6's lowering reads as BLOCK_M1 = BLOCK_N2 = BLOCK1, BLOCK_N1 = BLOCK_M2 =
# BLOCK2. Under max_autotune every process timed its own candidates, and near
# ties went another way in another process (a resumed run, another rank,
# another node): 8 fresh processes on one node did not all pick one forward
# tile, and computed q's and k's gradients in two bit patterns (the settings
# below say why), so the steps after a resumed 8-rank run drifted. These
# are max_autotune's winners on an H100 at the production pack (36,864 tokens,
# 16 and 8 heads of 128, bf16). The backward is 4-warp: the same lowering
# skips every 8-warp backward (pytorch issue 141603). They hold for
# CHAT_KERNEL_DTYPE, the autocast dtype the trainer runs the kernel at; q at
# another dtype keeps torch's own default (float32's is 4-warp both ways), one
# config, untimed too.
CHAT_KERNEL_FORWARD = (64, 64, 4, 3)
CHAT_KERNEL_BACKWARD = (32, 64, 4, 4)
CHAT_KERNEL_DTYPE = torch.bfloat16
# The inductor settings the chat kernel compiles under, whatever the
# environment says (TORCHINDUCTOR_MAX_AUTOTUNE and the like). No timing anywhere
# in its graphs: max_autotune also hands a reduction six configurations and
# autotune_pointwise several, timed at the kernel's first call in each process
# (the backward's delta, the sum of out x grad_out over the head, took XBLOCK 8
# in some processes and 32 in others, and q's and k's gradients two bit
# patterns), and coordinate descent tunes at run time too. Reductions looped,
# not persistent: untimed, a persistent delta takes its first config, XBLOCK 1,
# 0.36 ms a backward against the timed 0.10 ms, where the looped one has a
# single config (XBLOCK 16) at the timed speed. No graph cache: its key does
# not hold the pins above, so a node's cache could serve a kernel compiled at
# others.
CHAT_KERNEL_INDUCTOR = {
    "max_autotune": False,
    "max_autotune_pointwise": False,
    "coordinate_descent_tuning": False,
    "triton.autotune_pointwise": False,
    "triton.persistent_reductions": False,
    "fx_graph_cache": False,
}
# Query rows the self-test checks against a float32 reference, spread over the
# pack: the reference costs heads x rows x S floats, 151 MB at 16 x 64 x 36,864.
SELF_TEST_ROWS = 64
# What the self-test allows between the kernel and the reference, by the dtype
# q, k and v reach the kernel in: bf16 rounds the probabilities before P @ V.
SELF_TEST_TOLERANCE = {torch.float32: 1e-4, torch.bfloat16: 2e-2, torch.float16: 1e-2}

# The submodules of a decoder layer that its key and value are computed from.
# Everything else in the layer only shapes the hidden state it passes upward.
KEY_VALUE_MODULES = (
    "input_layernorm",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.k_norm",
    "self_attn.k_norm_hw",
)


@dataclass(frozen=True)
class BackboneConfig:
    """The geometry of one NEO backbone.

    num_layers is how many decoder layers are built, which arch/*.yaml sets
    through model.backbone_num_layers. A build shallower than the checkpoint
    drops the top layers outright: they are never allocated, never in the
    optimizer, and their checkpoint tensors are discarded at load. A model
    built that way has no final hidden state and cannot be co-trained.

    checkpoint_num_layers is the depth the checkpoint holds, which num_layers
    may be below. It defaults to num_layers, which is what a geometry stated
    outright means: nothing above the built stack exists to drop.
    """

    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    rope_theta_hw: float
    vocab_size: int
    vision_hidden: int
    vision_patch: int
    vision_rope_theta: float
    downsample_ratio: float
    pad_token_id: int
    checkpoint_num_layers: int | None = None

    def __post_init__(self) -> None:
        if self.checkpoint_num_layers is None:
            object.__setattr__(self, "checkpoint_num_layers", self.num_layers)
        if not 1 <= self.num_layers <= self.checkpoint_num_layers:
            raise ValueError(
                f"num_layers is {self.num_layers} against a "
                f"{self.checkpoint_num_layers}-layer checkpoint.\n"
                f"Expected 1..{self.checkpoint_num_layers}.\n"
                "Fix model.backbone_num_layers."
            )

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, *, num_layers: int | None = None
    ) -> BackboneConfig:
        """Read config.json. num_layers None keeps the checkpoint's own depth.

        Raises when the checkpoint declares an attention variant this module
        does not implement (attention bias, rope scaling, sliding window), and
        when num_layers is outside 1..the checkpoint's depth.
        """
        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        language = config["llm_config"]
        vision = config["vision_config"]

        def scalar(value):
            # Several vision fields ship as one-element lists.
            return value[0] if isinstance(value, (list, tuple)) else value

        if language.get("attention_bias", False):
            raise ValueError(
                f"{path}/config.json sets attention_bias true.\n"
                "Expected false: the vendored projections carry no bias.\n"
                "Point at a checkpoint whose attention has no bias."
            )
        if language.get("rope_scaling") is not None:
            raise ValueError(
                f"{path}/config.json sets rope_scaling to "
                f"{language['rope_scaling']!r}.\n"
                "Expected null: the vendored rope has no scaling.\n"
                "Point at a checkpoint with unscaled rope."
            )
        if language.get("use_sliding_window", False):
            raise ValueError(
                f"{path}/config.json sets use_sliding_window true.\n"
                "Expected false: every layer here attends the whole prefix.\n"
                "Point at a checkpoint without a sliding window."
            )

        depth = int(language["num_hidden_layers"])
        built = depth if num_layers is None else int(num_layers)
        if not 1 <= built <= depth:
            raise ValueError(
                f"model.backbone_num_layers is {num_layers!r}.\n"
                f"Expected 1..{depth}, the depth {path} holds.\n"
                "Lower backbone_num_layers, or load a deeper checkpoint."
            )
        hidden = int(language["hidden_size"])
        return cls(
            hidden_size=hidden,
            num_layers=built,
            num_attention_heads=int(language["num_attention_heads"]),
            num_key_value_heads=int(language["num_key_value_heads"]),
            head_dim=int(
                language.get("head_dim", hidden // int(language["num_attention_heads"]))
            ),
            intermediate_size=int(language["intermediate_size"]),
            rms_norm_eps=float(language.get("rms_norm_eps", 1e-6)),
            rope_theta=float(language.get("rope_theta", 5e6)),
            rope_theta_hw=float(language.get("rope_theta_hw", 1e4)),
            vocab_size=int(language["vocab_size"]),
            vision_hidden=int(scalar(vision["hidden_size"])),
            vision_patch=int(
                scalar(vision.get("patch_size", config.get("patch_size", 16)))
            ),
            vision_rope_theta=float(scalar(vision.get("rope_theta_vision", 1e4))),
            downsample_ratio=float(scalar(config.get("downsample_ratio", 0.5))),
            pad_token_id=int(config.get("pad_token_id", 151643)),
            checkpoint_num_layers=depth,
        )

    @property
    def is_truncated(self) -> bool:
        """True when fewer layers are built than the checkpoint holds."""
        return self.num_layers < self.checkpoint_num_layers


def rope_cos_sin(
    positions: torch.Tensor, dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """positions [...] -> (cos, sin) [..., dim], float32.

    dim is the rotated sub-dimension and the frequencies are cat(f, f), which
    is the convention the checkpoint's own rotary code uses.
    """
    steps = torch.arange(0, dim, 2, dtype=torch.float32, device=positions.device)
    inverse = 1.0 / (theta ** (steps / dim))
    freqs = positions.float()[..., None] * inverse
    table = torch.cat([freqs, freqs], dim=-1)
    return table.cos(), table.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap and negate the two halves of the last axis."""
    left, right = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-right, left), dim=-1)


def apply_rope_part(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotate one [B, H, S, d] sub-part with cos and sin [B, S, d].

    The rotation is part-local: rotate_half runs inside the part, not across
    the whole head, which is what the merged t/h/w split needs.
    """
    cos = cos.unsqueeze(1).to(x.dtype)
    sin = sin.unsqueeze(1).to(x.dtype)
    return (x * cos) + (rotate_half(x) * sin)


class RmsNorm(nn.Module):
    """Root mean square norm: float32 variance, gain applied after downcast."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        wide = x.float()
        wide = wide * torch.rsqrt(wide.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * wide.to(dtype)


def kernel_dtype(tensor: torch.Tensor) -> torch.dtype:
    """The dtype attention runs at: autocast's on the tensor's device, else its own.

    Under full fine-tuning the query and key leave their norms in float32 and
    the value in bf16; SDPA under autocast casts all three to bf16, and the
    chat kernel casts to the same, since flex_attention is not an autocast op.
    """
    kind = tensor.device.type
    if torch.is_autocast_enabled(kind):
        return torch.get_autocast_dtype(kind)
    return tensor.dtype


class ChatKernel:
    """flex_attention under a BlockMask: the packed chat half's attention.

    On cuda, torch.compile(flex_attention, dynamic=False, fullgraph=True),
    compiled at the first pack and not at import (the server and every tool
    import this module): NEO's trainer compiles the same function at module
    level (internlm/model/modules/mha_keepnorm.py:25-27 of the NEO++ trainer,
    mha.py:25-27 of Neo_plus_train) and calls it with enable_gqa and scale
    1/sqrt(head_dim) (:993-998 there); this is a port by hand, not an import.
    dynamic=False because the pack's shape never changes, and fullgraph so a
    graph break is an error rather than an eager fallback. frames counts the
    graphs the compiler built, through the backend it is handed; a call past
    CHAT_KERNEL_FRAMES raises. The backend compiles one Triton config a
    direction, CHAT_KERNEL_FORWARD and CHAT_KERNEL_BACKWARD, so a fresh and a
    resumed process, every rank and every node run the same kernels and the
    same sums, bit for bit. Elsewhere (the cpu tests and rehearsal)
    flex_attention runs eagerly, which materialises the scores and is exact.

    self_test runs once a process, at the first pack's own shape and mask,
    before its layers: a forward and a backward on random q, k and v, the
    forward held against a float32 reference on SELF_TEST_ROWS query rows.
    It logs chat_pack.KERNEL_LOG_LINE, which the stage-2 log check expects.

    Every compiled call runs at the CPU thread count of the first one
    (call_pinned): dynamo guards the process's global state, num_threads
    among it, and a recomputed layer calls the kernel again from the autograd
    engine's thread, whose count need not be the main thread's.
    """

    name = "flex"

    def __init__(self) -> None:
        self.compiled = None
        self.frames = 0
        self.tested_tokens: int | None = None
        self.threads: int | None = None
        self.pinned_logged = False

    def count(self, graph, example_inputs):
        """The compiler backend: inductor at the pinned configurations, counting every graph.

        torch 2.6's default backward config for bf16 at head_dim 128 on an
        H100 is (64, 128, 8 warps, 3 stages), and the same lowering skips every
        8-warp config (pytorch issue 141603), so NoValidChoicesError.
        max_autotune added the 4-warp configs but timed
        them in each process, so processes compiled different kernels. Here the
        lowering is offered CHAT_KERNEL_FORWARD and CHAT_KERNEL_BACKWARD only
        (pinned), under CHAT_KERNEL_INDUCTOR: nothing is timed, and every
        process compiles the same kernels.
        """
        from torch._inductor.compile_fx import compile_fx, compile_fx_inner

        self.frames += 1
        logger.info("chat kernel: compiling at forward %s, backward %s (BLOCK, BLOCK, warps, "
                    "stages; no autotuning)", CHAT_KERNEL_FORWARD, CHAT_KERNEL_BACKWARD)
        return compile_fx(graph, example_inputs, inner_compile=self.pinned(compile_fx_inner),
                          config_patches=CHAT_KERNEL_INDUCTOR)

    @staticmethod
    def pinned(inner):
        """inner (inductor's compile of one graph), with flex's lowering offered the pins only.

        The lowering takes its one default config from _get_default_config_fwd
        and _get_default_config_bwd of torch._inductor.kernel.flex_attention,
        read as module globals at each compile, the forward's and the
        backward's (which AOTAutograd runs later, at the first backward): both
        run inside inner, and each sees the pins there (for q at
        CHAT_KERNEL_DTYPE; torch's own default for another) and torch's own
        after. A torch without the two raises here rather than compiling unpinned.
        """
        # inductor's lowerings first: on torch 2.6 kernel/ imported alone is a
        # cycle (cpp_gemm_template, partially initialised).
        import torch._inductor.lowering  # noqa: F401
        from torch._inductor.kernel import flex_attention as lowering

        pins = {"_get_default_config_fwd": CHAT_KERNEL_FORWARD,
                "_get_default_config_bwd": CHAT_KERNEL_BACKWARD}
        missing = [name for name in pins if not callable(vars(lowering).get(name))]
        if missing:
            raise RuntimeError(
                f"torch {torch.__version__}'s flex_attention lowering has no {missing}.\n"
                "Expected torch 2.6's _get_default_config_fwd and _get_default_config_bwd, "
                "where the chat kernel pins its one config a direction.\n"
                "Find this torch's way to pin the flex configs before compiling the chat kernel."
            )

        def compile_pinned(*args, **kwargs):
            saved = {name: getattr(lowering, name) for name in pins}

            def pinned_default(name):
                def default(query):
                    if query.get_dtype() == CHAT_KERNEL_DTYPE:
                        return pins[name]
                    return saved[name](query)

                return default

            for name in pins:
                setattr(lowering, name, pinned_default(name))
            try:
                return inner(*args, **kwargs)
            finally:
                for name, function in saved.items():
                    setattr(lowering, name, function)

        return compile_pinned

    def attend(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        block_mask: BlockMask, scale: float,
    ) -> torch.Tensor:
        """q [1, H, S, d], k and v [1, H_kv, S, d] -> [1, H, S, d] at kernel_dtype.

        q, k and v are made contiguous: the value arrives as a transposed
        view, and inductor's lowering refuses non-contiguous symbolic views
        (Neo_plus_train mha.py:1002-1008 records it, and calls .contiguous()).
        """
        dtype = kernel_dtype(query)
        query, key, value = (part.to(dtype).contiguous() for part in (query, key, value))
        if not query.is_cuda:
            return flex_attention(query, key, value, block_mask=block_mask, scale=scale,
                                  enable_gqa=True)
        if self.compiled is None:
            self.compiled = torch.compile(flex_attention, dynamic=False, fullgraph=True,
                                          backend=self.count)
        out = self.call_pinned(self.compiled, query, key, value, block_mask=block_mask,
                               scale=scale, enable_gqa=True)
        self.check_frames()
        return out

    def call_pinned(self, compiled, *args, **kwargs):
        """compiled(*args, **kwargs) at the CPU thread count of the kernel's first call.

        dynamo's guards hold the process's global state, torch.get_num_threads()
        among it, and a checkpointed layer's recompute runs on the autograd
        engine's device thread, which at::init_num_threads gives its own count:
        an 8-rank stage-2 run compiled a second graph there,
        "GLOBAL_STATE changed: num_threads" (TORCH_LOGS=recompiles_verbose,
        one GPU, the same recipe). The kernel computes on the GPU, so the count
        is set to the first call's for the call and restored after it, and the
        forward and the recompute share one graph. Logged once when it acts.
        """
        threads = torch.get_num_threads()
        if self.threads is None:
            self.threads = threads
        if threads == self.threads:
            return compiled(*args, **kwargs)
        if not self.pinned_logged:
            self.pinned_logged = True
            logger.info("chat kernel: a call at %d CPU threads runs at the first call's %d "
                        "(dynamo guards num_threads)", threads, self.threads)
        torch.set_num_threads(self.threads)
        try:
            return compiled(*args, **kwargs)
        finally:
            torch.set_num_threads(threads)

    def check_frames(self) -> None:
        """Raise once the compiler has built more graphs than CHAT_KERNEL_FRAMES."""
        if self.frames > CHAT_KERNEL_FRAMES:
            raise RuntimeError(
                f"the chat kernel has compiled {self.frames} graphs in this process.\n"
                f"Expected {CHAT_KERNEL_FRAMES}: the pack's shape is fixed, so another graph "
                "is a guard that failed, and past dynamo's cache limit flex_attention runs "
                "eagerly and materialises every score (81 GiB a layer at 36,864).\n"
                "Find the failing guard with TORCH_LOGS=recompiles on this torch."
            )

    def self_test(self, block_mask: BlockMask, config: BackboneConfig,
                  device: torch.device) -> None:
        """Forward and backward once at the pack's shape and mask; raise unless both hold."""
        length = int(block_mask.seq_lengths[0])
        generator = torch.Generator(device=device).manual_seed(0)
        heads, kv_heads, width = (config.num_attention_heads, config.num_key_value_heads,
                                  config.head_dim)
        scale = width**-0.5
        query, key, value = (
            torch.randn((1, count, length, width), generator=generator, device=device)
            for count in (heads, kv_heads, kv_heads))
        dtype = kernel_dtype(query)
        query, key, value = (part.to(dtype).requires_grad_() for part in (query, key, value))
        with torch.enable_grad():
            out = self.attend(query, key, value, block_mask, scale)
            grads = torch.autograd.grad(out.float().square().sum(), (query, key, value))
        rows = torch.linspace(0, length - 1, SELF_TEST_ROWS, device=device).long()
        columns = torch.arange(length, device=device)
        # Batch and head as tensors: and_masks and or_masks build on them.
        zero = torch.zeros((), dtype=torch.long, device=device)
        allowed = block_mask.mask_mod(zero, zero, rows[:, None], columns[None, :])
        # The reference in float32 even inside the trainer's autocast.
        with torch.autocast(device.type, enabled=False):
            keys = key.detach().float().repeat_interleave(heads // kv_heads, dim=1)
            values = value.detach().float().repeat_interleave(heads // kv_heads, dim=1)
            scores = query.detach().float()[:, :, rows] @ keys.transpose(-1, -2) * scale
            scores = scores.masked_fill(~allowed, float("-inf"))
            expected = scores.softmax(-1) @ values
        error = float((out.detach().float()[:, :, rows] - expected).abs().max())
        finite = bool(torch.isfinite(out).all()) and all(
            bool(torch.isfinite(grad).all()) for grad in grads)
        tolerance = SELF_TEST_TOLERANCE.get(dtype, 2e-2)
        if not finite or not error <= tolerance or not bool(grads[0].abs().sum() > 0):
            raise RuntimeError(
                f"the chat kernel's self-test at {length} tokens: forward error {error:.3g} "
                f"against float32 on {SELF_TEST_ROWS} rows, finite {finite}.\n"
                f"Expected an error at most {tolerance} at {dtype} and a finite, nonzero "
                "backward.\n"
                "Check this node's torch and GPU; the alternative is two flash varlen calls."
            )
        if device.type == "cuda" and self.frames != CHAT_KERNEL_FRAMES:
            raise RuntimeError(
                f"the chat kernel built {self.frames} graphs in its self-test.\n"
                f"Expected {CHAT_KERNEL_FRAMES}.\n"
                "Find the failing guard with TORCH_LOGS=recompiles on this torch."
            )
        from ...data.chat_pack import KERNEL_LOG_LINE

        self.tested_tokens = length
        logger.info(KERNEL_LOG_LINE.format(kernel=self.name, tokens=length))


# The process's one chat kernel: one compiled graph, one self-test.
CHAT_KERNEL = ChatKernel()


class MergedAttention(nn.Module):
    """One query and key projection, split per head into [t:64 | h:32 | w:32].

    The t half is normed by q_norm and k_norm and rotated at rope_theta; the h
    and w quarters are normed by q_norm_hw and k_norm_hw and rotated at
    rope_theta_hw. Returns the attention output and the post-rope key and
    value, because the action expert attends into those.
    """

    def __init__(self, config: BackboneConfig):
        super().__init__()
        hidden = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, hidden, bias=False)
        half = self.head_dim // 2
        self.q_norm = RmsNorm(half, config.rms_norm_eps)
        self.q_norm_hw = RmsNorm(half, config.rms_norm_eps)
        self.k_norm = RmsNorm(half, config.rms_norm_eps)
        self.k_norm_hw = RmsNorm(half, config.rms_norm_eps)
        # compiled_layers.compile_robot(eager_numerics=True) sets it: the additive-bias
        # branch's query leaves project_and_rotate at the attention's dtype. A plain
        # attribute, no state; the bits the attention reads are the same either way.
        self.cast_query = False

    def project_and_rotate(
        self,
        x: torch.Tensor,
        proj: nn.Linear,
        norm_t: RmsNorm,
        norm_hw: RmsNorm,
        n_heads: int,
        rope: RopeTables,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """x [B, S, hidden] -> [B, n_heads, S, head_dim], normed and rotated.

        dtype casts the three rotated parts before they are joined: the same values as
        casting the joined tensor (a cast is elementwise), so the attention reads the same
        bits; a compiled robot layer that keeps its attention for the backward then keeps
        the query at that dtype instead of float32.
        """
        batch, length, _ = x.shape
        cos_t, sin_t, cos_h, sin_h, cos_w, sin_w = rope
        time, space = proj(x).view(batch, length, n_heads, self.head_dim).chunk(2, dim=-1)
        time = norm_t(time).transpose(1, 2)
        height, width = norm_hw(space).transpose(1, 2).chunk(2, dim=-1)
        time = apply_rope_part(time, cos_t, sin_t)
        height = apply_rope_part(height, cos_h, sin_h)
        width = apply_rope_part(width, cos_w, sin_w)
        if dtype is not None:
            time, height, width = time.to(dtype), height.to(dtype), width.to(dtype)
        return torch.cat([time, height, width], dim=-1)

    def forward(
        self, x: torch.Tensor, attention_bias: torch.Tensor | BlockMask, rope: RopeTables
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x [B, S, hidden] -> (output [B, S, hidden], key, value).

        attention_bias is additive [B, 1, S, S], read by SDPA, or a chat
        pack's BlockMask, read by CHAT_KERNEL with the key and value heads
        shared by the kernel (enable_gqa) rather than repeated. The key and
        value are [B, n_kv_heads, S, head_dim].
        """
        batch, length, _ = x.shape
        # cast_query: the additive-bias branch's query leaves at the attention's dtype
        # (the bits SDPA reads either way); the chat pack's kernel casts its own.
        cast = self.cast_query and not isinstance(attention_bias, BlockMask)
        query = self.project_and_rotate(
            x, self.q_proj, self.q_norm, self.q_norm_hw, self.n_heads, rope,
            kernel_dtype(x) if cast else None,
        )
        key = self.project_and_rotate(
            x, self.k_proj, self.k_norm, self.k_norm_hw, self.n_kv_heads, rope
        )
        value = (
            self.v_proj(x)
            .view(batch, length, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        if isinstance(attention_bias, BlockMask):
            out = CHAT_KERNEL.attend(query, key, value, attention_bias, self.scaling)
            out = out.transpose(1, 2).reshape(batch, length, -1)
            return self.o_proj(out), key, value
        repeat = self.n_heads // self.n_kv_heads
        keys = key.repeat_interleave(repeat, dim=1) if repeat > 1 else key
        values = value.repeat_interleave(repeat, dim=1) if repeat > 1 else value
        # The bias arrives in float32 and carries -inf; clamping to the compute
        # dtype's minimum keeps a fully masked row finite instead of nan. The key's
        # dtype is the query's before its cast (the same norms and rotations).
        bias = attention_bias.to(key.dtype).clamp(min=torch.finfo(key.dtype).min)
        out = F.scaled_dot_product_attention(
            query, keys, values, attn_mask=bias, scale=self.scaling
        )
        out = out.transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(out), key, value


class FeedForward(nn.Module):
    """Gated feed-forward: down(silu(gate(x)) * up(x))."""

    def __init__(self, config: BackboneConfig):
        super().__init__()
        hidden, inner = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(hidden, inner, bias=False)
        self.up_proj = nn.Linear(hidden, inner, bias=False)
        self.down_proj = nn.Linear(inner, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """Pre-norm attention and feed-forward, both residual."""

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.self_attn = MergedAttention(config)
        self.mlp = FeedForward(config)
        self.input_layernorm = RmsNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RmsNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self, x: torch.Tensor, attention_bias: torch.Tensor | BlockMask, rope: RopeTables,
        mlp_chunk: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x [B, S, hidden] -> (x, key, value) with the same shapes.

        mlp_chunk, the chat pack's relief flag (model.chat_mlp_chunk), runs the
        feed-forward over that many positions at a time, each recomputed in
        the backward: per position, so the same sums, equal up to the matmul's
        rounding (gradients measured within 1e-10 of none), and the
        layer's backward holds one chunk's gate and up activations instead of
        the pack's (1.69 GiB at 36,864 x 6,144 in bf16, packing_spec 5).
        """
        attention, key, value = self.self_attn(
            self.input_layernorm(x), attention_bias, rope
        )
        x = x + attention
        if mlp_chunk is None:
            x = x + self.mlp(self.post_attention_layernorm(x))
        else:
            x = x + self.chunked_mlp(x, int(mlp_chunk))
        return x, key, value

    def mlp_block(self, x: torch.Tensor) -> torch.Tensor:
        """The feed-forward half without its residual: mlp(post_attention_layernorm(x))."""
        return self.mlp(self.post_attention_layernorm(x))

    def chunked_mlp(self, x: torch.Tensor, chunk: int) -> torch.Tensor:
        """mlp_block over `chunk` positions at a time, each recomputed when a graph is built."""
        parts = []
        for part in x.split(chunk, dim=1):
            if torch.is_grad_enabled():
                parts.append(recompute(self.mlp_block, part, use_reentrant=False))
            else:
                parts.append(self.mlp_block(part))
        return torch.cat(parts, dim=1)


class NeoBackbone(nn.Module):
    """embed_tokens, num_layers decoder layers, and the final norm.

    Parameter names mirror the checkpoint's language_model.model.*, so loading
    is a name-mapped copy.
    """

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(config) for _ in range(config.num_layers)
        )
        self.norm = RmsNorm(config.hidden_size, config.rms_norm_eps)
        # A memory setting rather than part of the layout, so it is a plain
        # attribute the trainer sets: co-training runs this stack at chat
        # length and keeps every layer's activations, which is what needs it.
        self.gradient_checkpointing = False
        # The robot path's compiled layers (compiled_layers.RobotStack), set by
        # compiled_layers.compile_robot for training (trainer.compile_robot); None
        # runs every layer eager. A plain attribute: no parameter, no state.
        self.robot_stack = None
        if config.is_truncated:
            logger.info(
                "backbone built %d of the checkpoint's %d layers; %d..%d are dropped, "
                "so there is no final hidden state and no co-training later",
                config.num_layers,
                config.checkpoint_num_layers,
                config.num_layers,
                config.checkpoint_num_layers - 1,
            )

    def rope_tables(
        self, t_index: torch.Tensor, h_index: torch.Tensor, w_index: torch.Tensor
    ) -> RopeTables:
        """Three [B, S] index rows -> the six cos and sin tables, float32."""
        shapes = {tuple(index.shape) for index in (t_index, h_index, w_index)}
        if len(shapes) != 1 or t_index.dim() != 2:
            raise ValueError(
                f"rope index rows are {sorted(shapes)}.\n"
                "Expected one shared [batch, sequence] shape.\n"
                "Build t, h and w from the same identifier row."
            )
        head_dim = self.config.head_dim
        cos_t, sin_t = rope_cos_sin(t_index, head_dim // 2, self.config.rope_theta)
        cos_h, sin_h = rope_cos_sin(h_index, head_dim // 4, self.config.rope_theta_hw)
        cos_w, sin_w = rope_cos_sin(w_index, head_dim // 4, self.config.rope_theta_hw)
        return (cos_t, sin_t, cos_h, sin_h, cos_w, sin_w)

    def forward(
        self,
        embeds: torch.Tensor,
        attention_bias: torch.Tensor | BlockMask,
        rope: RopeTables,
        *,
        keep_last_hidden: bool,
        keep_key_values: bool = True,
        offload_checkpoints: bool = False,
        mlp_chunk: int | None = None,
        recompute_layers: int | None = None,
    ) -> tuple[list[LayerKeyValue], torch.Tensor | None]:
        """embeds [B, S, hidden] -> (one (key, value) per layer, hidden or None).

        keep_last_hidden is a call argument and not a stored flag: only
        co-training reads the final hidden state, and a flag left armed on a
        shared module would norm a hidden state the caller never asked for.
        It raises when the build is shallower than the checkpoint, because
        there is then no top layer and the returned tensor would be a
        well-formed value computed at the wrong depth. keep_key_values False
        returns no keys and values, so a chat row that reads only the hidden
        state does not hold every layer's pair until the stack returns.

        A BlockMask (the chat pack) runs CHAT_KERNEL, whose self-test runs
        here, outside the layers, at the first pack. offload_checkpoints and
        mlp_chunk are the pack's memory relief (model.chat_offload_checkpoints,
        model.chat_mlp_chunk), off unless the encoder passes them:
        offload_checkpoints puts what the recomputed layers keep for the
        backward, their inputs, in pinned host memory (save_on_cpu around the
        layer loop: a non-reentrant checkpoint saves its inputs through the
        hooks active outside it), and so needs the layers recomputed.

        recompute_layers, under gradient_checkpointing, recomputes the first
        that many layers and keeps the rest's activations
        (trainer.robot_recompute_layers, the policy passes it for a robot-only
        micro-batch); None recomputes them all. It changes what is held for
        the backward, never a number. A robot prefix (an additive bias, no
        final hidden state) runs through robot_stack when compile_robot set
        one; the chat pack (a BlockMask) and chat rows (keep_last_hidden) never.
        """
        if keep_last_hidden and self.config.is_truncated:
            raise RuntimeError(
                f"keep_last_hidden on a backbone built {self.config.num_layers} of "
                f"{self.config.checkpoint_num_layers} layers.\n"
                "Expected the full depth: layer "
                f"{self.config.checkpoint_num_layers - 1} does not exist here.\n"
                "Raise model.backbone_num_layers to the checkpoint's depth."
            )
        x = embeds
        key_values: list[LayerKeyValue] = []
        depth = len(self.layers)
        if recompute_layers is not None and not (
                isinstance(recompute_layers, int) and not isinstance(recompute_layers, bool)
                and 0 <= recompute_layers <= depth):
            raise ValueError(
                f"recompute_layers is {recompute_layers!r}.\n"
                f"Expected None (every layer) or a count from 0 to {depth}.\n"
                "Fix trainer.robot_recompute_layers."
            )
        recomputed = depth if recompute_layers is None else int(recompute_layers)
        # Recomputation only pays when there is a graph to trade it against.
        recompute_on = self.gradient_checkpointing and torch.is_grad_enabled()
        robot = (self.robot_stack is not None and not keep_last_hidden
                 and not isinstance(attention_bias, BlockMask))
        if offload_checkpoints and torch.is_grad_enabled() and not recompute_on:
            raise ValueError(
                "model.chat_offload_checkpoints is on and the layers are not recomputed.\n"
                "Expected trainer.gradient_checkpointing true: the relief moves the "
                "recomputed layers' inputs to the host; without recomputation it would move "
                "every activation.\n"
                "Turn trainer.gradient_checkpointing on, or the relief off."
            )
        if isinstance(attention_bias, BlockMask) and CHAT_KERNEL.tested_tokens is None:
            CHAT_KERNEL.self_test(attention_bias, self.config, embeds.device)
        extra = () if mlp_chunk is None else (int(mlp_chunk),)
        offload = offload_checkpoints and recompute_on
        with save_on_cpu(pin_memory=True) if offload else nullcontext():
            for index, layer in enumerate(self.layers):
                again = recompute_on and index < recomputed
                if robot:
                    # The compiled robot layer holds its own recomputation.
                    x, key, value = self.robot_stack(layer, x, attention_bias, rope, again)
                elif again:
                    # use_reentrant False keeps the autocast state across the
                    # recompute and composes with find_unused_parameters.
                    x, key, value = recompute(
                        layer, x, attention_bias, rope, *extra, use_reentrant=False
                    )
                else:
                    x, key, value = layer(x, attention_bias, rope, *extra)
                if keep_key_values:
                    key_values.append((key, value))
        return key_values, self.norm(x) if keep_last_hidden else None

    def unread_module_names(self) -> list[str]:
        """Submodules a reader of the keys and values cannot see the output of.

        Above the top layer's own key and value nothing is read: the final norm
        and that layer's query, output and feed-forward halves are computed and
        dropped. Only a caller that takes the final hidden state reaches them,
        so it is the caller that decides whether this list applies.
        """
        top = f"layers.{len(self.layers) - 1}"
        live = set()
        for path in KEY_VALUE_MODULES:
            # Resolves or raises: a renamed projection would otherwise turn the
            # whole layer unread, and the reader would freeze what it needs.
            self.get_submodule(f"{top}.{path}")
            live.add(f"{top}.{path}")
        names = ["norm"]
        for name, module in self.named_modules():
            if not name.startswith(f"{top}."):
                continue
            if any(path == name or path.startswith(f"{name}.") for path in live):
                continue
            if next(module.parameters(recurse=False), None) is not None:
                names.append(name)
        return names

    def qk_norm_gains(self) -> list[tuple[torch.Tensor, ...]]:
        """Per layer (q_norm, q_norm_hw, k_norm, k_norm_hw) gains, detached.

        The expert copies these so its queries and keys enter the joint softmax
        at the scale the backbone's keys were trained at.
        """
        return [
            (
                layer.self_attn.q_norm.weight.detach(),
                layer.self_attn.q_norm_hw.weight.detach(),
                layer.self_attn.k_norm.weight.detach(),
                layer.self_attn.k_norm_hw.weight.detach(),
            )
            for layer in self.layers
        ]


def dropped_layer_index(name: str, config: BackboneConfig) -> int | None:
    """The layer index of a checkpoint tensor this build has no room for."""
    if not config.is_truncated:
        return None
    found = LAYER_TENSOR.match(name)
    if found is None:
        return None
    index = int(found.group(1))
    if config.num_layers <= index < config.checkpoint_num_layers:
        return index
    return None


def load_backbone_weights(
    path: str | Path,
    backbone: NeoBackbone,
    vision: nn.Module,
    *,
    dtype: torch.dtype,
) -> None:
    """Copy the checkpoint into the backbone and the vision tower.

    Streams one tensor at a time, so peak host memory is one tensor and not the
    whole file. Strict in both directions: an unfilled module parameter and an
    unmatched checkpoint tensor each raise. The one allowance is a build
    shallower than the checkpoint, which drops exactly the contiguous tail
    layers num_layers..checkpoint_num_layers - 1 and nothing else. The language
    head is not read here; cotrain.CoTrainHead loads it.
    """
    config = backbone.config
    backbone_params = dict(backbone.named_parameters())
    vision_params = dict(vision.named_parameters())
    written: set[str] = set()
    unmatched: list[str] = []
    dropped: set[int] = set()

    tensors = Path(path) / "model.safetensors"
    with safe_open(str(tensors), framework="pt", device="cpu") as reader:
        for name in reader.keys():
            if name.startswith(LANGUAGE_HEAD_PREFIX):
                continue
            if name.startswith(BACKBONE_PREFIX):
                key = name[len(BACKBONE_PREFIX) :]
                destination, tag = backbone_params.get(key), f"backbone.{key}"
            elif name.startswith(VISION_PREFIX):
                key = name[len(VISION_PREFIX) :]
                destination, tag = vision_params.get(key), f"vision.{key}"
            else:
                destination, tag = None, name
            if destination is None:
                index = dropped_layer_index(name, config)
                if index is None:
                    unmatched.append(name)
                else:
                    dropped.add(index)
                continue
            tensor = reader.get_tensor(name)
            if tuple(destination.shape) != tuple(tensor.shape):
                raise RuntimeError(
                    f"{name} is {tuple(tensor.shape)} in {tensors}.\n"
                    f"Expected {tuple(destination.shape)}, the built module's shape.\n"
                    "Load a checkpoint whose geometry matches model.backbone."
                )
            with torch.no_grad():
                destination.copy_(tensor.to(dtype))
            written.add(tag)

    if unmatched:
        raise RuntimeError(
            f"{len(unmatched)} checkpoint tensors have nowhere to go, first "
            f"{unmatched[:3]}.\n"
            "Expected every tensor to name a built parameter.\n"
            "Load a checkpoint this module implements, or extend the module."
        )
    if config.is_truncated:
        expected_drop = set(range(config.num_layers, config.checkpoint_num_layers))
        if dropped != expected_drop:
            raise RuntimeError(
                f"the load dropped layers {sorted(dropped)}.\n"
                f"Expected exactly {sorted(expected_drop)}, the declared tail.\n"
                "Check the checkpoint's layer numbering before truncating it."
            )
        logger.info(
            "discarded %d checkpoint layers (%d..%d) for a %d-layer build",
            len(expected_drop),
            config.num_layers,
            config.checkpoint_num_layers - 1,
            config.num_layers,
        )
    expected = {f"backbone.{key}" for key in backbone_params}
    expected |= {f"vision.{key}" for key in vision_params}
    if written != expected:
        missing = sorted(expected - written)
        raise RuntimeError(
            f"{len(missing)} built parameters were never written, first {missing[:3]}.\n"
            f"Expected all {len(expected)} to be filled from {tensors}.\n"
            "Load a checkpoint that carries every parameter this module builds."
        )
