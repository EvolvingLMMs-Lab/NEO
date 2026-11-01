"""Build the policy from cfg.model, then apply the mode as a freeze pass.

One factory, registered under the name a run puts in model.name. It resolves
the backbone geometry, builds the tower, the encoder, the tokenizers, the
expert and the decoder, and finishes with apply_trainable_mode, which is the
only place requires_grad is set.

The mode is a pass over parameters, never a change of code path: full_ft and
cotrain build the same modules, frozen_vlm builds them and then freezes the
vision-language subtrees. Freezing is never done by wrapping the encoder in
no_grad, which would also cut the gradient to the conditioning tokenizers.
"""
from __future__ import annotations

import logging

import torch
from omegaconf import DictConfig, OmegaConf

from .. import constants
# chat_pack is imported as a module and read at call time: it imports the
# backbone's image ids, so a process that imports it first runs this package's
# __init__ while chat_pack is half built, and a from-import of one of its names
# would fail there.
from ..data import chat_pack
from ..data.images import NEO_NATIVE, check_convention, resize_filter, tower_norm
from ..engine.registry import MODELS
from .backbone.neo_llm import BackboneConfig, load_backbone_weights
from .cotrain import CoTrainHead
from .decoder import FlowDecoder
from .encoder import PrefixEncoder
from .expert import ActionExpert
from .freq_loss import FreqLossConfig
from .policy import NeoVlaPolicy
from .tokenizers import SuffixTokenizers

logger = logging.getLogger(__name__)

# What model.trainable may say. cotrain sets full_ft too: it differs by the
# chat stream and the head, not by which parameters the freeze pass touches.
TRAINABLE_MODES = ("full_ft", "frozen_vlm", "frozen_tokenizers",
                   "frozen_vlm_tokenizers")

# Where the two vision-language subtrees sit in the policy's module tree.
BACKBONE = "encoder.backbone"
VISION = "encoder.vision"

# What frozen_vlm freezes. The freeze walks the module, the declaration walks
# the parameter names: two ways of writing the same partition, so the engine's
# check compares two answers rather than one answer with itself.
FROZEN_VLM_SUBTREES = (BACKBONE, VISION)

# What frozen_tokenizers freezes, on top of everything full_ft freezes. The
# banks each carry their own list of the same partition, and the two are held
# against each other below rather than trusted apart.
FROZEN_TOKENIZER_SUBTREES = ("encoder.tokenizers", "decoder.tokenizers")

# The module classes a sharded run makes into units of their own, as the dotted
# paths trainer.strategy_options.transformer_layer_cls takes. The vision tower
# is named: it holds a convolution stack, and a tower this list does not name is
# gathered with everything else the root unit holds.
FSDP_WRAP_CLASSES = (
    "neo_vla.model.backbone.neo_llm.DecoderLayer",
    "neo_vla.model.backbone.vision.VisionTower",
    "neo_vla.model.expert.ExpertLayer",
)

DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

# The dtype every parameter an optimizer writes to is held in, whatever
# model.dtype builds the rest of the tree at.
TRAINED_DTYPE = torch.float32


def resolve_dtype(name: str) -> torch.dtype:
    """model.dtype -> the dtype the parameters are built and loaded at."""
    if name not in DTYPES:
        raise ValueError(
            f"model.dtype is {name!r}.\n"
            f"Expected one of {sorted(DTYPES)}.\n"
            "Fix model.dtype."
        )
    return DTYPES[name]


def resolve_backbone_config(cfg) -> BackboneConfig:
    """cfg.model -> the geometry the backbone is built at.

    model.backbone_path names a checkpoint directory and the geometry is read
    from its config.json, capped at model.backbone_num_layers. model.backbone
    instead states the geometry outright, which is how a small randomly
    initialised model is built on a machine with no checkpoint. Exactly one of
    the two: stating both would let a config and a checkpoint disagree
    silently, and stating neither leaves the shape undefined. The depth is
    model.backbone_num_layers in both, so arch/*.yaml is its one authority.
    """
    model = cfg.model
    path, stated = model.backbone_path, model.backbone
    if (path is None) == (stated is None):
        raise ValueError(
            f"model.backbone_path is {path!r} and model.backbone states a "
            f"geometry: {stated is not None}.\n"
            "Expected exactly one of them.\n"
            "Name a checkpoint directory, or state the geometry, not both."
        )
    depth = int(model.backbone_num_layers)
    if path is not None:
        return BackboneConfig.from_checkpoint(path, num_layers=depth)
    fields = OmegaConf.to_container(stated, resolve=True)
    if "num_layers" in fields:
        raise ValueError(
            "model.backbone states num_layers.\n"
            "Expected the depth in model.backbone_num_layers, which arch/*.yaml sets.\n"
            "Remove num_layers from model.backbone."
        )
    return BackboneConfig(num_layers=depth, **fields)


def resolve_pixel_convention(cfg) -> str:
    """cfg.model.vision.pixel_convention, which must be neo_native.

    Required rather than defaulted, and neo_native is the one convention this
    tree builds: NEO's ImageNet mean and std at the tower's entry, bicubic
    resizing. A config without the key, or naming legacy_minus1_1, describes a
    checkpoint of the old lineage, which the old tree (dcd90a5) serves.
    """
    convention = optional(cfg.model, "vision.pixel_convention")
    if convention is None:
        raise ValueError(
            "the config states no model.vision.pixel_convention.\n"
            f"Expected {NEO_NATIVE!r}, the one pixel convention this tree builds.\n"
            "Compose from config/model/neo2b.yaml, or state the key in the config."
        )
    if convention != NEO_NATIVE:
        raise ValueError(
            f"model.vision.pixel_convention is {convention!r}.\n"
            f"Expected {NEO_NATIVE!r}: the legacy [-1, 1] convention is retired, and its "
            "checkpoints are served by the tree that wrote them (dcd90a5).\n"
            f"Set model.vision.pixel_convention={NEO_NATIVE}."
        )
    return check_convention(convention, "model.vision.pixel_convention")


def build_encoder(cfg, config: BackboneConfig, dtype: torch.dtype) -> PrefixEncoder:
    """The encoder, its weights, and the reserved anchor rows."""
    model = cfg.model
    encoder = PrefixEncoder(
        config=config,
        tokenizer_path=model.tokenizer_path,
        n_views=int(model.prefix.n_views),
        n_slots=int(model.action.n_slots),
        max_grid=tuple(int(size) for size in model.prefix.max_grid),
        max_text_tokens=int(model.prefix.max_text_tokens),
        intrinsics_mode=str(model.prefix.intrinsics_mode),
        pixel_convention=resolve_pixel_convention(cfg),
        dtype=dtype,
    )
    logger.info(
        "pixel convention %s: the tower normalises %s, frames resize %s",
        encoder.pixel_convention, tower_norm(encoder.pixel_convention),
        resize_filter(encoder.pixel_convention),
    )
    if model.backbone_path is not None:
        load_backbone_weights(
            model.backbone_path, encoder.backbone, encoder.vision, dtype=dtype
        )
        logger.info("loaded the backbone and the vision tower from %s", model.backbone_path)
    if bool(model.prefix.init_anchor_rows):
        rows = encoder.tokenizers.init_anchor_rows(encoder.backbone.embed_tokens.weight)
        logger.info("redrew %d reserved anchor rows", rows)
    return encoder


def optional(node, name):
    """node.name, or None when the node has no such key.

    OmegaConf raises on a missing key under struct mode, which every Hydra
    config is, and a plain object raises AttributeError. Both mean the same
    thing here, and both mean off.
    """
    if node is None:
        return None
    if isinstance(node, DictConfig):
        return OmegaConf.select(node, name, default=None)
    if hasattr(node, "get"):
        return node.get(name, None)
    try:
        return node.__getattribute__(name)
    except AttributeError:
        return None


def build_freq_loss(action) -> FreqLossConfig:
    """model.action.dct -> the auxiliary loss config. Absent means off.

    Absent rather than required, and read field by field rather than by
    structured conversion, because a run resumed from a config.yaml written
    before this block existed must still build -- and must build with the term
    off, which is what a missing block says. The three live runs at the time
    this landed are exactly that case.

    Nothing here validates: an enabled block is checked against the chunk by
    freq_loss.resolve, inside the decoder, where the chunk is known.
    """
    block = optional(action, "dct")
    if block is None or not bool(optional(block, "enabled") or False):
        return FreqLossConfig()
    index = optional(block, "split_index")
    fraction = optional(block, "split")
    return FreqLossConfig(
        enabled=True,
        weight=float(optional(block, "weight") or 0.0),
        split_index=None if index is None else int(index),
        split=None if fraction is None else float(fraction),
        low_weight=float(optional(block, "low_weight") or 0.0),
        high_weight=float(optional(block, "high_weight") or 0.0),
        similarity=str(optional(block, "similarity") or "mse"),
    )


def action_widths(action) -> tuple[int, int]:
    """model.action.action_dim and body_dim, refused unless they are the v7 rows' (T6).

    Every width but ACTION_DIM (26) and BODY_DIM (8) is refused outright, not
    only checked against the gripper when it happens to equal one value: a
    19-wide v6 config, or a 26-wide one whose gripper sits at 6, would
    otherwise build, and the decode clamp would hold joint 6 to [-1, 1] while
    the real gripper at 13 ran to 1 + 2 alpha.
    """
    width = int(action.action_dim)
    body = optional(action, "body_dim")
    if width != constants.ACTION_DIM:
        old = " (the retired v6 width)" if width in constants.OLD_ACTION_DIMS else ""
        raise ValueError(
            f"model.action.action_dim is {width}{old}.\n"
            f"Expected {constants.ACTION_DIM}, the v7 arm row {constants.ACTION_RECORD}: "
            "the loss groups, the decode clamp and the server read its columns by name.\n"
            f"Set model.action.action_dim: {constants.ACTION_DIM}."
        )
    if body is None or int(body) != constants.BODY_DIM:
        raise ValueError(
            f"model.action.body_dim is {body!r}.\n"
            f"Expected {constants.BODY_DIM}, the v7 body row {constants.BODY_RECORD}, "
            "stated in the config beside action_dim.\n"
            f"Set model.action.body_dim: {constants.BODY_DIM}."
        )
    return width, int(body)


def gripper_columns(action) -> tuple[int, ...]:
    """model.action.gripper_dims as the decoder takes it, refused where the clamp would be wrong.

    The decode clamp holds a gripper column to [-1, 1] and every other column
    to 1 + 2 alpha (CX 05), so a gripper column the config does not name is
    clamped at the range bound in every reply, and serving builds no dataset
    whose gripper check would notice. The v7 action row (ACTION_DIM columns)
    has its gripper at GRIPPER_COLUMN and nowhere else, and any other width is
    refused (action_widths).
    """
    action_widths(action)
    columns = tuple(int(column) for column in action.gripper_dims)
    if columns != (constants.GRIPPER_COLUMN,):
        raise ValueError(
            f"model.action.gripper_dims is {list(columns)} for {int(action.action_dim)} "
            "action columns.\n"
            f"Expected [{constants.GRIPPER_COLUMN}]: the decode clamp holds a gripper "
            "column to [-1, 1] and the range columns to 1 + 2 alpha, and a column left out "
            "is clamped at the range bound (CX 05).\n"
            f"Set model.action.gripper_dims: [{constants.GRIPPER_COLUMN}]."
        )
    return columns


def build_decoder(cfg, encoder: PrefixEncoder, dtype: torch.dtype) -> FlowDecoder:
    """The action expert, the suffix tokenizers, and the flow head over them."""
    model = cfg.model
    config = encoder.config
    expert = ActionExpert(
        width=int(model.expert.width),
        intermediate=int(model.expert.intermediate),
        layer_map=tuple(int(index) for index in model.layer_map),
        backbone_num_layers=config.num_layers,
        n_heads=config.num_attention_heads,
        n_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        init_std=float(model.expert.init_std),
    ).to(dtype)
    if bool(model.expert.qk_norm_calibration):
        expert.calibrate_qk_norm_gains(encoder.backbone.qk_norm_gains())
    if optional(model.action, "include_state") is not None:
        raise ValueError(
            "model.action.include_state is set; the suffix [state] token it switched is "
            "retired.\n"
            "Expected no include_state: the state is six <state> columns of the prefix "
            "(template v7), in every build.\n"
            "Delete model.action.include_state from the config."
        )
    action_dim, body_dim = action_widths(model.action)
    gripper = gripper_columns(model.action)
    tokenizers = SuffixTokenizers(
        int(model.expert.width),
        n_slots=int(model.action.n_slots),
        init_std=float(model.expert.init_std),
    ).to(dtype)
    return FlowDecoder(
        expert,
        tokenizers,
        action_dim=action_dim,
        body_dim=body_dim,
        chunk=int(model.action.chunk),
        n_slots=int(model.action.n_slots),
        group_weights=optional(model.action, "group_weights"),
        num_sample_steps=int(model.action.num_sample_steps),
        gripper_dims=gripper,
        freq_loss=build_freq_loss(model.action),
    ).to(dtype)


def build_cotrain_head(
    cfg, config: BackboneConfig, dtype: torch.dtype
) -> tuple[CoTrainHead | None, float]:
    """The co-train head and its loss weight, or (None, 0.0) without one.

    Co-training reads the final hidden state and therefore the top layer, so a
    backbone built shallower than its checkpoint raises: that build has no top
    layer, and it cannot be co-trained after the fact either.
    """
    cotrain = cfg.model.cotrain
    if cotrain is None:
        return None, 0.0
    if config.is_truncated:
        raise ValueError(
            f"model.cotrain is set on a backbone built {config.num_layers} of "
            f"{config.checkpoint_num_layers} layers.\n"
            "Expected the checkpoint's full depth: co-training reads the final "
            "hidden state.\n"
            "Raise model.backbone_num_layers, or train this arm without co-training."
        )
    head = CoTrainHead(
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
        checkpoint_path=cfg.model.backbone_path,
        dtype=dtype,
    )
    return head, float(cotrain.weight)


def check_tokenizer_subtrees(policy: NeoVlaPolicy) -> None:
    """Hold each conditioning bank against its own frozen_after_pretrain_names.

    Each bank writes out what it holds rather than reading it off its children,
    so a parameter added to one and not to that list is a parameter a freeze
    would leave training, and a bank dropped from the list while its parameters
    stay is one nothing would ever pin. Comparing the two lists turns either
    into a raise, and the order is compared too: the freeze reads names.

    Runs under every mode, not only the two that freeze the banks. That list is
    the tree's one written record of what a bank holds, so checking it only
    where it is read would let a pretrain build a bank the finetune that pins
    it cannot name.
    """
    named = {
        "encoder.tokenizers": policy.encoder.tokenizers,
        "decoder.tokenizers": policy.decoder.tokenizers,
    }
    for name, bank in named.items():
        declared = bank.frozen_after_pretrain_names(name)
        held = [f"{name}.{leaf}" for leaf, _ in bank.named_parameters()]
        if declared != held:
            difference = sorted(set(declared) ^ set(held))
            raise ValueError(
                f"{name} names {len(declared)} frozen parameters and holds "
                f"{len(held)}; {difference[:2] or 'only the order'} differs.\n"
                "Expected the bank's own list to cover everything in it, in order.\n"
                "Fix frozen_after_pretrain_names, or the bank list it reads."
            )


def apply_trainable_mode(policy: NeoVlaPolicy, trainable: str) -> list[str]:
    """Freeze what the mode freezes; return the names it declares trainable.

    frozen_vlm calls requires_grad_(False) on the backbone and the vision
    tower, which covers the decoder layers, the embedding table and the final
    norm; the conditioning tokenizers, the expert and the flow head keep
    training, and gradient still reaches the tokenizers because nothing here
    touches the graph. The prefix's per-view geometry banks sit under
    encoder.tokenizers and train with the rest of it.

    full_ft freezes what no objective in the build can reach: without a co-train
    head nothing reads the backbone above the top layer's keys and values, so
    those parameters would sit in the optimizer, in the moving average and in
    the gradient reduction while never receiving a gradient.

    frozen_vlm_tokenizers freezes both: the vision-language subtrees and the
    conditioning banks, so only the action expert trains. It is the mode for a
    real-robot finetune on a few episodes, where moving either the backbone or
    the banks on that little data is the risk being avoided.

    frozen_tokenizers is full_ft with the two conditioning banks pinned. It is
    the mode for a stage that unfreezes a backbone a pretrain fitted those
    banks against. The freeze names the two subtrees, and check_tokenizer_subtrees
    holds each bank's own parameter list against what it holds, so the two ways
    of writing the partition are compared rather than trusted apart.

    Runs once, in build_policy, after the whole module tree exists and before
    anything reads requires_grad. Raises on an unknown mode, and raises when
    the pass would leave nothing trainable.
    """
    if trainable not in TRAINABLE_MODES:
        raise ValueError(
            f"model.trainable is {trainable!r}.\n"
            f"Expected one of {list(TRAINABLE_MODES)}.\n"
            "Fix model.trainable, or choose a config from config/mode."
        )
    check_tokenizer_subtrees(policy)
    if trainable in ("frozen_vlm", "frozen_vlm_tokenizers"):
        frozen = list(FROZEN_VLM_SUBTREES)
    elif policy.cotrain_head is None:
        backbone = policy.encoder.backbone
        frozen = [f"{BACKBONE}.{name}" for name in backbone.unread_module_names()]
    else:
        frozen = []
    if trainable in ("frozen_tokenizers", "frozen_vlm_tokenizers"):
        frozen += list(FROZEN_TOKENIZER_SUBTREES)
    for name in frozen:
        policy.get_submodule(name).requires_grad_(False)
    prefixes = tuple(f"{name}." for name in frozen)
    declared = [
        name for name, _ in policy.named_parameters() if not name.startswith(prefixes)
    ]
    if not declared:
        raise ValueError(
            f"model.trainable {trainable!r} leaves nothing to train.\n"
            "Expected at least one parameter outside the frozen subtrees.\n"
            "Check the freeze list against the module tree."
        )
    return declared


def build_dtype_subtrees(policy: NeoVlaPolicy) -> tuple[str, ...]:
    """Of the two vision-language subtrees, the ones with nothing to train.

    These are the only two places in the tree the rest of the model reads
    across a cast it makes itself: the encoder writes its tokens into the
    backbone's embedding dtype, and the expert casts the prefix keys and
    values to its queries'. A subtree frozen end to end can therefore stay at
    model.dtype while everything around it is wider, and no matmul is ever
    handed two dtypes. A mode that trains any part of one gets the whole of it
    widened instead, because the parts it does not train are read, without a
    cast, by the parts it does.
    """
    return tuple(
        name
        for name in FROZEN_VLM_SUBTREES
        if not any(
            parameter.requires_grad
            for parameter in policy.get_submodule(name).parameters()
        )
    )


def promote_trained_subtrees(policy: NeoVlaPolicy) -> tuple[int, int]:
    """Hold everything but those subtrees in float32. Elements, bytes a rank.

    model.dtype is the dtype the weights are read from the checkpoint at and
    the frozen trunk runs at, and for that trunk bfloat16 is the right answer:
    float32 there is gigabytes a rank for weights that never take a gradient.
    It is the wrong answer for anything AdamW writes back to. Those same
    tensors are what AdamW keeps its two moments in, and at bfloat16 the gap
    between 1.0 and the next value there is stands at 7.8e-3, far above any
    rate a pretrain reaches: the update is added and rounds straight back to
    the weight it started from, so tensors near unit scale, the norm gains
    above all, never move for the whole run.

    Autocast still runs the matmuls in bfloat16, so what this buys is
    resolution in the update rather than precision in the forward pass. The
    bytes counted are what a rank pays over the build dtype: two more per
    element for the parameter, and six more again for the gradient and the two
    moments of the ones that train.
    """
    kept = tuple(f"{name}." for name in build_dtype_subtrees(policy))
    elements, extra = 0, 0
    for name, parameter in policy.named_parameters():
        if name.startswith(kept) or parameter.dtype is TRAINED_DTYPE:
            continue
        parameter.data = parameter.data.to(TRAINED_DTYPE)
        elements += parameter.numel()
        extra += parameter.numel() * (8 if parameter.requires_grad else 2)
    return elements, extra


@MODELS.register("neo_vla")
def build_policy(cfg, *, for_training: bool = True) -> NeoVlaPolicy:
    """cfg -> a policy with its mode already applied.

    Reads model.layer_map and model.backbone_num_layers for the architecture,
    model.trainable for the mode, and model.cotrain for the co-training head.
    Without a head the build must stop at the top attach point: nothing reads a
    layer above it, so building one is memory the run cannot train.

    for_training decides whether promote_trained_subtrees runs. The wider dtype
    is there for what AdamW writes back to, so a build with no optimizer behind
    it keeps every parameter at model.dtype and pays neither the bytes nor the
    float32 matmuls. Logs one line per subsystem, no more.
    """
    model = cfg.model
    # The packed chat half's memory relief,
    # off by default: model.chat_offload_checkpoints and model.chat_mlp_chunk.
    # Neither changes a number the loss sees, and only a chat pack reads them.
    relief = chat_pack.relief_flags(
        {name: optional(model, name) for name in chat_pack.RELIEF_FLAGS})
    config = resolve_backbone_config(cfg)
    dtype = resolve_dtype(str(model.dtype))
    encoder = build_encoder(cfg, config, dtype)
    encoder.chat_relief = relief
    decoder = build_decoder(cfg, encoder, dtype)
    layer_map = decoder.expert.layer_map

    # What lets the freeze pass read the top built layer as the top attached
    # one, which is where the keys and values stop being read.
    if model.cotrain is None and layer_map[-1] + 1 != config.num_layers:
        raise ValueError(
            f"model.layer_map ends at layer {layer_map[-1]} of a "
            f"{config.num_layers}-layer build.\n"
            "Expected the build to end there: without co-training nothing reads a "
            "layer above the top attach point.\n"
            f"Set model.backbone_num_layers to {layer_map[-1] + 1}."
        )
    head, weight = build_cotrain_head(cfg, config, dtype)
    policy = NeoVlaPolicy(
        encoder=encoder, decoder=decoder, cotrain_head=head, cotrain_weight=weight
    )
    declared = apply_trainable_mode(policy, str(model.trainable))
    widened, extra = promote_trained_subtrees(policy) if for_training else (0, 0)
    policy.declare_trainable(declared)
    logger.info(
        "expert on %d of %d backbone layers, %d..%d",
        len(layer_map), config.num_layers, layer_map[0], layer_map[-1],
    )
    logger.info(
        "mode %s: %d of %d parameters train%s",
        model.trainable,
        len(declared),
        len(list(policy.named_parameters())),
        "" if head is None else f", co-train weight {weight}",
    )
    if widened:
        logger.info(
            "float32 for %d elements, %.2f GiB a rank above %s",
            widened, extra / 2 ** 30, model.dtype,
        )
    if any(relief.values()):
        logger.info(
            "chat pack relief: %s",
            ", ".join(f"{name} {'on' if on else 'off'}" for name, on in relief.items()),
        )
    return policy
