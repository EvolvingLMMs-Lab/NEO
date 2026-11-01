"""The policy: the one object the training loop sees.

It holds the encoder, the decoder and, under co-training, the head. Its whole
contract to the engine is forward(batch, ctx) returning a dict with "loss",
trainable_parameter_names(), and configure_optimizers(cfg).

The order inside forward is fixed and is the point of the module: the batch's
key set is held to the contract's, the arms' and the body's states are
normalised through the run's embodiments file (and perturbed in training), the
batch's slot count comes from the action tensor, the decoder states the suffix
position vector, and the encoder is handed that vector. Nothing else may decide
the suffix layout.
"""
from __future__ import annotations

import logging
import math
from contextlib import nullcontext

import torch
import torch.nn as nn

from .. import constants
# chat_pack is imported (as packing) as a module and read at call time: it imports the
# backbone's image ids, so a process that imports it first runs this package's
# __init__ while chat_pack is half built, and a from-import of one of its names
# would fail there.
from ..data import chat_pack as packing
from ..data.images import (
    CONVENTION_META_KEY,
    NEO_NATIVE,
    TILE_META_KEY,
    check_saved_convention,
)
from ..data.images import TILE_STAMP as TILE_RULE
from ..engine.checkpoint import file_digest
from ..engine.context import StepContext
from ..engine.loop import PACK_COUNTER_OUTPUT
from ..engine.hostcopy import host_of, to_host_device
from . import compiled_layers
from .deferred import deferred, require
from .cotrain import CoTrainHead, row_targets, square_weighted_loss, step_weight_mean
from .decoder import (
    OLD_SUFFIX_LAYOUTS,
    SUFFIX_LAYOUT,
    SUFFIX_LAYOUT_KEY,
    UNSTATED_SUFFIX_LAYOUT,
    FlowDecoder,
)
from .encoder import PrefixEncoder
from .prefix import PrefixContext
from .tokenizers import (
    BODY_DIM,
    BODY_NOISE_WIDTH,
    BODY_RECORD,
    NOISE_APERTURE,
    NOISE_HAND,
    NOISE_JOINTS,
    NOISE_POSITION,
    NOISE_ROTATION,
    NOISE_WIDTH,
    STATE_ABSENT,
    STATE_APERTURE,
    STATE_DIM,
    STATE_HAND,
    STATE_JOINTS,
    STATE_MEASURED,
    STATE_POSITION,
    STATE_RECORD,
    STATE_ROTATION,
    StateInputs,
    SuffixInputs,
)

logger = logging.getLogger(__name__)

# Logged once a process, at its first micro-batch with a chat pack: the order
# the two halves run in (speed C-2; NeoVlaPolicy.chat_pack_first), for the
# run's provenance, since the order is a class attribute and no config key.
CHAT_ORDER_LOG_LINE = "chat pack order: {order} (speed C-2, chat_pack_first {first})"

# The vision-language model, as named in this policy's named_parameters(): the
# backbone, the vision tower and the co-train head, which is initialised from
# NEO's lm_head. schedule.vlm_learning_rate is the rate of exactly these three
# subtrees; everything else -- the expert, the flow head and both conditioning
# banks, the tensors a frozen_vlm pretrain trains -- runs at
# schedule.learning_rate.
VLM_PREFIXES = ("encoder.backbone.", "encoder.vision.", "cotrain_head.")
# What a sharding wrap inserts into a parameter name. Stripped before the
# prefix test so the partition does not silently become empty under fsdp.
WRAP_SEGMENT = "_fsdp_wrapped_module."

# The co-train batch's chat half, collate_chat's five fields: the rows, their
# mask and labels, and the images at their own sizes with the row each belongs
# to. A robot batch has no such key.
CHAT = "chat"
CHAT_IDS = "input_ids"
CHAT_MASK = "attention_mask"
CHAT_LABELS = "labels"
CHAT_IMAGES = "images"
CHAT_IMAGE_ROWS = "image_rows"
CHAT_FIELDS = frozenset((CHAT_IDS, CHAT_MASK, CHAT_LABELS, CHAT_IMAGES, CHAT_IMAGE_ROWS))
# The legacy chat half's stacked canvas. Its format is retired with the rest of
# the legacy chat path; a half that still carries it is refused by name.
RETIRED_CHAT_FIELDS = {"pixels": "the legacy chat half's stacked [-1, 1] canvas"}
# A packed chat half carries exactly
# chat_pack.PACK_FIELDS instead: one pack of pack_tokens positions a
# micro-batch. Its counters come back from the forward under
# loop.PACK_COUNTER_OUTPUT, which the loop sums over a log window and all-reduces on
# every rank.

# Batch fields an older template retired, and what each became
# (constants.RETIRED_BATCH_KEYS). The key-set check refuses any field it does
# not expect; these are named so the refusal says why.
RETIRED_BATCH_FIELDS = dict(constants.RETIRED_BATCH_KEYS)

# The training-only fields a serving batch is refused by name for, first (R21).
NOISE_FIELDS = ("state_noise", "body_state_noise")

# What meta.json states about the state, action and body records and the
# history, the stamps a resume, an init_from and a serving load hold against
# this tree: a v6 step (10-wide state, 19-wide action, no body) is refused
# by name, before a tensor shape could.
STATE_STAMP = dict(constants.STATE_STAMP)
ACTION_STAMP = dict(constants.ACTION_STAMP)
BODY_STAMP = dict(constants.BODY_STAMP)
TILE_STAMP = dict(TILE_RULE)
HISTORY_STAMP = {
    "H": constants.HISTORY_SIZE,
    "min_gap": constants.MIN_GAP,
    "sampling": constants.SAMPLING,
    "no_history": constants.NO_HISTORY_FORM,
}
PREFIX_LAYOUT_KEY = "prefix_layout"
EMBODIMENTS_KEY = "embodiments"
STATE_KEY = "state"
ACTION_KEY = "action"
BODY_KEY = "body"
HISTORY_KEY = "history"
TILE_KEY = TILE_META_KEY
# Every stamp held exactly, in the order check_checkpoint_meta reads them.
RECORD_STAMPS = (
    (STATE_KEY, STATE_STAMP),
    (ACTION_KEY, ACTION_STAMP),
    (BODY_KEY, BODY_STAMP),
    (HISTORY_KEY, HISTORY_STAMP),
    (TILE_KEY, TILE_STAMP),
)
# The prefix geometry the encoder was built with (PrefixEncoder.layout_stamp).
# Stamped from the build, not from a constant: a test shape states its own.
PREFIX_KEY = "prefix"


_IDENTITY_6D: dict[torch.device, torch.Tensor] = {}


def identity_6d(device: torch.device) -> torch.Tensor:
    """The 6D identity rotation [1, 0, 0, 0, 1, 0] float32 on device, built once per device."""
    if device not in _IDENTITY_6D:
        _IDENTITY_6D[device] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device)
    return _IDENTITY_6D[device]


def refuse_perturbed_state() -> None:
    """perturb_state's refusal: a MEASURED entry left the clip not finite."""
    raise ValueError(
        "the perturbed state is not finite on a MEASURED entry.\n"
        "Expected the clip to leave finite numbers: a NaN in the draw or the "
        "record would reach the state bank as a measurement.\n"
        "Fix the noise draw; nothing is cleaned into a trained zero."
    )


def batch_field(batch: dict, name: str) -> torch.Tensor:
    """One required tensor out of the batch, or a raise that names it."""
    if name not in batch:
        raise ValueError(
            f"the batch carries no {name!r}.\n"
            "Expected every batch field (constants.batch_keys), validity masks included.\n"
            "Fix the collate function."
        )
    return batch[name]


def chat_targets(chat: dict) -> torch.Tensor:
    """A chat half -> the targets of each row it holds: a pack's segment_targets, else per row."""
    if set(chat) == set(packing.PACK_FIELDS):
        return chat[packing.PACK_SEGMENT_TARGETS]
    return row_targets(chat[CHAT_LABELS])


def dataset_field(dataset, name: str):
    """One attribute check_dataset reads off the training dataset, or a refusal naming it.

    An attribute, not vars(): RobotDataset states embodiments through a property.
    """
    if not hasattr(dataset, name):
        raise ValueError(
            f"the training dataset ({type(dataset).__name__}) states no {name}.\n"
            "Expected a RobotDataset, which states pixel_convention, understanding and "
            "embodiments.\nTrain through neo_vla.data.dataset.RobotDataset."
        )
    return getattr(dataset, name)


def learning_rate_factor(step: int, *, warmup: int, horizon: int, floor: float) -> float:
    """Multiplier on the base rate: linear warmup into a cosine decay to floor.

    floor is lr_final / learning_rate, so the schedule ends at the final rate
    the config states rather than at zero.
    """
    if step < warmup:
        return (step + 1) / warmup
    progress = min((step - warmup) / max(horizon - warmup, 1), 1.0)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def check_suffix_layout(meta, where: str) -> None:
    """R22 unless meta states this tree's suffix layout; a step stating none is v1."""
    stated = (meta or {}).get(SUFFIX_LAYOUT_KEY, UNSTATED_SUFFIX_LAYOUT)
    if stated == SUFFIX_LAYOUT:
        return
    tree = OLD_SUFFIX_LAYOUTS.get(stated)
    old = (f" ({stated!r} is the retired pre-token suffix, served and continued by "
           f"neo-official {tree})") if tree else ""
    unstated = "" if SUFFIX_LAYOUT_KEY in (meta or {}) else " (it states none)"
    raise ValueError(
        f"{constants.refusal_prefix(22)} {where} states suffix layout "
        f"{stated!r}{unstated}{old}.\n"
        f"Expected {SUFFIX_LAYOUT!r}: each suffix block is its K action tokens with "
        "the block's frame and identity rows added to their input, under a "
        "key-padding mask. A v1 step holds the same tensors under the same names, so "
        "a load would succeed and serve every row through the wrong layout.\n"
        "Serve or continue it with the tree that wrote it, or name a step written "
        "by this one."
    )


class NeoVlaPolicy(nn.Module):
    """The NEO backbone, the action expert, and the flow objective."""

    encoder: PrefixEncoder
    decoder: FlowDecoder
    cotrain_head: CoTrainHead | None
    # A micro-batch with a chat pack runs the pack's forward before the robot
    # rows' (speed C-2), so autograd runs the robot backward first: the
    # backbone's buckets then fill during the pack's long backward and their
    # all_reduce overlaps it, and the robot path's saved tensors are freed
    # before the pack's peak. The robot draws are unchanged (the chat path draws
    # no random number). The two halves' gradients meet in a different order,
    # so the sums agree to rounding, not bit for bit (speed_plan_final 2.2).
    # Rows that are not packed keep the robot half first. A class attribute
    # and no config key, read per micro-batch: the speed harness sets it on an
    # arm that isolates the order. CHAT_ORDER_LOG_LINE states it once.
    # Packed chat runs after the robot rows by default. Unpacked chat always
    # runs after the robot rows.
    chat_pack_first = False
    chat_order_logged = False

    def __init__(
        self,
        *,
        encoder: PrefixEncoder,
        decoder: FlowDecoder,
        cotrain_head: CoTrainHead | None = None,
        cotrain_weight: float = 0.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cotrain_head = cotrain_head
        self.cotrain_weight = float(cotrain_weight)
        if (cotrain_head is None) != (self.cotrain_weight == 0.0):
            raise ValueError(
                f"cotrain_weight is {self.cotrain_weight} with "
                f"{'no' if cotrain_head is None else 'a'} co-train head.\n"
                "Expected a positive weight with a head and zero without one.\n"
                "Set model.cotrain, or leave it null."
            )
        # Filled by declare_trainable in build_policy. None until then, so a
        # policy built any other way says so instead of declaring nothing.
        self.trainable_names: list[str] | None = None
        # The loop shows a co-train policy each step's micro-batches before the
        # first forward (prepare_step). The denominator of NEO's square loss for
        # that step, and how many native micro-batches may still use it.
        self.prepares_steps = cotrain_head is not None
        self.step_weight: torch.Tensor | None = None
        self.step_uses_left = 0
        # The micro-batches of the prepared step still to run its forward: the
        # last of them is the one whose backward DDP synchronises (loop.py
        # is_sync), where a step without any chat half adds chat_participation.
        # 0 outside a prepared step (a tool calling forward), which never adds it.
        self.step_micro_left = 0
        self.step_has_chat = True
        # trainer.robot_recompute_layers: the backbone layers a robot-only
        # micro-batch recomputes under gradient checkpointing; None, all of them.
        # The runner sets it; a micro-batch with a chat half always recomputes all.
        self.robot_recompute_layers: int | None = None
        # The chat corpus's chat_pack.PackConfig, read by check_dataset: None
        # for one chat row per robot row. The loop reads it to sum the pack
        # counters on every rank; a pack of another length is refused.
        self.chat_pack = None
        # The run's embodiments file (neo_vla.data.embodiment.Embodiments): set
        # by the runner from the data config, or by the serving loader from the
        # step directory. Every state is normalised through it and every
        # checkpoint carries it, so forward, predict and checkpoint_meta refuse
        # to run without it rather than inventing a normaliser.
        self.embodiments = None
        # Range clips (arm and body) of the last prefix_state call, for the step
        # log and the serving reply's clips counter.
        self.state_clips = 0

    def declare_trainable(self, names: list[str]) -> None:
        """Record the partition the mode declared. Called once, by the builder.

        Stored rather than derived on demand, so trainable_parameter_names
        answers the question "what did the mode say" and the engine can compare
        that with what actually requires a gradient.
        """
        known = {name for name, _ in self.named_parameters()}
        unknown = sorted(set(names) - known)
        if unknown:
            raise ValueError(
                f"{len(unknown)} declared names are not parameters here, "
                f"first {unknown[:3]}.\n"
                "Expected names from this policy's named_parameters().\n"
                "Declare the partition from the built tree."
            )
        self.trainable_names = list(names)

    def trainable_parameter_names(self) -> list[str]:
        """The names the mode declared trainable, in named_parameters order."""
        if self.trainable_names is None:
            raise RuntimeError(
                "no mode has been applied to this policy.\n"
                "Expected build.apply_trainable_mode to have run at build.\n"
                "Build through neo_vla.model.build.build_policy."
            )
        return list(self.trainable_names)

    def trim_prefix(self, bucket: int) -> None:
        """Run each prefix at its longest row rounded up to bucket columns (trainer.trim_prefix)."""
        if int(bucket) <= 0:
            raise ValueError(
                f"trim_prefix bucket {bucket}, expected a positive number of columns.\n"
                "Leave trainer.trim_prefix at 0 to keep every Lp column."
            )
        self.encoder.trim_bucket = int(bucket)

    def compile_layers(self) -> dict[str, int]:
        """The backbone's and the expert's layers through torch.compile (trainer.compile_layers)."""
        return compiled_layers.compile_layers(self)

    def compile_robot(self, backend: str = "inductor", **options):
        """The robot path's layers through torch.compile, the chat's eager (trainer.compile_robot)."""
        return compiled_layers.compile_robot(self, backend, **options)

    def chat_only_parameters(self) -> list[nn.Parameter]:
        """The trainable parameters only the chat half reaches, in named_parameters order.

        The co-train head and what the backbone computes above the top layer's
        key and value (NeoBackbone.unread_module_names: the final norm and the
        top layer's query, output and feed-forward halves): the robot prefix
        reads every layer's key and value and nothing above. Static, from the
        build, never from what a pack happened to reach (a text-only pack never
        reaches the tower).
        """
        if self.cotrain_head is None:
            return []
        backbone = self.encoder.backbone
        # unread_module_names reads the top built layer as the top attach point: the expert
        # must read that layer's key and value, or every layer above its own top attach
        # point would be robot-unread too and miss from this list (DDP would then wait on
        # them on a chat-less step). skip28 attaches up to layer 39 of 40.
        top = int(self.decoder.expert.layer_map[-1])
        if top + 1 != len(backbone.layers):
            raise ValueError(
                f"the expert's layer_map ends at backbone layer {top} of {len(backbone.layers)}.\n"
                "Expected it to read the top built layer's key and value: the chat-only "
                "parameters are then the final norm and that layer's query, output and "
                "feed-forward halves, and nothing below.\n"
                f"Build {top + 1} backbone layers (model.backbone_num_layers), or extend "
                "NeoBackbone.unread_module_names to the layers above the top attach point."
            )
        names = [f"encoder.backbone.{name}." for name in backbone.unread_module_names()]
        names.append("cotrain_head.")
        return [param for name, param in self.named_parameters()
                if param.requires_grad and name.startswith(tuple(names))]

    def chat_participation(self) -> torch.Tensor:
        """0 x the sum of every chat-only parameter: a zero scalar that reaches each of them.

        A step without chat (pack_every_steps) runs its DDP sync micro-batch on
        robot rows alone, and DDP, find_unused_parameters false, finishes the
        step's reduction only once every parameter's gradient is ready: without
        this the chat-only buckets never are, the later buckets are never
        reduced, and the next synchronised forward raises (968 of 981 tensors
        differed across ranks before it did). Added to
        that micro-batch's loss it changes no value (x + 0 is x) and gives each
        chat-only parameter an exact zero, the gradient the bucket views already
        hold on such a step.
        """
        terms = [param.sum() for param in self.chat_only_parameters()]
        if not terms:
            raise RuntimeError(
                "chat_participation on a policy with no chat-only parameter.\n"
                "Expected a co-train head, whose parameters only the chat half reaches.\n"
                "Build the policy with model.cotrain set."
            )
        return torch.stack([term.float() for term in terms]).sum() * 0.0

    def require_embodiments(self, where: str):
        """policy.embodiments, or a raise naming what needed it."""
        if self.embodiments is None:
            raise RuntimeError(
                f"{where} needs the run's embodiments file and policy.embodiments is unset.\n"
                "Expected the runner (from data.train.normaliser) or the serving loader "
                "(from the step's embodiments.json) to have set it.\n"
                "Build through neo_vla.engine.runner."
            )
        return self.embodiments

    def checkpoint_files(self) -> dict[str, bytes]:
        """What checkpoint.save writes beside state.pt: the embodiments file."""
        embodiments = self.require_embodiments("a checkpoint")
        return {constants.EMBODIMENTS_FILE: embodiments.dumps()}

    def checkpoint_meta(self) -> dict:
        """What checkpoint.save writes into meta.json beside the step.

        The pixel convention, the prefix layout, the digest of the embodiments
        file written beside it with each key's entry digest, the state, action,
        body, history and tile stamps, the prefix geometry the encoder was
        built with, and the suffix layout (decoder.SUFFIX_LAYOUT). A resume, an
        init_from and a serving load hold each of them against this build
        before reading a weight.
        """
        embodiments = self.require_embodiments("a checkpoint")
        meta = {
            CONVENTION_META_KEY: self.encoder.pixel_convention,
            PREFIX_LAYOUT_KEY: constants.PREFIX_LAYOUT,
            EMBODIMENTS_KEY: {
                "digest": file_digest(embodiments.dumps()),
                "keys": {key: entry.digest for key, entry in sorted(embodiments.entries.items())},
            },
        }
        meta.update({key: dict(stamp) for key, stamp in RECORD_STAMPS})
        meta[PREFIX_KEY] = self.encoder.layout_stamp()
        meta[SUFFIX_LAYOUT_KEY] = SUFFIX_LAYOUT
        return meta

    def check_checkpoint_meta(self, meta, where: str) -> None:
        """Raise unless the checkpoint whose meta.json this is suits this build.

        Called before a resume, an init_from and a serving load read the
        weights, so a mismatch costs a json read and not minutes of torch.load:
        the pixel convention, then R11 on a prefix layout other than v7, R1 on a
        meta that names no embodiments file, R22 on a state, action, body,
        history or tile stamp other than this tree's, R22 on a prefix stamp other than this
        build's (review train-eval-safety m5: n_views, max_grid, the text budget
        and the tokenizer make no parameter, so the strict load cannot see them),
        and R22 on a suffix layout other than SUFFIX_LAYOUT: a v1 step (pre
        tokens, 96 at K 30) holds the very tensors v2 reads, under the same
        names and shapes, so only this stamp tells them apart.
        The file's own digest (R1, R2) is held by
        checkpoint.check_embodiments_file, which has the directory.
        """
        check_saved_convention(meta, self.encoder.pixel_convention, where)
        meta = meta or {}
        layout = meta.get(PREFIX_LAYOUT_KEY)
        if layout != constants.PREFIX_LAYOUT:
            old = (f" ({layout!r} is a retired layout)"
                   if layout in constants.OLD_PREFIX_LAYOUTS else "")
            raise ValueError(
                f"{constants.refusal_prefix(11)} {where} states prefix layout {layout!r}{old}.\n"
                f"Expected {constants.PREFIX_LAYOUT!r}: this tree reads the v7 prefix (six "
                "<state> columns after the instruction, the body's among them; an Earlier "
                "and a Now block; no <scale>).\n"
                "Serve or continue an older checkpoint with the tree that wrote it."
            )
        stated = meta.get(EMBODIMENTS_KEY)
        if not isinstance(stated, dict) or not {"digest", "keys"} <= set(stated):
            raise ValueError(
                f"{constants.refusal_prefix(1)} {where} names no embodiments file in "
                "meta.json.\n"
                f"Expected meta.{EMBODIMENTS_KEY} with a digest and each key's entry digest, "
                f"beside {constants.EMBODIMENTS_FILE}.\n"
                "Name a step written by this tree; a step without its normaliser cannot "
                "be trained from or served."
            )
        for key, own in RECORD_STAMPS:
            if meta.get(key) != own:
                raise ValueError(
                    f"{constants.refusal_prefix(22)} {where} states {key} {meta.get(key)!r}.\n"
                    f"Expected {own!r}, this tree's {key} stamp: the checkpoint was "
                    "trained on another record, another history or other tiles.\n"
                    "Name a step written under the same stamps."
                )
        built = self.encoder.layout_stamp()
        stated = meta.get(PREFIX_KEY)
        if stated != built:
            differ = sorted(key for key in built
                            if not isinstance(stated, dict) or stated.get(key) != built[key])
            raise ValueError(
                f"{constants.refusal_prefix(22)} {where} states prefix {stated!r}; this build "
                f"lays down {built!r} (differs in {differ}).\n"
                "Expected the prefix the step was trained under: n_views, max_grid, "
                "max_text_tokens, the tower's pixels per token and the tokenizer make no "
                "parameter, so a strict load would serve the step on other columns.\n"
                "Build from the run's own config (.hydra/config.yaml, no --config override), "
                "or name a step written under this one."
            )
        check_suffix_layout(meta, where)

    def check_dataset(self, dataset) -> None:
        """Raise unless the loader feeds this model what it was built for.

        Three holds. The frames resize under the model's pixel convention. A
        chat corpus renders NEO's own rows. And the dataset's
        normaliser agrees with policy.embodiments key by key: every entry the
        dataset normalises through has the digest this policy's file gives it
        (R8), because the targets and the state were normalised by the one and
        the checkpoint will be served by the other. The file digests differ
        once the run fills the trained section, so the entries are compared. A
        dataset that states no file (targets_only, or not a RobotDataset) is
        refused: it could not have normalised what this model trains on.
        """
        stated = dataset_field(dataset, "pixel_convention")
        if stated != self.encoder.pixel_convention:
            raise ValueError(
                f"the dataset resizes under pixel convention {stated!r}; the model "
                f"is built {self.encoder.pixel_convention!r}.\n"
                "Expected one convention: the loader and the serving canvas resize with "
                "the model's.\nDrop the data.train.pixel_convention override."
            )
        chat = dataset_field(dataset, "understanding")
        # The COCO ShareGPT corpus states no pack in both co-training stages.
        self.chat_pack = chat.pack if chat is not None and hasattr(chat, "pack") else None
        if chat is not None:
            from ..data.understanding import NATIVE_FORMAT

            chat_format = chat.chat_format if hasattr(chat, "chat_format") else None
            if chat_format != NATIVE_FORMAT:
                raise ValueError(
                    f"the chat corpus renders chat_format {chat_format!r}; the model is "
                    f"built {NEO_NATIVE!r}.\n"
                    f"Expected chat_format {NATIVE_FORMAT!r}, the one chat format this "
                    "tree has: legacy chat rows would pass the tower's ImageNet gate as "
                    "pixels NEO never trained on.\n"
                    "Select a data group whose understanding node sets chat_format: neo_native."
                )
        # Read as an attribute: RobotDataset states its file through a property,
        # which vars() never holds, and a check that returned early on that
        # passed on every real dataset.
        normaliser = dataset_field(dataset, "embodiments")
        if normaliser is None:
            raise ValueError(
                f"the training dataset ({type(dataset).__name__}) states no embodiments "
                "file.\n"
                "Expected RobotDataset.embodiments, the file its targets and states are "
                "normalised through, to hold against the policy's.\n"
                "Build the dataset in train mode with data.train.normaliser set."
            )
        own = self.require_embodiments("the dataset check").entries
        for key, entry in sorted(normaliser.entries.items()):
            held = own.get(key)
            if held is None or held.digest != entry.digest:
                raise ValueError(
                    f"{constants.refusal_prefix(8)} the dataset normalises {key!r} through "
                    f"entry {entry.digest}; the policy holds "
                    f"{'no such key' if held is None else held.digest}.\n"
                    "Expected one entry per key for the loader, the checkpoint and the "
                    "server: the numbers and the clause are in the digest.\n"
                    "Point data.train.normaliser at the file the policy was given."
                )

    def parameter_groups(self, schedule, learning_rate: float) -> list:
        """The trainable parameters as AdamW takes them: one list, or two groups.

        schedule.vlm_learning_rate absent or null gives one group at
        learning_rate, the optimizer every frozen_vlm pretrain has had. Set, it
        gives two: everything outside VLM_PREFIXES at learning_rate, first, and
        the backbone, the tower and the co-train head at that rate, second. The
        scheduler multiplies both by one factor, so they share the warmup and
        the cosine, and each ends at its own rate x lr_final / learning_rate.

        A guard and not a filter: a prefix that matched nothing would train the
        whole model at the head rate while the log said otherwise, so an empty
        group raises and the two are held to partition the trainable set.
        """
        if schedule.get("backbone_learning_rate", None) is not None:
            raise ValueError(
                "schedule.backbone_learning_rate is set; it slowed encoder.backbone "
                "alone and left the tower and the co-train head at the head rate.\n"
                f"Expected schedule.vlm_learning_rate, which covers {list(VLM_PREFIXES)}.\n"
                "Rename the key."
            )
        trainable = [param for param in self.parameters() if param.requires_grad]
        vlm_rate = schedule.get("vlm_learning_rate", None)
        if vlm_rate is None:
            return trainable
        vlm_rate = float(vlm_rate)
        if not 0.0 < vlm_rate <= learning_rate:
            raise ValueError(
                f"schedule.vlm_learning_rate is {vlm_rate} against learning_rate "
                f"{learning_rate}.\nExpected a positive rate at or below the head rate.\n"
                "Fix schedule.vlm_learning_rate."
            )
        slow, fast, slow_names = [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            plain = name.replace(WRAP_SEGMENT, "")
            if plain.startswith(VLM_PREFIXES):
                slow.append(param)
                slow_names.append(plain)
            else:
                fast.append(param)
        if not slow or not fast:
            raise ValueError(
                f"schedule.vlm_learning_rate is {vlm_rate} and the groups hold "
                f"{len(slow)} vision-language and {len(fast)} other tensors of "
                f"{len(trainable)} trainable.\n"
                f"Expected both non-empty: {list(VLM_PREFIXES)} and an action head.\n"
                "Leave vlm_learning_rate null under frozen_vlm, or check model.trainable."
            )
        if len(slow) + len(fast) != len(trainable):
            raise RuntimeError(
                f"the two groups hold {len(slow)} + {len(fast)} of {len(trainable)} "
                "trainable tensors.\nExpected them to partition the trainable set."
            )
        logger.info(
            "vlm lr: %d tensors / %d parameters under %s at %g; %d tensors / %d "
            "parameters at %g; first vlm %s",
            len(slow), sum(p.numel() for p in slow), list(VLM_PREFIXES), vlm_rate,
            len(fast), sum(p.numel() for p in fast), learning_rate, slow_names[:2],
        )
        return [
            {"params": fast, "lr": learning_rate},
            {"params": slow, "lr": vlm_rate},
        ]

    def configure_optimizers(
        self, cfg
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """AdamW over the parameters that require a gradient, and its schedule.

        Reads cfg.schedule for learning_rate, lr_final, warmup_steps,
        adam_betas, adam_eps, weight_decay and max_steps; a missing key raises.
        The schedule is a linear warmup into a cosine decay towards lr_final.
        """
        schedule = cfg.schedule
        learning_rate = float(schedule.learning_rate)
        final = float(schedule.lr_final)
        warmup = int(schedule.warmup_steps)
        horizon = int(schedule.max_steps)
        if not 0.0 <= final <= learning_rate:
            raise ValueError(
                f"schedule.lr_final is {final} against learning_rate {learning_rate}.\n"
                "Expected a floor between zero and the peak rate.\n"
                "Fix schedule.lr_final."
            )
        if warmup >= horizon:
            raise ValueError(
                f"schedule.warmup_steps is {warmup} of {horizon} steps.\n"
                "Expected a warmup shorter than the run.\n"
                "Lower schedule.warmup_steps, or raise schedule.max_steps."
            )
        # foreach=True is torch's CUDA default and costs one full extra copy
        # of the parameter set inside the step: _multi_tensor_adamw allocates
        # exp_avg_sq_sqrt = torch._foreach_sqrt(exp_avg_sq) over the whole group.
        # For a 3.14B-parameter full finetune that is 11.71 GiB, live exactly
        # when the moments are also live, which is where this arm runs out. The
        # single-tensor path does the same arithmetic one tensor at a time, so
        # the transient is the largest tensor rather than the whole model.
        # Absent from the schedule => the kwarg is not passed => torch's choice,
        # unchanged. geoA passes False (wm-engine vla_flow_fullft_mem.py:180).
        adam_extra = {}
        foreach = schedule.get("adam_foreach", None)
        if foreach is not None:
            adam_extra["foreach"] = bool(foreach)
        groups = self.parameter_groups(schedule, learning_rate)
        optimizer = torch.optim.AdamW(
            groups,
            lr=learning_rate,
            betas=tuple(float(value) for value in schedule.adam_betas),
            eps=float(schedule.adam_eps),
            weight_decay=float(schedule.weight_decay),
            **adam_extra,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: learning_rate_factor(
                step, warmup=warmup, horizon=horizon, floor=final / learning_rate
            ),
        )
        return optimizer, scheduler

    def geometry_keys(self) -> tuple[str, ...]:
        """The geometry records every batch carries: all four (intrinsics_mode tokens)."""
        return tuple(constants.GEOMETRY_KEYS)

    def check_batch(self, batch: dict, *, train: bool) -> None:
        """Raise unless the robot half of the batch has exactly the contract's keys.

        constants.batch_keys is the one statement of the batch both producers
        emit: 35 keys in training under tokens, 27 at serving. Equality, not a
        subset, so a field another tree would read (scale, state_valid,
        action_loss_valid, a control rate) is refused rather than ignored, and a
        missing history key or loss mask stops the step instead of defaulting.
        At serving, the two noise fields are refused first and by name (R21): a
        perturbed state is a training device.
        """
        robot = set(batch) - {CHAT}
        if not train:
            for name in NOISE_FIELDS:
                if name in robot:
                    raise ValueError(
                        f"{constants.refusal_prefix(21)} the batch carries {name} at "
                        "predict.\n"
                        "Expected no state noise outside training: it perturbs the state "
                        "the model reads, and a served state is the one the robot "
                        "measured.\n"
                        f"Drop {list(NOISE_FIELDS)} from the serving batch; noise_seed "
                        "stays valid."
                    )
        wanted = constants.batch_keys(self.geometry_keys(), train=train)
        if robot != wanted:
            missing = sorted(wanted - robot)
            extra = sorted(robot - wanted)
            retired = [f"{key} (now {RETIRED_BATCH_FIELDS[key]})" for key in extra
                       if key in RETIRED_BATCH_FIELDS]
            raise ValueError(
                f"the {'training' if train else 'serving'} batch misses {missing} and "
                f"carries {extra} it should not"
                + (f"; retired: {retired}" if retired else "") + ".\n"
                f"Expected exactly constants.batch_keys for intrinsics_mode "
                f"{self.encoder.intrinsics_mode!r}, train={train}: {sorted(wanted)}.\n"
                "Build the batch with RobotDataset.sample_fields or Predictor.batch."
            )

    def check_body(self, batch: dict, *, train: bool) -> None:
        """Raise unless the body's masks agree with its code (review v7spec, consistency).

        One source of truth: body_state_code. A key without a body is ABSENT in
        both roles, and then body_dim_valid is all False and, in training,
        body_action_valid and body_loss_dims are too, so its block is no key
        and trains nothing; the encoder masks both of its prefix columns from
        the same code. A body that exists may be UNKNOWN (no dimension read):
        its masks are whatever the loader read, all False when nothing was.
        """
        # On the host copies the loop kept (engine.hostcopy): no device wait.
        code = host_of(batch_field(batch, "body_state_code"))
        absent = code == STATE_ABSENT
        if code.dim() != 2 or not torch.equal(absent[:, 0], absent[:, -1]):
            raise ValueError(
                f"body_state_code is {tuple(code.shape)} and says ABSENT in one role and "
                "not the other.\n"
                "Expected [batch, 2], ABSENT in both roles exactly where the key has no "
                "body.\nApply STATE_CODE_PRECEDENCE in the producer."
            )
        absent = absent[:, 0]
        fields = ["body_dim_valid"] + (["body_action_valid", "body_loss_dims"] if train else [])
        for name in fields:
            value = host_of(batch_field(batch, name))
            if bool(value[absent].any()):
                raise ValueError(
                    f"{name} is True on a sample whose key has no body (body_state_code "
                    "ABSENT).\n"
                    "Expected an ABSENT body to be no key and no target: at every "
                    "benchmark the arms must see what a fixed-base sample showed them.\n"
                    "Fix the producer's body masks."
                )

    def prefix_state(self, batch: dict) -> tuple[StateInputs, int]:
        """The batch's physical records -> (StateInputs in model-input space, clips).

        state [B, 2, S, 29] and body_state [B, 2, 8] are the physical records,
        NaN where invalid; state_dims and body_state_dims mark their valid
        dimensions and state_code and body_state_code hold their codes. A
        MEASURED arm record goes through embodiment.normalise_state with the
        entry of the sample's own key and its slot, and a MEASURED body record
        through embodiment.normalise_body_state with the key's entry: joints,
        position, hand and body by their per-(key, slot, dim) maps into
        [-1, 1], clipped to 1 + 2 alpha and counted, the 6D through the key's
        frame fix, the aperture to 2a - 1. Only the valid dimensions are kept;
        every other number is 0, a placeholder the bank's mask removes. The
        same call on the dataset's batch and the server's gives the same input.
        The clips count every range dimension clipped, arm and body alike.
        """
        embodiments = self.require_embodiments("the state")
        state = batch_field(batch, "state")
        dims = batch_field(batch, "state_dims")
        code = batch_field(batch, "state_code")
        body = batch_field(batch, "body_state")
        body_dims = batch_field(batch, "body_state_dims")
        body_code = batch_field(batch, "body_state_code")
        keys = list(batch_field(batch, "embodiment"))
        if state.dim() != 4 or int(state.shape[-1]) != STATE_DIM:
            raise ValueError(
                f"state is {tuple(state.shape)}, expected [batch, 2, slots, {STATE_DIM}].\n"
                f"The record is {STATE_RECORD}, start then now.\nFix the producer."
            )
        if tuple(body.shape) != (int(state.shape[0]), int(state.shape[1]), BODY_DIM):
            raise ValueError(
                f"body_state is {tuple(body.shape)}, expected "
                f"{(int(state.shape[0]), int(state.shape[1]), BODY_DIM)}.\n"
                f"The record is {BODY_RECORD}, start then now.\nFix the producer."
            )
        if len(keys) != int(state.shape[0]):
            raise ValueError(
                f"embodiment names {len(keys)} keys for a batch of {int(state.shape[0])}.\n"
                "Expected one normaliser key per sample.\nFix the producer."
            )
        alpha = float(embodiments.band_alpha)
        # On the host, whatever device the batch is on: the records are a few
        # hundred numbers, the normaliser builds its bounds as host tensors, and
        # the per-sample bookkeeping below would otherwise be one device sync
        # per sample and slot. In training the loop's host copies are read and
        # the result goes back from pinned memory, so neither way waits on the
        # device (engine.hostcopy); elsewhere one copy in, one copy out.
        host = StateInputs(
            state=host_of(state).to(torch.float32),
            state_dims=host_of(dims),
            state_code=host_of(code),
            body_state=host_of(body).to(torch.float32),
            body_state_dims=host_of(body_dims),
            body_state_code=host_of(body_code),
        )
        host.validate(int(state.shape[2]))
        arm_input, arm_clips = self.normalise_records(
            host.state, host.state_dims, host.state_code, keys, alpha, body=False
        )
        body_input, body_clips = self.normalise_records(
            host.body_state[:, :, None], host.body_state_dims[:, :, None],
            host.body_state_code[:, :, None], keys, alpha, body=True,
        )
        clips = arm_clips + body_clips
        self.state_clips = clips
        inputs = StateInputs(
            state=to_host_device(arm_input, state.device),
            state_dims=dims,
            state_code=code,
            body_state=to_host_device(body_input[:, :, 0].contiguous(), body.device),
            body_state_dims=body_dims,
            body_state_code=body_code,
        )
        return inputs, clips

    def normalise_records(
        self,
        records: torch.Tensor,
        dims: torch.Tensor,
        code: torch.Tensor,
        keys: list[str],
        alpha: float,
        *,
        body: bool,
    ) -> tuple[torch.Tensor, int]:
        """Host records [B, 2, S, D] -> (model input, clips), through each sample's own entry.

        A dimension enters the normaliser only where the code is MEASURED and
        its mask is set; every other number goes in as NaN and comes out 0. A
        valid dimension that normalises to a number that is not finite raises:
        the producer called it read and the normaliser could not place it.
        """
        measured = code == STATE_MEASURED
        valid = dims & measured[..., None]
        out = torch.zeros(records.shape, dtype=torch.float32)
        clips = 0
        if not bool(valid.any()):
            return out, clips
        # Imported by name from the module, at the call, so a test that installs
        # the contract's functions in its place is read here.
        if body:
            from ..data.embodiment import normalise_body_state as normalise
        else:
            from ..data.embodiment import normalise_state as normalise
        embodiments = self.require_embodiments("the state")
        for key in sorted(set(keys)):
            entry = embodiments.entry(key)
            for slot in range(int(records.shape[2])):
                rows = [index for index, name in enumerate(keys)
                        if name == key and bool(valid[index, :, slot].any())]
                if not rows:
                    continue
                picked = torch.tensor(rows)
                keep = valid[picked, :, slot]
                record = torch.where(keep, records[picked, :, slot],
                                     torch.full_like(records[picked, :, slot], float("nan")))
                if body:
                    normalised, clipped = normalise(record, entry, alpha=alpha)
                else:
                    normalised, clipped = normalise(record, entry, slot, alpha=alpha)
                normalised = torch.as_tensor(normalised, dtype=torch.float32, device="cpu")
                out[picked, :, slot] = torch.where(keep, normalised, torch.zeros_like(normalised))
                clips += int(clipped)
        if not bool(torch.isfinite(out[valid]).all()):
            raise ValueError(
                f"a valid dimension of a MEASURED {'body ' if body else ''}state normalised to "
                "a value that is not finite.\n"
                "Expected a finite physical value, and a bound for it in the key's entry, "
                "wherever the mask marks the dimension read.\n"
                "Clear that dimension's mask in the producer when it was not read."
            )
        return out, clips

    @staticmethod
    def perturb_state(
        state: StateInputs,
        noise: torch.Tensor,
        body_noise: torch.Tensor,
        *,
        alpha: float,
    ) -> StateInputs:
        """The training-only state noise, applied to the valid dimensions of MEASURED records alone.

        noise [B, 2, S, 26] is the loader's pre-scaled draw in the noise order,
        which is not the record's (constants.NOISE_*): position 3, rotation
        vector 3 in radians, aperture 1, joints 7, hand 12. body_noise
        [B, 2, 8] is in the body's order. Each named slice moves its own field
        of the record (T8): position, joints, hand and body become
        clip(y + n, 1 + 2 alpha); the rotation R' = exp([dw]x) R, re-emitted as
        6D; the aperture clip(y + n, [-1, 1]). A dimension is moved only where
        the code is MEASURED and its mask is set (the 6D as one group); the
        zero start, UNKNOWN, ABSENT and an invalid dimension are never
        perturbed, and a draw that is not zero on them is refused as a producer
        defect. An entry with no rotation noise keeps its 6D bit for bit, so a
        draw of zeros is the identity. A NaN never survives: the result must be
        finite wherever a MEASURED record's dimension is valid.
        """
        arm_shape = (*state.state_code.shape, NOISE_WIDTH)
        body_shape = (*state.body_state_code.shape, BODY_NOISE_WIDTH)
        for name, value, shape in (("state_noise", noise, arm_shape),
                                   ("body_state_noise", body_noise, body_shape)):
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"{name} is {tuple(value.shape)}, expected {shape}.\n"
                    "Position 3, rotation vector 3, aperture 1, joints 7, hand 12 per role "
                    "and arm; the body's 8 per role."
                )
        # The producer checks read host copies of the draws, the masks and the
        # codes (engine.hostcopy): the same values, without a device wait.
        host_dims = host_of(state.state_dims)
        host_measured = host_of(state.state_code) == STATE_MEASURED
        host_noise = host_of(noise).to(torch.float32)
        host_rotation_valid = host_dims[..., STATE_ROTATION].all(dim=-1)
        # Where each noise column may be non-zero: the valid dimensions it moves.
        allowed = torch.zeros(host_noise.shape, dtype=torch.bool)
        allowed[..., NOISE_POSITION] = host_dims[..., STATE_POSITION]
        allowed[..., NOISE_ROTATION] = host_rotation_valid[..., None]
        allowed[..., NOISE_APERTURE] = host_dims[..., STATE_APERTURE]
        allowed[..., NOISE_JOINTS] = host_dims[..., STATE_JOINTS]
        allowed[..., NOISE_HAND] = host_dims[..., STATE_HAND]
        host_body_dims = host_of(state.body_state_dims)
        host_body_measured = host_of(state.body_state_code) == STATE_MEASURED
        host_body_noise = host_of(body_noise).to(torch.float32)
        for draw, live, valid in ((host_noise, host_measured, allowed),
                                  (host_body_noise, host_body_measured, host_body_dims)):
            if bool((draw[~live] != 0).any()):
                raise ValueError(
                    "state_noise is not zero on a state that is not MEASURED.\n"
                    "Expected noise only where the record was measured: never on the zero "
                    "start, an UNKNOWN or an ABSENT record.\nFix history.state_noise."
                )
            if bool((draw[live] != 0)[~valid[live]].any()):
                raise ValueError(
                    "state_noise is not zero on a dimension a MEASURED state marks invalid.\n"
                    "Expected noise only on the valid dimensions it moves (the 6D as one "
                    "group): an unread joint or hand dimension is never perturbed.\n"
                    "Fix history.state_noise."
                )
        record = state.state
        dims = state.state_dims.to(record.device)
        measured = (state.state_code == STATE_MEASURED).to(record.device)
        noise = noise.to(device=record.device, dtype=torch.float32)
        rotation_valid = dims[..., STATE_ROTATION].all(dim=-1)
        body_record = state.body_state
        body_dims = state.body_state_dims.to(body_record.device)
        body_measured = (state.body_state_code == STATE_MEASURED).to(body_record.device)
        body_noise = body_noise.to(device=body_record.device, dtype=torch.float32)
        bound = 1.0 + 2.0 * float(alpha)
        out = record.clone()
        for field, column in ((STATE_POSITION, NOISE_POSITION), (STATE_JOINTS, NOISE_JOINTS),
                              (STATE_HAND, NOISE_HAND)):
            out[..., field] = (record[..., field] + noise[..., column]).clamp(-bound, bound)
        out[..., STATE_APERTURE] = (
            record[..., STATE_APERTURE] + noise[..., NOISE_APERTURE]
        ).clamp(-1.0, 1.0)
        turned = measured & rotation_valid & (noise[..., NOISE_ROTATION] != 0).any(dim=-1)
        host_turned = (host_measured & host_rotation_valid
                       & (host_noise[..., NOISE_ROTATION] != 0).any(dim=-1))
        if bool(host_turned.any()):
            from ..data.state import ee_rotation_6d, rotation_from_6d, so3_exp

            rotation = record[..., STATE_ROTATION]
            identity = identity_6d(rotation.device).expand_as(rotation)
            safe = torch.where(turned[..., None], rotation, identity)
            matrix = so3_exp(noise[..., NOISE_ROTATION]) @ rotation_from_6d(safe)
            out[..., STATE_ROTATION] = torch.where(
                turned[..., None], ee_rotation_6d(matrix).to(rotation.dtype), rotation
            )
        moved = dims & measured[..., None]
        arm = torch.where(moved, out, record)
        body_moved = body_dims & body_measured[..., None]
        body = torch.where(
            body_moved, (body_record + body_noise).clamp(-bound, bound), body_record
        )
        finite = (torch.isfinite(arm) | ~moved).all() & (torch.isfinite(body) | ~body_moved).all()
        require(finite, refuse_perturbed_state)
        return state.replace(state=arm, body_state=body)

    def suffix_inputs(self, batch: dict) -> SuffixInputs:
        """The per-arm suffix conditioning: the frame each arm's actions are in, and slot_valid."""
        return SuffixInputs(
            frame_ref=batch_field(batch, "frame_ref"),
            slot_valid=batch_field(batch, "slot_valid"),
        )

    def geometry_records(self, batch: dict, role: str) -> dict[str, torch.Tensor]:
        """One role's four geometry fields as encode's keywords."""
        return {f"{role}{name}": batch_field(batch, f"{role}{name}")
                for name in self.geometry_keys()}

    def encode_prefix(
        self,
        batch: dict,
        suffix_pos: torch.Tensor,
        *,
        state: StateInputs,
        recompute_layers: int | None = None,
    ) -> PrefixContext:
        """The prefix for this batch, under the position vector it is given.

        The one call site of encoder.encode, so training and serving cannot
        hand the encoder different prefixes. state is model-input space, from
        prefix_state (and perturb_state in training). recompute_layers is the
        backbone's (NeoBackbone.forward); None recomputes every layer.
        """
        return self.encoder.encode(
            pixels=batch_field(batch, "pixels"),
            image_shapes=batch_field(batch, "image_shapes"),
            view_valid=batch_field(batch, "view_valid"),
            history_pixels=batch_field(batch, "history_pixels"),
            history_image_shapes=batch_field(batch, "history_image_shapes"),
            history_view_valid=batch_field(batch, "history_view_valid"),
            history_valid=batch_field(batch, "history_valid"),
            prompts=list(batch_field(batch, "prompt")),
            **self.geometry_records(batch, ""),
            **self.geometry_records(batch, "history_"),
            state=state,
            slot_valid=batch_field(batch, "slot_valid"),
            suffix_pos=suffix_pos,
            recompute_layers=recompute_layers,
        )

    def flow_step(self, batch: dict, *,
                  recompute_layers: int | None = None) -> dict[str, torch.Tensor]:
        """The robot half of a step, in the order that matters.

        1. The batch's keys are the contract's, training form, and the body's
           masks agree with its code.
        2. The arms' and the body's states are normalised, then perturbed by
           the loader's draws.
        3. n_slots and K come from batch["actions"], shape [B, S, K, A], and the
           decoder states suffix_pos for them (S arm blocks and the body's).
        4. The encoder is called with suffix_pos as a keyword and returns the
           PrefixContext.
        5. The decoder takes the loss: keys by action_valid and
           body_action_valid, inputs by dim_valid and body_dim_valid, weights
           by action_loss_dims and body_loss_dims.
        """
        self.check_batch(batch, train=True)
        self.check_body(batch, train=True)
        # The checks on values the device computes (the perturbed state, the
        # tower's pixels) are read together once the forward is queued.
        with deferred():
            return self.flow_terms(batch, recompute_layers=recompute_layers)

    def flow_terms(self, batch: dict, *,
                   recompute_layers: int | None = None) -> dict[str, torch.Tensor]:
        """flow_step after the batch checks: steps 2 to 5."""
        actions = batch_field(batch, "actions")
        if actions.dim() != 4:
            raise ValueError(
                f"actions is {tuple(actions.shape)}, expected [batch, slots, "
                "chunk, dims].\nThe slot count of the batch is read from it."
            )
        batch_size, n_slots, chunk = (int(size) for size in actions.shape[:3])
        state, clips = self.prefix_state(batch)
        state = self.perturb_state(
            state, batch_field(batch, "state_noise"), batch_field(batch, "body_state_noise"),
            alpha=float(self.embodiments.band_alpha),
        )
        suffix_pos = self.decoder.suffix_positions(
            n_slots * chunk, n_slots=n_slots, device=actions.device
        )
        inputs = self.suffix_inputs(batch)
        inputs.validate(batch_size, n_slots)
        prefix = self.encode_prefix(batch, suffix_pos, state=state,
                                    recompute_layers=recompute_layers)
        result = self.decoder.loss(
            prefix,
            inputs,
            actions=actions,
            action_valid=batch_field(batch, "action_valid"),
            action_loss_dims=batch_field(batch, "action_loss_dims"),
            dim_valid=batch_field(batch, "dim_valid"),
            body_actions=batch_field(batch, "body_actions"),
            body_action_valid=batch_field(batch, "body_action_valid"),
            body_loss_dims=batch_field(batch, "body_loss_dims"),
            body_dim_valid=batch_field(batch, "body_dim_valid"),
        )
        # A fill, not a copy of a host scalar: the same float32 without a wait.
        result["state_clipped"] = torch.full((), float(clips), device=actions.device)
        return result

    def prepare_step(self, batches: list[dict], step: int | None = None) -> None:
        """A step's micro-batches, before its first forward: NEO's loss denominator.

        NEO divides the square-weighted loss by the weight sum of its whole
        step. A step here is len(batches) micro-batches whose labels arrive with
        them, so that sum over the step and every rank is one all_reduce here
        rather than one inside every micro-batch's forward, which held all 64
        ranks to the slowest one's chat batch sixteen times a step. Every rank
        calls this once a step, with or without native rows in its window. A
        packed micro-batch weighs each of its rows as a row of its own, by its
        segment_targets.

        The sum is averaged over ALL len(batches) micro-batches of the step,
        those without a chat half included, because the loop divides every
        micro-batch's loss by that same accumulation: the two cancel, and the
        chat gradient is cotrain_weight x the step's global weighted mean
        however few micro-batches carry a pack (shape B: 2 of 8). Averaged over
        the chat micro-batches alone it would be scaled by their share, 2/8.
        A run that names its chat micro-batches (chat_pack.PackConfig
        micro_batches) is held to them here, once a step: the packs on exactly
        those, the step's last among them. A run with a chat cadence
        (pack_every_steps above 1) is held to it too: at global step `step` (the
        loop passes it) exactly the micro-batches PackConfig.step_micro_batches
        names, none on a chat-less step; without a step, those or none.
        """
        pack = self.chat_pack
        if pack is not None and (pack.micro_batches is not None or pack.every != 1):
            named = pack.chat_micro_batches(len(batches))
            allowed = ((pack.step_micro_batches(step, len(batches)),) if step is not None
                       else (named, ()) if pack.every != 1 else (named,))
            carrying = tuple(at for at, batch in enumerate(batches) if CHAT in batch)
            if carrying not in allowed:
                raise ValueError(
                    f"this step's chat halves are on micro-batches {list(carrying)} of "
                    f"{len(batches)}"
                    f"{'' if step is None else f' at step {int(step)}'}.\n"
                    f"Expected exactly {' or '.join(str(list(one)) for one in allowed)}, the "
                    f"run's pack_micro_batches on the steps its pack_every_steps "
                    f"({pack.every}) picks.\n"
                    "Build the step with collate_step, which packs the micro-batches the "
                    "chat corpus names on the steps it names."
                )
        self.step_micro_left = len(batches)
        self.step_has_chat = any(CHAT in batch for batch in batches)
        targets = [
            chat_targets(batch[CHAT]) for batch in batches
            if CHAT in batch and CHAT_IMAGES in batch[CHAT]
        ]
        device = next(self.cotrain_head.parameters()).device
        self.step_weight = step_weight_mean(targets, len(batches), device)
        self.step_uses_left = len(targets)

    def cotrain_step(self, chat: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """The vision-language half -> (scalar loss, detached diagnostics).

        NEO's own rows only: collate_chat's five fields, run through
        encode_chat_rows and NEO's square-weighted loss. The policy logs this
        rank's weighted mean as cotrain_loss beside the token mean. The legacy
        half (one stacked [-1, 1] canvas, the token mean) is retired and
        refused by name. Raises when the batch carries a chat half and no head
        was built, rather than skipping it: a co-train run that quietly trains
        on robot data only looks exactly like one that works.
        """
        if self.cotrain_head is None:
            raise ValueError(
                "the batch carries a chat half and no co-train head was built.\n"
                "Expected model.cotrain to be set for a batch with chat rows.\n"
                "Select mode=cotrain, or drop the chat half from the batch."
            )
        retired = sorted(set(chat) & set(RETIRED_CHAT_FIELDS))
        if retired:
            raise ValueError(
                f"the chat half carries {retired}: "
                f"{', '.join(RETIRED_CHAT_FIELDS[name] for name in retired)}.\n"
                "Expected native chat rows only (collate_chat's input_ids, "
                "attention_mask, labels, images and image_rows): the legacy chat "
                "format is retired.\n"
                "Build the chat corpus with chat_format neo_native."
            )
        if set(chat) == set(packing.PACK_FIELDS):
            return self.cotrain_pack(chat)
        if set(chat) != CHAT_FIELDS:
            raise ValueError(
                f"the chat half carries {sorted(chat)}.\n"
                f"Expected exactly {sorted(CHAT_FIELDS)}, what collate_chat builds, or "
                f"{sorted(packing.PACK_FIELDS)}, what collate_chat_pack builds.\n"
                "Build the chat half with collate_chat, or collate_chat_pack in pack mode."
            )
        hidden = self.encoder.encode_chat_rows(
            batch_field(chat, CHAT_IDS),
            batch_field(chat, CHAT_MASK),
            batch_field(chat, CHAT_IMAGES),
            batch_field(chat, CHAT_IMAGE_ROWS),
        )
        sums, counts = self.cotrain_head.row_losses(hidden, batch_field(chat, CHAT_LABELS))
        shared = None
        if self.step_uses_left > 0:
            shared, self.step_uses_left = self.step_weight, self.step_uses_left - 1
        loss, local = square_weighted_loss(sums, counts, shared)
        token_mean = (sums.sum() / counts.sum().clamp(min=1)).detach()
        return loss, {"cotrain_loss": local, "cotrain_token_ce": token_mean}

    def cotrain_pack(self, pack: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """One packed chat half -> (scalar loss, detached diagnostics and counters).

        The pack is held to the frozen layout (chat_pack.check_pack) on its
        host copies, so the device is not waited on; run through
        encode_chat_pack; and scored per packed row, so NEO's square weight
        sees each row as a row (row_losses with segments). Its counters come
        back under PACK_COUNTER_OUTPUT. Raises on a pack of another length
        than the run's pack_tokens, and when the labels' count per row is not
        segment_targets, both read on the host.
        """
        host = {name: [host_of(image) for image in value] if name == packing.PACK_IMAGES
                else host_of(value) for name, value in pack.items()}
        length = int(host[packing.PACK_IDS].shape[-1])
        if self.chat_pack is not None and length != int(self.chat_pack.tokens):
            raise ValueError(
                f"a chat pack of {length} positions reached a run packing "
                f"{self.chat_pack.tokens}.\n"
                "Expected every pack at data.train.understanding.pack_tokens.\n"
                "Pad each pack to pack_tokens in collate_chat_pack."
            )
        rows = packing.check_pack(host, pack_tokens=length)
        hidden = self.encoder.encode_chat_pack(
            batch_field(pack, packing.PACK_IDS),
            batch_field(pack, packing.PACK_SEGMENTS),
            batch_field(pack, packing.PACK_POSITIONS),
            batch_field(pack, packing.PACK_IMAGES),
            batch_field(pack, packing.PACK_GRID_HW),
        )
        segments = batch_field(pack, packing.PACK_SEGMENTS)
        sums, counts = self.cotrain_head.row_losses(
            hidden, batch_field(pack, packing.PACK_LABELS), segments=segments, rows=rows)
        targets = batch_field(pack, packing.PACK_SEGMENT_TARGETS)
        # Both on the host: row_losses counted a pack's rows there and moved the
        # counts without waiting (speed C-8+), and targets has the loader's copy.
        host_counts, host_targets = host_of(counts), host_of(targets)
        if not torch.equal(host_counts, host_targets):
            raise ValueError(
                f"the pack's labels supervise {host_counts.tolist()[:8]} per row; "
                f"segment_targets is {host_targets.tolist()[:8]}.\n"
                "Expected one count per row, the same on both sides.\n"
                "Count labels != -100 per segment in collate_chat_pack."
            )
        shared = None
        if self.step_uses_left > 0:
            shared, self.step_uses_left = self.step_weight, self.step_uses_left - 1
        loss, local = square_weighted_loss(sums, counts, shared)
        token_mean = (sums.sum() / counts.sum().clamp(min=1)).detach()
        return loss, {"cotrain_loss": local, "cotrain_token_ce": token_mean,
                      PACK_COUNTER_OUTPUT: batch_field(pack, packing.PACK_COUNTERS).detach()}

    def forward(self, batch: dict, ctx: StepContext) -> dict[str, torch.Tensor]:
        """One micro-batch -> {"loss": scalar, ...detached diagnostics}.

        The diagnostics are flow_loss, the decoder's six group diagnostics
        (loss_ee, loss_gripper, loss_joint, loss_hand, loss_body_base,
        loss_body_posture) each with its loss-triple count (loss_ee_triples and
        so on), tau_mean and state_clipped, plus cotrain_loss and
        cotrain_token_ce when the batch carries a chat half. "loss" is the only
        entry the loop trains on; the rest are for the step log, which pools the
        group losses over its window by their counts. ctx says where the run
        stands; the objective does not depend on it. A chat pack runs after
        the robot rows unless chat_pack_first (off), and both halves' device checks are
        read once at the end; unpacked chat rows run after them, as always.
        """
        # A packing run that names its chat micro-batches (shape B), or chats on
        # some steps only (pack_every_steps), runs the others on robot rows
        # alone, with no chat forward; prepare_step holds each step's packs to
        # the micro-batches and steps named.
        pack = self.chat_pack
        robot_only = pack is not None and (pack.micro_batches is not None or pack.every != 1)
        if self.cotrain_head is not None and CHAT not in batch and not robot_only:
            raise ValueError(
                f"a co-train head was built and the batch has no {CHAT!r} half.\n"
                "Expected chat rows in every co-train micro-batch, unless the chat corpus "
                "names its chat micro-batches (pack_micro_batches) or steps "
                "(pack_every_steps).\n"
                "Mix the corpus into every batch, or select another mode."
            )
        chat = batch.get(CHAT)
        packed = chat is not None and set(chat) == set(packing.PACK_FIELDS)
        # The prepared step's last micro-batch is the one DDP synchronises.
        sync = self.step_micro_left == 1
        self.step_micro_left = max(0, self.step_micro_left - 1)
        # A micro-batch with a chat half recomputes every backbone layer of its
        # robot rows too; a robot-only one the runner's robot_recompute_layers.
        recompute = None if chat is not None else self.robot_recompute_layers
        # A chat pack's checks on device values (the tower's pixel range) and
        # flow_step's are read once, after both halves are queued (speed C-8+).
        if packed and not self.chat_order_logged:
            logger.info(CHAT_ORDER_LOG_LINE.format(
                order="the pack first, then the robot rows" if self.chat_pack_first
                else "the robot rows first, then the pack", first=bool(self.chat_pack_first)))
            self.chat_order_logged = True
        with deferred() if packed else nullcontext():
            if packed and self.chat_pack_first:
                cotrain, diagnostics = self.cotrain_step(chat)
                flow = self.flow_step(batch, recompute_layers=recompute)
            else:
                flow = self.flow_step(batch, recompute_layers=recompute)
                if chat is not None:
                    cotrain, diagnostics = self.cotrain_step(chat)
        loss = flow.pop("loss")
        result = {"loss": loss, "flow_loss": loss.detach(), **flow}
        if chat is not None:
            result["loss"] = loss + self.cotrain_weight * cotrain
            result.update(diagnostics)
        elif self.cotrain_head is not None and sync and not self.step_has_chat:
            # A chat-less step's sync micro-batch: every chat-only parameter
            # takes part in DDP's reduction at an exact zero (chat_participation).
            # Only there: a step with chat carries a pack on its sync micro-batch
            # (chat_pack.check_micro_batches), and prepare_step refuses one without.
            result["loss"] = loss + self.chat_participation()
        return result

    @torch.no_grad()
    def predict(
        self,
        batch: dict,
        *,
        noise: torch.Tensor | None = None,
        body_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A serving batch -> normalised (arms [B, n_slots, K, 26], body [B, K, 8]).

        The batch is the contract's serving form: the noise fields are refused
        by name (R21) and any other key-set difference after them. The slot
        count comes from batch["slot_valid"], since there is no action tensor,
        and K is the decoder's chunk; the position vector is built exactly as
        training builds it. The range columns come back inside 1 + 2 alpha and
        the gripper inside [-1, 1]; the body is 0 wherever body_dim_valid is
        False, so all of it for a key without a body. The server de-normalises
        both through the request key's entry. noise and body_noise, both or
        neither, are the request's own draws (FlowDecoder.sample_actions).
        state_clips holds this call's range clips.
        """
        self.check_batch(batch, train=False)
        self.check_body(batch, train=False)
        slot_valid = batch_field(batch, "slot_valid")
        if slot_valid.dim() != 2:
            raise ValueError(
                f"slot_valid is {tuple(slot_valid.shape)}, expected [batch, slots].\n"
                "One flag per arm; it is what says how many arms to predict."
            )
        batch_size, n_slots = (int(size) for size in slot_valid.shape)
        state, _ = self.prefix_state(batch)
        chunk = self.decoder.chunk
        suffix_pos = self.decoder.suffix_positions(
            n_slots * chunk, n_slots=n_slots, device=slot_valid.device
        )
        inputs = self.suffix_inputs(batch)
        inputs.validate(batch_size, n_slots)
        prefix = self.encode_prefix(batch, suffix_pos, state=state)
        return self.decoder.sample_actions(
            prefix,
            inputs,
            n_slots=n_slots,
            slot_valid=slot_valid,
            dim_valid=batch_field(batch, "dim_valid"),
            body_dim_valid=batch_field(batch, "body_dim_valid"),
            range_bound=1.0 + 2.0 * float(self.embodiments.band_alpha),
            noise=noise,
            body_noise=body_noise,
        )
