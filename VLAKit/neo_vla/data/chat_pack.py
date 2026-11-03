"""Optional packed-chat interfaces shared by the loader and model.

The COCO ShareGPT recipes use unpacked rows. With pack_tokens null this module
does not change a batch or loss. Custom packed corpora use these field names,
configuration checks, row-chain seeds and validation helpers.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch

from .mixture import stream_seed

# <IMG_CONTEXT>'s id in NEO's vocabulary, restated from
# neo_vla/model/backbone/neo_llm.py IMG_CONTEXT_ID: importing neo_llm runs
# neo_vla.model's __init__, whose encoder imports this module, so the shared
# interface would import itself half-built. The data side imports no model
# module. Keep this value aligned with the backbone constant.
IMG_CONTEXT_ID = 151_669

# The config keys. pack_tokens null is the unpacked loader: one chat row per
# robot row. force_pack is a test hook that forces fixture packs on the leading
# steps; training runs leave it null.
PACK_TOKENS_KEY = "data.train.understanding.pack_tokens"
FORCE_PACK_KEY = "data.train.understanding.force_pack"
# Which micro-batches of a step carry a pack: null is every one; a list of
# indices is those only, the rest robot rows alone, with no chat forward at all. The last micro-batch of a step, accumulation - 1, must
# be one of them: it is the one whose backward DDP synchronises (loop.py
# is_sync; no_sync on the others; find_unused_parameters false), so without a
# pack there the chat-only parameters (cotrain_head) reach that backward with
# no gradient, and DDP either refuses the next step or leaves them unsynchronised
# across ranks. PackConfig.chat_micro_batches refuses such a list wherever the
# accumulation is known (runner.build_loader, the policy's prepare_step).
PACK_MICRO_BATCHES_KEY = "data.train.understanding.pack_micro_batches"
# How often a step carries chat at all: null or 1 is every step; N is the
# steps whose number is a multiple of N (step 0 first), and a forced fixture
# step (force_pack) always carries. A step without chat runs its
# micro-batches on robot rows alone, its DDP sync micro-batch included, so the
# chat-only parameters take no gradient from the data there: the policy adds
# them to that backward at weight zero (NeoVlaPolicy.chat_participation), or
# DDP, find_unused_parameters false, never finishes the step's reduction.
# The chat steps share chains as consecutive steps do at 1: chat step c (global
# step c x N, PackConfig.chat_step) takes slot c % chain_steps of its group's
# chain, so no slot of a chain goes undelivered.
PACK_EVERY_STEPS_KEY = "data.train.understanding.pack_every_steps"

# Memory relief for the chat pack, under model, off by default:
# chat_offload_checkpoints keeps the pack's recomputation
# inputs in pinned host memory (torch.autograd.graph.save_on_cpu);
# chat_mlp_chunk runs the recomputed layer's MLP over CHAT_MLP_CHUNK_TOKENS
# positions at a time. Neither changes a number the loss sees.
RELIEF_FLAGS = ("chat_offload_checkpoints", "chat_mlp_chunk")
CHAT_MLP_CHUNK_TOKENS = 4_096

# NEO's pack: max_packed_tokens = seq_len = model_max_length = 36,864 in the
# NEO++ SFT config (survey_neoplus; packing_spec 1.1), padded to full length.
NEO_PACK_TOKENS = 36_864
# The BlockMask block size. A pack length that is a multiple of it needs no
# padding of q, k and v (36,864 = 288 x 128).
PACK_BLOCK = 128

# A pack's fields, the chat half of one micro-batch in pack mode, in this
# order. S is pack_tokens, n the rows packed, K the images.
#   input_ids        int64 [1, S]     CHAT_PAD_ID on padding
#   labels           int64 [1, S]     IGNORE_LABEL on padding and at every row's first position
#   segments         int64 [1, S]     row ordinal 0..n-1, contiguous and in order; PAD_SEGMENT on padding
#   segment_targets  int64 [n]        each row's supervised labels, at least 1
#   positions        int64 [3, 1, S]  t, h, w per position; t restarts at 0 on each row; 0 on padding
#   images           list of K uint8 [3, H, W], in <IMG_CONTEXT> order; H, W multiples of TOKEN_PIXELS
#   grid_hw          int64 [K, 2]     each image's token grid, (H, W) // TOKEN_PIXELS
#   counters         int64 [len(COUNTERS)]
PACK_IDS = "input_ids"
PACK_LABELS = "labels"
PACK_SEGMENTS = "segments"
PACK_SEGMENT_TARGETS = "segment_targets"
PACK_POSITIONS = "positions"
PACK_IMAGES = "images"
PACK_GRID_HW = "grid_hw"
PACK_COUNTERS = "counters"
PACK_FIELDS = (PACK_IDS, PACK_LABELS, PACK_SEGMENTS, PACK_SEGMENT_TARGETS, PACK_POSITIONS,
               PACK_IMAGES, PACK_GRID_HW, PACK_COUNTERS)
# Padding is no row's: the loss runs over the n real rows, and the BlockMask
# builder alone splits the pad tail into NEO's 1,024-token pseudo-documents
#.
PAD_SEGMENT = -1
# understanding.IGNORE_INDEX, understanding.CHAT_PAD_ID and
# understanding.NEO_TOKEN_PIXELS, restated so this module never imports the
# corpus (which imports it).
IGNORE_LABEL = -100
PAD_ID = 0
TOKEN_PIXELS = 32

# What one pack counts, in this order: the rows it holds, their real tokens,
# their images, the rows dropped over max_tokens while it was filled (the
# corpus's "too_long" skip), and the step's straddling row that no
# pack took (1 on the step's last pack, else 0). Fill is tokens over S, taken
# when logged. The loop logs counter c as COUNTER_PREFIX + c, summed over a
# log window and all-reduced on every rank.
COUNTER_ROWS = "rows"
COUNTER_TOKENS = "tokens"
COUNTER_IMAGES = "images"
COUNTER_DROPPED = "dropped_overlong"
COUNTER_SPILLED = "spilled"
COUNTERS = (COUNTER_ROWS, COUNTER_TOKENS, COUNTER_IMAGES, COUNTER_DROPPED, COUNTER_SPILLED)
COUNTER_PREFIX = "chat_pack_"

# The worst-case fixtures force_pack names, one per leading step: f_tok is one row of two
# 4,096 x 4,096 images and its text, a single segment of exactly S tokens;
# f_cnt is 558 images of 64 tokens (66 with <img> and </img>) and 36 text
# tokens, S = 36,864 exactly.
FORCE_PACK_FIXTURES = ("f_tok", "f_cnt")

# The two lines a packed run logs once at start: the loader's, when it builds step items,
# and the backbone's, after its self-test at the production shape. kernel is
# one of CHAT_KERNELS: compiled flex_attention, or the two flash varlen calls
# of gate G-K's fallback.
PACK_LOG_LINE = "chat pack: {tokens} tokens per micro-batch, chain {chain}"
# Appended to PACK_LOG_LINE by a run that states pack_micro_batches: the chat
# micro-batches, comma separated, the step's accumulation, and the chat steps
# that share one chain (PackConfig.chain_steps). A run that states none logs
# PACK_LOG_LINE alone, as before (pack_log_line).
PACK_MICRO_LOG_SUFFIX = ", on micro-batches {micro} of {accumulation}, {steps} steps a chain"
# Appended after that by a run that states pack_every_steps above 1: its chat
# steps, of which the suffix's {steps} share one chain.
PACK_EVERY_LOG_SUFFIX = ", on steps divisible by {every}"
KERNEL_LOG_LINE = "chat kernel: {kernel}, forward and backward ok at {tokens} tokens"
CHAT_KERNELS = ("flex", "varlen")

# The step item. In pack mode one loader
# item is one optimizer step of one rank. The robot samples carry their draw
# number under DRAW_INDEX_KEY to the collate, which strips it before
# default_collate, so the robot batches are today's.
DRAW_INDEX_KEY = "draw_index"
# Row j of a step's chain is seeded (stream_seed(key) << CHAIN_ROW_BITS) | j.
CHAIN_ROW_BITS = 16
CHAIN_ROWS = 1 << CHAIN_ROW_BITS


@dataclass(frozen=True)
class PackConfig:
    """data.train.understanding's pack: its length, its forced fixture steps, its micro-batches.

    micro_batches None packs every micro-batch of a step; a tuple names the
    ones that carry a pack (PACK_MICRO_BATCHES_KEY), checked against the
    step's accumulation by chat_micro_batches.
    """

    tokens: int
    force: tuple[str, ...] = ()
    micro_batches: tuple[int, ...] | None = None
    every: int = 1

    def fixture(self, step: int) -> str | None:
        """The fixture global step `step` packs on every rank, or None for drawn rows."""
        step = int(step)
        return self.force[step] if 0 <= step < len(self.force) else None

    def carries(self, step: int) -> bool:
        """Whether global step `step` carries chat packs (PACK_EVERY_STEPS_KEY).

        Every step at every 1; otherwise the multiples of every, step 0 first,
        and every step force_pack names (a forced worst case runs whatever
        the cadence). The same on every rank: it reads the step only.
        """
        step = int(step)
        return self.every == 1 or step % self.every == 0 or self.fixture(step) is not None

    def step_micro_batches(self, step: int, accumulation: int) -> tuple[int, ...]:
        """The micro-batches of global step `step` that carry a pack: () on a chat-less step."""
        carrying = self.chat_micro_batches(accumulation)
        return carrying if self.carries(step) else ()

    def chat_step(self, step: int) -> int:
        """Global step `step`'s ordinal among the cadence's chat steps: step // every.

        The step itself at every 1. The chat steps, not the global ones, share
        chains (chain_steps, group_key), so at every 2 the 8 chat steps 16g to
        16g + 14 of a rank take the 8 slots of one chain and none is drawn
        without being delivered. A forced step between chat steps
        (force_pack) packs its fixture and draws nothing, whatever its ordinal.
        """
        return int(step) // int(self.every)

    def chat_micro_batches(self, accumulation: int) -> tuple[int, ...]:
        """The micro-batches of a step of `accumulation` that carry a pack, in order.

        Every one when micro_batches is None. Raises unless each stated index is
        below accumulation and the last micro-batch, accumulation - 1, the one
        DDP synchronises, is among them (PACK_MICRO_BATCHES_KEY).
        """
        return check_micro_batches(self.micro_batches, accumulation)

    def chain_steps(self, accumulation: int) -> int:
        """The consecutive chat steps of a rank that share one chain: ceil(accumulation / packs a step).

        1 when every micro-batch packs: the chain is the step's own, accumulation
        packs (S8). A run that names fewer chat micro-batches chains at least
        accumulation packs all the same, over this many chat steps, chat step c
        (chat_step: the global step at every 1) taking its packs from slot
        c % steps (group_key): a chain of the
        step's own 2 packs would discard a straddler every 2 packs, the long
        rows first (measured at chain 2: 0.79 of rows over 9,216 tokens
        delivered, against 0.94 at chain 8).
        """
        return -(-int(accumulation) // len(self.chat_micro_batches(accumulation)))

    def chain_packs(self, accumulation: int) -> int:
        """The packs one chain holds: a step's packs x chain_steps, at least accumulation."""
        return len(self.chat_micro_batches(accumulation)) * self.chain_steps(accumulation)


def check_pack_micro_batches(value, where: str = PACK_MICRO_BATCHES_KEY) -> tuple[int, ...] | None:
    """pack_micro_batches as stated -> a strictly increasing tuple of indices; None is every one."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise ValueError(
            f"{where} is {value!r}.\n"
            "Expected null (every micro-batch carries a pack), or a list of micro-batch "
            "indices that ends with the step's last, accumulation - 1.\n"
            f"Pass a list: {where}=[3,7]."
        )
    values = list(value)
    if not values or not all(isinstance(one, int) and not isinstance(one, bool) and one >= 0
                             for one in values) \
            or any(after <= before for before, after in zip(values, values[1:])):
        raise ValueError(
            f"{where} is {values!r}.\n"
            "Expected at least one micro-batch index, each a non-negative integer, in "
            "increasing order and none twice.\n"
            f"Pass a list: {where}=[3,7]."
        )
    return tuple(int(one) for one in values)


def check_micro_batches(micro_batches: tuple[int, ...] | None,
                        accumulation: int) -> tuple[int, ...]:
    """A run's stated chat micro-batches against its accumulation -> the indices that pack.

    None is every micro-batch. A stated list is refused when an index is not
    below accumulation, or when it leaves out accumulation - 1: that micro-batch's
    backward is the one DDP synchronises (loop.py is_sync; no_sync on the others;
    find_unused_parameters false), so the chat-only parameters must take their
    gradient there, or DDP refuses the next step or leaves them unsynchronised.
    """
    accumulation = int(accumulation)
    if accumulation < 1:
        raise ValueError(
            f"accumulation {accumulation}.\n"
            "Expected at least 1 micro-batch a step.\n"
            "Fix schedule.gradient_accumulation_steps."
        )
    if micro_batches is None:
        return tuple(range(accumulation))
    micro = check_pack_micro_batches(micro_batches)
    if micro[-1] >= accumulation or (accumulation - 1) not in micro:
        raise ValueError(
            f"{PACK_MICRO_BATCHES_KEY} is {list(micro)} at accumulation {accumulation}.\n"
            f"Expected indices below {accumulation}, the last of them {accumulation - 1}: the "
            "step's last micro-batch is the one whose backward DDP synchronises, and the "
            "chat-only parameters (cotrain_head) must take a gradient in it, or DDP refuses "
            "the next step or leaves them unsynchronised across ranks.\n"
            f"State a list ending with {accumulation - 1}, or null for every micro-batch."
        )
    return micro


def pack_log_line(tokens: int, micro_batches: tuple[int, ...] | None, accumulation: int,
                  every: int = 1) -> str:
    """The loader's start line: PACK_LOG_LINE, the chat micro-batches and cadence when stated."""
    config = PackConfig(tokens=int(tokens), micro_batches=micro_batches)
    micro = config.chat_micro_batches(accumulation)
    line = PACK_LOG_LINE.format(tokens=int(tokens), chain=config.chain_packs(accumulation))
    if micro_batches is not None:
        line += PACK_MICRO_LOG_SUFFIX.format(micro=",".join(str(one) for one in micro),
                                             accumulation=int(accumulation),
                                             steps=config.chain_steps(accumulation))
    if int(every) != 1:
        line += PACK_EVERY_LOG_SUFFIX.format(every=int(every))
    return line


def check_pack_every(value, where: str = PACK_EVERY_STEPS_KEY) -> int:
    """pack_every_steps as stated -> a whole number of steps at or above 1; null is 1."""
    if value is None:
        return 1
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"{where} is {value!r}.\n"
            "Expected null or 1 (chat on every step), or a whole number N above 1: chat on "
            "the steps whose number is a multiple of N.\n"
            f"Pass an integer: {where}=2."
        )
    return int(value)


def check_force_pack(force_pack, where: str = FORCE_PACK_KEY) -> tuple[str, ...]:
    """The forced fixtures: one per leading step, or () for none."""
    if force_pack is None:
        return ()
    if isinstance(force_pack, (str, bytes)) or not hasattr(force_pack, "__iter__"):
        raise ValueError(
            f"{where} is {force_pack!r}.\n"
            f"Expected null, or a list of fixtures from {list(FORCE_PACK_FIXTURES)}, one per "
            "leading step.\n"
            f"Pass a list: {where}=[f_tok,f_cnt,f_tok]."
        )
    values = tuple(str(one) for one in force_pack)
    if not values or not set(values) <= set(FORCE_PACK_FIXTURES):
        raise ValueError(
            f"{where} is {list(values)}.\n"
            f"Expected at least one of {list(FORCE_PACK_FIXTURES)}.\n"
            f"Pass a list: {where}=[f_tok,f_cnt,f_tok]."
        )
    return values


def pack_config(pack_tokens, force_pack, *, max_tokens: int,
                pack_micro_batches=None, pack_every_steps=None) -> PackConfig | None:
    """pack_tokens, force_pack, pack_micro_batches, pack_every_steps -> a PackConfig, or None.

    None is one chat row per robot row. pack_tokens is null or a whole number
    of PACK_BLOCK positions no shorter than max_tokens, so every row the
    renderer keeps fits an empty pack and only the step's straddler is ever
    left over. force_pack, pack_micro_batches and a pack_every_steps above 1
    need a pack; the micro-batches are checked against the accumulation where
    it is known (PackConfig.chat_micro_batches).
    """
    force = check_force_pack(force_pack)
    micro = check_pack_micro_batches(pack_micro_batches)
    every = check_pack_every(pack_every_steps)
    if pack_tokens is None:
        if force:
            raise ValueError(
                f"{FORCE_PACK_KEY} is {list(force)} and {PACK_TOKENS_KEY} is null.\n"
                "Expected a pack length: a forced pack fills a pack.\n"
                f"State {PACK_TOKENS_KEY}, or drop {FORCE_PACK_KEY}."
            )
        if micro is not None:
            raise ValueError(
                f"{PACK_MICRO_BATCHES_KEY} is {list(micro)} and {PACK_TOKENS_KEY} is null.\n"
                "Expected a pack length: the micro-batches named carry a pack, and a run "
                "without one carries a chat row beside every robot row.\n"
                f"State {PACK_TOKENS_KEY}, or set {PACK_MICRO_BATCHES_KEY} null."
            )
        if every != 1:
            raise ValueError(
                f"{PACK_EVERY_STEPS_KEY} is {every} and {PACK_TOKENS_KEY} is null.\n"
                "Expected a pack length: the cadence chooses the steps that carry packs, and a "
                "run without one carries a chat row beside every robot row.\n"
                f"State {PACK_TOKENS_KEY}, or set {PACK_EVERY_STEPS_KEY} null."
            )
        return None
    if isinstance(pack_tokens, bool) or not isinstance(pack_tokens, int) \
            or pack_tokens < max(int(max_tokens), PACK_BLOCK) or pack_tokens % PACK_BLOCK:
        raise ValueError(
            f"{PACK_TOKENS_KEY} is {pack_tokens!r} with max_tokens {max_tokens}.\n"
            f"Expected null, or an integer multiple of {PACK_BLOCK} at least max_tokens: "
            "every kept row must fit an empty pack (NEO: 36,864 for both).\n"
            f"State {PACK_TOKENS_KEY}={NEO_PACK_TOKENS} with max_tokens at most that, or null."
        )
    return PackConfig(tokens=int(pack_tokens), force=force, micro_batches=micro, every=every)


def relief_flags(stated: Mapping[str, object]) -> dict[str, bool]:
    """model.<flag> for each RELIEF_FLAGS name -> a bool; absent or null is off."""
    flags = {}
    for name in RELIEF_FLAGS:
        value = stated.get(name)
        if value is not None and not isinstance(value, bool):
            raise ValueError(
                f"model.{name} is {value!r}.\n"
                "Expected true or false.\n"
                f"State model.{name}=false, or true to turn the relief on."
            )
        flags[name] = bool(value)
    return flags


class StepItem(NamedTuple):
    """One loader item in pack mode: global step `step` of one rank.

    indices are the per_device x accumulation draw numbers ResumableSampler
    yields for that step on this rank, in its order; micro-batch m is
    indices[m * per_device : (m + 1) * per_device], the micro-batch today's
    loader would have read there.
    """

    step: int
    indices: tuple[int, ...]


def split_step_item(item: StepItem, per_device: int, accumulation: int) -> list[tuple[int, ...]]:
    """A step item -> its accumulation micro-batches of draw numbers, in sampler order."""
    per_device, accumulation = int(per_device), int(accumulation)
    if len(item.indices) != per_device * accumulation:
        raise ValueError(
            f"step {item.step}'s item holds {len(item.indices)} draws.\n"
            f"Expected per_device x accumulation = {per_device} x {accumulation}.\n"
            "Build step items with the pack-mode batch sampler."
        )
    return [tuple(item.indices[m * per_device:(m + 1) * per_device])
            for m in range(accumulation)]


def step_key(first_index: int, draws: int) -> int:
    """The chain key of a step whose chain is its own: its first draw on this rank, modulo draws.

    The draw number is the sampler's own (sampler.py), unique per pass, step
    and rank and reproduced by a resume, so a step's packs are a pure
    function of (seed, key) and a resumed step packs what the uninterrupted
    run packed. Modulo draws, as today's row reads understanding[index % draws].
    """
    return int(first_index) % int(draws)


def group_key(step: int, steps: int, *, rank: int | None, world: int | None) -> int:
    """Step `step`'s chain key when `steps` steps share one: (step // steps) x world + rank.

    PackConfig.chain_steps > 1 (shape B): the steps of group step // steps on
    one rank draw one chain, each its own slot of it, so no step's first draw
    can key it. The key is the group's and the rank's instead: every (group,
    rank) its own over the run, a pure function of (step, rank, world), so a
    resume anywhere in a group packs what the uninterrupted run packed, and
    the ranks draw different rows. collate_step uses step_key when steps is 1.
    Under a chat cadence (pack_every_steps N) `step` is the chat step's
    ordinal (PackConfig.chat_step, global step // N), so the groups are of
    chat steps: at N 1 the global step itself.
    """
    steps = int(steps)
    if rank is None or world is None or steps < 1 or not 0 <= int(rank) < int(world):
        raise ValueError(
            f"the chain of step {step} spans {steps} steps on rank {rank} of world {world}.\n"
            "Expected at least 1 step and a rank below the world size: a chain shared by "
            "several steps is keyed by its group and the rank.\n"
            "Build the loader with runner.build_loader, which passes both to collate_step."
        )
    return (int(step) // steps) * int(world) + int(rank)


def chain_row_seed(key: int, j: int, *, seed: int) -> int:
    """The seed of row j of the chain keyed `key`: (stream_seed(key) << 16) | j.

    Every (key, j) slot is its own seed, so no row is drawn twice in a pass
    (fresh draws). A Python int seed, so the shift loses nothing.
    """
    j = int(j)
    if not 0 <= j < CHAIN_ROWS:
        raise ValueError(
            f"row {j} of the chain keyed {key}.\n"
            f"Expected 0 to {CHAIN_ROWS - 1}: a step's chain has {CHAIN_ROW_BITS} bits of rows.\n"
            "Lower accumulation or pack_tokens, or widen CHAIN_ROW_BITS with a new stamp."
        )
    return (stream_seed(int(key), seed=int(seed)) << CHAIN_ROW_BITS) | j


def step_item_prefetch(prefetch: int, accumulation: int) -> int:
    """--prefetch, which counts micro-batches, -> step items: ceil(prefetch / accumulation).

    The flag keeps one unit in every mode: 16 at
    accumulation 8 reads 2 steps ahead per worker, not 16.
    """
    prefetch, accumulation = int(prefetch), int(accumulation)
    if prefetch < 1 or accumulation < 1:
        raise ValueError(
            f"prefetch {prefetch} micro-batches at accumulation {accumulation}.\n"
            "Expected both at least 1.\n"
            "Give the loader a positive prefetch."
        )
    return math.ceil(prefetch / accumulation)


def counter_vector(**counts: int) -> torch.Tensor:
    """counts by COUNTERS name -> the int64 [len(COUNTERS)] counters field; absent are 0."""
    unknown = sorted(set(counts) - set(COUNTERS))
    if unknown:
        raise ValueError(
            f"counters {unknown} are not pack counters.\n"
            f"Expected names from {list(COUNTERS)}.\n"
            "Add the counter to COUNTERS."
        )
    return torch.tensor([int(counts.get(name, 0)) for name in COUNTERS], dtype=torch.long)


def _tensor(pack: dict, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    value = pack[name]
    if not isinstance(value, torch.Tensor) or value.dtype != torch.long \
            or tuple(value.shape) != shape:
        got = (type(value).__name__ if not isinstance(value, torch.Tensor)
               else f"{value.dtype} {tuple(value.shape)}")
        raise ValueError(
            f"the pack's {name} is {got}.\n"
            f"Expected torch.int64 {shape}.\n"
            "Build the pack with collate_chat_pack."
        )
    return value


def check_pack(pack: dict, *, pack_tokens: int) -> int:
    """Raise unless one micro-batch's chat pack is the frozen layout; return its rows n.

    Holds the field set, dtypes and shapes; segments 0..n-1 in contiguous runs
    then PAD_SEGMENT to the end; IGNORE_LABEL at each row's first position, so
    the whole-pack label shift equals the per-row one, and on padding;
    segment_targets equal to each row's supervised labels, at least 1; t
    restarting at 0 on each row and every position 0 on padding; one grid per
    image matching its size and covering every <IMG_CONTEXT>; and the
    counters' rows, tokens and images.
    """
    fields = set(pack)
    if fields != set(PACK_FIELDS):
        raise ValueError(
            f"the pack holds {sorted(fields)}; {sorted(fields ^ set(PACK_FIELDS))} differ.\n"
            f"Expected exactly {list(PACK_FIELDS)}.\n"
            "Build the pack with collate_chat_pack."
        )
    size = int(pack_tokens)
    ids = _tensor(pack, PACK_IDS, (1, size))[0]
    labels = _tensor(pack, PACK_LABELS, (1, size))[0]
    segments = _tensor(pack, PACK_SEGMENTS, (1, size))[0]
    positions = _tensor(pack, PACK_POSITIONS, (3, 1, size))[:, 0]
    real = int((segments != PAD_SEGMENT).sum())
    body, tail = segments[:real], segments[real:]
    steps = body[1:] - body[:-1]
    if real < 1 or int(body[0]) != 0 or bool((tail != PAD_SEGMENT).any()) \
            or bool((body < 0).any()) or bool(((steps != 0) & (steps != 1)).any()):
        raise ValueError(
            f"the pack's segments run {torch.unique_consecutive(segments).tolist()[:8]}.\n"
            f"Expected 0, 1, ..., n-1, each row one contiguous run in order, and {PAD_SEGMENT} "
            "only after the last row.\n"
            "Lay the rows out in order and pad at the end."
        )
    opens = torch.ones(real, dtype=torch.bool)
    opens[1:] = steps != 0
    rows = int(body[-1]) + 1
    if bool((labels[:real][opens] != IGNORE_LABEL).any()):
        first = int(torch.nonzero(opens & (labels[:real] != IGNORE_LABEL))[0])
        raise ValueError(
            f"segment {int(body[first])} opens with label {int(labels[first])} at {first}.\n"
            f"Expected {IGNORE_LABEL}: the whole-pack shift would score the previous row's "
            "last token against this row's first.\n"
            "Keep every row's masked header."
        )
    if bool((labels[real:] != IGNORE_LABEL).any()) or bool((ids[real:] != PAD_ID).any()):
        raise ValueError(
            f"the pack's padding from {real} carries a label or an id other than {PAD_ID}.\n"
            f"Expected id {PAD_ID} and label {IGNORE_LABEL} on every padding position.\n"
            "Pad with CHAT_PAD_ID and the ignore index."
        )
    supervised = labels[:real] != IGNORE_LABEL
    counts = torch.bincount(body[supervised], minlength=rows)
    targets = _tensor(pack, PACK_SEGMENT_TARGETS, (rows,))
    if not torch.equal(targets, counts) or bool((counts < 1).any()):
        raise ValueError(
            f"segment_targets is {targets.tolist()[:8]} and the labels supervise "
            f"{counts.tolist()[:8]} per row.\n"
            "Expected one count per row, equal to its supervised labels, each at least 1.\n"
            "Count labels != -100 per segment in collate_chat_pack."
        )
    inside = ids == IMG_CONTEXT_ID
    if bool((positions[:, real:] != 0).any()) or bool((positions[0, :real][opens] != 0).any()) \
            or bool((positions[1:, ~inside] != 0).any()):
        raise ValueError(
            "the pack's positions do not restart per row.\n"
            "Expected t = 0 at each row's first position, h = w = 0 off <IMG_CONTEXT>, and "
            "t = h = w = 0 on padding.\n"
            "Compute positions per segment (chat_positions with segments)."
        )
    images = pack[PACK_IMAGES]
    if not isinstance(images, (list, tuple)) or not all(
            isinstance(image, torch.Tensor) and image.dtype == torch.uint8 and image.dim() == 3
            and int(image.shape[0]) == 3 and int(image.shape[1]) % TOKEN_PIXELS == 0
            and int(image.shape[2]) % TOKEN_PIXELS == 0 for image in images):
        raise ValueError(
            "the pack's images are not a list of uint8 [3, H, W].\n"
            f"Expected each image at its own size, H and W multiples of {TOKEN_PIXELS}.\n"
            "List the rows' images in <IMG_CONTEXT> order."
        )
    grids = _tensor(pack, PACK_GRID_HW, (len(images), 2))
    sizes = torch.tensor([[int(image.shape[1]), int(image.shape[2])] for image in images],
                         dtype=torch.long).reshape(-1, 2)
    tokens = int((grids[:, 0] * grids[:, 1]).sum())
    if not torch.equal(grids, sizes // TOKEN_PIXELS) or tokens != int(inside.sum()):
        raise ValueError(
            f"grid_hw covers {tokens} image tokens; the ids hold {int(inside.sum())}.\n"
            f"Expected grid_hw = each image's (H, W) // {TOKEN_PIXELS}, covering every "
            "<IMG_CONTEXT>.\n"
            "Emit grid_hw from the images collate_chat_pack lists."
        )
    counters = _tensor(pack, PACK_COUNTERS, (len(COUNTERS),))
    held = dict(zip(COUNTERS, counters.tolist()))
    if min(held.values()) < 0 or (held[COUNTER_ROWS], held[COUNTER_TOKENS],
                                  held[COUNTER_IMAGES]) != (rows, real, len(images)):
        raise ValueError(
            f"the pack's counters are {held} for {rows} rows, {real} tokens and "
            f"{len(images)} images.\n"
            "Expected non-negative counts whose rows, tokens and images are the pack's own.\n"
            "Count while packing, with counter_vector."
        )
    return rows
