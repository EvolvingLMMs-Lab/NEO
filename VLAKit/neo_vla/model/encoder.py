"""The prefix encoder: prompt, images and state in, PrefixContext out.

It builds the v7 prompt row -- Earlier and Now blocks of views, the rendered
text, the six <state> columns (start then now; each arm, then the body) and
the close -- runs the vision tower over both blocks' images, writes each
view's calibration and camera rotation into the twelve columns after its
<cam_k>, writes the state banks' tokens after the instruction, runs the
backbone under a block-causal mask, and returns the per-layer keys and values
with the suffix's rotary geometry.

encode takes suffix_pos as a required keyword with no default: the suffix
layout is the decoder's to state and the encoder's to honour. The anchor is
per-sample and the position vector is not, so the rotary phase of a sample
never depends on the batch it arrived in. Freezing is done to parameters: the
encoder is never wrapped in torch.no_grad.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from .. import constants
# chat_pack is imported as a module and read at call time: it imports the
# backbone's image ids, so a process that imports it first runs this package's
# __init__ while chat_pack is half built, and a from-import of one of its names
# would fail there.
from ..data import chat_pack
from ..data.images import PIXEL_HIGH, PIXEL_LOW, check_convention, tower_norm
from ..data.rays import CALIBRATION_COLUMNS
from ..engine.hostcopy import host_of, remember, to_host_device
from .backbone.neo_llm import (
    IMG_CONTEXT_ID,
    IMG_END_ID,
    IMG_START_ID,
    BackboneConfig,
    NeoBackbone,
    rope_cos_sin,
)
from .backbone.vision import VisionTower
from .prefix import PrefixContext
from .tokenizers import (
    ANCHOR_IDS,
    EXTRINSICS_DIM,
    EXTRINSICS_TOKENS_PER_VIEW,
    INTRINSICS_TOKENS_PER_VIEW,
    MAX_CAMERAS,
    N_ROLES,
    RETIRED_ANCHOR_IDS,
    ROLE_NOW,
    ROLE_START,
    STATE_ABSENT,
    STATE_NO_HISTORY,
    PrefixTokenizers,
    StateInputs,
)

logger = logging.getLogger(__name__)

INTRINSICS_ID = ANCHOR_IDS["<intrinsics>"]
EXTRINSICS_ID = ANCHOR_IDS["<extrinsics>"]
STATE_ID = ANCHOR_IDS["<state>"]
CAMERA_IDS = tuple(ANCHOR_IDS[f"<cam_{view}>"] for view in range(MAX_CAMERAS))

# The scaffold's own text and its fixed widths, stated in neo_vla/constants.py
# and re-exported here. PREFIX_BEFORE_TEXT, PREFIX_AFTER_TEXT and PREFIX_LEN are
# the shipped shape's (3 views of 8 x 11, 2 arms and the body, 256 text tokens)
# under the NEOpp-2B tokenizer; a smaller test shape computes its own from the
# same rule.
HISTORY_HEADER = constants.HISTORY_HEADER
NOW_HEADER = constants.NOW_HEADER
STATE_LINE_OPEN = constants.STATE_LINE_OPEN
PREFIX_BEFORE_TEXT = constants.PREFIX_BEFORE_TEXT
PREFIX_AFTER_TEXT = constants.PREFIX_AFTER_TEXT
PREFIX_LEN = constants.PREFIX_LEN
FREQUENCY_OPEN = constants.FREQUENCY_OPEN

# An instruction holding one of these would drop a scaffold id into the text
# region. The retired <scale> row is here too: it is never handed out again, and
# a prompt that still holds it was rendered for a layout this tree does not read.
# So is <rays>, which no v7 row holds (tokenizers.ANCHOR_IDS says why it stays).
RESERVED_PROMPT_IDS = frozenset(
    (IMG_CONTEXT_ID, IMG_START_ID, IMG_END_ID, *ANCHOR_IDS.values(),
     *RETIRED_ANCHOR_IDS.values())
)

# Section 1 of the prompt template opens the row and the chat close ends it.
# Sections 2 to 5 are rendered into the prompt by the data side and share the
# max_text_tokens budget; the headers of the two view blocks and the line the
# state tokens stand on are encoder constants, not template keys.
CHAT_OPEN = "<|im_start|>user\n"
AXIS_HEADER = "1. axis information:\n"
CHAT_CLOSE = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

# Distinct instruction strings repeat across steps, and re-tokenising them is
# the only per-sample host work in this module.
PROMPT_CACHE_LIMIT = 65536

# The texts whose ids stand for the tokenizer in a checkpoint's prefix stamp
# (layout_stamp): the scaffold's own pieces, the frequency and instruction
# openings, a line ending in a full stop, '#' runs (the merge a newer
# transformers adds), digits and CJK. A tokenizer that reads any of them
# differently lays down another row.
TOKENIZER_PROBE_TEXTS = (
    CHAT_OPEN, AXIS_HEADER, HISTORY_HEADER, NOW_HEADER, STATE_LINE_OPEN, CHAT_CLOSE,
    FREQUENCY_OPEN + "NA", FREQUENCY_OPEN + "16.67 Hz\n", constants.INSTRUCTION_OPEN,
    "Instruction: pick up the bowl.\n", "## x", "#include", "3.75 Hz, 20 Hz, 1200 mm",
    "\u628a\u7897\u62ff\u8d77\u6765",
)


def block_columns(firsts: list[int], width: int) -> torch.Tensor:
    """Block starts and one block's width -> every column, in block order.

    Explicit indexes rather than a boolean mask, because a mask would be right
    only while a view's columns stay contiguous and in view order.
    """
    return torch.tensor(
        [column for first in firsts for column in range(first, first + width)],
        dtype=torch.long,
    )


def token_positions(
    ids: torch.Tensor, grid_width: torch.Tensor, tokens_per_image: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ids [B, S] and each column's image width [B, S] -> t, h and w [B, S].

    The checkpoint's convention, derived from content rather than from a fixed
    layout: a text token advances t and keeps h and w at zero, and the tokens
    of one image share a t and carry their rank in that image's own grid. Both
    the robot prefix and the chat row come through here, so the two cannot
    drift apart.
    """
    starts = (ids == IMG_START_ID).long()
    shifted = torch.cat([torch.zeros_like(starts[:, :1]), starts[:, :-1]], dim=1)
    inside = ids == IMG_CONTEXT_ID
    ordinary = (~inside).long()
    t_index = (shifted + ordinary).cumsum(1) - 1

    rank = (inside.long().cumsum(1) - 1) % tokens_per_image
    zero = torch.zeros_like(rank)
    h_index = torch.where(inside, rank // grid_width, zero)
    w_index = torch.where(inside, rank % grid_width, zero)
    return t_index, h_index, w_index


# What VisionTower.input_norm is under the NEO-native pixel convention (branch
# neo-native): ImageNet mean and std at the tower's entry. A native chat half
# is NEO's own pixels only behind it; see understanding.NATIVE_PIXEL_CONVENTION.
NATIVE_CHAT_TOWER_NORM = "imagenet"

# NEO's pad pseudo-documents: the padding of a pack is cut into documents this
# long (split_size of _offsets_to_doc_ids_tensor in NEO's
# modeling_internvl_chat.py), each attending causally to itself only.
PAD_DOCUMENT_TOKENS = 1024


def chat_positions(
    ids: torch.Tensor, grids: torch.Tensor, segments: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ids [B, S] and each image's token grid [K, 2] -> t, h and w [B, S].

    NEO's own indexes for a chat row with images of any size: t advances on
    every column that is not <IMG_CONTEXT> and once more after <img>, so an
    image's tokens share one t; h and w are each token's row and column in
    its own image's grid (get_thw_indexes and build_abs_positions_from_grid_hw
    in the checkpoint's modeling_neo_chat.py). The K grids are in the order
    the <IMG_CONTEXT> columns are met, row by row, and must cover them all.
    token_positions is not reused: its rank modulo one grid size is exact
    only while every image in the call has the same number of tokens.

    segments [B, S], given for a chat pack,
    numbers the rows packed back to back, PAD_SEGMENT on the padding. t then
    restarts at 0 on each row's first column, so every row gets exactly the
    indexes it gets alone, as NEO's collator computes t one segment at a time
    (get_cu_seqlens_and_indexes in its dataset_interleaved_iterable.py); h and
    w need no restart, being per image already. Padding is 0 in all three.
    """
    starts = (ids == IMG_START_ID).long()
    shifted = torch.cat([torch.zeros_like(starts[:, :1]), starts[:, :-1]], dim=1)
    inside = ids == IMG_CONTEXT_ID
    if segments is None:
        t_index = (shifted + (~inside).long()).cumsum(1) - 1
    else:
        if tuple(segments.shape) != tuple(ids.shape):
            raise ValueError(
                f"segments is {tuple(segments.shape)} for ids {tuple(ids.shape)}.\n"
                "Expected one segment number per column.\n"
                "Pass the pack's segments field unchanged."
            )
        segments = segments.to(ids.device)
        first = torch.ones_like(inside)
        first[:, 1:] = segments[:, 1:] != segments[:, :-1]
        # A row opens on its own first column, never on the previous row's
        # last one, so the <img> shift does not cross the boundary.
        steps = torch.where(first, 0, shifted) + (~inside).long()
        running = steps.cumsum(1)
        columns = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
        opening = torch.where(first, columns, 0).cummax(1).values
        t_index = running - (running - steps).gather(1, opening) - 1
        t_index = t_index.masked_fill(segments == chat_pack.PAD_SEGMENT, 0)
    h_index = torch.zeros_like(ids)
    w_index = torch.zeros_like(ids)
    if grids.numel():
        counts = (grids[:, 0] * grids[:, 1]).to(ids.device)
        total = int(counts.sum())
        widths = grids[:, 1].to(ids.device).repeat_interleave(counts, output_size=total)
        firsts = (counts.cumsum(0) - counts).repeat_interleave(counts, output_size=total)
        rank = torch.arange(total, device=ids.device) - firsts
        h_index[inside] = rank // widths
        w_index[inside] = rank % widths
    return t_index, h_index, w_index


def chat_mask_ids(
    segments: torch.Tensor, ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A pack's segments and ids [1, S] -> its document and image numbers [S], int64.

    doc is the row a column belongs to, with the padding cut into
    PAD_DOCUMENT_TOKENS-long pseudo-documents numbered after the last row, as
    NEO's _offsets_to_doc_ids_tensor cuts its pad tail. They exist only here
   : segments keeps PAD_SEGMENT on the padding for
    the loss. img numbers the images over the whole pack, 1 upward, on
    <IMG_CONTEXT> columns and 0 elsewhere, so <img> and </img> stay causal;
    NEO writes -1 off the image, and its rule reads only img > 0. Computed on
    segments' device without a host read.
    """
    if segments.dim() != 2 or int(segments.shape[0]) != 1 or tuple(ids.shape) != tuple(
            segments.shape):
        raise ValueError(
            f"segments is {tuple(segments.shape)} and ids {tuple(ids.shape)}.\n"
            "Expected both [1, S]: a pack is one sequence.\n"
            "Pass the pack's segments and input_ids fields unchanged."
        )
    segment, token = segments[0], ids[0].to(segments.device)
    real = segment != chat_pack.PAD_SEGMENT
    columns = torch.arange(segment.shape[0], device=segment.device)
    pad_rank = columns - real.sum()
    pseudo = segment.max().clamp(min=-1) + 1 + pad_rank.div(
        PAD_DOCUMENT_TOKENS, rounding_mode="floor")
    doc = torch.where(real, segment, pseudo)
    starts = (token == IMG_START_ID).long()
    shifted = torch.cat([starts.new_zeros(1), starts[:-1]])
    img = torch.where(token == IMG_CONTEXT_ID, shifted.cumsum(0), 0)
    return doc, img


def chat_mask_mod(doc: torch.Tensor, img: torch.Tensor):
    """NEO's packed attention rule over doc and img, as a flex_attention mask_mod.

    allowed(q, kv) = doc[q] == doc[kv] and (q >= kv, or q is an image token
    and kv a token of the same image). Ported by hand from NEO's
    create_flex_mask_padding (NEO++ trainer, denghanming's neo_train_plus,
    internvl/model/internvl_moe_chat/modeling_internvl_chat.py:88-127; the
    Neo_plus_train copy, wupenghao's, at :89-127), whose
    or_masks(and_masks(causal, samedoc), sameimg & samedoc) is the same set
    written as one function. Within one row it is encode_chat_rows' dense rule
    (same t or causal): text t is unique and one image's tokens share one t.
    The same-document term is what keeps a row from seeing another once t
    restarts per row.
    """

    def allowed(batch, head, query, key):
        return (doc[query] == doc[key]) & (
            (query >= key) | ((img[query] > 0) & (img[query] == img[key])))

    return allowed


def chat_dense_mask(segments: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """The pack's rule as a dense bool [S, S], True where a query may read a key.

    The reference the BlockMask kernel is tested against (SDPA under this
    mask); the kernel itself, the CPU rehearsal's included, takes the
    BlockMask. It is S * S bytes, so never at a production pack.
    """
    doc, img = chat_mask_ids(segments, ids)
    columns = torch.arange(doc.shape[0], device=doc.device)
    return chat_mask_mod(doc, img)(0, 0, columns[:, None], columns[None, :])


def chat_block_mask(segments: torch.Tensor, ids: torch.Tensor) -> BlockMask:
    """A pack's segments and ids [1, S] -> NEO's BlockMask, for every head and the batch.

    BLOCK_SIZE PACK_BLOCK, so a pack of pack_tokens columns needs no padding
    of q, k and v (NEO pads to it). Built once per pack and handed to every
    layer. On cuda the block map is computed compiled, as NEO builds it
    (_compile=True): the uncompiled builder evaluates the rule over all
    S * S pairs at once. Elsewhere the uncompiled one, at test sizes.
    """
    length = int(segments.shape[-1])
    if length % chat_pack.PACK_BLOCK:
        raise ValueError(
            f"the pack is {length} columns long.\n"
            f"Expected a multiple of {chat_pack.PACK_BLOCK}, the BlockMask block.\n"
            "State data.train.understanding.pack_tokens as a whole number of blocks."
        )
    doc, img = chat_mask_ids(segments, ids)
    return create_block_mask(
        chat_mask_mod(doc, img), B=None, H=None, Q_LEN=length, KV_LEN=length,
        device=segments.device, BLOCK_SIZE=chat_pack.PACK_BLOCK, _compile=segments.is_cuda,
    )


def check_native_tower(vision: nn.Module) -> None:
    """Raise unless the tower normalises its input to ImageNet's, as NEO's chat pixels need.

    The gate both chat paths pass, rows (encode_chat_rows) and pack
    (encode_chat_pack): without it their pixels are not NEO's.
    """
    norm = vars(vision).get("input_norm")
    if norm != NATIVE_CHAT_TOWER_NORM:
        raise ValueError(
            f"native chat rows reached a vision tower whose input_norm is {norm!r}.\n"
            f"Expected {NATIVE_CHAT_TOWER_NORM!r}: NEO's chat pixels are ToTensor plus "
            "ImageNet mean and std, and this path hands the tower [-1, 1].\n"
            "Build with model.vision.pixel_convention neo_native."
        )


def chat_image_tokens(
    vision: nn.Module, images: list[torch.Tensor], shapes: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """uint8 images [3, H, W] and their sizes [K, 2] -> their tokens, image after image.

    The one place native chat images are scaled, for rows and packs alike:
    each into the [-1, 1] contract, in float32, then through the same vision
    tower call the robot path makes, grouped by size. The tokens come back in
    the images' order, [sum h * w, hidden].
    """
    tokens: list[torch.Tensor | None] = [None] * len(images)
    for shape in torch.unique(shapes, dim=0):
        height, width = int(shape[0]), int(shape[1])
        members = [
            index for index, image in enumerate(images)
            if int(image.shape[-2]) == height and int(image.shape[-1]) == width
        ]
        stacked = torch.stack([images[index] for index in members]).to(device)
        scaled = stacked.to(torch.float32) / 255.0 * (PIXEL_HIGH - PIXEL_LOW) + PIXEL_LOW
        encoded = vision(scaled)
        for slot, index in enumerate(members):
            tokens[index] = encoded[slot]
    return torch.cat(tokens, dim=0)


class PrefixEncoder(nn.Module):
    """The backbone, the vision tower, the prompt scaffold and the state bank.

    The scaffold is built once at construction and its fixed part has constant
    column indexes for the run: the prefix length Lp is len(pre) +
    max_text_tokens + len(post) and never depends on the batch, because a
    prefix whose length moved would move every suffix rotary phase with it. The
    <state> columns follow the instruction, so their indexes are the one part
    of the row computed per sample, from the instruction's length. The body's
    two columns are always laid down; a key without a body masks them.
    """

    # These names are part of the interface: build.apply_trainable_mode freezes
    # encoder.backbone, and the tokenizers sit outside it so they keep training.
    backbone: NeoBackbone
    vision: VisionTower
    tokenizers: PrefixTokenizers

    def __init__(
        self,
        *,
        config: BackboneConfig,
        tokenizer_path: str | Path,
        n_views: int,
        n_slots: int,
        max_grid: tuple[int, int],
        max_text_tokens: int,
        intrinsics_mode: str,
        pixel_convention: str,
        dtype: torch.dtype,
    ):
        super().__init__()
        if not 1 <= int(n_views) <= MAX_CAMERAS:
            raise ValueError(
                f"n_views is {n_views}, expected 1..{MAX_CAMERAS}.\n"
                "A ninth camera needs a ninth reserved anchor row, which freezes "
                "after pretraining; widen ANCHOR_IDS deliberately."
            )
        self.config = config
        self.n_views = int(n_views)
        self.n_slots = int(n_slots)
        self.max_grid = (int(max_grid[0]), int(max_grid[1]))
        self.max_text_tokens = int(max_text_tokens)
        self.intrinsics_mode = str(intrinsics_mode)
        # model.vision.pixel_convention: what the tower does with the [-1, 1]
        # pixels it is handed, and what a checkpoint of this model states.
        self.pixel_convention = check_convention(
            pixel_convention, "model.vision.pixel_convention"
        )
        self.pad_token_id = int(config.pad_token_id)

        self.backbone = NeoBackbone(config).to(dtype)
        self.vision = VisionTower(
            vision_hidden=config.vision_hidden,
            backbone_hidden=config.hidden_size,
            patch=config.vision_patch,
            downsample_ratio=config.downsample_ratio,
            rope_theta=config.vision_rope_theta,
            input_norm=tower_norm(self.pixel_convention),
        ).to(dtype)
        self.tokenizers = PrefixTokenizers(
            config.hidden_size,
            intrinsics_mode=self.intrinsics_mode,
            n_slots=self.n_slots,
        ).to(dtype)
        # The chat half's loader, so the robot prompt and the chat rows read one
        # tokenizer under one check (DESIGN 3.1): a NEO tokenizer must give
        # NEO's special ids and split '##' as NEO's trainer did, and one that
        # merges '#' is refused here too, not only when a chat half is built.
        # Stage 1 and the server have no chat half.
        from ..data.understanding import load_chat_tokenizer

        self.text_tokenizer = load_chat_tokenizer(tokenizer_path)
        # Each prompt's (text ids, line ids); see prompt_ids.
        self.prompt_cache: dict[str, tuple[list[int], list[int]]] = {}
        self.logged_prefix_len = False
        # trainer.trim_prefix: 0 runs every prefix at Lp; a positive bucket cuts
        # the trailing pad to the batch's longest row rounded up to it (encode).
        # Not part of the layout: the server never sets it.
        self.trim_bucket = 0
        self.logged_widths: set[int] = set()
        # The chat pack's memory relief (chat_pack.RELIEF_FLAGS), off unless
        # build_policy reads model.chat_offload_checkpoints or
        # model.chat_mlp_chunk on. A memory setting like the backbone's
        # gradient_checkpointing: no layout, no parameter, no stamp.
        self.chat_relief = dict.fromkeys(chat_pack.RELIEF_FLAGS, False)
        self.build_scaffold()
        logger.info(
            "native prefix: Earlier and Now blocks of %d views of %d columns, %d "
            "calibration and %d rotation columns each, "
            "%d instruction tokens, %d state columns, Lp %d (%d before the text, "
            "%d after), intrinsics %s",
            self.n_views,
            self.tokens_per_view,
            self.tokens_per_calibration,
            self.tokens_per_extrinsics,
            self.max_text_tokens,
            self.n_state_columns,
            self.prefix_len,
            len(self.pre_ids),
            len(self.post_ids),
            self.intrinsics_mode,
        )

    @property
    def n_blocks(self) -> int:
        """View blocks in the row: an Earlier and a Now block of every camera slot."""
        return 2 * self.n_views

    @property
    def n_state_columns(self) -> int:
        """<state> columns: one per role and arm or body, role-major, the body last in each role.

        N_ROLES * (n_slots + 1), so 6 at the shipped shape: (start, arm 0),
        (start, arm 1), (start, body), (now, arm 0), (now, arm 1), (now, body).
        """
        return N_ROLES * (self.n_slots + 1)

    def build_scaffold(self) -> None:
        """Lay down the fixed part of the row and record its column indexes.

        pre_ids is the chat open, section 1's header, then "Earlier:" and one
        block per camera slot for the frames at the history step p, then "Now:"
        and one block per slot for the current frames. Block b names camera
        b % n_views: a past view and a current one of the same camera share the
        anchor, and only the header and the position tell them apart. A block
        reads <cam_k>, then its calibration, then its camera rotation, then its
        picture, so nothing about a camera is carried past its picture. Every
        sample keeps every block; an absent view, a view absent at p and the
        whole Earlier half of a draw without history are masked, never removed.

        post_ids is a newline, the <state> columns and the chat close: the width
        the row reserves after the text. The row is pre_ids + the text's line
        (line_ids: the instruction and its newline, tokenised as one string) +
        after_line_ids (the <state> columns and the close), padded at the end,
        so the state columns follow the instruction and move with its length.
        """
        image_block = [IMG_START_ID] + [IMG_CONTEXT_ID] * self.tokens_per_view + [IMG_END_ID]
        calibration_block = [INTRINSICS_ID] * self.tokens_per_calibration
        rotation_block = [EXTRINSICS_ID] * self.tokens_per_extrinsics
        newline = self.text_ids("\n")
        row = self.text_ids(CHAT_OPEN) + self.text_ids(AXIS_HEADER)
        self.image_columns: list[int] = []
        self.intrinsics_columns: list[int] = []
        self.extrinsics_columns: list[int] = []
        self.view_spans: list[tuple[int, int]] = []
        self.header_spans: list[tuple[int, int]] = []
        for block in range(self.n_blocks):
            if block in (0, self.n_views):
                header = HISTORY_HEADER if block == 0 else NOW_HEADER
                self.header_spans.append((len(row), len(row) + len(self.text_ids(header))))
                row = row + self.text_ids(header)
            start = len(row)
            row = row + [CAMERA_IDS[block % self.n_views]]
            # Straight after the anchor that names the camera, so nothing
            # stands between the numbers and the view they describe.
            self.intrinsics_columns.append(len(row))
            row = row + calibration_block
            # And the pose straight after the calibration: the two describe
            # one camera, and a reader that found them apart would have to
            # learn which picture the second block belongs to.
            self.extrinsics_columns.append(len(row))
            row = row + rotation_block
            self.image_columns.append(len(row) + 1)
            row = row + image_block + newline
            # A view's span covers its <cam_k> and its numbers too: masking only
            # the image would leave a camera named, and described, for a camera
            # the sample does not have.
            self.view_spans.append((start, len(row)))
        self.pre_ids = row
        self.state_line_ids = self.text_ids(STATE_LINE_OPEN)
        self.after_line_ids = [STATE_ID] * self.n_state_columns + self.text_ids(CHAT_CLOSE)
        self.post_ids = self.state_line_ids + self.after_line_ids
        self.register_buffer(
            "image_column_index",
            block_columns(self.image_columns, self.tokens_per_view),
            persistent=False,
        )
        self.register_buffer(
            "intrinsics_column_index",
            block_columns(self.intrinsics_columns, self.tokens_per_calibration),
            persistent=False,
        )
        self.register_buffer(
            "extrinsics_column_index",
            block_columns(self.extrinsics_columns, self.tokens_per_extrinsics),
            persistent=False,
        )
        # The same indexes kept on the host (a plain attribute, which .to()
        # leaves alone): the scaffold checks of embed_prefix read the host ids.
        self.host_columns = {
            "image": block_columns(self.image_columns, self.tokens_per_view),
            "intrinsics": block_columns(self.intrinsics_columns, self.tokens_per_calibration),
            "extrinsics": block_columns(self.extrinsics_columns, self.tokens_per_extrinsics),
        }

    def text_ids(self, text: str) -> list[int]:
        """Tokenise one piece of text, adding no special tokens of its own."""
        return self.text_tokenizer(text, add_special_tokens=False)["input_ids"]

    def instruction_ids(self, prompt: str) -> list[int]:
        """One rendered prompt (sections 2 to 5) -> its ids alone, checked.

        What the instruction budget counts, the same count the cache build's fit
        makes. The row holds line_ids, which NEO's tokenizer makes differ from
        these only where the text ends in punctuation. Every check prompt_ids
        makes has run.
        """
        return self.prompt_ids(prompt)[0]

    def line_ids(self, prompt: str) -> list[int]:
        """One rendered prompt -> the ids the row holds between pre_ids and the <state> columns.

        The prompt and the newline that ends its line, tokenised as one string,
        as NEO's tokenizer reads that text: a prompt ending in punctuation ends
        in one merged id ('.' then '\\n' is 624, not 13 then 198); any other
        ends in the newline's own id. Every check prompt_ids makes has run.
        """
        return self.prompt_ids(prompt)[1]

    def prompt_ids(self, prompt: str) -> tuple[list[int], list[int]]:
        """One rendered prompt -> (its ids alone, its line's ids), refusing to truncate it.

        The prompt must open with the frequency section as a string. Checked on
        the text rather than on the ids: "Frequency: NA" and "Frequency: 30 Hz"
        tokenise their shared opening differently at its last two ids, so an id
        prefix would refuse a valid prompt. A v5 prompt ("2. SCALE ...", or
        sections 3 to 5 alone) was rendered for a layout this tree does not read.
        The text alone is held to max_text_tokens, and the line to the columns
        the row reserves for it (max_text_tokens plus the newline's).
        """
        cached = self.prompt_cache.get(prompt)
        if cached is not None:
            return cached
        if not isinstance(prompt, str) or not prompt.startswith(FREQUENCY_OPEN):
            raise ValueError(
                f"the prompt does not open with the frequency section: {str(prompt)[:60]!r}.\n"
                f"Expected a template v7 rendering, which starts {FREQUENCY_OPEN!r}.\n"
                "Re-render the prompt cache and the eval frames under policy.yaml."
            )
        if prompt != prompt.rstrip():
            raise ValueError(
                f"the prompt ends with whitespace: {prompt[-40:]!r}.\n"
                "Expected it to end with the task: the <state> columns are "
                "counted from the text's length, so a trailing space or newline moves them.\n"
                "Render the prompt through neo_vla.data.prompt.render, which strips the task."
            )
        ids = self.text_ids(prompt)
        if len(ids) > self.max_text_tokens:
            raise ValueError(
                f"instruction is {len(ids)} tokens, expected at most "
                f"{self.max_text_tokens}.\nRaise model.prefix.max_text_tokens; "
                "pretraining and finetuning must share the value."
            )
        line = self.text_ids(prompt + STATE_LINE_OPEN)
        for held in (ids, line):
            if not RESERVED_PROMPT_IDS.isdisjoint(held):
                raise ValueError(
                    f"instruction holds a reserved scaffold id: {prompt!r}.\n"
                    "Expected text only: the tokenizer matches added tokens in plain "
                    "text, and every column index here is a scaffold constant or is "
                    "counted from the text's length."
                )
        reserved = self.max_text_tokens + len(self.state_line_ids)
        if len(line) > reserved:
            raise ValueError(
                f"the instruction and its newline are {len(line)} tokens, expected at most "
                f"{reserved}: the row reserves max_text_tokens {self.max_text_tokens} and "
                f"the newline's {len(self.state_line_ids)}.\n"
                "Raise model.prefix.max_text_tokens; pretraining and finetuning must share "
                "the value."
            )
        if len(self.prompt_cache) < PROMPT_CACHE_LIMIT:
            self.prompt_cache[prompt] = (ids, line)
        return ids, line

    def text_lengths(self, prompts: list[str]) -> torch.Tensor:
        """B prompts -> L [B] long: the columns each text's line takes before its newline's.

        len(line_ids) - len(state_line_ids). For a prompt whose newline keeps
        its own id that is the prompt's token count, as the budget counts it;
        for one ending in punctuation that NEO's tokenizer merges with the
        newline, one fewer. Column 636 + L at the shipped shape holds the id
        that ends the line. Never carried in the batch.
        """
        return torch.tensor(
            [len(self.line_ids(prompt)) - len(self.state_line_ids) for prompt in prompts],
            dtype=torch.long,
        )

    def state_columns(self, lengths: torch.Tensor) -> torch.Tensor:
        """L [B] -> the <state> columns [B, n_state_columns], role-major, each role's body last.

        L is text_lengths(prompts). len(pre_ids) + L + the newline + arange:
        the columns follow the text's line, so two samples with texts of
        different lengths hold their states at different columns. At the
        shipped shape that is PREFIX_BEFORE_TEXT + L + 1 + arange(6), so
        637 + L .. 642 + L.
        """
        offsets = torch.arange(self.n_state_columns, device=lengths.device)
        first = len(self.pre_ids) + len(self.state_line_ids)
        return first + lengths.long()[:, None] + offsets[None, :]

    def prefix_width(self, prompts: list[str]) -> int:
        """The columns encode runs these prompts' prefix at: Lp, or the cut of trim_bucket.

        Every row is pre_ids + its text's line + after_line_ids and then pad,
        so no real column lies past the longest row. That row's length rounded
        up to trim_bucket, capped at Lp, keeps every real column and leaves a
        compiled layer a handful of widths. Counted on the host from the cached
        token ids, so it costs no device synchronisation.
        """
        if self.trim_bucket <= 0:
            return self.prefix_len
        longest = max(len(self.pre_ids) + len(self.line_ids(prompt)) + len(self.after_line_ids)
                      for prompt in prompts)
        bucket = int(self.trim_bucket)
        return min(self.prefix_len, -(-longest // bucket) * bucket)

    @property
    def prefix_len(self) -> int:
        """Lp, fixed at construction and logged there."""
        return len(self.pre_ids) + self.max_text_tokens + len(self.post_ids)

    def layout_stamp(self) -> dict:
        """What a checkpoint states about the prefix this encoder lays down (R22 at every load).

        n_views, max_grid, max_text_tokens, the tower's pixels per token and the
        tokenizer create no parameter, so a strict load would take a step built
        under one prefix into a build of another (a --config override, a run
        config rewritten on relaunch) and serve it on other columns and another
        token grid. The numbers, Lp, the scaffold ids and the ids the tokenizer
        gives TOKENIZER_PROBE_TEXTS; JSON types only, so the stamp read back
        from meta.json compares equal.
        """
        return {
            "n_views": self.n_views,
            "n_slots": self.n_slots,
            "max_grid": list(self.max_grid),
            "max_text_tokens": self.max_text_tokens,
            "pixels_per_token": int(self.vision.pixels_per_token),
            "prefix_len": int(self.prefix_len),
            "scaffold": [[int(i) for i in self.pre_ids], [int(i) for i in self.post_ids]],
            "tokenizer": [[text, [int(i) for i in self.text_ids(text)]]
                          for text in TOKENIZER_PROBE_TEXTS],
        }

    @property
    def tokens_per_view(self) -> int:
        """max_grid[0] * max_grid[1], the column budget one camera gets."""
        return self.max_grid[0] * self.max_grid[1]

    @property
    def tokens_per_calibration(self) -> int:
        """A camera's calibration columns: INTRINSICS_TOKENS_PER_VIEW, six.

        Lp holds this number times n_blocks.
        """
        return INTRINSICS_TOKENS_PER_VIEW

    @property
    def tokens_per_extrinsics(self) -> int:
        """A camera's rotation columns: EXTRINSICS_TOKENS_PER_VIEW, six.

        Separate from tokens_per_calibration rather than folded into it, so the
        two blocks can be laid down, checked and scattered as the two records
        they are.
        """
        return EXTRINSICS_TOKENS_PER_VIEW

    def build_prompt_ids(
        self,
        prompts: list[str],
        view_valid: torch.Tensor,
        image_shapes: torch.Tensor,
        slot_valid: torch.Tensor,
        body_present: torch.Tensor,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """B prompts -> ids [B, Lp] long and a real-token mask [B, Lp] bool.

        view_valid [B, n_blocks] and image_shapes [B, n_blocks, 2] cover the
        Earlier blocks, then the Now blocks. Every sample keeps every block, so
        the fixed part of the row is one width. An absent view -- at t, at p, or
        all of p in a draw without history -- and the unused cells of a present
        one are cleared from the mask, which leaves the columns unattendable and
        keeps their rotary step. slot_valid [B, n_slots] clears both <state>
        columns of an arm the source does not have, and body_present [B] both
        of the body's for a key without one (StateInputs.body_present). Raises
        when a prompt is not v7, holds a reserved id or is longer than
        max_text_tokens: truncating would drop part of the instruction silently.
        Built on the host from host copies of the masks and sizes, then copied
        to device (default: image_shapes'), so the device is never waited on.
        """
        batch = len(prompts)
        expected = (batch, self.n_blocks)
        if tuple(view_valid.shape) != expected:
            raise ValueError(
                f"view_valid is {tuple(view_valid.shape)}, expected {expected}.\n"
                "One flag per view block, the Earlier views then the Now views."
            )
        if tuple(image_shapes.shape) != (*expected, 2):
            raise ValueError(
                f"image_shapes is {tuple(image_shapes.shape)}, expected "
                f"{(*expected, 2)}.\nOne pixel size per view block."
            )
        if tuple(slot_valid.shape) != (batch, self.n_slots) or slot_valid.dtype is not torch.bool:
            raise ValueError(
                f"slot_valid is {tuple(slot_valid.shape)} {slot_valid.dtype}, expected "
                f"{(batch, self.n_slots)} torch.bool.\nOne flag per arm slot."
            )
        if tuple(body_present.shape) != (batch,) or body_present.dtype is not torch.bool:
            raise ValueError(
                f"body_present is {tuple(body_present.shape)} {body_present.dtype}, expected "
                f"({batch},) torch.bool.\nOne flag per sample: does its key have a body."
            )

        device = image_shapes.device if device is None else device
        rows = [self.pre_ids + self.line_ids(p) + self.after_line_ids for p in prompts]
        ids = torch.full((batch, self.prefix_len), self.pad_token_id, dtype=torch.long)
        real = torch.zeros((batch, self.prefix_len), dtype=torch.bool)
        for index, row in enumerate(rows):
            ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
            real[index, : len(row)] = True

        grid = self.vision.grid_of(host_of(image_shapes))
        counts = grid[..., 0] * grid[..., 1]
        valid = host_of(view_valid).bool()
        if bool((valid & (counts == 0)).any()):
            raise ValueError(
                "a sample marks a camera valid and gives it no pixels.\n"
                "Expected a positive image_shapes entry wherever view_valid is true.\n"
                "Clear view_valid for the cameras the source does not have."
            )
        if bool((counts > self.tokens_per_view).any()):
            raise ValueError(
                f"a view needs {int(counts.max())} columns, the budget is "
                f"{self.tokens_per_view}.\nWiden model.prefix.max_grid; "
                "truncating would drop image content."
            )
        rank = torch.arange(self.tokens_per_view)[None, :]
        for block, (start, end) in enumerate(self.view_spans):
            real[:, start:end] = real[:, start:end] & valid[:, block : block + 1]
        for block, first in enumerate(self.image_columns):
            covered = rank < counts[:, block : block + 1]
            span = slice(first, first + self.tokens_per_view)
            real[:, span] = real[:, span] & covered
        columns = self.state_columns(self.text_lengths(prompts))
        if not bool((ids.gather(1, columns) == STATE_ID).all()):
            raise RuntimeError(
                "the computed <state> columns do not hold STATE_ID.\n"
                "Expected the state columns straight after the text's line.\n"
                "The row has been rebuilt under the scaffold; rebuild it here."
            )
        # Role-major, each role's arms then its body: the order state_columns
        # and the state banks lay the columns down in.
        holders = torch.cat([host_of(slot_valid), host_of(body_present)[:, None]], dim=1)
        present = holders.repeat(1, N_ROLES)
        real.scatter_(1, columns, real.gather(1, columns) & present)
        return to_host_device(ids, device), to_host_device(real, device)

    def position_indexes(
        self, ids: torch.Tensor, image_shapes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ids [B, Lp] and image_shapes [B, n_blocks, 2] -> t, h and w, each [B, Lp] long.

        A text token advances t and keeps h and w at zero. The tokens of one
        image share a single t and carry their position in that image's own
        grid, which is why the grid comes from image_shapes and not from a
        shared constant. A masked column keeps its t, so the anchor depends on
        the text's length alone and never on which views a sample holds.
        """
        grid = self.vision.grid_of(image_shapes)
        width = torch.ones_like(ids)
        for block, first in enumerate(self.image_columns):
            span = slice(first, first + self.tokens_per_view)
            # An absent camera has width zero and no attendable column; one
            # keeps the division defined without touching a readable index.
            width[:, span] = grid[:, block, 1].clamp(min=1)[:, None]
        return token_positions(ids, width, self.tokens_per_view)

    def block_causal_bias(
        self, t_index: torch.Tensor, pad_mask: torch.Tensor
    ) -> torch.Tensor:
        """Additive [B, 1, Lp, Lp] float32: (same t or causal) and not padding.

        Same t is evaluated within a row, never across the batch. Built here
        because the checkpoint's own forward ignores an attention mask when it
        is given embeddings.
        """
        blocked = torch.finfo(torch.float32).min
        columns = torch.arange(t_index.shape[-1], device=t_index.device)
        causal = (columns[None, :] <= columns[:, None])[None]
        same_step = t_index[:, None, :] == t_index[:, :, None]
        bias = torch.where(same_step | causal, 0.0, blocked)[:, None]
        return bias.masked_fill(~pad_mask.bool()[:, None, None, :], blocked)

    def check_geometry(
        self,
        intrinsics: torch.Tensor | None,
        intrinsics_valid: torch.Tensor | None,
        camera_extrinsics: torch.Tensor | None,
        camera_extrinsics_valid: torch.Tensor | None,
        batch: int,
        role: str = "",
    ) -> None:
        """Raise unless one role's camera geometry is the two records the prefix reads.

        The calibration and the camera's rotation, each [B, V, 6] with a flag
        per view: the twelve geometry columns of every view block project them.
        Checked separately, because they are separately measured. role is "" for
        the Now views and "history_" for the Earlier ones, which carry the same
        records under the same rule.
        """
        self.check_view_record(
            f"{role}intrinsics",
            intrinsics,
            intrinsics_valid,
            len(CALIBRATION_COLUMNS),
            f"One row per camera slot, the columns {CALIBRATION_COLUMNS}.",
            batch,
        )
        self.check_view_record(
            f"{role}camera_extrinsics",
            camera_extrinsics,
            camera_extrinsics_valid,
            EXTRINSICS_DIM,
            "One row per camera slot, the six numbers of the camera's rotation.",
            batch,
        )

    def check_view_record(
        self,
        name: str,
        record: torch.Tensor | None,
        valid: torch.Tensor | None,
        dim: int,
        columns: str,
        batch: int,
    ) -> None:
        """One per-view record, required, against its shape and its flag."""
        expected = (batch, self.n_views, dim)
        if record is None or valid is None:
            raise ValueError(
                f"model.prefix.intrinsics_mode is {self.intrinsics_mode!r} and the "
                f"batch carries no {name}.\nExpected {name} {expected} and "
                f"{name}_valid {expected[:2]}.\n"
                "Rebuild the corpus for this mode, or set the mode it was built for."
            )
        if tuple(record.shape) != expected:
            raise ValueError(
                f"{name} is {tuple(record.shape)}, expected {expected}.\n"
                f"{columns}\nFix the producer of this channel."
            )
        if tuple(valid.shape) != expected[:2]:
            raise ValueError(
                f"{name}_valid is {tuple(valid.shape)}, expected "
                f"{expected[:2]}.\nOne flag per camera slot."
            )
        if valid.dtype is not torch.bool:
            raise TypeError(
                f"{name}_valid is {valid.dtype}, expected torch.bool.\n"
                "A float mask would let 0.5 mean measured; cast it upstream."
            )

    def check_history(
        self, history_valid: torch.Tensor, history_view_valid: torch.Tensor, batch: int
    ) -> None:
        """Raise unless history_valid says exactly whether any Earlier view is present.

        A draw without history masks every Earlier view, and a draw with it has
        at least one: a history step with no camera is the no-history form, not
        a history of masked views. One flag per sample, held against the views
        it summarises, so a producer that disagrees with itself stops here.
        """
        if tuple(history_valid.shape) != (batch,) or history_valid.dtype is not torch.bool:
            raise ValueError(
                f"history_valid is {tuple(history_valid.shape)} {history_valid.dtype}, "
                f"expected ({batch},) torch.bool.\nOne flag per sample."
            )
        present = host_of(history_view_valid).bool().any(dim=-1)
        history_valid = host_of(history_valid)
        if not torch.equal(history_valid, present):
            raise ValueError(
                f"history_valid is {history_valid.tolist()} and the Earlier views say "
                f"{present.tolist()}.\n"
                "Expected history_valid == history_view_valid.any(-1): a draw without "
                "history masks every Earlier view, one with it has at least one.\n"
                "Fix the producer's no-history form."
            )

    def check_state(
        self,
        state: StateInputs,
        slot_valid: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> None:
        """Raise unless the state codes follow STATE_CODE_PRECEDENCE with the batch.

        For each arm: ABSENT exactly where the arm does not exist, in both
        roles. For the body: ABSENT in both roles or in neither. NO_HISTORY only
        in the start role, and there exactly where the arm or the body exists
        and the draw has no history. UNKNOWN and MEASURED fill the rest. The
        encoder holds these because the mask and the tokens both read the
        codes: a start column coded NO_HISTORY in a draw that has history would
        feed the model the zero state beside real Earlier frames.
        """
        batch = int(history_valid.shape[0])
        expected = (batch, N_ROLES, self.n_slots)
        state_code = state.state_code
        if tuple(state_code.shape) != expected:
            raise ValueError(
                f"state_code is {tuple(state_code.shape)}, expected {expected}.\n"
                f"The prefix holds {self.n_state_columns} <state> columns: one per role "
                "and arm slot, and the body's. Emit every slot, absent or not."
            )
        if state.validate(self.n_slots) != batch:
            raise ValueError(
                f"the state inputs hold {int(state_code.shape[0])} samples, the views "
                f"{batch}.\nPass policy.prefix_state's inputs for this batch."
            )
        if tuple(slot_valid.shape) != (batch, self.n_slots) or slot_valid.dtype is not torch.bool:
            raise ValueError(
                f"slot_valid is {tuple(slot_valid.shape)} {slot_valid.dtype}, expected "
                f"{(batch, self.n_slots)} torch.bool.\nOne flag per arm slot."
            )
        # On host copies (engine.hostcopy): the same codes, without a device wait.
        state_code = host_of(state_code)
        slots = host_of(slot_valid)
        history = host_of(history_valid)
        absent = state_code == STATE_ABSENT
        if not torch.equal(absent, (~slots)[:, None, :].expand_as(absent)):
            raise ValueError(
                "state_code says ABSENT where slot_valid says the arm exists, or the "
                "other way round.\n"
                "Expected ABSENT in both roles exactly where slot_valid is False.\n"
                "Apply STATE_CODE_PRECEDENCE in the producer."
            )
        if bool((state_code[:, ROLE_NOW] == STATE_NO_HISTORY).any()):
            raise ValueError(
                "state_code says NO_HISTORY in the now role.\n"
                "Expected NO_HISTORY only on a start token: the current state is "
                "measured, unknown or absent.\nFix the producer."
            )
        wanted = (~history)[:, None] & slots
        if not torch.equal(state_code[:, ROLE_START] == STATE_NO_HISTORY, wanted):
            raise ValueError(
                "the start codes disagree with history_valid.\n"
                "Expected NO_HISTORY on every existing arm's start token exactly when "
                "the draw has no history, and nowhere else.\n"
                "Apply STATE_CODE_PRECEDENCE in the producer."
            )
        body = host_of(state.body_state_code)
        body_absent = body == STATE_ABSENT
        if not torch.equal(body_absent[:, ROLE_START], body_absent[:, ROLE_NOW]):
            raise ValueError(
                "body_state_code says ABSENT in one role and not the other.\n"
                "Expected ABSENT in both roles exactly where the key has no body: the key, "
                "not the draw, decides whether a body exists.\n"
                "Apply STATE_CODE_PRECEDENCE in the producer."
            )
        if bool((body[:, ROLE_NOW] == STATE_NO_HISTORY).any()):
            raise ValueError(
                "body_state_code says NO_HISTORY in the now role.\n"
                "Expected NO_HISTORY only on a start token.\nFix the producer."
            )
        body_wanted = (~history) & ~body_absent[:, ROLE_NOW]
        if not torch.equal(body[:, ROLE_START] == STATE_NO_HISTORY, body_wanted):
            raise ValueError(
                "the body's start code disagrees with history_valid.\n"
                "Expected NO_HISTORY on the body's start token exactly when the key has a "
                "body and the draw has no history, and nowhere else.\n"
                "Apply STATE_CODE_PRECEDENCE in the producer."
            )

    def embed_prefix(
        self,
        ids: torch.Tensor,
        pixels: torch.Tensor,
        image_shapes: torch.Tensor,
        intrinsics: torch.Tensor,
        intrinsics_valid: torch.Tensor,
        camera_extrinsics: torch.Tensor,
        camera_extrinsics_valid: torch.Tensor,
        state: StateInputs,
        state_columns: torch.Tensor,
    ) -> torch.Tensor:
        """ids, images and state -> [B, Lp, hidden] with every projection in place.

        pixels [B, n_blocks, 3, H, W] and the per-block records hold the Earlier
        views first, then the Now views, and go through one tower pass and one
        set of geometry banks: a past view is read exactly as a current one.
        The image columns take the vision tower's tokens, each block's
        calibration and rotation take one column per number, and the six
        <state> columns take the state banks' tokens, role-major with each
        role's body token after its arms'. A column whose id is not the one the
        scaffold laid down raises rather than being written over.
        """
        batch, blocks = pixels.shape[0], pixels.shape[1]
        canvas = pixels.shape[-2:]
        # The sizes, the ids and the state columns are read on their host
        # copies (engine.hostcopy); only the rows each tower call takes go over.
        host_shapes = host_of(image_shapes)
        host_ids = host_of(ids)
        if bool((host_shapes > torch.tensor(canvas)).any()):
            raise ValueError(
                f"an image is larger than the {tuple(canvas)} canvas it arrived on.\n"
                "Expected every image_shapes entry to fit inside the canvas.\n"
                "Widen model.prefix.max_grid or resize in the loader."
            )
        flat = pixels.reshape(batch * blocks, *pixels.shape[2:])
        shapes = host_shapes.reshape(-1, 2)
        budget = self.tokens_per_view
        tokens = None
        # Grouped by distinct shape rather than run per image: the tower is two
        # convolutions, so a handful of calls covers the batch.
        for shape in torch.unique(shapes, dim=0):
            height, width = int(shape[0]), int(shape[1])
            if height == 0 or width == 0:
                continue
            rows = ((shapes[:, 0] == height) & (shapes[:, 1] == width)).nonzero()
            rows = to_host_device(rows.flatten(), flat.device)
            cropped = self.vision(flat[rows][:, :, :height, :width])
            # Sized from the tower's own output, so the buffer cannot be a
            # dtype the tower has since been cast away from.
            if tokens is None:
                tokens = cropped.new_zeros((flat.shape[0], budget, cropped.shape[-1]))
            tokens[rows, : cropped.shape[1]] = cropped
        if tokens is None:
            raise ValueError(
                "every camera slot in this batch has a zero image shape.\n"
                "Expected at least one view with pixels.\n"
                "The sampler has produced a batch the model cannot see."
            )

        columns = self.image_column_index
        if not bool((host_ids[:, self.host_columns["image"]] == IMG_CONTEXT_ID).all()):
            raise RuntimeError(
                "the image columns do not hold IMG_CONTEXT_ID.\n"
                "Expected the scaffold's own ids; the row has been rebuilt "
                "under it, so every column index here is wrong."
            )
        embeds = self.backbone.embed_tokens(ids)
        embeds[:, columns, :] = tokens.reshape(batch, blocks * budget, -1).to(
            embeds.dtype
        )
        calibration = self.intrinsics_column_index
        if not bool((host_ids[:, self.host_columns["intrinsics"]] == INTRINSICS_ID).all()):
            raise RuntimeError(
                "the calibration columns do not hold their own id.\n"
                "Expected INTRINSICS_ID at the columns the scaffold reserved.\n"
                "Rebuild the prefix through build_prompt_ids."
            )
        block = self.tokenizers.calibration_tokens(intrinsics, intrinsics_valid)
        embeds[:, calibration, :] = block.reshape(
            batch, calibration.numel(), -1
        ).to(embeds.dtype)
        rotation = self.extrinsics_column_index
        if not bool((host_ids[:, self.host_columns["extrinsics"]] == EXTRINSICS_ID).all()):
            raise RuntimeError(
                "the camera rotation columns do not hold their own id.\n"
                "Expected EXTRINSICS_ID at the columns the scaffold reserved.\n"
                "Rebuild the prefix through build_prompt_ids."
            )
        pose = self.tokenizers.camera_extrinsics_tokens(
            camera_extrinsics, camera_extrinsics_valid
        )
        embeds[:, rotation, :] = pose.reshape(batch, rotation.numel(), -1).to(
            embeds.dtype
        )
        if tuple(state_columns.shape) != (batch, self.n_state_columns):
            raise ValueError(
                f"state_columns is {tuple(state_columns.shape)}, expected "
                f"{(batch, self.n_state_columns)}.\nPass state_columns(text_lengths(prompts))."
            )
        held = host_ids.gather(1, host_of(state_columns))
        if not bool((held == STATE_ID).all()):
            raise RuntimeError(
                f"the <state> columns hold {sorted(set(held.flatten().tolist()))}, "
                f"expected only STATE_ID {STATE_ID}.\n"
                "The state follows the text, so its columns are counted from the text's "
                "length; a column taken from anywhere else writes a state over text.\n"
                "Pass state_columns(text_lengths(prompts))."
            )
        states = self.tokenizers.state_tokens(state)
        samples = torch.arange(batch, device=ids.device)[:, None].expand(held.shape)
        embeds[samples, state_columns.to(ids.device)] = states.reshape(
            batch, self.n_state_columns, -1
        ).to(embeds.dtype)
        return embeds

    def encode(
        self,
        *,
        pixels: torch.Tensor,
        image_shapes: torch.Tensor,
        view_valid: torch.Tensor,
        history_pixels: torch.Tensor,
        history_image_shapes: torch.Tensor,
        history_view_valid: torch.Tensor,
        history_valid: torch.Tensor,
        prompts: list[str],
        intrinsics: torch.Tensor | None = None,
        intrinsics_valid: torch.Tensor | None = None,
        camera_extrinsics: torch.Tensor | None = None,
        camera_extrinsics_valid: torch.Tensor | None = None,
        history_intrinsics: torch.Tensor | None = None,
        history_intrinsics_valid: torch.Tensor | None = None,
        history_camera_extrinsics: torch.Tensor | None = None,
        history_camera_extrinsics_valid: torch.Tensor | None = None,
        state: StateInputs,
        slot_valid: torch.Tensor,
        suffix_pos: torch.Tensor,
        recompute_layers: int | None = None,
    ) -> PrefixContext:
        """One batch's prefix.

        pixels, history_pixels   [B, V, 3, H, W] float in [-1, 1], the views at t
                                 and at the history step p; all zero where absent
        image_shapes (and history_)  [B, V, 2] long, (0, 0) where absent
        view_valid (and history_)    [B, V] bool
        history_valid            [B] bool, == history_view_valid.any(-1)
        prompts                  B rendered texts, sections 2 to 5, v7
        intrinsics, camera_extrinsics and their history_ twins, with _valid:
                                 [B, V, 6] and [B, V], required; a missing one
                                 raises (check_geometry)
        state                    StateInputs: the arms' [B, 2, S, 29] and the
                                 body's [B, 2, 8] model-input records, their
                                 per-dimension masks and their codes
                                 (policy.prefix_state), start then now
        slot_valid               [B, S] bool
        suffix_pos               [n_suffix] long, the decoder's position vector;
                                 required, no default
        recompute_layers         the backbone layers recomputed under gradient
                                 checkpointing (NeoBackbone.forward); None, all

        The suffix table is anchor[:, None] + 1 + suffix_pos[None, :], where
        anchor [B] is the t index of each row's last real prefix token: 129 + L
        at the shipped shape, whatever the views and the codes.
        """
        if suffix_pos.dim() != 1:
            raise ValueError(
                f"suffix_pos is {tuple(suffix_pos.shape)}, expected one dimension.\n"
                "A per-sample vector would tie the rotary phase to batch "
                "composition.\nPass FlowDecoder.suffix_positions unchanged."
            )
        if suffix_pos.dtype != torch.long:
            raise TypeError(
                f"suffix_pos is {suffix_pos.dtype}, expected torch.long.\n"
                "These are rotary offsets, counted in tokens."
            )
        n_suffix = int(suffix_pos.numel())
        if n_suffix == 0:
            raise ValueError(
                "suffix_pos is empty, expected one entry per suffix token.\n"
                "The decoder states the layout; an empty one is not a layout."
            )
        batch = len(prompts)
        for name, record in (("pixels", pixels), ("history_pixels", history_pixels)):
            if tuple(record.shape[:2]) != (batch, self.n_views):
                raise ValueError(
                    f"{name} is {tuple(record.shape)}, expected "
                    f"{(batch, self.n_views)} in the first two axes.\n"
                    "Every sample carries every camera slot at t and at p, masked or not."
                )
        self.check_history(history_valid, history_view_valid, batch)
        self.check_geometry(
            intrinsics, intrinsics_valid, camera_extrinsics, camera_extrinsics_valid, batch
        )
        self.check_geometry(
            history_intrinsics, history_intrinsics_valid, history_camera_extrinsics,
            history_camera_extrinsics_valid, batch, role="history_",
        )
        self.check_state(state, slot_valid, history_valid)

        def blocks(history, now):
            """Earlier then Now along the view axis."""
            return torch.cat([history, now], dim=1)

        block_pixels = blocks(history_pixels, pixels)
        # The sizes and the masks are only read on the host (the ids, the
        # positions, the tower's grouping), so their blocks are built there.
        block_shapes = blocks(host_of(history_image_shapes), host_of(image_shapes))
        block_valid = blocks(host_of(history_view_valid).bool(), host_of(view_valid).bool())
        block_intrinsics = blocks(history_intrinsics, intrinsics)
        block_intrinsics_valid = blocks(history_intrinsics_valid, intrinsics_valid)
        block_extrinsics = blocks(history_camera_extrinsics, camera_extrinsics)
        block_extrinsics_valid = blocks(history_camera_extrinsics_valid, camera_extrinsics_valid)

        device = pixels.device
        body_present = host_of(state.body_state_code)[:, ROLE_NOW] != STATE_ABSENT
        ids, pad_mask = self.build_prompt_ids(
            prompts, block_valid, block_shapes, slot_valid, body_present, device
        )
        host_ids = host_of(ids)
        width = self.prefix_width(prompts)
        if width < self.prefix_len:
            # Only pad goes: positions count along the row, so every kept
            # column's t, h and w, the anchor and the state columns are unmoved.
            ids, pad_mask, host_ids = ids[:, :width], pad_mask[:, :width], host_ids[:, :width]
            remember(ids, host_ids)
            if width not in self.logged_widths:
                self.logged_widths.add(width)
                logger.info("encode: prefix cut to %d of Lp %d columns", width, self.prefix_len)
        host_state_columns = self.state_columns(self.text_lengths(prompts))
        state_columns = to_host_device(host_state_columns, device)
        t_index, h_index, w_index = (
            to_host_device(index, device)
            for index in self.position_indexes(host_ids, block_shapes)
        )
        embeds = self.embed_prefix(
            ids,
            block_pixels,
            block_shapes,
            block_intrinsics,
            block_intrinsics_valid,
            block_extrinsics,
            block_extrinsics_valid,
            state,
            state_columns,
        )
        bias = self.block_causal_bias(t_index, pad_mask)
        rope = self.backbone.rope_tables(t_index, h_index, w_index)
        key_values, _ = self.backbone(embeds, bias, rope, keep_last_hidden=False,
                                      recompute_layers=recompute_layers)

        columns = torch.arange(pad_mask.shape[1], device=pad_mask.device)
        last_real = (pad_mask * columns).argmax(1, keepdim=True)
        anchor = t_index.gather(1, last_real).squeeze(1)
        suffix_t = anchor[:, None] + 1 + suffix_pos[None, :]
        cos, sin = rope_cos_sin(
            suffix_t, self.config.head_dim // 2, self.config.rope_theta
        )
        if not self.logged_prefix_len:
            self.logged_prefix_len = True
            logger.info(
                "encode: Lp %d, state columns from %d (text length %d in the first row)",
                self.prefix_len, int(host_state_columns[0, 0]), int(host_state_columns[0, 0])
                - len(self.pre_ids) - len(self.state_line_ids),
            )
        return PrefixContext(
            key_values=key_values,
            pad_mask=pad_mask,
            anchor=anchor,
            suffix_pos=suffix_pos,
            suffix_cos=cos,
            suffix_sin=sin,
            scaling=float(self.config.head_dim**-0.5),
            n_heads=self.config.num_attention_heads,
            n_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
        )

    def encode_chat_rows(
        self,
        ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: list[torch.Tensor],
        image_rows: torch.Tensor,
    ) -> torch.Tensor:
        """Native chat rows -> the final hidden state [B, T, hidden].

        ids and attention_mask are [B, T], padded on the right to the longest
        row; images are uint8 [3, H, W] at each image's own native size, in row
        then placeholder order, and image_rows names each one's row. Each image
        is scaled into the [-1, 1] contract by chat_image_tokens, in float32,
        and goes through the same vision tower call the robot path makes,
        grouped by size. Raises
        when a row's <IMG_CONTEXT> columns and its images' tokens differ in
        count, which is a renderer and tower disagreement, never a data skip,
        and when the tower does not normalise its input to ImageNet's mean and
        std: without that gate these rows are not NEO's pixels.
        """
        check_native_tower(self.vision)
        size = self.vision.pixels_per_token
        shapes = torch.tensor(
            [[int(image.shape[-2]), int(image.shape[-1])] for image in images],
            dtype=torch.long,
        ).reshape(-1, 2)
        grids = self.vision.grid_of(shapes)
        inside = ids == IMG_CONTEXT_ID
        rows = image_rows.to(ids.device)
        if int(rows.numel()) != len(images) or (
            rows.numel() > 1 and bool((rows[1:] < rows[:-1]).any())
        ):
            raise ValueError(
                f"{len(images)} images with image_rows {rows.tolist()}.\n"
                "Expected one row index per image, in row order.\n"
                "Build the batch with collate_chat."
            )
        wanted = torch.zeros(ids.shape[0], dtype=torch.long, device=ids.device)
        if len(images):
            wanted.index_add_(0, rows, (grids[:, 0] * grids[:, 1]).to(ids.device))
        found = inside.sum(1)
        if not torch.equal(found, wanted):
            raise ValueError(
                f"the rows hold {found.tolist()} image columns for images of "
                f"{wanted.tolist()} tokens.\n"
                "Expected one column per token; the renderer and the tower "
                f"disagree about an image's {size}-pixel grid."
            )
        embeds = self.backbone.embed_tokens(ids)
        if len(images):
            # The mask traverses row-major and the images arrive in row, then
            # placeholder, order, so image k fills the k-th run of columns.
            tokens = chat_image_tokens(self.vision, images, shapes, ids.device)
            embeds[inside] = tokens.to(embeds.dtype)
        t_index, h_index, w_index = chat_positions(ids, grids)
        bias = self.block_causal_bias(t_index, attention_mask)
        rope = self.backbone.rope_tables(t_index, h_index, w_index)
        _, hidden = self.backbone(
            embeds, bias, rope, keep_last_hidden=True, keep_key_values=False
        )
        return hidden

    def encode_chat_pack(
        self,
        input_ids: torch.Tensor,
        segments: torch.Tensor,
        positions: torch.Tensor,
        images: list[torch.Tensor],
        grid_hw: torch.Tensor,
    ) -> torch.Tensor:
        """One micro-batch's chat pack -> the final hidden state [1, S, hidden].

        The pack's fields: input_ids and
        segments [1, S], positions [3, 1, S], images uint8 [3, H, W] at native
        size in <IMG_CONTEXT> order, grid_hw [K, 2]. The rows attend under
        NEO's BlockMask (chat_block_mask), each row to itself only, and the
        backbone runs them in one call without keys and values, with the
        relief chat_relief turns on. Every row comes out as encode_chat_rows
        gives it alone: positions restart per row and the mask isolates it.

        Raises, reading the host copies (engine.hostcopy) so the device is
        never waited on, when grid_hw is not the tower's grid of each image;
        when an image's <img> block holds other than its grid's tokens, which
        holds every row's count as encode_chat_rows holds it, image by image;
        and when positions are not chat_positions' with segments, the one rule
        the collate and the model share. And, as for rows, unless the tower
        normalises to ImageNet's mean and std.
        """
        check_native_tower(self.vision)
        host_ids, host_segments = host_of(input_ids), host_of(segments)
        host_positions, host_grids = host_of(positions), host_of(grid_hw)
        length = int(host_ids.shape[-1])
        if tuple(host_ids.shape) != (1, length) or tuple(host_segments.shape) != (1, length) \
                or tuple(host_positions.shape) != (3, 1, length):
            raise ValueError(
                f"the pack's ids, segments and positions are {tuple(host_ids.shape)}, "
                f"{tuple(host_segments.shape)} and {tuple(host_positions.shape)}.\n"
                "Expected [1, S], [1, S] and [3, 1, S].\n"
                "Build the pack with collate_chat_pack."
            )
        shapes = torch.tensor(
            [[int(image.shape[-2]), int(image.shape[-1])] for image in images],
            dtype=torch.long,
        ).reshape(-1, 2)
        grids = self.vision.grid_of(shapes)
        if not torch.equal(host_grids.long().reshape(-1, 2), grids):
            raise ValueError(
                f"grid_hw is {host_grids.tolist()[:4]} for images whose tower grids are "
                f"{grids.tolist()[:4]}.\n"
                f"Expected each image's size over {self.vision.pixels_per_token} pixels a token.\n"
                "Emit grid_hw from the images collate_chat_pack lists."
            )
        inside = host_ids[0] == IMG_CONTEXT_ID
        _, image_numbers = chat_mask_ids(host_segments, host_ids)
        counts = torch.bincount(image_numbers[inside], minlength=1)
        held, stray = counts[1:][counts[1:] > 0], int(counts[0])
        wanted = grids[:, 0] * grids[:, 1]
        if stray or not torch.equal(held, wanted):
            raise ValueError(
                f"the pack's image blocks hold {held.tolist()[:8]} columns ({stray} outside "
                f"any <img>) for images of {wanted.tolist()[:8]} tokens.\n"
                "Expected each <img> block to hold one column per token of its image; the "
                "renderer and the tower disagree about an image's grid.\n"
                "Pack the rows' images in <IMG_CONTEXT> order with their own grids."
            )
        expected = torch.stack(chat_positions(host_ids, grids, host_segments))
        if not torch.equal(host_positions, expected):
            column = int((host_positions != expected).any(0)[0].nonzero()[0])
            raise ValueError(
                f"the pack's positions at column {column} are "
                f"{host_positions[:, 0, column].tolist()}; chat_positions gives "
                f"{expected[:, 0, column].tolist()}.\n"
                "Expected the collate's positions to be chat_positions(ids, grid_hw, "
                "segments), t restarting per row.\n"
                "Compute positions in collate_chat_pack with chat_positions."
            )
        embeds = self.backbone.embed_tokens(input_ids)
        if len(images):
            # The <IMG_CONTEXT> columns found on the host copy and moved without
            # waiting (speed C-8+): the indices a boolean mask on the device
            # would give, row then column, so the same index_put_ and its
            # backward's gather, with no wait for a nonzero.
            places = tuple(to_host_device(index, input_ids.device)
                           for index in (host_ids == IMG_CONTEXT_ID).nonzero(as_tuple=True))
            embeds.index_put_(places, chat_image_tokens(
                self.vision, images, shapes, input_ids.device).to(embeds.dtype))
        mask = chat_block_mask(segments, input_ids)
        rope = self.backbone.rope_tables(positions[0], positions[1], positions[2])
        chunk = chat_pack.CHAT_MLP_CHUNK_TOKENS if self.chat_relief["chat_mlp_chunk"] else None
        _, hidden = self.backbone(
            embeds, mask, rope, keep_last_hidden=True, keep_key_values=False,
            offload_checkpoints=self.chat_relief["chat_offload_checkpoints"], mlp_chunk=chunk,
        )
        return hidden
