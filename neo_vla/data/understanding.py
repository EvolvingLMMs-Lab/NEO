"""The vision-language half of a co-train batch: plain chat, no robot tokens.

One chat format, neo_native: a record renders the way NEOpp-2B-SFT's own
trainer did (its tag-mode chat function, every assistant turn supervised), with
0..k images per row, each resized by smart_resize at 32 pixels a token with
PIL's bicubic and carried as uint8. encode_chat_rows scales them into the
[-1, 1] contract and the tower's ImageNet gate turns that into NEO's pixels.
The batch keeps rows at their own lengths.

A row carries no sections, no anchor ids and no conditioning column, which is
the asymmetry that keeps the backbone's language ability rather than adapting
it to the robot scaffold. The COCO ShareGPT corpus (ShareGPT4VCoco) reads this
way. The earlier 'legacy' format is refused by name, and so are its keys.
"""
from __future__ import annotations

import io
import json
import logging
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from ..model.backbone.neo_llm import IMG_CONTEXT_ID, IMG_END_ID, IMG_START_ID
from .chat_pack import (
    PACK_COUNTERS,
    PACK_GRID_HW,
    PACK_IDS,
    PACK_IMAGES,
    PACK_LABELS,
    PACK_POSITIONS,
    PACK_SEGMENT_TARGETS,
    PACK_SEGMENTS,
    PAD_SEGMENT,
    check_pack,
    counter_vector,
)
from .mixture import stream_seed

logger = logging.getLogger(__name__)

# Where the chat half sits in a co-train sample. The policy raises when a
# co-train run produces a batch without this key.
CHAT_KEY = "chat"

# Positions the loss ignores. Every column that is not the assistant's own
# answer carries this, so the image and the question are context and never
# targets.
IGNORE_INDEX = -100

# Where an image block goes in the rendered row. The corpora write their own
# placeholders in prose, so this one is a control character no text carries and
# a record that carries it anyway is skipped rather than rendered twice.
IMAGE_MARK = "\x00image\x00"

# How a chat corpus renders its rows: NEO's own way, and no other. The node
# may state it and need not; a node naming a retired format, or setting a
# retired format's keys, is refused by name so a config written for it cannot
# run under this tree with its keys silently ignored.
NATIVE_FORMAT = "neo_native"
CHAT_FORMATS = (NATIVE_FORMAT,)
RETIRED_CHAT_FORMATS = {
    "legacy": "the Qwen3 jinja template on one square [-1, 1] canvas with only the last "
              "answer supervised",
}
RETIRED_CHAT_KEYS = ("image_size", "pixels_per_token")

# The native format's row, as a corpus emits it, and the batch collate_chat
# makes of several: rows padded to the longest one in the batch, every image of
# the batch in row then placeholder order, and the row each image belongs to.
NATIVE_ROW_KEYS = ("input_ids", "labels", "images")
NATIVE_BATCH_KEYS = ("input_ids", "attention_mask", "labels", "images", "image_rows")
# A padding column is masked as a key and carries no label, so its id only has
# to be a row of the embedding table; 0 is one in every vocabulary.
CHAT_PAD_ID = 0

# NEOpp-2B-SFT is step 100000 of a NEO++ run whose trainer rendered every chat
# row through preprocess_internlm_v3_mm_chat with thinking_method 'tag' (all
# 527 tensors are bit-identical to that step's export). The names below port
# that function, its image loader and its smart_resize: neo_train_plus
# internvl/data/dataset.py:1427-1637, :1795-1835, :313-337 and
# multimodal_dataset.py:743-907, :1059-1117.
NEO_REASON_PROMPT = (
    "Reason step by step and place the thought process within the <think></think> "
    "tags, and provide the final conclusion at the end."
)
NEO_ROLES = {
    "human": "user",
    "system": "system",
    "knowledge": "knowledge",
    "gpt": "assistant",
    "plugin": "system name=<|plugin|>",
    "plugin-return": "environment name=<|plugin|>",
    "interpreter": "system name=<|interpreter|>",
    "interpreter-return": "environment name=<|interpreter|>",
}
NEO_TURN_START = "<|im_start|>"
NEO_TURN_END = "<|im_end|>\n"
NEO_EMPTY_THINK = "<think>\n\n</think>\n\n"
NEO_IMAGE_PLACEHOLDER = "<image>"
# The trainer's only thinking method for this checkpoint. 'sys' and a plain
# neo1_0 render exist in NEO's other trees and did not produce these weights.
NEO_THINKING_METHODS = ("tag",)
# Pixels per backbone token on each side: a 16 pixel patch and a 2x2 merge.
NEO_TOKEN_PIXELS = 32
NEO_MAX_ASPECT = 200
# The messages dialect, renamed into the speakers NEO's function reads. NEO's
# own loader reads only 'conversations'.
MESSAGE_SPEAKERS = {"user": "human", "assistant": "gpt", "system": "system"}
# Where a native layout holds an image block that has no size yet.
IMAGE_SLOT = -1
# The pixel convention a native chat half needs. Its images cross the loader
# as uint8 and reach the tower as x / 255 * 2 - 1, which is NEO's ToTensor plus
# ImageNet mean and std only behind the tower's entry gate that
# model.vision.pixel_convention neo_native selects. Without
# the gate the tower reads raw [-1, 1], an affine map away from NEO's pixels
# (max |diff| 1.19 to 1.64 in review), so a native corpus states the convention
# and refuses any other; encode_chat_rows checks the tower itself.
NATIVE_PIXEL_CONVENTION = "neo_native"

# The ids a NEO-family tokenizer must give these tokens, the backbone's own.
NEO_TOKEN_IDS = {
    "<IMG_CONTEXT>": IMG_CONTEXT_ID, "<img>": IMG_START_ID, "</img>": IMG_END_ID,
    "<|im_start|>": 151644, "<|im_end|>": 151645, "<think>": 151667, "</think>": 151668,
}
# The chat tokenizer is NEOpp-2B-SFT's own files, the one the robot prompt
# reads too. NEO's trainer built its tokenizer from the Qwen3-14B files under
# transformers 4.37.2 (build_dataloader.py:103-121, use_fast=False), whose slow
# Qwen2Tokenizer skips every merges.txt line that starts with '#': 151291
# merges, not 151387. So it split '##' into two '#', and the checkpoint's own
# merges.txt (151292 lines, header included, written by that trainer) is that
# tokenizer; the weights agree (NLL lower under the split ids on 9 of 9 texts).
# The Qwen3-14B files read under a newer transformers merge '#' and give ids
# the checkpoint never trained on. These probes refuse such a tokenizer.
NEO_TOKENIZER_PROBES = (("##", (2, 2)), ("###", (2, 2, 2)), ("#include", (2, 997)))


class ChatSkip(ValueError):
    """One record the native format cannot turn into a row, and why.

    key is the skip counter it lands in. Raised inside a draw and caught by the
    draw, never by the loader: a record is data, and a bad one is counted and
    replaced rather than allowed to end the worker.
    """

    def __init__(self, key: str, message: str = ""):
        super().__init__(message or key)
        self.key = key


class MissingObject(LookupError):
    """An image file a record names is absent or unreadable.

    One bad record, counted under 'unfetchable' like any other skip and
    replaced by a redraw. An image root that does not answer at all still
    stops the run, through the corpus's consecutive-failure guard.
    """


def round_by_factor(number: float, factor: int) -> int:
    """The multiple of factor nearest number, rounding as Python's round does."""
    return round(number / factor) * factor


def ceil_by_factor(number: float, factor: int) -> int:
    """The smallest multiple of factor at or above number."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: float, factor: int) -> int:
    """The largest multiple of factor at or below number."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, *, min_pixels: int, max_pixels: int,
    factor: int = NEO_TOKEN_PIXELS,
) -> tuple[int, int]:
    """An image's size -> NEO's native size for it, both sides multiples of factor.

    Each side rounds to the nearest multiple; a result over max_pixels is
    scaled down and floored, one under min_pixels scaled up and ceiled, so the
    aspect is kept as closely as whole tokens allow. An aspect over 200 raises,
    which drops the row, as in NEO's trainer.
    """
    if max(height, width) / min(height, width) > NEO_MAX_ASPECT:
        raise ChatSkip(
            "aspect",
            f"absolute aspect ratio must be smaller than {NEO_MAX_ASPECT}, "
            f"got {max(height, width) / min(height, width)}",
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def image_budget(count: int, *, min_pixels: int, max_pixels: int) -> int:
    """The max_pixels one image of a count-image row is resized under.

    A row of several images shares twice the single-image budget, never less
    than min_pixels an image (multimodal_dataset.py:826).
    """
    return max_pixels if count == 1 else max(max_pixels * 2 // count, min_pixels)


def contrasting_background(image: Image.Image) -> tuple[int, int, int] | None:
    """NEO's backdrop for an RGBA image with transparent pixels, or None.

    Ported as written: the foreground mean is taken over 0..255 values and
    compared with 382.5, so a backdrop, when there is one, is always white.
    """
    array = np.array(image)
    if not (array[:, :, 3] == 0).any():
        return None
    opaque = array[:, :, :3][array[:, :, 3] > 0]
    if opaque.size == 0:
        return None
    return (0, 0, 0) if opaque.mean() > 382.5 else (255, 255, 255)


def neo_rgb(image: Image.Image) -> Image.Image:
    """A decoded image file -> RGB, the way NEO's loader converts it.

    An RGBA image with transparent pixels is first laid on the contrasting
    backdrop, as TCSLoader.__call__ in NEO's trainer does for a file.
    """
    if image.mode == "RGBA":
        backdrop = contrasting_background(image)
        if backdrop:
            laid = Image.new("RGB", image.size, backdrop)
            laid.paste(image, mask=image.split()[3])
            image = laid
    return image.convert("RGB")


def neo_conversation(record: dict) -> list[dict]:
    """A record -> its turns as NEO's function reads them: from, value, is_input.

    'conversations' is read as it stands; the messages dialect is renamed into
    the same speakers. A turn that is not a dict with string speaker and text
    drops the row, as the same turn stops NEO's function.
    """
    conversations = record.get("conversations")
    if isinstance(conversations, list):
        turns, speaker, said, rename = conversations, "from", "value", None
    elif isinstance(record.get("messages"), list):
        turns, speaker, said, rename = record["messages"], "role", "content", MESSAGE_SPEAKERS
    else:
        raise ChatSkip("no_turns", "the record holds no conversation")
    converted = []
    for turn in turns:
        if not isinstance(turn, dict):
            raise ChatSkip("no_turns", "a turn is not a mapping")
        who, text = turn.get(speaker), turn.get(said)
        if not isinstance(who, str) or not isinstance(text, str):
            raise ChatSkip("no_turns", "a turn has no speaker or no text")
        one = {"from": rename.get(who, who) if rename else who, "value": text}
        if "is_input" in turn:
            one["is_input"] = turn["is_input"]
        converted.append(one)
    if not converted:
        raise ChatSkip("no_turns", "the conversation is empty")
    return converted


def neo_image_refs(record: dict) -> list[str] | None:
    """The images a record names, in order, or None for a row without an image field.

    NEO reads 'image' (a string or a list) and otherwise 'images'. A null
    'image' beside an 'images' list, which ShareGPT4V-COCO writes, reads the
    list. A record with no image field and a video is skipped: this port reads
    stills only. A record with a video and an 'image' frame list reads the
    frames, as NEO does, since NEO reads 'image' first.
    """
    value = record.get("image")
    if isinstance(value, str) and value:
        refs = [value]
    elif isinstance(value, list):
        refs = value
    elif isinstance(record.get("images"), list):
        refs = record["images"]
    elif "image" in record or "images" in record:
        raise ChatSkip("unresolvable", "the image field names no image")
    else:
        video = record.get("video")
        if video is not None and video != "":
            raise ChatSkip("video", "a video row; this corpus reads stills")
        return None
    if not all(isinstance(ref, str) and ref for ref in refs):
        raise ChatSkip("unresolvable", "an image reference is not a path")
    return list(refs)


def load_chat_tokenizer(path: str | Path):
    """The native chat tokenizer, checked against the one NEO's trainer used.

    A tokenizer that names <IMG_CONTEXT> with the backbone's id is a NEO
    tokenizer, and it must then give every id in NEO_TOKEN_IDS and
    NEO_TOKENIZER_PROBES, or this raises: a tokenizer that merges text
    differently trains the chat half on ids NEO never saw. One that does not
    know <IMG_CONTEXT> (a test's byte-level one) is read as is.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>") != IMG_CONTEXT_ID:
        return tokenizer
    wrong = {
        token: tokenizer.convert_tokens_to_ids(token)
        for token, wanted in NEO_TOKEN_IDS.items()
        if tokenizer.convert_tokens_to_ids(token) != wanted
    }
    for text, wanted in NEO_TOKENIZER_PROBES:
        got = tuple(tokenizer(text, add_special_tokens=False)["input_ids"])
        if got != wanted:
            wrong[text] = got
    if wrong:
        raise ValueError(
            f"the chat tokenizer at {path} gives {wrong}, where NEO's training "
            f"tokenizer gives {dict(NEO_TOKEN_IDS)} and {dict(NEO_TOKENIZER_PROBES)}.\n"
            "A tokenizer that merges '#' is the Qwen3 merges.txt read by a transformers "
            "newer than NEO's 4.37.2, which kept the lines starting with '#'.\n"
            "Point tokenizer_path at NEOpp-2B-SFT itself (model.tokenizer_path)."
        )
    return tokenizer


@dataclass(frozen=True)
class ChatLayout:
    """A native row before its images are sized: text ids with a slot per image.

    ids holds IMAGE_SLOT where an image block goes, supervised says which ids
    are targets, and images is how many slots there are, in placeholder order.
    """

    ids: tuple[int, ...]
    supervised: tuple[bool, ...]
    images: int

    @property
    def text_tokens(self) -> int:
        """The row's length without its image blocks."""
        return len(self.ids) - self.images


class NeoChatRenderer:
    """A record -> a native chat row: ids, labels and the images at their own sizes.

    render_turns is NEO's tag-mode function with the image block written as a
    mark; layout tokenises it one piece at a time, as NEO does, and assemble
    splices each image's block in once its size is known. A row is the whole
    record or nothing: one longer than max_tokens is skipped, never cut.
    """

    def __init__(
        self,
        *,
        tokenizer_path: str | Path,
        min_pixels: int,
        max_pixels: int,
        max_tokens: int,
        thinking_method: str = "tag",
    ):
        if thinking_method not in NEO_THINKING_METHODS:
            raise ValueError(
                f"thinking_method is {thinking_method!r}.\n"
                f"Expected one of {NEO_THINKING_METHODS}: the method NEOpp-2B-SFT's "
                "trainer ran.\nPort another method before selecting it."
            )
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        self.max_tokens = int(max_tokens)
        if not NEO_TOKEN_PIXELS**2 <= self.min_pixels <= self.max_pixels:
            raise ValueError(
                f"min_pixels {self.min_pixels} and max_pixels {self.max_pixels}.\n"
                f"Expected {NEO_TOKEN_PIXELS ** 2} <= min_pixels <= max_pixels.\n"
                "NEO's own are 65536 and 16777216."
            )
        if self.max_tokens < 2:
            raise ValueError(
                f"max_tokens is {self.max_tokens}.\n"
                "Expected room for a turn and one supervised token.\n"
                "Raise max_tokens."
            )
        self.tokenizer = load_chat_tokenizer(tokenizer_path)
        self.thinking_method = thinking_method

    def text_ids(self, text: str) -> list[int]:
        """Tokenise one piece of text, adding no special tokens of its own."""
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def render_turns(
        self, conversation: list[dict], image_count: int, text_only: bool
    ) -> tuple[list[tuple[str, bool]], int]:
        """NEO's tag-mode function over one conversation -> (pieces, images placed).

        Each piece is (text, supervised) and IMAGE_MARK stands where NEO writes
        an image block. The steps, in NEO's order: a one-image row whose first
        human turn has no placeholder gets one; each placeholder, in order, is
        replaced in place while images remain; the reasoning prompt leaves every
        system turn and an emptied one is dropped; a turn renders as
        <|im_start|>role, newline, content, <|im_end|> and newline, masked,
        except an assistant turn, whose header is masked and whose content and
        end are the target. That header carries the empty think block unless
        the row asked for reasoning or the answer opens its own <think>.
        """
        turns = [dict(turn) for turn in conversation]
        if any(IMAGE_MARK in turn["value"] for turn in turns):
            raise ChatSkip("unrenderable", "the record writes this module's image mark")
        placed = 0
        if not text_only:
            if image_count == 1:
                for turn in turns:
                    if turn["from"] == "human":
                        if NEO_IMAGE_PLACEHOLDER not in turn["value"]:
                            turn["value"] = NEO_IMAGE_PLACEHOLDER + "\n" + turn["value"]
                        break
            for turn in turns:
                while NEO_IMAGE_PLACEHOLDER in turn["value"] and placed < image_count:
                    turn["value"] = turn["value"].replace(NEO_IMAGE_PLACEHOLDER, IMAGE_MARK, 1)
                    placed += 1
        thinking = False
        kept = []
        for turn in turns:
            if turn["from"] == "system":
                if NEO_REASON_PROMPT in turn["value"]:
                    turn["value"] = re.sub(
                        r"(\n?)(" + re.escape(NEO_REASON_PROMPT) + ")", "", turn["value"]
                    )
                    thinking = True
                if turn["value"] == "":
                    continue
            kept.append(turn)
        pieces: list[tuple[str, bool]] = []
        for turn in kept:
            role = NEO_ROLES.get(turn["from"])
            if role is None:
                continue
            content = turn["value"]
            if role != "assistant" or turn.get("is_input", False):
                pieces.append((f"{NEO_TURN_START}{role}\n{content}{NEO_TURN_END}", False))
                continue
            if not thinking and not content.startswith("<think>"):
                header = f"{NEO_TURN_START}{role}\n{NEO_EMPTY_THINK}"
            else:
                header = f"{NEO_TURN_START}{role}\n"
            pieces.append((header, False))
            pieces.append((f"{content}{NEO_TURN_END}", True))
        return pieces, placed

    def layout(
        self, conversation: list[dict], image_count: int, text_only: bool
    ) -> ChatLayout:
        """A conversation -> its ChatLayout, or a ChatSkip naming why there is none.

        Each piece is tokenised on its own and split at the image mark, so an
        image slot always sits between two tokenizer segments, which is where
        NEO's own <img> special token splits the same text. Skips: fewer
        placeholders than images, or a placed image block lost with the
        unknown-role turn it sat in; a literal <IMG_CONTEXT> in prose; no
        target; and text plus each image's fewest tokens longer than max_tokens.
        """
        pieces, placed = self.render_turns(conversation, image_count, text_only)
        if not text_only and placed != image_count:
            raise ChatSkip(
                "placeholder", f"{placed} image placeholders for {image_count} images"
            )
        ids: list[int] = []
        supervised: list[bool] = []
        for text, target in pieces:
            for index, part in enumerate(text.split(IMAGE_MARK)):
                if index:
                    ids.append(IMAGE_SLOT)
                    supervised.append(target)
                encoded = self.text_ids(part) if part else []
                ids.extend(encoded)
                supervised.extend([target] * len(encoded))
        # A placeholder placed in a turn NEO then drops (an unknown role) takes
        # its image block with it, and NEO's count assert drops the row
        # (multimodal_dataset.py:892-893). Counted here, from what was emitted.
        emitted = ids.count(IMAGE_SLOT)
        if emitted != placed:
            raise ChatSkip(
                "placeholder", f"{emitted} of {placed} image blocks survive the render"
            )
        if IMG_CONTEXT_ID in ids:
            raise ChatSkip("reserved_token", "the prose writes <IMG_CONTEXT>")
        if not any(flag for token, flag in zip(ids, supervised) if token != IMAGE_SLOT):
            raise ChatSkip("no_answer", "no assistant turn is supervised")
        layout = ChatLayout(tuple(ids), tuple(supervised), emitted)
        if layout.text_tokens + self.fewest_block_tokens() * layout.images > self.max_tokens:
            raise ChatSkip("too_long", f"{layout.text_tokens} text tokens")
        return layout

    def fewest_block_tokens(self) -> int:
        """The fewest columns an image block can take: <img>, its tokens, </img>.

        smart_resize keeps an image at min_pixels or more unless it floors
        after scaling down, and the floor loses at most half the pixels: an
        image just under 64 pixels on its short side keeps one 32-pixel row.
        So min_pixels // 2048 tokens, exhaustively checked at the budget floor.
        The pre-fetch check uses it, so a row too long in its text and its
        images' fewest tokens is dropped before any image is fetched.
        """
        return 2 + max(1, self.min_pixels // (2 * NEO_TOKEN_PIXELS**2))

    def grids(self, sizes: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Each image's (width, height) -> its token grid (rows, columns), native."""
        if not sizes:
            return []
        budget = image_budget(
            len(sizes), min_pixels=self.min_pixels, max_pixels=self.max_pixels
        )
        grids = []
        for width, height in sizes:
            rows, columns = smart_resize(
                height, width, min_pixels=self.min_pixels, max_pixels=budget
            )
            grids.append((rows // NEO_TOKEN_PIXELS, columns // NEO_TOKEN_PIXELS))
        return grids

    def assemble(
        self, layout: ChatLayout, grids: list[tuple[int, int]]
    ) -> tuple[list[int], list[int]]:
        """A layout and one grid per slot -> ids and labels, IGNORE_INDEX off target."""
        if len(grids) != layout.images:
            raise ValueError(
                f"{len(grids)} grids for a layout of {layout.images} images.\n"
                "Expected one grid per image slot, in placeholder order."
            )
        ids: list[int] = []
        labels: list[int] = []
        sized = iter(grids)
        for token, target in zip(layout.ids, layout.supervised):
            if token == IMAGE_SLOT:
                rows, columns = next(sized)
                block = [IMG_START_ID] + [IMG_CONTEXT_ID] * (rows * columns) + [IMG_END_ID]
                ids.extend(block)
                labels.extend(block if target else [IGNORE_INDEX] * len(block))
            else:
                ids.append(token)
                labels.append(token if target else IGNORE_INDEX)
        return ids, labels

    def pixels(self, picture: Image.Image, grid: tuple[int, int]) -> torch.Tensor:
        """An opened image and its grid -> uint8 [3, H, W] at NEO's native size.

        PIL's bicubic, which is what NEO's image.resize applied by default to
        an RGB image. uint8 rather than float is what crosses the loader;
        encode_chat_rows scales it into the [-1, 1] contract on the device.
        """
        rgb = neo_rgb(picture)
        size = (grid[1] * NEO_TOKEN_PIXELS, grid[0] * NEO_TOKEN_PIXELS)
        resized = rgb.resize(size, Image.Resampling.BICUBIC)
        array = np.asarray(resized, dtype=np.uint8)
        return torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()

    def sample(self, layout: ChatLayout, blobs: list[bytes]) -> dict:
        """A layout and its images' encoded bytes -> one native row.

        The size is read from each image's header first, so a row that the
        images make too long is dropped before anything is decoded.
        """
        pictures = [Image.open(io.BytesIO(blob)) for blob in blobs]
        grids = self.grids([picture.size for picture in pictures])
        ids, labels = self.assemble(layout, grids)
        if len(ids) > self.max_tokens:
            raise ChatSkip("too_long", f"{len(ids)} tokens with the images")
        # What encode_chat_rows raises on, so a row that would stop a rank is a
        # skip in the worker instead.
        columns = ids.count(IMG_CONTEXT_ID)
        if len(pictures) != layout.images or columns != sum(r * c for r, c in grids):
            raise ChatSkip(
                "placeholder",
                f"{columns} image columns for {len(pictures)} images of grids {grids}",
            )
        images = [self.pixels(picture, grid) for picture, grid in zip(pictures, grids)]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "images": images,
        }


def native_draw(
    renderer: NeoChatRenderer,
    record: object,
    *,
    locate: Callable[[str], str],
    fetch: Callable[[str], bytes],
    images_per_row: str,
    count: Callable[[str, str], None],
) -> dict | None:
    """One record -> a native row, or None after counting why there is none.

    locate turns a reference into a uri and raises ChatSkip for one it cannot;
    fetch reads a uri's bytes and raises MissingObject for one that is not
    there. Any other fetch failure is not caught: a mount that does not
    answer is a launch fault, not a bad record. Everything else is the
    record's own data, so every failure there is a counted skip, as NEO's
    get_sample drops a row on any exception: a ChatSkip under its own key,
    anything else (a lone surrogate the tokenizer will not take, say) under
    'unrenderable' with its type. The consecutive-failure guard of the corpus
    still stops a run whose renderer fails on every record.
    """
    try:
        if not isinstance(record, dict):
            raise ChatSkip("unparsable", "the line is not a record")
        conversation = neo_conversation(record)
        refs = neo_image_refs(record)
        if images_per_row == "one" and (refs is None or len(refs) != 1):
            raise ChatSkip("image_count", "the corpus keeps one-image rows only")
        uris = [locate(ref) for ref in refs or []]
        layout = renderer.layout(conversation, len(uris), refs is None)
    except ChatSkip as skip:
        count(skip.key, str(skip))
        return None
    except Exception as error:  # noqa: BLE001
        count("unrenderable", f"{type(error).__name__}: {error}")
        return None
    blobs = []
    for uri in uris:
        try:
            blobs.append(fetch(uri))
        except MissingObject as missing:
            count("unfetchable", str(missing))
            return None
    try:
        return renderer.sample(layout, blobs)
    except ChatSkip as skip:
        count(skip.key, str(skip))
        return None
    # A record's own bytes can fail to decode in more ways than PIL names
    # (OSError, SyntaxError, EOFError, zlib and struct errors, a decompression
    # bomb); each is one bad record, so each is a counted skip.
    except Exception as error:  # noqa: BLE001
        count("undecodable", f"{type(error).__name__}: {error}")
        return None


def collate_chat(rows: list[dict]) -> dict:
    """Native rows -> one batch: ids, mask and labels padded to the longest row.

    Rows keep their own lengths up to here, so a batch costs its longest row
    rather than max_tokens. images is a list over the whole batch in row, then
    placeholder, order, which is the order encode_chat_rows reads image columns.
    """
    for row in rows:
        missing = [key for key in NATIVE_ROW_KEYS if key not in row]
        if missing or tuple(row["input_ids"].shape) != tuple(row["labels"].shape):
            raise ValueError(
                f"a native chat row lacks {missing} or its ids and labels differ in length.\n"
                f"Expected {NATIVE_ROW_KEYS}, ids and labels of one length.\n"
                "Emit rows from NeoChatRenderer.sample."
            )
    width = max(int(row["input_ids"].shape[0]) for row in rows)
    input_ids = torch.full((len(rows), width), CHAT_PAD_ID, dtype=torch.long)
    labels = torch.full((len(rows), width), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.bool)
    images: list[torch.Tensor] = []
    owners: list[int] = []
    for index, row in enumerate(rows):
        length = int(row["input_ids"].shape[0])
        input_ids[index, :length] = row["input_ids"]
        labels[index, :length] = row["labels"]
        mask[index, :length] = True
        images.extend(row["images"])
        owners.extend([index] * len(row["images"]))
    return {
        "input_ids": input_ids,
        "attention_mask": mask,
        "labels": labels,
        "images": images,
        "image_rows": torch.tensor(owners, dtype=torch.long),
    }


def collate_chat_pack(
    rows: list[dict], *, pack_tokens: int, dropped_overlong: int = 0, spilled: int = 0
) -> dict:
    """Native rows -> one pack of exactly pack_tokens positions.

    The rows go back to back in order, row n its own segment n, and the rest
    is padding: CHAT_PAD_ID, IGNORE_INDEX and PAD_SEGMENT to the end. A row
    keeps its own labels, so it opens on its masked <|im_start|> header and
    the whole-pack label shift never scores one row's token against the row
    before; a row that opens on a target is refused here, before any loss
    sees it. positions are chat_positions' (neo_vla/model/encoder.py) taken
    row by row, so t restarts at 0 on each row and the loader and the model
    share one rule. images lists every row's images in <IMG_CONTEXT> order,
    grid_hw their token grids, and the counters are the pack's own plus the
    two the chain passes in. The pack is held to check_pack before it leaves.
    """
    # Imported here: the model's encoder imports this module's tokenizer loader.
    from ..model.encoder import chat_positions

    if not rows:
        raise ValueError(
            "a chat pack of no rows.\n"
            "Expected at least one: every pack opens with a row, which fits an empty pack.\n"
            "Pass the rows of one pack, at least one."
        )
    size = int(pack_tokens)
    input_ids = torch.full((1, size), CHAT_PAD_ID, dtype=torch.long)
    labels = torch.full((1, size), IGNORE_INDEX, dtype=torch.long)
    segments = torch.full((1, size), PAD_SEGMENT, dtype=torch.long)
    positions = torch.zeros((3, 1, size), dtype=torch.long)
    images: list[torch.Tensor] = []
    grids: list[torch.Tensor] = []
    targets: list[int] = []
    at = 0
    for segment, row in enumerate(rows):
        missing = [key for key in NATIVE_ROW_KEYS if key not in row]
        if missing or tuple(row["input_ids"].shape) != tuple(row["labels"].shape):
            raise ValueError(
                f"a native chat row lacks {missing} or its ids and labels differ in length.\n"
                f"Expected {NATIVE_ROW_KEYS}, ids and labels of one length.\n"
                "Emit rows from NeoChatRenderer.sample."
            )
        length = int(row["input_ids"].shape[0])
        if at + length > size:
            raise ValueError(
                f"row {segment} of {length} tokens ends at {at + length}, past the pack's {size}.\n"
                "Expected the rows to fit the pack: a pack closes before a row "
                "that does not fit.\n"
                "Pass fewer rows, or a larger pack_tokens."
            )
        if int(row["labels"][0]) != IGNORE_INDEX:
            raise ValueError(
                f"row {segment} opens with label {int(row['labels'][0])}.\n"
                f"Expected {IGNORE_INDEX}: the whole-pack shift would score the previous "
                "row's last token against this row's first.\n"
                "Emit rows from NeoChatRenderer.sample, whose rows open on a masked header."
            )
        row_grids = torch.tensor(
            [[int(image.shape[-2]) // NEO_TOKEN_PIXELS, int(image.shape[-1]) // NEO_TOKEN_PIXELS]
             for image in row["images"]], dtype=torch.long,
        ).reshape(-1, 2)
        t_index, h_index, w_index = chat_positions(row["input_ids"][None], row_grids)
        span = slice(at, at + length)
        input_ids[0, span] = row["input_ids"]
        labels[0, span] = row["labels"]
        segments[0, span] = segment
        positions[0, 0, span] = t_index[0]
        positions[1, 0, span] = h_index[0]
        positions[2, 0, span] = w_index[0]
        images.extend(row["images"])
        grids.append(row_grids)
        targets.append(int((row["labels"] != IGNORE_INDEX).sum()))
        at += length
    pack = {
        PACK_IDS: input_ids,
        PACK_LABELS: labels,
        PACK_SEGMENTS: segments,
        PACK_SEGMENT_TARGETS: torch.tensor(targets, dtype=torch.long),
        PACK_POSITIONS: positions,
        PACK_IMAGES: images,
        PACK_GRID_HW: torch.cat(grids),
        PACK_COUNTERS: counter_vector(
            rows=len(rows), tokens=at, images=len(images),
            dropped_overlong=dropped_overlong, spilled=spilled,
        ),
    }
    check_pack(pack, pack_tokens=size)
    return pack


def check_chat_format(chat_format: str, **given: object) -> None:
    """Raise unless the node names NEO's format and gives exactly its keys.

    neo_native reads min_pixels and max_pixels. The retired legacy format read
    image_size and pixels_per_token; a node that sets either, or names that
    format, was written for it and is refused by name rather than run with the
    keys ignored.
    """
    if chat_format in RETIRED_CHAT_FORMATS:
        raise ValueError(
            f"chat_format {chat_format!r} is retired: {RETIRED_CHAT_FORMATS[chat_format]}.\n"
            f"Expected {NATIVE_FORMAT!r}, the only chat format this tree renders.\n"
            "Select a data group whose understanding node is neo_native."
        )
    if chat_format not in CHAT_FORMATS:
        raise ValueError(
            f"chat_format is {chat_format!r}, expected {NATIVE_FORMAT!r}.\n"
            "State the format the understanding node renders, or leave the key out."
        )
    retired = [name for name in RETIRED_CHAT_KEYS if given.get(name) is not None]
    if retired:
        raise ValueError(
            f"the chat node sets {retired}, keys of the retired legacy format.\n"
            "Expected min_pixels and max_pixels only: each image keeps its own size.\n"
            "Remove them from the understanding node."
        )
    missing = [name for name in ("min_pixels", "max_pixels") if given.get(name) is None]
    if missing:
        raise ValueError(
            f"chat_format {chat_format!r} lacks {missing}.\n"
            "Expected both min_pixels and max_pixels, the bounds smart_resize reads.\n"
            "Set them in the understanding node (NEO's own min_pixels is 65536)."
        )


def check_pixel_convention(pixel_convention: str | None) -> None:
    """Raise unless a chat corpus runs under the NEO-native pixel convention.

    Its images reach the tower as uint8 / 255 * 2 - 1, which is NEO's pixels
    only behind the tower's ImageNet gate that neo_native selects.
    """
    if pixel_convention != NATIVE_PIXEL_CONVENTION:
        raise ValueError(
            f"a chat corpus was given pixel_convention {pixel_convention!r}.\n"
            f"Expected {NATIVE_PIXEL_CONVENTION!r}: NEO's chat pixels are ImageNet-normalised, "
            "and only that convention's tower gate turns this corpus's [-1, 1] into them.\n"
            "Set pixel_convention: ${model.vision.pixel_convention} and build the model "
            "neo_native."
        )


def read_sharegpt4v_entries(path: Path) -> list[dict]:
    """The corpus file -> its raw records, for the native format.

    Only the file's shape is checked here, in the process that launches the
    run; each record is parsed when it is drawn, and a bad one is a counted
    skip there rather than a refusal of the whole corpus.
    """
    with path.open(encoding="utf-8") as handle:
        entries = json.load(handle)
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            f"{path} holds {type(entries).__name__} rather than a list of records.\n"
            "Expected the ShareGPT4V json: a non-empty list of conversations.\n"
            "Point records at the corpus file."
        )
    return entries


def check_no_pack(**stated) -> None:
    """Refuse a pack key stated non-null on a corpus that never packs (ShareGPT4V-COCO).

    A group on this corpus states pack_tokens, pack_micro_batches and force_pack
    null: a value here would be a packed recipe reaching a corpus that cannot
    build packs.
    """
    packed = {name: value for name, value in stated.items() if value is not None}
    if packed:
        raise ValueError(
            f"ShareGPT4V-COCO is given {packed}.\n"
            "Expected pack_tokens, pack_micro_batches and force_pack null: this corpus is "
            "one chat row per robot row, never a pack.\n"
            "State them null in the group (config/data/*_cotrain_cocogptshare.yaml)."
        )


class ShareGPT4VCoco(Dataset):
    """The COCO ShareGPT co-train corpus, as a map from an index to a chat sample.

    Each image keeps its native size under min_pixels and max_pixels, and a
    record that cannot make a row is replaced by a deterministic redraw from
    its index's stream. image_size and pixels_per_token are accepted only to be
    refused by name: they belong to the retired legacy format. pack_tokens,
    pack_micro_batches and force_pack are accepted only as null, the value the
    COCO ShareGPT groups state: this corpus is one chat row per robot row,
    never a pack, and a stated pack is refused by name.
    """

    def __init__(
        self,
        *,
        records: str | Path,
        images: str | Path,
        tokenizer_path: str | Path,
        max_tokens: int,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        pixel_convention: str | None = None,
        chat_format: str = NATIVE_FORMAT,
        thinking_method: str = "tag",
        seed: int = 0,
        attempts: int = 256,
        image_size: int | None = None,
        pixels_per_token: int | None = None,
        pack_tokens: int | None = None,
        pack_micro_batches: list | None = None,
        force_pack: list | None = None,
    ):
        check_no_pack(pack_tokens=pack_tokens, pack_micro_batches=pack_micro_batches,
                      force_pack=force_pack)
        check_chat_format(
            chat_format,
            image_size=image_size,
            pixels_per_token=pixels_per_token,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        check_pixel_convention(pixel_convention)
        self.chat_format = chat_format
        self.pixel_convention = pixel_convention
        self.images = Path(images)
        self.renderer = NeoChatRenderer(
            tokenizer_path=tokenizer_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_tokens=max_tokens,
            thinking_method=thinking_method,
        )
        self.records = read_sharegpt4v_entries(Path(records))
        # This corpus never packs its chat rows.
        self.pack = None
        self.seed = int(seed)
        self.attempts = int(attempts)
        self.skips: dict[str, int] = {}
        self.last_error: str | None = None
        logger.info(
            "sharegpt4v_coco: %d records, native images in %d..%d pixels, rows up to %d tokens",
            len(self.records), self.renderer.min_pixels, self.renderer.max_pixels,
            self.renderer.max_tokens,
        )

    def __len__(self) -> int:
        """The number of records that survived the read."""
        return len(self.records)

    def record(self, index: int) -> dict:
        """The conversation at an index, before tokenising. For the verify stage."""
        return self.records[index]

    def count(self, key: str, message: str) -> None:
        """Record one skipped draw and why."""
        self.skips[key] = self.skips.get(key, 0) + 1
        self.last_error = message.splitlines()[0] if message else key

    def locate(self, ref: str) -> str:
        """A record's image reference -> the file under the corpus's image root."""
        return str(self.images / ref)

    @staticmethod
    def fetch(uri: str) -> bytes:
        """A file's bytes, or MissingObject when it is absent or unreadable."""
        try:
            return Path(uri).read_bytes()
        except OSError as missing:
            raise MissingObject(f"{uri}: {missing}") from missing

    def __getitem__(self, index: int) -> dict:
        """One chat sample: input_ids and labels at the row's own length, images uint8.

        collate_chat pads a batch of them. A draw that fails is replaced by
        index from the index's own stream, and `attempts` failures in a row
        raise.
        """
        rng = random.Random(stream_seed(int(index), seed=self.seed))
        chosen = int(index)
        for _ in range(self.attempts):
            sample = native_draw(
                self.renderer,
                self.records[chosen],
                locate=self.locate,
                fetch=self.fetch,
                images_per_row="any",
                count=self.count,
            )
            if sample is not None:
                return sample
            chosen = rng.randrange(len(self.records))
        raise RuntimeError(
            f"{self.attempts} consecutive draws held no usable record, skips {self.skips}.\n"
            f"The last failure was {self.last_error}.\n"
            "Expected most records to render; check the corpus and its image root."
        )
