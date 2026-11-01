"""The vision-language half of a co-train batch: plain chat, no robot tokens.

One chat format, neo_native: a record renders the way NEOpp-2B-SFT's own
trainer did (its tag-mode chat function, every assistant turn supervised), with
0..k images per row, each resized by smart_resize at 32 pixels a token with
PIL's bicubic and carried as uint8. encode_chat_rows scales them into the
[-1, 1] contract and the tower's ImageNet gate turns that into NEO's pixels.
The batch keeps rows at their own lengths.

A row carries no sections, no anchor ids and no conditioning column, which is
the asymmetry that keeps the backbone's language ability rather than adapting
it to the robot scaffold. ShareGPT4V-COCO and MT_Und both read this way. The
earlier 'legacy' format is refused by name, and so are its keys.
"""
from __future__ import annotations

import bisect
import functools
import io
import json
import logging
import math
import os
import random
import re
import sys
import threading
import time
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
from typing import BinaryIO, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_file
from transformers import AutoTokenizer

from ..model.backbone.neo_llm import IMG_CONTEXT_ID, IMG_END_ID, IMG_START_ID
from .chat_pack import (
    FORCE_PACK_FIXTURES,
    PACK_COUNTERS,
    PACK_GRID_HW,
    PACK_IDS,
    PACK_IMAGES,
    PACK_LABELS,
    PACK_POSITIONS,
    PACK_SEGMENT_TARGETS,
    PACK_SEGMENTS,
    PAD_SEGMENT,
    chain_row_seed,
    check_pack,
    counter_vector,
    pack_config,
)
from .line_index import LineIndex, entry_need
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
              "answer supervised, and MT_Und drawn by byte offset",
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
# own loader reads only 'conversations'; 3 of 6,000 MT_Und draws use this one.
MESSAGE_SPEAKERS = {"user": "human", "assistant": "gpt", "system": "system"}
# Whether a native corpus keeps every row NEO trains on or only one-image rows.
IMAGES_PER_ROW = ("any", "one")
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
# the checkpoint never trained on, in about 28% of MT_Und rows. These probes
# refuse such a tokenizer.
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


def neo_rgb(image: Image.Image, *, from_object_store: bool) -> Image.Image:
    """A decoded image -> RGB, the way NEO's loader converts it.

    From the object store the image is converted straight to RGB, which drops
    any alpha. From a file an RGBA image with transparent pixels is first laid
    on the contrasting backdrop. Both are TCSLoader.__call__ in NEO's trainer.
    """
    if from_object_store:
        return image.convert("RGB")
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

    def pixels(
        self, picture: Image.Image, grid: tuple[int, int], *, from_object_store: bool
    ) -> torch.Tensor:
        """An opened image and its grid -> uint8 [3, H, W] at NEO's native size.

        PIL's bicubic, which is what NEO's image.resize applied by default to
        an RGB image. uint8 rather than float is what crosses the loader;
        encode_chat_rows scales it into the [-1, 1] contract on the device.
        """
        rgb = neo_rgb(picture, from_object_store=from_object_store)
        size = (grid[1] * NEO_TOKEN_PIXELS, grid[0] * NEO_TOKEN_PIXELS)
        resized = rgb.resize(size, Image.Resampling.BICUBIC)
        array = np.asarray(resized, dtype=np.uint8)
        return torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()

    def sample(
        self, layout: ChatLayout, blobs: list[bytes], stored: list[bool]
    ) -> dict:
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
        images = [
            self.pixels(picture, grid, from_object_store=from_store)
            for picture, grid, from_store in zip(pictures, grids, stored)
        ]
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
    there. Any other fetch failure is not caught: a store or mount that does
    not answer is a launch fault, not a bad record. Everything else is the
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
        return renderer.sample(
            layout, blobs, [uri.startswith(OBJECT_PREFIX) for uri in uris]
        )
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
            "Build packs with MtUnd.step_packs."
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
                "Expected the rows to fit the pack: the chain closes a pack before a row "
                "that does not fit.\n"
                "Build packs with MtUnd.step_packs."
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
    null: a value here would be stage 2's NEO pack reaching a run this corpus
    cannot build it for.
    """
    packed = {name: value for name, value in stated.items() if value is not None}
    if packed:
        raise ValueError(
            f"ShareGPT4V-COCO is given {packed}.\n"
            "Expected pack_tokens, pack_micro_batches and force_pack null: post-train chat is "
            "one row per robot row, never a pack.\n"
            "State them null in the post-train group (config/data/*_v7_cotrain.yaml)."
        )


class ShareGPT4VCoco(Dataset):
    """The post-train co-train corpus, as a map from an index to a chat sample.

    Each image keeps its native size under min_pixels and max_pixels, and a
    record that cannot make a row is replaced by a deterministic redraw from
    its index's stream. image_size and pixels_per_token are accepted only to be
    refused by name: they belong to the retired legacy format. pack_tokens,
    pack_micro_batches and force_pack are accepted only as null, the value the
    post-train groups state: post-train chat is one row per
    robot row, never stage 2's pack, and a stated pack is refused by name.
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
        # Post-train chat rows are never packed.
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


# The internal object store MT_Und's images sit behind, the client that reads
# it, and the two packages that client needs. Nothing else reads this store:
# the ordinary clients do not speak its protocol.
OBJECT_PREFIX = "s3://"
OBJECT_CLIENT = "aoss_client"
OBJECT_CLIENT_NEEDS = ("environs", "marshmallow", "boto3")

# How long one object-store request may wait. The sdk builds its boto3 clients
# with no config, so botocore's defaults applied: 60 s to connect and 60 s of
# silence between reads, then retries. A request the store stopped answering
# held a loader worker, its rank and, through the gradient sync, every rank for
# a minute or more (in stage 2, 38 of the 42 stalls caught were this GET). Now a
# connection silent for 15 s is dropped and the SAME object is asked
# for again on a fresh one, so no draw changes. The read bound is silence
# between reads, not a limit on the whole transfer.
OBJECT_CONNECT_SECONDS = 5
OBJECT_READ_SECONDS = 15
# Few attempts per request: botocore's standard mode draws each retry from a
# per-client quota, and read_bytes below does the long waiting itself.
OBJECT_ATTEMPTS = 3
# How long read_bytes keeps asking for one object while the store does not
# answer, drops the connection or reports itself busy (5xx, throttling):
# botocore does not retry a body that stalls after its headers, nor anything
# once its quota is spent. read_bytes is the one long retry loop: the sdk's own
# get retry is cut to a single try (single_sdk_try), so one ask is at most 3
# attempts of 15 s. The patience, plus that one ask, stays under the 30-minute
# collective timeout (NEO_VLA_DIST_TIMEOUT_MINUTES), so a worker whose read
# gives up does so before the whole job is aborted. The bound is per read, not
# per batch. Until the store has answered a worker once, the patience is
# short: a wrong endpoint, name, certificate or proxy is a launch fault.
OBJECT_PATIENCE_SECONDS = 1200
OBJECT_FIRST_PATIENCE_SECONDS = 120
OBJECT_BACKOFF_MAX_SECONDS = 30
OBJECT_WARN_EVERY_SECONDS = 60
# Backed by os.urandom, so forked workers do not share a jitter sequence and no
# draw's generator is touched.
OBJECT_JITTER = random.SystemRandom()
OBJECT_BUSY_STATUS = frozenset({429, 500, 502, 503, 504})
OBJECT_BUSY_CODES = frozenset({
    "SlowDown", "RequestTimeout", "ServiceUnavailable", "InternalError", "Throttling",
    "ThrottlingException", "TooManyRequests", "TooManyRequestsException",
    "RequestLimitExceeded", "BandwidthLimitExceeded",
})
BOUNDED_CLASSES: set[type] = set()
SINGLE_TRY_CLASSES: set[type] = set()
BOUND_LOCK = threading.RLock()


def object_store_transient(failure: BaseException) -> bool:
    """No answer, a dropped connection, or a busy store (5xx, throttling)."""
    from botocore.exceptions import (
        ClientError,
        ConnectionError,
        HTTPClientError,
        IncompleteReadError,
    )

    if isinstance(failure, (HTTPClientError, ConnectionError, IncompleteReadError)):
        return True
    # The sdk wraps any other botocore ClientError as S3ClientError(err).
    if not any(kind.__name__ == "S3ClientError" for kind in type(failure).__mro__):
        return False
    wrapped = failure.args[0] if failure.args else None
    if not isinstance(wrapped, ClientError):
        return False
    status = wrapped.response.get("ResponseMetadata", {}).get("HTTPStatusCode") or 0
    code = wrapped.response.get("Error", {}).get("Code")
    return status in OBJECT_BUSY_STATUS or code in OBJECT_BUSY_CODES


class MissingObject(LookupError):
    """The object store answered, and does not hold this uri.

    Distinct from the client raising, which means the store could not be
    reached at all. The first is one bad record among 32.7M draws and is
    counted like any other skip; the second is a launch fault and still stops
    the run. Before this existed they were the same event: the client returns
    None for a missing object, bytearray(None) raised TypeError, nothing
    caught it, and the first missing object killed a dataloader worker and
    with it the run.
    """


# How many open annotation files one worker keeps.
HANDLE_LIMIT = 32
# How many records of one entry a native draw tries before it picks another
# entry. NEO's reader never gives up on an entry, it reads the next line; at our
# max_tokens more rows are dropped than at NEO's, and replacing them inside the
# entry keeps the entry's share at NEO's len rather than len times its keep
# rate. An entry that keeps one row in ten is left after 8 misses 43% of a draw
# (0.9 ** 8); one that keeps half, 0.4%.
ENTRY_PATIENCE = 8

# The forced fixtures (MtUnd.fixture_pack): image i of a fixture is
# uint8 noise from a generator seeded FIXTURE_SEED + i, and f_cnt's images are
# 8 x 8 token grids, 256 x 256 pixels, NEO's min_pixels exactly.
FIXTURE_SEED = 20_261_006
FIXTURE_COUNT_GRID = (8, 8)


def object_store_client(sdk_path: Path) -> type:
    """The client class MT_Und's images are read through, or a raise naming it.

    Imported here rather than at the top of the module because it installs on
    the cluster and nowhere else, and the shipped corpus must load without it.
    The class is returned instead of an instance: the client holds sockets and
    does not survive a fork, so it is built inside the worker that reads with
    it, after the loader has forked.
    """
    if str(sdk_path) not in sys.path:
        sys.path.append(str(sdk_path))
    try:
        from aoss_client.ceph.s3.s3_client import S3Client
        from aoss_client.client import Client
        from aoss_client.mixed_client import MixedClient
    except ImportError as missing:
        raise ImportError(
            f"MT_Und reads its images through {OBJECT_CLIENT}, which did not "
            f"import: {missing}.\n"
            f"Expected the internal sdk under {sdk_path}, plus "
            f"{' and '.join(OBJECT_CLIENT_NEEDS)}.\n"
            "Install them, select object_store 'exclude', or co-train on ShareGPT4VCoco."
        ) from missing
    # Here, in the process that forks the loader, so every worker inherits both.
    bound_object_requests(S3Client)
    single_sdk_try(MixedClient)
    return Client


def single_sdk_try(mixed_class: type) -> None:
    """Make every sdk client ask the store once per get.

    The sdk wraps each get in its own retry, get_retry_max times (10 unless
    aoss.conf says otherwise) with no pause, and read_bytes only reaches its
    deadline between gets, so one get could last 10 x 3 x 15 s. read_bytes is
    the one retry loop, with the backoff and the patience, so each get tries
    once. The sdk reads the count from the instance on every get.
    """
    if mixed_class in SINGLE_TRY_CLASSES:
        return
    construct = mixed_class.__init__

    @functools.wraps(construct)
    def construct_trying_once(self, *args, **kwargs):
        construct(self, *args, **kwargs)
        logger.info("object store: the sdk's %d tries per get become 1; read_bytes retries",
                    self._get_retry_max)
        self._get_retry_max = 1

    mixed_class.__init__ = construct_trying_once
    SINGLE_TRY_CLASSES.add(mixed_class)


def bound_object_requests(s3_class: type) -> None:
    """Give every boto3 client and resource an s3 class builds the deadline.

    The sdk passes boto3 no config for its signed clients, so the deadline is
    merged into whatever config each build asks for, and only while the class's
    own constructor runs: boto3 elsewhere in the process is left as it was.
    """
    if s3_class in BOUNDED_CLASSES:
        return
    import boto3.session
    from botocore.config import Config

    deadline = Config(
        connect_timeout=OBJECT_CONNECT_SECONDS,
        read_timeout=OBJECT_READ_SECONDS,
        retries={"mode": "standard", "total_max_attempts": OBJECT_ATTEMPTS},
    )

    def with_deadline(build):
        def build_with_deadline(self, *args, config=None, **kwargs):
            merged = deadline if config is None else config.merge(deadline)
            return build(self, *args, config=merged, **kwargs)
        return build_with_deadline

    construct = s3_class.__init__

    @functools.wraps(construct)
    def construct_with_deadline(self, *args, **kwargs):
        # One construction at a time: the swap is on the boto3 class, so a second
        # thread building meanwhile would get, or leave behind, the wrong one.
        with BOUND_LOCK:
            session = boto3.session.Session
            client, resource = session.client, session.resource
            session.client, session.resource = with_deadline(client), with_deadline(resource)
            try:
                construct(self, *args, **kwargs)
            finally:
                session.client, session.resource = client, resource

    s3_class.__init__ = construct_with_deadline
    BOUNDED_CLASSES.add(s3_class)


def resolve_reference(root: object, ref: str) -> str | None:
    """An entry's root and a record's image reference -> one uri, or nothing.

    An absolute reference stands on its own; a relative one needs a root, and
    the entries whose root is empty or absent carry absolute references. One
    that does not is unresolvable and its record is skipped, rather than joined
    onto the filesystem root where some other file may answer to the name.
    """
    if not ref:
        return None
    if ref.startswith(OBJECT_PREFIX) or ref.startswith("/"):
        return ref
    if not isinstance(root, str) or not root:
        return None
    return root.rstrip("/") + "/" + ref


class MtUnd(Dataset):
    """The stage-2 chat corpus, drawn from a registry of jsonl files.

    An index seeds a draw rather than addressing a record. The draw is NEO's
    trainer's read (line_index.py): an entry by ceil(length * repeat_time),
    then one of the line positions NEO reads in it, uniformly, through a line
    index built once. A record that makes no row is replaced by another line
    of the same entry, as NEO's reader moves to the next line, and after
    ENTRY_PATIENCE misses by another entry. It keeps every still-image and
    text-only row NEO's trainer reads (images_per_row 'any'), or one-image rows
    only ('one'). Every skip is counted and reported.

    denylist names the entry keys to drop. It is a required argument because
    this corpus contains verbatim MMMU and MMMU-Pro test items; those two
    remain invalid as evaluations for any model trained here, denylist or not.
    """

    def __init__(
        self,
        *,
        registry: str | Path,
        denylist: str | Path,
        tokenizer_path: str | Path,
        max_tokens: int,
        draws: int,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        pixel_convention: str | None = None,
        line_index: str | Path | None = None,
        chat_format: str = NATIVE_FORMAT,
        seed: int = 0,
        object_store: str = "require",
        sdk_path: str | Path | None = None,
        client_config: str | Path | None = None,
        attempts: int = 256,
        thinking_method: str = "tag",
        images_per_row: str = "any",
        image_size: int | None = None,
        pixels_per_token: int | None = None,
        pack_tokens: int | None = None,
        force_pack: list | None = None,
        pack_micro_batches: list | None = None,
        pack_every_steps: int | None = None,
    ):
        check_chat_format(
            chat_format,
            image_size=image_size,
            pixels_per_token=pixels_per_token,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        check_pixel_convention(pixel_convention)
        # The packed chat half: None is one row per
        # robot row, today's draw by index; a PackConfig is one pack of its
        # length per chat micro-batch (every one, or those pack_micro_batches
        # names), drawn by step_packs for a whole step; on the steps pack_every_steps
        # picks, every step at its default.
        self.pack = pack_config(pack_tokens, force_pack, max_tokens=max_tokens,
                                pack_micro_batches=pack_micro_batches,
                                pack_every_steps=pack_every_steps)
        self.pixel_convention = pixel_convention
        if images_per_row not in IMAGES_PER_ROW:
            raise ValueError(
                f"images_per_row is {images_per_row!r}, expected one of {IMAGES_PER_ROW}.\n"
                "'any' keeps every row NEO trains on; 'one' keeps one-image rows."
            )
        self.chat_format = chat_format
        self.images_per_row = images_per_row
        if line_index is None:
            raise ValueError(
                "MtUnd was given no line index.\n"
                "Expected the directory tools/build_mtund_line_index.py wrote over this "
                "registry: the draw reads the line positions NEO's trainer reads.\n"
                "Build one and name it in line_index (NEO_VLA_MTUND_LINE_INDEX)."
            )
        self.line_index = LineIndex(line_index)
        # The index's record for each kept entry.
        self.index_records: dict[str, dict] = {}
        if object_store not in ("require", "exclude"):
            raise ValueError(
                f"object_store is {object_store!r}, expected 'require' or 'exclude'.\n"
                "'require' reads the whole registry, 'exclude' drops its object store "
                "entries.\n"
                "State which, rather than letting a missing client decide the mixture."
            )
        self.client_class: type | None = None
        self.client = None
        self.store_answered = False
        self.client_config = client_config
        if object_store == "require":
            if sdk_path is None or client_config is None:
                raise ValueError(
                    "object_store 'require' was selected without sdk_path and "
                    "client_config.\n"
                    "Expected both: they name the internal sdk and its configuration.\n"
                    "State them in the config, or select object_store 'exclude'."
                )
            self.client_class = object_store_client(Path(sdk_path))
        self.renderer = NeoChatRenderer(
            tokenizer_path=tokenizer_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_tokens=max_tokens,
            thinking_method=thinking_method,
        )
        self.draws = int(draws)
        self.seed = int(seed)
        self.attempts = int(attempts)
        self.skips = {
            "image_count": 0,
            "unresolvable": 0,
            "no_turns": 0,
            "no_client": 0,
            # A record whose text will not render into a supervised row, and
            # one whose image bytes will not decode. Both were raises out of
            # __getitem__ before, which a dataloader worker does not survive:
            # 32.7M samples is enough draws that a corpus-wide rate of one in
            # ten thousand is still a dead run on a multi-day pretrain. They
            # are skips, counted, and the consecutive-failure guard below is
            # what still stops a run whose renderer is broken outright.
            "unrenderable": 0,
            "undecodable": 0,
            # An object the store answered for but does not hold.
            "unfetchable": 0,
            # A line that is not json, a video row, fewer placeholders than
            # images, a literal <IMG_CONTEXT> in prose, no supervised turn, an
            # aspect over 200, and a row longer than max_tokens, which is
            # dropped, never cut.
            "unparsable": 0,
            "video": 0,
            "placeholder": 0,
            "reserved_token": 0,
            "no_answer": 0,
            "aspect": 0,
            "too_long": 0,
        }
        # The last message behind an unrenderable or undecodable skip, so the
        # guard that fires after `attempts` consecutive failures names a cause
        # rather than only a count.
        self.last_error: str | None = None
        # Opened on first use inside the worker: file handles carry an offset,
        # and one inherited across a fork is shared by every worker that seeks it.
        # handles_pid is the process that opened them, so a worker forked from a
        # parent that already drew opens its own (handle).
        self.handles: dict[str, BinaryIO] = {}
        self.sizes: dict[str, int] = {}
        self.handles_pid = os.getpid()
        self.entries, self.keys, self.cumulative = self.read_registry(
            Path(registry), Path(denylist)
        )

    def read_registry(
        self, registry: Path, denylist: Path
    ) -> tuple[dict, list[str], list[float]]:
        """The registry -> the entries kept, their keys and their mass totals."""
        denied = set(json.loads(denylist.read_text(encoding="utf-8")))
        listed = json.loads(registry.read_text(encoding="utf-8"))
        kept: dict[str, dict] = {}
        keys: list[str] = []
        weights: list[float] = []
        declared = 0.0
        unread = 0
        for key, entry in listed.items():
            # NEO's len and its dataset weight (multimodal_dataset.py:164).
            mass = float(entry_need(entry))
            declared += mass
            stored = self.client_class is None and str(entry.get("root", "")).startswith(
                OBJECT_PREFIX
            )
            if mass <= 0.0 or key in denied or stored:
                continue
            record = self.line_index.check(key, entry)
            if record["used"] < 1:
                unread += 1
                continue
            self.index_records[key] = record
            kept[key] = entry
            keys.append(key)
            weights.append(mass)
        if unread:
            logger.warning("mt_und: %d entries have an empty annotation file and are dropped",
                           unread)
        if not keys:
            raise ValueError(
                f"{registry} kept no entries against {denylist}.\n"
                "Expected at least one: every entry was denied, empty or object stored.\n"
                "Widen the denylist or select object_store 'require'."
            )
        cumulative = list(accumulate(weights))
        logger.info(
            "mt_und: %d of %d entries, %.2f%% of the declared mass, %d denied keys",
            len(keys),
            len(listed),
            100.0 * cumulative[-1] / declared,
            len(denied),
        )
        return kept, keys, cumulative

    def __len__(self) -> int:
        """draws: how many chat samples one pass over the index stream is."""
        return self.draws

    def handle(self, path: str) -> BinaryIO:
        """One open annotation file of this process, keeping at most HANDLE_LIMIT of them."""
        if self.handles_pid != os.getpid():
            # A loader worker forked from a process that had drawn: the handles
            # it inherited share one file offset with the parent's and every
            # sibling's, so a seek in one moves the line the others read, and
            # the same draw reads another record. Close this process's copies,
            # which releases its descriptors and leaves the shared offset
            # alone, and open its own; the parent's handles are untouched.
            for inherited in self.handles.values():
                inherited.close()
            self.handles, self.sizes, self.handles_pid = {}, {}, os.getpid()
        opened = self.handles.get(path)
        if opened is None:
            if len(self.handles) >= HANDLE_LIMIT:
                self.handles.pop(next(iter(self.handles))).close()
            opened = self.handles[path] = open(path, "rb")
            self.sizes[path] = opened.seek(0, 2)
        return opened

    def read_bytes(self, uri: str) -> torch.Tensor:
        """One image uri -> its encoded bytes, from the object store or a file."""
        if not uri.startswith(OBJECT_PREFIX):
            # A path the registry names but the filesystem does not hold is the
            # same event as an object the store answers "no" for: one bad record
            # among 32.7M, not a launch fault. Raise the same exception so the
            # same guard in native_draw counts it. Without this, read_file raises
            # RuntimeError, nothing catches it, and the worker -- then its rank,
            # then the whole job via the NCCL watchdog -- dies. That is exactly
            # how the first stage-2 production launch died at step 20, on
            # .../CASIA-CSDB/data/val_images/23526.png.
            # A file that is present and unreadable is counted below; a mount
            # that is gone still stops the run through the consecutive-failure
            # guard in __getitem__.
            if not os.path.isfile(uri):
                raise MissingObject(uri)
            try:
                return read_file(uri)
            except (RuntimeError, OSError, ValueError) as broken:
                # PRESENT but unreadable: a non-ASCII name under another user's
                # tree, a permission, a truncated write. One such file among
                # 32.7M records is a BAD RECORD, and a per-read raise here kills
                # a DataLoader worker, then its rank, then the job -- which is
                # how step 140 died on
                # .../huangkaili/.../202510_dmt_sessions/.../<non-ascii>.
                # The systemic case -- a mount that is actually gone -- is still
                # caught, and caught better: __getitem__ raises after `attempts`
                # consecutive failures and prints the skip census with
                # last_error, which distinguishes "one bad file" from "nothing
                # reads" in a way a single raise cannot.
                raise MissingObject(f"{uri}: unreadable ({broken})") from broken
        if self.client is None:
            # Under spawn or forkserver the worker imported its own sdk, which the
            # binding made before the fork never reached; under fork this is a no-op.
            from aoss_client.ceph.s3.s3_client import S3Client
            from aoss_client.mixed_client import MixedClient
            bound_object_requests(S3Client)
            single_sdk_try(MixedClient)
            self.client = self.client_class(str(self.client_config))
        patience = OBJECT_PATIENCE_SECONDS if self.store_answered else OBJECT_FIRST_PATIENCE_SECONDS
        give_up = time.monotonic() + patience
        failures, warned_at = 0, float("-inf")
        while True:
            try:
                encoded = self.client.get(uri)
                break
            except Exception as failure:
                now = time.monotonic()
                if not object_store_transient(failure) or now >= give_up:
                    raise
                failures += 1
                if now - warned_at >= OBJECT_WARN_EVERY_SECONDS:
                    first = (str(failure).splitlines() or [type(failure).__name__])[0]
                    logger.warning("object store did not answer for %s (%d failures so far: %s); "
                                   "asking again", uri, failures, first)
                    warned_at = now
                backoff = OBJECT_JITTER.uniform(0, min(OBJECT_BACKOFF_MAX_SECONDS, 2 ** failures))
                time.sleep(max(0.0, min(give_up - now, backoff)))
        self.store_answered = True
        if encoded is None:
            raise MissingObject(uri)
        return torch.frombuffer(bytearray(encoded), dtype=torch.uint8)

    def count(self, key: str, message: str) -> None:
        """Record one skipped native draw and why."""
        self.skips[key] = self.skips.get(key, 0) + 1
        self.last_error = message.splitlines()[0] if message else key

    def pick_entry(self, rng: random.Random) -> str:
        """An entry's key, by the mass read_registry gave it."""
        target = rng.random() * self.cumulative[-1]
        return self.keys[bisect.bisect_right(self.cumulative, target)]

    def read_line(self, key: str, rng: random.Random) -> object:
        """One of the line positions NEO reads in an entry, uniformly, parsed."""
        record = self.index_records[key]
        handle = self.handle(str(self.entries[key]["annotation"]))
        handle.seek(self.line_index.offset(record, rng.randrange(record["used"])))
        return json.loads(handle.readline())

    def attempt_native(self, rng: random.Random, key: str) -> dict | None:
        """One native draw from an entry, or nothing when the record cannot fill a row.

        A line that is not json is a counted skip here; a read that fails is
        still a raise, because that is the file, not the record.
        """
        entry = self.entries[key]
        try:
            record = self.read_line(key, rng)
        # json raises ValueError on a line that is not json or not utf-8, and
        # RecursionError on one nested past the interpreter's limit.
        except (ValueError, RecursionError) as error:
            self.count("unparsable", f"{entry.get('annotation')}: {type(error).__name__}: {error}")
            return None
        if record is None:
            self.count("unparsable", f"{entry.get('annotation')}: a null line")
            return None
        root = entry.get("root")

        def locate(ref: str) -> str:
            uri = resolve_reference(root, ref)
            if uri is None:
                raise ChatSkip("unresolvable", f"{ref} has no root to resolve against")
            if uri.startswith(OBJECT_PREFIX) and self.client_class is None:
                raise ChatSkip("no_client", f"{uri} sits in the object store")
            return uri

        def fetch(uri: str) -> bytes:
            try:
                return self.read_bytes(uri).numpy().tobytes()
            except ValueError as empty:
                # torch.frombuffer raises this on an object of zero bytes; read_bytes
                # itself is the deadline code and stays as it is.
                raise MissingObject(f"{uri}: {empty}") from empty

        return native_draw(
            self.renderer,
            record,
            locate=locate,
            fetch=fetch,
            images_per_row=self.images_per_row,
            count=self.count,
        )

    def draw_row(self, rng: random.Random, trace: list | None = None) -> dict:
        """One chat row from rng: an entry by mass, then lines of it until one renders.

        A record that makes no row is replaced by another line of the same
        entry and, after ENTRY_PATIENCE misses, by another entry; `attempts`
        failures in a row raise. A row over max_tokens is one such miss: it is
        dropped and redrawn, never cut. rng is the row's whole randomness, so a
        row is a pure function of its seed while storage answers alike. With
        trace, the draw appends {"entry", "attempts", "skips"}: the entry the
        row came from, the attempts it took and the skips met on the way.
        """
        before = dict(self.skips) if trace is not None else None
        key, misses = None, 0
        for attempt in range(self.attempts):
            if key is None or misses == ENTRY_PATIENCE:
                key, misses = self.pick_entry(rng), 0
            sample = self.attempt_native(rng, key)
            misses += 1
            if sample is not None:
                if trace is not None:
                    trace.append({"entry": key, "attempts": attempt + 1, "skips": {
                        name: count - before.get(name, 0)
                        for name, count in self.skips.items() if count != before.get(name, 0)
                    }})
                return sample
        raise RuntimeError(
            f"{self.attempts} consecutive draws held no usable record, skips {self.skips}.\n"
            f"The last render or decode failure was {self.last_error}.\n"
            "Expected most records to render into a supervised row.\n"
            "Widen the entry set, or select object_store 'require' to reach its images."
        )

    def __getitem__(self, index: int) -> dict:
        """One chat sample, in the fields ShareGPT4VCoco emits: the row index seeds."""
        return self.draw_row(random.Random(stream_seed(int(index), seed=self.seed)))

    def step_packs(self, key: int, packs: int, *, step: int,
                   trace: list | None = None, slot: int = 0, slots: int = 1) -> list[dict]:
        """The `packs` chat packs of one optimizer step of one rank.

        key is the step's chain key (chat_pack.step_key, or chat_pack.group_key
        when steps share a chain), and row j of the chain is draw_row on
        Random(chain_row_seed(key, j)), so the packs are a pure function of
        (seed, key, slot). The S8 chain of packs x slots packs: packs fill in
        order; a row is appended while the pack's tokens plus its own fit
        pack_tokens; the first row that does not fit closes the pack and opens
        the next. The step takes packs slot x packs to (slot + 1) x packs - 1
        of it, drawing the rows before them again and stopping at the first
        row past its own last pack, so slots 0..slots-1 together carry the
        packs of one chain (slots 1: the step's own). Only the chain's last
        straddler is discarded, and the last pack of the last slot counts it
        as spilled. The over-long rows draw_row dropped are counted on the
        pack that was filling. No cap on images: a pack closes on tokens only,
        as NEO's does under native resolution (num_images_expected 32768).
        A step the corpus's force_pack names carries that fixture in every
        pack instead, and draws nothing. With trace, each drawn row's record
        (draw_row's) gains its j, its tokens and the step's pack it went to,
        None for a row in another slot's pack and for the one past the step's
        last pack (at the last slot, the spilled one): what a digest mismatch
        prints per row.
        """
        if self.pack is None:
            raise ValueError(
                "step_packs was asked of a chat corpus with pack_tokens null.\n"
                "Expected a pack length: a corpus without one draws one row per index.\n"
                "State data.train.understanding.pack_tokens, or read rows by index."
            )
        packs, slot, slots = int(packs), int(slot), int(slots)
        if not 0 <= slot < slots:
            raise ValueError(
                f"step {step} asked for slot {slot} of a chain of {slots} slots.\n"
                "Expected 0 <= slot < slots: a step takes its own slot of the chain.\n"
                "Pass slot = PackConfig.chat_step(step) % PackConfig.chain_steps(accumulation)."
            )
        fixture = self.pack.fixture(step)
        if fixture is not None:
            pack = self.fixture_pack(fixture)
            return [dict(pack) for _ in range(packs)]
        size = self.pack.tokens
        # The chain's packs first .. last - 1 are this step's.
        first, last = slot * packs, (slot + 1) * packs
        filling: list[list[dict]] = [[] for _ in range(packs)]
        # chat_pack's dropped_overlong is the renderer's too_long skip.
        dropped = [0] * packs
        k, j, tokens = 0, 0, 0
        while True:
            before = self.skips["too_long"]
            row = self.draw_row(random.Random(chain_row_seed(key, j, seed=self.seed)), trace)
            if k >= first:
                dropped[k - first] += self.skips["too_long"] - before
            length = int(row["input_ids"].shape[0])
            if trace is not None:
                trace[-1].update(j=j, tokens=length, pack=None)
            j += 1
            if tokens + length > size:
                if k + 1 == last:
                    break
                k, tokens = k + 1, 0
            tokens += length
            if k >= first:
                filling[k - first].append(row)
                if trace is not None:
                    trace[-1]["pack"] = k - first
        return [
            collate_chat_pack(rows, pack_tokens=size, dropped_overlong=dropped[m],
                              spilled=int(slot == slots - 1 and m == packs - 1))
            for m, rows in enumerate(filling)
        ]

    def fixture_pack(self, name: str) -> dict:
        """One forced fixture as a pack: a single row of exactly pack_tokens, no padding.

        The packing worst cases, built locally
        with seeded uint8 noise for pixels, so nothing is read from the object
        store and every rank builds the same pack. The row is a masked user
        header, its images, a masked assistant header, then a supervised answer
        that fills the pack exactly. f_tok holds two square images as large as
        max_pixels and the pack allow: 4,096 x 4,096 each at NEO's budget, 2 x
        16,384 tokens. f_cnt holds as many 64-token (256 x 256) images as fit,
        66 columns each: 558 and 36 text tokens at 36,864. It is built anew on
        each call and not kept: force_pack names each fixture for one step, and
        a kept one would hold its ~100 MB of pixels in the worker, in shared
        memory once the loader has sent it, for the rest of the run.
        """
        size = self.pack.tokens
        user = self.renderer.text_ids(f"{NEO_TURN_START}user\n")
        reply = self.renderer.text_ids(f"{NEO_TURN_END}{NEO_TURN_START}assistant\n")
        filler = self.renderer.text_ids("a")[0]
        fixed = len(user) + len(reply) + 1
        if name == "f_tok":
            side = min(math.isqrt(self.renderer.max_pixels) // NEO_TOKEN_PIXELS,
                       math.isqrt((size - fixed - 4) // 2))
            grids = [(side, side)] * 2
        elif name == "f_cnt":
            grids = [FIXTURE_COUNT_GRID] * ((size - fixed) // (FIXTURE_COUNT_GRID[0]
                                                              * FIXTURE_COUNT_GRID[1] + 2))
        else:
            raise ValueError(
                f"force_pack names {name!r}.\n"
                f"Expected one of {FORCE_PACK_FIXTURES}.\n"
                "Name a fixture chat_pack.FORCE_PACK_FIXTURES lists."
            )
        ids = list(user)
        images = []
        for number, (rows, columns) in enumerate(grids):
            ids.extend([IMG_START_ID] + [IMG_CONTEXT_ID] * (rows * columns) + [IMG_END_ID])
            generator = torch.Generator().manual_seed(FIXTURE_SEED + number)
            images.append(torch.randint(
                0, 256, (3, rows * NEO_TOKEN_PIXELS, columns * NEO_TOKEN_PIXELS),
                generator=generator, dtype=torch.uint8,
            ))
        ids.extend(reply)
        context = len(ids)
        ids.extend([filler] * (size - context))
        row = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor([IGNORE_INDEX] * context + ids[context:], dtype=torch.long),
            "images": images,
        }
        return collate_chat_pack([row], pack_tokens=size)
