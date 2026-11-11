"""Frames to view slots: the loader decides the size, nothing upstream does.

snap_grid turns an on-disk frame size into the token grid it will occupy, and
place_views writes each view's tile into the top-left of its own canvas. The
canvas is one shape for the whole run so the prefix is one width; the tile is
the part the encoder reads, and image_shapes is how it knows where the tile
ends.

place_views is the only producer of image_shapes and view_valid. They are
returned together because they are one statement: a slot is either absent, with
a zero shape and a false flag, or present with a real tile. Two producers of
that statement is the defect this module exists to remove. It also measures:
every decoded frame is snapped here and raises unless it takes the grid the
manifest recorded for that view.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy
import torch
from PIL import Image

# Every producer -- this loader, the chat loader and the serving canvas -- hands
# the vision tower pixels in [-1, 1], uint8 / 255 * 2 - 1, and nothing else.
# What the tower does with them is the run's pixel convention, below. NEO's
# patch embedding holds no normalisation of its own (modeling_neo_vit.py has
# no mean or std); NEO's own training and eval feed it ImageNet mean/std after
# ToTensor, so under neo_native the tower converts at its entry.
PIXEL_LOW = -1.0
PIXEL_HIGH = 1.0

# model.vision.pixel_convention -> (what the tower does at its entry, how a
# frame is resized). neo_native is NEO-2B's own and the only convention: ImageNet
# mean/std and PIL's BICUBIC. resize_frame hands the frame's uint8 levels to
# PIL's own BICUBIC, the call NEO resizes with, so a tile is NEO's bit for bit;
# tests/data/test_images.py holds every corpus size equal to PIL on every value.
NEO_NATIVE = "neo_native"
PIXEL_CONVENTIONS = {NEO_NATIVE: ("imagenet", "bicubic")}
RESIZE_FILTERS = ("bicubic",)
# The PIL filter each name in RESIZE_FILTERS calls.
PIL_FILTERS = {"bicubic": Image.Resampling.BICUBIC}

# Conventions an earlier tree read, refused by name. legacy_minus1_1 fed
# [-1, 1] straight into the tower and resized bilinearly.
RETIRED_CONVENTIONS = {
    "legacy_minus1_1": "[-1, 1] straight into the tower and bilinear resizing, the "
                       "convention of an earlier tree",
}

# How a robot frame becomes its tile, consistent with NEO: NEO's smart_resize on
# the manifest's canvas (snap_grid), resampled by PIL's own BICUBIC
# (resize_frame). Every checkpoint's meta.json states it under TILE_META_KEY; a
# step trained under another rule (an earlier 180x320 view at 6x10 with a torch
# resample) is refused at a resume, an init_from and a serving load rather than
# run on other tiles.
TILE_META_KEY = "tile"
TILE_STAMP = {"rule": "neo_smart_resize", "resample": "pil_bicubic"}

# The field a checkpoint's meta.json states its convention in. A meta.json
# without it was written before the field existed, under the retired
# convention, and is refused rather than read as that convention.
CONVENTION_META_KEY = "pixel_convention"


def check_convention(convention, where: str) -> str:
    """Raise unless convention is the one this tree implements; return it."""
    if convention in RETIRED_CONVENTIONS:
        raise ValueError(
            f"{where} is {convention!r}, which is retired: "
            f"{RETIRED_CONVENTIONS[convention]}.\n"
            f"Expected {NEO_NATIVE!r}, the only pixel convention this tree builds.\n"
            "Serve an old checkpoint with the tree that trained it (dcd90a5)."
        )
    if convention not in PIXEL_CONVENTIONS:
        raise ValueError(
            f"{where} is {convention!r}.\n"
            f"Expected {NEO_NATIVE!r}.\n"
            "A run composes config/model/neo2b.yaml and gets neo_native."
        )
    return str(convention)


def tower_norm(convention: str) -> str:
    """The convention -> what VisionTower does at its entry."""
    return PIXEL_CONVENTIONS[check_convention(convention, "the pixel convention")][0]


def resize_filter(convention: str) -> str:
    """The convention -> the filter resize_frame uses."""
    return PIXEL_CONVENTIONS[check_convention(convention, "the pixel convention")][1]


def saved_convention(meta, where: str = "the checkpoint's meta.json") -> str:
    """A checkpoint's meta.json -> the convention its weights were fitted under.

    Raises for a meta.json without the field, or for no meta.json at all: it
    was written before the field existed, under the retired convention.
    """
    if not meta or CONVENTION_META_KEY not in meta:
        raise ValueError(
            f"{where} states no {CONVENTION_META_KEY}.\n"
            f"Expected {CONVENTION_META_KEY}: {NEO_NATIVE!r} in every meta.json this tree "
            "writes; one without it was written under the retired legacy_minus1_1.\n"
            "Serve or continue it with the tree that trained it (dcd90a5)."
        )
    return str(meta[CONVENTION_META_KEY])


def check_saved_convention(meta, built: str, where: str) -> None:
    """Raise unless the checkpoint at `where` was written under the convention built.

    A checkpoint's expert, tokenizers and, after a full finetune, its tower were
    fitted to the tower's outputs under one convention. Training from it,
    resuming it or serving it under another changes every image token by a
    cosine of about 0.9 with no error anywhere, so a disagreement stops here,
    and so does a meta.json that does not say.
    """
    saved = saved_convention(meta, where)
    if saved != built:
        raise ValueError(
            f"{where} states pixel convention {saved!r}; this model is built {built!r}.\n"
            "Expected the convention the checkpoint was written under: its weights were "
            "fitted to the tower's outputs under that one.\n"
            f"Name a checkpoint written under {built}."
        )


@dataclass(frozen=True)
class Tile:
    """Where one view lands: its grid in tokens and its size in pixels.

    grid_h, grid_w are token counts and height, width are grid times the
    token size, so the two can never disagree about the same view. A consumer
    that wants pixels takes height and width; one that wants token spend takes
    grid_h * grid_w.
    """

    grid_h: int
    grid_w: int
    height: int
    width: int


# A robot view's tile is the size NEO-2B's own smart_resize gives the frame:
# VLMTrainKit neo/data/utils.py:177-207
# and VLMEvalKit vlmeval/vlm/neo/utils.py:249-275, the Qwen2.5-VL rule. Its three
# numbers come from the manifest's canvas: factor is the token size, min_pixels
# the square of the short side and max_pixels the cell. On the shipped canvas
# (short side 256, token 32, cell 256x352) that is factor 32, min_pixels 65,536,
# NEO's own (VLMEvalKit utils.py:250, the harness's NEO_MIN), and max_pixels
# 90,112, the 88-token cell. A source that NEO's rule sizes past the cell on
# either side is refused by check_fits, never cropped.
# NEO refuses a frame whose long side is more than 200 times its short side.
NEO_MAX_ASPECT = 200


def smart_resize(height: int, width: int, *, factor: int, min_pixels: int,
                 max_pixels: int) -> tuple[int, int]:
    """NEO's smart_resize, operation for operation: (height, width) -> (h_bar, w_bar).

    Each side rounds to a multiple of factor with Python's round (ties to even,
    as NEO's round_by_factor); a product above max_pixels is scaled down by one
    factor for both sides and floored, one below min_pixels scaled up and
    ceiled. The same float operations in the same order as NEO's, so the two
    agree on every size, ties included.
    """
    if max(height, width) / min(height, width) > NEO_MAX_ASPECT:
        raise ValueError(
            f"frame is {height}x{width}, an aspect of "
            f"{max(height, width) / min(height, width)}.\n"
            f"Expected at most {NEO_MAX_ASPECT}, the aspect NEO's smart_resize accepts.\n"
            "Fix this view's shape in the manifest."
        )
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def snap_grid(shape: tuple[int, int], *, short_side: int, token: int,
              cell: tuple[int, int]) -> Tile:
    """An on-disk (height, width) -> the tile it is resized to: NEO's smart_resize.

    factor token, min_pixels short_side squared and max_pixels the cell's
    area, so a frame below NEO's minimum is upscaled as NEO upscales it (a
    180x320 view takes 192x352, 66 tokens) and one above the cell is brought
    under it. Every producer of a robot tile calls this: the manifest stage
    records its grid, the loader holds each decoded frame to it and the server
    sizes each live frame with it.
    """
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(
            f"frame size is {(height, width)}, expected both sides positive.\n"
            "An absent camera is stated by place_views, not by a zero shape.\n"
            "Fix this view's shape in the manifest."
        )
    cell_h, cell_w = int(cell[0]), int(cell[1])
    if short_side <= 0 or token <= 0 or cell_h <= 0 or cell_w <= 0:
        raise ValueError(
            f"canvas is short_side {short_side}, token {token} and cell {(cell_h, cell_w)}, "
            "expected every one positive.\nSet canvas.short_side, canvas.token and "
            "canvas.cell in the manifest."
        )
    tile_h, tile_w = smart_resize(height, width, factor=int(token),
                                  min_pixels=int(short_side) * int(short_side),
                                  max_pixels=cell_h * cell_w)
    return Tile(grid_h=tile_h // int(token), grid_w=tile_w // int(token),
                height=tile_h, width=tile_w)


def check_fits(tile: Tile, *, cell: tuple[int, int]) -> None:
    """Raise unless the tile fits the canvas, naming the view that does not.

    Called once per directory when the manifest is read, not once per frame:
    the size is a property of the directory, so a source that cannot fit is a
    build-time refusal rather than a failure part-way through a run. Cropping
    here would silently remove the field of view the native-aspect path exists
    to keep.
    """
    cell_h, cell_w = int(cell[0]), int(cell[1])
    if tile.height > cell_h or tile.width > cell_w:
        raise ValueError(
            f"tile is {tile.height}x{tile.width} pixels, the canvas is "
            f"{cell_h}x{cell_w}.\nExpected the tile to fit inside the canvas.\n"
            "Widen model.prefix.max_grid; this path will not crop the view."
        )


def resize_frame(frame: torch.Tensor, tile: Tile, *, resample: str) -> torch.Tensor:
    """One decoded frame [3, h, w] -> [3, tile.height, tile.width] in [-1, 1].

    resample is resize_filter(convention), which is bicubic: the frame's uint8
    levels go through PIL's own Image.resize with BICUBIC, the call NEO resizes
    with, and come back as those levels scaled to PIXEL_LOW..PIXEL_HIGH. The
    tile is snap_grid's, NEO's smart_resize: each side is resized to its own
    whole number of tokens, as NEO resizes it, and nothing is cropped.
    """
    if resample not in RESIZE_FILTERS:
        raise ValueError(
            f"resample is {resample!r}.\n"
            f"Expected one of {list(RESIZE_FILTERS)}.\n"
            "Pass resize_filter(model.vision.pixel_convention)."
        )
    if frame.dim() != 3 or int(frame.shape[0]) != 3:
        raise ValueError(
            f"frame is {tuple(frame.shape)}, expected [3, height, width].\n"
            "One decoded frame of one camera at one step.\n"
            "Give place_views the reader's frames, not a window of them."
        )
    if not frame.is_floating_point():
        raise TypeError(
            f"frame is {frame.dtype}, expected a floating type holding [0, 1].\n"
            "Decode to float in the reader; an integer frame scales wrong here."
        )
    span = (float(frame.amin()), float(frame.amax()))
    if span[0] < 0.0 or span[1] > 1.0:
        raise ValueError(
            f"frame spans {span[0]} to {span[1]}, expected [0, 1].\n"
            "A 0..255 frame arrives here as saturated white and trains on it.\n"
            "Normalise in the reader."
        )
    # PIL itself, not a float twin of it. PIL resamples width first, then height,
    # each pass with 22-bit fixed-point coefficients rounded to uint8. torch's
    # antialiased bicubic, one rounded pass per axis, stayed off by a level on
    # 44,467 and by two on 5 of 3,470,100,480 corpus values (the consistency
    # harness); PIL's own call is off by none, at +0.5 ms on a 180x320
    # frame and -0.5 ms on a 264x352 one (one thread).
    # A frame already at its tile passes through on its levels, so OXE and
    # LIBERO frames, 256x256 on disk, are unchanged.
    levels = torch.floor(frame.detach().cpu() * 255.0 + 0.5).clamp(0.0, 255.0).to(torch.uint8)
    if tuple(levels.shape[-2:]) != (tile.height, tile.width):
        picture = Image.fromarray(levels.permute(1, 2, 0).contiguous().numpy())
        picture = picture.resize((tile.width, tile.height), PIL_FILTERS[resample])
        levels = torch.from_numpy(numpy.array(picture)).permute(2, 0, 1)
    scaled = levels.to(frame.dtype) / 255.0 * (PIXEL_HIGH - PIXEL_LOW) + PIXEL_LOW
    return scaled.to(frame.device)


def check_snaps_to(frame: torch.Tensor, tile: Tile, *, key: str,
                   short_side: int, token: int, cell: tuple[int, int]) -> None:
    """Raise unless the frame decoded is one the tile was snapped from.

    The tile comes from the grid the manifest recorded, and that grid was
    snapped from a size a metadata file declared. Nothing else compares the two,
    so a directory whose frames were re-encoded to another size is resized into
    a tile of the declared shape for a whole run, image_shapes reports that
    shape as a measurement, and no stage raises.
    """
    found = snap_grid(
        (int(frame.shape[-2]), int(frame.shape[-1])), short_side=short_side, token=token,
        cell=cell,
    )
    if (found.grid_h, found.grid_w) != (tile.grid_h, tile.grid_w):
        raise ValueError(
            f"camera {key!r} decoded a {int(frame.shape[-2])}x{int(frame.shape[-1])} "
            f"frame, which snaps to grid ({found.grid_h}, {found.grid_w}).\n"
            f"Expected ({tile.grid_h}, {tile.grid_w}), the grid the manifest records.\n"
            "Rebuild the manifest against the frames on disk."
        )


def place_views(
    frames: dict[str, torch.Tensor],
    tiles: dict[str, Tile],
    *,
    slots: dict[str, int],
    n_views: int,
    cell: tuple[int, int],
    short_side: int,
    token: int,
    resample: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The frames of one step -> pixels, image_shapes and view_valid.

    pixels        [n_views, 3, cell_h, cell_w] float32, each tile at the
                  top-left of its slot's canvas and zeros elsewhere
    image_shapes  [n_views, 2] long, the tile size, and exactly (0, 0) for a
                  slot no camera fills
    view_valid    [n_views] bool, true where image_shapes is non-zero

    The two masks agree by construction because one function writes both. A
    slot with no camera keeps its canvas and its prefix columns, both
    unattendable, so every sample has the same prefix width.

    cell, short_side and token are the canvas the tiles were snapped against,
    and they are here so that each decoded frame is measured against the tile it is
    about to be written into rather than trusted to match it. resample is
    resize_filter(model.vision.pixel_convention), stated by every caller so the
    training loader and the serving canvas cannot resize one frame two ways.
    """
    if set(frames) != set(tiles) or set(frames) != set(slots):
        raise ValueError(
            f"frames name {sorted(frames)}, tiles {sorted(tiles)} and slots "
            f"{sorted(slots)}.\nExpected one entry per camera in all three.\n"
            "A camera missing from one of them would be silently absent."
        )
    cell_h, cell_w = int(cell[0]), int(cell[1])
    pixels = torch.zeros((n_views, 3, cell_h, cell_w), dtype=torch.float32)
    image_shapes = torch.zeros((n_views, 2), dtype=torch.long)
    view_valid = torch.zeros(n_views, dtype=torch.bool)
    for key in sorted(frames):
        slot = int(slots[key])
        if not 0 <= slot < n_views:
            raise ValueError(
                f"camera {key!r} asks for slot {slot}, the layout has "
                f"{n_views}.\nExpected a slot in 0..{n_views - 1}.\n"
                "Raise model.prefix.n_views or move the camera in the manifest."
            )
        if bool(view_valid[slot]):
            raise ValueError(
                f"camera {key!r} and another both claim slot {slot}.\n"
                "Expected one camera per slot: the second would overwrite the "
                "first.\nGive them different slots in the manifest."
            )
        tile = tiles[key]
        # resize_frame owns the frame's own shape and range checks, so it runs
        # first and the size check below reads a frame it has already accepted.
        resized = resize_frame(frames[key], tile, resample=resample)
        check_snaps_to(frames[key], tile, key=key, short_side=short_side, token=token,
                       cell=cell)
        pixels[slot, :, : tile.height, : tile.width] = resized
        image_shapes[slot, 0] = tile.height
        image_shapes[slot, 1] = tile.width
        view_valid[slot] = True
    return pixels, image_shapes, view_valid


def check_absence_agrees(image_shapes: torch.Tensor, view_valid: torch.Tensor) -> None:
    """Raise unless a zero shape and a false flag mark exactly the same slots.

    The check a test asserts per source. It is cheap and it is here because the
    previous loader padded the camera axis before the shapes were written, so
    every slot reported a real size and two of three cameras of a single-camera
    source were a black tile the encoder still ran the vision tower over.
    """
    if tuple(image_shapes.shape[:-1]) != tuple(view_valid.shape):
        raise ValueError(
            f"image_shapes is {tuple(image_shapes.shape)} and view_valid is "
            f"{tuple(view_valid.shape)}.\nExpected the same slot axes and a "
            "trailing 2.\nBoth come out of one place_views call."
        )
    if int(image_shapes.shape[-1]) != 2:
        raise ValueError(
            f"image_shapes ends in {int(image_shapes.shape[-1])}, expected 2.\n"
            "One height and one width per camera slot."
        )
    if bool((image_shapes < 0).any()):
        raise ValueError(
            f"image_shapes holds {int(image_shapes.min())}, expected zero or more.\n"
            "Zero says absent and a positive number says present; there is no third."
        )
    zero_side = image_shapes == 0
    absent = zero_side.all(-1)
    if not torch.equal(absent, zero_side.any(-1)):
        raise ValueError(
            "an image shape has one zero side and one real one.\n"
            "Expected both sides zero for an absent camera, both positive for a "
            "present one.\nFix the producer of image_shapes."
        )
    if not torch.equal(absent, ~view_valid.bool()):
        raise ValueError(
            f"{int((absent != ~view_valid.bool()).sum())} slots disagree about "
            "absence.\nExpected a zero image_shapes row exactly where view_valid "
            "is false.\nEmit both from one place_views call."
        )
