"""The camera calibration channel: six numbers per view, read and checked.

A per-view block states a camera's calibration beside its picture: fx, fy, cx,
cy and the raster they were measured against. The loader emits those six numbers
per camera slot, at the resolution they were measured against, and the model's
calibration columns (intrinsics_mode tokens) read them through
PrefixTokenizers.compress_calibration, which takes each focal length and
principal point as a ratio of its own side, so the record means the same thing
whatever tile the picture was resized to.

The module keeps its name from the ray maps it used to build. Those, and the
modes that read them (patch, tower), are removed; so are none and text.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .conditioning import mark_unknown

# How the calibration reaches the model, as model.prefix.intrinsics_mode names
# it. tokens projects the six calibration numbers and the six of the camera's
# rotation into twelve prefix columns of their own, one projection per column,
# right after each <cam_k>. It is the one mode, and the corpus stamp, the prompt
# template's geometry key and the
# checkpoint's run config all name it.
INTRINSICS_MODES = ("tokens",)

# The modes this tree built before and refuses by name now: none sent nothing,
# patch added a projected ray to each image token, text wrote the twelve
# numbers as words, tower embedded a ray map beside the picture. A checkpoint or
# a corpus stamped for one of them is not one this tree can read.
REMOVED_INTRINSICS_MODES = ("none", "patch", "text", "tower")


def check_intrinsics_mode(mode, where: str) -> str:
    """Return mode when it is the one mode; refuse a removed one by name."""
    mode = str(mode)
    if mode in REMOVED_INTRINSICS_MODES:
        raise ValueError(
            f"{where} is {mode!r}, an intrinsics mode this tree no longer builds.\n"
            f"Expected {INTRINSICS_MODES[0]!r}: the modes {list(REMOVED_INTRINSICS_MODES)} "
            "were removed.\n"
            "Set model.prefix.intrinsics_mode=tokens and use a corpus stamped for it."
        )
    if mode not in INTRINSICS_MODES:
        raise ValueError(
            f"{where} is {mode!r}.\n"
            f"Expected one of {INTRINSICS_MODES}.\n"
            "Set it to the mode this run's model was pretrained under."
        )
    return mode


# Calibrations this channel will use. A self-calibrated intrinsic recovered
# from the recording itself is excluded: it is a fit, and a fit that is wrong
# is indistinguishable here from one that is right.
PROVENANCE_ALLOWED = ("downloaded", "vendor")

# What the manifest may say about where a view's calibration comes from. The
# words are the ones extrinsics already uses, because the two answer the same
# question: one calibration for the directory, one per episode, or none.
INTRINSICS_KINDS = ("episode", "constant", "none")

# The columns of a per-episode calibration array, in order. tools/build_cache.py
# writes it and this module reads it, so the two cannot drift into two orders.
CALIBRATION_COLUMNS = ("fx", "fy", "cx", "cy", "width", "height")


@dataclass(frozen=True)
class Intrinsics:
    """A pinhole calibration as it was recorded, at its own resolution.

    width and height are the pixel size fx, fy, cx and cy were measured
    against. They are part of the record because the model reads each focal
    length and principal point as a ratio of its own side
    (PrefixTokenizers.compress_calibration), and a calibration stored against
    some other resize cannot be read that way without knowing which one.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    provenance: str


def check_provenance(provenance: str) -> None:
    """Raise unless the calibration came from a source in PROVENANCE_ALLOWED.

    Takes the word rather than a calibration because a per-episode view states
    it once for the whole array. Called when the manifest is read, so a corpus
    built with an excluded calibration fails before it trains rather than
    emitting numbers that look exactly like measured ones.
    """
    if provenance not in PROVENANCE_ALLOWED:
        raise ValueError(
            f"calibration provenance is {provenance!r}, expected one "
            f"of {PROVENANCE_ALLOWED}.\n"
            "Record the view as having no calibration rather than fitting one here."
        )


def calibration_from_row(row: Sequence[float], provenance: str) -> Intrinsics | None:
    """One row of a per-episode calibration array -> a calibration, or None.

    A row that is unknown throughout is an episode whose camera nobody
    calibrated, and None is what says so. A row unknown in part is not a
    weaker calibration but an unreadable one, so it raises: filling the missing
    number would put a measured-looking value where no measurement is.
    """
    values = [float(value) for value in row]
    if len(values) != len(CALIBRATION_COLUMNS):
        raise ValueError(
            f"calibration row holds {len(values)} numbers, expected "
            f"{len(CALIBRATION_COLUMNS)}.\nExpected the columns "
            f"{CALIBRATION_COLUMNS}.\nRebuild the intrinsics part of the cache."
        )
    unknown = [math.isnan(value) for value in values]
    if all(unknown):
        return None
    if any(unknown):
        named = [name for name, gone in zip(CALIBRATION_COLUMNS, unknown, strict=True) if gone]
        raise ValueError(
            f"calibration row states {named} as unknown and the rest as measured.\n"
            "Expected all six columns or none of them.\n"
            "Rebuild the intrinsics part of the cache."
        )
    return Intrinsics(
        fx=values[0], fy=values[1], cx=values[2], cy=values[3],
        width=int(values[4]), height=int(values[5]), provenance=provenance,
    )


def frame_band(shape: tuple[int, int]) -> tuple[float, float]:
    """The widths over heights a calibration may have to describe this frame.

    shape is the stored frame as the manifest records it, height first. Both
    sides are whole pixels, so a resize of the calibrated picture lands within
    half a pixel of the stored one on each axis and anything wider is another
    picture. This is the question every stage that knows the stored size asks.
    """
    height, width = int(shape[0]), int(shape[1])
    if width <= 0 or height <= 0:
        raise ValueError(
            f"frame is {width}x{height} pixels, expected both sides positive.\n"
            "A frame with no size has no aspect to compare a calibration with.\n"
            "Record the view's stored shape."
        )
    return ((width - 0.5) / (height + 0.5), (width + 0.5) / (height - 0.5))


def calibration_for_views(
    intrinsics: dict[str, Intrinsics | None],
    *,
    slots: dict[str, int],
    n_views: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One step's calibrations -> the six numbers per slot, and their validity.

    values [n_views, 6] float32, the columns CALIBRATION_COLUMNS names and in
    that order, and valid [n_views] bool. A view nobody calibrated, and every
    slot no camera fills, is unknown: the flag is false and the row holds the
    marker mark_unknown writes.

    The numbers stay at the resolution they were measured against: the model
    reads them as ratios of their own raster, which no resize of the picture
    moves.
    """
    if set(intrinsics) != set(slots):
        raise ValueError(
            f"intrinsics name {sorted(intrinsics)} and slots {sorted(slots)}.\n"
            "Expected one entry per camera in both, null where a view has no "
            "calibration.\n"
            "An omitted camera states nothing, and says nothing about why."
        )
    values = torch.zeros((n_views, len(CALIBRATION_COLUMNS)), dtype=torch.float32)
    valid = torch.zeros(n_views, dtype=torch.bool)
    for key in sorted(intrinsics):
        calibration = intrinsics[key]
        if calibration is None:
            continue
        slot = int(slots[key])
        if not 0 <= slot < n_views:
            raise ValueError(
                f"camera {key!r} asks for slot {slot}, the layout has "
                f"{n_views}.\nExpected a slot in 0..{n_views - 1}.\n"
                "Raise model.prefix.n_views or move the camera in the manifest."
            )
        if bool(valid[slot]):
            raise ValueError(
                f"camera {key!r} and another both claim slot {slot}.\n"
                "Expected one camera per slot: the second would overwrite the "
                "first.\nGive them different slots in the manifest."
            )
        values[slot] = torch.tensor(
            [calibration.fx, calibration.fy, calibration.cx, calibration.cy,
             float(calibration.width), float(calibration.height)],
            dtype=torch.float32,
        )
        valid[slot] = True
    return mark_unknown(values, valid), valid
