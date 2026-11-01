"""Stage 7: per-episode camera calibration, from the source's own files.

One record per directory, episode and camera, holding the pinhole numbers and
the pixel size they were measured against, which is what rays.Intrinsics needs
to rescale onto the tile the loader produces. A camera the source did not
calibrate gets no record: absence is expressed by absence, and the ray channel
says unknown for a view no record names.

--source picks the reader and --calib repeats when a source states its
calibration in more than one tree. Numbers whose principal point or field of
view says they were measured against another frame than the one they name are
refused, and the refusals are counted by reason in the header.

Usage:
    python tools/build_intrinsics.py --root /corpus --source agibot \
        --calib calib/agibot_intrinsics --dirs-file dirs.txt \
        --layouts config/layouts/gd7.json --out prep/intrinsics_agibot.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import struct
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import layouts as layout_table
import numpy as np
import stage
from scan_source import directory_inputs, read_dir_list

from neo_vla.data import rays

logger = logging.getLogger("build_intrinsics")

REPO = Path(__file__).resolve().parent.parent

# What a reader writes when its tree is the recording's own profile rather than
# a document with a provenance field of its own. rays.PROVENANCE_ALLOWED is what
# decides whether it may be read back; this tool never invents a word outside it.
PROVENANCE = "vendor"

# How far the principal point may sit from the centre of the frame the record
# names, as a share of each side. This is what catches a calibration measured
# against some other frame size: its centre lands where that frame's centre
# was. The vendors here sit within 2.1 percent, and the calibrations measured
# against a different frame sit at 25 to 49 percent.
PRINCIPAL_POINT_MARGIN = 0.10

# Angles a camera in this corpus sees. The band has to hold DROID's 101 degree
# ZED and RH20T's 42.8 degree vertical, so it is wide; it is here to catch a
# focal off by a factor, not to grade a lens.
FIELD_OF_VIEW_DEGREES = {"horizontal": (40.0, 140.0), "vertical": (30.0, 120.0)}

# Projections the pinhole fields of a record can express. A fisheye model names
# a mapping these four numbers do not describe, so a record carrying one is
# refused rather than read as a pinhole later.
PINHOLE_DISTORTION = ("none", "brown_conrady", "inverse_brown_conrady",
                      "modified_brown_conrady", "plumb_bob", "rational_polynomial")

# The order every record's distortion_coeffs are written in. A source that
# states a different one is refused rather than reordered, so the field means
# one thing across sources and a reader needs no per-source table.
DISTORTION_ORDER = "k1,k2,p1,p2,k3"

# AgiBot's sidecar, one record per episode.
AGIBOT_SIDECAR = "intrinsic_sidecar.jsonl"
AGIBOT_KEY = "raw_file_name"
AGIBOT_CAMERAS = "cameras"
AGIBOT_SLOT_ORDER = "view_slot_order"
AGIBOT_SIZE = "stored_hw"
AGIBOT_STATUS = "status"

# Where the sidecar states the order it wrote its distortion coefficients in.
AGIBOT_COEFF_ORDER = "distortion_order"

# Which layout view each source's own camera name belongs to. Checked against
# the layout file at run time, so a rename on either side raises here instead
# of writing a record against a camera the manifest does not have.
CAMERA_VIEWS = {
    "agibot": {
        "head": "observation.images.head",
        "hand_left": "observation.images.hand_left",
        "hand_right": "observation.images.hand_right",
    },
    # Only the per-camera RH20T tree, whose episode is one camera. The grid and
    # joined trees hold several cameras of one scene per episode and name the
    # stored column beside each camera's serial, so their columns come from the
    # episode rather than from this table; rh20t_cameras reads them.
    "rh20t": {"cam": "observation.images.cam"},
    "droid": {
        "exterior": "observation.images.image",
        "wrist": "observation.images.wrist_image",
    },
    # A sidecar of the remaining OXE trees has no vendor camera name to
    # translate, so it has to name the stored column itself. Nobody established
    # which raw stream the converter wrote into which column, and a table
    # guessing it would calibrate one camera with another's numbers.
    "oxe": {
        "observation.images.image": "observation.images.image",
        "observation.images.wrist_image": "observation.images.wrist_image",
    },
    # The three-camera DROID tree states each profile under the stored column
    # it belongs to, so there is no vendor camera name to translate here either.
    "droid3cam": {
        "observation.images.exterior_1": "observation.images.exterior_1",
        "observation.images.exterior_2": "observation.images.exterior_2",
        "observation.images.wrist": "observation.images.wrist",
    },
    # RoboTwin names its cameras in the recording and the RoboTwin converter
    # takes three of the four into the corpus, so this is that table read the
    # same way round: the vendor's name for a camera, then the column it fills.
    "robotwin2": {
        "head_camera": "observation.images.head",
        "left_camera": "observation.images.left_wrist",
        "right_camera": "observation.images.right_wrist",
    },
}

# Which layout source each reader's directories belong to. The one-camera DROID
# is one family inside the oxe layout rather than a source of its own, so the
# reader's name and the layout's source are two words and are kept apart here.
# The three-camera DROID tree is its own source: config/layouts/oxe3src_gd6.json
# and gd7.json assign oxe/droid3cam* to source droid3cam.
LAYOUT_SOURCE = {"agibot": "agibot", "rh20t": "rh20t", "droid": "oxe", "oxe": "oxe",
                 "droid3cam": "droid3cam", "robotwin2": "robotwin2"}

# Where the three-camera DROID tree states each camera's profile, in the
# episode metadata the DROID converter wrote it into. The profile is at the
# raster it was measured against, which is what the model's calibration columns
# read it as a ratio of (PrefixTokenizers.compress_calibration).
DROID3CAM_KEY = "droid_intrinsics"

# RoboTwin's sidecar, one record per episode, written from the recordings by
# the RoboTwin calibration stager. The join key is the capture and the index the
# converter wrote beside every episode, and the frame count both sides state
# is what confirms it.
ROBOTWIN_SIDECAR = "robotwin_calib_sidecar.jsonl"
ROBOTWIN_ARCHIVE = "archive"
ROBOTWIN_EPISODE = "episode"
ROBOTWIN_CAMERAS = "cameras"
ROBOTWIN_FRAMES = "frames"

# What the corpus calls the same two numbers, in the metadata the conversion
# wrote. length is the rows the directory holds for that episode.
ROBOTWIN_SOURCE_FIELD = "robotwin_source"
ROBOTWIN_EPISODE_FIELD = "robotwin_episode"
ROBOTWIN_LENGTH_FIELD = "length"

# RH20T's calibration tree: one directory per config, each holding one
# directory per calibration scene.
RH20T_CALIB = "calib"
RH20T_INTRINSICS = "intrinsics.npy"
RH20T_DEVICES = "devices.npy"
RH20T_IMAGES = "imgs"
RH20T_CONFIG_FIELD = "rh20t_config"
RH20T_SERIAL_FIELD = "rh20t_camera_serial"

# Where a multi-camera RH20T episode lists the cameras it holds, and the fields
# one entry names its stored column, its slot and its serial in. The list is
# absent from the per-camera tree, which is how the two shapes are told apart.
RH20T_VIEWS_FIELD = "rh20t_views"
RH20T_VIEW_KEY = "key"
RH20T_VIEW_SLOT = "slot"
RH20T_VIEW_SERIAL = "serial"

# What a camera of a multi-camera RH20T episode is called in its record, its
# slot appended. This is the name tools/build_rh20t_grid.py gives the same
# camera, so a record from either writer reads the same.
RH20T_SLOT_CAMERA = "cam_"

# DROID's calibration, two files of one tree: the sidecar that resolved which
# camera each episode stored, and the annotation file the numbers themselves
# come from. The second holds the profiles the ZED recordings carry, extracted
# from the recordings rather than fitted to them.
DROID_SIDECAR = "camera_calib_sidecar.jsonl"
DROID_ANNOTATIONS = "karlp/intrinsics.json"
DROID_REPO_FIELD = "lerobot_repo"
DROID_EPISODE_FIELD = "episode_id"
DROID_ROWS_FIELD = "n_frames"
DROID_MATRIX = "cameraMatrix"
DROID_COEFFS = "distCoeffs"

# Where the sidecar names each camera's serial. The exterior view is whichever
# of the two the converter stored, resolved per episode and named at the top
# level; the wrist is one entry of the serial list, and DROID records one.
DROID_SERIAL_AT = {"exterior": ("stored_cam_serial",),
                   "wrist": ("camera_serials", "wrist")}

# The calibration sidecars the remaining OXE trees write, and the fields each
# shape states its view, its numbers and its provenance in. The provenance word
# is compared with rays.PROVENANCE_ALLOWED as written: a table from a fitter's
# name to "vendor" would launder the estimate the allow list exists to refuse.
OXE_SHAPES = {
    "handeye_calib_sidecar.jsonl": {
        "view": "stored_view_key", "block": "intrinsics",
        "provenance": ("provenance", "kind"),
    },
    "camera_calib_sidecar.jsonl": {
        "view": "stored_view_rlds_key", "block": "intrinsics_native",
        "provenance": ("intrinsics_source",),
    },
}

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class Calibration:
    """One camera's numbers as a reader states them, before they are checked."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion_model: str | None
    distortion_coeffs: tuple[float, ...] | None
    distortion_order: str | None


def field_of_view(focal: float, extent: int) -> float:
    """The angle, in degrees, a pinhole of this focal sees across this many pixels."""
    return math.degrees(2.0 * math.atan(extent / (2.0 * focal)))


def refusal(calibration: Calibration) -> str | None:
    """Why this calibration cannot be stated, or None when it can.

    A refused camera emits no record, so every reason here removes a camera
    from the corpus rather than degrading one. The names are what the header
    counts, so reading one run's counts says which check bit.

    There is no check that the two focals are equal. A frame stored at an
    aspect its optics do not have has a real focal per axis, and AgiBot's are
    3 to 1 apart on purpose; a check calling that a defect would refuse the
    corpus's largest calibrated source. What is checked instead is whether the
    numbers were measured against the frame the record names, which is what a
    focal copied from another resize actually gets wrong.
    """
    if calibration.width <= 0 or calibration.height <= 0:
        return "size_not_positive"
    if not (calibration.fx > 0.0 and calibration.fy > 0.0):
        return "focal_not_positive"
    if not (math.isfinite(calibration.cx) and math.isfinite(calibration.cy)):
        return "principal_point_not_finite"
    if not 0.0 <= calibration.cx <= calibration.width - 1:
        return "principal_point_outside_frame"
    if not 0.0 <= calibration.cy <= calibration.height - 1:
        return "principal_point_outside_frame"
    for value, extent in ((calibration.cx, calibration.width),
                          (calibration.cy, calibration.height)):
        if abs(value - (extent - 1) / 2.0) > PRINCIPAL_POINT_MARGIN * extent:
            return "principal_point_off_centre"
    for axis, focal, extent in (("horizontal", calibration.fx, calibration.width),
                                ("vertical", calibration.fy, calibration.height)):
        low, high = FIELD_OF_VIEW_DEGREES[axis]
        if not low <= field_of_view(focal, extent) <= high:
            return "field_of_view_out_of_range"
    return distortion_refusal(calibration.distortion_model, calibration.distortion_coeffs,
                              calibration.distortion_order)


def distortion_refusal(
    model: str | None, coefficients: tuple[float, ...] | None, order: str | None,
) -> str | None:
    """Why a distortion statement cannot be carried, or None when it can."""
    if model is None:
        # The source records no distortion. That is not a claim that there is
        # none, so the record carries neither field and nothing reads one.
        return None if coefficients is None else "distortion_coeffs_without_model"
    if model not in PINHOLE_DISTORTION:
        return "distortion_model_not_pinhole"
    if coefficients is None:
        return "distortion_model_without_coeffs"
    if not all(math.isfinite(value) for value in coefficients):
        return "distortion_coeffs_not_finite"
    if model == "none" and any(value != 0.0 for value in coefficients):
        return "distortion_none_with_coeffs"
    # A source that names its own order is held to this file's one. Reordering
    # it here would make the record depend on a table nothing downstream reads.
    if order is not None and order != DISTORTION_ORDER:
        return "distortion_order_unexpected"
    return None


def distortion_name(raw: Any) -> str:
    """A source's spelling of a distortion model, in the one this tool compares.

    The vendor libraries print the enum qualified and spaced in several ways;
    folding case and separators once here is cheaper than an allow list that
    has to carry every spelling of the same model.
    """
    folded = "".join(
        character if character.isalnum() else "_" for character in str(raw).strip().lower()
    )
    while "__" in folded:
        folded = folded.replace("__", "_")
    folded = folded.strip("_")
    prefix = "distortion_"
    return folded[len(prefix):] if folded.startswith(prefix) else folded


def as_record(
    *, dataset: str, episode: int, camera: str, view: str, slot: int,
    calibration: Calibration, provenance: str | None = None,
) -> dict[str, Any]:
    """One checked calibration -> the record written for it.

    provenance is the word the source stated, for a reader whose source states
    one; a reader over a tree that ships the recording's own profile passes
    nothing and PROVENANCE stands. Either way the word goes through the loader's
    own allow list here, so one it would refuse raises rather than reaching the
    file.
    """
    stated = PROVENANCE if provenance is None else provenance
    rays.check_provenance(stated)
    record = {
        "dataset": dataset,
        "episode": int(episode),
        "camera": camera,
        "view": view,
        "slot": int(slot),
        "fx": float(calibration.fx),
        "fy": float(calibration.fy),
        "cx": float(calibration.cx),
        "cy": float(calibration.cy),
        "width": int(calibration.width),
        "height": int(calibration.height),
        "provenance": stated,
    }
    if calibration.distortion_model is not None:
        record["distortion_model"] = calibration.distortion_model
        record["distortion_coeffs"] = [float(v) for v in calibration.distortion_coeffs]
        record["distortion_order"] = DISTORTION_ORDER
    return record


def layout_slots(document: dict[str, Any], directory: str, source: str) -> dict[str, int]:
    """This directory's stored camera columns -> the slot each one fills.

    The layout is what every record's slot is taken from, because it is what
    the manifest stage checks a record's slot against. A directory holds as
    many columns as its layout entry names, which is one for some RH20T
    directories and three for others.
    """
    layout = layout_table.resolve(document, directory)
    expected = LAYOUT_SOURCE[source]
    if layout.source != expected:
        raise ValueError(
            f"directory {directory} has layout source {layout.source!r}\n"
            f"expected {expected!r}, the layout source --source {source} reads\n"
            f"pass only that source's directories, or change --source"
        )
    return {view["key"]: int(view["slot"]) for view in layout.views}


def view_slots(document: dict[str, Any], directory: str, source: str) -> dict[str, tuple[str, int]]:
    """This directory's cameras -> the layout view and slot each one fills."""
    by_key = layout_slots(document, directory, source)
    mapping = {}
    for camera, key in sorted(CAMERA_VIEWS[source].items()):
        if key not in by_key:
            raise KeyError(
                f"{directory} has no view {key!r} for camera {camera!r}\n"
                f"expected one of {sorted(by_key)}\n"
                f"fix CAMERA_VIEWS in this tool or the layout file's views"
            )
        mapping[camera] = (key, by_key[key])
    return mapping


def episode_metadata(directory: Path) -> list[dict[str, Any]]:
    """One directory's meta/episodes.jsonl, in file order."""
    with open(directory / "meta" / "episodes.jsonl", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _field(record: dict[str, Any], name: str, where: str) -> Any:
    if name not in record:
        raise KeyError(
            f"{where} record has no {name!r}\n"
            f"expected it among {sorted(record)[:8]}\n"
            f"check the file this run reads, or the conversion that wrote it"
        )
    return record[name]


def agibot_calibration(block: dict[str, Any], stored_hw: list[int], where: str) -> Calibration:
    """One camera's sidecar block -> the calibration this tool states for it.

    The numbers are the vendor's, copied. The sidecar's focals differ by axis
    because the delivered 640x480 frame is the sensor's 16:9 field resampled
    onto a 4:3 raster, so fx and fy describe that raster correctly and moving
    either would break them.
    """
    for name in ("fx", "fy", "cx", "cy"):
        if name not in block:
            raise KeyError(
                f"{where} camera block has no {name!r}\n"
                f"expected fx, fy, cx and cy among {sorted(block)[:8]}\n"
                f"check the sidecar this run reads"
            )
    height, width = int(stored_hw[0]), int(stored_hw[1])
    coefficients = block.get("distortion_coeffs")
    model = block.get("distortion_model")
    return Calibration(
        fx=float(block["fx"]), fy=float(block["fy"]),
        cx=float(block["cx"]), cy=float(block["cy"]),
        width=width, height=height,
        distortion_model=None if model is None else distortion_name(model),
        distortion_coeffs=None if coefficients is None else tuple(
            float(value) for value in coefficients
        ),
        distortion_order=block.get(AGIBOT_COEFF_ORDER),
    )


def agibot_wanted(
    root: Path, directories: list[Path], document: dict[str, Any],
) -> dict[str, list[tuple[str, int, dict[str, tuple[str, int]]]]]:
    """The episodes this run covers, keyed by the sidecar's own join key."""
    wanted: dict[str, list[tuple[str, int, dict[str, tuple[str, int]]]]] = {}
    for directory in directories:
        name = stage.relative(root, directory)
        slots = view_slots(document, name, "agibot")
        for episode in episode_metadata(directory):
            key = str(_field(episode, AGIBOT_KEY, name))
            index = int(_field(episode, "episode_index", name))
            wanted.setdefault(key, []).append((name, index, slots))
    return wanted


def sidecars_named(calib: list[Path], names: tuple[str, ...], expected: str) -> list[Path]:
    """Every file of these names under the calibration trees this run reads.

    Nothing is found by name alone: a tree the caller did not name is not
    searched, which is what keeps one source's sidecar out of another's reader
    when the two spell their file the same way.
    """
    files = sorted({path for tree in calib for name in names for path in tree.rglob(name)})
    if not files:
        raise FileNotFoundError(
            f"no {' or '.join(names)} under {[str(tree) for tree in calib]}\n"
            f"expected {expected}\n"
            f"pass --calib each directory holding them"
        )
    return files


def agibot_records(files: list[Path], wanted: set[str]) -> Iterator[dict[str, Any]]:
    """Every sidecar record this run needs, streamed rather than held."""
    for path in files:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if str(record.get(AGIBOT_KEY)) in wanted:
                    yield record


def read_agibot(
    root: Path, calib: list[Path], directories: list[Path], document: dict[str, Any],
) -> tuple[list[dict[str, Any]], Counter, dict[str, Path]]:
    """AgiBot's sidecar, joined to our directories by the key it carries."""
    wanted = agibot_wanted(root, directories, document)
    files = sidecars_named(calib, (AGIBOT_SIDECAR,),
                           "AgiBot's per-task calibration sidecars")
    counts: Counter = Counter()
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sidecar in agibot_records(files, set(wanted)):
        key = str(sidecar[AGIBOT_KEY])
        if key in seen:
            raise ValueError(
                f"two sidecar records name episode {key}\n"
                f"expected one calibration per episode; nothing here says which\n"
                f"remove the duplicate from the calibration tree"
            )
        seen.add(key)
        status = sidecar.get(AGIBOT_STATUS)
        if status != "ok":
            counts[f"episode_status_{status}"] += len(wanted[key])
            continue
        order = [str(name) for name in _field(sidecar, AGIBOT_SLOT_ORDER, key)]
        blocks = _field(sidecar, AGIBOT_CAMERAS, key)
        if not isinstance(blocks, dict):
            raise TypeError(
                f"sidecar {key} has {AGIBOT_CAMERAS!r} as {type(blocks).__name__}\n"
                f"expected an object keyed by the names in {AGIBOT_SLOT_ORDER!r}\n"
                f"point AGIBOT_CAMERAS at the key the sidecar nests them under"
            )
        stored_hw = list(_field(sidecar, AGIBOT_SIZE, key))
        for dataset, episode, slots in wanted[key]:
            for position, camera in enumerate(order):
                if camera not in slots:
                    counts["camera_not_in_layout"] += 1
                    continue
                view, slot = slots[camera]
                if position != slot:
                    # The sidecar states its own slot order. Two orders for one
                    # camera means one of them names a different picture.
                    counts["slot_disagrees_with_layout"] += 1
                    continue
                block = blocks.get(camera)
                if block is None:
                    counts["camera_absent"] += 1
                    continue
                where = f"{dataset} episode {episode} camera {camera}"
                calibration = agibot_calibration(block, stored_hw, where)
                reason = refusal(calibration)
                if reason is not None:
                    counts[reason] += 1
                    continue
                records.append(as_record(
                    dataset=dataset, episode=episode, camera=camera, view=view,
                    slot=slot, calibration=calibration,
                ))
    counts["episode_absent_from_sidecar"] += sum(
        len(wanted[key]) for key in set(wanted) - seen
    )
    inputs = {stage.relative(root, path): path for path in files}
    return records, counts, inputs


def png_size(path: Path) -> tuple[int, int]:
    """A png's (height, width), from its header rather than its pixels."""
    with open(path, "rb") as handle:
        head = handle.read(24)
    if len(head) < 24 or head[:8] != PNG_MAGIC or head[12:16] != b"IHDR":
        raise ValueError(
            f"{path} does not start with a png header\n"
            f"expected the 24-byte signature and IHDR chunk\n"
            f"point --calib at the calibration tree holding the scene images"
        )
    width, height = struct.unpack(">II", head[16:24])
    return int(height), int(width)


def scene_size(scene: Path) -> tuple[int, int]:
    """The pixel size a scene's calibration was measured against.

    Read from the scene's own calibration images rather than declared, because
    it is the number every record states and a wrong one rescales every ray. A
    scene whose images disagree about it has no single answer.
    """
    images = sorted((scene / RH20T_IMAGES).glob("*.png"))
    if not images:
        raise FileNotFoundError(
            f"{scene} has no {RH20T_IMAGES}/*.png\n"
            f"expected the calibration images the intrinsics were fitted on\n"
            f"pass a calibration tree that carries them"
        )
    sizes = {png_size(path) for path in images}
    if len(sizes) != 1:
        raise ValueError(
            f"{scene} calibration images are {sorted(sizes)}\n"
            f"expected one pixel size for the scene\n"
            f"separate the sizes before reading this scene"
        )
    return sizes.pop()


def _load_object_array(path: Path) -> Any:
    # These are numpy object arrays of python dicts, so reading them runs the
    # pickle in the file. The calibration tree is read-only input to this
    # pipeline and is the tree the frames themselves came from.
    return np.load(path, allow_pickle=True)


def rh20t_scene(scene: Path) -> tuple[dict[str, Calibration], int]:
    """One calibration scene -> a calibration per device, and how many lack one."""
    height, width = scene_size(scene)
    matrices = _load_object_array(scene / RH20T_INTRINSICS).item()
    devices = [str(serial) for serial in _load_object_array(scene / RH20T_DEVICES).tolist()]
    found = {}
    for serial in devices:
        if serial not in matrices:
            continue
        matrix = np.asarray(matrices[serial], dtype=np.float64)
        if matrix.shape[0] < 3 or matrix.shape[1] < 3:
            raise ValueError(
                f"{scene} intrinsics for {serial} are {tuple(matrix.shape)}\n"
                f"expected at least 3 by 3 to read fx, fy, cx and cy\n"
                f"check the calibration file for this scene"
            )
        found[serial] = Calibration(
            fx=float(matrix[0, 0]), fy=float(matrix[1, 1]),
            cx=float(matrix[0, 2]), cy=float(matrix[1, 2]),
            width=width, height=height,
            # The file carries a projection matrix and no distortion. Saying
            # "none" here would claim the lens has none, which nobody measured.
            distortion_model=None, distortion_coeffs=None, distortion_order=None,
        )
    return found, len(devices) - len(found)


def config_number(name: Any) -> int:
    """The config a directory name or an episode field names, as its number."""
    digits = ""
    for character in reversed(str(name).strip()):
        if not character.isdigit():
            break
        digits = character + digits
    if not digits:
        raise ValueError(
            f"config {name!r} ends in no digits\n"
            f"expected a name or number saying which RH20T config it is\n"
            f"check the episode metadata and the calibration directory names"
        )
    return int(digits)


def rh20t_scenes(calib: list[Path]) -> list[Path]:
    """Every calibration scene under the trees that carries an intrinsics file."""
    scenes = sorted(
        scene for tree in calib for scene in tree.glob(f"*/{RH20T_CALIB}/*")
        if (scene / RH20T_INTRINSICS).is_file()
    )
    if not scenes:
        raise FileNotFoundError(
            f"no */{RH20T_CALIB}/*/{RH20T_INTRINSICS} under "
            f"{[str(tree) for tree in calib]}\nexpected RH20T's per-config "
            f"calibration scenes there\npass --calib the config directories' parent"
        )
    return scenes


def rh20t_calibrations(
    scenes: list[Path],
) -> tuple[dict[tuple[int, str], Calibration], Counter]:
    """The calibration scenes -> one calibration per config and camera serial.

    A serial two scenes of one config disagree about is dropped rather than
    resolved by order: the two scenes are equally good evidence and nothing
    here can say which frame belongs to which.
    """
    counts: Counter = Counter()
    per_key: dict[tuple[int, str], dict[tuple, Calibration]] = {}
    for scene in scenes:
        config = config_number(scene.parent.parent.name)
        found, missing = rh20t_scene(scene)
        counts["device_without_intrinsics"] += missing
        for serial, calibration in found.items():
            numbers = (calibration.fx, calibration.fy, calibration.cx,
                       calibration.cy, calibration.width, calibration.height)
            per_key.setdefault((config, serial), {})[numbers] = calibration
    resolved = {}
    for key, values in per_key.items():
        if len(values) != 1:
            counts["serial_disagrees_across_scenes"] += 1
            continue
        resolved[key] = next(iter(values.values()))
    return resolved, counts


def rh20t_cameras(
    episode: dict[str, Any], where: str,
) -> list[tuple[str, str, int | None, str]]:
    """One episode's cameras: each one's name, stored column, stated slot and serial.

    The per-camera tree records one camera per episode and names its serial at
    the top level, so the layout's one view is the column that camera was
    written into and there is no slot to compare.

    The grid and joined trees write up to three cameras of one scene into one
    episode. Which camera a column holds is a property of the episode and not
    of the directory: the same column is another serial in the next episode,
    and how many columns there are varies by directory. So each column's serial
    is read from the entry that names the column. The writer built the column
    name and the serial from one record per camera, which is why the pair here
    is what it wrote; taking the serials as a list and pairing them with the
    columns by position would give one camera's focal length to another
    camera's pictures wherever the two orders differ, and no later stage reads
    pixels to notice.
    """
    stated = episode.get(RH20T_VIEWS_FIELD)
    if stated is None:
        camera, key = next(iter(CAMERA_VIEWS["rh20t"].items()))
        return [(camera, key, None,
                 str(_field(episode, RH20T_SERIAL_FIELD, where)).strip())]
    cameras = []
    for entry in stated:
        slot = int(_field(entry, RH20T_VIEW_SLOT, where))
        cameras.append((
            f"{RH20T_SLOT_CAMERA}{slot}",
            str(_field(entry, RH20T_VIEW_KEY, where)),
            slot,
            str(_field(entry, RH20T_VIEW_SERIAL, where)).strip(),
        ))
    return cameras


def read_rh20t(
    root: Path, calib: list[Path], directories: list[Path], document: dict[str, Any],
) -> tuple[list[dict[str, Any]], Counter, dict[str, Path]]:
    """RH20T's vendor calibration, joined by the config and serial per camera."""
    scenes = rh20t_scenes(calib)
    per_key, counts = rh20t_calibrations(scenes)
    records = []
    for directory in directories:
        name = stage.relative(root, directory)
        by_key = layout_slots(document, name, "rh20t")
        for episode in episode_metadata(directory):
            index = int(_field(episode, "episode_index", name))
            where = f"{name} episode {index}"
            config = config_number(_field(episode, RH20T_CONFIG_FIELD, name))
            for camera, view, stated, serial in rh20t_cameras(episode, where):
                if view not in by_key:
                    counts["camera_not_in_layout"] += 1
                    continue
                slot = by_key[view]
                if stated is not None and stated != slot:
                    # The episode and the layout disagree about which slot this
                    # column is, so one of the two names a different picture
                    # and nothing here can say which.
                    counts["slot_disagrees_with_layout"] += 1
                    continue
                calibration = per_key.get((config, serial))
                if calibration is None:
                    counts["camera_without_calibration"] += 1
                    continue
                reason = refusal(calibration)
                if reason is not None:
                    counts[reason] += 1
                    continue
                records.append(as_record(
                    dataset=name, episode=index, camera=camera, view=view,
                    slot=slot, calibration=calibration,
                ))
    inputs = {}
    for scene in scenes:
        for name in (RH20T_INTRINSICS, RH20T_DEVICES):
            inputs[stage.relative(root, scene / name)] = scene / name
    return records, counts, inputs


def one_named(calib: list[Path], name: str) -> Path:
    """The one calibration tree holding this file, refusing none and refusing two."""
    found = [tree / name for tree in calib if (tree / name).is_file()]
    if len(found) != 1:
        raise FileNotFoundError(
            f"{len(found)} of the calibration trees hold {name}\n"
            f"expected exactly one; nothing here says which of two to read\n"
            f"pass --calib the single tree that holds it"
        )
    return found[0]


def droid_serial(sidecar: dict[str, Any], camera: str) -> str | None:
    """The serial the sidecar names for one camera, or None when it names none."""
    value: Any = sidecar
    for key in DROID_SERIAL_AT[camera]:
        if not isinstance(value, dict) or value.get(key) is None:
            return None
        value = value[key]
    return str(value).strip()


def droid_calibration(entry: dict[str, Any], where: str) -> Calibration:
    """One camera's annotation entry -> the calibration this tool states for it.

    The annotation writes the matrix as fx, cx, fy, cy and the pixel size beside
    it, so nothing here is rescaled: the record states the resolution the
    profile was measured at, and the model reads each number as a ratio of it.
    """
    matrix = list(_field(entry, DROID_MATRIX, where))
    if len(matrix) != 4:
        raise ValueError(
            f"{where} camera matrix holds {len(matrix)} numbers\n"
            f"expected four, fx then cx then fy then cy\n"
            f"check the annotation file this run reads"
        )
    coefficients = tuple(float(value) for value in _field(entry, DROID_COEFFS, where))
    # The file states coefficients and no model name. All zero is the rectified
    # stream's own statement and is carried as "none"; anything else names a
    # model nobody wrote down, which distortion_refusal then refuses.
    model = "none" if all(value == 0.0 for value in coefficients) else None
    return Calibration(
        fx=float(matrix[0]), cx=float(matrix[1]),
        fy=float(matrix[2]), cy=float(matrix[3]),
        width=int(_field(entry, "width", where)),
        height=int(_field(entry, "height", where)),
        distortion_model=model, distortion_coeffs=coefficients,
        distortion_order=None,
    )


def droid_wanted(
    root: Path, directories: list[Path], document: dict[str, Any],
) -> dict[tuple[str, int], tuple[str, int, dict[str, tuple[str, int]]]]:
    """The episodes this run covers, keyed and sized the way the sidecar is."""
    wanted: dict[tuple[str, int], tuple[str, int, dict[str, tuple[str, int]]]] = {}
    claimed: dict[str, str] = {}
    for directory in directories:
        name = stage.relative(root, directory)
        slots = view_slots(document, name, "droid")
        repo = directory.name
        if repo in claimed:
            raise ValueError(
                f"{name} and {claimed[repo]} are both named {repo!r}\n"
                f"expected one directory per name the sidecar keys by\n"
                f"pass only one of them, or rename the other"
            )
        claimed[repo] = name
        for episode in episode_metadata(directory):
            index = int(_field(episode, "episode_index", name))
            wanted[(repo, index)] = (name, int(_field(episode, "length", name)), slots)
    return wanted


def read_droid(
    root: Path, calib: list[Path], directories: list[Path], document: dict[str, Any],
) -> tuple[list[dict[str, Any]], Counter, dict[str, Path]]:
    """DROID's recorded profiles, joined per episode through its own sidecar.

    The sidecar keys an episode by its position in a directory, which is a key
    only if the two agree about that episode. So the join is confirmed against
    the frame count both sides state, and a pair that disagrees emits nothing
    rather than calibrating a neighbouring episode.
    """
    sidecar_path = one_named(calib, DROID_SIDECAR)
    annotations_path = one_named(calib, DROID_ANNOTATIONS)
    wanted = droid_wanted(root, directories, document)
    counts: Counter = Counter()
    resolved: dict[tuple[str, int], dict[str, Any]] = {}
    seen: set[tuple[str, int]] = set()
    with open(sidecar_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sidecar = json.loads(line)
            key = (str(_field(sidecar, DROID_REPO_FIELD, DROID_SIDECAR)),
                   int(_field(sidecar, "episode_index", DROID_SIDECAR)))
            if key not in wanted:
                continue
            seen.add(key)
            if int(_field(sidecar, DROID_ROWS_FIELD, DROID_SIDECAR)) != wanted[key][1]:
                counts["episode_rows_disagree"] += 1
                continue
            resolved[key] = sidecar
    counts["episode_absent_from_sidecar"] += len(set(wanted) - seen)

    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    records = []
    for key in sorted(resolved):
        sidecar = resolved[key]
        dataset, _, slots = wanted[key]
        episode = key[1]
        cameras = annotations.get(str(_field(sidecar, DROID_EPISODE_FIELD, dataset)))
        if cameras is None:
            counts["episode_without_calibration"] += 1
            continue
        for camera in sorted(slots):
            serial = droid_serial(sidecar, camera)
            if serial is None:
                counts["camera_serial_unknown"] += 1
                continue
            entry = cameras.get(serial)
            if entry is None:
                counts["camera_absent"] += 1
                continue
            where = f"{dataset} episode {episode} camera {camera}"
            calibration = droid_calibration(entry, where)
            reason = refusal(calibration)
            if reason is not None:
                counts[reason] += 1
                continue
            view, slot = slots[camera]
            records.append(as_record(
                dataset=dataset, episode=episode, camera=camera, view=view,
                slot=slot, calibration=calibration,
            ))
    inputs = {stage.relative(root, path): path
              for path in (sidecar_path, annotations_path)}
    return records, counts, inputs


@dataclass(frozen=True)
class Stated:
    """One calibration an OXE sidecar states, in the words it states it in."""

    dataset: str
    view: str
    provenance: str
    calibration: Calibration | None


def oxe_calibration(block: dict[str, Any], where: str) -> Calibration:
    """An intrinsics block of either OXE sidecar -> the calibration it states."""
    distortion = block.get("distortion") or {}
    model, coefficients = distortion.get("model"), distortion.get("coeffs")
    return Calibration(
        fx=float(_field(block, "fx", where)), fy=float(_field(block, "fy", where)),
        cx=float(_field(block, "cx", where)), cy=float(_field(block, "cy", where)),
        width=int(_field(block, "width", where)),
        height=int(_field(block, "height", where)),
        distortion_model=None if model is None else distortion_name(model),
        distortion_coeffs=(None if coefficients is None
                           else tuple(float(value) for value in coefficients)),
        distortion_order=None,
    )


def oxe_statement(record: dict[str, Any], shape: dict[str, Any], where: str) -> Stated:
    """One OXE sidecar record -> the dataset, view, provenance and numbers in it."""
    origin: Any = record
    for key in shape["provenance"]:
        origin = _field(origin, key, where)
    block = record.get(shape["block"])
    return Stated(
        dataset=str(_field(record, "dataset", where)),
        view=str(_field(record, shape["view"], where)),
        provenance=str(origin),
        calibration=None if not block else oxe_calibration(block, where),
    )


def oxe_wanted(
    root: Path, directories: list[Path], document: dict[str, Any],
) -> dict[str, list[tuple[str, dict[str, tuple[str, int]], Path]]]:
    """The directories this run covers, under the dataset name a sidecar uses.

    The episode list is not read here. Most of OXE has no calibration that
    survives the provenance gate, and parsing a million episode records to
    expand a statement that is then refused reads the corpus for nothing.
    """
    wanted: dict[str, list[tuple[str, dict[str, tuple[str, int]], Path]]] = {}
    for directory in directories:
        name = stage.relative(root, directory)
        slots = view_slots(document, name, "oxe")
        layout = layout_table.resolve(document, name)
        wanted.setdefault(layout.name, []).append((name, slots, directory))
    return wanted


def read_oxe(
    root: Path, calib: list[Path], directories: list[Path], document: dict[str, Any],
) -> tuple[list[dict[str, Any]], Counter, dict[str, Path]]:
    """The remaining OXE calibration trees, gated on the provenance each states.

    Nothing is translated here. A record's view has to be a stored column the
    layout names and its provenance has to be a word the allow list holds; one
    that is neither is counted under what it said, so a tree that supplies no
    usable calibration says so in numbers rather than in silence.
    """
    paths = sidecars_named(calib, tuple(sorted(OXE_SHAPES)),
                           "the OXE calibration sidecars")
    wanted = oxe_wanted(root, directories, document)
    counts: Counter = Counter()
    records = []
    for path in paths:
        shape = OXE_SHAPES[path.name]
        where = stage.relative(root, path)
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                statement = oxe_statement(json.loads(line), shape, where)
                counts[f"stated_{statement.provenance}"] += 1
                if statement.provenance not in rays.PROVENANCE_ALLOWED:
                    counts["provenance_not_allowed"] += 1
                    continue
                # Before the numbers are judged, so every later count is about a
                # camera this run covers rather than one it never asked for.
                if statement.dataset not in wanted:
                    counts["dataset_not_in_run"] += 1
                    continue
                if statement.calibration is None:
                    counts["camera_absent"] += 1
                    continue
                reason = refusal(statement.calibration)
                if reason is not None:
                    counts[reason] += 1
                    continue
                for dataset, slots, directory in wanted[statement.dataset]:
                    if statement.view not in slots:
                        counts["view_not_a_stored_column"] += 1
                        continue
                    view, slot = slots[statement.view]
                    records.extend(
                        as_record(dataset=dataset,
                                  episode=int(_field(episode, "episode_index", dataset)),
                                  camera=statement.view, view=view, slot=slot,
                                  calibration=statement.calibration,
                                  provenance=statement.provenance)
                        for episode in episode_metadata(directory)
                    )
    return records, counts, {stage.relative(root, path): path for path in paths}


def droid3cam_calibration(block: dict[str, Any], where: str) -> Calibration:
    """One camera's profile as the converted tree states it, unchanged.

    The conversion already refused a profile measured against another raster
    than the recording, so a block that is here describes the frames beside it.
    """
    for name in ("fx", "fy", "cx", "cy", "width", "height"):
        _field(block, name, where)
    coefficients = block.get("distortion_coeffs")
    model = block.get("distortion_model")
    return Calibration(
        fx=float(block["fx"]), fy=float(block["fy"]),
        cx=float(block["cx"]), cy=float(block["cy"]),
        width=int(block["width"]), height=int(block["height"]),
        distortion_model=None if model is None else distortion_name(model),
        distortion_coeffs=None if coefficients is None else tuple(
            float(value) for value in coefficients
        ),
        distortion_order=block.get("distortion_order"),
    )


def read_droid3cam(
    root: Path, calib: list[Path], directories: list[Path], document: dict[str, Any],
) -> tuple[list[dict[str, Any]], Counter, dict[str, Path]]:
    """The three-camera DROID tree's own profiles, one record per calibrated camera.

    There is no join: the conversion wrote each episode's profiles into that
    episode's metadata, keyed by the stored column. A camera with no block, and
    a camera whose block is null, get no record and no calibration.
    """
    counts: Counter = Counter()
    records = []
    for directory in directories:
        name = stage.relative(root, directory)
        slots = view_slots(document, name, "droid3cam")
        for episode in episode_metadata(directory):
            index = int(_field(episode, "episode_index", name))
            stated = episode.get(DROID3CAM_KEY) or {}
            for camera in sorted(slots):
                block = stated.get(camera)
                if block is None:
                    counts["camera_absent"] += 1
                    continue
                where = f"{name} episode {index} camera {camera}"
                calibration = droid3cam_calibration(block, where)
                reason = refusal(calibration)
                if reason is not None:
                    counts[reason] += 1
                    continue
                view, slot = slots[camera]
                records.append(as_record(
                    dataset=name, episode=index, camera=camera, view=view,
                    slot=slot, calibration=calibration,
                ))
    return records, counts, {}


def robotwin2_calibration(block: dict[str, Any], where: str) -> Calibration:
    """One camera's sidecar block -> the calibration this tool states for it.

    The four numbers are the matrix the recording stored, copied, and the size
    is the frame they were measured against, read off that frame. The simulator
    renders a plain pinhole and states no distortion, so the record carries
    neither distortion field rather than a model of zeroes nobody wrote.
    """
    for name in ("fx", "fy", "cx", "cy", "width", "height"):
        if name not in block:
            raise KeyError(
                f"{where} camera block has no {name!r}\n"
                f"expected fx, fy, cx, cy, width and height among {sorted(block)[:8]}\n"
                f"rebuild the sidecar with the RoboTwin calibration stager"
            )
    return Calibration(
        fx=float(block["fx"]), fy=float(block["fy"]),
        cx=float(block["cx"]), cy=float(block["cy"]),
        width=int(block["width"]), height=int(block["height"]),
        distortion_model=None, distortion_coeffs=None, distortion_order=None,
    )


def robotwin2_wanted(
    root: Path, directories: list[Path], document: dict[str, Any],
) -> dict[tuple[str, int], list[tuple[str, int, int, dict[str, tuple[str, int]]]]]:
    """The episodes this run covers, keyed by the capture and index they came from."""
    wanted: dict[tuple[str, int], list[tuple[str, int, int, dict[str, tuple[str, int]]]]] = {}
    for directory in directories:
        name = stage.relative(root, directory)
        slots = view_slots(document, name, "robotwin2")
        for episode in episode_metadata(directory):
            key = (str(_field(episode, ROBOTWIN_SOURCE_FIELD, name)),
                   int(_field(episode, ROBOTWIN_EPISODE_FIELD, name)))
            wanted.setdefault(key, []).append((
                name, int(_field(episode, "episode_index", name)),
                int(_field(episode, ROBOTWIN_LENGTH_FIELD, name)), slots))
    return wanted


def read_robotwin2(
    root: Path, calib: list[Path], directories: list[Path], document: dict[str, Any],
) -> tuple[list[dict[str, Any]], Counter, dict[str, Path]]:
    """RoboTwin's staged sidecar, joined to our directories by the key it carries."""
    wanted = robotwin2_wanted(root, directories, document)
    files = sidecars_named(calib, (ROBOTWIN_SIDECAR,),
                           "the sidecar the RoboTwin calibration stager writes")
    counts: Counter = Counter()
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for path in files:
        _, lines = stage.read_jsonl(path)
        for sidecar in lines:
            key = (str(_field(sidecar, ROBOTWIN_ARCHIVE, str(path))),
                   int(_field(sidecar, ROBOTWIN_EPISODE, str(path))))
            if key not in wanted:
                continue
            if key in seen:
                raise ValueError(
                    f"two sidecar records name {key[0]} episode {key[1]}\n"
                    f"expected one calibration per episode; nothing here says which\n"
                    f"remove the duplicate from the calibration tree"
                )
            seen.add(key)
            blocks = _field(sidecar, ROBOTWIN_CAMERAS, str(key))
            frames = int(_field(sidecar, ROBOTWIN_FRAMES, str(key)))
            for dataset, episode, rows, slots in wanted[key]:
                if frames != rows:
                    # The key alone would join a renumbered capture just as
                    # readily; a recording of another length is a different one.
                    raise ValueError(
                        f"{dataset} episode {episode} holds {rows} rows and "
                        f"its sidecar record {frames} frames\n"
                        f"expected the recording the conversion read\n"
                        f"rebuild the sidecar from the captures the corpus came from"
                    )
                for camera in sorted(slots):
                    block = blocks.get(camera)
                    if block is None:
                        counts["camera_absent"] += 1
                        continue
                    where = f"{dataset} episode {episode} camera {camera}"
                    calibration = robotwin2_calibration(block, where)
                    reason = refusal(calibration)
                    if reason is not None:
                        counts[reason] += 1
                        continue
                    view, slot = slots[camera]
                    records.append(as_record(
                        dataset=dataset, episode=episode, camera=camera, view=view,
                        slot=slot, calibration=calibration,
                    ))
    counts["episode_absent_from_sidecar"] += sum(
        len(wanted[key]) for key in set(wanted) - seen
    )
    inputs = {stage.relative(root, path): path for path in files}
    return records, counts, inputs


READERS = {"agibot": read_agibot, "droid": read_droid, "droid3cam": read_droid3cam,
           "oxe": read_oxe, "rh20t": read_rh20t, "robotwin2": read_robotwin2}


def check_one_per_camera(records: list[dict[str, Any]]) -> None:
    """Raise when two records calibrate the same camera of the same episode.

    A reader that joins one episode to two calibrations would write both, and
    a consumer reading by position would take whichever came last without
    anything saying a choice was made.
    """
    seen: Counter = Counter(
        (record["dataset"], record["episode"], record["slot"]) for record in records
    )
    repeated = sorted(key for key, count in seen.items() if count > 1)
    if not repeated:
        return
    raise ValueError(
        f"{len(repeated)} cameras have more than one record, first {repeated[:3]}\n"
        f"expected one record per directory, episode and slot\n"
        f"fix the join in the reader for this source"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, help="the one corpus root")
    parser.add_argument("--source", required=True, choices=sorted(READERS),
                        help="which reader to use")
    # Optional because a source may state its calibration in its own tree. A
    # reader that needs one raises by name when none was passed rather than
    # reading a tree the caller did not choose.
    parser.add_argument("--calib", action="append", default=[],
                        help="a calibration tree of that source, under the root")
    parser.add_argument("--dir", action="append", default=[], help="a directory to read")
    parser.add_argument("--dirs-file", help="a file of directories, one per line")
    parser.add_argument("--layouts", required=True, help="the layout file naming the views")
    parser.add_argument("--out", required=True,
                        help="where this source's intrinsics records go")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    root = stage.resolve_root(args.root)
    dirs_file = stage.under_root(root, args.dirs_file) if args.dirs_file else None
    directories = read_dir_list(root, args.dir, dirs_file)
    calib = [stage.under_root(root, path) for path in args.calib]
    out = stage.under_root(root, args.out)
    stage.refuse_write_into_input(out, [*directories, *calib])

    layouts_path = stage.under_root(root, args.layouts)
    document = layout_table.load(layouts_path)
    records, counts, inputs = READERS[args.source](root, calib, directories, document)
    records.sort(key=lambda record: (record["dataset"], record["episode"], record["slot"]))
    check_one_per_camera(records)

    covered = {(record["dataset"], record["episode"]) for record in records}
    head = stage.header(
        "build_intrinsics",
        root=root,
        inputs={
            **directory_inputs(root, directories, dirs_file),
            **inputs,
            # The layout names each camera's view key and slot, so a record is
            # only current for the layout it was mapped through.
            "layouts": layouts_path,
        },
        tree=REPO,
        extra={"source": args.source, "directories": len(directories),
               "episodes_with_a_camera": len(covered), "records": len(records),
               "counts": dict(sorted(counts.items()))},
    )
    written = stage.write_jsonl(out, head, records)
    logger.info("%s: %d directories, %d episodes with a camera, %d records",
                args.source, len(directories), len(covered), len(records))
    for reason, count in sorted(counts.items()):
        logger.info("%s %d", reason, count)
    logger.info("%s %s", "wrote" if written else "unchanged", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
