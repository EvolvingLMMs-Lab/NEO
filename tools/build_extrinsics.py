"""Stage 7b: one camera rotation per episode, from the source's own files.

One record per directory, episode and camera this run covers. A camera fixed to
the room holds the rotation from its frame into the robot base frame; a camera
bolted to an arm holds the rotation from its frame into that arm's, which the
loader composes with the arm's own pose. A record with a defect holds neither,
so that episode is read in robot frame and the suffix says the extrinsics are
unknown.

The join is the identity key the source writes beside each episode, never a
position, and it is confirmed against the frame count both sides state.

Usage:
    python tools/build_extrinsics.py --root /corpus --source agibot \
        --calib calib/agibot_extrinsics --dirs-file config/agibot_dirs.txt \
        --layouts config/oxe3src.json --out prep/extrinsics_agibot.jsonl
    python tools/build_extrinsics.py --root /corpus --source agibot_wrist \
        --calib calib/agibot_geometry --dirs-file config/agibot_dirs.txt \
        --layouts config/oxe3src.json --out prep/extrinsics_agibot_wrist.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import layouts as layout_table
import numpy as np
import pyarrow.parquet as pq
import stage
from build_intrinsics import episode_metadata, sidecars_named
from scan_source import directory_inputs, read_dir_list
from validate_episodes import rotation_error, rotation_of

from neo_vla.data import rays, registry

logger = logging.getLogger("build_extrinsics")

REPO = Path(__file__).resolve().parent.parent

# The word this stage writes for a pose the source measured and published. It
# goes through the loader's own allow list before it can reach the file, so a
# fitted pose cannot be written here under a word that reads as a measurement.
PROVENANCE = "vendor"

# Why a pose cannot be stated. One name per record, so a run's counts say which
# check removed each episode from the camera-frame corpus.
DEFECTS = (
    "episode_absent_from_sidecar",
    "duplicate_record",
    "provenance_not_allowed",
    "pose_missing",
    "episode_rows_disagree",
    "pose_not_rotation",
    "pose_implausible",
    "pose_varies_within_episode",
    "view_absent_from_sidecar",
    "track_not_aligned",
    "track_rows_disagree",
    "arm_pose_missing",
)

# How far a matrix may sit from orthonormal and still be read as a rotation.
# The loader repeats this audit on the array the cache stage writes, so a value
# looser than the loader's would state a pose the loader then refuses.
ROTATION_TOLERANCE = 1e-4

# How far the camera may turn inside one episode before a single pose per
# episode stops describing it. AgiBot's own distribution has a gap here: 30
# episodes lie between 0.10 and 0.45 degrees and the next two are at 4.85 and
# 47.44, so any threshold from 0.5 to 4.8 refuses the same two episodes.
#
# This is a gate on the head, and only on the head. A camera bolted to an arm
# turns by a median 55 degrees inside one episode, so routing it through here
# would leave it with no poses at all under a build that still succeeded.
# read_agibot_wrist asks whether the vendor's aligned track covers the episode
# instead, and stores the arm offset, which does hold still.
WITHIN_EPISODE_DEGREES = 1.0

# What a pose that names this robot's own base frame looks like: an optical
# axis within a quarter turn of the base's forward direction, and a camera
# inside a box around the base. Wide on purpose. The bands catch a pose written
# against some other frame, not a badly aimed mount.
OPTICAL_AXIS_YAW_DEGREES = 90.0
CAMERA_FORWARD_METRES = 1.5
CAMERA_SIDEWAYS_METRES = 1.0
CAMERA_HEIGHT_METRES = (0.3, 2.5)

# AgiBot's head pose sidecar, one record per episode, and the fields it states
# each episode's identity, pose, vendor frame count and head motion in.
AGIBOT_SIDECAR = "head_extrinsic_sidecar.jsonl"
AGIBOT_KEY = "raw_file_name"
AGIBOT_POSE = "T_base_from_cam_4x4"
AGIBOT_ROWS = "n_aligned"
AGIBOT_MOTION = "within_episode_max_rot_deg"
AGIBOT_ORIGIN = "calib_source"

# The origins whose numbers are the vendor's own, copied. A record naming
# anything else is a pose somebody derived, and the two are told apart here
# rather than downstream, where they have the same shape and both are
# rotations.
AGIBOT_VENDOR_ORIGINS = ("agibot_head_extrinsic_aligned_frame0",)

# The field a restated pose keeps the vendor's own beside it in. A record
# carrying it states a pose that is not the published one whatever its origin
# field says, so its presence alone refuses the record.
PUBLISHED_POSE = "T_base_from_cam_4x4_published"

# AgiBot's geometry sidecar, one record per episode holding all three cameras.
# It states each camera's aligned track: how many frames it covers, whether
# that matches the episode, and the pose at the track's first frame.
AGIBOT_GEOMETRY = "geometry_sidecar.jsonl"
AGIBOT_VIEWS = "views"
AGIBOT_FRAME_ZERO = "R_base_from_cam"
AGIBOT_ALIGNED = "aligned"
AGIBOT_STATUS = "status"

# The field a mounted camera's offset is written under, which is the key the
# layout names for it. A pose in the base frame and an offset from an arm are
# both rotations, so they are never written under one name.
POSE_FIELD = "rotation_base_from_camera"
MOUNT_FIELD = "rotation_end_effector_from_camera"

# The rotation encoding this reader decodes from a stored state row. AgiBot's
# quaternions are xyzw, identified rather than assumed: read as wxyz the mount
# offset wanders by 66 to 102 degrees inside an episode instead of 2.7e-06.
QUATERNION_KIND = "quat_xyzw"

# Which layout views each reader's poses describe, and the extrinsics kind the
# layout has to give each one. Checked against the cameras the manifest will
# offer as action frames, so a pose cannot be written against one camera and
# read as another's, and an arm offset cannot land where a base-frame pose goes.
POSE_VIEWS = {
    "agibot": (("head", "observation.images.head", "episode"),),
    "agibot_wrist": (
        ("hand_left", "observation.images.hand_left", "mounted"),
        ("hand_right", "observation.images.hand_right", "mounted"),
    ),
}

# Which layout source each reader's directories belong to, as build_intrinsics
# spells it: the reader's name and the layout's source are two words.
LAYOUT_SOURCE = {"agibot": "agibot", "agibot_wrist": "agibot"}

# What each reader joins on, and the quantity that confirms the join. Recorded
# in the header so a reader of the output knows what was checked and what was
# only assumed.
JOIN = {
    "agibot": {
        "key": AGIBOT_KEY,
        "confirmed_on": "the vendor pose count against the episode's rows",
    },
    "agibot_wrist": {
        "key": AGIBOT_KEY,
        "confirmed_on": "the aligned track's frame count against the episode's rows",
    },
}


def _defect(name: str, **detail: Any) -> dict[str, Any]:
    if name not in DEFECTS:
        raise KeyError(
            f"defect {name!r} is not named in DEFECTS\n"
            f"expected one of the names this stage records\n"
            f"add the name before reporting it, so a reader can key on it"
        )
    return {"name": name, **detail}


def optical_axis_yaw(rotation: np.ndarray) -> float:
    """How far the camera looks off the base's forward direction, in degrees.

    The optical axis is the third column: the camera convention writes z
    forward, so the column is that axis expressed in the robot base frame.
    """
    forward = rotation[:, 2]
    return math.degrees(math.atan2(float(forward[1]), float(forward[0])))


def plausible(rotation: np.ndarray, origin: list[float] | None) -> bool:
    """Whether the pose sits where this robot's own camera can sit."""
    if abs(optical_axis_yaw(rotation)) > OPTICAL_AXIS_YAW_DEGREES:
        return False
    if origin is None or len(origin) != 3:
        return False
    x, y, z = (float(value) for value in origin)
    if not all(math.isfinite(value) for value in (x, y, z)):
        return False
    low, high = CAMERA_HEIGHT_METRES
    return (abs(x) <= CAMERA_FORWARD_METRES and abs(y) <= CAMERA_SIDEWAYS_METRES
            and low <= z <= high)


def reference_views(
    document: dict[str, Any], directory: str, source: str
) -> list[tuple[str, str, int]]:
    """The cameras this run's poses describe, each with its camera slot.

    Resolved from POSE_VIEWS and checked against the extrinsics kind the layout
    gives each one, rather than from the views that happen to have a pose:
    several cameras of a directory carry one now, and a reader writes for the
    ones it was built to read. A layout that gives this reader's camera another
    kind raises here rather than writing an arm offset where a base-frame pose
    goes, which is the same shape and the same audit.
    """
    layout = layout_table.resolve(document, directory)
    expected = LAYOUT_SOURCE[source]
    if layout.source != expected:
        raise ValueError(
            f"directory {directory} has layout source {layout.source!r}\n"
            f"expected {expected!r}, the layout source --source {source} reads\n"
            f"pass only that source's directories, or change --source"
        )
    slots = {view["key"]: int(view["slot"]) for view in layout.views}
    found = []
    for camera, key, kind in POSE_VIEWS[source]:
        stated = layout_table.view_extrinsics(layout, key).get("kind")
        if key not in slots or stated != kind:
            raise KeyError(
                f"{directory} gives {key!r} extrinsics kind {stated!r}\n"
                f"expected kind {kind!r}, what --source {source} writes for it\n"
                f"fix POSE_VIEWS in this tool or the layout file's extrinsics block"
            )
        found.append((camera, key, slots[key]))
    return found


def as_record(
    *, dataset: str, episode: int, camera: str, view: str, slot: int,
    source_key: str, rotation: np.ndarray | None, motion: float | None,
    defects: list[dict[str, Any]], rotation_key: str = POSE_FIELD,
) -> dict[str, Any]:
    """One episode's rotation for one camera, or the defect there instead.

    The rotation and the provenance word are written together and only when
    the record has no defect, so nothing downstream can read a number out of a
    record this stage refused. rotation_key names which rotation it is: a pose
    in the base frame, or the offset from the arm a mounted camera is bolted
    to. Both are proper 3x3 rotations and only the name tells them apart, so
    the two never share a field.
    """
    record = {
        "dataset": dataset,
        "episode": int(episode),
        "camera": camera,
        "view": view,
        "slot": int(slot),
        "source_key": source_key,
        "provenance": None,
        POSE_FIELD: None,
        MOUNT_FIELD: None,
        "within_episode_deg": None if motion is None else float(motion),
        "defects": defects,
        "pass": not defects,
    }
    if not defects and rotation is not None:
        rays.check_provenance(PROVENANCE)
        record["provenance"] = PROVENANCE
        record[rotation_key] = [
            float(value) for value in np.asarray(rotation, dtype=np.float64).reshape(9)
        ]
    return record


def agibot_wanted(
    root: Path, directories: list[Path], document: dict[str, Any],
) -> dict[str, list[tuple[str, int, int, str, str, int]]]:
    """The episodes this run covers, keyed by the sidecar's own join key.

    The rows each episode declares travel with it, because they are what
    confirms the join: a record joined to the wrong episode states a different
    recording's frame count.
    """
    wanted: dict[str, list[tuple[str, int, int, str, str, int]]] = {}
    for directory in directories:
        name = stage.relative(root, directory)
        camera, view, slot = reference_views(document, name, "agibot")[0]
        for episode in episode_metadata(directory):
            key = str(episode[AGIBOT_KEY])
            index = int(episode["episode_index"])
            rows = int(episode["length"])
            wanted.setdefault(key, []).append((name, index, rows, camera, view, slot))
    return wanted


def sidecar_records(files: list[Path], wanted: set[str]) -> Iterator[dict[str, Any]]:
    """Every sidecar record this run needs, streamed rather than held."""
    for path in files:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if str(record.get(AGIBOT_KEY)) in wanted:
                    yield record


def agibot_defect(
    sidecar: dict[str, Any] | None, rows: int, tolerance: float, motion_limit: float,
) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    """The one thing wrong with this episode's pose, and the pose when nothing is.

    The order is the order the evidence arrives in. A record that restates the
    vendor's numbers is refused before its rotation is judged, because a
    restated pose is a proper rotation too. The frame count comes next, since
    nothing a record says about a pose means anything until the record is known
    to belong to this episode. The pose itself is judged last.
    """
    if sidecar is None:
        return [_defect("episode_absent_from_sidecar")], None
    origin = str(sidecar.get(AGIBOT_ORIGIN))
    if origin not in AGIBOT_VENDOR_ORIGINS or PUBLISHED_POSE in sidecar:
        return [_defect("provenance_not_allowed", stated=origin,
                        restated=PUBLISHED_POSE in sidecar)], None
    stored = sidecar.get(AGIBOT_POSE)
    if stored is None:
        return [_defect("pose_missing", key=AGIBOT_POSE)], None
    stated_rows = sidecar.get(AGIBOT_ROWS)
    if stated_rows is None or int(stated_rows) != rows:
        return [_defect("episode_rows_disagree", stated=stated_rows, rows=rows)], None
    matrix = rotation_of(stored)
    if matrix is None:
        return [_defect("pose_not_rotation", reason="shape")], None
    error = rotation_error(matrix)
    if error > tolerance:
        return [_defect("pose_not_rotation", error=error)], None
    if not plausible(matrix, sidecar.get("cam_origin_m")):
        return [_defect("pose_implausible", yaw=optical_axis_yaw(matrix),
                        origin=sidecar.get("cam_origin_m"))], None
    motion = sidecar.get(AGIBOT_MOTION)
    if motion is None or float(motion) > motion_limit:
        return [_defect("pose_varies_within_episode", degrees=motion)], None
    return [], matrix


def read_agibot(
    root: Path, calib: list[Path], directories: list[Path], document: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], Counter, dict[str, Path]]:
    """AgiBot's head poses, joined to our directories by the key they carry."""
    wanted = agibot_wanted(root, directories, document)
    files = sidecars_named(calib, (AGIBOT_SIDECAR,), "AgiBot's per-task head poses")
    found: dict[str, dict[str, Any]] = {}
    repeated: set[str] = set()
    for sidecar in sidecar_records(files, set(wanted)):
        key = str(sidecar[AGIBOT_KEY])
        if key in found:
            repeated.add(key)
        found[key] = sidecar

    counts: Counter = Counter()
    records = []
    for key in sorted(wanted):
        for dataset, episode, rows, camera, view, slot in wanted[key]:
            if key in repeated:
                defects, matrix = [_defect("duplicate_record", key=key)], None
            else:
                defects, matrix = agibot_defect(
                    found.get(key), rows, args.rotation_tolerance,
                    args.within_episode_tolerance,
                )
            for defect in defects:
                counts[defect["name"]] += 1
            motion = (found.get(key) or {}).get(AGIBOT_MOTION)
            records.append(as_record(
                dataset=dataset, episode=episode, camera=camera, view=view, slot=slot,
                source_key=key, rotation=matrix, motion=motion, defects=defects,
            ))
    counts["sidecar_records_not_in_run"] += len(set(found) - set(wanted))
    inputs = {stage.relative(root, path): path for path in files}
    return records, counts, inputs


def wrist_defect(
    view: dict[str, Any] | None, rows: int, tolerance: float,
) -> list[dict[str, Any]]:
    """The one thing wrong with this camera's published pose, or nothing.

    The question is whether the vendor's aligned track is there and covers the
    episode, never whether the pose holds still: a camera bolted to an arm
    turns by a median 55 degrees inside one episode, which is the reason it is
    stored as an offset. The
    frame-zero rotation is audited because that is the pose the offset is
    measured from.
    """
    if view is None:
        return [_defect("view_absent_from_sidecar")]
    if str(view.get(AGIBOT_STATUS)) != "ok" or not view.get(AGIBOT_ALIGNED):
        return [_defect("track_not_aligned", status=view.get(AGIBOT_STATUS))]
    aligned = view.get(AGIBOT_ROWS)
    if aligned is None or int(aligned) != rows:
        return [_defect("track_rows_disagree", stated=aligned, rows=rows)]
    matrix = rotation_of(view.get(AGIBOT_FRAME_ZERO))
    if matrix is None:
        return [_defect("pose_not_rotation", reason="shape")]
    if rotation_error(matrix) > tolerance:
        return [_defect("pose_not_rotation", error=rotation_error(matrix))]
    return []


def read_agibot_wrist(
    root: Path, calib: list[Path], directories: list[Path], document: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], Counter, dict[str, Path]]:
    """AgiBot's two wrist cameras, as the offset from the arm each is bolted to.

    One record per directory, episode and camera, holding
    R_end_effector<-camera rather than a pose in the base frame. The camera is
    rigid to the arm, measured: over 150,822 frames the offset wanders by at
    most 2.7e-06 degrees inside an episode, so the whole moving pose is this
    one matrix and a column the loader already reads.

    The offset is measured against the corpus's own state column rather than
    the vendor's proprioception file, because that column is what the loader
    composes with. The two hold the same numbers, checked bit for bit.
    """
    files = sidecars_named(calib, (AGIBOT_GEOMETRY,), "AgiBot's per-task geometry")
    counts: Counter = Counter()
    records: list[dict[str, Any]] = []
    for directory in directories:
        name = stage.relative(root, directory)
        cameras = reference_views(document, name, "agibot_wrist")
        layout = layout_table.resolve(document, name)
        episodes = episode_metadata(directory)
        found = {str(record[AGIBOT_KEY]): record
                 for record in sidecar_records(files, {str(episode[AGIBOT_KEY])
                                                       for episode in episodes})}
        for episode in episodes:
            key = str(episode[AGIBOT_KEY])
            index = int(episode["episode_index"])
            arms = first_row_rotations(directory, index, layout)
            for camera, view, slot in cameras:
                stated = (found.get(key) or {}).get(AGIBOT_VIEWS, {}).get(camera)
                defects = wrist_defect(stated, int(episode["length"]),
                                       args.rotation_tolerance)
                offset = None
                if not defects:
                    offset, defects = mount_offset(
                        rotation_of(stated[AGIBOT_FRAME_ZERO]),
                        arms.get(arm_slot_of(layout, view)),
                        args.rotation_tolerance,
                    )
                for defect in defects:
                    counts[f"{camera}.{defect['name']}"] += 1
                records.append(as_record(
                    dataset=name, episode=index, camera=camera, view=view,
                    slot=slot, source_key=key, rotation=offset,
                    rotation_key=MOUNT_FIELD,
                    motion=(stated or {}).get(AGIBOT_MOTION), defects=defects,
                ))
    counts["directories"] += len(directories)
    inputs = {stage.relative(root, path): path for path in files}
    return records, counts, inputs


def arm_slot_of(layout: layout_table.Layout, key: str) -> int:
    """The action slot of the arm one camera is mounted on, from the layout."""
    for view in layout.views:
        if view["key"] == key:
            return int(view["arm_slot"])
    raise KeyError(
        f"layout for {layout.directory} has no view {key!r}\n"
        f"expected it among {[view['key'] for view in layout.views]}\n"
        f"fix POSE_VIEWS in this tool or the layout file's views"
    )


def first_row_rotations(
    directory: Path, index: int, layout: layout_table.Layout
) -> dict[int, np.ndarray]:
    """Each arm's R_base<-end_effector at the episode's first row, by action slot.

    Read from the stored state column the loader reads, at the columns the
    layout established, so the offset measured against it is exact for the
    composition the loader makes rather than close to it.
    """
    stored = first_state_row(directory, index, layout.columns["state"])
    found: dict[int, np.ndarray] = {}
    for position, arm in enumerate(layout.state["arms"]):
        at = int(arm["rotation_at"])
        if str(arm["rotation_kind"]) != QUATERNION_KIND:
            raise ValueError(
                f"{layout.directory} arm {arm['name']} states rotation kind "
                f"{arm['rotation_kind']!r}\n"
                f"expected {QUATERNION_KIND!r}, the order this reader decodes\n"
                f"establish the arm's rotation encoding, or extend this reader"
            )
        found[int(arm.get("slot", position + 1))] = quaternion_matrix(
            stored[at:at + 4]
        )
    return found


def first_state_row(directory: Path, index: int, column: str) -> np.ndarray:
    """One episode's first stored state row, by the path the source states."""
    info = json.loads((directory / "meta" / "info.json").read_text(encoding="utf-8"))
    relative = info["data_path"].format(
        episode_chunk=index // int(info["chunks_size"]), episode_index=index
    )
    table = pq.read_table(directory / relative, columns=[column])
    return np.asarray(table.column(column)[0].as_py(), dtype=np.float64)


def quaternion_matrix(values: np.ndarray) -> np.ndarray:
    """One xyzw quaternion -> its rotation matrix, normalised first."""
    x, y, z, w = (np.asarray(values, dtype=np.float64)
                  / np.linalg.norm(np.asarray(values, dtype=np.float64)))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mount_offset(
    base_from_camera: np.ndarray | None,
    base_from_arm: np.ndarray | None,
    tolerance: float,
) -> tuple[np.ndarray | None, list[dict[str, Any]]]:
    """R_end_effector<-camera from the two poses at the episode's first frame."""
    if base_from_arm is None:
        return None, [_defect("arm_pose_missing")]
    offset = base_from_arm.T @ np.asarray(base_from_camera, dtype=np.float64)
    if rotation_error(offset) > tolerance:
        return None, [_defect("pose_not_rotation", error=rotation_error(offset))]
    return offset, []


READERS = {"agibot": read_agibot, "agibot_wrist": read_agibot_wrist}


def check_one_per_episode(records: list[dict[str, Any]]) -> None:
    """Raise when two records state a pose for the same episode and camera.

    The cache stage reads one record per directory, episode and camera, so a
    second would be taken or dropped by file order with nothing saying a choice
    was made.
    """
    seen: Counter = Counter(
        (record["dataset"], record["episode"], record["view"]) for record in records
    )
    repeated = sorted(key for key, count in seen.items() if count > 1)
    if not repeated:
        return
    raise ValueError(
        f"{len(repeated)} episodes have more than one record, first {repeated[:3]}\n"
        f"expected one record per directory, episode and camera\n"
        f"fix the join in the reader for this source"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, help="the one corpus root")
    parser.add_argument("--source", required=True, choices=sorted(READERS),
                        help="which reader to use")
    parser.add_argument("--calib", action="append", required=True,
                        help="a pose tree of that source, under the root")
    parser.add_argument("--dir", action="append", default=[], help="a directory to read")
    parser.add_argument("--dirs-file", help="a file of directories, one per line")
    parser.add_argument("--layouts", required=True, help="the layout file naming the views")
    parser.add_argument("--out", required=True,
                        help="where this source's extrinsics records go")
    parser.add_argument("--rotation-tolerance", type=float, default=ROTATION_TOLERANCE)
    parser.add_argument("--within-episode-tolerance", type=float,
                        default=WITHIN_EPISODE_DEGREES,
                        help="degrees the camera may turn inside one episode")
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
    records, counts, inputs = READERS[args.source](root, calib, directories, document, args)
    records.sort(key=lambda record: (record["dataset"], record["episode"]))
    check_one_per_episode(records)

    posed = sum(1 for record in records if record["pass"])
    head = stage.header(
        "build_extrinsics",
        root=root,
        inputs={
            **directory_inputs(root, directories, dirs_file),
            **inputs,
            # The layout names the reference camera each pose is written
            # against, so a record is only current for the layout it was
            # resolved through.
            "layouts": layouts_path,
        },
        tree=REPO,
        extra={"source": args.source, "directories": len(directories),
               "records": len(records), "poses": posed,
               "views": [key for _, key, _ in POSE_VIEWS[args.source]],
               "rotation_tolerance": args.rotation_tolerance,
               "within_episode_tolerance": args.within_episode_tolerance,
               "join": JOIN[args.source], "defect_names": list(DEFECTS),
               "counts": dict(sorted(counts.items()))},
    )
    written = stage.write_jsonl(out, head, records)
    logger.info("%s: %d directories, %d records, %d with a pose",
                args.source, len(directories), len(records), posed)
    for reason, count in sorted(counts.items()):
        logger.info("%s %d", reason, count)
    logger.info("%s %s", "wrote" if written else "unchanged", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
