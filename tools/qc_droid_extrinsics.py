"""Stage 7c: whether DROID's stored pose describes the camera the frames show.

No labels and no detector. The arm is the only moving thing in a DROID scene,
so a right pose puts the projected gripper on moving pixels and a stale one
leaves it on static background. The score is the motion energy at the projected
points over the motion energy of the whole frame.

One record per episode and its stored exterior camera: the score, the samples
behind it and a verdict of ok, fail or inconclusive. With --control the episode
is scored again under a pose recorded in another lab, which reads as wrong.

Usage:
    python tools/qc_droid_extrinsics.py --root /corpus --control \
        --calib calib/droid_extrinsics --dirs-file config/droid_dirs.txt \
        --urdf <a Franka Panda URDF you supply> --out prep/qc_droid.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import xml.etree.ElementTree as elementtree
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pyarrow.parquet as parquet
import stage
from scan_source import directory_inputs, episode_files, read_dir_list
from scipy.spatial.transform import Rotation

logger = logging.getLogger("qc_droid_extrinsics")

REPO = Path(__file__).resolve().parent.parent

# The kit this is ported from measured motion on a 640x360 map made from
# DROID's own 1280x720 stream. Our copy stores one 256x256 frame per row, so
# both of the kit's pixel constants are rescaled to the raster they land on
# rather than carried across as numbers.
KIT_MOTION_WIDTH = 640.0
KIT_MOTION_HEIGHT = 360.0
KIT_NATIVE_WIDTH = 1280.0
KIT_NATIVE_HEIGHT = 720.0
KIT_BLUR_SIGMA = 5.0
KIT_MIN_UV_SPEED = 1.5

# Below this mean absolute difference a frame holds no motion, and dividing the
# joint energy by it would report the ratio of two kinds of noise.
STATIC_FRAME_ENERGY = 0.2

# How many (frame, point) samples a score has to rest on. Fewer is not a low
# score but no reading at all: the arm was still, or out of view.
MIN_SAMPLES = 30

# Where the kit puts the line. A pose that describes the camera scores well
# above one and a pose that does not scores near it, so anything in this
# region is a wide gap rather than a tuned edge.
SCORE_THRESHOLD = 2.5

# How many stored frames the difference is taken across.
STRIDE = 2

# The points the score is sampled at. The kit samples the forearm and the
# fingers because base joints barely move; our copy publishes no joint angles,
# so the arm cannot be reconstructed above the wrist and these are the links
# the flange pose alone places. They are the fastest-moving ones either way.
HAND_LINKS = ("panda_hand", "panda_hand_tcp", "panda_leftfinger",
              "panda_rightfinger", "panda_leftfingertip", "panda_rightfingertip")

# The link the stored pose is the pose of. Checked by eye against the pixels:
# the hand links hang off this frame and land on the gripper, and reading the
# pose as the tool centre instead puts them a hand's length back up the wrist.
FLANGE_LINK = "panda_link8"

# The widest the fingers open, in metres, and the gripper reading that means
# open. The kit drives the two finger joints from the same pair.
FINGER_TRAVEL = 0.04

# Where in the stored state vector the flange pose and the gripper sit. The
# first six are DROID's cartesian_position verbatim, in metres and radians
# about fixed axes, which the pose sidecar states and validated against the
# vendor's own trajectory file.
POSE_COLUMNS = slice(0, 6)
GRIPPER_COLUMN = 6

# The stored camera column the score is about, and the roles the sidecar names
# the two exterior cameras by.
STORED_VIEW = "observation.images.image"
EXTERIOR_ROLES = ("ext1", "ext2")

# Sources that publish a pose the vendor measured. Everything else is an
# estimate or nothing, and neither is scored here: an estimate is refused by
# the build stage before it can reach a token.
VENDOR_SOURCES = ("droid_metadata_json", "karlp_verified_gt")

VERDICTS = ("ok", "fail", "inconclusive")

# Why a score could not be read. Recorded beside the verdict so a run's counts
# say whether the arm was invisible or the calibration was.
NO_CALIBRATION = "no stored calibration"
NO_SAMPLES = "arm still or out of view"

_chains: dict[str, dict[str, tuple]] = {}


def hand_chain(path: Path) -> dict[str, tuple]:
    """The URDF's joints, keyed by child link: parent, origin, type and axis."""
    root = elementtree.parse(path).getroot()
    chain: dict[str, tuple] = {}
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        translation = np.zeros(3)
        angles = np.zeros(3)
        if origin is not None:
            translation = np.array(
                [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()])
            angles = np.array(
                [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()])
        axis = joint.find("axis")
        chain[joint.find("child").attrib["link"]] = (
            joint.find("parent").attrib["link"], translation, angles,
            joint.attrib["type"],
            np.array([float(v) for v in axis.attrib["xyz"].split()])
            if axis is not None else None,
        )
    return chain


def hand_points(chain: dict[str, tuple], aperture: np.ndarray) -> np.ndarray:
    """The sampled links' positions in the flange frame, one row per step.

    Only the fixed joints and the two finger slides lie below the flange, so
    this needs no joint angles: the aperture is the whole configuration.
    """
    points = np.zeros((len(aperture), len(HAND_LINKS), 3))
    for column, link in enumerate(HAND_LINKS):
        steps = []
        current = link
        while current != FLANGE_LINK:
            parent, translation, angles, kind, axis = chain[current]
            steps.append((translation, angles, kind, axis))
            current = parent
        placed = np.broadcast_to(np.eye(4), (len(aperture), 4, 4)).copy()
        for translation, angles, kind, axis in reversed(steps):
            fixed = np.eye(4)
            fixed[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
            fixed[:3, 3] = translation
            placed = placed @ fixed
            if kind == "prismatic":
                slide = np.broadcast_to(np.eye(4), (len(aperture), 4, 4)).copy()
                slide[:, :3, 3] = axis[None] * aperture[:, None]
                placed = placed @ slide
        points[:, column] = placed[:, :3, 3]
    return points


def project(
    state: np.ndarray, chain: dict[str, tuple], camera_from_base: np.ndarray,
    calibration: dict[str, Any], size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """The sampled links in stored pixels, and which of them the camera sees."""
    aperture = (1.0 - np.clip(state[:, GRIPPER_COLUMN], 0.0, 1.0)) * FINGER_TRAVEL
    rotation = Rotation.from_euler("xyz", state[:, 3:6]).as_matrix()
    local = hand_points(chain, aperture)
    world = np.einsum("tij,tkj->tki", rotation, local) + state[:, None, 0:3]
    camera = np.einsum(
        "ij,tkj->tki", camera_from_base[:3, :3], world) + camera_from_base[:3, 3]
    depth = camera[..., 2]
    uv = np.stack([
        calibration["fx"] * camera[..., 0] / depth + calibration["cx"],
        calibration["fy"] * camera[..., 1] / depth + calibration["cy"],
    ], -1)
    width, height = size
    seen = ((depth > 0.01) & (uv[..., 0] >= 0) & (uv[..., 0] < width)
            & (uv[..., 1] >= 0) & (uv[..., 1] < height))
    return uv, seen


def qc_camera(
    frames: list[np.ndarray], uv_all: np.ndarray, ok_all: np.ndarray, stride: int,
    *, size: tuple[int, int], sigma: float, min_uv_speed: float,
) -> tuple[float | None, int, int]:
    """The kit's score for one camera: joint motion energy over frame energy.

    Three things differ from the kit and nothing else does. The frames arrive
    as encoded bytes because our copy stores a picture per row and has no video
    file to open. The motion map is the stored raster rather than half of it,
    since 256 pixels do not survive another halving, so the projected points
    index it directly. The blur width and the speed floor are passed in, having
    been rescaled from the raster the kit measured them on.
    """
    width, height = size
    previous = None
    joint_energy: list[float] = []
    frame_energy: list[float] = []
    used, moving = 0, 0
    for index in range(min(len(uv_all), len(frames))):
        if index % stride:
            continue
        gray = cv2.cvtColor(
            cv2.imdecode(frames[index], cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
        if previous is not None:
            motion = cv2.GaussianBlur(cv2.absdiff(gray, previous), (0, 0), sigma)
            whole = float(motion.mean())
            if whole > STATIC_FRAME_ENERGY:
                moving += 1
                step = uv_all[index] - uv_all[index - stride]
                for point in range(uv_all.shape[1]):
                    if not (ok_all[index, point] and ok_all[index - stride, point]):
                        continue
                    if np.linalg.norm(step[point]) < min_uv_speed:
                        continue
                    column = int(np.clip(uv_all[index, point, 0], 0, width - 1))
                    row = int(np.clip(uv_all[index, point, 1], 0, height - 1))
                    joint_energy.append(float(motion[row, column]))
                    frame_energy.append(whole)
                    used += 1
        previous = gray
    if used < MIN_SAMPLES:
        return None, used, moving
    return float(np.mean(joint_energy) / np.mean(frame_energy)), used, moving


def verdict_of(score: float | None, threshold: float) -> str:
    """What one score says: nothing, a camera that is there, or one that is not."""
    if score is None:
        return "inconclusive"
    return "ok" if score >= threshold else "fail"


def scaled_constants(size: tuple[int, int]) -> tuple[float, float]:
    """The kit's blur width and speed floor, rescaled to our stored raster.

    Both are lengths in pixels, so each is scaled by the geometric mean of the
    two axis ratios: the stored frame is a squashed 16:9 view and a single
    ratio would be wrong on one axis.
    """
    width, height = size
    blur = KIT_BLUR_SIGMA * math.sqrt(
        (width / KIT_MOTION_WIDTH) * (height / KIT_MOTION_HEIGHT))
    speed = KIT_MIN_UV_SPEED * math.sqrt(
        (width / KIT_NATIVE_WIDTH) * (height / KIT_NATIVE_HEIGHT))
    return blur, speed


def stored_size(directory: Path) -> tuple[int, int]:
    """The width and height of the stored exterior frames, from info.json."""
    info = json.loads((directory / "meta" / "info.json").read_text(encoding="utf-8"))
    shape = info["features"][STORED_VIEW]["shape"]
    return int(shape[1]), int(shape[0])


def pose_records(calib: Path, name: str) -> list[dict[str, Any]]:
    """One shard's pose sidecar, in file order."""
    path = calib / name / "meta" / "camera_poses.jsonl"
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def calibration_of(record: dict[str, Any], size: tuple[int, int]) -> dict[str, Any] | None:
    """The stored-raster calibration, refused when it describes another raster."""
    stored = record.get("intrinsics_stored_256x256")
    if stored is None:
        return None
    if (int(stored["width"]), int(stored["height"])) != size:
        raise ValueError(
            f"calibration is for {stored['width']}x{stored['height']}, "
            f"frames are {size[0]}x{size[1]}\n"
            f"expected a calibration measured against the stored frames\n"
            f"rebuild the pose sidecar against this copy of the corpus"
        )
    return stored


def work_items(
    root: Path, directories: list[Path], calib: Path, urdf: Path, sample: int,
) -> list[dict[str, Any]]:
    """One item per vendor-posed episode, with the pose and the frames to read.

    The sidecar joins on the episode index it is keyed by, and the frame count
    both sides state confirms the join before an item is made, so a shard read
    out of order stops here rather than scoring one room's arm in another's.
    """
    items: list[dict[str, Any]] = []
    for directory in directories:
        name = stage.relative(root, directory)
        size = stored_size(directory)
        parquets = episode_files(directory, "data", ".parquet")
        episodes = {}
        with open(directory / "meta" / "episodes.jsonl", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    episode = json.loads(line)
                    episodes[int(episode["episode_index"])] = int(episode["length"])
        for record in pose_records(calib, Path(name).name):
            index = int(record["episode_index"])
            rows = episodes.get(index)
            if rows is None or rows != int(record["n_frames"]):
                raise ValueError(
                    f"{name} episode {index} has {rows} rows, sidecar says "
                    f"{record['n_frames']}\n"
                    f"expected the two to agree, which is what confirms the join\n"
                    f"rebuild the sidecar from the matching source recording"
                )
            if record["calib_source"] not in VENDOR_SOURCES:
                continue
            items.append({
                "dataset": name, "episode": index, "parquet": str(parquets[index]),
                "camera": record["stored_exterior"], "serial": record["stored_cam_serial"],
                "lab": record["lab"], "calib_source": record["calib_source"],
                "size": size, "urdf": str(urdf),
                "camera_from_base": record["T_cam_from_base_4x4"],
                "calibration": calibration_of(record, size),
            })
    items.sort(key=lambda item: (item["dataset"], item["episode"]))
    if sample:
        step = max(1, len(items) // sample)
        items = items[::step][:sample]
    return items


def add_controls(items: list[dict[str, Any]]) -> None:
    """Give each item a pose recorded in another lab, as the wrong-pose reading.

    Half the list away and then forward to the first different lab, so the
    control is a camera in another room rather than the same camera under
    another name: neighbouring episodes of one shard often share a session.
    """
    posed = [item for item in items if item["calibration"] is not None]
    for position, item in enumerate(posed):
        offset = (position + len(posed) // 2) % len(posed)
        while posed[offset]["lab"] == item["lab"]:
            offset = (offset + 1) % len(posed)
            if offset == position:
                raise ValueError(
                    f"every posed episode is from lab {item['lab']!r}\n"
                    f"expected at least two labs, so a control can be wrong\n"
                    f"pass more shards, or run without --control"
                )
        item["control_from_base"] = posed[offset]["camera_from_base"]
        item["control_lab"] = posed[offset]["lab"]
        item["control_episode"] = posed[offset]["episode"]


def chain_for(path: str) -> dict[str, tuple]:
    """The parsed URDF for this worker, parsed once per process."""
    if path not in _chains:
        _chains[path] = hand_chain(Path(path))
    return _chains[path]


def score_item(item: dict[str, Any]) -> dict[str, Any]:
    """One episode's record: its score, its control's score and both verdicts.

    One thread per worker: the frames are small and a pool of processes that
    each open their own thread pool spends its time in the scheduler.
    """
    cv2.setNumThreads(1)
    record: dict[str, Any] = {
        "dataset": item["dataset"], "episode": item["episode"],
        "view": STORED_VIEW, "camera": item["camera"], "serial": item["serial"],
        "lab": item["lab"], "calib_source": item["calib_source"],
        "score": None, "samples": 0, "moving_frames": 0,
        "verdict": "inconclusive", "reason": NO_CALIBRATION,
        "control_score": None, "control_samples": 0,
        "control_verdict": None, "control_lab": None, "control_episode": None,
    }
    if item["calibration"] is None:
        return record

    table = parquet.ParquetFile(item["parquet"]).read(
        columns=["observation.state", STORED_VIEW])
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    frames = [np.frombuffer(value.as_py()["bytes"], dtype=np.uint8)
              for value in table[STORED_VIEW]]
    size = (int(item["size"][0]), int(item["size"][1]))
    sigma, min_uv_speed = scaled_constants(size)
    chain = chain_for(item["urdf"])

    poses = [("", np.asarray(item["camera_from_base"], dtype=np.float64))]
    if "control_from_base" in item:
        poses.append(("control_", np.asarray(item["control_from_base"], dtype=np.float64)))
    for prefix, pose in poses:
        uv, seen = project(state, chain, pose, item["calibration"], size)
        score, used, moving = qc_camera(
            frames, uv, seen, STRIDE, size=size, sigma=sigma, min_uv_speed=min_uv_speed)
        record[f"{prefix}score"] = score
        record[f"{prefix}samples"] = used
        record[f"{prefix}verdict"] = verdict_of(score, SCORE_THRESHOLD)
        if not prefix:
            record["moving_frames"] = moving
            record["reason"] = None if score is not None else NO_SAMPLES
    if "control_from_base" in item:
        record["control_lab"] = item["control_lab"]
        record["control_episode"] = item["control_episode"]
    return record


def scored(items: list[dict[str, Any]], workers: int) -> Iterable[dict[str, Any]]:
    """Every item's record, run here or across a pool of processes."""
    if workers <= 1:
        return [score_item(item) for item in items]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(score_item, items, chunksize=8))


def distribution(scores: list[float]) -> dict[str, float]:
    """The percentiles a reader needs to see where two populations sit."""
    if not scores:
        return {}
    points = [1, 5, 10, 25, 50, 75, 90, 99]
    values = np.percentile(np.asarray(scores), points)
    return {f"p{point}": round(float(value), 3) for point, value in zip(points, values)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, help="the one corpus root")
    parser.add_argument("--calib", required=True,
                        help="the pose tree, one directory per shard, under the root")
    parser.add_argument("--dir", action="append", default=[], help="a directory to read")
    parser.add_argument("--dirs-file", help="a file of directories, one per line")
    parser.add_argument("--urdf", required=True, help="the arm model the points come from")
    parser.add_argument("--out", required=True, help="where the scores go")
    parser.add_argument("--control", action="store_true",
                        help="also score each episode under another lab's pose")
    parser.add_argument("--sample", type=int, default=0,
                        help="score this many episodes, spread evenly over the shards")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=SCORE_THRESHOLD)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    root = stage.resolve_root(args.root)
    dirs_file = stage.under_root(root, args.dirs_file) if args.dirs_file else None
    directories = read_dir_list(root, args.dir, dirs_file)
    calib = stage.under_root(root, args.calib)
    urdf = Path(args.urdf).resolve()
    out = stage.under_root(root, args.out)
    stage.refuse_write_into_input(out, [*directories, calib])

    items = work_items(root, directories, calib, urdf, args.sample)
    if args.control:
        add_controls(items)
    records = list(scored(items, args.workers))
    records.sort(key=lambda record: (record["dataset"], record["episode"]))

    counts: Counter = Counter(record["verdict"] for record in records)
    control_counts: Counter = Counter(
        record["control_verdict"] for record in records
        if record["control_verdict"] is not None)
    measured = [record["score"] for record in records if record["score"] is not None]
    controls = [record["control_score"] for record in records
                if record["control_score"] is not None]
    head = stage.header(
        "qc_droid_extrinsics",
        root=root,
        inputs={**directory_inputs(root, directories, dirs_file),
                stage.relative(root, calib): calib},
        tree=REPO,
        extra={"directories": len(directories), "episodes": len(records),
               "threshold": args.threshold, "stride": STRIDE,
               "min_samples": MIN_SAMPLES, "blur_sigma_and_speed_are_rescaled": True,
               "urdf": str(urdf), "urdf_sha256": stage.sha256_of(urdf),
               "verdicts": dict(sorted(counts.items())),
               "control_verdicts": dict(sorted(control_counts.items())),
               "score_percentiles": distribution(measured),
               "control_percentiles": distribution(controls)},
    )
    written = stage.write_jsonl(out, head, records)
    logger.info("%d episodes scored over %d directories", len(records), len(directories))
    for verdict in VERDICTS:
        logger.info("%s %d", verdict, counts[verdict])
    if controls:
        logger.info("control median %.2f against measured median %.2f",
                    float(np.median(controls)), float(np.median(measured)))
    logger.info("%s %s", "wrote" if written else "unchanged", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
