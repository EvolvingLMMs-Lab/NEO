"""Stage 7d: whether a source's stored pose describes the camera its frames show.

No labels and no detector. The arm is the only thing that moves through a
scene, so a right pose puts the projected arm points on moving pixels and a
wrong one leaves them on static background. The score is the motion energy at
the projected points over the motion energy of the whole frame.

One record per episode and its reference camera: the score, the samples behind
it and a verdict of ok, fail or inconclusive. With --control the episode is
scored again under a pose recorded elsewhere, which reads as wrong.

Usage:
    python tools/qc_extrinsics.py --root /corpus --source agibot --control \
        --calib calib/agibot_extrinsics --intrinsics metadata/C/intrinsics_agibot.jsonl \
        --layouts config/layouts/action_pretrain.json --out metadata/C/qc_extrinsics_agibot.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Iterator

import av
import cv2
import layouts as layout_table
import numpy as np
import pyarrow.parquet as parquet
import stage
from scan_source import directory_inputs, episode_files, read_dir_list, video_files

logger = logging.getLogger("qc_extrinsics")

REPO = Path(__file__).resolve().parent.parent

# The scoring core is carried over from an earlier motion-energy QC script
# (the "kit" below), which measured it over 36,685 DROID episodes on their
# native video. It is copied here so the corpus tools depend on nothing
# outside this repository. The ratio, the blur, the
# static-frame floor and the speed floor are the kit's numbers unchanged.
#
# The kit measures motion on half of a 1280 by 720 frame and samples a point of
# that 640 by 360 map. AgiBot's stored frame is 640 by 480, so the map is the
# stored frame itself: halving it again would spread the blur over twice the
# share of the scene the kit spreads it over. Dropping the halving is what
# keeps both pixel constants meaning here what they mean on the kit's raster.
KIT_BLUR_SIGMA = 5.0
KIT_MIN_UV_SPEED = 1.5
STATIC_FRAME_ENERGY = 0.2
MIN_SAMPLES = 30
SCORE_THRESHOLD = 2.5
STRIDE = 2

VERDICTS = ("ok", "fail", "inconclusive")

# How far apart two poses have to be before one can stand as the other's
# wrong-pose control. Another directory is not enough on its own: AgiBot's head
# poses cluster, and a quarter of pairs drawn from two task directories sit
# within five degrees of each other. Ten degrees moves a projected point by
# about 57 pixels at this corpus's focal length, at any depth, which is more
# than ten times the blur the score reads through.
MIN_CONTROL_ROTATION_DEGREES = 10.0

# Why a score could not be read, recorded beside the verdict so a run's counts
# say whether the arm was invisible or the calibration was.
NO_CALIBRATION = "no stored calibration"
NO_POSE = "no stored pose"
NO_VIDEO = "no video for the reference camera"
NO_SAMPLES = "arm still or out of view"

# What each unit a state layout may state its positions in is in metres. A unit
# the table does not hold raises: a factor guessed here scales every point.
METRES_PER_UNIT = {"m": 1.0, "cm": 0.01, "mm": 0.001}

# The camera role each source's stored pose is the pose of. AgiBot's head is
# the only view the layout gives that role; the two wrist views are not a
# reference and no pose is stated for them.
REFERENCE_ROLE = {"agibot": "head"}

# AgiBot's pose sidecar, one record per episode, and the fields the join and
# the pose come from. The prepared extrinsics file carries the rotation of this
# same pose and no translation, and a point cannot be projected without one.
AGIBOT_SIDECAR = "head_extrinsic_sidecar.jsonl"
AGIBOT_KEY = "raw_file_name"
AGIBOT_POSE = "T_base_from_cam_4x4"
AGIBOT_SOURCE = "calib_source"


def motion_scores(
    frames: Iterable[tuple[int, np.ndarray]],
    projections: list[tuple[np.ndarray, np.ndarray]],
    stride: int,
    *,
    size: tuple[int, int],
    sigma: float,
    min_uv_speed: float,
) -> tuple[list[tuple[float | None, int]], int]:
    """The motion-energy score of each projection of one camera, in one pass.

    frames yields the stored frames the difference is taken across, as gray
    planes with the index each one has in the video. Everything the score rests
    on is the kit's: the blurred absolute difference, the whole-frame floor
    below which a pair of frames holds no motion, the speed floor a point has
    to clear in the image to be sampled, and the ratio of the two means.

    Several projections are scored together so that a pose and its control read
    one decode of the video rather than two. They share the motion map, which
    is what makes the two scores comparable per episode.
    """
    width, height = size
    sampled: list[tuple[list[float], list[float]]] = [([], []) for _ in projections]
    previous = None
    moving = 0
    for index, gray in frames:
        if previous is not None:
            motion = cv2.GaussianBlur(cv2.absdiff(gray, previous), (0, 0), sigma)
            whole = float(motion.mean())
            if whole > STATIC_FRAME_ENERGY:
                moving += 1
                for slot, (uv, seen) in enumerate(projections):
                    step = uv[index] - uv[index - stride]
                    for point in range(uv.shape[1]):
                        if not (seen[index, point] and seen[index - stride, point]):
                            continue
                        if float(np.linalg.norm(step[point])) < min_uv_speed:
                            continue
                        column = int(np.clip(uv[index, point, 0], 0, width - 1))
                        row = int(np.clip(uv[index, point, 1], 0, height - 1))
                        sampled[slot][0].append(float(motion[row, column]))
                        sampled[slot][1].append(whole)
        previous = gray
    results = []
    for point_energy, whole_energy in sampled:
        used = len(point_energy)
        if used < MIN_SAMPLES:
            results.append((None, used))
            continue
        results.append((float(np.mean(point_energy) / np.mean(whole_energy)), used))
    return results, moving


def gray_frames(path: Path, rows: int, stride: int) -> Iterator[tuple[int, np.ndarray]]:
    """Every stride-th stored frame of one video, as a gray plane, in order.

    One decode thread: the caller runs a process per episode, and a decoder
    that opens its own pool inside each of them spends its time scheduling.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        for index, frame in enumerate(container.decode(video=0)):
            if index >= rows:
                break
            if index % stride:
                continue
            yield index, frame.to_ndarray(format="gray")


def project(
    points: np.ndarray, camera_from_base: np.ndarray, calibration: dict[str, Any],
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """The sampled points in stored pixels, and which of them the camera sees.

    A pinhole projection, as the kit's is. A record's distortion model is not
    applied: nobody established whether the delivered frames are the raw stream
    or a rectified one, so undistorting could move a point that was already in
    place. On AgiBot it would move a sampled point by three to five pixels,
    which is inside the blur the score reads through.
    """
    camera = np.einsum(
        "ij,tkj->tki", camera_from_base[:3, :3], points) + camera_from_base[:3, 3]
    depth = camera[..., 2]
    uv = np.stack([
        calibration["fx"] * camera[..., 0] / depth + calibration["cx"],
        calibration["fy"] * camera[..., 1] / depth + calibration["cy"],
    ], -1)
    width, height = size
    seen = ((depth > 0.01) & (uv[..., 0] >= 0) & (uv[..., 0] < width)
            & (uv[..., 1] >= 0) & (uv[..., 1] < height))
    return uv, seen


def state_points(state: np.ndarray, spec: list[dict[str, Any]]) -> np.ndarray:
    """The stored state -> the points to sample, in metres, one row per step.

    The kit samples the forearm and the fingers of a reconstructed arm, and
    leaves the base joints out because they barely move and dilute the ratio.
    Our conversion of AgiBot publishes end-effector poses and no joint angles,
    so the arm cannot be reconstructed at all; the end effector is the most
    distal point there is, which is the point the kit's rule asks for.
    """
    return np.stack(
        [state[:, arm["at"]:arm["at"] + 3] * arm["metres"] for arm in spec], axis=1)


POINTS = {"agibot": state_points}


def reference_view(layout: layout_table.Layout, source: str, directory: str) -> tuple[str, int]:
    """The layout view this source's stored pose is the pose of, and its slot."""
    role = REFERENCE_ROLE[source]
    found = [view for view in layout.views if view.get("role") == role]
    if len(found) != 1:
        raise ValueError(
            f"{directory} has {len(found)} views with role {role!r}\n"
            f"expected one, the camera the stored pose describes\n"
            f"fix the views of this source in the layout file"
        )
    return str(found[0]["key"]), int(found[0]["slot"])


def points_spec(layout: layout_table.Layout, directory: str) -> tuple[list[dict[str, Any]], int]:
    """Which state columns hold a point, what puts it in metres, and the width."""
    state = layout.state
    if state is None:
        raise ValueError(
            f"{directory} states no state layout\n"
            f"expected the columns the arm positions sit in\n"
            f"write the source's state block before scoring it"
        )
    spec = []
    for arm in state["arms"]:
        unit = str(arm["position_unit"])
        if unit not in METRES_PER_UNIT:
            raise ValueError(
                f"{directory} states arm positions in {unit!r}\n"
                f"expected one of {sorted(METRES_PER_UNIT)}\n"
                f"add the factor to METRES_PER_UNIT, or fix the layout"
            )
        spec.append({"name": str(arm["name"]), "at": int(arm["position_at"]),
                     "metres": METRES_PER_UNIT[unit]})
    return spec, int(state["width"])


def rotation_degrees(first: np.ndarray, second: np.ndarray) -> float:
    """The angle between the rotations of two poses, in degrees."""
    product = np.asarray(first)[:3, :3].T @ np.asarray(second)[:3, :3]
    cosine = (float(np.trace(product)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def stage_records(path: Path) -> Iterator[dict[str, Any]]:
    """The records of a jsonl stage output, streamed rather than held.

    The calibration file carries one record per camera of every episode, and
    holding a source's whole file as parsed objects costs more memory than the
    scores it is read for.
    """
    with open(path, encoding="utf-8") as handle:
        first = handle.readline()
        if not first or "provenance" not in json.loads(first):
            raise ValueError(
                f"{path} does not start with a provenance header\n"
                f"expected the header every stage output writes on line 1\n"
                f"re-run the stage that writes this file"
            )
        for line in handle:
            if line.strip():
                yield json.loads(line)


def calibrations(path: Path, slot: int, wanted: set[str]) -> dict[tuple[str, int], dict[str, Any]]:
    """The reference camera's calibration, per directory and episode."""
    found: dict[tuple[str, int], dict[str, Any]] = {}
    for record in stage_records(path):
        if int(record["slot"]) != slot or record["dataset"] not in wanted:
            continue
        found[(record["dataset"], int(record["episode"]))] = {
            "fx": float(record["fx"]), "fy": float(record["fy"]),
            "cx": float(record["cx"]), "cy": float(record["cy"]),
            "width": int(record["width"]), "height": int(record["height"]),
        }
    return found


def agibot_sidecars(calib: Path) -> tuple[dict[str, dict[str, Any]], list[Path]]:
    """Every pose the calibration tree states, keyed by the sidecar's join key.

    The files are returned alongside the poses because every one of them is an
    input of the run. They all carry the same file name, so they can only be
    told apart by their path.
    """
    files = sorted(calib.rglob(AGIBOT_SIDECAR))
    if not files:
        raise FileNotFoundError(
            f"no {AGIBOT_SIDECAR} under {calib}\n"
            f"expected AgiBot's per-task pose sidecars\n"
            f"pass --calib the directory holding them"
        )
    poses: dict[str, dict[str, Any]] = {}
    for path in files:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                key = str(record[AGIBOT_KEY])
                if key in poses:
                    raise ValueError(
                        f"two sidecar records name episode {key}\n"
                        f"expected one pose per episode; nothing says which\n"
                        f"remove the duplicate from the calibration tree"
                    )
                poses[key] = record
    return poses, files


def read_agibot(
    root: Path, calib: Path, intrinsics: Path, directories: list[Path],
    document: dict[str, Any],
) -> tuple[list[dict[str, Any]], Counter, dict[str, Path]]:
    """AgiBot's episodes, each with the pose and the calibration it will use.

    The sidecar is joined on the raw file name both sides carry, and the frame
    count both sides state confirms the join before an item is made, so a
    directory read against the wrong sidecar stops here rather than scoring one
    scene's arm against another's pose.
    """
    poses, files = agibot_sidecars(calib)
    names = [stage.relative(root, directory) for directory in directories]
    counts: Counter = Counter()
    items: list[dict[str, Any]] = []
    references = set()
    for directory, name in zip(directories, names):
        layout = layout_table.resolve(document, name)
        view, slot = reference_view(layout, "agibot", name)
        references.add((view, slot))
        spec, width = points_spec(layout, name)
        parquets = episode_files(directory, "data", ".parquet")
        videos = video_files(directory, [view])[view]
        with open(directory / "meta" / "episodes.jsonl", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                episode = json.loads(line)
                index = int(episode["episode_index"])
                key = str(episode[AGIBOT_KEY])
                sidecar = poses.get(key)
                if sidecar is None:
                    counts["episode_absent_from_sidecar"] += 1
                    continue
                if sidecar["status"] != "ok":
                    counts[f"episode_status_{sidecar['status']}"] += 1
                    continue
                if int(sidecar["length"]) != int(episode["length"]):
                    raise ValueError(
                        f"{name} episode {index} has {episode['length']} rows, "
                        f"sidecar {key} says {sidecar['length']}\n"
                        f"expected the two to agree, which confirms the join\n"
                        f"re-stage the pose tree against this copy of the corpus"
                    )
                items.append({
                    "source": "agibot", "dataset": name, "episode": index,
                    "camera": REFERENCE_ROLE["agibot"], "view": view,
                    "source_key": key, "calib_source": sidecar.get(AGIBOT_SOURCE),
                    "pose": sidecar[AGIBOT_POSE], "spec": spec, "state_width": width,
                    "column": layout.columns["state"],
                    "parquet": str(parquets[index]) if index in parquets else None,
                    "video": str(videos[index]) if index in videos else None,
                })
    if len(references) != 1:
        raise ValueError(
            f"the directories name {len(references)} reference cameras\n"
            f"expected one view and slot across a run, so one file is read\n"
            f"pass directories that share a reference camera"
        )
    stored = calibrations(intrinsics, references.pop()[1], set(names))
    for item in items:
        item["calibration"] = stored.get((item["dataset"], item["episode"]))
        if item["calibration"] is None:
            counts["episode_without_calibration"] += 1
    inputs = {stage.relative(root, path): path for path in [*files, intrinsics]}
    return items, counts, inputs


READERS = {"agibot": read_agibot}


def add_controls(items: list[dict[str, Any]]) -> None:
    """Give each item another directory's pose, as the wrong-pose reading.

    Half the list away and then forward to the first pose recorded in another
    directory that also points somewhere else. Both conditions are needed: one
    directory's episodes share a session, and two directories of this corpus
    often share a head pose, so a different directory alone would compare a
    pose against itself and call the agreement a control.
    """
    posed = [item for item in items
             if item["calibration"] is not None and item["video"] is not None]
    if not posed:
        return
    matrices = [np.asarray(item["pose"], dtype=np.float64) for item in posed]
    for position, item in enumerate(posed):
        offset = (position + len(posed) // 2) % len(posed)
        start = offset
        while (posed[offset]["dataset"] == item["dataset"]
               or rotation_degrees(matrices[position], matrices[offset])
               < MIN_CONTROL_ROTATION_DEGREES):
            offset = (offset + 1) % len(posed)
            if offset == start:
                raise ValueError(
                    f"no pose is {MIN_CONTROL_ROTATION_DEGREES} degrees from "
                    f"{item['dataset']} episode {item['episode']}\n"
                    f"expected one, so the control reads as a wrong pose\n"
                    f"pass more directories, or run without --control"
                )
        item["control_pose"] = posed[offset]["pose"]
        item["control_dataset"] = posed[offset]["dataset"]
        item["control_episode"] = posed[offset]["episode"]
        item["control_rotation_deg"] = round(
            rotation_degrees(matrices[position], matrices[offset]), 3)


def verdict_of(score: float | None, threshold: float) -> str:
    """What one score says: nothing, a camera that is there, or one that is not."""
    if score is None:
        return "inconclusive"
    return "ok" if score >= threshold else "fail"


def read_state(item: dict[str, Any]) -> np.ndarray:
    """One episode's stored state, checked against the width the layout states."""
    table = parquet.ParquetFile(item["parquet"]).read(columns=[item["column"]])
    state = np.asarray(table[item["column"]].to_pylist(), dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != item["state_width"]:
        raise ValueError(
            f"{item['dataset']} episode {item['episode']} state is "
            f"{tuple(state.shape)}\n"
            f"expected a width of {item['state_width']}, as the layout states\n"
            f"fix the layout's state block, or the conversion that wrote it"
        )
    return state


def score_item(item: dict[str, Any]) -> dict[str, Any]:
    """One episode's record: its score, its control's score and both verdicts."""
    cv2.setNumThreads(1)
    record: dict[str, Any] = {
        "dataset": item["dataset"], "episode": item["episode"],
        "view": item["view"], "camera": item["camera"],
        "source_key": item["source_key"], "calib_source": item["calib_source"],
        "score": None, "samples": 0, "moving_frames": 0,
        "verdict": "inconclusive", "reason": NO_CALIBRATION,
        "control_score": None, "control_samples": 0, "control_verdict": None,
        "control_dataset": None, "control_episode": None,
        "control_rotation_deg": item.get("control_rotation_deg"),
    }
    if item["pose"] is None:
        record["reason"] = NO_POSE
        return record
    if item["calibration"] is None:
        return record
    if item["video"] is None or item["parquet"] is None:
        record["reason"] = NO_VIDEO
        return record

    state = read_state(item)
    points = POINTS[item["source"]](state, item["spec"])
    calibration = item["calibration"]
    size = (calibration["width"], calibration["height"])
    poses = [np.asarray(item["pose"], dtype=np.float64)]
    if "control_pose" in item:
        poses.append(np.asarray(item["control_pose"], dtype=np.float64))
    projections = [project(points, np.linalg.inv(pose), calibration, size)
                   for pose in poses]

    frames = gray_frames(Path(item["video"]), len(state), STRIDE)
    results, moving = motion_scores(
        frames, projections, STRIDE, size=size, sigma=KIT_BLUR_SIGMA,
        min_uv_speed=KIT_MIN_UV_SPEED)
    score, used = results[0]
    record["score"] = score
    record["samples"] = used
    record["moving_frames"] = moving
    record["verdict"] = verdict_of(score, SCORE_THRESHOLD)
    record["reason"] = None if score is not None else NO_SAMPLES
    if len(results) > 1:
        control, control_used = results[1]
        record["control_score"] = control
        record["control_samples"] = control_used
        record["control_verdict"] = verdict_of(control, SCORE_THRESHOLD)
        record["control_dataset"] = item["control_dataset"]
        record["control_episode"] = item["control_episode"]
    return record


def scored(items: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    """Every item's record, run here or across a pool of processes."""
    if workers <= 1:
        return [score_item(item) for item in items]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(score_item, items, chunksize=4))


def distribution(scores: list[float]) -> dict[str, float]:
    """The percentiles a reader needs to see where two populations sit."""
    if not scores:
        return {}
    points = [5, 25, 50, 75, 95]
    values = np.percentile(np.asarray(scores), points)
    return {f"p{point}": round(float(value), 3) for point, value in zip(points, values)}


def win_rate(records: list[dict[str, Any]]) -> float | None:
    """The share of episodes whose own pose outscores its own control.

    The statistic the whole check rests on. Two medians say the populations
    differ; this says the pose beats the wrong pose on the same frames, which
    is what a per-episode verdict needs to be worth anything.
    """
    pairs = [(record["score"], record["control_score"]) for record in records
             if record["score"] is not None and record["control_score"] is not None]
    if not pairs:
        return None
    return round(sum(1 for own, other in pairs if own > other) / len(pairs), 4)


def spread(items: list[dict[str, Any]], sample: int) -> list[dict[str, Any]]:
    """The items to score: all of them, or an even stride through the sorted list."""
    if not sample or sample >= len(items):
        return items
    step = max(1, len(items) // sample)
    return items[::step][:sample]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, help="the one corpus root")
    parser.add_argument("--source", required=True, choices=sorted(READERS),
                        help="which reader to use")
    parser.add_argument("--calib", required=True,
                        help="the pose tree of that source, under the root")
    parser.add_argument("--intrinsics", required=True,
                        help="the calibration records the points are projected through")
    parser.add_argument("--dir", action="append", default=[], help="a directory to read")
    parser.add_argument("--dirs-file", help="a file of directories, one per line")
    parser.add_argument("--layouts", required=True, help="the layout file naming the views")
    parser.add_argument("--out", required=True, help="where the scores go")
    parser.add_argument("--control", action="store_true",
                        help="also score each episode under another directory's pose")
    parser.add_argument("--sample", type=int, default=0,
                        help="score this many episodes, spread evenly over the directories")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=SCORE_THRESHOLD)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    root = stage.resolve_root(args.root)
    dirs_file = stage.under_root(root, args.dirs_file) if args.dirs_file else None
    directories = read_dir_list(root, args.dir, dirs_file)
    calib = stage.under_root(root, args.calib)
    intrinsics = stage.under_root(root, args.intrinsics)
    out = stage.under_root(root, args.out)
    stage.refuse_write_into_input(out, [*directories, calib])

    layouts_path = stage.under_root(root, args.layouts)
    document = layout_table.load(layouts_path)
    read, counts, inputs = READERS[args.source](
        root, calib, intrinsics, directories, document)
    read.sort(key=lambda item: (item["dataset"], item["episode"]))
    # The header states both counts. A sampled run scores a part of what it
    # read, and a file that said only the one number would read as a census.
    items = spread(read, args.sample)
    if args.control:
        add_controls(items)
    records = scored(items, args.workers)
    records.sort(key=lambda record: (record["dataset"], record["episode"]))

    verdicts: Counter = Counter(record["verdict"] for record in records)
    control_verdicts: Counter = Counter(
        record["control_verdict"] for record in records
        if record["control_verdict"] is not None)
    measured = [record["score"] for record in records if record["score"] is not None]
    controls = [record["control_score"] for record in records
                if record["control_score"] is not None]
    beats = win_rate(records)
    head = stage.header(
        "qc_extrinsics",
        root=root,
        inputs={**directory_inputs(root, directories, dirs_file), **inputs,
                "layouts": layouts_path},
        tree=REPO,
        extra={"source": args.source, "directories": len(directories),
               "episodes": len(records), "episodes_read": len(read),
               "sample": args.sample, "threshold": args.threshold,
               "stride": STRIDE, "min_samples": MIN_SAMPLES,
               "blur_sigma": KIT_BLUR_SIGMA, "min_uv_speed": KIT_MIN_UV_SPEED,
               "control_rotation_floor_deg": MIN_CONTROL_ROTATION_DEGREES,
               "verdicts": dict(sorted(verdicts.items())),
               "control_verdicts": dict(sorted(control_verdicts.items())),
               "score_percentiles": distribution(measured),
               "control_percentiles": distribution(controls),
               "win_rate": beats,
               "counts": dict(sorted(counts.items()))},
    )
    written = stage.write_jsonl(out, head, records)
    logger.info("%s: %d episodes scored over %d directories",
                args.source, len(records), len(directories))
    for verdict in VERDICTS:
        logger.info("%s %d", verdict, verdicts[verdict])
    if controls:
        logger.info("median %.2f against control median %.2f, own pose wins %s",
                    float(np.median(measured)), float(np.median(controls)), beats)
    for reason, count in sorted(counts.items()):
        logger.info("%s %d", reason, count)
    logger.info("%s %s", "wrote" if written else "unchanged", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
