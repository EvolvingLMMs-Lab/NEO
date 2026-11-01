"""RH20T: arm joints from the lowdim archives, keyed onto the grid rows.

transformed/joint.npy exists only inside RH20T_cfg<N>.tar.gz. extract streams
one archive once and keeps, for every scene a grid directory uses, joint.npy
and tcp_base.npy as one npz of plain arrays per scene, then an index naming
what it found; build reads only those. A grid row is keyed as the grid was
built: the clock camera's source_frame picks a row of the clock camera's own
v3 episode, whose source_timestamp_ms is the key into joint.npy[clock serial]
(the grid row's own timestamp is not). tcp_base shares joint.npy's clock, so
the directory's position must equal tcp_base at the same key on every keyed
row. The leading 7 values are the joints (ur5: 6, its joint 6 is NaN). A
reading that cannot be an angle (common.implausible_joints: beyond 2 pi, or a
spike of more than pi that comes back within two rows) is NaN, one value at a
time, and counted per robot and joint in the checks.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import tarfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
import stage

from neo_vla.data.reader import VIEW_FRAME_SUFFIX
from sidecars import common

logger = logging.getLogger("sidecars.rh20t")

ARMS = 1
JOINT_MEMBER = "transformed/joint.npy"
TCP_MEMBER = "transformed/tcp_base.npy"
INDEX = "_index.json"
INDEX_FORMAT = "neo_vla_rh20t_lowdim_v1"

# Joints per robot word of meta/episodes.jsonl rh20t_robot (rh20t_api
# configs.json robot_joint_field [0, 7]; the UR5 has 6).
ROBOT_JOINTS = {"flexiv": 7, "franka": 7, "kuka": 7, "ur5": 6}

# How far the directory's position may sit from tcp_base at the same key: the
# state is the metre value times 100 stored as float32, exact to
# TCP_TOLERANCE_CM at an arm's reach and to TCP_RELATIVE of the value at the
# impossible magnitudes some RH20T readings hold (measured: -1.4e6 and
# -9.6e37 cm, the same reading at the same key in both files, so the key is
# still checked there). A position beyond PLAUSIBLE_CM is counted, not judged.
TCP_TOLERANCE_CM = 1e-3
TCP_RELATIVE = 1e-6
PLAUSIBLE_CM = 1e4

DESCRIBED = {
    common.COLUMN_JOINTS: {"arms": ARMS, "unit": "rad", "recorded": True,
                           "source": ("RH20T_cfg<N>.tar.gz <scene>/transformed/joint.npy"
                                      "[clock serial][source_timestamp_ms of the clock "
                                      "camera's source_frame], leading 7 (ur5 6)")},
    common.COLUMN_HAND: {"arms": ARMS, "unit": "rad", "recorded": False,
                         "source": "not recorded (parallel gripper)"},
    common.COLUMN_BODY: {"unit": "", "recorded": False,
                         "source": "not recorded (fixed base)"},
}


def scene_of(raw_file_name: str) -> tuple[str, str]:
    """RH20T_cfg3/<scene>#group01 -> (RH20T_cfg3, <scene>)."""
    config, rest = raw_file_name.split("/", 1)
    return config, rest.split("#", 1)[0]


def wanted_scenes(directories: Iterable[Path]) -> dict[str, set[str]]:
    """Every scene the grid directories use, by configuration."""
    wanted: dict[str, set[str]] = {}
    for directory in directories:
        with open(directory / "meta" / "episodes.jsonl", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    config, scene = scene_of(json.loads(line)["raw_file_name"])
                    wanted.setdefault(config, set()).add(scene)
    return wanted


def _joint_arrays(loaded: Any) -> dict[str, np.ndarray]:
    arrays = {}
    for serial, by_time in loaded.items():
        times = np.asarray(sorted(by_time), dtype=np.int64)
        arrays[f"joint_ts__{serial}"] = times
        arrays[f"joint__{serial}"] = np.asarray([by_time[int(t)] for t in times],
                                                dtype=np.float64)
    return arrays


def _tcp_arrays(loaded: Any) -> dict[str, np.ndarray]:
    arrays = {}
    for serial, records in loaded.items():
        kept = [record for record in records
                if record.get("tcp") is not None and len(record["tcp"]) == 7]
        arrays[f"tcp_ts__{serial}"] = np.asarray([r["timestamp"] for r in kept],
                                                 dtype=np.int64)
        arrays[f"tcp__{serial}"] = np.asarray([r["tcp"] for r in kept],
                                              dtype=np.float64).reshape(-1, 7)
    return arrays


def _write_scene(path: Path, arrays: dict[str, np.ndarray]) -> str:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    payload = buffer.getvalue()
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    if stage.sha256_of(temporary) != digest:
        temporary.unlink()
        raise OSError(
            f"{temporary} does not hold the {len(payload)} bytes written to it\n"
            "expected the file to read back as what was produced\n"
            "free space under the output root, then run the archive again"
        )
    os.replace(temporary, path)
    return digest


def extract(archive: Path, wanted: set[str], work: Path) -> dict[str, Any]:
    """Stream one lowdim archive and keep the wanted scenes' joint and tcp_base.

    A scene's members sit together in the archive, so a scene is written as
    soon as the stream moves on to another one, and memory holds one scene.
    The index is written last: a run stopped before it leaves no index, and
    build refuses a configuration without one.
    """
    config = archive.name.split(".", 1)[0]
    target = work / config
    target.mkdir(parents=True, exist_ok=True)
    stale = target / INDEX
    if stale.exists():
        stale.unlink()
    kinds_found: dict[str, set[str]] = {}
    unreadable: dict[str, str] = {}
    files: dict[str, str] = {}
    pending: dict[str, dict[str, np.ndarray]] = {}
    streamed = 0

    def flush(scene: str) -> None:
        arrays = pending.pop(scene)
        path = target / f"{scene}.npz"
        if scene in files:
            with np.load(path) as earlier:
                arrays = {**{key: earlier[key] for key in earlier.files}, **arrays}
        arrays["has_joint"] = np.asarray("joint" in kinds_found[scene])
        arrays["has_tcp"] = np.asarray("tcp" in kinds_found[scene])
        files[scene] = _write_scene(path, arrays)

    with tarfile.open(archive, mode="r|gz") as stream:
        for member in stream:
            if not member.isfile():
                continue
            parts = member.name.split("/")
            parts = parts[1:] if parts and parts[0] == config else parts
            if len(parts) != 3 or "/".join(parts[1:]) not in (JOINT_MEMBER, TCP_MEMBER):
                continue
            scene = parts[0]
            if scene not in wanted:
                continue
            for other in [name for name in pending if name != scene]:
                flush(other)
            payload = stream.extractfile(member).read()
            streamed += len(payload)
            kind = "joint" if parts[2] == "joint.npy" else "tcp"
            try:
                loaded = np.load(io.BytesIO(payload), allow_pickle=True).item()
                arrays = _joint_arrays(loaded) if kind == "joint" else _tcp_arrays(loaded)
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                unreadable[f"{scene}/{kind}"] = str(error)[:200]
                continue
            kinds_found.setdefault(scene, set()).add(kind)
            pending.setdefault(scene, {}).update(arrays)
    for scene in list(pending):
        flush(scene)
    stat = archive.stat()
    with_joint = {scene for scene, kinds in kinds_found.items() if "joint" in kinds}
    index = {
        "format": INDEX_FORMAT, "complete": True, "config": config,
        "archive": str(archive), "archive_bytes": stat.st_size,
        "archive_mtime": int(stat.st_mtime), "member_bytes_kept": streamed,
        "scenes_wanted": len(wanted), "scenes_with_joint": len(with_joint),
        "scenes_with_tcp_base": sum("tcp" in kinds for kinds in kinds_found.values()),
        "scenes_without_joint": sorted(wanted - with_joint),
        "unreadable": unreadable, "files": dict(sorted(files.items())),
    }
    (target / INDEX).write_text(stage.dumps_json(index), encoding="utf-8")
    logger.info("%s: %d of %d wanted scenes hold joint.npy, %d tcp_base, %d unreadable",
                config, index["scenes_with_joint"], len(wanted),
                index["scenes_with_tcp_base"], len(unreadable))
    return index


def extracted(archive: Path, work: Path) -> dict[str, Any] | None:
    """The index of an archive already extracted from these bytes, if it holds."""
    config = archive.name.split(".", 1)[0]
    path = work / config / INDEX
    if not path.exists():
        return None
    index = json.loads(path.read_text(encoding="utf-8"))
    stat = archive.stat()
    if (index.get("format") != INDEX_FORMAT or not index.get("complete")
            or index.get("archive_bytes") != stat.st_size
            or index.get("archive_mtime") != int(stat.st_mtime)):
        return None
    for scene, digest in index["files"].items():
        if stage.sha256_of(work / config / f"{scene}.npz") != digest:
            return None
    return index


def load_index(work: Path, config: str, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """A configuration's extraction index, refused when absent or incomplete."""
    if config not in cache:
        path = work / config / INDEX
        if not path.exists():
            raise ValueError(
                f"no extraction index {path}\n"
                "expected tools/sidecars/run.py rh20t-extract to have streamed this archive\n"
                "run the extraction first; a missing scene is only known after it"
            )
        index = json.loads(path.read_text(encoding="utf-8"))
        if index.get("format") != INDEX_FORMAT or not index.get("complete"):
            raise ValueError(
                f"{path} is not a complete {INDEX_FORMAT} index\n"
                "expected the record a finished extraction writes\n"
                "run the extraction of this configuration again"
            )
        cache[config] = index
    return cache[config]


def source_table(source_root: Path, source: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """The clock camera's v3 episode: its timestamps (ms) and its state."""
    directory = source_root / source["directory"]
    info = json.loads((directory / "meta" / "info.json").read_text(encoding="utf-8"))
    index = int(source["episode_index"])
    path = directory / info["data_path"].format(
        episode_chunk=index // int(info.get("chunks_size", 1000)), episode_index=index)
    table = pq.read_table(path, columns=["source_timestamp_ms", "observation.state"])
    times = np.asarray(table.column("source_timestamp_ms").to_numpy(), dtype=np.int64)
    state = np.stack(table.column("observation.state").to_numpy(zero_copy_only=False))
    return times, state.astype(np.float64)


def lookup(times: np.ndarray, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each key, whether the sorted times hold it and at which row."""
    at = np.searchsorted(times, keys)
    inside = at < times.shape[0]
    hit = np.zeros(keys.shape[0], dtype=bool)
    hit[inside] = times[at[inside]] == keys[inside]
    return hit, np.where(hit, at, 0)


def episode_joints(record: dict[str, Any], grid: dict[str, np.ndarray],
                   scene: dict[str, np.ndarray] | None, source_root: Path,
                   digest: common.SourceDigest, where: str) -> dict[str, Any]:
    """One grid episode's joints and its keying counts."""
    rows = grid["state"].shape[0]
    robot = record["rh20t_robot"]
    if robot not in ROBOT_JOINTS:
        raise common.Misaligned(
            f"{where} names robot {robot!r}\n"
            f"expected one of {sorted(ROBOT_JOINTS)}\n"
            "add the robot's joint count to ROBOT_JOINTS"
        )
    count = ROBOT_JOINTS[robot]
    joints = np.full((rows, 7), np.nan, dtype=np.float64)
    result = {"robot": robot, "rows": rows, "keyed": 0, "scene_missing": scene is None,
              "tcp_rows": 0, "tcp_error_cm": 0.0, "joints": joints,
              "implausible_joint_values": [0] * joints.shape[1]}
    times, state = source_table(source_root, grid["source"])
    frames = grid["frames"]
    inside = (frames >= 0) & (frames < times.shape[0])
    if not np.array_equal(grid["state"][inside], state[frames[inside]]):
        raise common.Misaligned(
            f"{where} state is not its clock camera's source state at source_frame\n"
            "expected the grid row to be the source row it was taken from\n"
            "rebuild the grid directory or check rh20t_views[clock].source"
        )
    keys = np.where(inside, times[np.clip(frames, 0, max(times.shape[0] - 1, 0))], -1)
    digest.text(f"{record['raw_file_name']}|{record['rh20t_grid_clock_serial']}")
    digest.array(keys)
    if scene is None or not bool(scene["has_joint"]):
        result["scene_missing"] = True
        return result
    serial = record["rh20t_grid_clock_serial"]
    if f"joint_ts__{serial}" not in scene:
        result["serial_missing"] = True
        return result
    for kind in ("joint", "tcp"):
        if f"{kind}_ts__{serial}" in scene:
            order = np.argsort(scene[f"{kind}_ts__{serial}"], kind="stable")
            scene[f"{kind}_ts__{serial}"] = scene[f"{kind}_ts__{serial}"][order]
            scene[f"{kind}__{serial}"] = scene[f"{kind}__{serial}"][order]
    hit, at = lookup(scene[f"joint_ts__{serial}"], keys)
    hit &= inside
    values = scene[f"joint__{serial}"]
    if values.shape[1] < count:
        raise common.Misaligned(
            f"{where} joint.npy rows are {values.shape[1]} wide for robot {robot}\n"
            f"expected at least {count} joint values\n"
            "check the archive member"
        )
    joints[hit, :count] = values[at[hit], :count]
    digest.array(joints[hit])
    block = joints[:, :count]
    implausible = common.implausible_joints(block)
    block[implausible] = np.nan
    digest.array(implausible)
    result["implausible_joint_values"][:count] = implausible.sum(axis=0).tolist()
    result["keyed"] = int(hit.sum())
    if f"tcp_ts__{serial}" in scene:
        tcp_hit, tcp_at = lookup(scene[f"tcp_ts__{serial}"], keys)
        tcp_hit &= inside
        if tcp_hit.any():
            tcp = scene[f"tcp__{serial}"][tcp_at[tcp_hit], :3] * 100.0
            error = np.abs(grid["state"][tcp_hit, 0:3] - tcp)
            outside = error > TCP_TOLERANCE_CM + TCP_RELATIVE * np.abs(tcp)
            if outside.any():
                raise common.Misaligned(
                    f"{where} position is {float(error[outside].max()):.3e} cm from "
                    "tcp_base at the same key\n"
                    f"expected at most {TCP_TOLERANCE_CM} cm plus {TCP_RELATIVE} of the value: "
                    "joint.npy and tcp_base share a clock\n"
                    "the keying is wrong for this episode; check the clock slot and serial"
                )
            plausible = (np.abs(tcp) < PLAUSIBLE_CM).all(axis=1)
            result["tcp_rows"] = int(tcp_hit.sum())
            result["tcp_error_cm"] = float(error[plausible].max()) if plausible.any() else 0.0
            result["implausible_rows"] = int((~plausible).sum())
    return result


def build(entry: dict[str, Any], directory: Path, work: Path,
          source_root: Path) -> dict[str, Any]:
    """Every row of one RH20T grid directory: columns, stamp parts and checks."""
    episodes = common.directory_episodes(directory)
    total = sum(episode.rows for episode in episodes)
    columns = common.empty_columns(total, ARMS, command=False)
    episode_column = np.empty(total, dtype=np.int64)
    frame_column = np.empty(total, dtype=np.int64)
    digest = common.SourceDigest()
    indexes: dict[str, dict[str, Any]] = {}
    checks: dict[str, Any] = {"episodes": len(episodes), "by_robot": {},
                              "tcp_error_cm_max": 0.0, "tcp_rows": 0,
                              "implausible_state_rows": 0, "implausible_joint_values": 0}
    start = 0
    for episode in episodes:
        record = episode.record
        slot = int(record["rh20t_grid_clock_slot"])
        view = next(v for v in record["rh20t_views"] if int(v["slot"]) == slot)
        frame_name = f"{view['key']}{VIEW_FRAME_SUFFIX}"
        table = pq.read_table(episode.data_file, columns=[
            "observation.state", frame_name, common.FRAME_COLUMN])
        grid = {"state": np.stack(table.column("observation.state")
                                  .to_numpy(zero_copy_only=False)).astype(np.float64),
                "frames": np.asarray(table.column(frame_name).to_numpy(), dtype=np.int64),
                "source": view["source"]}
        config, scene_name = scene_of(record["raw_file_name"])
        index = load_index(work, config, indexes)
        scene = None
        if scene_name in index["files"]:
            with np.load(work / config / f"{scene_name}.npz") as loaded:
                scene = {key: loaded[key] for key in loaded.files}
        result = episode_joints(record, grid, scene, source_root, digest,
                                f"{entry['name']} episode {episode.index}")
        stop = start + episode.rows
        episode_column[start:stop] = episode.index
        frame_column[start:stop] = table.column(common.FRAME_COLUMN).to_numpy()
        columns[common.COLUMN_JOINTS][start:stop] = common.finite(result["joints"])
        robot = checks["by_robot"].setdefault(result["robot"], {
            "episodes": 0, "rows": 0, "keyed_rows": 0, "scene_missing_episodes": 0,
            "serial_missing_episodes": 0,
            "implausible_joint_values": [0] * len(result["implausible_joint_values"])})
        robot["implausible_joint_values"] = [
            held + found for held, found in zip(robot["implausible_joint_values"],
                                                result["implausible_joint_values"])]
        checks["implausible_joint_values"] += sum(result["implausible_joint_values"])
        robot["episodes"] += 1
        robot["rows"] += result["rows"]
        robot["keyed_rows"] += result["keyed"]
        robot["scene_missing_episodes"] += int(result["scene_missing"])
        robot["serial_missing_episodes"] += int(result.get("serial_missing", False))
        checks["tcp_rows"] += result["tcp_rows"]
        checks["implausible_state_rows"] += result.get("implausible_rows", 0)
        checks["tcp_error_cm_max"] = max(checks["tcp_error_cm_max"], result["tcp_error_cm"])
        start = stop
    return {"episodes": episodes, "columns": columns, "episode_column": episode_column,
            "frame_column": frame_column, "described": DESCRIBED, "arms": ARMS,
            "source_digest": digest.hexdigest(), "checks": checks,
            "source_root": str(source_root)}
