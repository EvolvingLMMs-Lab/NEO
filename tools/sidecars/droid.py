"""DROID (droid3cam): the measured joint state and the commanded joint position.

Row i of a directory episode is row i of the session's trajectory.h5 (the
converter keeps min(trajectory rows, video frames) rows from the start), so
the episode's cartesian track must be the directory's state on every row.
observation.joints is observation/robot_state/joint_positions and
action.joints is action/joint_position, the command, which leads the state by
4 or 5 rows; both 7 wide, in rad, Panda URDF order. The trajectory root is the
one the directory's meta/info.json names unless the caller passes another.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow.parquet as pq

from sidecars import common

ARMS = 1

TRAJECTORY_FILE = "trajectory.h5"
JOINT_STATE = "observation/robot_state/joint_positions"
JOINT_COMMAND = "action/joint_position"
CARTESIAN = "observation/robot_state/cartesian_position"

# How far the directory's state may sit from the trajectory's cartesian pose,
# in its own units (m and rad): the state is the float64 pose stored as float32.
CARTESIAN_TOLERANCE = 1e-5

DESCRIBED = {
    common.COLUMN_JOINTS: {"arms": ARMS, "unit": "rad", "recorded": True,
                           "source": f"trajectory.h5 {JOINT_STATE}"},
    common.COLUMN_HAND: {"arms": ARMS, "unit": "rad", "recorded": False,
                         "source": "not recorded (parallel gripper)"},
    common.COLUMN_BODY: {"unit": "", "recorded": False,
                         "source": "not recorded (fixed base)"},
    common.COLUMN_JOINTS_COMMAND: {"arms": ARMS, "unit": "rad", "recorded": True,
                                   "source": f"trajectory.h5 {JOINT_COMMAND} (command)"},
}


def raw_root_of(directory: Path) -> Path:
    """The trajectory root the directory's converter read, from meta/info.json."""
    info = json.loads((directory / "meta" / "info.json").read_text(encoding="utf-8"))
    try:
        return Path(info["droid_provenance"]["settings"]["raw_root"])
    except KeyError:
        raise ValueError(
            f"{directory}/meta/info.json names no droid_provenance.settings.raw_root\n"
            "expected the converter's provenance block\n"
            "pass --droid-raw with the trajectory root"
        ) from None


def episode_arrays(path: Path, rows: int, state: np.ndarray) -> dict[str, Any]:
    """One episode's joint state and command, checked against the directory's state."""
    with h5py.File(path, "r") as handle:
        stored = {name: (np.asarray(handle[name][:rows], dtype=np.float64)
                         if name in handle else None)
                  for name in (JOINT_STATE, JOINT_COMMAND, CARTESIAN)}
    cartesian = stored[CARTESIAN]
    if cartesian is None or cartesian.shape != (rows, 6):
        raise common.Misaligned(
            f"{path} {CARTESIAN} is "
            f"{None if cartesian is None else cartesian.shape} for {rows} rows\n"
            "expected the trajectory to hold at least the episode's rows\n"
            "the trajectory is not this episode's source; check droid_session"
        )
    error = float(np.max(np.abs(state[:, 0:6] - cartesian)))
    if not error <= CARTESIAN_TOLERANCE:
        raise common.Misaligned(
            f"{path} cartesian pose is {error:.3e} from the directory's state\n"
            f"expected at most {CARTESIAN_TOLERANCE} on every row\n"
            "the trajectory is not this episode's source; check droid_session"
        )
    for name in (JOINT_STATE, JOINT_COMMAND):
        value = stored[name]
        if value is not None and value.shape != (rows, 7):
            raise common.Misaligned(
                f"{path} {name} is {value.shape} for {rows} rows\n"
                "expected 7 joints on every row the episode keeps\n"
                "check the source session"
            )
    return {"state": stored[JOINT_STATE], "command": stored[JOINT_COMMAND],
            "error": error}


def build(entry: dict[str, Any], directory: Path, raw_root: Path | None) -> dict[str, Any]:
    """Every row of one DROID directory: columns, stamp parts and checks."""
    root = raw_root if raw_root is not None else raw_root_of(directory)
    episodes = common.directory_episodes(directory)
    total = sum(episode.rows for episode in episodes)
    columns = common.empty_columns(total, ARMS, command=True)
    episode_column = np.empty(total, dtype=np.int64)
    frame_column = np.empty(total, dtype=np.int64)
    checks = {"episodes": len(episodes), "cartesian_error_max": 0.0,
              "state_missing_episodes": 0, "command_missing_episodes": 0}
    start = 0
    for episode in episodes:
        table = pq.read_table(episode.data_file,
                              columns=["observation.state", common.FRAME_COLUMN])
        state = np.stack(table.column("observation.state").to_numpy(zero_copy_only=False))
        session = episode.record["droid_session"]
        arrays = episode_arrays(root / session / TRAJECTORY_FILE, episode.rows,
                                state.astype(np.float64))
        stop = start + episode.rows
        episode_column[start:stop] = episode.index
        frame_column[start:stop] = table.column(common.FRAME_COLUMN).to_numpy()
        for key, column, missing in (
                ("state", common.COLUMN_JOINTS, "state_missing_episodes"),
                ("command", common.COLUMN_JOINTS_COMMAND, "command_missing_episodes")):
            if arrays[key] is None:
                checks[missing] += 1
            else:
                columns[column][start:stop] = common.finite(arrays[key])
        checks["cartesian_error_max"] = max(checks["cartesian_error_max"], arrays["error"])
        start = stop
    return {"episodes": episodes, "columns": columns, "episode_column": episode_column,
            "frame_column": frame_column, "described": DESCRIBED, "arms": ARMS,
            "checks": checks, "source_root": str(root)}
