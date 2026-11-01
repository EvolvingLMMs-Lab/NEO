"""AgiBot G1: arm joints and the body block, from one pass over the raw h5.

Row i of a directory episode is row i of AgiBotWorld-Beta's proprio_stats.h5
(the converter emits every frame), so the episode's h5 must hold exactly the
directory's rows and its end-effector track must be the directory's state.
Joints are state/joint/position, left [:, :7], right [:, 7:], in rad. The body
block is base dx, dy (cm per step) and dyaw (rad per step) from the odometry,
waist pitch and lift as torso 0 and 1 (torso 2 is not recorded), and head yaw
and pitch. The odometry is held: it updates on about 2 of every 3 rows, so it
is resampled to row time before it is differenced (base_motion).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow.parquet as pq

from sidecars import common

ARMS = 2

# The h5 datasets read, by what they are.
TIMESTAMP = "timestamp"
JOINTS = "state/joint/position"
END_STATE = "state/end/position"
END_ACTION = "action/end/position"
WAIST = "state/waist/position"
HEAD = "state/head/position"
ODOMETRY_POSITION = "state/robot/position"
ODOMETRY_ORIENTATION = "state/robot/orientation"

# The directory's state columns holding each arm's position in cm.
STATE_POSITION = ((0, 3), (8, 11))

# How far the directory's position may sit from the h5's, in cm. The state is
# the h5 metre value times 100 stored as float32, which is exact to a few
# micrometres at the reach of this arm; a misaligned row is centimetres off.
END_TOLERANCE_CM = 1e-3

# A row's odometry is a reading when its quaternion is one: an all-zero
# quaternion is how an episode without odometry stores it.
QUATERNION_FLOOR = 0.5

# The longest gap, in rows, between two odometry updates that is read as one
# update period. Measured cadence: updates come 1 or 2 rows apart while the
# base moves; a longer gap is the base standing still, which holds its pose
# until 2 rows before the next update instead of creeping towards it.
LONGEST_UPDATE_GAP = 3
ONSET_ROWS = 2

# The census motion rule, for the moving-row diagnostics only.
MOVING_PLANAR_CM = 0.3
MOVING_YAW_RAD = 3e-3


def yaw_of(orientation: np.ndarray) -> np.ndarray:
    """Yaw in rad from a yaw-only quaternion stored xyzw."""
    return 2.0 * np.arctan2(orientation[:, 2], orientation[:, 3])


def update_rows(position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """The rows where the odometry changed, the first row always among them."""
    changed = np.zeros(position.shape[0], dtype=bool)
    changed[0] = True
    changed[1:] = ((position[1:] != position[:-1]).any(axis=1)
                   | (orientation[1:] != orientation[:-1]).any(axis=1))
    return np.flatnonzero(changed)


def knots_of(updates: np.ndarray) -> np.ndarray:
    """The rows the resampled pose passes through exactly.

    Every update row, and for a gap longer than LONGEST_UPDATE_GAP the row
    ONSET_ROWS before the next update, which holds the previous pose: the base
    was still until then, and the update spreads over the rows one update
    period covers rather than over the whole stop.
    """
    rows = list(updates)
    for first, second in zip(updates[:-1], updates[1:]):
        if second - first > LONGEST_UPDATE_GAP:
            rows.append(second - ONSET_ROWS)
    return np.unique(np.asarray(rows, dtype=np.int64))


def resampled_pose(times: np.ndarray, position: np.ndarray,
                   orientation: np.ndarray) -> np.ndarray:
    """x, y (m) and unwrapped yaw (rad) at every row of one valid stretch.

    1. Keep the rows where the pose changed (and the stop holds, knots_of).
    2. Interpolate x, y and yaw linearly over the row times between them.
    3. Average each row with its two neighbours. With updates 1 and 2 rows
       apart, interpolation alone moves a constant-speed base 1.5 and 0.75
       times its speed on alternate rows; any 3 consecutive rows hold exactly
       two updates, so the 3-row mean gives the constant speed back.
    The stretch's first and last pose are kept, so the summed motion is kept.
    A knot knots_of adds for a stop lies where the pose is held, so the value
    the pose holds there is the update before it.
    """
    yaw = np.unwrap(yaw_of(orientation))
    knots = knots_of(update_rows(position, orientation))
    values = np.stack([position[:, 0], position[:, 1], yaw], axis=1)
    pose = np.stack([np.interp(times, times[knots], values[knots, axis])
                     for axis in range(3)], axis=1)
    padded = np.concatenate([pose[:1], pose, pose[-1:]])
    smoothed = (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0
    smoothed[0], smoothed[-1] = pose[0], pose[-1]
    return smoothed


def displacement(pose: np.ndarray) -> np.ndarray:
    """Per-step motion r-1 -> r in the base frame at r-1: dx, dy in cm, dyaw in rad."""
    step = np.diff(pose, axis=0)
    heading = pose[:-1, 2]
    cosine, sine = np.cos(heading), np.sin(heading)
    forward = cosine * step[:, 0] + sine * step[:, 1]
    lateral = -sine * step[:, 0] + cosine * step[:, 1]
    return np.stack([forward * 100.0, lateral * 100.0, step[:, 2]], axis=1)


def valid_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    """Maximal [start, stop) runs of True."""
    edges = np.diff(np.concatenate([[0], valid.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def base_motion(timestamps: np.ndarray, position: np.ndarray | None,
                orientation: np.ndarray | None) -> np.ndarray:
    """Body columns B0-B2 per row: the last step's motion, NaN where unread.

    A row's motion is read only when it and the row before it both hold an
    odometry reading, so row 0 and the first row of every valid stretch are
    NaN, and an episode without odometry is NaN throughout, never 0.
    """
    rows = timestamps.shape[0]
    out = np.full((rows, 3), np.nan, dtype=np.float64)
    if (position is None or orientation is None or position.shape[0] != rows
            or orientation.shape[0] != rows or rows < 2):
        return out
    planar = position[:, :2]
    valid = ((np.linalg.norm(orientation, axis=1) > QUATERNION_FLOOR)
             & np.isfinite(planar).all(axis=1) & np.isfinite(orientation).all(axis=1))
    times = (timestamps - timestamps[0]).astype(np.float64)
    if np.any(np.diff(times) <= 0):
        times = np.arange(rows, dtype=np.float64)
    for start, stop in valid_runs(valid):
        if stop - start < 2:
            continue
        pose = resampled_pose(times[start:stop], planar[start:stop],
                              orientation[start:stop])
        out[start + 1:stop] = displacement(pose)
    return out


def moving_rows(position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """Rows the census calls moving: over a 3-row window, > 3 mm or > 3e-3 rad."""
    rows = position.shape[0]
    moving = np.zeros(rows, dtype=bool)
    if rows < 4:
        return moving
    yaw = np.unwrap(yaw_of(orientation))
    planar = np.linalg.norm(position[3:, :2] - position[:-3, :2], axis=1) * 100.0
    turn = np.abs(yaw[3:] - yaw[:-3])
    moving[3:] = (planar > MOVING_PLANAR_CM) | (turn > MOVING_YAW_RAD)
    return moving


def _read(handle: h5py.File, name: str) -> np.ndarray | None:
    if name not in handle or handle[name].size == 0:
        return None
    return np.asarray(handle[name][()], dtype=np.float64)


def episode_arrays(h5_path: Path, rows: int, state: np.ndarray,
                   digest: common.SourceDigest) -> dict[str, Any]:
    """One episode's joints and body rows, checked against the directory's state."""
    with h5py.File(h5_path, "r") as handle:
        timestamps = np.asarray(handle[TIMESTAMP][()], dtype=np.int64)
        joints = _read(handle, JOINTS)
        end = _read(handle, END_STATE)
        end_source = END_STATE
        if end is None:
            end, end_source = _read(handle, END_ACTION), END_ACTION
        waist = _read(handle, WAIST)
        head = _read(handle, HEAD)
        position = _read(handle, ODOMETRY_POSITION)
        orientation = _read(handle, ODOMETRY_ORIENTATION)
    if timestamps.shape[0] != rows:
        raise common.Misaligned(
            f"{h5_path} holds {timestamps.shape[0]} rows, the directory episode {rows}\n"
            "expected row i of the h5 to be row i of the episode\n"
            "re-convert this episode or drop it from the directory"
        )
    if end is None or end.reshape(rows, -1).shape[1] != 6:
        raise common.Misaligned(
            f"{h5_path} has no two-arm end-effector track to check rows against\n"
            f"expected {END_STATE} of shape (N, 2, 3)\n"
            "check the source episode"
        )
    flat = end.reshape(rows, 6) * 100.0
    error = max(float(np.max(np.abs(state[:, a:b] - flat[:, 3 * arm:3 * arm + 3])))
                for arm, (a, b) in enumerate(STATE_POSITION))
    if not error <= END_TOLERANCE_CM:
        raise common.Misaligned(
            f"{h5_path} end-effector position is {error:.3e} cm from the directory's state\n"
            f"expected at most {END_TOLERANCE_CM} cm on every row\n"
            "the h5 is not this episode's source; check meta/episodes.jsonl raw_file_name"
        )
    for name, value in (("joints", joints), ("waist", waist), ("head", head)):
        if value is not None and value.shape[0] != rows:
            raise common.Misaligned(
                f"{h5_path} {name} holds {value.shape[0]} rows, the episode {rows}\n"
                "expected every h5 group on the timestamp's rows\n"
                "check the source episode"
            )
    for value in (timestamps, joints, waist, head, position, orientation):
        digest.array(value)
    body = np.full((rows, 8), np.nan, dtype=np.float64)
    body[:, 0:3] = base_motion(timestamps, position, orientation)
    if waist is not None and waist.shape[1] >= 2:
        body[:, 3] = waist[:, 0]
        body[:, 4] = waist[:, 1]
    if head is not None and head.shape[1] >= 2:
        body[:, 6] = head[:, 0]
        body[:, 7] = head[:, 1]
    odometry = (position is not None and orientation is not None
                and position.shape[0] == rows and orientation.shape[0] == rows)
    moving = (moving_rows(position, orientation) if odometry
              else np.zeros(rows, dtype=bool))
    still = moving & (body[:, 0:3] == 0.0).all(axis=1)
    return {
        "joints": None if joints is None else joints[:, :14],
        "body": body,
        "end_error_cm": error,
        "end_source": end_source,
        "moving_rows": int(moving.sum()),
        "moving_rows_read_still": int(still.sum()),
        "odometry_rows": int(np.isfinite(body[:, 0]).sum()),
    }


DESCRIBED = {
    common.COLUMN_JOINTS: {"arms": ARMS, "unit": "rad", "recorded": True,
                           "source": f"proprio_stats.h5 {JOINTS}, left [:, :7], right [:, 7:]"},
    common.COLUMN_HAND: {"arms": ARMS, "unit": "rad", "recorded": False,
                         "source": "not recorded (parallel grippers)"},
    common.COLUMN_BODY: {
        "unit": "B0-B1 cm/step, B2 rad/step, T0 rad, T1 m, H rad", "recorded": True,
        "source": (f"B0-B2 {ODOMETRY_POSITION} and {ODOMETRY_ORIENTATION} resampled to row "
                   f"time, differenced in the base frame of the step's start; T0, T1 "
                   f"{WAIST} pitch, lift; T2 not recorded; H0, H1 {HEAD} yaw, pitch"),
    },
}


def build(entry: dict[str, Any], directory: Path, raw_root: Path) -> dict[str, Any]:
    """Every row of one AgiBot directory: columns, stamp parts and checks."""
    episodes = common.directory_episodes(directory)
    total = sum(episode.rows for episode in episodes)
    columns = common.empty_columns(total, ARMS, command=False)
    episode_column = np.empty(total, dtype=np.int64)
    frame_column = np.empty(total, dtype=np.int64)
    digest = common.SourceDigest()
    checks = {"episodes": len(episodes), "end_error_cm_max": 0.0, "end_source": {},
              "joints_missing_episodes": 0, "odometry_episodes": 0,
              "odometry_rows": 0, "moving_rows": 0, "moving_rows_read_still": 0}
    start = 0
    for episode in episodes:
        table = pq.read_table(episode.data_file,
                              columns=["observation.state", common.FRAME_COLUMN])
        state = np.stack(table.column("observation.state").to_numpy(zero_copy_only=False))
        raw = episode.record["raw_file_name"]
        digest.text(raw)
        arrays = episode_arrays(raw_root / raw / "proprio_stats.h5", episode.rows,
                                state.astype(np.float64), digest)
        stop = start + episode.rows
        episode_column[start:stop] = episode.index
        frame_column[start:stop] = table.column(common.FRAME_COLUMN).to_numpy()
        if arrays["joints"] is None:
            checks["joints_missing_episodes"] += 1
        else:
            columns[common.COLUMN_JOINTS][start:stop] = common.finite(arrays["joints"])
        columns[common.COLUMN_BODY][start:stop] = common.finite(arrays["body"])
        checks["end_error_cm_max"] = max(checks["end_error_cm_max"], arrays["end_error_cm"])
        checks["end_source"][arrays["end_source"]] = (
            checks["end_source"].get(arrays["end_source"], 0) + 1)
        checks["odometry_episodes"] += int(arrays["odometry_rows"] > 0)
        for key in ("odometry_rows", "moving_rows", "moving_rows_read_still"):
            checks[key] += arrays[key]
        start = stop
    return {"episodes": episodes, "columns": columns, "episode_column": episode_column,
            "frame_column": frame_column, "described": DESCRIBED, "arms": ARMS,
            "source_digest": digest.hexdigest(), "checks": checks}
