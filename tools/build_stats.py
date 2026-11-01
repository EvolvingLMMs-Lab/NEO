"""Stage: embodiments.json v2, the per-(key, slot, dim) normaliser of the v7 layout.

Every key is measured on the loader's own targets and state (RobotDataset in
targets-only mode), drawn per stratum and weighted by stratum mass, over the
26-wide arm row, the 29-wide arm state and the 8-wide body. A full scan of
every directory finds constant columns, dead readings (dead_dims), joints
that wrap past pi and columns never read, and each directory's record masks
them. The gates refuse by name, with counts and their
denominators. The file is written and validated through
neo_vla.data.embodiment, the one implementation.

Usage:
    python tools/build_stats.py --root /corpus --manifest manifests/gd7_tokens_nobw.json \
        --quantiles 1 99 --band-alpha 1.0 --windows-per-stratum 15000 \
        --frame-policies as-drawn --constant-scan full --out prep/embodiments.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import multiprocessing
import os
import re
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import key_table
import numpy as np
import stage
import unify
from pyarrow import feather

from neo_vla.constants import (
    ACTION_DIM,
    ACTION_DPOS,
    ACTION_DROT,
    ACTION_HAND,
    ACTION_JOINTS,
    ARM_JOINTS,
    BODY_BASE,
    BODY_DIM,
    BODY_HEAD,
    BODY_TORSO,
    EMBODIMENTS_SCHEMA,
    GRIPPER_COLUMN,
    HAND_JOINTS,
    MANIFEST_SCHEMA_VERSION,
    OLD_MANIFEST_SCHEMA_VERSIONS,
    RANGE_COLUMNS,
    STATE_APERTURE,
    STATE_DIM,
    STATE_HAND,
    STATE_JOINTS,
    STATE_POSITION,
    STATE_ROTATION,
    refusal_prefix,
)
from neo_vla.data.embodiment import (
    BODY_DIM_NAMES,
    DIM_NAMES,
    DROT_RECORD,
    FIT_FIELDS,
    FIT_FLAGGED_MAX,
    FIT_P99_BAND,
    MASK,
    QUANTILES,
    STATE_DIM_NAMES,
    STATE_RANGE,
    TRANSFORM,
    UNTRAINED_DROT,
    FitReport,
    base_moving,
    fit_base_scale,
    header_of,
    inherited_columns,
    weighted_quantile,
)

logger = logging.getLogger("build_stats")

REPO = Path(__file__).resolve().parent.parent

# The row and record this stage measures, from the contract and the normaliser
# (no copy of their own): one arm's 26 action columns, its 29 state dims and
# the 8 body dims. RANGE_COLUMNS are the q01/q99-mapped ones, every column but
# the gripper.
ARM_DIM = ACTION_DIM
GRIPPER = GRIPPER_COLUMN
JOINT_COLUMNS = tuple(range(ACTION_JOINTS.start, ACTION_JOINTS.stop))
HAND_COLUMNS = tuple(range(ACTION_HAND.start, ACTION_HAND.stop))
EE_RANGE = tuple(range(ACTION_DPOS.start, ACTION_DROT.stop))
DROT_COLUMNS = tuple(range(ACTION_DROT.start, ACTION_DROT.stop))
STATE_POSE = tuple(range(STATE_POSITION.start, STATE_ROTATION.stop))
BASE_DIMS = tuple(range(BODY_BASE.start, BODY_BASE.stop))
POSTURE_DIMS = tuple(range(BODY_TORSO.start, BODY_HEAD.stop))

# The action column each state dim mirrors, for the joints and the hand (a dead
# reading is masked in both); every other dim points at the gripper column,
# which is never dead.
STATE_MIRROR = np.full(STATE_DIM, GRIPPER_COLUMN, dtype=np.int64)
STATE_MIRROR[STATE_JOINTS] = np.arange(ACTION_JOINTS.start, ACTION_JOINTS.stop)
STATE_MIRROR[STATE_HAND] = np.arange(ACTION_HAND.start, ACTION_HAND.stop)

# The only scope. The two it replaced are refused by name.
SCOPE = "embodiment"
RETIRED_SCOPES = {
    "directory": "one normaliser per directory",
    "global": "one pooled bound over every directory, with a <scale> token beside it",
}

# The frame policy that keeps each directory's own manifest frame_ref and the
# loader's own reference draw. Any other policy overrides every entry's
# frame_ref with that value, and several policies are pooled at equal weight
# (D9, V6 4.7b).
AS_DRAWN = "as-drawn"

# How far a directory's canonical gripper may sit outside [0, 1] before its
# record is judged not to be this directory's (G27).
GRIPPER_TOLERANCE = 0.02

# A per-episode key array stores int8 indices.
EPISODE = "episode"
CONSTANT = "constant"

# The stages of config/embodiments.yaml.
STAGES = ("pretrain", "finetune")

# Families are strata: a directory's source, or for OXE the dataset it shards.
SHARD = re.compile(r"_shard\d+of\d+$")

DEFAULT_WINDOWS = 15000

# Rows are keyed by directory index * SLOT_CODE + slot when their usable dims
# are looked up; slots are below it.
SLOT_CODE = 64

# The dead-reading gate: a joint or hand column constant
# over a whole episode in at least this share of a directory's episodes that
# read it is a dead reading, masked for that directory and listed in its
# record's dead_dims (MMK2's hand_joint_8, FACTS_hand_data 2).
DEAD_SHARE = 0.9

# The wrap gate (spec 1.3): a step of a joint larger than this between two
# consecutive rows of one episode is a reading that wrapped at +-pi, and any
# one masks that joint for the key until it is resolved.
WRAP_STEP = math.pi

# The held-signal gate on the body base: at
# most this share of the steps that move a base dim may read that dim exactly
# 0. Held odometry differenced per row reads 0 between its updates, on a third
# of the moving rows. A step moves a dim when the census window around it
# moves that dim on its own (dim_moving), so a dim legitimately still while
# another moves is not judged: dx = dy = 0 turning in place, dyaw = 0 driving
# straight (the union of the three refused AgiBot at base_dx 0.4561 on the
# resampled sidecars).
HELD_SHARE_MAX = 0.05

# The own-arm gate (review data MINOR): per (key, slot), the R^2 of the arm's
# EE position on the sin and cos of its own joints must beat the R^2 on the
# other slot's joints. A swapped joint block passes every other gate. Below
# this many rows with both arms' joints and poses read, the gate has no
# evidence and refuses.
OWN_ARM_MIN_ROWS = 100

# fit_check's statistic: the 99th percentile of |y| on each decoded range column.
FIT_QUANTILE = 0.99
FIT_WEIGHTING = ("stratum mass over the stratum's trained rows per slot and column "
                 "(measure_key's quantile weights); trained columns as loss_dims")


def refuse(message: str, expected: str, fix: str) -> ValueError:
    return ValueError(f"{message}\nexpected {expected}\n{fix}")


def delta_frame(entry: dict[str, Any]) -> str:
    """The manifest entry's action.rotation_delta_frame, refused when it is not stated."""
    frame = (entry.get("action") or {}).get("rotation_delta_frame")
    if frame not in DROT_RECORD:
        raise refuse(f"directory {entry.get('name')!r} states rotation_delta_frame {frame!r}",
                     f"one of {sorted(DROT_RECORD)}",
                     "rebuild the manifest with tools/build_manifest.py")
    return str(frame)


def drot_of(entry: dict[str, Any]) -> str:
    """The datasets record's drot word for a manifest entry: DROT_RECORD of its frame."""
    return DROT_RECORD[delta_frame(entry)]


def degenerate(where: str, lo: float, hi: float, share: float | None,
               directories: Iterable[str]) -> ValueError:
    """DEGENERATE_RANGE (CX 11): a mapped dim or state axis whose bounds cannot map."""
    named = sorted(directories)
    listed = ", ".join(named[:5]) + (f" and {len(named) - 5} more" if len(named) > 5 else "")
    nonzero = "n/a" if share is None else f"{share:.4%}"
    return ValueError(
        f"DEGENERATE_RANGE: {where} has q01 {lo!r} and q99 {hi!r} (nonzero share {nonzero}; "
        f"directories {listed})\n"
        f"expected finite bounds with q01 < q99 on every mapped dim that is neither constant "
        f"nor absent\n"
        f"declare the dim invalid for the key (mask it in the layout) or fix the data")


def family_of(entry: dict[str, Any]) -> str:
    """A directory's stratum within its key: the source, or for OXE the unsharded dataset."""
    if entry["source"] == "oxe":
        return SHARD.sub("", entry["name"])
    return str(entry["source"])


def gripper_trained(entry: dict[str, Any]) -> bool:
    gripper = entry["action"].get("gripper") or {}
    return gripper.get("method") in ("measured", "declared")


def structural(entry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Per arm, the action columns [arms, 26] and state dims [arms, 29] the layout fills.

    dpos where an arm names it, drot where it names it and the convention
    trains a rotation, the gripper where it names it and its method trains
    it, joints and hand on their first count columns; the state's pose where
    the arm has one, the aperture where it has a gripper spec its method
    trains, joints and hand likewise (actions.structural_dims, in numpy).
    """
    arms = entry["arms"]
    action = np.zeros((len(arms), ARM_DIM), dtype=bool)
    state = np.zeros((len(arms), STATE_DIM), dtype=bool)
    trains_drot = drot_of(entry) not in UNTRAINED_DROT
    grip = gripper_trained(entry)
    for index, (arm, fields) in enumerate(zip(arms, entry["action"]["arms"])):
        if fields.get("dpos") is not None:
            action[index, ACTION_DPOS] = True
        if fields.get("drot") is not None and trains_drot:
            action[index, ACTION_DROT] = True
        if fields.get("gripper") is not None and grip:
            action[index, GRIPPER] = True
        layout = arm.get("state")
        if layout is not None and layout.get("position_at") is not None:
            state[index, STATE_POSE] = True
        if layout is not None and layout.get("gripper") is not None and grip:
            state[index, STATE_APERTURE] = True
        for part, action_span, state_span in (("joints", ACTION_JOINTS, STATE_JOINTS),
                                              ("hand", ACTION_HAND, STATE_HAND)):
            block = arm.get(part)
            if block is not None:
                count = int(block["count"])
                action[index, action_span.start:action_span.start + count] = True
                state[index, state_span.start:state_span.start + count] = True
    return action, state


def body_dims(body: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
    """The manifest body block's action dims [8] and state dims [8] (data.body.BodyLayout)."""
    action = np.zeros(BODY_DIM, dtype=bool)
    state = np.zeros(BODY_DIM, dtype=bool)
    if body is None:
        return action, state
    base = body["base"]
    if base in ("odometry", "still", "command"):
        action[BODY_BASE] = True
    if base in ("odometry", "still"):
        state[BODY_BASE] = True
    if base == "imu_yaw":
        action[BODY_BASE.stop - 1] = state[BODY_BASE.stop - 1] = True
    for span, count in ((BODY_TORSO, int(body["torso"])), (BODY_HEAD, int(body["head"]))):
        action[span.start:span.start + count] = True
        state[span.start:span.start + count] = True
    return action, state


@dataclass
class Directory:
    """One manifest entry, as this stage reads it."""

    index: int
    name: str
    path: str
    source: str
    family: str
    mass: float
    starts: np.ndarray
    trajectory: np.ndarray
    keys: list[str]
    slots: tuple[int, ...]
    stated: dict[str, Any]
    has_state: bool
    action_dims: np.ndarray = field(default_factory=lambda: np.zeros((0, ARM_DIM), bool))
    state_dims: np.ndarray = field(default_factory=lambda: np.zeros((0, STATE_DIM), bool))
    body: dict[str, Any] | None = None

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.starts)


def read_directories(root: Path, manifest: dict[str, Any]) -> list[Directory]:
    """Every manifest entry with its mass, episode layout, per-position key and structure."""
    version = manifest.get("schema_version", 0)
    if version != MANIFEST_SCHEMA_VERSION:
        old = (" (a v6 manifest: its rows are 19 wide, with no joints, hand or body)"
               if version in OLD_MANIFEST_SCHEMA_VERSIONS else "")
        raise refuse(f"the manifest states schema_version {version}{old}",
                     f"schema {MANIFEST_SCHEMA_VERSION} (per-arm joints and hand, the body, the "
                     f"stored v7 columns)",
                     "rebuild the manifest with this tree's build_manifest")
    entries = manifest["datasets"]
    total = sum(float(entry["weight"]) * int(entry["rows"]) for entry in entries)
    if not total > 0:
        raise refuse("the manifest's mixture mass is 0", "a positive weight x rows",
                     "check the weights")
    found = []
    for index, entry in enumerate(entries):
        starts = np.load(root / entry["episode_starts"]).astype(np.int64)
        episodes = len(starts) - 1
        if entry.get("trajectory_weights"):
            trajectory = np.load(root / entry["trajectory_weights"]).astype(np.float64)
        else:
            trajectory = np.ones(episodes)
        block = entry.get("embodiment")
        if not isinstance(block, dict):
            raise refuse(f"{entry['name']} has no embodiment block",
                         "the manifest's embodiment field",
                         "rebuild the manifest with --embodiment")
        if block["kind"] == CONSTANT:
            keys = [str(block["key"])] * episodes
        elif block["kind"] == EPISODE:
            indices = np.load(root / block["path"])
            if indices.shape[0] != episodes:
                raise refuse(f"{entry['name']} key array holds {indices.shape[0]} positions for "
                             f"{episodes} kept episodes", "one key per kept episode",
                             "rebuild the arrays with build_embodiment (R16)")
            keys = [str(block["keys"][int(i)]) for i in indices]
        else:
            raise refuse(f"{entry['name']} embodiment kind {block['kind']!r}",
                         "constant or episode", "rebuild the manifest")
        action, state = structural(entry)
        found.append(Directory(
            index=index, name=entry["name"], path=entry["path"], source=entry["source"],
            family=family_of(entry), mass=float(entry["weight"]) * int(entry["rows"]) / total,
            starts=starts, trajectory=trajectory, keys=keys,
            slots=tuple(int(arm["slot"]) for arm in entry["arms"]), stated=entry,
            has_state=bool(state.any()), action_dims=action, state_dims=state,
            body=entry.get("body")))
    return found


@dataclass
class Stratum:
    """One (key, family) group: its directories, the key's episodes in each, and masses."""

    key: str
    family: str
    members: list[tuple[Directory, np.ndarray, np.ndarray, float]] = field(default_factory=list)

    @property
    def mass(self) -> float:
        return sum(member[3] for member in self.members)


def strata_of(directories: list[Directory]) -> dict[str, dict[str, Stratum]]:
    """Per key, per family: (directory, the key's episodes, their draw probabilities, mass)."""
    found: dict[str, dict[str, Stratum]] = defaultdict(dict)
    for directory in directories:
        keys = np.asarray(directory.keys)
        total = directory.trajectory.sum()
        for key in sorted(set(directory.keys)):
            episodes = np.nonzero(keys == key)[0]
            held = directory.trajectory[episodes]
            if not held.sum() > 0:
                continue
            stratum = found[key].setdefault(directory.family, Stratum(key, directory.family))
            stratum.members.append((directory, episodes, held / held.sum(),
                                    directory.mass * held.sum() / total))
    return found


@dataclass(frozen=True)
class Window:
    """One drawn window, the fields of neo_vla.data.mixture.Draw."""

    index: int
    dataset_index: int
    name: str
    episode: int
    start: int


def draw_windows(stratum: Stratum, count: int, seed: str, base: int) -> list[Window]:
    """count windows: directory by mass within the stratum, episode by trajectory, start uniform."""
    rng = np.random.default_rng(unify.stable_seed(f"{seed}:{stratum.key}:{stratum.family}"))
    masses = np.asarray([member[3] for member in stratum.members])
    chosen = rng.choice(len(masses), size=count, p=masses / masses.sum())
    windows = []
    for number, member_index in enumerate(chosen):
        directory, episodes, probability, _ = stratum.members[member_index]
        episode = int(episodes[rng.choice(len(episodes), p=probability)])
        length = int(directory.lengths[episode])
        windows.append(Window(index=base + number, dataset_index=directory.index,
                              name=directory.name, episode=episode,
                              start=int(rng.integers(0, max(length, 1)))))
    return windows


@dataclass
class Pool:
    """One (policy, key, stratum)'s rows: arm targets, arm state, body targets and state."""

    values: list[np.ndarray] = field(default_factory=list)
    valid: list[np.ndarray] = field(default_factory=list)
    slots: list[np.ndarray] = field(default_factory=list)
    directories: list[np.ndarray] = field(default_factory=list)
    state: list[np.ndarray] = field(default_factory=list)
    state_valid: list[np.ndarray] = field(default_factory=list)
    state_slots: list[np.ndarray] = field(default_factory=list)
    state_windows: list[np.ndarray] = field(default_factory=list)
    body: list[np.ndarray] = field(default_factory=list)
    body_valid: list[np.ndarray] = field(default_factory=list)
    body_moving: list[np.ndarray] = field(default_factory=list)
    body_dim_moving: list[np.ndarray] = field(default_factory=list)
    body_directories: list[np.ndarray] = field(default_factory=list)
    body_state: list[np.ndarray] = field(default_factory=list)
    body_state_valid: list[np.ndarray] = field(default_factory=list)
    windows: int = 0
    _joined: dict[str, np.ndarray] | None = None

    def joined(self) -> dict[str, np.ndarray]:
        """The pool's parts as one array each, joined once and kept."""
        def cat(parts, shape, dtype):
            return np.concatenate(parts) if parts else np.zeros(shape, dtype=dtype)
        if self._joined is not None:
            return self._joined
        self._joined = {"values": cat(self.values, (0, ARM_DIM), np.float64),
                "valid": cat(self.valid, (0, ARM_DIM), bool),
                "slots": cat(self.slots, (0,), np.int64),
                "directories": cat(self.directories, (0,), np.int64),
                "state": cat(self.state, (0, STATE_DIM), np.float64),
                "state_valid": cat(self.state_valid, (0, STATE_DIM), bool),
                "state_slots": cat(self.state_slots, (0,), np.int64),
                "state_windows": cat(self.state_windows, (0,), np.int64),
                "body": cat(self.body, (0, BODY_DIM), np.float64),
                "body_valid": cat(self.body_valid, (0, BODY_DIM), bool),
                "body_moving": cat(self.body_moving, (0,), bool),
                "body_dim_moving": cat(self.body_dim_moving, (0, len(BASE_DIMS)), bool),
                "body_directories": cat(self.body_directories, (0,), np.int64),
                "body_state": cat(self.body_state, (0, BODY_DIM), np.float64),
                "body_state_valid": cat(self.body_state_valid, (0, BODY_DIM), bool)}
        return self._joined


def state_at(dataset: Any, draw: Any) -> tuple[Any, Any]:
    """The loader's arm state at t, [slots, 29] records and their valid dims.

    RobotDataset.state_rows(draw) is the record the sample carries (base
    offsets subtracted, before the directory's masks and the state noise), so
    the pooled bounds are the ones the loader builds: the G28 rule, for the state.
    """
    return dataset.state_rows(draw)


def body_state_at(dataset: Any, draw: Any) -> tuple[Any, Any]:
    """The loader's body state at t, the [8] record and its valid dims."""
    return dataset.body_state_rows(draw)


def as_array(value: Any, dtype=np.float64) -> np.ndarray:
    return np.asarray(value.cpu() if hasattr(value, "cpu") else value, dtype=dtype)


def dim_moving(base: np.ndarray) -> np.ndarray:
    """[N, 3] base steps of one window -> [N, 3]: the steps whose census window moves each dim.

    base_moving over the dim alone, the other two set to 0: |sum dx| or
    |sum dy| over the window above planar_cm, |sum dyaw| above dyaw_rad. A
    step whose own value of the dim is not finite is never marked. This is
    what the held-signal gate judges a dim on (check_held).
    """
    values = np.asarray(base, dtype=np.float64).reshape(-1, len(BASE_DIMS))
    marked = np.zeros(values.shape, dtype=bool)
    if values.shape[0] == 0:
        return marked
    for dim in range(values.shape[1]):
        alone = np.where(np.arange(values.shape[1]) == dim, values, 0.0)
        marked[:, dim] = base_moving(alone) & np.isfinite(values[:, dim])
    return marked


def collect(dataset: Any, windows: list[Window], draw_type: Callable[..., Any] | None,
            key: str, want_state: bool, *, command: bool = False) -> Pool:
    """The loader's targets for each window as arm-rows and body rows; the state at t.

    Every value comes from dataset.targets(draw) and the state rows: the rows
    this stage pools are the rows training normalises (G28). A value counts
    where it was read (targets.read) on a step that exists, in a column the
    directory's layout fills. A window whose key is not the stratum's is a
    join defect and refuses. A base step moves by the census rule over the
    window's own steps (base_moving; a command where it is not 0), and moves
    a base dim when that rule moves the dim on its own (dim_moving).
    """
    pool = Pool()
    for window in windows:
        draw = window if draw_type is None else draw_type(
            index=window.index, dataset_index=window.dataset_index, name=window.name,
            episode=window.episode, start=window.start)
        targets = dataset.targets(draw)
        if str(targets.key) != key:
            raise refuse(f"{window.name} episode {window.episode} was drawn for key {key} and "
                         f"the loader says {targets.key}", "one key per kept position",
                         "rebuild the embodiment arrays (R16)")
        chunk = as_array(targets.chunk)
        steps = as_array(targets.steps_valid, bool)
        dims = as_array(targets.dims_valid, bool)
        read = as_array(targets.read, bool)
        slots = np.asarray(targets.slots, dtype=np.int64)
        for arm in range(chunk.shape[0]):
            kept = steps[arm]
            pool.values.append(chunk[arm][kept])
            pool.valid.append(read[arm][kept] & dims[arm][None, :])
            pool.slots.append(np.full(int(kept.sum()), slots[arm]))
            pool.directories.append(np.full(int(kept.sum()), window.dataset_index))
        if chunk.shape[0]:
            kept = steps[0]
            body = as_array(targets.body)[kept]
            valid = as_array(targets.body_read, bool)[kept] & as_array(targets.body_dims,
                                                                        bool)[None, :]
            base = np.where(valid[:, BODY_BASE], body[:, BODY_BASE], np.nan)
            pool.body.append(body)
            pool.body_valid.append(valid)
            pool.body_moving.append(base_moving(base, command=command) if len(base)
                                    else np.zeros(0, dtype=bool))
            pool.body_dim_moving.append(dim_moving(base))
            pool.body_directories.append(np.full(int(kept.sum()), window.dataset_index))
        if want_state:
            record, measured = state_at(dataset, draw)
            record, measured = as_array(record), as_array(measured, bool)
            for slot in np.nonzero(measured.any(axis=1))[0]:
                pool.state.append(record[slot][None, :])
                pool.state_valid.append(measured[slot][None, :])
                pool.state_slots.append(np.asarray([slot]))
                pool.state_windows.append(np.asarray([window.index]))
            record, measured = body_state_at(dataset, draw)
            record, measured = as_array(record), as_array(measured, bool)
            if measured.any():
                pool.body_state.append(record[None, :])
                pool.body_state_valid.append(measured[None, :])
        pool.windows += 1
    return pool


def matrix(table: Any, column: str) -> np.ndarray:
    """An arrow column of fixed-width lists (or scalars) as a float64 [rows, width] array."""
    chunked = table.column(column).combine_chunks()
    rows = len(chunked)
    if hasattr(chunked.type, "value_type"):
        flat = np.asarray(chunked.flatten().to_numpy(zero_copy_only=False), dtype=np.float64)
        return flat.reshape(rows, -1) if rows else flat.reshape(0, 1)
    return np.asarray(chunked.to_numpy(zero_copy_only=False), dtype=np.float64).reshape(rows, 1)


@dataclass
class Scan:
    """One directory's full scan, per arm: what never moves, what is dead, wraps, is never read.

    constant/value [arms, 26]: a column the directory fills that never moves
    over its valid rows (and its value); dead [arms, 26] joint and hand
    columns whose reading is constant over a whole episode in DEAD_SHARE of
    the episodes that read it; unread [arms, 26] and state_unread [arms, 29]
    columns the layout fills and no row reads; wraps [arms, 7] steps beyond
    pi; dead_counts per column (dead episodes, read episodes); body_constant,
    body_value and body_unread [8], body_state_unread [8].
    """

    name: str
    constant: np.ndarray
    value: np.ndarray
    rows: int
    gripper: tuple[float, float] | None
    dead: np.ndarray = field(default_factory=lambda: np.zeros((0, ARM_DIM), bool))
    unread: np.ndarray = field(default_factory=lambda: np.zeros((0, ARM_DIM), bool))
    state_unread: np.ndarray = field(default_factory=lambda: np.zeros((0, STATE_DIM), bool))
    wraps: np.ndarray = field(default_factory=lambda: np.zeros((0, ARM_JOINTS), np.int64))
    wrap_steps: int = 0
    dead_counts: dict[str, tuple[int, int]] = field(default_factory=dict)
    body_constant: np.ndarray = field(default_factory=lambda: np.zeros(BODY_DIM, bool))
    body_value: np.ndarray = field(default_factory=lambda: np.full(BODY_DIM, np.nan))
    body_unread: np.ndarray = field(default_factory=lambda: np.zeros(BODY_DIM, bool))
    body_state_unread: np.ndarray = field(default_factory=lambda: np.zeros(BODY_DIM, bool))


def constancy(block: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per column of [rows, width]: (constant, its value, never read) over the finite values."""
    finite = np.isfinite(block)
    read = finite.any(axis=0)
    low = np.where(finite, block, np.inf).min(axis=0, initial=np.inf)
    high = np.where(finite, block, -np.inf).max(axis=0, initial=-np.inf)
    constant = read & (low == high)
    return constant, np.where(constant, low, np.nan), ~read


def episode_dead(block: np.ndarray, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per column of [rows, width]: episodes constant over their whole length, episodes read."""
    width = block.shape[1]
    if block.shape[0] == 0 or len(starts) < 2:
        return np.zeros(width, np.int64), np.zeros(width, np.int64)
    finite = np.isfinite(block)
    bounds = starts[:-1]
    keep = np.diff(starts) > 0
    low = np.minimum.reduceat(np.where(finite, block, np.inf), bounds, axis=0)[keep]
    high = np.maximum.reduceat(np.where(finite, block, -np.inf), bounds, axis=0)[keep]
    counted = np.add.reduceat(finite.astype(np.int64), bounds, axis=0)[keep]
    read = counted > 0
    return (read & (low == high)).sum(axis=0), read.sum(axis=0)


def wrap_counts(block: np.ndarray, starts: np.ndarray) -> tuple[np.ndarray, int]:
    """Per column, consecutive in-episode steps with |q[r+1] - q[r]| > pi; and steps checked."""
    if block.shape[0] < 2:
        return np.zeros(block.shape[1], np.int64), 0
    step = np.abs(block[1:] - block[:-1])
    inside = np.ones(block.shape[0] - 1, dtype=bool)
    inner = starts[1:-1]
    inside[inner[(inner > 0) & (inner < block.shape[0])] - 1] = False
    both = np.isfinite(step) & inside[:, None]
    return (both & (step > WRAP_STEP)).sum(axis=0).astype(np.int64), int(both.any(axis=1).sum())


def scan_directory(root: Path, directory: Directory) -> Scan:
    """Every valid row of a directory: constancy, the gripper range, dead and wrapping parts.

    Robot-frame directories: an EE column is constant when it never moves.
    Camera-frame ones: only a translation or rotation triple identically 0 is
    constant, because the drawn frame mixes the three (V6 1.2). Joints and
    hand are read from the column their target comes from (the stored state,
    or the command), every row; the dead and wrap gates read the stored state.
    """
    entry = directory.stated
    action = entry["action"]
    names = [entry["columns"]["action"]]
    if action.get("valid_column"):
        names.append(action["valid_column"])
    stored = {name: entry["columns"].get(name) for name in
              ("joints", "hand", "joints_command", "body", "body_command")}
    names += sorted({value for value in stored.values() if value is not None})
    table = feather.read_table(root / entry["cache"]["arrow"], columns=names, memory_map=True)
    raw = matrix(table, names[0])
    valid = np.ones(raw.shape[0], dtype=bool)
    if action.get("valid_column"):
        valid = matrix(table, action["valid_column"])[:, 0] > 0.5
    arms = len(action["arms"])
    camera = str(entry.get("frame_ref", "robot")).startswith("camera")
    constant = np.zeros((arms, ARM_DIM), dtype=bool)
    value = np.full((arms, ARM_DIM), np.nan)
    unread = np.zeros((arms, ARM_DIM), dtype=bool)
    state_unread = np.zeros((arms, STATE_DIM), dtype=bool)
    dead = np.zeros((arms, ARM_DIM), dtype=bool)
    wraps = np.zeros((arms, ARM_JOINTS), dtype=np.int64)
    wrap_steps = 0
    dead_counts: dict[str, tuple[int, int]] = {}
    gripper = entry["action"].get("gripper") or {}
    lows, highs = [], []
    factor = action.get("translation_cm_per_unit")
    radians = action.get("rotation_rad_per_unit")
    for index, fields in enumerate(action["arms"]):
        block = np.full((int(valid.sum()), ARM_DIM), np.nan)
        if fields.get("dpos") is not None:
            block[:, ACTION_DPOS] = raw[valid][:, list(fields["dpos"])] * np.asarray(factor)
        if fields.get("drot") is not None:
            block[:, ACTION_DROT] = raw[valid][:, list(fields["drot"])] * float(radians)
        if fields.get("gripper") is not None:
            block[:, GRIPPER] = raw[valid][:, fields["gripper"][0]]
        arm = entry["arms"][index]
        for part, span, state_span in (("joints", ACTION_JOINTS, STATE_JOINTS),
                                       ("hand", ACTION_HAND, STATE_HAND)):
            if arm.get(part) is None:
                continue
            count = int(arm[part]["count"])
            width = ARM_JOINTS if part == "joints" else HAND_JOINTS
            source = stored["joints_command"] if part == "joints" and str(
                arm[part]["target"]).startswith("command:") else stored[part]
            readings = matrix(table, stored[part])[:, index * width:index * width + count]
            readings = readings * float(arm[part]["unit_factor"])
            target = matrix(table, source)[:, index * width:index * width + count] * float(
                arm[part]["unit_factor"])
            block[:, span.start:span.start + count] = target[valid]
            dead_episodes, read_episodes = episode_dead(readings, directory.starts)
            for offset in range(count):
                name = DIM_NAMES[span.start + offset]
                dead_counts[f"{index}:{name}"] = (int(dead_episodes[offset]),
                                                  int(read_episodes[offset]))
                if read_episodes[offset] and dead_episodes[offset] >= DEAD_SHARE * read_episodes[
                        offset]:
                    dead[index, span.start + offset] = True
            state_unread[index, state_span.start:state_span.start + count] = ~np.isfinite(
                readings).any(axis=0)
            if part == "joints":
                counted, checked = wrap_counts(readings, directory.starts)
                wraps[index, :count] = counted
                wrap_steps = max(wrap_steps, checked)
        column_constant, column_value, column_unread = constancy(block)
        constant[index], value[index], unread[index] = column_constant, column_value, \
            column_unread
        if camera:
            for span in (ACTION_DPOS, ACTION_DROT):
                still = bool(constant[index, span].all() and (value[index, span] == 0).all())
                constant[index, span] = still
                value[index, span] = 0.0 if still else np.nan
        if gripper.get("method") in ("measured", "declared") and fields.get("gripper") \
                is not None and block.shape[0]:
            open_, closed = float(gripper["raw_open"]), float(gripper["raw_closed"])
            canonical = (block[:, GRIPPER] - open_) / (closed - open_)
            canonical = canonical[np.isfinite(canonical)]
            if canonical.size:
                lows.append(float(canonical.min()))
                highs.append(float(canonical.max()))
    scan = Scan(name=directory.name, constant=constant, value=value, rows=int(valid.sum()),
                gripper=(min(lows), max(highs)) if lows else None, dead=dead, unread=unread,
                state_unread=state_unread, wraps=wraps, wrap_steps=wrap_steps,
                dead_counts=dead_counts)
    if directory.body is not None and stored["body"] is not None:
        body = matrix(table, stored["body"])
        target = body.copy()
        if stored["body_command"] is not None:
            target[:, BODY_BASE] = matrix(table, stored["body_command"])[:, BODY_BASE]
        scan.body_constant, scan.body_value, scan.body_unread = constancy(target[valid])
        _, _, scan.body_state_unread = constancy(body)
    return scan


def check_gripper_ranges(key: str, scans: dict[str, Scan]) -> None:
    """G27: every directory of a key canonicalises its gripper into [0, 1] under its record."""
    outside = {name: scan.gripper for name, scan in scans.items() if scan.gripper is not None
               and (scan.gripper[0] < -GRIPPER_TOLERANCE
                    or scan.gripper[1] > 1 + GRIPPER_TOLERANCE)}
    if outside:
        listed = "; ".join(f"{name} [{lo:.3f}, {hi:.3f}]"
                           for name, (lo, hi) in sorted(outside.items())[:5])
        raise refuse(f"key {key}: directories disagree on the canonical gripper value set: "
                     f"{listed}", f"every directory's canonical gripper within [0, 1] "
                     f"(+-{GRIPPER_TOLERANCE})", "re-measure the raw_open/raw_closed of those "
                     "directories (measure_gripper_polarity); one record per directory")


def check_effector(key: str, measured: dict[str, Any], spec: dict[str, Any]) -> None:
    """A key trains only what its robot has (build_stats.check_effector, spec 8.3).

    A key with no gripper (effector none) trains no gripper column; a key
    with fewer than 7 joints trains no joint past them (a 6-DoF arm's j6 is
    invalid, never 0); a key with no hand trains no hand column, and one with
    a hand none past its declared order. measured["beyond"] counts the pooled
    values read past each declaration, which the absent columns do not show.
    """
    if spec.get("effector") == "none" and any(measured["gripper_valid"]):
        slots = [slot for slot, valid in zip(measured["slots"], measured["gripper_valid"])
                 if valid]
        raise refuse(f"key {key} states effector none in config/embodiments.yaml and trains a "
                     f"gripper at slots {slots} (directories "
                     f"{measured['reference']['directories'][:4]})",
                     "every directory of a key with no gripper masking its gripper column",
                     "mask the gripper in the layout (method masked, with its evidence), or fix "
                     "the key's effector")
    for part, count in measured.get("beyond", {}).items():
        if count:
            declared = measured["declared"].get(part)
            raise refuse(f"key {key} declares {part} {declared} and its directories read "
                         f"{count} {part} values past that ({measured['reference']['rows']} "
                         f"arm-rows)", f"no {part} column trained past the key's declaration "
                         "(a 6-DoF arm's j6 and a gripper key's hand are invalid, never 0)",
                         f"fix the key's {part} in config/embodiments.yaml or the layout's count")


def wrap_gated(key: str, slots: list[int], members: list[Directory],
               scans: dict[str, Scan]) -> dict[int, list[int]]:
    """Per slot, the joints of a key some directory steps by more than pi (spec 1.3).

    Any one such step masks the joint for the key until it is resolved;
    logged with the count and the steps it was found among.
    """
    gated: dict[int, list[int]] = {}
    for slot in slots:
        counts = np.zeros(ARM_JOINTS, dtype=np.int64)
        checked = 0
        for directory in members:
            if slot not in directory.slots:
                continue
            scan = scans[directory.name]
            counts += scan.wraps[directory.slots.index(slot)]
            checked += scan.wrap_steps
        joints = [int(joint) for joint in np.flatnonzero(counts)]
        if joints:
            gated[slot] = joints
            logger.warning("key %s slot %d: joints %s step past pi %s times over %d in-episode "
                           "steps: masked for the key (wrap gate)", key, slot,
                           [DIM_NAMES[joint] for joint in joints],
                           [int(counts[joint]) for joint in joints], checked)
    return gated


def directory_masks(directory: Directory, scan: Scan,
                    gated: dict[int, list[int]] | None = None) -> np.ndarray:
    """[n_arms, 26] True where this directory contributes a target.

    A column its layout fills, not constant in the directory, read somewhere,
    not a dead reading, and not a joint the key's wrap gate masked.
    """
    usable = directory.action_dims & ~scan.constant & ~scan.unread & ~scan.dead
    for arm, slot in enumerate(directory.slots):
        for joint in (gated or {}).get(slot, ()):
            usable[arm, joint] = False
    return usable


def band_rate(values: np.ndarray, weights: np.ndarray, valids: np.ndarray,
              slot_of: np.ndarray, slots: list[int], low: np.ndarray, high: np.ndarray,
              mapped: np.ndarray, alpha: float) -> dict[str, Any]:
    """The weighted share of arm-rows the loader will loss-mask at this alpha (V6 1.3, 6-9).

    A row is out of band when any valid mapped column lies outside
    [lo - alpha (hi - lo), hi + alpha (hi - lo)] (the gripper is exempt); the
    loader masks only that column's group, so this is an upper bound on the
    share of any one group. Each row weighs what the quantiles weighed it.
    """
    counted = np.zeros(values.shape[0], dtype=bool)
    outside = np.zeros(values.shape[0], dtype=bool)
    for position, slot in enumerate(slots):
        rows = slot_of == slot
        for column in RANGE_COLUMNS:
            if not mapped[position, column]:
                continue
            use = rows & valids[:, column]
            span = high[position, column] - low[position, column]
            values_column = values[:, column]
            outside |= use & ((values_column < low[position, column] - alpha * span)
                              | (values_column > high[position, column] + alpha * span))
            counted |= use
    row_weight = weights[:, list(RANGE_COLUMNS)].max(axis=1) if weights.shape[0] else \
        np.zeros(0)
    total = float(row_weight[counted].sum())
    return {"alpha": float(alpha), "rows": int(counted.sum()),
            "rows_out": int((outside & counted).sum()),
            "rate": float(row_weight[outside & counted].sum() / total) if total > 0 else None}


def declared_columns(declaration: dict[str, Any]) -> np.ndarray:
    """[26] bool: the joint and hand columns a key's declaration leaves out (absent)."""
    absent = np.zeros(ARM_DIM, dtype=bool)
    joints = declaration.get("joints")
    hand = declaration.get("hand")
    absent[ACTION_JOINTS.start + (0 if joints is None else int(joints["count"])):
           ACTION_JOINTS.stop] = True
    absent[ACTION_HAND.start + (0 if hand is None else int(hand["count"])):ACTION_HAND.stop] = True
    return absent


def declared_state(declaration: dict[str, Any]) -> np.ndarray:
    """[29] bool: the state joint and hand dims a key's declaration leaves out (absent)."""
    absent = np.zeros(STATE_DIM, dtype=bool)
    joints = declaration.get("joints")
    hand = declaration.get("hand")
    absent[STATE_JOINTS.start + (0 if joints is None else int(joints["count"])):
           STATE_JOINTS.stop] = True
    absent[STATE_HAND.start + (0 if hand is None else int(hand["count"])):STATE_HAND.stop] = True
    return absent


def pooled(strata: dict[str, Stratum], pools: dict[tuple[str, str], Pool],
           policies: list[str]) -> list[tuple[Stratum, dict[str, np.ndarray]]]:
    """Each (stratum, joined pool) of a key under every frame policy."""
    found = []
    for policy in policies:
        for family, stratum in strata.items():
            found.append((stratum, pools[(policy, family)].joined()))
    return found


def measure_key(key: str, strata: dict[str, Stratum], pools: dict[tuple[str, str], Pool],
                policies: list[str], scans: dict[str, Scan],
                directories: dict[int, Directory], alpha: float = 1.0,
                declaration: dict[str, Any] | None = None) -> dict[str, Any]:
    """One key's action, state and body blocks from its pooled rows and its scans."""
    declaration = declaration or {"joints": None, "hand": None, "body": None}
    members = [member for stratum in strata.values() for member in stratum.members]
    key_dirs = {member[0].index: member[0] for member in members}
    slots = sorted({slot for directory in key_dirs.values() for slot in directory.slots})
    gated = wrap_gated(key, slots, list(key_dirs.values()), scans)
    usable_by_dir = {index: directory_masks(directory, scans[directory.name], gated)
                     for index, directory in key_dirs.items()}
    beyond_action = declared_columns(declaration)
    values, weights, valids, slot_rows, dir_rows = [], [], [], [], []
    windows = 0
    beyond = {"joints": 0, "hand": 0}
    for stratum, joined in pooled(strata, pools, policies):
        valid = joined["valid"].copy()
        for part, columns in (("joints", JOINT_COLUMNS), ("hand", HAND_COLUMNS)):
            past = [column for column in columns if beyond_action[column]]
            beyond[part] += int(valid[:, past].sum()) if past else 0
        if valid.shape[0]:
            code = joined["directories"] * SLOT_CODE + joined["slots"]
            unique, inverse = np.unique(code, return_inverse=True)
            table = np.stack([
                usable_by_dir[int(c // SLOT_CODE)][
                    directories[int(c // SLOT_CODE)].slots.index(int(c % SLOT_CODE))]
                for c in unique])
            valid &= table[inverse]
        weight = np.zeros_like(valid, dtype=np.float64)
        for slot in slots:
            in_slot = joined["slots"] == slot
            weight[in_slot] = fit_weights(valid[in_slot], stratum.mass) / len(policies)
        values.append(joined["values"])
        weights.append(weight)
        valids.append(valid)
        slot_rows.append(joined["slots"])
        dir_rows.append(joined["directories"])
    for policy in policies:
        for family in strata:
            windows += pools[(policy, family)].windows
    values = np.concatenate(values) if values else np.zeros((0, ARM_DIM))
    weights = np.concatenate(weights) if weights else np.zeros((0, ARM_DIM))
    valids = np.concatenate(valids) if valids else np.zeros((0, ARM_DIM), dtype=bool)
    slot_of = np.concatenate(slot_rows) if slot_rows else np.zeros(0, dtype=np.int64)
    dir_of = np.concatenate(dir_rows) if dir_rows else np.zeros(0, dtype=np.int64)
    if not valids.any():
        raise refuse(f"key {key} pooled 0 valid arm-rows over {windows} windows (n = 0)",
                     "rows to measure", "check the key's directories and the draw")
    low = np.full((len(slots), ARM_DIM), np.nan)
    high = np.full((len(slots), ARM_DIM), np.nan)
    constant = np.zeros((len(slots), ARM_DIM), dtype=bool)
    absent = np.zeros((len(slots), ARM_DIM), dtype=bool)
    gripper_valid = []
    for position, slot in enumerate(slots):
        holders = [directory for directory in key_dirs.values() if slot in directory.slots]
        for column in RANGE_COLUMNS:
            filled = [directory for directory in holders
                      if directory.action_dims[directory.slots.index(slot)][column]]
            if beyond_action[column] or column in gated.get(slot, ()) or not filled:
                absent[position, column] = True
                continue
            usable = [directory for directory in filled
                      if usable_by_dir[directory.index][directory.slots.index(slot)][column]]
            if not usable:
                stills = {float(scans[directory.name].value[directory.slots.index(slot), column])
                          for directory in filled}
                stills = {value for value in stills if np.isfinite(value)}
                if len(stills) == 1:
                    constant[position, column] = True
                    low[position, column] = high[position, column] = stills.pop()
                    continue
                if not stills:
                    absent[position, column] = True
                    continue
                raise refuse(f"key {key} slot {slot} {DIM_NAMES[column]} is constant in every "
                             f"directory, at different values {sorted(stills)[:4]}",
                             "one constant value for a key-constant dim",
                             "split the key or mask the dim")
            rows = (slot_of == slot) & valids[:, column]
            lo = weighted_quantile(values[rows, column], weights[rows, column],
                                   QUANTILES[0] / 100) if rows.any() else float("nan")
            hi = weighted_quantile(values[rows, column], weights[rows, column],
                                   QUANTILES[1] / 100) if rows.any() else float("nan")
            if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                share = float((values[rows, column] != 0).mean()) if rows.any() else None
                names = {directories[int(index)].name for index in np.unique(dir_of[rows])}
                raise degenerate(f"key {key} slot {slot} dim {column} ({DIM_NAMES[column]})", lo,
                                 hi, share, names or {directory.name for directory in usable})
            low[position, column], high[position, column] = lo, hi
        grip = [directory for directory in holders
                if usable_by_dir[directory.index][directory.slots.index(slot)][GRIPPER]]
        # The gripper column is never constant or absent: its map is 2 g_c - 1
        # whatever the data, so it always states low 0 and high 1, and whether
        # the key trains it is gripper_valid alone.
        gripper_valid.append(bool(grip))
        low[position, GRIPPER], high[position, GRIPPER] = 0.0, 1.0
    band = band_rate(values, weights, valids, slot_of, slots, low, high, ~(constant | absent),
                     alpha)
    state = measure_state(key, slots, strata, pools, policies, declaration, gated,
                          [directory for directory in key_dirs.values()], scans)
    body = measure_body(key, strata, pools, policies, declaration.get("body"), key_dirs, scans)
    reference = {
        "directories": sorted(directory.name for directory in key_dirs.values()),
        "rows": int(valids.any(axis=1).sum()), "windows": windows,
        "strata": {family: round(stratum.mass, 8) for family, stratum in sorted(strata.items())},
        "frame_policies": {policy: 1.0 / len(policies) for policy in policies},
    }
    return {"slots": slots, "low": low, "high": high, "constant": constant, "absent": absent,
            "gripper_valid": gripper_valid, "state": state, "body": body,
            "reference": reference, "band": band, "gated": gated, "beyond": beyond,
            "declared": {"joints": None if declaration.get("joints") is None
                         else declaration["joints"]["count"],
                         "hand": None if declaration.get("hand") is None
                         else declaration["hand"]["count"]},
            "drawn": {int(index) for index in np.unique(dir_of)}}


def measure_state(key: str, slots: list[int], strata: dict[str, Stratum],
                  pools: dict[tuple[str, str], Pool], policies: list[str],
                  declaration: dict[str, Any], gated: dict[int, list[int]],
                  members: list[Directory], scans: dict[str, Scan]) -> dict[str, Any]:
    """Per (slot, dim) the state's weighted q01/q99, constant and absent flags [slots, 29].

    Position, joints and hand are ranged; a position axis needs lo < hi
    (DEGENERATE_RANGE), a joint or hand dim with lo == hi is constant. The 6D
    and the aperture are absent where no row read them and are never
    constant. Dims past the key's declaration, wrap-gated joints and dims no
    pooled row read are absent.
    """
    beyond = declared_state(declaration)
    low = np.full((len(slots), STATE_DIM), np.nan)
    high = np.full((len(slots), STATE_DIM), np.nan)
    constant = np.zeros((len(slots), STATE_DIM), dtype=bool)
    absent = np.zeros((len(slots), STATE_DIM), dtype=bool)
    parts = pooled(strata, pools, policies)
    for position, slot in enumerate(slots):
        filled = np.zeros(STATE_DIM, dtype=bool)
        for directory in members:
            if slot in directory.slots:
                arm = directory.slots.index(slot)
                filled |= directory.state_dims[arm] & ~scans[directory.name].state_unread[arm] \
                    & ~scans[directory.name].dead[arm][STATE_MIRROR]
        rows, valid, weight = [], [], []
        for stratum, joined in parts:
            in_slot = joined["state_slots"] == slot
            count = int(in_slot.sum())
            if count:
                rows.append(joined["state"][in_slot])
                valid.append(joined["state_valid"][in_slot])
                weight.append(np.full(count, stratum.mass / count / len(policies)))
        if rows:
            values, mask, weights = np.concatenate(rows), np.concatenate(valid), \
                np.concatenate(weight)
        else:
            values = np.zeros((0, STATE_DIM))
            mask = np.zeros((0, STATE_DIM), bool)
            weights = np.zeros(0)
        for dim in range(STATE_DIM):
            joint = dim - STATE_JOINTS.start if dim in range(STATE_JOINTS.start,
                                                              STATE_JOINTS.stop) else None
            use = mask[:, dim] & np.isfinite(values[:, dim])
            if beyond[dim] or not filled[dim] or not use.any() or (
                    joint is not None and joint in gated.get(slot, ())):
                absent[position, dim] = True
                continue
            if dim not in STATE_RANGE:
                continue
            lo = weighted_quantile(values[use, dim], weights[use], QUANTILES[0] / 100)
            hi = weighted_quantile(values[use, dim], weights[use], QUANTILES[1] / 100)
            name = STATE_DIM_NAMES[dim]
            if dim in range(STATE_POSITION.start, STATE_POSITION.stop):
                if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                    raise degenerate(f"key {key} slot {slot} state {name}", lo, hi, None, [key])
            elif lo == hi:
                constant[position, dim] = True
            elif not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                raise degenerate(f"key {key} slot {slot} state {name}", lo, hi, None, [key])
            low[position, dim], high[position, dim] = lo, hi
        for span in (STATE_POSITION, STATE_ROTATION):
            if absent[position, span].any():
                absent[position, span] = True
                low[position, span] = high[position, span] = np.nan
    return {"low": low, "high": high, "constant": constant, "absent": absent}


def measure_body(key: str, strata: dict[str, Stratum], pools: dict[tuple[str, str], Pool],
                 policies: list[str], declared: dict[str, Any] | None,
                 directories: dict[int, Directory], scans: dict[str, Scan]) -> dict[str, Any]:
    """The key's body block: the base by its symmetric moving-row scale, torso and head q01/q99.

    The base action is fitted on the pooled steps the census rule marks
    moving (a command where it is not 0), per row weighted by its stratum's
    mass; a dim with no moving row, or a scale below its floor, is constant at
    0 (a still base), and one never read is absent (fit_base_scale). The base
    state is the last step's motion, the same quantity one row on, so it takes
    the action's scale; a commanded base has no state (D5). The held-signal
    gate refuses a base dim exactly 0 too often on the steps that move it.
    """
    if declared is None:
        return {"declared": "absent"}
    base = declared["base"]
    torso = len(declared["torso"] or ())
    head = len(declared["head"] or ())
    rows, valid, moving, weight, state_rows, state_valid, state_weight = [], [], [], [], [], [], []
    dim_moves = []
    for stratum, joined in pooled(strata, pools, policies):
        count = int(joined["body"].shape[0])
        if count:
            rows.append(joined["body"])
            valid.append(joined["body_valid"])
            moving.append(joined["body_moving"])
            dim_moves.append(joined["body_dim_moving"])
            weight.append(np.full(count, stratum.mass / count / len(policies)))
        held = int(joined["body_state"].shape[0])
        if held:
            state_rows.append(joined["body_state"])
            state_valid.append(joined["body_state_valid"])
            state_weight.append(np.full(held, stratum.mass / held / len(policies)))
    values = np.concatenate(rows) if rows else np.zeros((0, BODY_DIM))
    mask = np.concatenate(valid) if valid else np.zeros((0, BODY_DIM), bool)
    moves = np.concatenate(moving) if moving else np.zeros(0, bool)
    along = np.concatenate(dim_moves) if dim_moves else np.zeros((0, len(BASE_DIMS)), bool)
    weights = np.concatenate(weight) if weight else np.zeros(0)
    allowed = np.zeros(BODY_DIM, dtype=bool)
    allowed[BODY_TORSO.start:BODY_TORSO.start + torso] = True
    allowed[BODY_HEAD.start:BODY_HEAD.start + head] = True
    action = {"low": [None] * BODY_DIM, "high": [None] * BODY_DIM,
              "constant": [False] * BODY_DIM, "absent": [True] * BODY_DIM}
    state = {"low": [None] * BODY_DIM, "high": [None] * BODY_DIM,
             "constant": [False] * BODY_DIM, "absent": [True] * BODY_DIM}
    rows_used = 0
    action_base = {"odometry": BASE_DIMS, "still": BASE_DIMS, "command": BASE_DIMS,
                   "imu_yaw": BASE_DIMS[-1:], "none": ()}[base]
    state_base = {"odometry": BASE_DIMS, "still": BASE_DIMS, "imu_yaw": BASE_DIMS[-1:],
                  "command": (), "none": ()}[base]
    if action_base:
        steps = np.where(mask[:, BODY_BASE], values[:, BODY_BASE], np.nan)
        for dim in BASE_DIMS:
            if dim not in action_base:
                steps[:, dim] = np.nan
        scale = fit_base_scale(steps, moves, weights) if steps.shape[0] else None
        if scale is not None:
            rows_used = scale.rows
            bounds = scale.bounds()
            for dim in action_base:
                if base == "still" and bounds["absent"][dim] is False \
                        and bounds["constant"][dim] is False:
                    raise refuse(f"key {key} reads a still base and its {BODY_DIM_NAMES[dim]} "
                                 f"moves on {scale.rows} rows", "a still base constant at 0",
                                 "fix the key's body base in config/embodiments.yaml")
                for name in ("low", "high", "constant", "absent"):
                    action[name][dim] = bounds[name][dim]
                if base != "command":
                    check_held(key, dim, steps[:, dim], along[:, dim - BODY_BASE.start])
    for dim in state_base:
        for name in ("low", "high", "constant", "absent"):
            state[name][dim] = action[name][dim]
    state_values = np.concatenate(state_rows) if state_rows else np.zeros((0, BODY_DIM))
    state_mask = np.concatenate(state_valid) if state_valid else np.zeros((0, BODY_DIM), bool)
    state_weights = np.concatenate(state_weight) if state_weight else np.zeros(0)
    for dim in POSTURE_DIMS:
        if not allowed[dim]:
            continue
        for side, (pool_values, pool_mask, pool_weights) in (
                (action, (values, mask, weights)),
                (state, (state_values, state_mask, state_weights))):
            use = pool_mask[:, dim] & np.isfinite(pool_values[:, dim])
            if not use.any():
                continue
            lo = weighted_quantile(pool_values[use, dim], pool_weights[use], QUANTILES[0] / 100)
            hi = weighted_quantile(pool_values[use, dim], pool_weights[use], QUANTILES[1] / 100)
            if lo == hi:
                side["constant"][dim] = True
            elif not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                raise degenerate(f"key {key} body {BODY_DIM_NAMES[dim]}", lo, hi, None, [key])
            side["low"][dim], side["high"][dim] = float(lo), float(hi)
            side["absent"][dim] = False
    return {"declared": "present", "base": base, "torso": declared["torso"],
            "head": declared["head"], "action": action, "state": state,
            "base_scale_rows": int(rows_used)}


def check_held(key: str, dim: int, steps: np.ndarray, moving: np.ndarray) -> None:
    """The held-signal gate: a base dim exactly 0 on at most HELD_SHARE_MAX of its moving steps.

    moving [N] marks the steps whose census window moves this dim on its own
    (dim_moving). A reading held between its updates and differenced per row
    reads 0 inside such a window, whichever dims are held; a dim that is
    still while another moves (dx = dy = 0 turning in place, dyaw = 0 driving
    straight) has no window that moves it and is not judged there.
    """
    rows = moving & np.isfinite(steps)
    count = int(rows.sum())
    name = BODY_DIM_NAMES[dim]
    if count == 0:
        logger.info("key %s %s: no step moves it (held-signal gate not judged)", key, name)
        return
    share = float((steps[rows] == 0.0).mean())
    logger.info("key %s %s: %.6f of %d steps moving along it exactly 0 (held-signal gate)",
                key, name, share, count)
    if share > HELD_SHARE_MAX:
        raise refuse(f"key {key} {name}: {share:.4f} of {count} steps moving along {name} are "
                     f"exactly 0", f"at most {HELD_SHARE_MAX} (a reading held between its "
                     "updates and differenced per row reads 0 between them)",
                     "resample the held readings before differencing (the AgiBot sidecar writer "
                     "for AgiBot) and rebuild the sidecar")


def own_arm_r2(key: str, strata: dict[str, Stratum], pools: dict[tuple[str, str], Pool],
               policies: list[str], slots: list[int], declaration: dict[str, Any],
               state: dict[str, Any]) -> dict[str, Any] | None:
    """The own-arm gate: per slot, R^2 of EE xyz on its own joints against the other slot's.

    Rows are draws where both slots' joints (the declared count) and poses
    were read at t. Features are sin and cos of the joints and a constant;
    R^2 is the mean over the three axes. None for a key without two slots,
    joints, or a pose on both slots (nothing to swap or no EE to hold).
    """
    joints = declaration.get("joints")
    if len(slots) != 2 or joints is None:
        return None
    count = int(joints["count"])
    if any(state["absent"][position, STATE_POSITION].any() for position in range(2)):
        return None
    pairs: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)
    for _, joined in pooled(strata, pools, policies):
        for row in range(joined["state"].shape[0]):
            values, valid = joined["state"][row], joined["state_valid"][row]
            needed = list(range(STATE_JOINTS.start, STATE_JOINTS.start + count)) \
                + list(range(STATE_POSITION.start, STATE_POSITION.stop))
            if valid[needed].all():
                pairs[int(joined["state_windows"][row])][int(joined["state_slots"][row])] = (
                    values[STATE_JOINTS.start:STATE_JOINTS.start + count],
                    values[STATE_POSITION])
    both = [found for found in pairs.values() if set(found) == set(slots)]
    if len(both) < OWN_ARM_MIN_ROWS:
        raise refuse(f"key {key}: the own-arm gate has {len(both)} draws with both slots' "
                     f"joints and poses read", f"at least {OWN_ARM_MIN_ROWS} (a gate over "
                     "nothing passes on no evidence)",
                     "draw more windows of the key (--windows-per-stratum)")
    report = {"rows": len(both)}
    for slot in slots:
        other = [one for one in slots if one != slot][0]
        target = np.stack([found[slot][1] for found in both])
        own = np.stack([found[slot][0] for found in both])
        cross = np.stack([found[other][0] for found in both])
        report[slot] = {"own": r_squared(own, target), "cross": r_squared(cross, target)}
    return report


def r_squared(joints: np.ndarray, target: np.ndarray) -> float:
    """Mean over target columns of R^2 of a least-squares fit on [sin q, cos q, 1]."""
    features = np.concatenate([np.sin(joints), np.cos(joints), np.ones((joints.shape[0], 1))],
                              axis=1)
    coefficients, *_ = np.linalg.lstsq(features, target, rcond=None)
    residual = ((target - features @ coefficients) ** 2).sum(axis=0)
    total = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    return float(np.mean(1.0 - residual / np.where(total > 0, total, 1.0)))


def check_own_arm(key: str, report: dict[str, Any] | None) -> None:
    """Refuse a key whose arm's EE is better explained by the other arm's joints."""
    if report is None:
        return
    for slot, found in report.items():
        if slot == "rows":
            continue
        logger.info("key %s slot %s: EE R^2 %.4f on its own joints, %.4f on the other's "
                    "(%d draws; own-arm gate)", key, slot, found["own"], found["cross"],
                    report["rows"])
        if not found["own"] > found["cross"]:
            raise refuse(f"key {key} slot {slot}: the EE position fits the other slot's joints "
                         f"(R^2 {found['cross']:.4f}) at least as well as its own "
                         f"({found['own']:.4f}) over {report['rows']} draws",
                         "each arm's EE explained best by its own joints (a swapped joint "
                         "block passes every other gate)",
                         "check the converter's arm order for the key's directories")


def rounded(array: np.ndarray, absent: np.ndarray | None = None) -> list[list[Any]]:
    """Bounds as JSON: 8 decimals, None where absent (C10)."""
    rows = []
    for position, row in enumerate(np.asarray(array)):
        rows.append([None if (absent is not None and absent[position][dim])
                     or not np.isfinite(value) else round(float(value), 8)
                     for dim, value in enumerate(row)])
    return rows


def body_document(body: dict[str, Any]) -> dict[str, Any]:
    """The key's body block as embodiments.json states it."""
    if body["declared"] == "absent":
        return {"declared": "absent"}

    def side(block: dict[str, Any], name: str) -> list[Any]:
        return [None if block["absent"][dim] or block[name][dim] is None
                else round(float(block[name][dim]), 8) for dim in range(BODY_DIM)]

    document = {"declared": "present", "base": body["base"], "torso": body["torso"],
                "head": body["head"], "base_scale_rows": int(body["base_scale_rows"])}
    for prefix in ("action", "state"):
        block = body[prefix]
        document[f"{prefix}_low"] = side(block, "low")
        document[f"{prefix}_high"] = side(block, "high")
        document[f"{prefix}_constant"] = [bool(flag) for flag in block["constant"]]
        document[f"{prefix}_absent"] = [bool(flag) for flag in block["absent"]]
    return document


def entry_document(key: str, measured: dict[str, Any], spec: dict[str, Any],
                   manifest_name: str, manifest_sha: str) -> dict[str, Any]:
    """One key's embodiments.json v2 entry, before its digest."""
    absent = measured["absent"]
    state = measured["state"]
    joints = key_table.joints_of(key, spec)
    hand = key_table.hand_of(key, spec)

    def block(span: slice, bounds: np.ndarray) -> list[list[Any]]:
        return rounded(bounds[:, span], state["absent"][:, span])

    return {
        "slots": measured["slots"],
        "joints": joints,
        "hand": hand,
        "action": {"low": rounded(measured["low"], absent),
                   "high": rounded(measured["high"], absent),
                   "constant": measured["constant"].tolist(), "absent": absent.tolist(),
                   "mask": list(MASK), "gripper_valid": measured["gripper_valid"],
                   "from": None},
        "state": {"position_low": block(STATE_POSITION, state["low"]),
                  "position_high": block(STATE_POSITION, state["high"]),
                  "joint_low": block(STATE_JOINTS, state["low"]),
                  "joint_high": block(STATE_JOINTS, state["high"]),
                  "hand_low": block(STATE_HAND, state["low"]),
                  "hand_high": block(STATE_HAND, state["high"]),
                  "constant": state["constant"].tolist(), "absent": state["absent"].tolist()},
        "body": body_document(measured["body"]),
        "ee_frame_fix": spec.get("ee_frame_fix"),
        "clause": spec["clause"],
        "reference": {"manifest": manifest_name, "manifest_sha256": manifest_sha,
                      **measured["reference"]},
        "stage": spec.get("stage"),
    }


def read_keys(path: Path) -> dict[str, Any]:
    """config/embodiments.yaml's key table (neo_vla_embodiment_keys_v2; a v1 one is refused)."""
    return key_table.load(path)["keys"]


def scope_type(value: str) -> str:
    if value in RETIRED_SCOPES:
        raise argparse.ArgumentTypeError(
            f"--scope {value} is retired ({RETIRED_SCOPES[value]}); the normaliser is per "
            f"embodiment key (V6 1). Use --scope {SCOPE}")
    if value != SCOPE:
        raise argparse.ArgumentTypeError(f"--scope {value!r}; expected {SCOPE}")
    return value


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True, help="the manifest that is trained")
    parser.add_argument("--out", required=True)
    parser.add_argument("--embodiments", default="config/embodiments.yaml",
                        help="key table (neo_vla_embodiment_keys_v2): clause, stage, frame fix "
                             "and the joints, hand and body each key declares")
    parser.add_argument("--scope", type=scope_type, default=SCOPE)
    parser.add_argument("--quantiles", type=float, nargs=2, default=list(QUANTILES))
    parser.add_argument("--band-alpha", type=float, default=1.0)
    parser.add_argument("--windows-per-stratum", type=int, default=DEFAULT_WINDOWS)
    parser.add_argument("--frame-policies", default=AS_DRAWN,
                        help="as-drawn, or a comma list of frame_ref values pooled at equal "
                             "weight (finetune keys, D9)")
    parser.add_argument("--constant-scan", choices=["full"], default="full")
    parser.add_argument("--inherit-action", action="append", default=[],
                        help="KEY=SRC: copy SRC's action block from --from for KEY")
    parser.add_argument("--from", dest="source_file",
                        help="the lineage's embodiments.json (the init step's)")
    parser.add_argument("--reuse", action="append", default=[],
                        help="FILE#KEY: take KEY's entry from FILE unchanged")
    parser.add_argument("--check-fit", action="store_true",
                        help="hold every key to R14 under each frame policy")
    parser.add_argument("--cache-index",
                        help="the cache index the arrow caches were written under; hashed into "
                             "the provenance so prepare_data can tell a stale normaliser")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", default="stats")
    parser.add_argument("--n-slots", type=int, default=2)
    parser.add_argument("--n-views", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=30)
    parser.add_argument("--action-dim", type=int, default=ACTION_DIM)
    parser.add_argument("--max-grid", type=int, nargs=2, default=[8, 11])
    args = parser.parse_args(argv)
    if tuple(args.quantiles) != tuple(float(q) for q in QUANTILES):
        parser.error(f"--quantiles {args.quantiles}: the transform is {TRANSFORM} at "
                     f"{list(QUANTILES)} (Embodiments.load refuses any other)")
    if not args.band_alpha > 0:
        parser.error(f"--band-alpha {args.band_alpha}: expected > 0")
    if args.windows_per_stratum < 1:
        parser.error("--windows-per-stratum must be at least 1")
    if args.inherit_action and not args.source_file:
        parser.error("--inherit-action needs --from, the file whose entries it copies")
    if args.action_dim != ACTION_DIM:
        parser.error(f"--action-dim {args.action_dim}: the v7 row is {ACTION_DIM} wide")
    return args


def policies_of(text: str) -> list[str]:
    policies = [part.strip() for part in text.split(",") if part.strip()]
    if not policies or (AS_DRAWN in policies and len(policies) > 1):
        raise refuse(f"--frame-policies {text!r}", f"{AS_DRAWN}, or a list of frame_ref values",
                     "name the policies the key's arms plan to train under")
    return policies


# The dataset each worker process opens per policy. Built lazily in the worker
# so a forked pool shares nothing but the arguments.
_FACTORY: Callable[[str], Any] | None = None
_DATASETS: dict[str, Any] = {}


def _dataset_for(policy: str) -> Any:
    if policy not in _DATASETS:
        _DATASETS[policy] = _FACTORY(policy)
    return _DATASETS[policy]


def _collect_job(job: tuple[str, str, list[Window], bool, bool]) -> Pool:
    policy, key, windows, want_state, command = job
    dataset = _dataset_for(policy)
    return collect(dataset, windows, dataset.draw_type, key, want_state, command=command)


def pool_all(jobs: list[tuple[str, str, list[Window], bool, bool]], workers: int) -> list[Pool]:
    if workers <= 1:
        return [_collect_job(job) for job in jobs]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        return pool.map(_collect_job, jobs, chunksize=1)


def declarations_of(keys_table: dict[str, Any] | None, key: str) -> dict[str, Any]:
    """A key's joints, hand and body declarations (none when the table does not hold it)."""
    if keys_table is None or key not in keys_table:
        return {"joints": None, "hand": None, "body": None}
    spec = keys_table[key]
    return {"joints": key_table.joints_of(key, spec), "hand": key_table.hand_of(key, spec),
            "body": key_table.body_of(key, spec)}


def measure_all(root: Path, manifest: dict[str, Any], factory: Callable[[str], Any],
                args: argparse.Namespace, skip: set[str],
                keys_table: dict[str, Any] | None = None,
                ) -> tuple[dict[str, Any], dict[str, Scan], list[Directory]]:
    """Every measured key's blocks, and every directory's scan."""
    global _FACTORY
    _FACTORY = factory
    _DATASETS.clear()
    directories = read_directories(root, manifest)
    by_index = {directory.index: directory for directory in directories}
    policies = policies_of(args.frame_policies)
    strata = strata_of(directories)
    if args.workers > 1:
        with multiprocessing.get_context("fork").Pool(args.workers) as pool:
            scanned = pool.starmap(scan_directory, [(root, d) for d in directories])
    else:
        scanned = [scan_directory(root, directory) for directory in directories]
    scans = {scan.name: scan for scan in scanned}
    log_gates(scans, directories)
    jobs, where = [], []
    base = 0
    for key in sorted(strata):
        if key in skip:
            continue
        check_gripper_ranges(key, {member[0].name: scans[member[0].name]
                                   for stratum in strata[key].values()
                                   for member in stratum.members})
        want_state = any(member[0].has_state or member[0].body is not None
                         for stratum in strata[key].values() for member in stratum.members)
        body = declarations_of(keys_table, key)["body"]
        command = body is not None and body["base"] == "command"
        for policy in policies:
            for family, stratum in sorted(strata[key].items()):
                windows = draw_windows(stratum, args.windows_per_stratum,
                                       f"{args.seed}:{policy}", base)
                base += len(windows)
                jobs.append((policy, key, windows, want_state, command))
                where.append((key, policy, family))
    pools: dict[str, dict[tuple[str, str], Pool]] = defaultdict(dict)
    for (key, policy, family), found in zip(where, pool_all(jobs, args.workers)):
        pools[key][(policy, family)] = found
    measured = {}
    for key in sorted(pools):
        declaration = declarations_of(keys_table, key)
        measured[key] = measure_key(key, strata[key], pools[key], policies, scans, by_index,
                                    args.band_alpha, declaration)
        measured[key]["own_arm"] = own_arm_r2(key, strata[key], pools[key], policies,
                                              measured[key]["slots"], declaration,
                                              measured[key]["state"])
        check_own_arm(key, measured[key]["own_arm"])
        never = sorted(set(measured[key]["reference"]["directories"])
                       - {by_index[index].name for index in measured[key]["drawn"]})
        logger.info("key %s: %d arm-rows over %d windows in %d directories (%d never drawn%s)",
                    key, measured[key]["reference"]["rows"], measured[key]["reference"]["windows"],
                    len(measured[key]["reference"]["directories"]), len(never),
                    (": " + ", ".join(never[:8])) if never else "")
        band = measured[key]["band"]
        logger.info("key %s: %s of %d arm-rows out of band at alpha %g (%d rows; the loader "
                    "masks the column's group)", key, "n/a" if band["rate"] is None
                    else f"{band['rate']:.4%}", band["rows"], band["alpha"], band["rows_out"])
    return measured, scans, directories


def log_gates(scans: dict[str, Scan], directories: list[Directory]) -> None:
    """The dead-reading and never-read gates, per directory, with their denominators."""
    dead_total = 0
    for directory in directories:
        scan = scans[directory.name]
        for arm, slot in enumerate(directory.slots):
            dead = [DIM_NAMES[column] for column in np.flatnonzero(
                scan.dead[arm] & directory.action_dims[arm])]
            if dead:
                dead_total += len(dead)
                counts = {name: scan.dead_counts.get(f"{arm}:{name}") for name in dead}
                logger.warning("%s slot %d: dead readings %s (constant episodes of read "
                               "episodes: %s; masked for the directory)", directory.name, slot,
                               dead, counts)
    logger.info("dead-reading gate: %d columns masked over %d directories", dead_total,
                len(directories))


def real_factory(root: Path, manifest_path: Path, manifest: dict[str, Any],
                 args: argparse.Namespace, scratch: Path) -> Callable[[str], Any]:
    """RobotDataset in targets-only mode, per frame policy."""
    stamp = manifest["prompt"]["stamp"]

    def build(policy: str) -> Any:
        from neo_vla.data.dataset import RobotDataset
        from neo_vla.data.mixture import Draw
        path = manifest_path
        if policy != AS_DRAWN:
            document = json.loads(manifest_path.read_text(encoding="utf-8"))
            for entry in document["datasets"]:
                entry["frame_ref"] = policy
            path = scratch / f"frames_{policy.replace(':', '_')}.json"
            path.write_text(json.dumps(document), encoding="utf-8")
        dataset = RobotDataset(
            manifest=path, root=root, max_grid=tuple(args.max_grid), pixels_per_token=32,
            n_views=args.n_views, n_slots=args.n_slots, chunk=args.chunk,
            action_dim=args.action_dim, gripper_dims=(GRIPPER,),
            tokenizer_id=stamp["tokenizer_id"], max_text_tokens=stamp["max_text_tokens"],
            intrinsics_mode=stamp["intrinsics_mode"], pixel_convention="neo_native",
            epoch_samples=1, seed=0, normaliser=None, mode="targets_only")
        dataset.draw_type = Draw
        return dataset

    return build


def decodable_of(entries: list[Any], slot: int) -> np.ndarray:
    return np.any([entry.decodable(slot) for entry in entries], axis=0)


def holdable_of(entries: list[Any], slot: int) -> np.ndarray:
    return np.any([entry.state_valid(slot) for entry in entries], axis=0)


def directory_record_masks(directory: Directory, scan: Scan, entries: list[Any],
                           gated: dict[str, dict[int, list[int]]]) -> dict[str, Any]:
    """One directory's masks: what it trains and holds valid, per slot and body.

    columns: the layout's columns less the directory's constants, dead
    readings, never-read columns, the wrap-gated joints of its keys and what
    no key of it decodes; state_dims likewise over the 29 (the pose one group,
    the aperture with its trained gripper); the body's from the manifest's
    body block less its constants, never-read dims and what no key holds.
    """
    columns = directory_masks(directory, scan)
    states = directory.state_dims & ~scan.state_unread
    states[:, list(range(STATE_JOINTS.start, STATE_JOINTS.stop))] &= ~scan.dead[:, ACTION_JOINTS]
    states[:, list(range(STATE_HAND.start, STATE_HAND.stop))] &= ~scan.dead[:, ACTION_HAND]
    for arm, slot in enumerate(directory.slots):
        for key in sorted(set(directory.keys)):
            for joint in gated.get(key, {}).get(slot, ()):
                columns[arm, joint] = False
                states[arm, STATE_JOINTS.start + joint] = False
        columns[arm] &= decodable_of(entries, slot)
        states[arm] &= holdable_of(entries, slot)
        for span in (STATE_POSITION, STATE_ROTATION):
            if not states[arm, span].all():
                states[arm, span] = False
    body_action, body_state = body_dims(directory.body)
    present = [entry.body for entry in entries if entry.body.present]
    if present:
        body_columns = body_action & ~scan.body_constant & ~scan.body_unread \
            & np.any([body.decodable() for body in present], axis=0)
        body_states = body_state & ~scan.body_state_unread \
            & np.any([~body.state_absent for body in present], axis=0)
    else:
        body_columns = np.zeros(BODY_DIM, dtype=bool)
        body_states = np.zeros(BODY_DIM, dtype=bool)
    if body_states.all():
        raise refuse(f"directory {directory.name} holds all {BODY_DIM} body state dims valid",
                     "at most 7: a measured body with all 8 valid, a still base and a mid-range "
                     "posture equals the no-history input",
                     "mask a body dim the key does not record, or give NO_HISTORY its own input")
    dead = {str(slot): [DIM_NAMES[column] for column in np.flatnonzero(
        scan.dead[arm] & directory.action_dims[arm])] for arm, slot in enumerate(directory.slots)}
    return {"columns": columns.tolist(), "state_dims": states.tolist(),
            "body_columns": body_columns.tolist(), "body_state_dims": body_states.tolist(),
            "dead_dims": {slot: names for slot, names in dead.items() if names}}


def datasets_records(directories: list[Directory], scans: dict[str, Scan],
                     entries: dict[str, Any] | None = None,
                     gated: dict[str, dict[int, list[int]]] | None = None) -> dict[str, Any]:
    """Every manifest directory's record: the six client inputs, digested, and its masks.

    entries are the built Entry objects per key; without them (a file being
    assembled before its entries exist) the masks are left out.
    """
    from neo_vla.data.embodiment import record_digest, record_from_manifest
    records = {}
    for directory in directories:
        entry = directory.stated
        inputs = record_from_manifest(entry)
        scan = scans[directory.name]
        # Keyed by the manifest entry's name, as the loader's loss_dims, the
        # runner's coverage check and the client handshake look records up.
        record = {
            **inputs,
            "constant": scan.constant.tolist(),
            "drot": drot_of(entry),
            "history_blocked": len((entry.get("history") or {}).get("blocked", [])),
            "record_digest": record_digest(inputs),
        }
        if entries is not None:
            keys = [entries[key] for key in sorted(set(directory.keys))]
            record.update(directory_record_masks(directory, scan, keys, gated or {}))
        records[entry["name"]] = record
    return records


def assemble(args: argparse.Namespace, measured: dict[str, Any], scans: dict[str, Scan],
             directories: list[Directory], keys_table: dict[str, Any], manifest_rel: str,
             manifest_sha: str, provenance: dict[str, Any]) -> Any:
    """The embodiments.json document through the one implementation, validated on the way."""
    from neo_vla.data.embodiment import (
        INHERITED_GROUPS_ALL,
        Embodiments,
        _read_entry,
        entry_digest,
    )
    header = header_of(args.band_alpha)
    lineage = None
    if args.source_file:
        lineage = Embodiments.load(Path(args.source_file))
        if lineage.header != header:
            raise refuse(f"--from {args.source_file} states normaliser {lineage.header}",
                         f"the header this build writes ({header})",
                         "build with the lineage's transform, quantiles and alpha")
    entries: dict[str, Any] = {}
    for key, blocks in sorted(measured.items()):
        if key not in keys_table:
            raise refuse(f"key {key} is not in config/embodiments.yaml",
                         f"one of {sorted(keys_table)}", "add the key and its clause")
        check_effector(key, blocks, keys_table[key])
        entries[key] = entry_document(key, blocks, keys_table[key], manifest_rel, manifest_sha)
    inherited: dict[str, Any] = {}
    for spec in args.inherit_action:
        key, _, source = spec.partition("=")
        if key not in entries:
            raise refuse(f"--inherit-action {spec}: key {key} is not in this manifest",
                         "a key this manifest trains", "check the spelling")
        stated = lineage.document()["embodiments"].get(source)
        if stated is None:
            raise refuse(f"--inherit-action {spec}: {args.source_file} has no key {source}",
                         f"one of {sorted(lineage.entries)}", "name a key of the lineage file")
        missing = sorted(set(entries[key]["slots"]) - set(stated["slots"]))
        if missing:
            raise ValueError(
                f"{refusal_prefix(15)} --inherit-action {spec}: slots {missing} of {key} are "
                f"not in {source}'s slots {stated['slots']}\nExpected every slot the key's "
                f"directories fill.\nInherit from a key that covers them, or measure the key.")
        # The groups the key table names (action_from_groups; every group when it names
        # none, or names no action_from, which check_lineage then refuses): their columns
        # are the source's rows, value for value; the other columns stay as measured.
        groups = key_table.action_groups_of(key, keys_table[key]) or INHERITED_GROUPS_ALL
        columns = inherited_columns(groups)
        rows = [stated["slots"].index(slot) for slot in entries[key]["slots"]]
        own = entries[key]["action"]
        # A copied column the source never measured (constant or absent there) would be served
        # as that constant, or not at all, on an axis the key's own directories move: refused
        # (the gripper agreement below is the same rule).
        lost = [(entries[key]["slots"][index], column) for index, row in enumerate(rows)
                for column in columns
                if (stated["action"]["constant"][row][column]
                    or stated["action"]["absent"][row][column])
                and not (own["constant"][index][column] or own["absent"][index][column])]
        if lost:
            raise refuse(f"--inherit-action {spec}: {source} is constant or absent on (slot, "
                         f"column) {lost[:6]}, which {key}'s directories move",
                         "every copied column measured by the source wherever the key moves it",
                         f"inherit from a key that measured them; copying fewer groups for "
                         f"{key} is a separate decision")
        action = {field_name: [list(own[field_name][index]) for index in range(len(rows))]
                  for field_name in ("low", "high", "constant", "absent")}
        for index, row in enumerate(rows):
            for field_name in action:
                for column in columns:
                    action[field_name][index][column] = stated["action"][field_name][row][column]
        gripper_valid = list(own["gripper_valid"])
        if GRIPPER in columns:
            copied = [bool(stated["action"]["gripper_valid"][row]) for row in rows]
            if copied != [bool(flag) for flag in gripper_valid]:
                raise refuse(f"--inherit-action {spec}: {source} states gripper_valid {copied} "
                             f"on slots {entries[key]['slots']} and {key}'s directories give "
                             f"{gripper_valid}", "a copied gripper the key's directories "
                             "train exactly where the source does",
                             "fix the key's gripper declarations (dropping ee from "
                             f"{key}'s action_from_groups would give up the executed-scale "
                             "inheritance)")
            gripper_valid = copied
        entries[key]["action"] = {**action, "gripper_valid": gripper_valid,
                                  "mask": list(stated["action"]["mask"]), "from": source,
                                  "from_groups": list(groups)}
        inherited[key] = {"from": source, "groups": list(groups), "columns": list(columns),
                          "source_digest": stated.get("digest"),
                          "lineage_sha256": stage.sha256_of(Path(args.source_file))}
        logger.info("%s copies %s's %s columns %s from %s; the other columns are its own",
                    key, source, list(groups), list(columns), args.source_file)
    for spec in args.reuse:
        path, _, key = spec.partition("#")
        reused = Embodiments.load(Path(path))
        if reused.header != header:
            raise refuse(f"--reuse {spec} states normaliser {reused.header}", f"{header}",
                         "reuse only within one lineage")
        entries[key] = dict(reused.document()["embodiments"][key])
        entries[key].pop("digest", None)
    clauses: dict[str, str] = {}
    for key, entry in entries.items():
        if entry["clause"] in clauses:
            raise ValueError(
                f"{refusal_prefix(9)} keys {clauses[entry['clause']]} and {key} share the "
                f"clause {entry['clause']!r}\nExpected one clause per key: the clause names "
                f"the normaliser.\nGive each key its own clause in config/embodiments.yaml.")
        clauses[entry["clause"]] = key
        entry["digest"] = entry_digest(entry, header)
    check_lineage(entries, keys_table, None if lineage is None else lineage.document(),
                  dict(spec.partition("=")[::2] for spec in args.inherit_action),
                  {spec.partition("#")[2] for spec in args.reuse})
    built = {key: _read_entry(key, entry, header, str(args.out)) for key, entry in entries.items()}
    gated = {key: blocks.get("gated", {}) for key, blocks in measured.items()}
    if inherited:
        provenance = {**provenance, "inherited": inherited}
    document = {"schema": EMBODIMENTS_SCHEMA, "provenance": provenance, "normaliser": header,
                "parent": None, "embodiments": entries,
                "datasets": datasets_records(directories, scans, built, gated), "trained": None}
    return Embodiments.from_document(document, where=str(args.out))


def check_lineage(entries: dict[str, Any], keys_table: dict[str, Any],
                  lineage: dict[str, Any] | None, inherited: dict[str, str],
                  reused: set[str]) -> None:
    """V6 4.7 at build time, on the finished entries (digests included).

    Every key this build measures states a stage; a finetune key is built
    against a lineage (--from), whose header assemble() has already held this
    build to; a key whose table entry names action_from copies exactly that
    key's action block (the groups action_from_groups names, every group when it
    names none; assemble() copies them) and no other key copies one. Given a
    lineage, a key it holds keeps that entry digest for digest (4.7a; R8 at the runner), and no
    key shares a clause with another key of the lineage (R9 across files).
    """
    for key in sorted(set(entries) - reused):
        spec = keys_table[key]
        stage_name = spec.get("stage")
        if stage_name not in STAGES:
            raise refuse(f"key {key} has stage {stage_name!r} in config/embodiments.yaml",
                         f"one of {list(STAGES)}", "state the key's stage in the key table")
        if stage_name == "finetune" and lineage is None:
            raise refuse(f"key {key} is a finetune key and no --from names the lineage's "
                         f"embodiments.json", "a key new to the lineage built with the "
                         "header of the init step's file (V6 4.7b)",
                         "pass --from <init step>/embodiments.json")
        source, given = spec.get("action_from"), inherited.get(key)
        if source is not None and given != source:
            raise refuse(f"key {key} copies its action block from {source} (config/"
                         f"embodiments.yaml action_from) and this build "
                         + ("measures it" if given is None else f"copies it from {given}"),
                         "the action block of the key the table names, digest for digest "
                         "(V6 4.7, D6(i))",
                         f"pass --inherit-action {key}={source} with --from <init step>/"
                         f"embodiments.json")
        if source is None and given is not None:
            raise refuse(f"--inherit-action {key}={given}, and config/embodiments.yaml names no "
                         f"action_from for {key}", "a copied action block only where the key "
                         "table says so", f"state action_from: {given} for {key} in the key "
                         "table, or drop the flag and measure the key")
    if lineage is None:
        return
    held = lineage["embodiments"]
    for key, entry in sorted(entries.items()):
        if key in held and held[key]["digest"] != entry["digest"]:
            raise ValueError(
                f"{refusal_prefix(8)} key {key} is in the lineage file at {held[key]['digest']} "
                f"and this build writes {entry['digest']}\nExpected a key the lineage holds to "
                f"keep its entry, digest for digest (V6 4.7a).\nPass --reuse <the --from "
                f"file>#{key} instead of measuring it again.")
        for other, stated in sorted(held.items()):
            if other != key and stated["clause"] == entry["clause"]:
                raise ValueError(
                    f"{refusal_prefix(9)} key {key} shares the clause {entry['clause']!r} with "
                    f"the lineage's key {other}\nExpected one clause per key across the "
                    f"lineage: the clause names the normaliser.\nGive {key} its own clause in "
                    f"config/embodiments.yaml.")


def fit_weights(valid: np.ndarray, mass: float) -> np.ndarray:
    """[rows, columns] weights of one stratum's rows at one slot: mass over each column's rows.

    The weights measure_key gives the rows it takes q01/q99 over, so every
    stratum weighs its mixture mass in every column it trains, however many
    windows were drawn from it and however many rows each window holds.
    """
    valid = np.asarray(valid, dtype=bool)
    counts = np.maximum(valid.sum(axis=0), 1)
    return mass / counts * valid


def fit_field(values: np.ndarray, weights: np.ndarray, valid: np.ndarray, columns: list[int],
              low: np.ndarray, high: np.ndarray, *, alpha: float,
              symmetric: tuple[int, ...] = ()) -> tuple[dict[int, float | None], float, bool,
                                                         int]:
    """One field's weighted verdict: p99 |y| per column, flagged share, ok, rows counted.

    A row counts for the field when it trains one of its columns. It is
    flagged when a trained column is outside [lo - a span, hi + a span] or not
    finite; the share weighs each row by its largest column weight, as
    band_rate does. Each column's p99 |y| must lie in FIT_P99_BAND.
    """
    counted = valid[:, columns].any(axis=1)
    flagged = np.zeros(values.shape[0], dtype=bool)
    p99: dict[int, float | None] = {}
    ok = True
    for column in columns:
        lo, hi = float(low[column]), float(high[column])
        use = valid[:, column]
        span = hi - lo
        column_values = values[:, column]
        finite = np.isfinite(column_values)
        flagged |= use & (~finite | (column_values < lo - alpha * span)
                          | (column_values > hi + alpha * span))
        pick = use & finite
        if column in symmetric:
            scaled = np.abs(column_values[pick] / hi)
        else:
            scaled = np.abs(2.0 * (column_values[pick] - lo) / span - 1.0)
        p99[column] = (weighted_quantile(scaled, weights[pick, column], FIT_QUANTILE)
                       if scaled.size and weights[pick, column].sum() > 0 else None)
        if p99[column] is None or not FIT_P99_BAND[0] <= p99[column] <= FIT_P99_BAND[1]:
            ok = False
    row_weight = weights[:, columns].max(axis=1)
    total = float(row_weight[counted].sum())
    share = float(row_weight[flagged & counted].sum() / total) if total > 0 else float("nan")
    if not share <= FIT_FLAGGED_MAX:
        ok = False
    return p99, share, ok, int(counted.sum())


def fit_report(values: np.ndarray, weights: np.ndarray, valid: np.ndarray, entry: Any,
               slot: int, *, alpha: float) -> FitReport:
    """embodiment.fit_check on a weighted pool, per field: R14 of the mixture a key trains.

    values [rows, 26] are physical targets, valid [rows, 26] the columns each
    row trains, weights [rows, 26] what each row weighs in each column
    (fit_weights). Each field (joints, EE range, hand) with a decoded range
    column is judged on its own; the gripper is exempt. Refuses n = 0.
    """
    row = entry.slot_index(slot)
    low = np.asarray(entry.low[row], dtype=np.float64)
    high = np.asarray(entry.high[row], dtype=np.float64)
    decoded = np.asarray(entry.decodable(slot), dtype=bool)
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & decoded[None, :]
    weights = np.where(valid, np.asarray(weights, dtype=np.float64), 0.0)
    counted = valid[:, list(RANGE_COLUMNS)].any(axis=1)
    if not counted.any():
        raise ValueError(
            f"fit for key {entry.key!r} slot {slot} was given 0 trained arm-rows over "
            f"{values.shape[0]} drawn.\nExpected at least one row that trains a range column; "
            f"a gate over nothing passes on no evidence.\nDraw windows of this key before "
            f"checking its fit.")
    p99_all: list[float | None] = [None] * ARM_DIM
    fields = {}
    for name, columns in FIT_FIELDS.items():
        trained = [column for column in columns if decoded[column] and valid[:, column].any()]
        if not trained:
            continue
        for column in trained:
            lo, hi = float(low[column]), float(high[column])
            if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                raise degenerate(f"key {entry.key} slot {slot} dim {column} "
                                 f"({DIM_NAMES[column]})", lo, hi, None, [entry.key])
        p99, share, ok, rows = fit_field(values, weights, valid, trained, low, high, alpha=alpha)
        for column, value in p99.items():
            p99_all[column] = value
        fields[name] = {"columns": trained, "rows": rows, "flagged_share": share, "ok": ok}
    worst = max(one["flagged_share"] for one in fields.values())
    return FitReport(p99_abs_y=p99_all, flagged_share=worst,
                     ok=all(one["ok"] for one in fields.values()), rows=int(counted.sum()),
                     fields=fields)


def fit_rows(dataset: Any, windows: list[Window], key: str, built: Any,
             entry: Any) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per slot, the drawn windows' trained rows and the columns each trains.

    A row is a step the window's steps_valid keeps; its columns are what the
    loader trains (RobotDataset.loss_dims): the draw's dims_valid AND the
    directory's trained columns at that slot under the key (the record this
    build wrote), and read at that step. A window whose key is not the
    stratum's is a join defect and refuses.
    """
    found: dict[int, tuple[list[np.ndarray], list[np.ndarray]]] = defaultdict(lambda: ([], []))
    for window in windows:
        draw = window if dataset.draw_type is None else dataset.draw_type(**window.__dict__)
        targets = dataset.targets(draw)
        if str(targets.key) != key:
            raise refuse(f"{window.name} episode {window.episode} was drawn for key {key} and "
                         f"the loader says {targets.key}", "one key per kept position",
                         "rebuild the embodiment arrays (R16)")
        chunk = as_array(targets.chunk)
        steps = as_array(targets.steps_valid, bool)
        dims = as_array(targets.dims_valid, bool)
        read = as_array(targets.read, bool)
        for arm, slot in enumerate(int(slot) for slot in targets.slots):
            if slot not in entry.slots:
                entry.slot_index(slot)  # R15, stated by the one implementation
            trained = dims[arm] & np.asarray(built.columns(window.name, key, slot), dtype=bool)
            rows = chunk[arm][steps[arm]]
            found[slot][0].append(rows)
            found[slot][1].append(read[arm][steps[arm]] & trained[None, :])
    return {slot: (np.concatenate(values), np.concatenate(valid))
            for slot, (values, valid) in found.items()}


def check_fit(built: Any, root: Path, manifest: dict[str, Any], factory: Callable[[str], Any],
              args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """R14 per key, planned frame policy and field, on the mixture this manifest trains.

    Each (key, family) stratum draws windows_per_stratum windows, as
    measure_all does, and every row weighs what it weighs in the key's q01/q99
    (fit_weights), so the gate judges the mixture the key trains. Only the
    columns each row trains count (fit_rows).
    """
    from neo_vla.data.embodiment import require_fit
    directories = read_directories(root, manifest)
    strata = strata_of(directories)
    reports: dict[str, dict[str, Any]] = defaultdict(dict)
    failures = []
    for policy in policies_of(args.frame_policies):
        dataset = factory(policy)
        for key in sorted(strata):
            entry = built.entry(key)
            parts: dict[int, list[tuple[str, float, np.ndarray, np.ndarray]]] = defaultdict(list)
            for family, stratum in sorted(strata[key].items()):
                windows = draw_windows(stratum, args.windows_per_stratum,
                                       f"{args.seed}:fit:{policy}", 0)
                for slot, (values, valid) in fit_rows(dataset, windows, key, built,
                                                      entry).items():
                    parts[slot].append((family, stratum.mass, values, valid))
            for slot, held in sorted(parts.items()):
                values = np.concatenate([part[2] for part in held])
                valid = np.concatenate([part[3] for part in held])
                if not valid[:, list(RANGE_COLUMNS)].any():
                    logger.info("fit %s %s slot %d: no trained range column, not judged", key,
                                policy, slot)
                    continue
                weights = np.concatenate([fit_weights(part[3], part[1]) for part in held])
                report = fit_report(values, weights, valid, entry, slot, alpha=args.band_alpha)
                reports[key][f"{policy}:{slot}"] = {"p99_abs_y": list(report.p99_abs_y),
                                                    "flagged_share": report.flagged_share,
                                                    "ok": report.ok, "rows": report.rows,
                                                    "fields": report.fields,
                                                    "strata": len(held),
                                                    "weighting": FIT_WEIGHTING}
                logger.info("fit %s %s slot %d: %s over %d rows in %d strata (mass-weighted), %s",
                            key, policy, slot, {name: round(one["flagged_share"], 4)
                                                for name, one in report.fields.items()},
                            report.rows, len(held), "ok" if report.ok else "FAILS")
                if not report.ok:
                    failures.append((report, f"key {key} under {policy} slot {slot}"))
    for report, where in failures:
        require_fit(report, where=where)
    return reports


def lineage_paths(args: argparse.Namespace, root: Path) -> None:
    """--from and each --reuse FILE#KEY: absolute as given, a relative FILE under --root."""
    def under(path: str) -> str:
        return path if os.path.isabs(path) else os.path.normpath(root / path)

    if args.source_file:
        args.source_file = under(args.source_file)
    args.reuse = [f"{under(path)}#{key}" for path, _, key in
                  (spec.partition("#") for spec in args.reuse)]


def gate_stamps(measured: dict[str, Any]) -> dict[str, Any]:
    """Per key the wrap and own-arm gates' findings, as the provenance stamps them.

    JSON keys are strings: a slot number beside the own-arm report's "rows"
    would make the sorted dump refuse int and str keys together.
    """
    return {key: {"wrap": {str(slot): joints for slot, joints in blocks["gated"].items()},
                  "own_arm": None if blocks["own_arm"] is None else {
                      str(name): found for name, found in blocks["own_arm"].items()}}
            for key, blocks in sorted(measured.items())}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    root = stage.resolve_root(args.root)
    lineage_paths(args, root)
    manifest_path = stage.under_root(root, args.manifest)
    manifest = stage.read_json(manifest_path)
    out = stage.under_root(root, args.out)
    table = Path(args.embodiments)
    if not table.is_absolute():
        table = (REPO / table if (REPO / table).exists()
                 else stage.under_root(root, args.embodiments))
    keys_table = read_keys(table)
    skip = {spec.partition("#")[2] for spec in args.reuse}
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="build_stats_") as scratch:
        factory = real_factory(root, manifest_path, manifest, args, Path(scratch))
        measured, scans, directories = measure_all(root, manifest, factory, args, skip,
                                                   keys_table)
        inputs = {"manifest": manifest_path}
        if args.cache_index:
            inputs["cache"] = stage.under_root(root, args.cache_index)
        header = stage.header("build_stats", root=root, inputs=inputs, tree=REPO)
        provenance = {**header,
                      "argv": list(sys.argv[1:] if argv is None else argv),
                      "manifest": stage.relative(root, manifest_path),
                      "manifest_sha256": manifest_sha, "seed": args.seed,
                      "windows_per_stratum": args.windows_per_stratum,
                      "frame_policies": args.frame_policies,
                      "band_rates": {key: blocks["band"] for key, blocks in sorted(
                          measured.items())},
                      "gates": gate_stamps(measured)}
        built = assemble(args, measured, scans, directories, keys_table,
                         stage.relative(root, manifest_path), manifest_sha, provenance)
        if args.check_fit:
            fits = check_fit(built, root, manifest, factory, args)
            document = built.document()
            for key, report in fits.items():
                document["embodiments"][key]["fit"] = report
            from neo_vla.data.embodiment import Embodiments
            built = Embodiments.from_document(document, where=str(out))
    payload = built.dumps()
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(out)
    if stage.sha256_of(out) != hashlib.sha256(payload).hexdigest():
        raise OSError(
            f"{out} does not hold the {len(payload)} bytes written to it\n"
            f"expected the file to read back as what the stage produced\n"
            f"free space under the root, then run this stage again")
    from neo_vla.data.embodiment import Embodiments
    Embodiments.load(out)
    logger.info("%d keys, %d directories; wrote %s (sha256 %s)", len(built.entries),
                len(built.datasets), out, hashlib.sha256(payload).hexdigest()[:16])
    return 0


if __name__ == "__main__":
    sys.exit(main())
