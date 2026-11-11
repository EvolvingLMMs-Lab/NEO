"""The shared body block: base motion, torso and head, one 8-wide record per sample.

[base_dx base_dy base_dyaw | torso0 torso1 torso2 | head_yaw head_pitch], the
record BODY_RECORD names. The base is the motion of one
step in the base frame at the step's start, cm and rad per step: the state at
row r holds the step r-1 -> r, and the action's step j the step t+j -> t+j+1,
which is the state at row t+j+1. Torso and head are absolute joint positions.
The block is in the base frame always and never turns with frame_ref.

Validity is per dimension, NaN never 0. This module holds the writer's half
(odometry to base motion, with AgiBot's held readings resampled, and the yaw
wrap), the loader's half (targets, the per-sample masks) and the invariants
the body's masks keep with its codes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..constants import (
    BODY_BASE,
    BODY_DIM,
    BODY_HEAD,
    BODY_POSTURE,
    BODY_RECORD,
    BODY_TORSO,
    COLUMN_BODY,
    COLUMN_BODY_COMMAND,
    STATE_ABSENT,
)
from ..model.tokenizers import within_range
from .actions import sample_masks

__all__ = [
    "BODY_DIM", "BODY_RECORD", "BASE", "TORSO", "HEAD", "POSTURE", "BODY_GROUPS",
    "TORSO_JOINTS", "HEAD_JOINTS", "BASE_SOURCES", "CM_PER_M", "CADENCE_GAP_ROWS",
    "QUATERNION_MIN_NORM", "BodyLayout", "wrap_angle", "yaw_of_quaternion",
    "base_motion", "yaw_steps", "body_record", "read_body", "body_targets",
    "pack_body", "check_body_masks", "zero_step_share",
]

# Where each field sits, from the contract. POSTURE is torso and head, the
# group the band takes out together; the base is the other group.
BASE = BODY_BASE
TORSO = BODY_TORSO
HEAD = BODY_HEAD
POSTURE = BODY_POSTURE
BODY_GROUPS = {"base": BASE, "posture": POSTURE}
TORSO_JOINTS = TORSO.stop - TORSO.start
HEAD_JOINTS = HEAD.stop - HEAD.start

# Where a key's base reading comes from (config/embodiments.yaml v2 body.base).
# odometry and still are differenced poses (still is a recording of a base
# that never moved, so it reads 0); imu_yaw has the yaw only (Leju); command is
# a recorded twist (Galaxea), a target and never a state (D5); none records
# nothing. Which of B0..B2 each makes valid, in the state and in the action.
BASE_SOURCES = ("odometry", "command", "imu_yaw", "still", "none")
_BASE_STATE = {"odometry": (True, True, True), "still": (True, True, True),
               "imu_yaw": (False, False, True), "command": (False, False, False),
               "none": (False, False, False)}
_BASE_ACTION = {"odometry": (True, True, True), "still": (True, True, True),
                "imu_yaw": (False, False, True), "command": (True, True, True),
                "none": (False, False, False)}

# Odometry in metres, the base columns in centimetres per step.
CM_PER_M = 100.0

# Two pose updates at most this many rows apart belong to one stream of
# messages. AgiBot's odometry runs at about 20 Hz on 30 Hz rows: in moving
# stretches the gaps between changed rows are 1 or 2 rows (16,869 and 16,808
# of 36,047 gaps sampled, review rv_agibot_odo3.py) and a late message makes a
# 3; a longer gap is a base at rest, or a reading too small to change the
# millimetre-quantised position, and is not interpolated across.
CADENCE_GAP_ROWS = 3

# A stored quaternion this short is no reading: AgiBot writes all zeros for an
# episode without odometry (19,844 episodes).
QUATERNION_MIN_NORM = 0.5


@dataclass(frozen=True)
class BodyLayout:
    """What one key's body records, as config/embodiments.yaml v2 states it.

    base    a member of BASE_SOURCES
    torso   torso joints filled from T0 (0 to 3; AgiBot 2: pitch, lift)
    head    head joints filled from H0 (0 to 2: yaw, pitch)

    A key whose body is absent (a fixed-base robot) has no BodyLayout at all
    (None): its body token and block are masked. A layout that records nothing
    (base none, no torso, no head) is a body nobody measured: UNKNOWN.
    """

    base: str
    torso: int
    head: int

    def __post_init__(self) -> None:
        if self.base not in BASE_SOURCES:
            raise ValueError(
                f"body base is {self.base!r}, expected one of {BASE_SOURCES}.\n"
                "Name where the base reading comes from; none when nothing records it.\n"
                "Fix the key's body block in config/embodiments.yaml."
            )
        for name, count, widest in (("torso", self.torso, TORSO_JOINTS),
                                    ("head", self.head, HEAD_JOINTS)):
            if isinstance(count, bool) or not isinstance(count, int) \
                    or not 0 <= count <= widest:
                raise ValueError(
                    f"body {name} is {count!r}, expected 0 to {widest} joints.\n"
                    f"The {name} fills its columns from the first; the rest stay invalid.\n"
                    "Fix the key's body block in config/embodiments.yaml."
                )

    @property
    def target(self) -> str:
        """The base's target rule: the recorded command, or the next state's motion."""
        return "command:" + COLUMN_BODY_COMMAND if self.base == "command" else "next_state"

    def _dims(self, base: tuple[bool, bool, bool]) -> torch.Tensor:
        dims = torch.zeros(BODY_DIM, dtype=torch.bool)
        dims[BASE] = torch.tensor(base)
        dims[TORSO.start:TORSO.start + self.torso] = True
        dims[HEAD.start:HEAD.start + self.head] = True
        return dims

    def state_dims(self) -> torch.Tensor:
        """[BODY_DIM] bool: the dims the key's state can hold (a command is no state)."""
        return self._dims(_BASE_STATE[self.base])

    def action_dims(self) -> torch.Tensor:
        """[BODY_DIM] bool: the dims the key's action can train."""
        return self._dims(_BASE_ACTION[self.base])


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Angles in radians -> the same angles in (-pi, pi]."""
    values = torch.as_tensor(angle, dtype=torch.float64)
    return values - 2.0 * math.pi * torch.ceil((values - math.pi) / (2.0 * math.pi))


def yaw_of_quaternion(quaternion: torch.Tensor, order: str = "xyzw"):
    """Quaternions [N, 4] -> (yaw [N] float64 in (-pi, pi], read [N] bool).

    The yaw of the rotation's z-y-x reading, atan2(2 (w z + x y), 1 - 2 (y^2 +
    z^2)); for AgiBot's yaw-only quaternions it equals 2 atan2(z, w) wrapped. A
    quaternion shorter than QUATERNION_MIN_NORM, or not finite, is no reading.
    """
    if order not in ("xyzw", "wxyz"):
        raise ValueError(
            f"quaternion order is {order!r}, expected 'xyzw' or 'wxyz'.\n"
            "The component order is part of the reading."
        )
    values = torch.as_tensor(quaternion, dtype=torch.float64)
    if values.dim() != 2 or int(values.shape[-1]) != 4:
        raise ValueError(
            f"quaternions are {tuple(values.shape)}, expected [rows, 4].\n"
            "One quaternion per row."
        )
    if order == "wxyz":
        values = values[:, [1, 2, 3, 0]]
    norm = torch.linalg.vector_norm(values, dim=-1)
    read = torch.isfinite(values).all(dim=-1) & (norm >= QUATERNION_MIN_NORM)
    x, y, z, w = (values / norm.clamp_min(QUATERNION_MIN_NORM)[:, None]).unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.where(read, wrap_angle(yaw), torch.full_like(yaw, float("nan"))), read


def _runs(read: torch.Tensor) -> list[tuple[int, int]]:
    """[N] bool -> the (first, last) rows of each run of consecutive True."""
    edge = torch.zeros(1, dtype=torch.int8)
    change = torch.diff(torch.cat([edge, read.to(torch.int8), edge]))
    starts = torch.nonzero(change == 1).flatten().tolist()
    stops = (torch.nonzero(change == -1).flatten() - 1).tolist()
    return list(zip(starts, stops))


def _knots(times: torch.Tensor, pose: torch.Tensor, cadence: int):
    """One run's rows -> the (time, pose) knots its held readings are resampled through.

    A changed row is one whose pose differs from the row before. Changes at
    most cadence rows apart form one stretch of messages, and the stretch's
    messages are spaced evenly in time between its first and last change (a
    20 Hz stream on 30 Hz rows lands on rows 1 and 2 apart in turn, so the row
    times of the changes are not the message times). Before a stretch the
    previous pose is held until one message period before its first change.
    """
    count = int(times.shape[0])
    changed = torch.nonzero((pose[1:] != pose[:-1]).any(dim=-1)).flatten() + 1
    knot_times, knot_poses = [times[:1]], [pose[:1]]
    last = times[0]
    if changed.numel():
        breaks = torch.nonzero(torch.diff(changed) > cadence).flatten() + 1
        bounds = [0, *breaks.tolist(), int(changed.numel())]
        for first, stop in zip(bounds[:-1], bounds[1:]):
            rows = changed[first:stop]
            size = int(rows.numel())
            if size > 1:
                period = (times[rows[-1]] - times[rows[0]]) / (size - 1)
                stamps = times[rows[0]] + period * torch.arange(size, dtype=times.dtype)
            else:
                period = times[rows[0]] - times[rows[0] - 1]
                stamps = times[rows]
            held = stamps[0] - period
            if held > last:
                knot_times.append(held.reshape(1))
                knot_poses.append(pose[rows[0] - 1].reshape(1, -1))
            knot_times.append(stamps)
            knot_poses.append(pose[rows])
            last = stamps[-1]
    if times[count - 1] > last:
        knot_times.append(times[count - 1:])
        knot_poses.append(pose[count - 1:])
    return torch.cat(knot_times), torch.cat(knot_poses)


def _interpolate(at: torch.Tensor, knot_times: torch.Tensor, knot_poses: torch.Tensor):
    """Piecewise-linear values of the knots at the times at, held past either end."""
    if int(knot_times.numel()) == 1:
        return knot_poses.expand(int(at.numel()), -1).clone()
    right = torch.searchsorted(knot_times, at, right=True).clamp(1, int(knot_times.numel()) - 1)
    left = right - 1
    span = knot_times[right] - knot_times[left]
    weight = ((at - knot_times[left]) / span).clamp(0.0, 1.0)[:, None]
    return knot_poses[left] + weight * (knot_poses[right] - knot_poses[left])


def base_motion(times: torch.Tensor, x: torch.Tensor, y: torch.Tensor, yaw: torch.Tensor,
                read: torch.Tensor, *, cadence_gap_rows: int = CADENCE_GAP_ROWS):
    """Planar odometry per row -> each row's last-step motion [N, 3] and its flags [N].

    x, y in metres and yaw in radians in a fixed (world) frame, times in any
    one increasing unit, read the rows whose odometry is a reading. Row r's
    motion is the step r-1 -> r: forward and left in centimetres in the base
    frame at r-1, and the counter-clockwise yaw change in radians. Row 0 of
    every run of read rows has no step and is NaN.

    Held readings are resampled first:
    AgiBot's odometry updates on about 2 of 3 rows, so a plain difference reads
    exactly 0 on a third of moving rows and 1.5 times the motion on the rest.
    The pose is interpolated through _knots at the row times, then
    differenced; a base at rest stays exactly 0.
    """
    stamps = torch.as_tensor(times)
    stamps = (stamps - stamps.reshape(-1)[0]).to(torch.float64)
    planar = torch.stack([torch.as_tensor(x, dtype=torch.float64),
                          torch.as_tensor(y, dtype=torch.float64)], dim=-1)
    heading = torch.as_tensor(yaw, dtype=torch.float64)
    flags = torch.as_tensor(read, dtype=torch.bool)
    count = int(flags.shape[0])
    if stamps.shape != (count,) or planar.shape != (count, 2) or heading.shape != (count,):
        raise ValueError(
            f"times {tuple(stamps.shape)}, x and y {tuple(planar.shape)}, yaw "
            f"{tuple(heading.shape)} and read {tuple(flags.shape)} disagree.\n"
            "Expected one value of each per row.\nPass one episode's rows."
        )
    flags = flags & torch.isfinite(planar).all(dim=-1) & torch.isfinite(heading) \
        & torch.isfinite(stamps)
    motion = torch.full((count, 3), float("nan"), dtype=torch.float64)
    valid = torch.zeros(count, dtype=torch.bool)
    for first, last in _runs(flags):
        if last == first:
            continue
        rows = slice(first, last + 1)
        when = stamps[rows]
        if not bool((torch.diff(when) > 0).all()):
            raise ValueError(
                f"row times {first}..{last} do not increase.\n"
                "Expected strictly increasing timestamps inside an episode.\n"
                "Pass the episode's own row times."
            )
        turned = heading[rows][0] + torch.cat([
            torch.zeros(1, dtype=torch.float64), torch.cumsum(wrap_angle(torch.diff(heading[rows])), 0)
        ])
        pose = torch.cat([planar[rows], turned[:, None]], dim=-1)
        knot_times, knot_poses = _knots(when, pose, int(cadence_gap_rows))
        smooth = _interpolate(when, knot_times, knot_poses)
        step = smooth[1:] - smooth[:-1]
        cos, sin = torch.cos(smooth[:-1, 2]), torch.sin(smooth[:-1, 2])
        forward = cos * step[:, 0] + sin * step[:, 1]
        left = -sin * step[:, 0] + cos * step[:, 1]
        motion[first + 1:last + 1] = torch.stack(
            [forward * CM_PER_M, left * CM_PER_M, wrap_angle(step[:, 2])], dim=-1)
        valid[first + 1:last + 1] = True
    return motion, valid


def yaw_steps(yaw: torch.Tensor, read: torch.Tensor):
    """A yaw per row (an IMU's, Leju) -> the wrapped yaw change of each row's last step.

    No resampling: Leju's IMU yaw is held on 0 to 0.5% of steps (review
    rv_leju_oxe.py). Returns (dyaw [N] float64, NaN where row r or r-1 is not
    read, and its flags [N]).
    """
    heading = torch.as_tensor(yaw, dtype=torch.float64)
    flags = torch.as_tensor(read, dtype=torch.bool) & torch.isfinite(heading)
    steps = torch.full_like(heading, float("nan"))
    valid = torch.zeros_like(flags)
    valid[1:] = flags[1:] & flags[:-1]
    steps[1:] = torch.where(valid[1:], wrap_angle(heading[1:] - heading[:-1]), steps[1:])
    return steps, valid


def body_record(count: int, *, base: torch.Tensor | None = None,
                torso: torch.Tensor | None = None, head: torch.Tensor | None = None):
    """A writer's fields per row -> the stored record [N, BODY_DIM] float64 and its dims.

    base [N, 3] in cm and rad per step; torso [N, k <= 3] and head [N, k <= 2]
    absolute, filled from their first column. Every dimension not given, and
    every value no measurement could be, is NaN: never 0, which is a base at
    rest or a joint at its zero.
    """
    record = torch.full((int(count), BODY_DIM), float("nan"), dtype=torch.float64)
    for name, values, span in (("base", base, BASE), ("torso", torso, TORSO),
                               ("head", head, HEAD)):
        if values is None:
            continue
        given = torch.as_tensor(values, dtype=torch.float64)
        widest = span.stop - span.start
        if given.dim() != 2 or int(given.shape[0]) != int(count) \
                or not 1 <= int(given.shape[1]) <= widest \
                or (name == "base" and int(given.shape[1]) != widest):
            raise ValueError(
                f"body {name} is {tuple(given.shape)}, expected [{count}, "
                f"{widest if name == 'base' else f'1..{widest}'}].\n"
                "One row per record row; the base has all three of dx, dy, dyaw.\n"
                "Leave a field out (None) when the source does not record it."
            )
        record[:, span.start:span.start + int(given.shape[1])] = given
    dims = within_range(record)
    return torch.where(dims, record, torch.full_like(record, float("nan"))), dims


def read_body(rows: torch.Tensor | None, layout: BodyLayout | None, *, count: int):
    """Stored observation.body rows [N, 8] -> the state records [N, 8] float64 and dims.

    Only the dims the key's layout can hold in the state are read; the rest
    are NaN and False. An absent body (layout None) reads nothing; a declared
    body whose stored column is missing is refused, never read as unknown.
    """
    record = torch.full((int(count), BODY_DIM), float("nan"), dtype=torch.float64)
    dims = torch.zeros((int(count), BODY_DIM), dtype=torch.bool)
    if layout is None:
        return record, dims
    if rows is None:
        raise ValueError(
            f"the key declares a body ({layout}) and the directory gives no "
            f"{COLUMN_BODY} column.\n"
            "Expected the stored v7 column (or its sidecar) for a declared body.\n"
            "Re-emit the directory in v7; a missing column is refused, never masked."
        )
    values = torch.as_tensor(rows, dtype=torch.float64)
    if tuple(values.shape) != (int(count), BODY_DIM):
        raise ValueError(
            f"{COLUMN_BODY} rows are {tuple(values.shape)}, expected ({count}, {BODY_DIM}).\n"
            "One 8-wide body record per row."
        )
    dims = within_range(values) & layout.state_dims()[None, :]
    return torch.where(dims, values, record), dims


def body_targets(layout: BodyLayout | None, records: dict[str, torch.Tensor], *,
                 inside: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One window's body targets [K, 8] float64 and where each was read [K, 8].

    records maps the stored column names to rows t .. t+K. Step j's torso and
    head target is observation.body at row t+j+1, and so is its base target
    (the state at t+j+1 holds the step t+j -> t+j+1); a commanded base
    (Galaxea) reads action.body at row t+j instead. A step exists only when
    row t+j+1 is inside the episode. An absent body reads nothing.
    """
    if inside.dim() != 1 or inside.dtype is not torch.bool or int(inside.shape[0]) < 2:
        raise ValueError(
            f"inside is {tuple(inside.shape)} {inside.dtype}, expected [K + 1] torch.bool.\n"
            "One flag per row t .. t+K."
        )
    steps = int(inside.shape[0]) - 1
    values = torch.zeros((steps, BODY_DIM), dtype=torch.float64)
    read = torch.zeros((steps, BODY_DIM), dtype=torch.bool)
    if layout is None:
        return values, read
    wanted = layout.action_dims()
    sources = {COLUMN_BODY: (POSTURE if layout.base == "command" else slice(0, BODY_DIM), 1)}
    if layout.base == "command":
        sources[COLUMN_BODY_COMMAND] = (BASE, 0)
    for column, (span, shift) in sources.items():
        rows = records.get(column)
        if rows is None:
            raise ValueError(
                f"the body takes targets from {column!r} and the window has no such column.\n"
                f"Expected the stored v7 column; the window holds {sorted(records)}.\n"
                "Re-emit the directory in v7; a missing column is refused, never masked."
            )
        given = torch.as_tensor(rows, dtype=torch.float64)
        if tuple(given.shape) != (steps + 1, BODY_DIM):
            raise ValueError(
                f"{column} rows are {tuple(given.shape)}, expected ({steps + 1}, {BODY_DIM}).\n"
                "Rows t .. t+K, one more than the chunk."
            )
        picked = given[shift:shift + steps, span]
        ok = within_range(picked) & inside[1:, None] & wanted[span][None, :]
        values[:, span] = torch.where(ok, picked, torch.zeros_like(picked))
        read[:, span] = ok
    return values, read


def pack_body(targets: torch.Tensor, *, steps_valid: torch.Tensor, dims_valid: torch.Tensor,
              read: torch.Tensor, in_band: torch.Tensor):
    """Normalised body targets [K, 8] -> body_actions, body_action_valid, body_loss_dims, body_dim_valid.

    The arms' rule (actions.sample_masks): a step is a key when it exists and
    a trained dim was read; a dim is valid for the sample only when read on
    every kept step. An AgiBot episode without odometry so masks the base for
    the whole sample and keeps the posture; an absent or unknown body has no
    key and no valid dim. in_band is per dim, spread over BODY_GROUPS.
    """
    steps = int(targets.shape[0]) if targets.dim() == 2 else -1
    expected = {"targets": (steps, BODY_DIM), "steps_valid": (steps,),
                "dims_valid": (BODY_DIM,), "read": (steps, BODY_DIM),
                "in_band": (steps, BODY_DIM)}
    given = {"targets": targets, "steps_valid": steps_valid, "dims_valid": dims_valid,
             "read": read, "in_band": in_band}
    for name, shape in expected.items():
        value = given[name]
        if tuple(value.shape) != shape or (name != "targets" and value.dtype is not torch.bool):
            raise ValueError(
                f"body {name} is {tuple(value.shape)} {value.dtype}, expected {shape}"
                f"{'' if name == 'targets' else ' torch.bool'}.\n"
                "One body per sample: [K, 8] targets and flags, [8] trained dims."
            )
    if not bool(torch.isfinite(targets).all()):
        raise ValueError(
            "a normalised body target is not finite.\n"
            "Expected the normaliser to have set every non-finite value to 0."
        )
    kept, columns = sample_masks(steps_valid, dims_valid, read)
    live = kept[:, None] & columns[None, :]
    actions = torch.where(live, targets.to(torch.float32), torch.zeros(()))
    return actions, kept, live & in_band, columns


def check_body_masks(state_code: torch.Tensor, action_valid: torch.Tensor,
                     dim_valid: torch.Tensor) -> None:
    """Raise unless the body's codes and action masks tell one story.

    ABSENT is a property of the key, so it holds in both roles or neither,
    and an absent body has no key and no valid dim; a kept step needs a valid
    dim. An unknown body has none either: the codes, not the masks, tell it
    from an absent one.
    """
    absent = state_code == STATE_ABSENT
    if bool(absent.any()) and not bool(absent.all()):
        raise ValueError(
            f"body codes {state_code.tolist()} are absent in one role only.\n"
            "Expected absent in both roles or in neither: the key decides it.\n"
            "Build the codes with conditioning.build_conditioning."
        )
    if bool(absent.all()) and (bool(action_valid.any()) or bool(dim_valid.any())):
        raise ValueError(
            "an absent body has a valid action step or dimension.\n"
            "Expected an absent body masked in every key and column.\n"
            "Pack it with pack_body from a None layout's targets."
        )
    if bool(action_valid.any()) and not bool(dim_valid.any()):
        raise ValueError(
            "the body has a kept step and no valid dimension.\n"
            "Expected every key to carry at least one trained dimension.\n"
            "Pack it with pack_body."
        )


def zero_step_share(steps: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
    """Per dim, the share of moving rows whose step is exactly 0, [D] float64.

    The held-signal gate's statistic: a delta field whose
    source holds its readings shows a periodic exact zero inside motion. NaN
    for a dim with no moving row.
    """
    values = torch.as_tensor(steps, dtype=torch.float64)
    rows = torch.as_tensor(moving, dtype=torch.bool)
    if values.dim() == 1:
        values = values[:, None]
    picked = values[rows]
    if int(picked.shape[0]) == 0:
        return torch.full((int(values.shape[1]),), float("nan"), dtype=torch.float64)
    return (picked == 0.0).double().mean(dim=0)
