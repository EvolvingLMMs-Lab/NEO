"""One arm's proprioception as the 29-number v7 record the state tokens read.

The record is [j0..j6 | px py pz | r00 r10 r20 r01 r11 r21 | a | h0..h11]:
joints in rad, position in metres in the robot base frame, columns 0 and 1 of
R_base<-ee (Zhou's 6D), the canonical aperture a in [0, 1] with 0 open, and the
hand in its key's declared order. It is physical:
embodiment.normalise_state maps it per (key, slot, dim).

Validity is per dimension (section 3.2): the pose (position and 6D) is one
atomic group, the aperture needs an established GripperSpec, joints and hand
are per dimension, and an invalid dimension holds NaN, never 0. Pose and
aperture come from the directory's state row; joints and hand from their own
stored v7 columns (observation.joints A x 7, observation.hand A x 12, arm k at
block k), where NaN means not recorded.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch

from ..constants import (
    ARM_JOINTS,
    COLUMN_HAND,
    COLUMN_JOINTS,
    COLUMN_JOINTS_COMMAND,
    HAND_JOINTS,
    STATE_APERTURE,
    STATE_DIM,
    STATE_HAND,
    STATE_JOINTS,
    STATE_POSITION,
    STATE_ROTATION,
)
from ..model.tokenizers import within_range
from .conditioning import is_rotation, mark_unknown_dims

# Where each field sits in the record, from the contract. POSE is the atomic
# group: position and 6D are read together or not at all.
JOINTS = STATE_JOINTS
POSITION = STATE_POSITION
ROTATION = STATE_ROTATION
APERTURE = slice(STATE_APERTURE, STATE_APERTURE + 1)
HAND = STATE_HAND
POSE = slice(STATE_POSITION.start, STATE_ROTATION.stop)

# A stored position in centimetres is divided by this to reach metres, the
# unit the record carries. Every source reaches metres through it; the per-key
# map to [-1, 1] happens later, in the normaliser, and nowhere here.
POSITION_SCALE_CM = 100.0

# The two per-dimension fields read from their own stored v7 columns, how wide
# each arm's block of that column is, which stored column holds the record,
# and which stored command columns a target may name instead of the next state.
PART_FIELDS = ("joints", "hand")
PART_WIDTHS = {"joints": ARM_JOINTS, "hand": HAND_JOINTS}
PART_SLICES = {"joints": JOINTS, "hand": HAND}
PART_RECORDS = {"joints": COLUMN_JOINTS, "hand": COLUMN_HAND}
PART_COMMANDS = {"joints": (COLUMN_JOINTS_COMMAND,), "hand": ()}

# A part's target rule: the measured state one row on
# (q[t+j+1]), or a command the source recorded at row t+j, named by its stored
# column. A joint target and the EE target of one sample describe one motion:
# a constructed EE delta goes with next_state, a recorded EE command with the
# joint command.
TARGET_NEXT_STATE = "next_state"
TARGET_COMMAND_PREFIX = "command:"

# Rotation encodings the decoder knows. The component order is part of the
# name: a bare "quat" is the defect, because the two orders differ by a
# rotation of tens of degrees and neither one errors.
ROTATION_KINDS = ("quat_xyzw", "quat_wxyz", "rotmat9", "rotvec3", "euler_xyz")

# Multiply a stored position by this to reach centimetres.
POSITION_UNITS = {"cm": 1.0, "m": 100.0, "mm": 0.1}

# Aperture is 0 fully open and 1 fully closed, in every source. The direction
# is a fact about the recording, so it is measured once per dataset and stored
# in GripperSpec rather than inferred from the numbers at run time.
APERTURE_OPEN = 0.0
APERTURE_CLOSED = 1.0

# Raw columns each encoding occupies, so a layout knows how much it reads.
ROTATION_WIDTHS = {"quat_xyzw": 4, "quat_wxyz": 4, "rotmat9": 9, "rotvec3": 3, "euler_xyz": 3}

# Below this a quaternion has no direction, so the record says unknown; a
# rotation vector this short is the identity.
DEGENERATE_NORM = 1e-8

# Below this angle the Rodrigues and log-map coefficients use their series,
# which are exact at zero where the closed forms divide zero by zero.
SMALL_ANGLE = 1e-4

# The two Euler readings a manifest's rotation_source may name: extrinsic x
# then y then z (the matrix is Rz Ry Rx, the reading euler_xyz takes), and
# intrinsic X then Y then Z (the matrix is Rx Ry Rz).
EULER_CONVENTIONS = ("xyz_extrinsic", "XYZ_intrinsic")


@dataclass(frozen=True)
class GripperSpec:
    """Which column holds aperture and which end of it means closed.

    raw_open and raw_closed are the raw values the recording uses for the two
    extremes, and they may be in either order: their order is what encodes the
    polarity. A dataset whose polarity was not established has no GripperSpec
    at all, which is not the same as one with a symmetric range.
    """

    column: int
    raw_open: float
    raw_closed: float


@dataclass(frozen=True)
class PartLayout:
    """One arm's joints or hand inside their stored v7 column, and its target rule.

    count        dims this arm fills from the field's first: joints 1 to 7 (a
                 6-DoF arm fills j0..j5 and j6 stays invalid, never 0), hand 1
                 to 12 in the key's declared order
    unit_factor  model units per stored unit: 1.0 for radians and for a hand
                 the source normalises (Inspire), pi / 180 for degrees;
                 measured per directory, never assumed
    target       TARGET_NEXT_STATE, or TARGET_COMMAND_PREFIX + a stored command
                 column of the field (DROID's joints: "command:action.joints")

    The state reads count and unit_factor; the action target also reads
    target. The manifest states one block per arm and field, and the loader
    hands the same object to the arm's state layout and to its action layout,
    so the two cannot read one column two ways.
    """

    count: int
    unit_factor: float
    target: str


def parse_target(target: str, field: str) -> str | None:
    """A part's target rule -> None for the next state, or the command column it names.

    Raises for any other spelling, for a command column the field has none of
    (a hand never has one), and for a field outside PART_FIELDS.
    """
    if field not in PART_FIELDS:
        raise ValueError(
            f"field is {field!r}, expected one of {PART_FIELDS}.\n"
            "Only joints and hand are read from their own stored columns.\n"
            "The EE fields come from the flat action column."
        )
    text = str(target)
    if text == TARGET_NEXT_STATE:
        return None
    if text.startswith(TARGET_COMMAND_PREFIX):
        column = text[len(TARGET_COMMAND_PREFIX):]
        if column in PART_COMMANDS[field]:
            return column
        raise ValueError(
            f"{field} target is {text!r}, a command column the field has none of.\n"
            f"Expected {TARGET_NEXT_STATE!r} or one of "
            f"{[TARGET_COMMAND_PREFIX + name for name in PART_COMMANDS[field]]}.\n"
            "A command target names the stored v7 command column it is read from."
        )
    raise ValueError(
        f"{field} target is {text!r}.\n"
        f"Expected {TARGET_NEXT_STATE!r} or {TARGET_COMMAND_PREFIX!r} + a stored command column.\n"
        "State the rule the source's EE action follows."
    )


def check_part(part: PartLayout, field: str, owner: str) -> None:
    """Raise unless a part's count, factor and target are well formed for its field."""
    parse_target(part.target, field)
    widest = PART_WIDTHS[field]
    if isinstance(part.count, bool) or not isinstance(part.count, int) \
            or not 1 <= part.count <= widest:
        raise ValueError(
            f"{owner} {field} count is {part.count!r}, expected 1 to {widest}.\n"
            f"An arm fills its {field} from the first column; the rest stay invalid.\n"
            f"Leave {field} out (None) for an arm that records none."
        )
    factor = part.unit_factor
    if isinstance(factor, bool) or not isinstance(factor, (int, float)) \
            or not math.isfinite(float(factor)) or float(factor) <= 0.0:
        raise ValueError(
            f"{owner} {field} unit_factor is {factor!r}, expected a finite positive factor.\n"
            "Measure what one stored unit is worth: 1.0 for rad, pi / 180 for degrees.\n"
            "A zero factor turns every reading into a pose at the origin."
        )


@dataclass(frozen=True)
class ArmStateLayout:
    """Where one arm's state lives: its pose and aperture in the directory's raw
    state vector, its joints and hand in their own stored v7 columns.

    source, arm     the manifest's names for the directory's source and the arm
    position_at     first column of the position triple, or None
    rotation_at     first column of the rotation block, or None
    rotation_kind   a member of ROTATION_KINDS, or None
    position_unit   a key of POSITION_UNITS, or None
    gripper         the aperture column, or None when it was never established
    provenance      what established this layout, carried into the manifest
    joints, hand    the arm's block in observation.joints / observation.hand,
                    or None for an arm that records none

    The four pose fields are all stated or all None: a joints-only arm
    (BimanualYAM, Tien Kung) has no pose and keeps its joints and gripper. An
    arm without a gripper spec keeps its pose: the aperture alone is invalid.
    """

    source: str
    arm: str
    position_at: int | None
    rotation_at: int | None
    rotation_kind: str | None
    position_unit: str | None
    gripper: GripperSpec | None
    provenance: str
    joints: PartLayout | None = None
    hand: PartLayout | None = None

    @property
    def has_pose(self) -> bool:
        """Whether the layout reads a position and a rotation from the state row."""
        return self.position_at is not None

    def width(self) -> int:
        """The raw state columns this layout reads, so a short vector raises early."""
        reads = [0]
        if self.has_pose:
            reads += [
                int(self.position_at) + 3,
                int(self.rotation_at) + ROTATION_WIDTHS[self.rotation_kind],
            ]
        if self.gripper is not None:
            reads.append(self.gripper.column + 1)
        return max(reads)

    def columns(self) -> frozenset[int]:
        """Every raw state column this layout reads, for the overlap check."""
        read: set[int] = set()
        if self.has_pose:
            read |= set(range(int(self.position_at), int(self.position_at) + 3))
            start = int(self.rotation_at)
            read |= set(range(start, start + ROTATION_WIDTHS[self.rotation_kind]))
        if self.gripper is not None:
            read.add(self.gripper.column)
        return frozenset(read)

    def part(self, field: str) -> PartLayout | None:
        """The arm's joints or hand block, by field name."""
        if field not in PART_FIELDS:
            raise ValueError(
                f"field is {field!r}, expected one of {PART_FIELDS}.\n"
                "Only joints and hand are parts of a state layout."
            )
        return self.joints if field == "joints" else self.hand


def check_state_width(**claims: int) -> None:
    """Raise unless every claimed per-arm state width is STATE_DIM.

    The width has one authority, STATE_DIM in neo_vla/constants.py, and this is
    the only place a second claim is refused. Pass one keyword per claimant, so
    a run whose config, dataset and runner disagree stops with all three
    numbers in the message instead of building a projection at one of them.
    """
    fields = (
        (JOINTS.stop - JOINTS.start) + (POSITION.stop - POSITION.start)
        + (ROTATION.stop - ROTATION.start) + 1 + (HAND.stop - HAND.start)
    )
    stated = {"fields": fields, **{name: int(width) for name, width in claims.items()}}
    wrong = {name: width for name, width in stated.items() if width != STATE_DIM}
    if wrong:
        seen = ", ".join(f"{name} says {width}" for name, width in stated.items())
        raise ValueError(
            f"per-arm state width: {seen}, expected STATE_DIM {STATE_DIM}.\n"
            "STATE_DIM in neo_vla/constants.py is the only authority for it.\n"
            "Fix whichever producer disagrees; nothing here rewrites the width."
        )


def check_layout(layout: ArmStateLayout, raw_width: int) -> None:
    """Raise unless the layout is well formed and the raw vector is wide enough."""
    check_state_width()
    named = f"{layout.source}/{layout.arm}"
    pose = (layout.position_at, layout.rotation_at, layout.rotation_kind, layout.position_unit)
    if any(value is None for value in pose) and not all(value is None for value in pose):
        raise ValueError(
            f"{named} states position_at {layout.position_at}, rotation_at "
            f"{layout.rotation_at}, rotation_kind {layout.rotation_kind!r} and "
            f"position_unit {layout.position_unit!r}.\n"
            "Expected all four (an arm with a pose) or none (a joints-only arm).\n"
            "A pose with a missing part is not a pose."
        )
    if not layout.provenance:
        raise ValueError(
            f"{named} has no provenance, expected the file that established it.\n"
            "A layout without one is a guess; leave the arm out and it says unknown."
        )
    if not layout.has_pose and layout.gripper is None and layout.joints is None \
            and layout.hand is None:
        raise ValueError(
            f"{named} reads no pose, no gripper, no joints and no hand.\n"
            "Expected at least one field.\n"
            "Give the arm no layout (None) instead; it then says unknown."
        )
    if layout.has_pose:
        if layout.rotation_kind not in ROTATION_KINDS:
            raise ValueError(
                f"{named} rotation_kind is {layout.rotation_kind!r}, expected one of "
                f"{ROTATION_KINDS}.\nThe component order is part of the name."
            )
        if layout.position_unit not in POSITION_UNITS:
            raise ValueError(
                f"{named} position_unit is {layout.position_unit!r}, expected one of "
                f"{tuple(POSITION_UNITS)}.\nMeasure the unit and record it."
            )
        if layout.position_at < 0 or layout.rotation_at < 0:
            raise ValueError(
                f"{named} reads column {min(layout.position_at, layout.rotation_at)}, "
                "expected zero or more.\nColumns are indexes into the stored row."
            )
    if layout.gripper is not None:
        gripper = layout.gripper
        if gripper.column < 0:
            raise ValueError(
                f"{named} gripper column is {gripper.column}, expected zero or more.\n"
                "Read it from the layout; the last column is a pad in some sources."
            )
        if gripper.raw_open == gripper.raw_closed:
            raise ValueError(
                f"{named} gripper open and closed are both {gripper.raw_open}, "
                "expected two values.\nTheir order is what encodes the polarity."
            )
    for field in PART_FIELDS:
        part = layout.part(field)
        if part is not None:
            check_part(part, field, named)
    if raw_width < layout.width():
        raise ValueError(
            f"{named} reads {layout.width()} columns, got a state of {raw_width}.\n"
            "Either the layout names the wrong source or the cache is truncated."
        )


def ee_rotation_6d(rotation: torch.Tensor) -> torch.Tensor:
    """R [..., 3, 3] -> [..., 6]: column 0 then column 1, r00 r10 r20 r01 r11 r21.

    Rz(90 degrees) gives [0, 1, 0, -1, 0, 0]. Columns, not rows: a rows-first
    6D (pytorch3d's) is a different record under the same width.
    """
    return torch.cat([rotation[..., :, 0], rotation[..., :, 1]], dim=-1)


def rotation_from_6d(six: torch.Tensor) -> torch.Tensor:
    """[..., 6] -> R [..., 3, 3] by Gram-Schmidt, exact for a clean 6D record."""
    first, second = six[..., 0:3], six[..., 3:6]
    b1 = first / torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    along = (b1 * second).sum(dim=-1, keepdim=True)
    b2 = second - along * b1
    b2 = b2 / torch.linalg.vector_norm(b2, dim=-1, keepdim=True)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def _hat(vector: torch.Tensor) -> torch.Tensor:
    """[..., 3] -> the skew matrices [..., 3, 3] with hat(w) v = w x v."""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack([
        torch.stack([zero, -z, y], dim=-1),
        torch.stack([z, zero, -x], dim=-1),
        torch.stack([-y, x, zero], dim=-1),
    ], dim=-2)


def so3_exp(vector: torch.Tensor) -> torch.Tensor:
    """A rotation vector [..., 3] -> R [..., 3, 3] by Rodrigues, exactly I at zero."""
    theta = torch.linalg.vector_norm(vector, dim=-1)[..., None, None]
    small = theta < SMALL_ANGLE
    safe = torch.where(small, torch.ones_like(theta), theta)
    first = torch.where(small, 1.0 - theta ** 2 / 6.0, torch.sin(safe) / safe)
    second = torch.where(small, 0.5 - theta ** 2 / 24.0, (1.0 - torch.cos(safe)) / safe ** 2)
    hat = _hat(vector)
    eye = torch.eye(3, dtype=vector.dtype, device=vector.device).expand(hat.shape)
    return eye + first * hat + second * (hat @ hat)


def so3_log(rotation: torch.Tensor) -> torch.Tensor:
    """R [..., 3, 3] -> its rotation vector [..., 3], the inverse of so3_exp.

    Near a half turn the antisymmetric part vanishes, so the axis is read from
    the symmetric part there instead; the per-step deltas this serves are far
    from it, and the branch keeps the map total.
    """
    trace = rotation[..., 0, 0] + rotation[..., 1, 1] + rotation[..., 2, 2]
    theta = torch.arccos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0))
    vee = torch.stack([
        rotation[..., 2, 1] - rotation[..., 1, 2],
        rotation[..., 0, 2] - rotation[..., 2, 0],
        rotation[..., 1, 0] - rotation[..., 0, 1],
    ], dim=-1)
    small = theta < SMALL_ANGLE
    safe = torch.where(small, torch.ones_like(theta), theta)
    factor = torch.where(small, 0.5 + theta ** 2 / 12.0, safe / (2.0 * torch.sin(safe)))
    general = factor[..., None] * vee
    symmetric = (rotation + rotation.transpose(-1, -2)) / 2.0
    eye = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    outer = (symmetric - torch.cos(theta)[..., None, None] * eye)
    column = outer.diagonal(dim1=-2, dim2=-1).argmax(dim=-1)
    axis = torch.take_along_dim(outer, column[..., None, None].expand(*column.shape, 3, 1),
                                dim=-1)[..., 0]
    axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1e-12)
    sign = torch.where((axis * vee).sum(dim=-1, keepdim=True) < 0.0, -1.0, 1.0)
    near_pi = (math.pi - theta) < 1e-3
    return torch.where(near_pi[..., None], sign * axis * theta[..., None], general)


def _about(angle: torch.Tensor, axis: int) -> torch.Tensor:
    """Rotations of angle [...] radians about one base axis -> [..., 3, 3]."""
    cos, sin = torch.cos(angle), torch.sin(angle)
    one, zero = torch.ones_like(angle), torch.zeros_like(angle)
    if axis == 0:
        rows = [[one, zero, zero], [zero, cos, -sin], [zero, sin, cos]]
    elif axis == 1:
        rows = [[cos, zero, sin], [zero, one, zero], [-sin, zero, cos]]
    else:
        rows = [[cos, -sin, zero], [sin, cos, zero], [zero, zero, one]]
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)


def euler_matrix(angles: torch.Tensor, convention: str) -> torch.Tensor:
    """Euler angles [..., 3] in radians -> R [..., 3, 3] under one named reading."""
    if convention not in EULER_CONVENTIONS:
        raise ValueError(
            f"euler convention is {convention!r}, expected one of {EULER_CONVENTIONS}.\n"
            "The order and the frame are part of the name.\n"
            "State the reading the source's measurement established."
        )
    rx, ry, rz = (_about(angles[..., axis], axis) for axis in range(3))
    if convention == "xyz_extrinsic":
        return rz @ ry @ rx
    return rx @ ry @ rz


def rotation_matrices(block: torch.Tensor, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw rotation columns [N, width] -> (R [N, 3, 3] float64, decoded [N] bool).

    A quaternion too short to have a direction and a stored matrix that is not
    a rotation are not decoded, and their rows hold the identity. No sign is
    fixed anywhere: the 6D the record carries does not depend on it.
    """
    values = block.to(torch.float64)
    rows = int(values.shape[0])
    eye = torch.eye(3, dtype=torch.float64).expand(rows, 3, 3).clone()
    if kind in ("quat_xyzw", "quat_wxyz"):
        order = [0, 1, 2, 3] if kind == "quat_xyzw" else [1, 2, 3, 0]
        quaternion = values[:, order]
        length = torch.linalg.vector_norm(quaternion, dim=-1)
        decoded = length >= DEGENERATE_NORM
        x, y, z, w = (quaternion / length.clamp_min(DEGENERATE_NORM)[:, None]).unbind(-1)
        matrix = torch.stack([
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ], dim=-2)
    elif kind == "rotvec3":
        decoded = torch.ones(rows, dtype=torch.bool)
        matrix = so3_exp(values)
    elif kind == "euler_xyz":
        decoded = torch.ones(rows, dtype=torch.bool)
        matrix = euler_matrix(values, "xyz_extrinsic")
    elif kind == "rotmat9":
        matrix = values.reshape(rows, 3, 3)
        decoded = torch.tensor([is_rotation(one) for one in matrix], dtype=torch.bool)
    else:
        raise ValueError(
            f"rotation kind is {kind!r}, expected one of {ROTATION_KINDS}.\n"
            "The component order is part of the name.\n"
            "Fix the layout's rotation_kind."
        )
    return torch.where(decoded[:, None, None], matrix, eye), decoded


class ArmPoses(NamedTuple):
    """An arm's pose and aperture at each row, with one flag per field.

    position  [N, 3] float64 metres, NaN where the pose was not read
    rotation  [N, 3, 3] float64 R_base<-ee, the identity where not read
    aperture  [N] float64 canonical a, NaN where not read
    pose      [N] bool: position and rotation both read (one atomic group)
    gripper   [N] bool: a GripperSpec exists and its column read a number
    """

    position: torch.Tensor
    rotation: torch.Tensor
    aperture: torch.Tensor
    pose: torch.Tensor
    gripper: torch.Tensor


def arm_poses(rows: torch.Tensor, layout: ArmStateLayout, *,
              base_offset_m: torch.Tensor | None = None) -> ArmPoses:
    """Raw state rows [N, W] -> the arm's pose and aperture with per-field flags.

    The pose is read where every position and rotation column is a number a
    measurement could be (RECORD_LIMIT) and the rotation decodes; the aperture
    where the layout has a gripper spec and its column is such a number. The
    two are independent: a source with no established gripper keeps its pose.
    A row whose pose is not read holds NaN position and the identity rotation,
    which no consumer reads without its flag.
    """
    if rows.dim() != 2:
        raise ValueError(
            f"state rows are {tuple(rows.shape)}, expected [rows, width].\n"
            "Pass the window's state column, or one row as [1, width].\n"
            "arm_state takes a single row."
        )
    check_layout(layout, int(rows.shape[1]))
    values = rows.to(torch.float64)
    count = int(values.shape[0])
    eye = torch.eye(3, dtype=torch.float64).expand(count, 3, 3)
    nan3 = torch.full((count, 3), float("nan"), dtype=torch.float64)
    if layout.has_pose:
        end = int(layout.rotation_at) + ROTATION_WIDTHS[layout.rotation_kind]
        rotation, decoded = rotation_matrices(values[:, int(layout.rotation_at):end],
                                              layout.rotation_kind)
        # within_range rather than isfinite: a source can ship a garbled telemetry
        # sample as a finite number of impossible size, and RH20T does.
        pose_columns = sorted(set(range(int(layout.position_at), int(layout.position_at) + 3))
                              | set(range(int(layout.rotation_at), end)))
        pose = within_range(values[:, pose_columns]).all(dim=-1) & decoded
        scale = POSITION_UNITS[layout.position_unit] / POSITION_SCALE_CM
        position = values[:, int(layout.position_at):int(layout.position_at) + 3] * scale
        if base_offset_m is not None:
            position = position - torch.as_tensor(base_offset_m, dtype=torch.float64)
    else:
        rotation, pose, position = eye.clone(), torch.zeros(count, dtype=torch.bool), nan3
    aperture = torch.full((count,), float("nan"), dtype=torch.float64)
    gripper_read = torch.zeros(count, dtype=torch.bool)
    if layout.gripper is not None:
        gripper = layout.gripper
        raw = values[:, gripper.column]
        travelled = (raw - gripper.raw_open) / (gripper.raw_closed - gripper.raw_open)
        # A recording runs a little outside its own measured extremes, and
        # aperture is a fraction of a known travel rather than an open range.
        aperture = (APERTURE_OPEN + travelled * (APERTURE_CLOSED - APERTURE_OPEN)).clamp(
            APERTURE_OPEN, APERTURE_CLOSED
        )
        gripper_read = within_range(raw)
    return ArmPoses(
        position=torch.where(pose[:, None], position, nan3),
        rotation=torch.where(pose[:, None, None], rotation, eye),
        aperture=torch.where(gripper_read, aperture, torch.full_like(aperture, float("nan"))),
        pose=pose,
        gripper=gripper_read,
    )


def arm_pose(raw: torch.Tensor, layout: ArmStateLayout):
    """One raw state row -> (position m [3], R [3, 3], aperture, pose read, aperture read)."""
    if raw.dim() != 1:
        raise ValueError(
            f"state row is {tuple(raw.shape)}, expected one row.\n"
            "Index the window first; arm_poses takes a stack of rows.\n"
            "The pose of one step is one row."
        )
    poses = arm_poses(raw[None], layout)
    return (poses.position[0], poses.rotation[0], float(poses.aperture[0]),
            bool(poses.pose[0]), bool(poses.gripper[0]))


def read_part(rows: torch.Tensor | None, part: PartLayout | None, *, field: str,
              block: int, count_rows: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """A stored joints or hand column [N, A x width] -> one arm's [N, width] values and flags.

    block is the arm's index in the directory (arm k reads block k). Columns
    past part.count stay NaN and invalid whatever the store holds there: a
    6-DoF arm's j6 is invalid, never the 0 a writer may have put in it. A
    stored NaN is a dimension not recorded; a value no measurement could be
    (RECORD_LIMIT) is not read either. Values are multiplied by unit_factor.
    rows None is allowed only when part is None (count_rows then gives N): a
    declared part with no stored column is refused, never read as unknown.
    """
    if field not in PART_FIELDS:
        raise ValueError(
            f"field is {field!r}, expected one of {PART_FIELDS}.\n"
            "Only joints and hand are read from their own stored columns."
        )
    width = PART_WIDTHS[field]
    if rows is None:
        if part is not None:
            raise ValueError(
                f"the arm declares {field} ({part.count}) and the directory gives no "
                f"{PART_RECORDS[field]} column.\n"
                "Expected the stored v7 column (or its sidecar) for every declared part.\n"
                "Re-emit the directory in v7; a missing column is refused, never masked."
            )
        if count_rows is None:
            raise ValueError(
                f"no {PART_RECORDS[field]} rows and no row count.\n"
                "Pass count_rows for an arm without this part."
            )
        count = int(count_rows)
        return (torch.full((count, width), float("nan"), dtype=torch.float64),
                torch.zeros((count, width), dtype=torch.bool))
    if rows.dim() != 2:
        raise ValueError(
            f"{PART_RECORDS[field]} rows are {tuple(rows.shape)}, expected [rows, arms x {width}].\n"
            "Pass the window's stored column, one row per step."
        )
    count = int(rows.shape[0])
    values = torch.full((count, width), float("nan"), dtype=torch.float64)
    read = torch.zeros((count, width), dtype=torch.bool)
    if part is None:
        return values, read
    stored = int(rows.shape[1])
    if stored % width or not 0 <= int(block) < stored // width:
        raise ValueError(
            f"{PART_RECORDS[field]} is {stored} wide and the arm reads block {block}.\n"
            f"Expected arms x {width} columns with block {block} inside them.\n"
            "The stored column holds one block per arm of the directory, in arm order."
        )
    check_part(part, field, f"block {block}")
    start = int(block) * width
    raw = rows[:, start:start + part.count].to(torch.float64)
    ok = within_range(raw)
    values[:, :part.count] = torch.where(ok, raw * float(part.unit_factor),
                                         torch.full_like(raw, float("nan")))
    read[:, :part.count] = ok
    return values, read


def records_of(poses: ArmPoses, joints: tuple[torch.Tensor, torch.Tensor],
               hand: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-field readings [N] -> the records [N, STATE_DIM] float32 and their dims [N, STATE_DIM].

    NaN exactly where a dimension is invalid: the pose's nine as one group,
    the aperture with its spec, joints and hand per dimension.
    """
    count = int(poses.position.shape[0])
    record = torch.full((count, STATE_DIM), float("nan"), dtype=torch.float64)
    dims = torch.zeros((count, STATE_DIM), dtype=torch.bool)
    record[:, JOINTS], dims[:, JOINTS] = joints
    record[:, POSITION] = poses.position
    record[:, ROTATION] = ee_rotation_6d(poses.rotation)
    dims[:, POSE] = poses.pose[:, None]
    record[:, APERTURE] = poses.aperture[:, None]
    dims[:, APERTURE] = poses.gripper[:, None]
    record[:, HAND], dims[:, HAND] = hand
    return mark_unknown_dims(record.to(torch.float32), dims), dims


def arm_records(rows: torch.Tensor, layout: ArmStateLayout, *, block: int,
                joints: torch.Tensor | None = None, hand: torch.Tensor | None = None,
                base_offset_m: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw state rows [N, W] and the stored part rows -> records [N, 29] and dims [N, 29].

    joints and hand are the directory's observation.joints and
    observation.hand rows aligned with rows (None where the directory has no
    such column, which is refused for an arm that declares the part).
    """
    poses = arm_poses(rows, layout, base_offset_m=base_offset_m)
    count = int(rows.shape[0])
    for name, stored in (("joints", joints), ("hand", hand)):
        if stored is not None and int(stored.shape[0]) != count:
            raise ValueError(
                f"{PART_RECORDS[name]} has {int(stored.shape[0])} rows and the state {count}.\n"
                "Expected the stored columns row-aligned with the state.\n"
                "Read every column of the window over the same rows."
            )
    return records_of(
        poses,
        read_part(joints, layout.joints, field="joints", block=block, count_rows=count),
        read_part(hand, layout.hand, field="hand", block=block, count_rows=count),
    )


def arm_state(raw: torch.Tensor, layout: ArmStateLayout, *, block: int = 0,
              joints: torch.Tensor | None = None, hand: torch.Tensor | None = None,
              base_offset_m: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """One raw state row -> ([STATE_DIM] float32 record, [STATE_DIM] bool dims).

    Position is converted to metres, minus base_offset_m where the source's
    stored position is not in the robot base frame (LIBERO's scene offset);
    the rotation is decoded to R_base<-ee and written as its 6D; the aperture
    is the canonical a; joints and hand come from their stored rows (one row
    each, or None). The record is NaN exactly where its dims are False.
    """
    if raw.dim() != 1:
        raise ValueError(
            f"state row is {tuple(raw.shape)}, expected one row.\n"
            "split_arms passes the current step; index the window first.\n"
            "The record of one step is one row."
        )
    record, dims = arm_records(
        raw[None], layout, block=block,
        joints=None if joints is None else joints.reshape(1, -1),
        hand=None if hand is None else hand.reshape(1, -1),
        base_offset_m=base_offset_m,
    )
    return record[0], dims[0]


def split_arms(raw: torch.Tensor, arms: Sequence, *, n_slots: int,
               joints: torch.Tensor | None = None, hand: torch.Tensor | None = None,
               base_offsets: dict[int, torch.Tensor] | None = None):
    """A directory's state row -> state [n_slots, STATE_DIM] and dims [n_slots, STATE_DIM].

    Each arm names its own slot (arms[k].slot) and its layout (arms[k].state,
    None for a source whose state was never established); arm k reads block k
    of the stored joints and hand rows. Slots no arm fills, and arms with no
    layout, are all NaN with no valid dimension rather than zero. Raises when
    two layouts of one directory claim the same state column, which is how a
    single-arm layout applied twice would present itself.
    """
    taken: dict[int, str] = {}
    for arm in arms:
        if not 0 <= int(arm.slot) < n_slots:
            raise ValueError(
                f"arm {arm.slot} is outside the {n_slots} action slots.\n"
                "Expected every arm's slot below the model's n_slots.\n"
                "Raise model.action.n_slots, or fix the arm's slot in the manifest."
            )
        if arm.state is None:
            continue
        owner = f"{arm.state.source}/{arm.state.arm}"
        for column in sorted(arm.state.columns()):
            if column in taken:
                raise ValueError(
                    f"{owner} and {taken[column]} both read column {column}.\n"
                    "Two arms of one directory cannot share a pose; fix the offsets."
                )
            taken[column] = owner
    state = torch.full((n_slots, STATE_DIM), float("nan"), dtype=torch.float32)
    dims = torch.zeros((n_slots, STATE_DIM), dtype=torch.bool)
    for block, arm in enumerate(arms):
        if arm.state is None:
            continue
        offset = None if base_offsets is None else base_offsets.get(int(arm.slot))
        state[int(arm.slot)], dims[int(arm.slot)] = arm_state(
            raw, arm.state, block=block, joints=joints, hand=hand, base_offset_m=offset,
        )
    return mark_unknown_dims(state, dims), dims
