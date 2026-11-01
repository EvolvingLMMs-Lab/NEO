"""A source's action columns to the physical targets and the slot tensor.

The steps run in one order: unify each source into centimetres and radians per
step, canonicalise the gripper to 0 open and 1 closed, bring every rotation
delta to the spatial (base-frame) rotation vector, rotate into the frame the
sample's anchor names, then normalise per embodiment (neo_vla.data.embodiment)
and pack the arms into their slots.

One arm's row is the v7 record j7_dpos3_drot3_g1_h12:
joints 0-6 and hand 14-25 are absolute joint-space targets the camera frame
never turns; dpos 7-9 (cm per step), drot 10-12 (spatial rotation vector) and
the gripper 13 are v6's EE columns. Every site addresses a column through the
named slices below, never a number, and a column a source does not record is
invalid, which is not zero: a zero target is a motion the model reproduces.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..constants import (
    ACTION_DIM,
    ACTION_DPOS,
    ACTION_DROT,
    ACTION_EE,
    ACTION_GRIPPER,
    ACTION_HAND,
    ACTION_JOINTS,
    GRIPPER_COLUMN,
)
from .state import (
    PART_FIELDS,
    PART_RECORDS,
    TARGET_COMMAND_PREFIX,
    TARGET_NEXT_STATE,
    PartLayout,
    check_part,
    parse_target,
    read_part,
    so3_log,
)

# Where each field sits in a slot's row, from the contract. The widths are
# fixed by the model's action_dim; a config that states a different action_dim
# raises rather than shifting these, because the columns are frozen once
# pretraining starts.
JOINTS = ACTION_JOINTS
DPOS = ACTION_DPOS
DROT = ACTION_DROT
GRIPPER = ACTION_GRIPPER
HAND = ACTION_HAND
EE = ACTION_EE

# A unified block is one slot's whole row, so the slices above index the
# unified chunk too and the layout of a block cannot drift from the layout of
# the row it lands in. Block k of a chunk starts at k * ARM_DIM: the stride is
# the row width, never a field's stop (trap T1).
ARM_DIM = ACTION_DIM
ARM_FIELDS = {"dpos": DPOS, "drot": DROT, "gripper": GRIPPER}

# The two fields a layout fills from their stored v7 columns (observation.joints
# and observation.hand, or a command column), never from the flat action:
# where each lands in the row. TARGET_NEXT_STATE and TARGET_COMMAND_PREFIX
# (re-exported from neo_vla.data.state) spell a part's target rule.
PART_COLUMNS = {"joints": JOINTS, "hand": HAND}
TARGET_RULES = (TARGET_NEXT_STATE, TARGET_COMMAND_PREFIX)

# The row's groups. A value outside the band takes only its own group out of
# the loss at that step (a joint outlier must not delete the EE target), and
# the loss reports one mean per group. EE is the
# seven columns every benchmark executes, gripper included.
ARM_GROUPS = {"joints": JOINTS, "ee": EE, "hand": HAND}

# What the loader does with a chunk's rotation columns, by the rotation
# convention the source's layout states. After this step every valid rotation
# column is a spatial rotation vector, so the frame rotation is one path:
# spatial is kept, a body delta is carried to the base frame by the arm's own
# orientation at that step, an Euler difference is rebuilt as the spatial delta
# between the two recorded orientations, a source that records no rotation
# trains translation only, and an unresolved convention masks the rotation.
ROTATION_CHANNEL_POLICY = {
    "spatial": "rotate",
    "body": "body_to_spatial",
    "euler_diff": "euler_to_spatial",
    "none": "translation_only",
    "unresolved": "mask_drot",
}
ROTATION_POLICIES = tuple(ROTATION_CHANNEL_POLICY.values())

# The policies whose rotation columns are a target after conversion.
ROTATION_TRAINED = ("rotate", "body_to_spatial", "euler_to_spatial")

# The policies dropped from this tree, each with what it became, so a manifest
# naming one is refused by name rather than as an unknown word.
RETIRED_POLICIES = {
    "conjugate": "body_to_spatial (the body delta is carried to the base frame first)",
    "refuse": "mask_drot (an unresolved convention masks the rotation columns)",
}

# How far a frame rotation may miss orthonormality before rotate_chunk refuses
# it. A matrix that changes the length of a delta moves the target outside the
# range the normaliser was fitted on, and nothing downstream would notice.
ROTATION_TOLERANCE = 1e-4


@dataclass(frozen=True)
class ActionLayout:
    """Where one directory's action targets come from, per arm.

    arms        one mapping per arm, from an EE field name (dpos, drot,
                gripper) to the flat action columns that fill it; arm k is
                block k of a unified chunk, and the directory's arms[k].slot
                says which action slot it fills. An arm with no EE action (a
                joints-only key) maps nothing.
    flat_dim    the width of the flat action column, so a wrong file raises
    rotation    a member of ROTATION_POLICIES, from the source's rotation
                convention
    translation_cm_per_unit  centimetres one stored translation unit is worth,
                per axis; measured for this directory, never assumed; None
                when no arm reads dpos
    rotation_rad_per_unit    radians one stored rotation unit is worth; None
                when no arm reads drot
    joints, hand  per arm, the PartLayout its joint (hand) targets come from,
                or None; () when no arm has one. The same objects as the arms'
                state layouts hold: one manifest block per arm and field.

    A column a layout does not fill stays invalid.
    """

    arms: tuple[dict[str, tuple[int, ...]], ...]
    flat_dim: int
    rotation: str
    translation_cm_per_unit: tuple[float, float, float] | None
    rotation_rad_per_unit: float | None
    joints: tuple[PartLayout | None, ...] = ()
    hand: tuple[PartLayout | None, ...] = ()

    def __post_init__(self) -> None:
        if not self.arms:
            raise ValueError(
                "layout declares no arms, expected at least one.\n"
                "A directory with no arm has nothing to pack; leave it out."
            )
        if self.rotation in RETIRED_POLICIES:
            raise ValueError(
                f"rotation is {self.rotation!r}, a policy this tree retired.\n"
                f"It became {RETIRED_POLICIES[self.rotation]}.\n"
                "Rebuild the manifest with this tree's manifest stage."
            )
        if self.rotation not in ROTATION_POLICIES:
            raise ValueError(
                f"rotation is {self.rotation!r}, expected one of "
                f"{ROTATION_POLICIES}.\nTake it from the source's convention."
            )
        for field in PART_FIELDS:
            stated = tuple(getattr(self, field))
            if stated and len(stated) != len(self.arms):
                raise ValueError(
                    f"layout states {field} for {len(stated)} arms and has {len(self.arms)}.\n"
                    f"Expected one entry per arm (None for an arm without {field}), or ().\n"
                    "Fix the manifest entry's per-arm blocks."
                )
            for arm, part in enumerate(stated):
                if part is not None:
                    check_part(part, field, f"arm {arm}")
        self._check_factors()
        claimed: dict[int, str] = {}
        for k, fields in enumerate(self.arms):
            owner = f"arm {k}"
            if not fields and self.part("joints", k) is None and self.part("hand", k) is None:
                raise ValueError(
                    f"{owner} fills no EE field, no joints and no hand.\n"
                    "Expected at least one target.\n"
                    "Leave the arm out of the action layout instead."
                )
            for name, columns in fields.items():
                if name in PART_FIELDS:
                    raise ValueError(
                        f"{owner} names field {name!r} among its flat action columns.\n"
                        f"Joints and hand come from their stored v7 columns; state them in "
                        f"the layout's {name} parts.\nThe flat action holds dpos, drot "
                        "and gripper only."
                    )
                if name not in ARM_FIELDS:
                    raise ValueError(
                        f"{owner} names field {name!r}, expected one of "
                        f"{tuple(ARM_FIELDS)}.\nFix the manifest entry."
                    )
                span = ARM_FIELDS[name]
                if len(columns) != span.stop - span.start:
                    raise ValueError(
                        f"{owner} {name} has {len(columns)} columns, expected "
                        f"{span.stop - span.start}.\nFix the manifest entry."
                    )
                for column in columns:
                    if not 0 <= column < self.flat_dim:
                        raise ValueError(
                            f"{owner} {name} reads column {column}, expected "
                            f"0 to {self.flat_dim - 1}.\nFix flat_dim or the columns."
                        )
                    if column in claimed:
                        raise ValueError(
                            f"{owner} {name} and {claimed[column]} both read column "
                            f"{column}.\nOne layout applied twice looks exactly like this."
                        )
                    claimed[column] = f"{owner} {name}"

    def _check_factors(self) -> None:
        """Raise unless each unit factor is stated exactly when an arm reads its field.

        A read field needs a finite positive factor. Zero is refused with the
        rest: a zero factor turns every delta of the directory into a still
        step, which trains as a target and looks like a source that happens
        not to move. A factor for a field no arm reads was not measured on
        anything, so it is refused too.
        """
        reads = {
            "translation_cm_per_unit": any("dpos" in fields for fields in self.arms),
            "rotation_rad_per_unit": any("drot" in fields for fields in self.arms),
        }
        stated = {
            "translation_cm_per_unit": self.translation_cm_per_unit,
            "rotation_rad_per_unit": self.rotation_rad_per_unit,
        }
        for name, read in reads.items():
            if not read:
                if stated[name] is not None:
                    raise ValueError(
                        f"{name} is {stated[name]!r} and no arm reads its field.\n"
                        "Expected None: a factor for a field nobody reads measured nothing.\n"
                        "Drop it from the manifest entry."
                    )
                continue
            if stated[name] is None:
                raise ValueError(
                    f"{name} is None and an arm reads its field.\n"
                    "Expected a finite positive factor.\n"
                    "Measure what one stored unit is worth for this directory."
                )
        translation = self.translation_cm_per_unit
        if translation is not None and len(translation) != DPOS.stop - DPOS.start:
            raise ValueError(
                f"translation_cm_per_unit has {len(translation)} "
                f"axes, expected {DPOS.stop - DPOS.start}.\n"
                "One measured factor per translation axis.\nFix the manifest entry."
            )
        values = {
            "translation_cm_per_unit": () if translation is None else tuple(translation),
            "rotation_rad_per_unit": () if self.rotation_rad_per_unit is None
            else (self.rotation_rad_per_unit,),
        }
        for name, numbers in values.items():
            for value in numbers:
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError(
                        f"{name} holds {value}, expected a finite positive factor.\n"
                        "Measure what one stored unit is worth for this directory.\n"
                        "A directory with no measured factor has no unified action."
                    )

    @property
    def n_arms(self) -> int:
        """How many arms this directory fills, so slots 0 to n_arms - 1.

        Also the number of blocks in a unified chunk, because every slot is an
        arm. The two were different numbers only while a base slot existed, and
        keeping a second name for one number is how they came apart.
        """
        return len(self.arms)

    def part(self, field: str, arm: int) -> PartLayout | None:
        """Arm k's joints or hand PartLayout, or None."""
        if field not in PART_FIELDS:
            raise ValueError(
                f"field is {field!r}, expected one of {PART_FIELDS}.\n"
                "Only joints and hand are parts of an action layout."
            )
        stated = tuple(getattr(self, field))
        return stated[int(arm)] if stated else None


def structural_dims(layout: ActionLayout, *, gripper_trained: bool) -> torch.Tensor:
    """[arms, ARM_DIM] bool: the columns the layout, its rotation policy and the gripper fill.

    dpos where an arm names it; drot where it names it and the policy trains
    rotation; the gripper where it names it and the directory's gripper method
    trains it; joints and hand on their first count columns. Before the
    directory's constants and the key's decodable columns.
    """
    dims = torch.zeros((layout.n_arms, ARM_DIM), dtype=torch.bool)
    for arm, fields in enumerate(layout.arms):
        if "dpos" in fields:
            dims[arm, DPOS] = True
        if "drot" in fields and layout.rotation in ROTATION_TRAINED:
            dims[arm, DROT] = True
        if "gripper" in fields and gripper_trained:
            dims[arm, GRIPPER] = True
        for field, span in PART_COLUMNS.items():
            part = layout.part(field, arm)
            if part is not None:
                dims[arm, span.start:span.start + part.count] = True
    return dims


def _block_count(chunk: torch.Tensor) -> int:
    """A unified chunk -> how many slot blocks it holds. Raises on a remainder."""
    if chunk.dim() != 2:
        raise ValueError(
            f"chunk is {tuple(chunk.shape)}, expected [steps, blocks x {ARM_DIM}].\n"
            "unify returns that shape; pass its output."
        )
    width = int(chunk.shape[-1])
    if width == 0 or width % ARM_DIM:
        raise ValueError(
            f"chunk is {width} columns, expected a positive multiple of {ARM_DIM}.\n"
            "A partial block would put one slot's translation in another's gripper."
        )
    return width // ARM_DIM


def block_columns(index: int, span: slice) -> slice:
    """Block index and a within-block span -> the columns of a unified chunk.

    The one place a flat column of a multi-arm chunk is computed: arm 1's
    gripper is block_columns(1, GRIPPER), flat column 39. A stride written as a
    field's stop (arm * GRIPPER.stop + GRIPPER.start, trap T1) lands on arm 1's
    joint 1 instead and raises nowhere.
    """
    if int(index) < 0:
        raise IndexError(
            f"block {index} is negative.\n"
            "Expected the arm's index in the layout, 0 or more.\n"
            "Number the arms from 0, as the layout lists them."
        )
    start = int(index) * ARM_DIM
    return slice(start + span.start, start + span.stop)



def unit_factors(layout: ActionLayout) -> dict[str, torch.Tensor]:
    """Per field name, what one stored unit of it is worth, as a row of factors.

    The gripper is a fraction of its own stroke, so it passes through at one.
    """
    factors = {"gripper": torch.ones(GRIPPER.stop - GRIPPER.start)}
    if layout.translation_cm_per_unit is not None:
        factors["dpos"] = torch.tensor(layout.translation_cm_per_unit, dtype=torch.float32)
    if layout.rotation_rad_per_unit is not None:
        factors["drot"] = torch.full((DROT.stop - DROT.start,),
                                     float(layout.rotation_rad_per_unit))
    return factors


def unify(flat: torch.Tensor, layout: ActionLayout) -> torch.Tensor:
    """A directory's raw action window -> centimetres and radians per step, float64.

    Translation is a per-step delta in centimetres and rotation is the stored
    per-step rotation delta in radians, in the source's own convention until
    the conversion step. Neither is a velocity: a velocity needs a real clock.
    The factors come from the layout, which measured them for this directory.

    The result is one block per arm in the layout's arm order, so every step
    after this reads a fixed layout and none of them needs the source's
    columns again. Joint and hand columns stay 0 here: part_targets fills them
    from their stored columns, and the chunk is unify(...) + part_targets(...)[0].
    """
    if flat.dim() != 2:
        raise ValueError(
            f"action window is {tuple(flat.shape)}, expected [steps, "
            f"{layout.flat_dim}].\nPass one window, not a batch of them."
        )
    if int(flat.shape[-1]) != layout.flat_dim:
        raise ValueError(
            f"action window is {int(flat.shape[-1])} columns, expected "
            f"{layout.flat_dim}.\nA truncated action is a relabelled dataset."
        )
    factors = unit_factors(layout)
    window = flat.to(torch.float64)
    unified = window.new_zeros((int(window.shape[0]), layout.n_arms * ARM_DIM))
    groups = [(k, fields, ARM_FIELDS) for k, fields in enumerate(layout.arms)]
    for index, fields, allowed in groups:
        for name, columns in fields.items():
            source = window.index_select(-1, torch.as_tensor(columns, dtype=torch.long))
            unified[:, block_columns(index, allowed[name])] = source * factors[name].to(torch.float64)
    return unified


def part_targets(layout: ActionLayout, records: dict[str, torch.Tensor], *,
                 inside: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Each arm's joint and hand targets for one window, and where each was read.

    records maps a stored v7 column name (observation.joints, observation.hand,
    action.joints) to its rows t .. t+K, [K + 1, arms x width]; inside [K + 1]
    says which of those rows lie in the episode. Step j's target is, by the
    part's rule:

      next_state       the record at row t+j+1 (step j's own end state)
      command:<column> that command column at row t+j

    and either way it exists only when row t+j+1 is inside the episode.
    Returns values [K, arms x ARM_DIM] float64 (0 off the part columns and
    wherever not read) and read [K, arms x ARM_DIM] bool (True only on a part
    column whose value was read: inside, a number a measurement could be, and
    below the part's count). A part whose stored column is missing is refused.
    """
    if inside.dim() != 1 or inside.dtype is not torch.bool or int(inside.shape[0]) < 2:
        raise ValueError(
            f"inside is {tuple(inside.shape)} {inside.dtype}, expected [K + 1] torch.bool.\n"
            "One flag per row t .. t+K: a step's target is read one row on.\n"
            "Read chunk + 1 rows of the window, as the Euler conversion does."
        )
    steps = int(inside.shape[0]) - 1
    values = torch.zeros((steps, layout.n_arms * ARM_DIM), dtype=torch.float64)
    read = torch.zeros((steps, layout.n_arms * ARM_DIM), dtype=torch.bool)
    for field, span in PART_COLUMNS.items():
        for arm in range(layout.n_arms):
            part = layout.part(field, arm)
            if part is None:
                continue
            command = parse_target(part.target, field)
            column = PART_RECORDS[field] if command is None else command
            rows = records.get(column)
            if rows is None:
                raise ValueError(
                    f"arm {arm} takes its {field} target from {column!r} and the window "
                    "has no such column.\n"
                    f"Expected the stored v7 column; the window holds {sorted(records)}.\n"
                    "Re-emit the directory in v7; a missing column is refused, never masked."
                )
            if int(rows.shape[0]) != steps + 1:
                raise ValueError(
                    f"{column} holds {int(rows.shape[0])} rows, expected {steps + 1}.\n"
                    "Rows t .. t+K, one more than the chunk.\n"
                    "Read the stored column over the same rows as inside."
                )
            got, ok = read_part(rows, part, field=field, block=arm)
            got, ok = (got[1:], ok[1:]) if command is None else (got[:-1], ok[:-1])
            ok = ok & inside[1:, None]
            columns = block_columns(arm, span)
            values[:, columns] = torch.where(ok, got, torch.zeros_like(got))
            read[:, columns] = ok
    return values, read


def blocks_of(chunk: torch.Tensor) -> torch.Tensor:
    """A unified chunk [steps, arms x ARM_DIM] -> [arms, steps, ARM_DIM]."""
    blocks = _block_count(chunk)
    return chunk.reshape(int(chunk.shape[0]), blocks, ARM_DIM).transpose(0, 1)


def chunk_of(blocks: torch.Tensor) -> torch.Tensor:
    """[arms, steps, ARM_DIM] -> the unified chunk [steps, arms x ARM_DIM], the inverse of blocks_of."""
    return blocks.transpose(0, 1).reshape(int(blocks.shape[1]), -1)


def body_to_spatial(chunk: torch.Tensor, base_from_ee: torch.Tensor) -> torch.Tensor:
    """A chunk whose rotation columns are body-frame deltas -> spatial ones.

    base_from_ee is [steps, arms, 3, 3], each arm's own R_base<-ee at the step
    its delta leaves from; the spatial delta is R_t w_b, because
    exp(w_s) R_t = R_t exp(w_b) holds exactly for w_s = R_t w_b. Every other
    column (joints, translation, gripper, hand) passes through.
    """
    blocks = _block_count(chunk)
    wanted = (int(chunk.shape[0]), blocks, 3, 3)
    if tuple(base_from_ee.shape) != wanted:
        raise ValueError(
            f"end-effector orientation is {tuple(base_from_ee.shape)}, expected {wanted}.\n"
            "One matrix per step of the window and per arm, in block order.\n"
            "Read the arm's pose at every step of the window."
        )
    turned = chunk.to(torch.float64).clone()
    matrices = base_from_ee.to(torch.float64)
    for index in range(blocks):
        span = block_columns(index, DROT)
        turned[:, span] = torch.einsum("nij,nj->ni", matrices[:, index], turned[:, span])
    return turned


def euler_to_spatial(chunk: torch.Tensor, orientations: torch.Tensor) -> torch.Tensor:
    """A chunk whose rotation columns are Euler differences -> spatial deltas.

    orientations is [steps + 1, arms, 3, 3]: each arm's recorded R_base<-ee at
    the step each delta leaves from and at the one after the last, read from
    the source's Euler angles. The spatial delta of step k is
    log(R_k+1 R_k^T). The stored columns are replaced, not adjusted: a
    difference of Euler angles is no rotation vector. Every other column
    passes through.
    """
    blocks = _block_count(chunk)
    wanted = (int(chunk.shape[0]) + 1, blocks, 3, 3)
    if tuple(orientations.shape) != wanted:
        raise ValueError(
            f"orientations are {tuple(orientations.shape)}, expected {wanted}.\n"
            "One matrix per step of the window plus the row after it, per arm.\n"
            "Read the state one row past the window."
        )
    matrices = orientations.to(torch.float64)
    delta = so3_log(matrices[1:] @ matrices[:-1].transpose(-1, -2))
    turned = chunk.to(torch.float64).clone()
    for index in range(blocks):
        turned[:, block_columns(index, DROT)] = delta[:, index]
    return turned


def _is_rotation(matrices: torch.Tensor) -> bool:
    """Whether every 3x3 in a stack is orthonormal with a positive determinant."""
    identity = torch.eye(3, dtype=matrices.dtype).expand_as(matrices)
    product = matrices.transpose(-1, -2) @ matrices
    if not torch.allclose(product, identity, atol=ROTATION_TOLERANCE):
        return False
    return bool((torch.linalg.det(matrices) > 0.0).all())


def rotate_chunk(
    chunk: torch.Tensor,
    base_from_reference: dict[int, torch.Tensor],
    policy: str,
) -> torch.Tensor:
    """Express the blocks of a unified chunk in their reference camera frames.

    base_from_reference is keyed by block (arm k of the layout). Each value is
    R_base_from_camera of the camera that arm's anchor names, and its delta is
    read in that reference frame as x @ it. A block with no entry is left
    alone, which is that arm in the robot base frame; keying it this way is
    what lets two arms of one sample be written against two different cameras.

    The rotation columns are spatial by the time they reach here under every
    policy that trains them, so they turn with the frame exactly as the
    translation does. Under the two policies that train no rotation they pass
    through: those columns are invalid and carry no target. Joints, gripper
    and hand are never turned: they are absolute joint-space and aperture
    targets, the same in every frame.
    """
    if policy in RETIRED_POLICIES:
        raise ValueError(
            f"policy is {policy!r}, a policy this tree retired.\n"
            f"It became {RETIRED_POLICIES[policy]}.\n"
            "Convert the rotation columns before rotating the chunk."
        )
    if policy not in ROTATION_POLICIES:
        raise ValueError(
            f"policy is {policy!r}, expected one of {ROTATION_POLICIES}.\n"
            "It comes from the source's rotation convention."
        )
    blocks = _block_count(chunk)
    matrices = _references(base_from_reference, blocks)
    turned = chunk.clone()
    spans = (DPOS, DROT) if policy in ROTATION_TRAINED else (DPOS,)
    for index, matrix in sorted(matrices.items()):
        for span in spans:
            # Row-vector convention: x @ R is R^T x, the delta read in the
            # reference frame rather than the base frame.
            turned[:, block_columns(index, span)] = chunk[:, block_columns(index, span)] @ matrix.to(chunk.dtype)
    return turned


def _references(
    given: dict[int, torch.Tensor], blocks: int
) -> dict[int, torch.Tensor]:
    """The per-block frame rotations, each audited before anything turns by it."""
    checked: dict[int, torch.Tensor] = {}
    for block, rotation in sorted(given.items()):
        if not 0 <= int(block) < blocks:
            raise IndexError(
                f"a reference claims block {block}, this chunk holds {blocks}.\n"
                "Expected a block inside the chunk; block k is the layout's arm k.\n"
                "Fix the caller's arm numbering."
            )
        if tuple(rotation.shape) != (3, 3):
            raise ValueError(
                f"block {block} has rotation {tuple(rotation.shape)}, expected (3, 3).\n"
                "Pass R_base_from_reference, the matrix the extrinsics token states."
            )
        matrix = rotation.to(torch.float64)
        if not _is_rotation(matrix):
            raise ValueError(
                f"block {block} was given a matrix that is not a rotation, so it "
                "would change the length of every delta.\n"
                "Emit that slot in robot mode rather than repairing it here."
            )
        checked[int(block)] = matrix
    return checked


def group_band(in_range: torch.Tensor, groups: dict[str, slice]) -> torch.Tensor:
    """[..., D] per-dimension in-range flags -> [..., D] in-band flags by group.

    A dimension out of range takes every dimension of its own group out of
    band at that step and nothing else: the arm's groups are ARM_GROUPS
    (joints, EE, hand), the body's neo_vla.data.body.BODY_GROUPS (base,
    posture). The groups must tile 0 .. D - 1 exactly once.
    """
    width = int(in_range.shape[-1])
    covered = sorted(c for span in groups.values() for c in range(span.start, span.stop))
    if covered != list(range(width)):
        raise ValueError(
            f"the groups cover columns {covered} of a {width}-wide row.\n"
            "Expected every column in exactly one group.\n"
            "Pass ARM_GROUPS for an arm row or body.BODY_GROUPS for the body."
        )
    if in_range.dtype is not torch.bool:
        raise TypeError(
            f"in_range is {in_range.dtype}, expected torch.bool.\n"
            "One flag per (step, dimension)."
        )
    band = in_range.clone()
    for span in groups.values():
        band[..., span] = in_range[..., span].all(dim=-1, keepdim=True)
    return band


def sample_masks(steps_valid: torch.Tensor, dims_valid: torch.Tensor,
                 read: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Which steps are keys and which columns the sample trains, both per sample.

    steps_valid [..., K]     the step exists (inside the episode, a stored action)
    dims_valid  [..., D]     the columns the directory and key train
    read        [..., K, D]  the value at (step, column) was read

    A step is kept (a key) when it exists and at least one trained column was
    read there. A column is valid for the sample when it is trained and read
    on every kept step, so the mask is constant over the chunk exactly as it is
    at serve, and no unread value enters the action input as a target (an
    AgiBot episode without odometry masks the base for the whole sample; a
    joint hole masks that joint, not the step). A kept step
    whose only read columns were dropped this way is no longer kept, and a
    sample with no kept step has no valid column. Returns (action_valid
    [..., K], dim_valid [..., D]).
    """
    if read.shape != steps_valid.shape + dims_valid.shape[-1:] \
            or steps_valid.shape[:-1] != dims_valid.shape[:-1]:
        raise ValueError(
            f"steps_valid {tuple(steps_valid.shape)}, dims_valid {tuple(dims_valid.shape)} "
            f"and read {tuple(read.shape)} do not agree.\n"
            "Expected [..., K], [..., D] and [..., K, D].\n"
            "One flag per step, per column, and per (step, column)."
        )
    available = steps_valid[..., :, None] & dims_valid[..., None, :] & read
    first = available.any(dim=-1)
    held = (read | ~first[..., :, None]).all(dim=-2)
    dim_valid = dims_valid & held & first.any(dim=-1, keepdim=True)
    action_valid = first & dim_valid.any(dim=-1, keepdim=True)
    return action_valid, dim_valid


def pack_slots(
    targets: torch.Tensor,
    *,
    slots: tuple[int, ...],
    n_slots: int,
    chunk_len: int,
    action_dim: int,
    steps_valid: torch.Tensor,
    dims_valid: torch.Tensor,
    read: torch.Tensor,
    in_band: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalised per-arm targets -> actions, action_valid, action_loss_dims and dim_valid.

    targets       [arms, chunk_len, ARM_DIM], normalise_actions' output per arm,
                  finite everywhere
    slots         the action slot each arm fills, arms[k].slot
    steps_valid   [arms, chunk_len] bool, the step exists
    dims_valid    [arms, ARM_DIM] bool, the columns the directory and key train
    read          [arms, chunk_len, ARM_DIM] bool, the value was read: the
                  pose reads of an EE conversion, a joint or hand reading
    in_band       [arms, chunk_len, ARM_DIM] bool, from normalise_actions,
                  already spread over its group (group_band)

    actions           [n_slots, chunk_len, action_dim] float32, finite, 0 off
                      the kept steps and valid columns and on empty slots
    action_valid      [n_slots, chunk_len] bool, the attention keys
                      (sample_masks); the band leaves them alone
    action_loss_dims  [n_slots, chunk_len, action_dim] bool, a key, a valid
                      column, and in band
    dim_valid         [n_slots, action_dim] bool, per sample (sample_masks)
    """
    if action_dim != ARM_DIM:
        raise ValueError(
            f"action_dim is {action_dim}, expected {ARM_DIM}.\n"
            "The columns are frozen at pretraining; change the config, not the slices."
        )
    arms = len(slots)
    if len(set(slots)) != arms or any(not 0 <= int(slot) < n_slots for slot in slots):
        raise ValueError(
            f"arms fill slots {list(slots)} of {n_slots}.\n"
            "Expected distinct slots below the model's n_slots.\n"
            "Fix the arms' slots in the manifest, or raise model.action.n_slots."
        )
    expected = {
        "targets": ((arms, chunk_len, ARM_DIM), None),
        "steps_valid": ((arms, chunk_len), torch.bool),
        "dims_valid": ((arms, ARM_DIM), torch.bool),
        "read": ((arms, chunk_len, ARM_DIM), torch.bool),
        "in_band": ((arms, chunk_len, ARM_DIM), torch.bool),
    }
    given = {"targets": targets, "steps_valid": steps_valid, "dims_valid": dims_valid,
             "read": read, "in_band": in_band}
    for name, (shape, dtype) in expected.items():
        value = given[name]
        if tuple(value.shape) != shape or (dtype is not None and value.dtype is not dtype):
            raise ValueError(
                f"{name} is {tuple(value.shape)} {value.dtype}, expected {shape}"
                f"{'' if dtype is None else f' {dtype}'}.\n"
                "One row per arm of the directory, in the layout's arm order.\n"
                "Pass RobotDataset.targets and normalise_actions' outputs."
            )
    if not bool(torch.isfinite(targets).all()):
        raise ValueError(
            "a normalised target is not finite.\n"
            "Expected normalise_actions to have set every non-finite value to 0.\n"
            "Normalise through neo_vla.data.embodiment.normalise_actions."
        )
    kept, columns = sample_masks(steps_valid, dims_valid, read)
    actions = torch.zeros((n_slots, chunk_len, action_dim), dtype=torch.float32)
    action_valid = torch.zeros((n_slots, chunk_len), dtype=torch.bool)
    action_loss_dims = torch.zeros((n_slots, chunk_len, action_dim), dtype=torch.bool)
    dim_valid = torch.zeros((n_slots, action_dim), dtype=torch.bool)
    for arm, slot in enumerate(slots):
        live = kept[arm][:, None] & columns[arm][None, :]
        actions[slot] = torch.where(live, targets[arm].to(torch.float32), torch.zeros(()))
        action_valid[slot] = kept[arm]
        action_loss_dims[slot] = live & in_band[arm]
        dim_valid[slot] = columns[arm]
    return actions, action_valid, action_loss_dims, dim_valid


def check_gripper_columns(gripper_dims: tuple[int, ...]) -> None:
    """Raise unless the model's gripper columns are this module's GRIPPER slice.

    The model config names the aperture columns for its diagnostic split and
    this module names them for the packer. They describe the same columns, so
    the run stops when they disagree rather than reporting one channel's error
    under another channel's name. v6's [6] is joint 6 under v7 (trap T6).
    """
    expected = (GRIPPER_COLUMN,)
    stated = tuple(int(column) for column in gripper_dims)
    if stated != expected:
        raise ValueError(
            f"model.action.gripper_dims is {list(stated)}, expected {list(expected)}.\n"
            "Aperture is one column per arm row, after the joints and the EE delta."
        )
