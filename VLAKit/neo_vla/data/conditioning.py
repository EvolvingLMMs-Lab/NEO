"""The state records' codes, the per-view pose record, and the one way a record says unknown.

Each sample carries two state records per action slot and two body records,
one at the history step p and one at the current step t, with a code per
(role, slot) and per role for the body: measured, unknown, the no-history
form, or absent. Validity is per dimension beside the record (state_dims,
body_state_dims): MEASURED means at least one valid dimension, and the record
is a number exactly there. build_conditioning writes the codes in the order
STATE_CODE_PRECEDENCE states and every non-measured record as all NaN with no
valid dimension, so a dropped mask fails loudly instead of teaching a pose.

The prefix's per-view channels are intrinsics and the camera extrinsics beside
them, each with its own flag: neo_vla.data.rays builds the first and
view_extrinsics here builds the second, both through mark_unknown.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..constants import (
    BODY_DIM,
    N_ROLES,
    ROLE_START,
    STATE_ABSENT,
    STATE_CODE_PRECEDENCE,
    STATE_DIM,
    STATE_MEASURED,
    STATE_NO_HISTORY,
    STATE_UNKNOWN,
)

# What an unmeasured number is filled with. Not zero: a zero that escapes its
# mask is a plausible pose and nothing downstream can tell it from a real one,
# while this cannot be mistaken for a measurement by any consumer.
UNKNOWN = float("nan")

# Extrinsics are the camera's forward and up axes written in the robot base
# frame: columns 2 and -1 of R_base_from_camera. The third axis is their cross
# product, so nothing about the rotation is lost by sending six numbers. The
# width is held against neo_vla.model.tokenizers.EXTRINSICS_DIM whenever the
# anchor table is read, which happens inside the functions: this module is
# imported by neo_vla.data.rays while neo_vla.model is still importing.
EXTRINSICS_DIM = 6
EXTRINSICS_FORWARD = slice(0, 3)
EXTRINSICS_UP = slice(3, EXTRINSICS_DIM)

# The two words a manifest may use for a directory's frame. The anchor row is
# finer than either: in camera mode it names which camera, which is why the row
# is built from a slot and not from one of these.
FRAME_REF_NAMES = ("camera", "robot")


def _anchors():
    """The model's anchor table: MAX_CAMERAS, the robot and camera rows, the row count."""
    from ..model import tokenizers

    if tokenizers.EXTRINSICS_DIM != EXTRINSICS_DIM:
        raise ValueError(
            f"the extrinsics tokenizer reads {tokenizers.EXTRINSICS_DIM} numbers and this "
            f"module writes {EXTRINSICS_DIM}.\n"
            "Expected one width; the record is forward then up.\n"
            "Change EXTRINSICS_DIM together with the tokenizer."
        )
    return tokenizers


def parse_frame_ref(spell):
    """A manifest frame spelling -> (kind, slot). "camera" is a draw; "camera:N" pins slot N.

    Bare "camera" returns slot None, which means "whichever camera the sample drew" -- the
    behaviour every manifest written before this had, and still the right answer whenever more
    than one camera carries a pose. "camera:N" returns N and asserts, in _check_camera_mode, that
    N is the only eligible camera: a pinned corpus is one whose draw is degenerate, and the
    spelling is how eval learns which slot that is without re-deriving the pool.
    """
    text = str(spell).strip()
    if text == "robot":
        return ("robot", None)
    if text == "camera":
        return ("camera", None)
    if text.startswith("camera:"):
        tail = text.split(":", 1)[1]
        if tail.isdigit():
            return ("camera", int(tail))
    raise ValueError(
        "frame_ref is %r.\n"
        "Expected 'robot', 'camera', or 'camera:N' pinning camera slot N.\n"
        "There is no third frame; fix the entry." % (spell,)
    )

# How far a matrix may sit from orthonormal and still be read as a rotation.
# Loose enough for float32 round-off in a stored calibration and far tighter
# than the smallest error worth teaching, a tenth of a degree being 1.7e-3.
ROTATION_TOLERANCE = 1e-4


@dataclass(frozen=True)
class Conditioning:
    """One sample's state and body records, their codes and dims, and the per-slot anchors.

    state            [N_ROLES, n_slots, STATE_DIM] float32, role 0 the history
                     step p and role 1 the current step t
    state_code       [N_ROLES, n_slots] long, by STATE_CODE_PRECEDENCE
    state_dims       [N_ROLES, n_slots, STATE_DIM] bool, the valid dimensions
    slot_valid       [n_slots] bool, the arm exists in this source
    history          whether the sample carries history (the no-history form when not)
    frame_ref        [n_slots] long, one anchor row per slot
    body_state       [N_ROLES, BODY_DIM] float32, the body at p and t
    body_state_code  [N_ROLES] long
    body_state_dims  [N_ROLES, BODY_DIM] bool
    body_present     the key declares a body (False: ABSENT in both roles)

    Under MEASURED a record is finite exactly where its dims are True and has
    at least one; under every other code it is all NaN with no valid dim (the
    no-history zeros and ones are the input's, masked_input). The state is in
    the robot base frame and the body in the base frame, in every role.
    """

    state: torch.Tensor
    state_code: torch.Tensor
    state_dims: torch.Tensor
    slot_valid: torch.Tensor
    history: bool
    frame_ref: torch.Tensor
    body_state: torch.Tensor
    body_state_code: torch.Tensor
    body_state_dims: torch.Tensor
    body_present: bool

    def check(self, n_slots: int) -> None:
        """Raise unless shapes, codes, precedence, dims and the NaN marker all agree."""
        shapes = {
            "state": (self.state, (N_ROLES, n_slots, STATE_DIM), None),
            "state_code": (self.state_code, (N_ROLES, n_slots), torch.long),
            "state_dims": (self.state_dims, (N_ROLES, n_slots, STATE_DIM), torch.bool),
            "slot_valid": (self.slot_valid, (n_slots,), torch.bool),
            "body_state": (self.body_state, (N_ROLES, BODY_DIM), None),
            "body_state_code": (self.body_state_code, (N_ROLES,), torch.long),
            "body_state_dims": (self.body_state_dims, (N_ROLES, BODY_DIM), torch.bool),
        }
        for name, (value, shape, dtype) in shapes.items():
            if tuple(value.shape) != shape or (dtype is not None and value.dtype is not dtype):
                raise ValueError(
                    f"{name} is {tuple(value.shape)} {value.dtype}, expected {shape}"
                    f"{'' if dtype is None else f' {dtype}'}.\n"
                    "Two roles; one record per action slot and one body.\n"
                    "Build it with build_conditioning."
                )
        for name, codes, present in (
            ("state", self.state_code, self.slot_valid),
            ("body_state", self.body_state_code[:, None],
             torch.tensor([bool(self.body_present)])),
        ):
            expected = state_codes(slot_valid=present, history=self.history,
                                   measured=codes == STATE_MEASURED)
            if not torch.equal(expected, codes):
                raise ValueError(
                    f"{name} codes {codes.tolist()} break the precedence, expected "
                    f"{expected.tolist()}.\n"
                    "Expected absent before no-history before unknown before measured.\n"
                    "Build the codes with state_codes."
                )
        check_records("state", self.state, self.state_code, self.state_dims)
        check_records("body_state", self.body_state, self.body_state_code,
                      self.body_state_dims)
        full = (self.body_state_code == STATE_MEASURED) & self.body_state_dims.all(dim=-1)
        if bool(full.any()):
            raise ValueError(
                f"body role {int(full.nonzero()[0])} is measured with all {BODY_DIM} dims valid.\n"
                "Expected at least one invalid dim: a still base and mid-range posture\n"
                "would then be the no-history input [0, 1] exactly. No source records all eight."
            )
        if self.frame_ref.dtype is not torch.long:
            raise TypeError(
                f"frame_ref is {self.frame_ref.dtype}, expected torch.long.\n"
                "It indexes the anchor rows; build it with frame_ref_rows."
            )
        if tuple(self.frame_ref.shape) != (n_slots,):
            raise ValueError(
                f"frame_ref is {tuple(self.frame_ref.shape)}, expected {(n_slots,)}.\n"
                "One anchor per slot; two arms may name different cameras.\n"
                "Build it with frame_ref_rows."
            )
        anchors = _anchors()
        outside = (self.frame_ref < 0) | (self.frame_ref >= anchors.FRAME_REF_KINDS)
        if bool(outside.any()):
            raise ValueError(
                f"frame_ref holds {int(self.frame_ref[outside][0])}, expected a "
                f"camera slot below {anchors.MAX_CAMERAS} or {anchors.FRAME_REF_ROBOT} "
                "for robot.\n"
                "Build it with frame_ref_rows; there is no third frame."
            )


def check_records(name: str, record: torch.Tensor, codes: torch.Tensor,
                  dims: torch.Tensor) -> None:
    """Raise unless each token's numbers and dims agree with its code.

    MEASURED: finite exactly where dims is True, and at least one dim. Every
    other code: all NaN and no valid dim. record [..., W], codes [...],
    dims [..., W].
    """
    measured = codes == STATE_MEASURED
    finite = torch.isfinite(record)
    agree = torch.where(measured[..., None], finite == dims, ~finite & ~dims).all(dim=-1)
    agree &= ~measured | dims.any(dim=-1)
    if not bool(agree.all()):
        index = tuple((~agree).nonzero()[0].tolist())
        code = int(codes[index])
        held = "no valid dimension" if code == STATE_MEASURED and not bool(dims[index].any()) \
            else "numbers and dims that disagree"
        raise ValueError(
            f"{name} {list(index)} has code {code} and {held}.\n"
            "Expected numbers exactly on the valid dims of a measured record, at least one,\n"
            "and NaN with no valid dim under every other code. Write it through mark_unknown_dims."
        )


def state_codes(*, slot_valid: torch.Tensor, history: bool,
                measured: torch.Tensor) -> torch.Tensor:
    """Per (role, slot), the first code of STATE_CODE_PRECEDENCE that applies.

    ABSENT where the source has no arm in the slot, in both roles; NO_HISTORY
    for the start role of every existing arm when the sample has no history;
    UNKNOWN where the record was not measured; MEASURED otherwise. NO_HISTORY
    never appears in the current role.
    """
    if STATE_CODE_PRECEDENCE != (STATE_ABSENT, STATE_NO_HISTORY, STATE_UNKNOWN, STATE_MEASURED):
        raise ValueError(
            f"STATE_CODE_PRECEDENCE is {STATE_CODE_PRECEDENCE}.\n"
            "Expected absent, no-history, unknown, measured, the order this function writes.\n"
            "Change the contract and this function together."
        )
    slots = int(slot_valid.shape[0])
    if tuple(measured.shape) != (N_ROLES, slots):
        raise ValueError(
            f"measured is {tuple(measured.shape)}, expected ({N_ROLES}, {slots}).\n"
            "One flag per role and slot.\nPass the records' own flags."
        )
    codes = torch.where(measured, STATE_MEASURED, STATE_UNKNOWN).to(torch.long)
    if not history:
        codes[ROLE_START] = STATE_NO_HISTORY
    return torch.where(slot_valid[None, :], codes, torch.full_like(codes, STATE_ABSENT))


def mark_unknown(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Fill the rows of value where valid is false with UNKNOWN.

    Every producer in this package ends with this call. It exists so that
    "unknown" is one operation rather than a convention each channel restates,
    and so a test can assert the marker and the flag agree on every sample.
    """
    if valid.dtype is not torch.bool:
        raise TypeError(
            f"validity is {valid.dtype}, expected torch.bool.\n"
            "A float mask would let 0.5 mean measured; cast it in the producer."
        )
    if tuple(valid.shape) != tuple(value.shape[:-1]):
        raise ValueError(
            f"validity is {tuple(valid.shape)}, expected {tuple(value.shape[:-1])}.\n"
            "Expected one flag per record.\nFix the producer of this channel."
        )
    numbers = value.to(torch.float32)
    return torch.where(valid[..., None], numbers, torch.full_like(numbers, UNKNOWN))


def mark_unknown_dims(value: torch.Tensor, dims: torch.Tensor) -> torch.Tensor:
    """Fill the dimensions of value where dims is false with UNKNOWN.

    The per-dimension twin of mark_unknown, for the records whose validity is
    per dimension (the v7 arm state and the body). dims has value's own
    shape, so a per-record flag cannot be passed by mistake.
    """
    if dims.dtype is not torch.bool:
        raise TypeError(
            f"validity is {dims.dtype}, expected torch.bool.\n"
            "A float mask would let 0.5 mean measured; cast it in the producer."
        )
    if tuple(dims.shape) != tuple(value.shape):
        raise ValueError(
            f"validity is {tuple(dims.shape)}, expected {tuple(value.shape)}.\n"
            "Expected one flag per dimension.\n"
            "Use mark_unknown for a channel with one flag per record."
        )
    numbers = value.to(torch.float32)
    return torch.where(dims, numbers, torch.full_like(numbers, UNKNOWN))


def is_rotation(matrix: torch.Tensor) -> bool:
    """True when matrix is a proper 3x3 rotation, to ROTATION_TOLERANCE.

    Right-handed as well as orthonormal. A quaternion read in the wrong
    component order and a flipped axis both land here, and the second one has a
    determinant of minus one while passing every orthonormality check.
    """
    if tuple(matrix.shape) != (3, 3) or not bool(torch.isfinite(matrix).all()):
        return False
    values = matrix.to(torch.float64)
    gram = values.T @ values - torch.eye(3, dtype=values.dtype)
    if float(gram.abs().max()) > ROTATION_TOLERANCE:
        return False
    return abs(float(torch.det(values)) - 1.0) <= ROTATION_TOLERANCE


def extrinsics_6(rotation_base_from_camera: torch.Tensor) -> torch.Tensor:
    """A 3x3 rotation -> the six numbers the extrinsics tokens read.

    The columns of the input are the camera's axes written in the robot base
    frame, in the camera convention: x right, y down, z forward. So forward is
    column 2 and up is the negation of column 1. Raises when the input is not a
    proper rotation, rather than projecting it onto one: an extrinsic that
    failed its audit is unknown, and a repaired one is a different claim that
    belongs in the manifest.
    """
    if not is_rotation(rotation_base_from_camera):
        raise ValueError(
            f"extrinsics {tuple(rotation_base_from_camera.shape)} is not a rotation.\n"
            "Expected a right-handed orthonormal 3x3 R_base_from_camera.\n"
            "Pass None for a calibration that failed its audit; the channel then says unknown."
        )
    values = rotation_base_from_camera.to(torch.float32)
    return torch.cat([values[:, 2], -values[:, 1]])


def frame_ref_row(camera_slot: int | None) -> int:
    """A camera slot -> its anchor row, or the robot row for None.

    Raises for a slot the anchor has no row for. There is no fallback: the
    anchor names the camera every action column of the arm is written against,
    so a slot outside the table cannot resolve to a default one.
    """
    anchors = _anchors()
    if camera_slot is None:
        return anchors.FRAME_REF_ROBOT
    slot = int(camera_slot)
    if not 0 <= slot < anchors.MAX_CAMERAS:
        raise ValueError(
            f"the reference is camera slot {slot}.\n"
            f"Expected a slot below {anchors.MAX_CAMERAS}, which the anchor has a row for.\n"
            "Renumber the view's camera slot, or raise MAX_CAMERAS before pretraining."
        )
    return anchors.FRAME_REF_CAMERA + slot


def frame_ref_rows(
    cameras: dict[int, int | None], *, n_slots: int
) -> torch.Tensor:
    """Each action slot's reference camera slot -> the [n_slots] anchor rows.

    A slot absent from the mapping, or present with None, reads the robot row:
    that is a slot whose action columns are in the robot base frame, which
    covers every arm no usable camera can serve (the pose audit runs before
    the draw) and every slot of a directory that offers no camera at all.
    """
    rows = torch.full((int(n_slots),), _anchors().FRAME_REF_ROBOT, dtype=torch.long)
    for slot, camera in sorted(cameras.items()):
        if not 0 <= int(slot) < int(n_slots):
            raise IndexError(
                f"a reference claims slot {slot}, this sample has {n_slots}.\n"
                "Expected a slot inside the action tensor's slot axis.\n"
                "Raise model.action.n_slots, or fix the arm's slot in the manifest."
            )
        rows[int(slot)] = frame_ref_row(camera)
    return rows


def frame_spelling(row: int) -> str:
    """An anchor row -> the spelling a trained stamp keys it by: "robot" or "camera:k".

    The inverse of frame_ref_row, and the one place the stamp's spellings are
    written, so the loader's realised frames and the server's R12 read one
    vocabulary.
    """
    anchors = _anchors()
    value = int(row)
    if value == anchors.FRAME_REF_ROBOT:
        return "robot"
    slot = value - anchors.FRAME_REF_CAMERA
    if not 0 <= slot < anchors.MAX_CAMERAS:
        first = anchors.FRAME_REF_CAMERA
        raise ValueError(
            f"anchor row {value} names no frame.\n"
            f"Expected {anchors.FRAME_REF_ROBOT} for the robot or a camera row "
            f"{first}..{first + anchors.MAX_CAMERAS - 1}.\n"
            "Build the row with frame_ref_row."
        )
    return f"camera:{slot}"


def view_extrinsics(
    rotations: dict[str, torch.Tensor | None],
    *,
    slots: dict[str, int],
    n_views: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One step's camera poses -> the six numbers per camera slot, and their validity.

    values [n_views, 6] float32, forward then up as EXTRINSICS_FORWARD and
    EXTRINSICS_UP cut them, and valid [n_views] bool. A camera the manifest
    gives no pose, one whose stored pose failed its audit, and every slot no
    camera fills, is unknown: the flag is false and the row holds the marker
    mark_unknown writes. The caller passes None for each of those; a pose that
    arrives here is one it already audited.

    This is the camera's own pose and it is keyed by camera slot, the slot
    <cam_k> names. It is not the frame an arm's actions are written in: that is
    the suffix anchor, keyed by action slot, and the two were one record only
    while the reference was per arm.

    Keyed and refused exactly as rays.calibration_for_views is, because the two
    channels describe the same views in the same block of the prefix: a view
    named in one and left out of the other is a producer that asked which
    cameras this step shows twice and got two answers.
    """
    if set(rotations) != set(slots):
        raise ValueError(
            f"poses name {sorted(rotations)} and slots {sorted(slots)}.\n"
            "Expected one entry per camera in both, null where a view has no "
            "measured pose.\n"
            "An omitted camera states nothing, and says nothing about why."
        )
    _anchors()
    values = torch.zeros((n_views, EXTRINSICS_DIM), dtype=torch.float32)
    valid = torch.zeros(n_views, dtype=torch.bool)
    for key in sorted(rotations):
        rotation = rotations[key]
        if rotation is None:
            continue
        slot = int(slots[key])
        if not 0 <= slot < n_views:
            raise ValueError(
                f"camera {key!r} asks for slot {slot}, the layout has "
                f"{n_views}.\nExpected a slot in 0..{n_views - 1}.\n"
                "Raise model.prefix.n_views or move the camera in the manifest."
            )
        if bool(valid[slot]):
            raise ValueError(
                f"camera {key!r} and another both claim slot {slot}.\n"
                "Expected one camera per slot: the second would overwrite the "
                "first.\nGive them different slots in the manifest."
            )
        values[slot] = extrinsics_6(rotation)
        valid[slot] = True
    return mark_unknown(values, valid), valid


def masked_input(record: torch.Tensor, dims: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """Records [..., W], dims [..., W] and codes [...] -> the state input [..., 2 W].

    [x * m, m] for a MEASURED record (x the normalised record, m its dims); the
    zero record with an all-ones mask for NO_HISTORY, the one input no real
    record can give (an arm's 6D is never all zero, and a body never has all
    eight dims); zeros for UNKNOWN and ABSENT, whose tokens are the learned
    unknown row and a masked column instead. NaN never leaves this function.
    """
    measured = (codes == STATE_MEASURED)[..., None]
    fresh = (codes == STATE_NO_HISTORY)[..., None]
    mask = torch.where(measured, dims, fresh.expand_as(dims)).to(torch.float32)
    values = torch.where(measured & dims, record.to(torch.float32), torch.zeros(()))
    return torch.cat([values, mask], dim=-1)


def build_conditioning(
    start: torch.Tensor,
    start_dims: torch.Tensor,
    now: torch.Tensor,
    now_dims: torch.Tensor,
    *,
    slot_valid: torch.Tensor,
    history: bool,
    frame_ref: torch.Tensor,
    body_start: torch.Tensor,
    body_start_dims: torch.Tensor,
    body_now: torch.Tensor,
    body_now_dims: torch.Tensor,
    body_present: bool,
) -> Conditioning:
    """Assemble one sample's records and codes, marking every unknown dimension.

    start and now are [n_slots, STATE_DIM] with their dims, body_start and
    body_now [BODY_DIM] with theirs; start records are ignored where the sample
    has no history. A record is measured when at least one dim is valid. The
    only constructor of Conditioning: it writes the codes, clears every record
    whose code is not measured to all NaN with no valid dim, and runs check,
    so a Conditioning that exists is one whose numbers, dims and codes agree.
    """
    for name, record, width in (("start", start, STATE_DIM), ("now", now, STATE_DIM)):
        if record.dim() != 2 or int(record.shape[-1]) != width:
            raise ValueError(
                f"{name} state is {tuple(record.shape)}, expected [slots, {width}].\n"
                "Expected the action tensor's slot axis.\n"
                "Build it with neo_vla.data.state.split_arms."
            )
    for name, record in (("body_start", body_start), ("body_now", body_now)):
        if tuple(record.shape) != (BODY_DIM,):
            raise ValueError(
                f"{name} is {tuple(record.shape)}, expected ({BODY_DIM},).\n"
                "One body record per role.\n"
                "Build it with neo_vla.data.body.read_body."
            )
    dims = torch.stack([start_dims, now_dims]).to(torch.bool)
    codes = state_codes(slot_valid=slot_valid, history=bool(history), measured=dims.any(dim=-1))
    dims = dims & (codes == STATE_MEASURED)[..., None]
    body_dims = torch.stack([body_start_dims, body_now_dims]).to(torch.bool)
    body_codes = state_codes(
        slot_valid=torch.tensor([bool(body_present)]), history=bool(history),
        measured=body_dims.any(dim=-1)[:, None],
    )[:, 0]
    body_dims = body_dims & (body_codes == STATE_MEASURED)[:, None]
    conditioning = Conditioning(
        state=mark_unknown_dims(torch.stack([start, now]), dims), state_code=codes,
        state_dims=dims, slot_valid=slot_valid.to(torch.bool), history=bool(history),
        frame_ref=frame_ref.to(torch.long),
        body_state=mark_unknown_dims(torch.stack([body_start, body_now]), body_dims),
        body_state_code=body_codes, body_state_dims=body_dims,
        body_present=bool(body_present),
    )
    conditioning.check(int(now.shape[0]))
    return conditioning
