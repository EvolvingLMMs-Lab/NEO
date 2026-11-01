"""The conditioning channels: numbers and anchors turned into tokens.

Two banks, because they emit into two different widths. PrefixTokenizers works
at the backbone width and owns the prefix's six <state> columns (start and now
for each arm and for the body), the camera anchor ids, and the twelve geometry
columns beside each view (intrinsics_mode tokens, the one mode). A state record
is read per dimension: the bank projects [x * m, m], the record with its
invalid dimensions zeroed beside the mask of its valid ones.

SuffixTokenizers works at the expert width and owns each suffix block's row:
the frame row (<frame_ref>; <robot_mode> for the body) plus the identity row
(<arm_i>, <body>), added by the decoder to every action token of the block.

Every projection carries a learned unknown row and takes a validity mask: an
unmeasured value is said, not projected, and these banks freeze after pretrain."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .. import constants
from ..engine.hostcopy import host_of
from .deferred import require

# The reserved vocabulary rows the prefix anchors use. NEOpp-2B ships 254
# unused rows, so the embedding table is never resized and the checkpoint's
# language head stays loadable for co-training.
ANCHOR_RESERVED_BASE = 151682
ANCHOR_RESERVED_COUNT = 254
MAX_CAMERAS = 8

# The per-arm state record, the body record and their codes, stated once in
# neo_vla/constants.py and re-exported here, where the bank that reads them
# lives. Assigned rather than imported by name, so each is the constants
# module's own object.
STATE_DIM = constants.STATE_DIM
STATE_RECORD = constants.STATE_RECORD
STATE_JOINTS = constants.STATE_JOINTS
STATE_POSITION = constants.STATE_POSITION
STATE_ROTATION = constants.STATE_ROTATION
STATE_APERTURE = constants.STATE_APERTURE
STATE_HAND = constants.STATE_HAND
STATE_INPUT_DIM = constants.STATE_INPUT_DIM
BODY_DIM = constants.BODY_DIM
BODY_RECORD = constants.BODY_RECORD
BODY_INPUT_DIM = constants.BODY_INPUT_DIM
STATE_MEASURED = constants.STATE_MEASURED
STATE_UNKNOWN = constants.STATE_UNKNOWN
STATE_NO_HISTORY = constants.STATE_NO_HISTORY
STATE_ABSENT = constants.STATE_ABSENT
STATE_CODE_PRECEDENCE = constants.STATE_CODE_PRECEDENCE
STATE_TOKEN_ID = constants.STATE_TOKEN_ID
RETIRED_ANCHOR_IDS = constants.RETIRED_ANCHOR_IDS
ROLE_START = constants.ROLE_START
ROLE_NOW = constants.ROLE_NOW
N_ROLES = constants.N_ROLES
NOISE_WIDTH = constants.NOISE_WIDTH
NOISE_POSITION = constants.NOISE_POSITION
NOISE_ROTATION = constants.NOISE_ROTATION
NOISE_APERTURE = constants.NOISE_APERTURE
NOISE_JOINTS = constants.NOISE_JOINTS
NOISE_HAND = constants.NOISE_HAND
BODY_NOISE_WIDTH = constants.BODY_NOISE_WIDTH

# <cam_k> names a camera; <intrinsics>, <extrinsics> and <state> are
# placeholders whose embedding rows are overwritten with a projection on every
# forward, the way the image context token is. <intrinsics> and <extrinsics> are
# separate ids for two blocks that sit next to each other, so the scaffold check
# in the encoder catches a row where the two have traded places rather than
# writing a rotation into the columns that promised a calibration.
#
# <rays> held the ray columns of the removed tower mode. No row of the v7
# scaffold holds it and a prompt holding it is refused, but it stays in this map:
# init_anchor_rows re-seeds every row here, and the confirmed v6 init re-seeds
# twelve (8 cameras, <intrinsics>, <rays>, <extrinsics>, <state>; DESIGN, OS), so
# dropping it would move one reserved row of the embedding table off that init.
#
# The row after the last camera held <scale> until template v6. It stays in
# RETIRED_ANCHOR_IDS and is never handed out again: a prompt holding it is
# refused, and a checkpoint that trained it cannot be mistaken for one that did
# not. <state> takes the next free reserved row after <extrinsics>.
ANCHOR_IDS: dict[str, int] = {
    **{f"<cam_{k}>": ANCHOR_RESERVED_BASE + k for k in range(MAX_CAMERAS)},
    "<intrinsics>": ANCHOR_RESERVED_BASE + MAX_CAMERAS + 1,
    "<rays>": ANCHOR_RESERVED_BASE + MAX_CAMERAS + 2,
    "<extrinsics>": ANCHOR_RESERVED_BASE + MAX_CAMERAS + 3,
    "<state>": STATE_TOKEN_ID,
}
if set(ANCHOR_IDS.values()) & set(RETIRED_ANCHOR_IDS.values()):
    raise RuntimeError(
        "ANCHOR_IDS hands out a retired row.\n"
        f"Expected none of {sorted(RETIRED_ANCHOR_IDS.values())} among the anchors.\n"
        "Take the next free reserved row instead."
    )

# Rows of SuffixTokenizers.frame_ref. The anchor names the frame the arm's
# action columns are written in. Row k names current camera slot k, the same
# camera the prefix's Now block calls <cam_k>, and the last row names the robot
# base frame. Two arms of one sample may name different cameras. The state is
# in the robot base frame.
FRAME_REF_CAMERA = 0
FRAME_REF_ROBOT = MAX_CAMERAS
FRAME_REF_KINDS = MAX_CAMERAS + 1

# The body's frame is the base frame, always: its block never turns with a
# camera, so its frame row is the robot row of the same table, <robot_mode>.
BODY_FRAME_REF = FRAME_REF_ROBOT

# The 6-D rotation: two columns of a rotation matrix, which the third is the
# cross product of. Read by the prefix, where it is the camera's own pose. It
# was also the suffix's per-arm [extrinsics], the arm's base rotation, and that
# token is gone; the dimension is the same six numbers either way.
EXTRINSICS_DIM = 6

# The calibration record: fx, fy, cx, cy, width and height, which is what
# CALIBRATION_COLUMNS in neo_vla/data/rays.py names. Written out here rather
# than imported, because that package imports this module; a test in
# tests/model/test_tokenizers.py holds the two numbers together.
INTRINSICS_DIM = 6

# Columns one view's calibration and camera rotation hold under intrinsics_mode
# tokens: one token per column of the record, never a record folded into fewer,
# which is why these are written as the record widths and not as bare numbers.
# The counts are part of the prefix layout -- they move every column after them
# and they freeze with the tokenizers -- so they are constants here and not
# config keys.
INTRINSICS_TOKENS_PER_VIEW = INTRINSICS_DIM
EXTRINSICS_TOKENS_PER_VIEW = EXTRINSICS_DIM

# The largest number a record of measured geometry can hold. Beyond it the
# value is not a reading, whatever produced it, and it is dangerous rather than
# merely wrong: it is finite, so torch.isfinite passes it and nan_to_num leaves
# it alone, and it only shows itself when something squares it. One did: a
# state element of -9.65e37 in rh20t_grid_k13_bysa_e2w0x1u0 episode 10 projected to
# 2.45e38, RmsNorm squared it past the float32 ceiling,
# rsqrt(inf) gave zero, and the forward loss came out a healthy 0.124 while the
# backward computed 0 * inf and returned nan into 394 parameter gradients.
#
# Measured over 16,000 draws of the live corpus, the largest legitimate value
# of any of these records is 1280 -- an intrinsics principal point in pixels;
# state reaches 1.27, a rotation 1.0, and nothing at all exceeded 1e4. The 80.2
# of that measurement was the TM channel, which no longer exists; the bound is
# unchanged because TM was never the number that set it.
# The bound sits about eight hundred times above the largest measurement seen
# and thirty-two orders of magnitude below the corruption, so it can discard
# neither a reading nor an unseen camera's wider raster.
RECORD_LIMIT = 1.0e6


def within_range(x: torch.Tensor) -> torch.Tensor:
    """Elementwise: is this a number a measurement could be?"""
    return torch.isfinite(x) & (x.abs() <= RECORD_LIMIT)


def brought_inside(x: torch.Tensor) -> torch.Tensor:
    """The same numbers, with the ones no measurement could be brought inside.

    Applied before the projection and never after, for the same reason the
    cleaning always was: torch.where evaluates both branches, so a row this
    module discards still reaches the weight gradients of the whole batch. A
    value that overflows when squared does its damage from the discarded
    branch exactly as a nan would.
    """
    return torch.nan_to_num(
        x.float(), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp(-RECORD_LIMIT, RECORD_LIMIT)


# What model.prefix.intrinsics_mode may say to this bank: tokens, the one mode,
# and the four removed ones refused by name. Written out here rather than
# imported from neo_vla/data/rays.py, which names the same, because that package
# imports this module; a test holds the two together.
BUILT_INTRINSICS_MODES = ("tokens",)
REMOVED_INTRINSICS_MODES = ("none", "patch", "text", "tower")

# The frame size a calibration was measured against, through the curve
# u = asinh(px / s) / asinh(x_max / s). That size is the one part of the record
# compress_calibration cannot read as a ratio, so it is the one part still
# counted in pixels, and a linear encoding would crush most rasters together.
FRAME_ASINH_S_PX = 1000.0
FRAME_ASINH_X_PX = 4096.0

# The stream the calibration and camera-rotation banks are drawn from. Fixed
# rather than the run's seed, for the reason init_anchor_rows gives, and separate
# from the global one so that these banks cannot decide what the rest of the
# model is.
CALIBRATION_INIT_SEED = 0

# The stream the state bank is drawn from, fixed and separate for the same
# reason: every build has it, and drawing it from the global stream would make
# the expert that follows depend on the bank's size.
STATE_INIT_SEED = 1

# One anchor row's stream is seed * this + the row it lands in. The seed is
# multiplied first so a row and a seed cannot trade places, which is what
# mixture.stream_seed does for a draw.
ANCHOR_SEED_STRIDE = 1_000_003


def _check_view_record(
    name: str, record: torch.Tensor, valid: torch.Tensor, dim: int
) -> None:
    """Raise unless a per-view record is [..., views, dim] with a flag per view.

    The per-column banks read record[..., k:k+1], which on a record of the wrong
    width would quietly take the first `dim` columns and project them as if they
    were the right ones. The width is checked here rather than left to the
    projection, whose own message would name an axis of 1 and send the reader
    looking for the wrong defect.
    """
    if record.dim() < 2 or int(record.shape[-1]) != dim:
        raise ValueError(
            f"{name} is {tuple(record.shape)}, expected a last axis of {dim}.\n"
            "Fix the producer of this channel."
        )
    if tuple(valid.shape) != tuple(record.shape[:-1]):
        raise ValueError(
            f"{name}_valid is {tuple(valid.shape)}, expected "
            f"{tuple(record.shape[:-1])}.\nOne flag per view."
        )


def _qualified_names(prefix: str, banks) -> list[str]:
    """[(attribute, module)] -> the parameter names under prefix, in order."""
    head = f"{prefix}." if prefix else ""
    return [
        f"{head}{owner}.{name}"
        for owner, bank in banks
        for name, _ in bank.named_parameters()
    ]


@dataclass(frozen=True)
class SuffixInputs:
    """The per-arm conditioning of the suffix for one batch, on the slot axis.

    Two fields. The state moved into the prefix, so what is left per arm is
    the frame its actions are written in and whether the arm exists at all.

    frame_ref   [B, S] long, a current camera slot below FRAME_REF_ROBOT, or
                FRAME_REF_ROBOT for the robot base frame; a slot that is not
                an arm holds FRAME_REF_ROBOT
    slot_valid  [B, S] bool, the arm exists in this source or is served
    """

    frame_ref: torch.Tensor
    slot_valid: torch.Tensor

    def n_slots(self) -> int:
        """S, from slot_valid.shape[1]: the one flag every arm carries."""
        if self.slot_valid.dim() != 2:
            raise ValueError(
                f"slot_valid is {tuple(self.slot_valid.shape)}, expected "
                "[batch, slots].\nGive it the action tensor's slot axis in the producer."
            )
        return int(self.slot_valid.shape[1])

    def validate(self, batch_size: int, n_slots: int) -> None:
        """Raise unless both fields are the right shape, dtype and slot count."""
        expected = (batch_size, n_slots)
        if tuple(self.slot_valid.shape) != expected:
            raise ValueError(
                f"slot_valid is {tuple(self.slot_valid.shape)}, expected {expected}.\n"
                "One flag per arm."
            )
        if self.slot_valid.dtype is not torch.bool:
            raise TypeError(
                f"slot_valid is {self.slot_valid.dtype}, expected torch.bool.\n"
                "A float mask would let 0.5 mean present; cast it upstream."
            )
        if self.frame_ref.dtype is not torch.long:
            raise TypeError(
                f"frame_ref is {self.frame_ref.dtype}, expected torch.long.\n"
                "It indexes anchor rows."
            )
        if tuple(self.frame_ref.shape) != expected:
            raise ValueError(
                f"frame_ref is {tuple(self.frame_ref.shape)}, expected {expected}.\n"
                "One anchor per arm; a slot that is not an arm names the robot "
                "frame. The per-sample form [B] is no longer read."
            )


@dataclass(frozen=True)
class StateInputs:
    """The six <state> columns' inputs for one batch, in model-input space.

    The arms' records and the body's, each with one code per role and the mask
    of its valid dimensions, named as the batch names them. policy.prefix_state
    builds this from the batch's physical records through the run's normaliser
    (and perturb_state moves it in training); the bank reads [x * m, m].

    state            [B, 2, S, STATE_DIM] float, a model-input value wherever
                     state_dims is True; ignored elsewhere
    state_dims       [B, 2, S, STATE_DIM] bool, the valid dimensions of a
                     MEASURED record, all False under every other code
    state_code       [B, 2, S] long, the codes of neo_vla.constants
    body_state       [B, 2, BODY_DIM] float, as state for the body
    body_state_dims  [B, 2, BODY_DIM] bool
    body_state_code  [B, 2] long; ABSENT in both roles is a key without a body

    Roles run start then now. NO_HISTORY's record is not carried: the bank
    writes it, the zero record with an all-ones mask.
    """

    state: torch.Tensor
    state_dims: torch.Tensor
    state_code: torch.Tensor
    body_state: torch.Tensor
    body_state_dims: torch.Tensor
    body_state_code: torch.Tensor

    def replace(self, **changes) -> "StateInputs":
        """The same inputs with the named fields swapped."""
        return dataclasses.replace(self, **changes)

    @property
    def body_present(self) -> torch.Tensor:
        """[B] bool: the sample's key has a body, read off the now role's code."""
        return self.body_state_code[:, ROLE_NOW] != STATE_ABSENT

    def validate(self, n_slots: int) -> int:
        """Raise unless every field has its shape and dtype; return the batch size."""
        code = self.state_code
        if code.dim() != 3 or tuple(code.shape[1:]) != (N_ROLES, int(n_slots)):
            raise ValueError(
                f"state_code is {tuple(code.shape)}, expected [batch, {N_ROLES}, {n_slots}].\n"
                "One code per role and arm, start then now."
            )
        batch = int(code.shape[0])
        expected = {
            "state": ((batch, N_ROLES, int(n_slots), STATE_DIM), None),
            "state_dims": ((batch, N_ROLES, int(n_slots), STATE_DIM), torch.bool),
            "state_code": ((batch, N_ROLES, int(n_slots)), torch.long),
            "body_state": ((batch, N_ROLES, BODY_DIM), None),
            "body_state_dims": ((batch, N_ROLES, BODY_DIM), torch.bool),
            "body_state_code": ((batch, N_ROLES), torch.long),
        }
        for name, (shape, dtype) in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"{name} is {tuple(value.shape)}, expected {shape}.\n"
                    f"The arm record is {STATE_RECORD} ({STATE_DIM}) and the body's "
                    f"{BODY_RECORD} ({BODY_DIM}), start then now.\n"
                    "Build the inputs with policy.prefix_state."
                )
            if dtype is not None and value.dtype is not dtype:
                raise TypeError(
                    f"{name} is {value.dtype}, expected {dtype}.\n"
                    "A float mask would let 0.5 mean valid; a code indexes the "
                    "codes of neo_vla.constants. Cast it upstream."
                )
            if dtype is None and not value.is_floating_point():
                raise TypeError(f"{name} is {value.dtype}, expected a float record.")
        return batch


def masked_record(
    record: torch.Tensor, dims: torch.Tensor, code: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A record, its valid dimensions and its codes -> ([x * m, m], who projects).

    The bank's input under each code: MEASURED reads the
    record with its invalid dimensions zeroed beside its mask; NO_HISTORY is
    the zero record with an all-ones mask, written here whatever the caller
    left in those rows; UNKNOWN and ABSENT take the unknown row, so the second
    result is False there. NaN and impossible values are cleaned before the
    product (brought_inside), so a dimension the mask drops cannot reach a
    gradient from the discarded side of a torch.where.
    """
    no_history = (code == STATE_NO_HISTORY)[..., None]
    mask = torch.where(no_history, torch.ones_like(dims), dims).to(torch.float32)
    values = torch.where(no_history, torch.zeros_like(record, dtype=torch.float32),
                         brought_inside(record))
    features = torch.cat([values * mask, mask], dim=-1)
    projects = (code == STATE_MEASURED) | (code == STATE_NO_HISTORY)
    return features, projects


def check_measured_record(
    name: str, record: torch.Tensor, dims: torch.Tensor, code: torch.Tensor
) -> None:
    """Raise unless the codes and the per-dimension masks agree.

    Under MEASURED at least one dimension is valid and every valid one is a
    number a measurement could be; under every other code no dimension is
    valid. The mask and the code are two statements of one fact, so a producer
    that disagrees with itself stops here rather than at a projection.
    """
    # The codes and the masks on their host copies (engine.hostcopy); only the
    # range check reads the device record, deferred (model.deferred).
    host_dims, host_code = host_of(dims), host_of(code)
    measured = host_code == STATE_MEASURED
    if bool(host_dims[~measured].any()):
        raise ValueError(
            f"{name}_dims marks a dimension valid on a code that is not MEASURED.\n"
            "Expected every dimension invalid under NO_HISTORY, UNKNOWN and ABSENT: "
            "the bank writes NO_HISTORY's all-ones mask itself.\n"
            "Fix the producer (Conditioning.check)."
        )
    if bool((measured & ~host_dims.any(dim=-1)).any()):
        raise ValueError(
            f"a {name} the code calls MEASURED has no valid dimension.\n"
            "Expected MEASURED to mean at least one dimension was read.\n"
            "Mark it UNKNOWN in the producer."
        )
    readable = within_range(record) | ~dims
    on_device = (code == STATE_MEASURED).to(readable.device)
    require((readable | ~on_device[..., None]).all(), lambda: refuse_unreadable(name))


def refuse_unreadable(name: str) -> None:
    """check_measured_record's refusal: a valid MEASURED dimension is not a model-input number."""
    raise ValueError(
        f"a valid dimension of a {name} the code calls MEASURED is not a finite "
        "model-input number.\n"
        "Expected policy.prefix_state to have normalised and clipped it.\n"
        "Clear that dimension's mask in the producer if it was not read."
    )


class NumericTokenizer(nn.Module):
    """A fixed-width numeric record -> one token, with a learned unknown row.

    Two layers, in_dim -> hidden -> silu -> width, because a single linear over
    a record is a map of that record's own rank into a token. The prefix's
    geometry columns arrive at in_dim 1, where that is sharpest: one linear
    would put every value a column can take on a single line through the token
    space, and the silu between the two layers is what bends it. Non-finite
    inputs are cleaned
    before the projection and not after: torch.where evaluates both branches,
    so one NaN in a masked-out row would otherwise reach the weight gradients
    of the whole batch.
    """

    def __init__(self, in_dim: int, width: int, *, init_std: float = 0.02):
        super().__init__()
        self.in_dim = int(in_dim)
        self.width = int(width)
        self.in_proj = nn.Linear(self.in_dim, self.width)
        self.out_proj = nn.Linear(self.width, self.width)
        # Zero at the output, so a run that adds this channel and a run without
        # it are the same run at step 0.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # Not zero: "this was not measured" is a statement the model reads, and
        # it has to be a different token from the zero a measured record
        # projects to before the projection has learned anything.
        self.unknown = nn.Parameter(torch.randn(self.width) * init_std)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x [..., in_dim], valid [...] bool -> [..., width].

        valid is required. A caller with nothing to say passes an all-False
        mask and gets the unknown row, which is a statement; a caller that
        could omit the mask would project whatever numbers were in the tensor.
        """
        if int(x.shape[-1]) != self.in_dim:
            raise ValueError(
                f"record is {tuple(x.shape)}, expected a last axis of "
                f"{self.in_dim}.\nFix the producer of this channel."
            )
        if tuple(valid.shape) != tuple(x.shape[:-1]):
            raise ValueError(
                f"validity is {tuple(valid.shape)}, expected "
                f"{tuple(x.shape[:-1])}.\nOne flag per record."
            )
        # A non-finite number is not a measurement, whatever the producer
        # claims, and neither is a finite one of impossible size. Without this
        # the sanitising below would hand the projection a confident zero,
        # which is the one thing this module must not emit.
        measured = valid.bool() & within_range(x).all(dim=-1)
        clean = brought_inside(x)
        hidden = nn.functional.silu(self.in_proj(clean.to(self.in_proj.weight.dtype)))
        token = self.out_proj(hidden)
        unknown = self.unknown.to(token.dtype).expand_as(token)
        return torch.where(measured[..., None], token, unknown)


class PrefixTokenizers(nn.Module):
    """The prefix's conditioning, at the backbone width.

    The state banks behind the six <state> columns (an arm bank and a body
    bank), the anchor id map, and the two blocks of columns that follow each
    <cam_k> -- six for the calibration, six for the camera rotation
    (intrinsics_mode tokens, the one mode). The camera anchors are real rows of
    the backbone's embedding table rather than parameters here, so this module
    holds the ids and their initialiser.
    """

    def __init__(
        self,
        width: int,
        *,
        intrinsics_mode: str,
        n_slots: int,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.width = int(width)
        self.n_slots = int(n_slots)
        self.intrinsics_mode = str(intrinsics_mode)
        if self.intrinsics_mode in REMOVED_INTRINSICS_MODES:
            raise ValueError(
                f"model.prefix.intrinsics_mode is {self.intrinsics_mode!r}, a mode this "
                "tree no longer builds.\n"
                f"Expected {BUILT_INTRINSICS_MODES[0]!r}: {list(REMOVED_INTRINSICS_MODES)} "
                "were removed.\n"
                "Set model.prefix.intrinsics_mode=tokens."
            )
        if self.intrinsics_mode not in BUILT_INTRINSICS_MODES:
            raise ValueError(
                f"model.prefix.intrinsics_mode is {self.intrinsics_mode!r}.\n"
                f"Expected one of {list(BUILT_INTRINSICS_MODES)}, the modes built here.\n"
                "Set it to the mode this run's model was pretrained under."
            )
        if self.n_slots < 1:
            raise ValueError(
                f"n_slots is {self.n_slots}, expected at least one.\n"
                "Every slot is an arm and has a start and a now <state> column.\n"
                "Set model.action.n_slots."
            )
        self._build_state_bank(init_std=init_std)
        self._build_calibration_banks(init_std=init_std)
        self.anchor_ids = dict(ANCHOR_IDS)
        self.frame_s_px = FRAME_ASINH_S_PX
        self.frame_denom = math.asinh(FRAME_ASINH_X_PX / FRAME_ASINH_S_PX)

    def _build_state_bank(self, *, init_std: float) -> None:
        """The arm and body state MLPs with their unknown rows, and the role, arm and body rows.

        One arm bank for both roles and both arms, so a start token and a now
        token of the same arm read the same record the same way and differ only
        by e_role; the body has a bank of its own (no column meaning is shared
        with an arm's) and shares e_role. Each MLP reads [x * m, m]
        (STATE_INPUT_DIM 58 for an arm, BODY_INPUT_DIM 16 for the body), the
        RDT availability vector, so a partial record -- EE without joints, joints
        without EE, a 6-DoF arm's joint 6, a partial hand or body -- is one
        mechanism. out_proj starts at zero (NumericTokenizer), so at step 0 a
        measured record carries its role and its arm or e_body and nothing else.
        Drawn from a fixed stream of its own, for the reason STATE_INIT_SEED
        gives. There is no row for the missing start: the no-history form feeds
        each MLP the zero record with an all-ones mask, a statement the same
        weights make.
        """
        stream = torch.Generator()
        stream.manual_seed(STATE_INIT_SEED)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(stream.get_state())
            self.state = NumericTokenizer(STATE_INPUT_DIM, self.width, init_std=init_std)
            self.body_state = NumericTokenizer(BODY_INPUT_DIM, self.width, init_std=init_std)
            self.state_role = nn.Embedding(N_ROLES, self.width)
            self.state_arm = nn.Embedding(self.n_slots, self.width)
            # e_body: one row beside the arm rows, so a body token is told from
            # every arm's by its own learned identity as well as by its bank.
            self.state_body = nn.Embedding(1, self.width)
            nn.init.normal_(self.state_role.weight, std=init_std)
            nn.init.normal_(self.state_arm.weight, std=init_std)
            nn.init.normal_(self.state_body.weight, std=init_std)

    def state_tokens(self, inputs: StateInputs) -> torch.Tensor:
        """StateInputs -> [B, 2, S + 1, width]: per role, the arms' tokens then the body's.

        An arm token is MLP_arm([x * m, m]) + e_role[r] + e_arm[s] where the
        code is MEASURED or NO_HISTORY, and the arm bank's learned unknown row
        + e_role[r] + e_arm[s] where it is UNKNOWN or ABSENT. The body token is
        the same over the body bank, with e_body in place of e_arm. The
        no-history record is the zero record with an all-ones mask, written here
        whatever the caller put in those rows, so the missing start is
        MLP([0, 1]) + e_role[0] + e_arm[s] by construction; every real record
        has at least one invalid dimension (an arm's zero 6D is not a rotation,
        and a MEASURED body with all eight dimensions valid is refused), so the
        all-ones mask is NO_HISTORY's alone.

        Both MLPs run on every entry and torch.where picks, so every parameter
        of both banks is in the graph of every batch, whichever codes it holds:
        DDP with find_unused_parameters False needs exactly that. An ABSENT
        column is masked by the encoder; its embedding is the unknown path and
        is never attended.
        """
        inputs.validate(self.n_slots)
        known = torch.tensor(STATE_CODE_PRECEDENCE)
        for name, code in (("state_code", host_of(inputs.state_code)),
                           ("body_state_code", host_of(inputs.body_state_code))):
            if not bool(torch.isin(code, known).all()):
                raise ValueError(
                    f"{name} holds {sorted(set(code.unique().tolist()))}, expected "
                    f"a subset of {sorted(STATE_CODE_PRECEDENCE)}.\n"
                    "Write the codes of neo_vla.constants."
                )
        check_measured_record("state", inputs.state, inputs.state_dims, inputs.state_code)
        check_measured_record(
            "body state", inputs.body_state, inputs.body_state_dims, inputs.body_state_code
        )
        complete = ((host_of(inputs.body_state_code) == STATE_MEASURED)
                    & host_of(inputs.body_state_dims).all(-1))
        if bool(complete.any()):
            raise ValueError(
                "a MEASURED body record has all eight dimensions valid.\n"
                "Expected at least one invalid dimension: with every dimension valid, a "
                "still base and mid-range posture, the body input is [0, 1], which is "
                "NO_HISTORY's (review v7spec, consistency).\n"
                "No source records all eight; mask the dimension the source does not read."
            )
        features, projects = masked_record(inputs.state, inputs.state_dims, inputs.state_code)
        base = self.state(features, projects)
        roles = self.state_role.weight[None, :, None, :].to(base.dtype)
        arms = self.state_arm.weight[None, None, :, :].to(base.dtype)
        body_features, body_projects = masked_record(
            inputs.body_state, inputs.body_state_dims, inputs.body_state_code
        )
        body = self.body_state(body_features, body_projects)
        body = body + self.state_role.weight[None, :, :].to(body.dtype) + self.state_body.weight[
            None
        ].to(body.dtype)
        return torch.cat([base + roles + arms, body[:, :, None, :].to(base.dtype)], dim=2)

    def _build_calibration_banks(self, *, init_std: float) -> None:
        """The calibration and camera-rotation banks, drawn from a stream of their own.

        The stream is forked and restored here, so this draws nothing from the
        global one, and the expert, suffix tokenizer and flow head built after
        it do not depend on these banks' size. Its seed is fixed, for the reason
        init_anchor_rows states. The draw order is the one the tokens mode has
        always had: the calibration bank, then the rotation bank.
        """
        stream = torch.Generator()
        stream.manual_seed(CALIBRATION_INIT_SEED)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(stream.get_state())
            # One projection per column, each reading that column alone, so a
            # token here states one number rather than a learned mixture of six
            # and a view nobody calibrated says so in six distinct unknown rows
            # rather than one row repeated six times.
            self.intrinsics = nn.ModuleList(
                NumericTokenizer(1, self.width, init_std=init_std)
                for _ in range(INTRINSICS_TOKENS_PER_VIEW)
            )
            # A separate bank rather than six more columns of the one above,
            # because the two records are separately measured: a directory can
            # know a camera's calibration and not its pose, and the unknown rows
            # have to be able to say so one record at a time.
            self.camera_extrinsics = nn.ModuleList(
                NumericTokenizer(1, self.width, init_std=init_std)
                for _ in range(EXTRINSICS_TOKENS_PER_VIEW)
            )

    def compress_calibration(self, intrinsics: torch.Tensor) -> torch.Tensor:
        """[..., 6] of pixels -> [..., 6] of about unit size, column by column.

        The record cannot go in as measured: fx, cx and the frame width are
        pixel counts, three orders of magnitude above every other number this
        module reads, and the first layer saturates on them. A focal length and
        a principal point are meaningful against the frame they were measured
        on, so each is divided by its own side, which is the optics rather than
        the raster and is what lets an unseen camera land between its
        neighbours. The two sides go through an asinh curve in pixels rather
        than being dropped: they are what the four ratios were taken against,
        and keeping them is what stops two cameras sharing one vector. Monotone
        in every column, and a row that is not finite stays that way, so the
        projection still reads it as unmeasured.

        A frame side of zero or less is made unmeasured rather than raised on:
        the record then says unknown in all six columns, and a calibration is
        refused where it is read, by tools/validate_intrinsics.py, not on the
        way into a projection.
        """
        numbers = intrinsics.float()
        sides = numbers[..., 4:]
        # nan rather than a clamp: a clamped side would divide cleanly and
        # reach the projection looking measured, which is the one thing this
        # channel must never do.
        sides = torch.where(sides > 0.0, sides, torch.full_like(sides, float("nan")))
        width, height = sides[..., 0], sides[..., 1]
        against = torch.stack([width, height, width, height], dim=-1)
        return torch.cat(
            [
                numbers[..., :4] / against,
                torch.asinh(sides / self.frame_s_px) / self.frame_denom,
            ],
            dim=-1,
        )

    def calibration_tokens(
        self, intrinsics: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """[B, V, 6] and [B, V] bool -> [B, V, INTRINSICS_TOKENS_PER_VIEW, width].

        The six numbers arrive at the resolution they were measured against,
        and compress_calibration is what makes them a record this bank can
        read. Column k reaches bank k and nothing else: a token here states one
        number, so the raster columns stay legible as the raster rather than
        dissolving into a mixture with the optics, and an uncalibrated view
        spends the same six positions on six unknown rows.

        The validity is the RECORD's and is computed once, here, rather than
        left to each bank to take over the one column it was handed. A bank
        given a column alone tests that column alone, so a view with an
        unreadable frame height would have had its focal length and its
        principal point projected as measurements while only the two raster
        tokens said unknown -- a reading assembled out of a defective record.
        """
        _check_view_record("intrinsics", intrinsics, valid, INTRINSICS_DIM)
        record = self.compress_calibration(intrinsics)
        measured = valid.bool() & within_range(record).all(dim=-1)
        return torch.stack(
            [
                bank(record[..., column : column + 1], measured)
                for column, bank in enumerate(self.intrinsics)
            ],
            dim=-2,
        )

    def camera_extrinsics_tokens(
        self, extrinsics: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """[B, V, 6] and [B, V] bool -> [B, V, EXTRINSICS_TOKENS_PER_VIEW, width].

        The camera's own rotation, one column per number, standing beside the
        calibration of the same picture. No curve on the way in and none needed:
        a 6-D rotation is two unit vectors, so every column is already inside
        [-1, 1] and the saturation compress_calibration exists to prevent cannot
        arise here. RECORD_LIMIT still guards it, because a producer defect is
        not bounded by what the quantity means.

        The validity is the record's, for the reason calibration_tokens states:
        five columns of a rotation and one that is not a number is not five
        sixths of a pose, it is not a pose, and projecting the five would state
        an orientation nothing measured.
        """
        _check_view_record("camera_extrinsics", extrinsics, valid, EXTRINSICS_DIM)
        measured = valid.bool() & within_range(extrinsics).all(dim=-1)
        return torch.stack(
            [
                bank(extrinsics[..., column : column + 1], measured)
                for column, bank in enumerate(self.camera_extrinsics)
            ],
            dim=-2,
        )

    def init_anchor_rows(
        self, embed_weight: torch.Tensor, *, std: float = 0.02, seed: int = 0
    ) -> int:
        """Re-initialise the reserved rows in ANCHOR_IDS. Returns the count.

        The checkpoint's reserved rows are near-identical to one another, so
        distinct anchors would otherwise start as the same vector and be
        separable only by position. Drawn from a fixed generator rather than
        the global one, because this runs on every rank before any broadcast,
        and one stream per row rather than one draw across them, so that adding
        an anchor leaves every row that already existed exactly where it was.
        """
        rows = sorted(set(self.anchor_ids.values()))
        reserved_end = ANCHOR_RESERVED_BASE + ANCHOR_RESERVED_COUNT
        outside = [row for row in rows if not ANCHOR_RESERVED_BASE <= row < reserved_end]
        if outside:
            raise ValueError(
                f"anchor ids {outside} are outside the reserved rows "
                f"[{ANCHOR_RESERVED_BASE}, {reserved_end}).\n"
                "Move them inside; overwriting a used row breaks the language head."
            )
        if embed_weight.dim() != 2 or int(embed_weight.shape[1]) != self.width:
            raise ValueError(
                f"table is {tuple(embed_weight.shape)}, expected "
                f"[vocab, {self.width}].\nPass the backbone's embed_tokens weight."
            )
        if rows[-1] >= int(embed_weight.shape[0]):
            raise IndexError(
                f"anchor id {rows[-1]} is past the table's "
                f"{int(embed_weight.shape[0])} rows.\n"
                "This is not the table the v7 scaffold emits ids into."
            )
        generator = torch.Generator()
        drawn = []
        for row in rows:
            generator.manual_seed(int(seed) * ANCHOR_SEED_STRIDE + row)
            drawn.append(torch.randn(self.width, generator=generator) * float(std))
        stacked = torch.stack(drawn)
        with torch.no_grad():
            embed_weight[rows] = stacked.to(dtype=embed_weight.dtype, device=embed_weight.device)
        return len(rows)

    def frozen_after_pretrain_names(self, prefix: str = "") -> list[str]:
        """Every parameter here, qualified under prefix, for the freeze pass.

        The state banks first (the arm MLP, the body MLP, then the role, arm
        and body rows), then the calibration and camera-rotation banks, in the
        order they are registered. The <cam_k> and <state> anchors are rows of
        the backbone's embedding table, not parameters of this module, so they
        are not named here: they follow the table they live in.
        """
        banks = [
            ("state", self.state),
            ("body_state", self.body_state),
            ("state_role", self.state_role),
            ("state_arm", self.state_arm),
            ("state_body", self.state_body),
            ("intrinsics", self.intrinsics),
            ("camera_extrinsics", self.camera_extrinsics),
        ]
        return _qualified_names(prefix, banks)


class SuffixTokenizers(nn.Module):
    """Every suffix block's conditioning row, at the expert width.

    An arm's row is FRAME + IDENT: the frame_ref table's row for the frame the
    arm's actions are written in, in its own sample (row k for current camera
    slot k, the robot row, <robot_mode>, for the base frame), plus the arm's
    identity row <arm_i>. The body's block, always last, takes the robot row
    (the body is in the base frame, always) plus its own identity row <body>,
    beside the <arm_i> rows. The decoder adds a block's row to every action
    token of that block, after the time MLP, as it adds the step embedding.

    Until neo-official 3d53a4b (suffix layout neo_vla_suffix_v1) the same rows
    were two pre tokens opening each block, and a block-local attention mask
    was the only way they reached their own action rows. Added to the token
    input they reach exactly their own block's rows with a regular mask, and
    the tables, their names and their shapes are unchanged.

    Three tokens left the arm block over time. [TM] restated [state] on every
    source that resets to a home pose; [extrinsics] held the arm's base
    rotation, which <frame_ref> already names; and [state] moved into the
    prefix (template v6), where the instruction and every view sit beside it
    and a start-of-chunk state can stand next to the current one.

    None of the rows here is a vocabulary id: the suffix never passes through
    the backbone's embedding table, and the expert is narrower than it.
    """

    def __init__(self, width: int, *, n_slots: int, init_std: float = 0.02):
        super().__init__()
        self.width = int(width)
        self.n_slots = int(n_slots)
        if self.n_slots < 1:
            raise ValueError(
                f"n_slots is {self.n_slots}, expected at least one.\n"
                "Every slot is an arm; set model.action.n_slots to the number the "
                "layout carries."
            )
        self.frame_ref = nn.Embedding(FRAME_REF_KINDS, self.width)
        # The one tensor n_slots shapes. Arm identity is positional -- slot s
        # takes row s -- so changing n_slots does not resize this table, it
        # renumbers the arms, and a checkpoint written under another count is a
        # different model rather than one that can be reshaped into this.
        self.arm = nn.Embedding(self.n_slots, self.width)
        # <body>: the body's block has its own identity row, never an arm's.
        self.body = nn.Embedding(1, self.width)
        # None of these is zero-initialised. Every block sees the same rotary
        # positions, so <arm_i> and <body>, added to every action token of
        # their block, are what tell one block from another, and a table of
        # zeros would make them interchangeable.
        nn.init.normal_(self.frame_ref.weight, std=init_std)
        nn.init.normal_(self.arm.weight, std=init_std)
        nn.init.normal_(self.body.weight, std=init_std)

    def banks(self) -> list[tuple[str, nn.Module]]:
        """The submodules the blocks are made of, in block order.

        Written out rather than read off named_children, so a parameter added
        here has to be given a place in the block and in the freeze list.
        """
        return [("frame_ref", self.frame_ref), ("arm", self.arm), ("body", self.body)]

    def frozen_after_pretrain_names(self, prefix: str = "") -> list[str]:
        """Every parameter here, qualified under prefix, for the freeze pass.

        At the expert width the anchor rows are this module's own embeddings
        rather than rows of a shared vocabulary, so they are named here.
        """
        return _qualified_names(prefix, self.banks())

    def forward(self, inputs: SuffixInputs) -> torch.Tensor:
        """SuffixInputs -> [B, S + 1, width]: each block's conditioning row, in block order.

        Row s < S is frame_ref[inputs.frame_ref[b, s]] + arm[s], gathered per
        sample and per arm, so two arms of one sample may name different
        cameras and two samples of one batch different frames. Row S is the
        body's, frame_ref[BODY_FRAME_REF] + body[0], whether or not the
        sample's key has a body (an absent body's tokens are masked as keys by
        the decoder, never dropped, so the layout is one length). Slot s always
        takes row s of the arm table, so an arm keeps its identity across
        sources that carry different numbers of arms. That identity is the row
        index and nothing else, which is why dropping a slot from the layout
        renumbers every arm after it rather than leaving them where they were.
        """
        batch_slots = inputs.n_slots()
        batch_size = int(inputs.slot_valid.shape[0])
        inputs.validate(batch_size, batch_slots)
        if batch_slots > self.n_slots:
            raise ValueError(
                f"batch carries {batch_slots} slots, this bank holds "
                f"{self.n_slots} arm rows.\n"
                "Raise model.action.n_slots: slot s takes row s."
            )
        frame_ref = inputs.frame_ref
        host_frame_ref = host_of(frame_ref)
        in_range = (host_frame_ref >= 0) & (host_frame_ref < FRAME_REF_KINDS)
        if not bool(in_range.all()):
            raise ValueError(
                f"frame_ref holds {int(host_frame_ref[~in_range][0])}, expected a "
                f"camera slot below {MAX_CAMERAS} or {FRAME_REF_ROBOT} for robot.\n"
                "There is no fallback row; the row names the block's frame."
            )
        arm_rows = torch.arange(batch_slots, device=self.arm.weight.device)
        arms = self.frame_ref(frame_ref) + self.arm(arm_rows)[None]
        body = self.frame_ref.weight[BODY_FRAME_REF] + self.body.weight[0]
        return torch.cat([arms, body[None, None].expand(batch_size, 1, -1).to(arms.dtype)],
                         dim=1)
