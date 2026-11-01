"""The dataset: read the manifest, draw a window, emit one sample.

The manifest is the interface. It is the only file this module opens to learn
what exists; every other path it reads is one the manifest named, plus the
embodiments file the data config names. Nothing here scans a directory, infers
a dataset from a filename or falls back to a default when a field is missing,
so what a run trained on is a file that can be read afterwards.

A sample is a dict of fixed-shape tensors and two strings: the current views,
the earlier views of the history step, their geometry, the arm and body state
records with their codes and valid dims, the normalised arm and body targets
with their masks, and the training-only state noise. Its key set is
constants.batch_keys.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate

from ..constants import (
    ACTION_DIM,
    ACTION_HAND,
    ACTION_JOINTS,
    ACTION_RECORD,
    ARM_JOINTS,
    BODY_DIM,
    COLUMN_BODY,
    COLUMN_BODY_COMMAND,
    COLUMN_HAND,
    COLUMN_JOINTS,
    COLUMN_JOINTS_COMMAND,
    GEOMETRY_KEYS,
    HAND_JOINTS,
    MANIFEST_SCHEMA_VERSION,
    N_ARM_SLOTS,
    OLD_ACTION_DIMS,
    OLD_MANIFEST_SCHEMA_VERSIONS,
    PREFIX_LAYOUT,
    STATE_DIM,
    STATE_HAND,
    STATE_JOINTS,
    batch_keys,
    refusal_prefix,
)
from ..model.tokenizers import within_range
from . import conditioning, embodiment, history, images, rays
from .actions import (
    ARM_FIELDS,
    GRIPPER,
    RETIRED_POLICIES,
    ROTATION_CHANNEL_POLICY,
    ROTATION_TRAINED,
    ActionLayout,
    block_columns,
    blocks_of,
    body_to_spatial,
    check_gripper_columns,
    euler_to_spatial,
    pack_slots,
    part_targets,
    rotate_chunk,
    structural_dims,
    unify,
)
from .body import BodyLayout, body_targets, check_body_masks, pack_body, read_body
from .chat_pack import DRAW_INDEX_KEY, StepItem, group_key, split_step_item, step_key
from .contact import ContactConfig, ContactFocus
from .embodiment import Embodiments
from .history import HistoryDraw
from .mixture import (
    SAMPLINGS,
    STREAM_SPAN,
    Draw,
    MixturePlan,
    MixtureRule,
    RowPlan,
    TrajectoryWeights,
    build_plan,
    build_row_plan,
    draw_rows,
    family_table,
    pass_table,
)
from .mixture import draw as draw_window
from .prompt.stamp import Stamp
from .rays import Intrinsics
from .reader import SIDECAR_PINS, CacheReader, StoredParts
from .reference import (
    PER_ARM,
    SHARED,
    Reference,
    ReferenceCamera,
    draw_reference,
    present_pool,
    strategies_for,
)
from .registry import ArmSpec, ViewSpec, reference_views, source_id
from .state import (
    EULER_CONVENTIONS,
    PART_FIELDS,
    TARGET_COMMAND_PREFIX,
    ArmStateLayout,
    GripperSpec,
    PartLayout,
    arm_poses,
    check_layout,
    euler_matrix,
    split_arms,
)
from .understanding import CHAT_KEY, collate_chat

logger = logging.getLogger(__name__)

# The manifest layout this loader understands. A manifest written under any
# other version is refused, an old one by name: a field that moved would
# otherwise be read as the field that took its place, and both are plain
# numbers. Version 3 adds, per arm, the joints and hand blocks; per directory
# the body block, the stored v7 columns and the sidecar they came from.
SCHEMA_VERSION = MANIFEST_SCHEMA_VERSION

# The three numeric columns an EpisodeReader returns for a window. Naming them
# here rather than at each call is what lets the reader and the assembler be
# written against one another without either one reading the manifest twice.
ROW_STATE = "state"
ROW_ACTION = "action"
ROW_VALID = "valid"

# The two modes a dataset runs in. train emits normalised samples through the
# embodiments file and refuses to run without one; targets_only (the stats
# build) returns physical targets and refuses to emit a sample.
MODES = ("train", "targets_only")

# The slot count the history uniforms are laid out for: two roles by two
# slots by the noise columns after the six fixed positions (constants
# EXTRA_NOISE_FIRST), then the body's.
HISTORY_SLOTS = N_ARM_SLOTS

# The stored v7 column of each part a manifest's columns block names, and how
# many values one arm block of it holds (the body has one 8-wide block). A
# command column holds the same per-arm blocks as the record it commands.
COLUMN_FIELDS = {"joints": (COLUMN_JOINTS, ARM_JOINTS), "hand": (COLUMN_HAND, HAND_JOINTS),
                 "joints_command": (COLUMN_JOINTS_COMMAND, ARM_JOINTS),
                 "body": (COLUMN_BODY, BODY_DIM), "body_command": (COLUMN_BODY_COMMAND, BODY_DIM)}
ARM_COLUMNS = ("joints", "hand", "joints_command")

# What a parts-only arm's state layout says established it: the arm's pose and
# gripper never were, and its joints and hand come from the stamped columns.
PARTS_ONLY = "joints and hand only: the stored v7 columns under their writer's stamp"

# The row width a layout is checked against at load, before any row is read:
# wider than any stored row, so only the layout's own rules can refuse it.
UNREAD_WIDTH = 1 << 30

# Windows per key the trained stamp's frames are realised over. A spelling a
# key takes on 0.6% of its arm-draws (AgiBot's robot fallback, DESIGN 3.1) is
# missed with probability exp(-12) at this count, and a miss only refuses
# (R12) a frame the key barely trained.
FRAME_DRAWS = 2000

# Fields a version-1 manifest carried that version 2 dropped, each with what it
# became, so an old manifest is refused by name.
RETIRED_ACTION_FIELDS = {
    "normaliser": "the embodiments file named by data.train.normaliser",
    "rotation_convention": "action.rotation_delta_frame",
}

# What the manifest may say about a directory's camera poses. A mounted camera
# is one bolted to an arm's end effector: its pose in the base frame is not a
# per-episode constant, so what is stored is the constant offset from the arm
# and the pose is composed here.
MOUNTED = "mounted"
EXTRINSICS_KINDS = ("episode", "constant", MOUNTED, "none")

# The two ways a manifest keys a directory's embodiment.
EMBODIMENT_KINDS = ("constant", "episode")

# The fields of one v6 prompt-cache record, each checked when a directory's
# prompts are first read.
PROMPT_FIELDS = ("position", "episode", "frame_ref", "embodiment", "control_hz", "tokens",
                 "head", "task", "tail", "text")

# The fields of one label-pool line (a data group's label_pools): line i is kept
# position i, its source episode as the prompt cache states it, the
# episode's phrasings in the order the passes take them and each one's text tokens rendered in
# that position's head and tail (as tools/build_cache.py counts a prompt). The offsets file beside
# it holds the byte offset of every line and the file's size, so a draw reads one line.
LABEL_POOL_FIELDS = ("position", "episode", "labels", "tokens")
LABEL_POOL_KEYS = ("dir", "sources")


@dataclass(frozen=True)
class Provenance:
    """Who wrote a stage output, from what, and when.

    inputs maps each input file to its hash, so a manifest that was rebuilt
    after its stats changed can be told from one that was not. Every
    preparation stage output carries one of these.
    """

    schema_version: int
    tool: str
    tool_version: str
    git_commit: str
    inputs: dict[str, str]
    created: str


@dataclass(frozen=True)
class IntrinsicsEntry:
    """Where one camera's calibration comes from, or that it has none.

    kind is one of rays.INTRINSICS_KINDS. A constant view carries the six
    numbers here, because every kept episode of it was measured the same way;
    an episode view carries a path to an array of them, read by kept position
    the way the extrinsics beside it are. A view with no calibration says so
    rather than being left out of the entry.
    """

    kind: str
    provenance: str | None
    constant: Intrinsics | None
    path: str | None


@dataclass(frozen=True)
class ExtrinsicsEntry:
    """Where one camera's pose comes from, or that it has none.

    kind is one of EXTRINSICS_KINDS. An episode camera carries a path to one
    3x3 per kept episode, a constant camera one 3x3 for the whole directory, a
    mounted camera one 3x3 per kept episode that is R_end_effector<-camera
    rather than R_base<-camera, and a camera with no measured pose says kind
    none rather than being left out: only a camera with a pose can be drawn as
    a reference, so the absence is what keeps it out of the draw.
    """

    kind: str
    path: str | None


@dataclass(frozen=True)
class Rotations:
    """One camera's stored rotations, and what the matrices in them are.

    The kind travels with the array because it is what says which frame the
    stored matrix relates: a pose in the base frame and an offset from an arm
    are both proper rotations of the same shape, and reading one as the other
    is a plausible camera in the wrong place.
    """

    kind: str
    values: np.ndarray

    def at(self, episode: int) -> torch.Tensor:
        """The matrix this camera holds for one episode, as the array stores it."""
        index = 0 if self.kind == "constant" else int(episode)
        return torch.from_numpy(
            np.ascontiguousarray(self.values[index], dtype=np.float64)
        )


@dataclass(frozen=True)
class ViewEntry:
    """One camera of one directory, as the manifest records it.

    spec        name, role, camera slot and the arm it is mounted on
    shape       the frames' size on disk, before the loader resizes anything
    grid        the token grid that shape snaps to under this run's canvas,
                recorded by the build so a change in the snapping rule is a
                mismatch rather than a quiet change of token spend
    intrinsics  where the calibration comes from, at its own resolution
    extrinsics  where this camera's pose comes from, or that it has none
    """

    spec: ViewSpec
    shape: tuple[int, int]
    grid: tuple[int, int]
    intrinsics: IntrinsicsEntry
    extrinsics: ExtrinsicsEntry


@dataclass(frozen=True)
class ArmEntry:
    """One arm of one directory: which slot it fills and where its pose lives.

    The two halves travel together because they are read from one object in the
    manifest. base_offset is the path of a [episodes, 3] array of metres the
    arm's stored position is off the robot base by (LIBERO's scene offset), or
    None where the stored position is already in the base frame.
    """

    spec: ArmSpec
    state: ArmStateLayout | None
    base_offset: str | None = None

    @property
    def slot(self) -> int:
        """The action slot this arm fills, arms[k].slot."""
        return int(self.spec.slot)


@dataclass(frozen=True)
class EmbodimentSpec:
    """Which normaliser key a directory's episodes take.

    constant names one key for every episode; episode names keys per kept
    episode through two arrays in kept order: path holds int8 indexes into
    keys, ids_path the source episode id each position was joined from (R16).
    """

    kind: str
    keys: tuple[str, ...]
    path: str | None
    ids_path: str | None


@dataclass(frozen=True)
class ActionGripper:
    """The action gripper column's polarity: raw_open and raw_closed, and how it was set."""

    raw_open: float
    raw_closed: float
    method: str


@dataclass(frozen=True)
class RotationSource:
    """Where a converted rotation reads the arm's orientation, when not its state layout.

    column is the state column (it must be the entry's own), at the first of
    three Euler angles in radians, euler the reading that turns them into R.
    """

    column: str
    at: int
    euler: str


@dataclass(frozen=True)
class DatasetEntry:
    """One directory of the corpus: everything the loader needs to read it.

    Paths are relative to the corpus root, so a manifest survives the corpus
    being moved. valid_column is the stored column saying which rows the
    source recorded an action for, or None for a directory that records no such
    column. The manifest's scale_cm_per_step is the tools' speed QC and is not
    read here: nothing trains on it since the scale token was retired.
    client_record is the six client-input fields the embodiments file's
    datasets record is digested over, recomputed from this entry. body is
    the key's body layout (None for a fixed base); stored names the v7
    columns the layouts read and their writer (None when they read none).
    """

    name: str
    source: str
    source_id: int
    path: str
    episodes: int
    rows: int
    views: tuple[ViewEntry, ...]
    arms: tuple[ArmEntry, ...]
    action: ActionLayout
    state_column: str
    action_column: str
    valid_column: str | None
    common_base_frame: bool
    frame_ref: str
    weight: float
    episode_starts_path: str
    trajectory_weights_path: str | None
    arrow_path: str
    frames_path: str
    embodiment: EmbodimentSpec
    control_hz: float | None
    control_hz_method: str
    control_hz_evidence: str
    gripper: ActionGripper
    rotation_delta_frame: str
    rotation_source: RotationSource | None
    history_blocked: tuple[int, ...]
    client_record: dict
    body: BodyLayout | None = None
    stored: StoredParts | None = None
    pin_keeps_other_poses: bool = False

    def reference_cameras(self) -> tuple[ReferenceCamera, ...]:
        """The cameras a sample of this directory may be written against.

        A camera enters when its role admits it and the manifest gives it a
        pose. Both halves are needed: a role decides whether the frame means
        anything for actions, and a pose is what the extrinsics token and the
        chunk rotation are built from.
        """
        eligible = {view.key for view in reference_views(
            tuple(view.spec for view in self.views)
        )}
        _, pinned = conditioning.parse_frame_ref(self.frame_ref)
        return tuple(
            ReferenceCamera(view.spec.key, view.spec.slot, view.spec.arm_slot)
            for view in self.views
            if view.spec.key in eligible and view.extrinsics.kind != "none"
            and (not self.pin_keeps_other_poses or pinned is None or view.spec.slot == pinned)
        )


@dataclass(frozen=True)
class Targets:
    """One draw's physical targets, before any normalisation.

    chunk        [arms, K, ARM_DIM] float64, the v7 row: joint targets (rad,
                 absolute), dpos in cm per step and the spatial rotation
                 vector in rad per step (both in the drawn frame), the
                 canonical gripper g_c in [0, 1], hand targets; 0 where not read
    steps_valid  [arms, K] bool: the step exists (row t+j holds an action and
                 its end row t+j+1 is inside the episode); one row, every arm
    read         [arms, K, ARM_DIM] bool: the value at (step, column) was read:
                 the EE group where its flat columns are numbers and every
                 pose its conversion reads was read, each joint and hand
                 column where its stored reading was
    dims_valid   [arms, ARM_DIM] bool: the columns the layout, the rotation
                 policy and the gripper method leave a target in (before the
                 directory's masks and the key's decodable columns)
    slots        each arm's action slot; key the draw's embodiment key;
                 frame_ref the [n_slots] anchor rows the chunk is written in;
                 directory the manifest directory the draw came from
    body         [K, BODY_DIM] float64: the body's targets in the base frame
                 (0 where not read; all 0 for a key with no body)
    body_read    [K, BODY_DIM] bool
    body_dims    [BODY_DIM] bool: the dims the key's body can train
    """

    chunk: torch.Tensor
    steps_valid: torch.Tensor
    dims_valid: torch.Tensor
    slots: tuple[int, ...]
    key: str
    frame_ref: torch.Tensor
    directory: str
    read: torch.Tensor
    body: torch.Tensor
    body_read: torch.Tensor
    body_dims: torch.Tensor


@dataclass(frozen=True)
class Manifest:
    """A whole corpus, as one file states it.

    root is the directory every relative path in the entries hangs off, and it
    comes from the run's environment rather than from the file: the same
    manifest is read on a node that mounts the corpus somewhere else.

    short_side, token and cell are the canvas the recorded grids were snapped
    against. A run whose model asks for a different canvas gets a raise, not a
    resnap, because the grid is what each source costs in tokens and the
    mixture was weighted knowing it.
    """

    schema_version: int
    corpus: str
    root: Path
    short_side: int
    token: int
    cell: tuple[int, int]
    entries: tuple[DatasetEntry, ...]
    prompt_template: str
    prompt_cache: str
    prompt_stamp: Stamp
    provenance: Provenance
    sha256: str = ""

    def index_of(self, name: str) -> int:
        """The position of a directory in entries. Raises for an unknown name."""
        for index, entry in enumerate(self.entries):
            if entry.name == name:
                return index
        raise KeyError(
            f"the manifest has no directory named {name!r}.\n"
            f"Expected one of its {len(self.entries)} entry names.\n"
            "Use the name as the manifest spells it, or rebuild the manifest."
        )

    def totals(self) -> dict[str, int]:
        """Directories, episodes and rows, for the one line the build logs."""
        return {
            "directories": len(self.entries),
            "episodes": sum(entry.episodes for entry in self.entries),
            "rows": sum(entry.rows for entry in self.entries),
        }


class EpisodeReader(Protocol):
    """What the loader needs from whatever reads a directory off disk.

    reader.CacheReader is the implementation; this states the surface, so a
    corpus staged some other way can be read by supplying it rather than by
    branching here. rows returns the numeric columns of a window and frames
    returns decoded images, both indexed by kept position inside one episode.
    """

    def rows(self, episode: int, start: int, length: int) -> dict[str, torch.Tensor]:
        """The numeric columns of one window, padded at the episode's end."""
        raise NotImplementedError

    def frames(self, episode: int, step: int) -> dict[str, torch.Tensor]:
        """One step's decoded frames, keyed by the manifest's view keys.

        A camera with no frame at that step is left out rather than given a
        neighbouring one, and the caller marks its slot absent.
        """
        raise NotImplementedError

    def frames_at(self, episode: int, steps: Sequence[int]) -> list[dict[str, torch.Tensor]]:
        """frames at each of several steps of one episode, in the order given.

        A sample asks for its step t and its history step p in one call, so a
        reader can open each file once for both; the answer per step is what
        frames returns for that step alone.
        """
        raise NotImplementedError

    def present_views(self, episode: int, step: int) -> frozenset[str]:
        """The keys frames would return at that step, without decoding them.

        The reference draw and the tools that audit it read this rather than
        the flag column, so the cameras a sample may be written against and the
        pictures it carries are one statement made in one place.
        """
        raise NotImplementedError

    def episode_ids(self) -> np.ndarray:
        """[episodes] int64: the source episode each kept position holds (R16's witness)."""
        raise NotImplementedError


def load_manifest(path: str | Path, *, root: str | Path,
                  pin_keeps_other_poses: bool = False) -> Manifest:
    """Read a manifest and check it against this loader.

    Raises when schema_version is not SCHEMA_VERSION, naming both, because a
    manifest from another version is not partially readable: its fields are
    numbers and a moved one is read as its neighbour. Also raises for a
    duplicate directory name, since the second entry would take the first
    entry's weight and be invisible in the mixture table, and for an entry with
    no weight, which is what stops an unlisted corpus entering at a default.
    """
    manifest_path = Path(path)
    data = manifest_path.read_bytes()
    document = json.loads(data.decode("utf-8"))
    where = str(manifest_path)
    version = _require(document, "schema_version", where)
    if int(version) != SCHEMA_VERSION:
        old = (f", a retired manifest schema ({list(OLD_MANIFEST_SCHEMA_VERSIONS)}): its "
               "rows are the v6 layout, with no joints, hand or body"
               if int(version) in OLD_MANIFEST_SCHEMA_VERSIONS else "")
        raise ValueError(
            f"{where} states schema_version {version}{old}.\n"
            f"Expected {SCHEMA_VERSION}; another version is not partially readable.\n"
            "Rebuild the manifest with this tree's manifest stage."
        )
    canvas = _require(document, "canvas", where)
    short_side = _positive(canvas, "short_side", where)
    token = _positive(canvas, "token", where)
    cell = _read_cell(canvas, token, where)
    prompt = _require(document, "prompt", where)

    raw_entries = _require(document, "datasets", where)
    if not raw_entries:
        raise ValueError(
            f"{where} lists no datasets.\n"
            "Expected at least one directory.\n"
            "Run the weights and manifest stages over a corpus that is not empty."
        )
    seen: set[str] = set()
    for raw in raw_entries:
        name = str(_require(raw, "name", where))
        if name in seen:
            raise ValueError(
                f"{where} lists the directory {name!r} twice.\n"
                "Expected every entry name to be unique.\n"
                "The second entry would take the first entry's weight; rename or drop it."
            )
        seen.add(name)

    entries = tuple(
        _read_entry(raw, where=where, short_side=short_side, token=token, cell=cell,
                    pin_keeps_other_poses=pin_keeps_other_poses)
        for raw in raw_entries
    )
    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        corpus=str(_require(document, "corpus", where)),
        root=Path(root),
        short_side=short_side,
        token=token,
        cell=cell,
        entries=entries,
        prompt_template=str(_require(prompt, "template", where)),
        prompt_cache=str(_require(prompt, "cache", where)),
        prompt_stamp=_read_stamp(_require(prompt, "stamp", where), where),
        provenance=_read_provenance(_require(document, "provenance", where), where),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    _check_totals(manifest, _require(document, "totals", where), where)
    from .prompt.embodiments import validate_manifest_prompts

    validate_manifest_prompts(document)
    return manifest


@dataclass(frozen=True)
class _Context:
    """What one draw's step t gives both the targets and the sample.

    now and now_dims are the arm records at t [n_slots, STATE_DIM] and their
    valid dims, body_now and body_now_dims the body's [BODY_DIM], all as read
    (before the directory's masks).
    """

    entry: DatasetEntry
    reader: EpisodeReader
    present: frozenset[str]
    now: torch.Tensor
    now_dims: torch.Tensor
    body_now: torch.Tensor
    body_now_dims: torch.Tensor
    poses: dict[str, torch.Tensor | None]
    rotations: dict[int, torch.Tensor]
    targets: Targets


class RobotDataset(Dataset):
    """The robot corpus as a map from a draw index to one sample.

    An index is a draw identifier, not an address in the corpus: index i is
    hashed with seed into the two-level draw in neo_vla.data.mixture, so
    __getitem__ is deterministic and a resumed run replays the windows it would
    have drawn. len is epoch_samples, a config number. An index at or past it
    is pass i // epoch_samples of window i % epoch_samples: the window repeats
    and every choice made about it after the window (reference camera, history
    step, dropout, noise) is drawn afresh for that pass.

    sampling picks how a window is drawn (neo_vla.data.mixture): "draws", each
    index independently, or "rows", one pass laid out over the rows, every row
    of a directory a window start once per wrap and every directory exactly its
    share. mixture, when the group states one, is the rule the manifest's
    weights must carry (families, exponent, fixed shares); it is checked here and
    logged per family, and None skips the check. contact_sampling, None or a block
    with enabled false by default, moves only the start inside each drawn episode
    towards commanded gripper events (neo_vla.data.contact).

    The model-facing numbers arrive here as arguments and are checked against
    the manifest once, at construction. normaliser names the embodiments file;
    history_dropout and state_noise are the two required training knobs. None
    of them has a default: a group that leaves one out is refused by name.
    """

    def __init__(
        self,
        *,
        manifest: str | Path,
        root: str | Path,
        max_grid: tuple[int, int],
        pixels_per_token: int,
        n_views: int,
        n_slots: int,
        chunk: int,
        action_dim: int,
        gripper_dims: tuple[int, ...],
        tokenizer_id: str,
        max_text_tokens: int,
        intrinsics_mode: str,
        pixel_convention: str,
        epoch_samples: int,
        seed: int,
        understanding: Dataset | None = None,
        normaliser: str | Path | None = None,
        history_dropout: float | None = None,
        state_noise: dict | None = None,
        mode: str = "train",
        sampling: str = "draws",
        mixture: dict | None = None,
        label_pools: dict | None = None,
        pin_keeps_other_poses: bool = False,
        frame_mode: str | None = None,
        frame_set: dict | None = None,
        frame_draw: str = "episode",
        contact_sampling: dict | None = None,
    ):
        if mode not in MODES:
            raise ValueError(
                f"the dataset mode is {mode!r}.\n"
                f"Expected one of {MODES}.\n"
                "Train with 'train'; the stats build reads 'targets_only'."
            )
        self.mode = mode
        self.manifest = load_manifest(manifest, root=root,
                                      pin_keeps_other_poses=pin_keeps_other_poses)
        self.frame_mode, self.frame_draw = frame_mode, frame_draw
        self.frame_set = frame_set
        if frame_mode is not None:
            from .frame_modes import configure
            self.manifest = configure(self.manifest, frame_mode, frame_set, frame_draw,
                                      pin_keeps_other_poses)
        elif frame_set:
            raise ValueError("frame_set requires frame_mode")
        self.root = Path(root)
        self.max_grid = (int(max_grid[0]), int(max_grid[1]))
        self.pixels_per_token = int(pixels_per_token)
        self.n_views = int(n_views)
        self.n_slots = int(n_slots)
        self.chunk = int(chunk)
        self.action_dim = int(action_dim)
        self.gripper_dims = tuple(int(dim) for dim in gripper_dims)
        self.tokenizer_id = str(tokenizer_id)
        self.max_text_tokens = int(max_text_tokens)
        self.intrinsics_mode = str(intrinsics_mode)
        # The model's model.vision.pixel_convention, through the data group:
        # it picks the resize filter, and the runner holds it against the model.
        self.pixel_convention = images.check_convention(
            pixel_convention, "data.train.pixel_convention"
        )
        self.resample = images.resize_filter(self.pixel_convention)
        self.epoch_samples = int(epoch_samples)
        self.seed = int(seed)
        self.understanding = understanding
        if not 1 <= self.epoch_samples <= STREAM_SPAN:
            raise ValueError(
                f"data.train.epoch_samples is {self.epoch_samples}.\n"
                f"Expected at least one draw and at most {STREAM_SPAN}, the draw streams a "
                "seed has; it is how long one pass is, not a limit.\n"
                "Set it to the total draws of the stage, steps times the global batch."
            )
        if self.n_slots != HISTORY_SLOTS:
            raise ValueError(
                f"model.action.n_slots is {self.n_slots}.\n"
                f"Expected {HISTORY_SLOTS}: the history uniforms are laid out for two roles "
                "by two slots by the noise columns, then the body's.\n"
                "Set model.action.n_slots to 2, or change the contract's layout first."
            )
        self.history_dropout, self.state_noise = None, None
        if mode == "train" or history_dropout is not None or state_noise is not None:
            # Required in train mode; in targets_only mode (the stats build and
            # the history-share audit) stated together or not at all.
            self.history_dropout, self.state_noise = history.check_history_config(
                history_dropout, state_noise
            )
        self._embodiments = self._load_normaliser(normaliser)

        self._check_canvas()
        self._check_slots()
        self._check_action_columns()
        self._check_prompt_stamp()
        self._check_intrinsics_mode()

        entries = self.manifest.entries
        self._plan = build_plan(
            tuple(entry.name for entry in entries),
            tuple(entry.source for entry in entries),
            tuple(entry.rows for entry in entries),
            tuple(entry.weight for entry in entries),
        )
        self._episode_starts = [self._read_episode_starts(e) for e in entries]
        self._label_pools = self._check_label_pools(label_pools)
        self._trajectory = [
            self._read_trajectory_weights(index, entry)
            for index, entry in enumerate(entries)
        ]
        self.sampling = str(sampling)
        if self.sampling not in SAMPLINGS:
            raise ValueError(
                f"data.train.sampling is {self.sampling!r}.\n"
                f"Expected one of {SAMPLINGS}.\n"
                "gd7 and the groups built on it (stage 1, stage 2, debug) state rows; "
                "post-train groups draw."
            )
        self._rule = None if mixture is None else MixtureRule.from_config(mixture)
        self._families = None if self._rule is None else family_table(self._plan, self._rule)
        self._rows: RowPlan | None = None
        if self.sampling == "rows":
            self._rows = self._row_plan()
        self._episode_keys = [self._read_episode_keys(entry) for entry in entries]
        self._blocked = [self._read_blocked(entry) for entry in entries]
        self._check_embodiments()
        # Contact-focused starts: None unless the group enables them, and then built from the
        # manifest alone. An episode's gripper events are read on its first draw in a copy.
        self._contact: ContactFocus | None = None
        contact = ContactConfig.from_config(contact_sampling, chunk=self.chunk)
        if contact is not None and contact.enabled:
            if self.mode != "train":
                raise ValueError(
                    f"data.train.contact_sampling is enabled in mode {self.mode!r}.\n"
                    "Expected train: the stats build and the audits measure the uniform draw.\n"
                    "Leave contact_sampling out of a targets_only dataset."
                )
            self._contact = ContactFocus(contact, entries, self._reader)
        # One reader, one prompt file, one rotation table and one offset table
        # per directory, opened on first use so that a dataloader worker pays
        # only for the directories its own draws reach.
        self._readers: dict[int, EpisodeReader] = {}
        self._prompts: dict[int, tuple[dict, ...]] = {}
        self._rotations: dict[int, dict[str, Rotations]] = {}
        self._cameras: dict[int, tuple[ReferenceCamera, ...]] = {}
        self._calibrations: dict[int, dict[str, np.ndarray]] = {}
        self._offsets: dict[int, dict[int, np.ndarray]] = {}
        # Directories whose prompts were held to their rows (R16), per copy.
        self._identified: set[int] = set()
        # Draws whose reference pool was cut by a camera being absent at the
        # step they opened at, per directory, and gripper values the
        # canonical clip moved; both per copy of this dataset.
        self._absences: dict[str, dict[str, int]] = {}
        # Draws whose pool lost a present camera to its pose audit (reference_audits).
        self._audits: dict[str, dict[str, int]] = {}
        self._gripper_clips: dict[str, int] = {}
        # The trained stamp's summary, realised once (trained_summary).
        self._summary: dict | None = None

        totals = self.manifest.totals()
        logger.info(
            "corpus %s: %d directories, %d episodes, %d rows",
            self.manifest.corpus, totals["directories"], totals["episodes"], totals["rows"],
        )
        shares = self._plan.by_source()
        logger.info(
            "declared mixture: %s",
            " ".join(f"{name}={share:.4f}" for name, share in sorted(shares.items())),
        )
        if self._rule is not None:
            logger.info("mixture rule: families n^%s, fixed %s; the manifest carries it",
                        self._rule.exponent, self._rule.fixed or "none")
            for row in self._families:
                logger.info("mixture family %s: rows %d, share %.4f (%s), manifest %.4f",
                            row["family"], row["rows"], row["rule"],
                            "fixed" if row["fixed"] else f"n^{self._rule.exponent}",
                            row["manifest"])
        if self._rows is not None:
            owner = None if self._rule is None else self._rule.owner()
            logger.info("sampling rows: one pass of %d draws over %d rows, every row a window "
                        "start once per wrap", self.epoch_samples, totals["rows"])
            for row in pass_table(self._rows, self._plan, owner):
                logger.info("passes %s: %d draws over %d rows = %.4f passes (whole wraps %d-%d "
                            "a directory)", row["family"], row["draws"], row["rows"],
                            row["passes"], *row["wraps"])
        if self._contact is not None:
            for line in self._contact.describe():
                logger.info("contact sampling %s", line)

    def __len__(self) -> int:
        """epoch_samples: how many draws one pass over the index stream is."""
        return self.epoch_samples

    @property
    def embodiments(self) -> Embodiments | None:
        """The embodiments file this dataset normalises through, or None in targets_only."""
        return self._embodiments

    def plan(self) -> MixturePlan:
        """The first-level mixture table, for the build log and the verify stage."""
        return self._plan

    def reference_absences(self) -> dict[str, dict[str, int]]:
        """Per directory, the draws whose reference pool absence cut down.

        restricted is the draws taken over fewer cameras than the manifest
        poses and robot_fallback the ones that found none of them. Counted per
        copy of this dataset, so a run with dataloader workers holds one of
        these per worker and a caller wanting the run's total adds them up.
        """
        return {name: dict(counts) for name, counts in sorted(self._absences.items())}

    def reference_audits(self) -> dict[str, dict[str, int]]:
        """Per directory, the draws whose pool the pose audit cut down.

        cut is the draws where a camera with a picture at the step had a pose
        that failed its audit, so the draw went on over the others; robot the
        ones of those left with no usable camera, every slot in the robot
        frame; arm_robot the ones where some arm kept a camera and another
        arm, which no usable camera could serve, is in the robot frame. Per
        copy of this dataset, as reference_absences is.
        """
        return {name: dict(counts) for name, counts in sorted(self._audits.items())}

    def gripper_clips(self) -> dict[str, int]:
        """Per directory, the valid gripper values the canonical clip to [0, 1] moved."""
        return dict(sorted(self._gripper_clips.items()))

    def draw(self, index: int) -> Draw:
        """The directory, episode and window an index stands for.

        Public because it is the provenance path: the sample itself carries no
        source id, so a diagnostic that wants to know where a sample came from
        asks here rather than reading a field out of the batch that the model
        would then have to be told to ignore. The window is that of index
        modulo epoch_samples, and the draw carries the pass the index stands
        for, which every later choice about the sample reads.
        """
        if int(index) < 0:
            raise ValueError(
                f"draw index is {int(index)}.\n"
                "Expected an index at or above 0.\n"
                "The sampler numbers draws from 0; pass the index it gave."
            )
        if self._contact is not None:
            # Contact sampling: the same draw with its start moved inside the episode.
            if self._rows is not None:
                window = draw_rows(self._rows, self._plan, int(index) % self.epoch_samples,
                                   episode_starts=self._episode_starts, focus=self._contact)
            else:
                window = draw_window(self._plan, int(index) % self.epoch_samples, seed=self.seed,
                                     episode_starts=self._episode_starts,
                                     trajectory=self._trajectory, focus=self._contact)
        elif self._rows is not None:
            window = draw_rows(self._rows, self._plan, int(index) % self.epoch_samples,
                               episode_starts=self._episode_starts)
        else:
            window = draw_window(
                self._plan,
                int(index) % self.epoch_samples,
                seed=self.seed,
                episode_starts=self._episode_starts,
                trajectory=self._trajectory,
            )
        return replace(window, pass_index=int(index) // self.epoch_samples)

    def row_plan(self) -> RowPlan | None:
        """The pass layout under rows sampling (quotas, wraps), or None under draws."""
        return self._rows

    def contact_stats(self) -> dict | None:
        """Contact sampling's counts for this copy (focus applied or not, episodes read, store).

        focused counts draws whose start went through the focused inverse CDF, including those
        that landed on the off start; unmoved counts selected draws that kept the off start
        (no window, or strength 1). None when contact sampling is off. Per copy of this dataset,
        as gripper_clips is: a run with loader workers holds one per worker.
        """
        return None if self._contact is None else self._contact.stats()

    def family_shares(self) -> list[dict] | None:
        """The mixture rule's per-family table, or None when the group states no rule."""
        return None if self._families is None else [dict(row) for row in self._families]

    def _row_plan(self) -> RowPlan:
        """rows sampling's pass layout, refusing a manifest it cannot honour."""
        weighted = [entry.name for entry, table in zip(self.manifest.entries, self._trajectory,
                                                       strict=True) if table is not None]
        if weighted:
            raise ValueError(
                f"{len(weighted)} directories carry trajectory weights, first {weighted[0]!r}, "
                "and data.train.sampling is rows.\n"
                "Expected no trajectory weights: rows sampling takes every row of a directory "
                "once per wrap, which a per-episode weight would contradict.\n"
                "Drop trajectory_weights from those entries, or sample this group by draws."
            )
        for entry, starts in zip(self.manifest.entries, self._episode_starts, strict=True):
            if int(starts[0]) != 0:
                raise ValueError(
                    f"directory {entry.name!r} has episode_starts beginning at "
                    f"{int(starts[0])}.\n"
                    "Expected 0: rows sampling numbers a directory's rows from 0.\n"
                    "Rebuild episode_starts for this directory."
                )
        return build_row_plan(self._plan, self.epoch_samples, seed=self.seed)

    def stream_number(self, draw: Draw) -> int:
        """The stream every choice about a draw after its window spends.

        pass * epoch_samples + window index: the sampler's own index for a draw
        this dataset made, so pass 0 reads the stream the window index always
        read, and a window replayed on a later pass reads a stream of its own,
        deterministic in (index, pass, seed), for its reference camera, history
        step, dropout, noise switch and noise.
        """
        return int(draw.pass_index) * self.epoch_samples + int(draw.index)

    def open_reader(self, entry: DatasetEntry) -> EpisodeReader:
        """The reader for one directory, over the cache the manifest names."""
        index = self.manifest.index_of(entry.name)
        return CacheReader(
            root=self.root,
            arrow_path=entry.arrow_path,
            frames_path=entry.frames_path,
            episode_starts=self._episode_starts[index],
            state_column=entry.state_column,
            action_column=entry.action_column,
            valid_column=entry.valid_column,
            view_keys=tuple(view.spec.key for view in entry.views),
            parts=entry.stored,
        )

    def embodiment_of(self, dataset_index: int, episode: int) -> str:
        """The normaliser key of one kept episode of one directory."""
        entry = self.manifest.entries[dataset_index]
        if entry.embodiment.kind == "constant":
            return entry.embodiment.keys[0]
        codes, _ = self._episode_keys[dataset_index]
        return entry.embodiment.keys[int(codes[int(episode)])]

    def history_draw(self, draw: Draw) -> HistoryDraw:
        """The history step a draw uses, or the no-history form and why.

        Rules 1 to 3 (short, blocked, dropout) come from history.decide_history
        on the draw's own uniforms; the fourth is here, because only the reader
        can say whether any camera has a picture at p. A targets_only dataset
        built without the history knobs skips the dropout rule.
        """
        uniforms = history.history_uniforms(self.stream_number(draw), seed=self.seed)
        blocked = int(draw.episode) in self._blocked[draw.dataset_index]
        drawn = history.decide_history(
            draw.start, uniforms, dropout=self.history_dropout or 0.0, blocked=blocked
        )
        if drawn.step is None:
            return drawn
        entry = self.manifest.entries[draw.dataset_index]
        present = self._reader(draw.dataset_index).present_views(draw.episode, drawn.step)
        if not present & {view.spec.key for view in entry.views}:
            return HistoryDraw(None, "no_camera")
        return drawn

    def targets(self, draw: Draw) -> Targets:
        """One draw's physical targets: unify, canonical gripper, spatial drot, drawn frame.

        The single path both training and the stats build read (G28): the
        sample normalises exactly these numbers, and build_stats measures them.
        Joint and hand targets come from their stored columns by the part's
        rule, the body's from its own. Reads no picture; the reference draw
        uses the reader's presence list.
        """
        return self._context(draw, prefix=False).targets

    def state_rows(self, draw: Draw) -> tuple[torch.Tensor, torch.Tensor]:
        """One draw's physical arm state at t: [n_slots, 29] records and their valid dims.

        The record the sample carries, before the directory's masks and the
        state noise: the row at t with the directory's base offsets
        subtracted and the stored joints and hand beside it, NaN wherever a
        dimension was not read. Reads one row and no picture;
        tools/build_stats.py pools the state bounds from it.
        """
        entry = self.manifest.entries[draw.dataset_index]
        rows = self._reader(draw.dataset_index).rows(draw.episode, draw.start, 1)
        offsets = self._base_offsets(draw.dataset_index, entry, draw.episode)
        state, dims, _ = self._arm_records(entry, rows, offsets)
        return state, dims

    def body_state_rows(self, draw: Draw) -> tuple[torch.Tensor, torch.Tensor]:
        """One draw's physical body at t: the [BODY_DIM] record and its valid dims.

        The base is the last step's motion, torso and head absolute, as the
        stored column holds them; NaN where not read, and everywhere for a key
        with no body. tools/build_stats.py pools the body's state bounds from it.
        """
        entry = self.manifest.entries[draw.dataset_index]
        rows = self._reader(draw.dataset_index).rows(draw.episode, draw.start, 1)
        return self._body_record(entry, rows)

    def loss_dims(self, targets: Targets) -> torch.Tensor:
        """[arms, ARM_DIM] bool: the columns of a draw that train under its key.

        The structural dims of targets AND the directory's trained columns at
        each arm's slot (embodiments.columns: its record's own constants, dead
        readings, gated joints, a masked drot or gripper, and the key's
        decodable columns), so the loader starts from the masks
        the server rebuilds. The per-sample read is applied when packing.
        Train mode only.
        """
        embodiments = self._require_embodiments("loss_dims")
        dims = targets.dims_valid.clone()
        for arm, slot in enumerate(targets.slots):
            dims[arm] &= torch.as_tensor(
                embodiments.columns(targets.directory, targets.key, slot), dtype=torch.bool)
        return dims

    def body_loss_dims(self, targets: Targets) -> torch.Tensor:
        """[BODY_DIM] bool: the body columns of a draw that train under its key.

        The dims the key's body layout can train AND the directory's trained
        body columns (embodiments.body_columns); none for a key with no body.
        Train mode only.
        """
        embodiments = self._require_embodiments("body_loss_dims")
        held = embodiments.body_columns(targets.directory, targets.key)
        return targets.body_dims & torch.as_tensor(held, dtype=torch.bool)

    def key_windows(self, key: str, count: int, *, seed: int) -> list[Draw]:
        """count draws of one embodiment key, as the mixture would draw them given the key.

        Directories are chosen by weight times rows times the share of their
        episode draws that carry the key, then an episode of that key by
        trajectory weight (by length under rows sampling, where every row is a
        start once per wrap), then a start uniformly inside it: the key's own
        slice of the training distribution, for the runner's fit gate (R14).
        """
        tables = []
        for index, entry in enumerate(self.manifest.entries):
            chosen = self._episodes_of_key(index, key)
            if chosen is None:
                continue
            weights, share = chosen
            mass = float(self._plan.weights[index]) * float(self._plan.rows[index]) * share
            if mass > 0.0:
                tables.append((index, mass, weights))
        if not tables:
            raise ValueError(
                f"no directory of manifest {self.manifest.corpus!r} draws key {key!r}.\n"
                "Expected at least one directory with that key and a positive weight.\n"
                "Check the key against the manifest's embodiment fields."
            )
        masses = torch.tensor([mass for _, mass, _ in tables], dtype=torch.float64)
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        picks = torch.multinomial(masses / masses.sum(), int(count), replacement=True,
                                  generator=generator)
        draws = []
        for position, pick in enumerate(picks.tolist()):
            index, _, weights = tables[pick]
            episode = int(torch.multinomial(weights, 1, generator=generator))
            starts = self._episode_starts[index]
            length = int(starts[episode + 1]) - int(starts[episode])
            start = min(int(float(torch.rand((), generator=generator,
                                             dtype=torch.float64)) * length), length - 1)
            draws.append(Draw(position, index, self.manifest.entries[index].name,
                              episode, start))
        return draws

    def trained_summary(self) -> dict:
        """What the runner stamps as trained: per key the frames, prompt heads and rates.

        frames are realised, not assumed: FRAME_DRAWS windows of the key drawn
        as the mixture draws them (key_windows), each arm's anchor row read the
        way its targets read it (realised_frame_ref), and the share of arm-draws
        each spelling took. A camera absent at the step, a stored pose that
        fails its audit and a mounted camera whose arm was not read leave the
        draw's pool, and an arm no usable camera can serve trains in the robot
        frame and counts as robot here; a camera whose pose never passes is
        never counted. R12 reads share > 0 as trained, so a
        spelling that realised nothing is refused at serve (fail closed) and
        one that realised something was trained. frame_arm_draws is the
        denominator. Computed once per dataset copy.
        """
        if self._summary is None:
            self._summary = self._build_summary()
        return copy.deepcopy(self._summary)

    def _build_summary(self) -> dict:
        keys: dict[str, dict] = {}
        for index, entry in enumerate(self.manifest.entries):
            for key in entry.embodiment.keys:
                chosen = self._episodes_of_key(index, key)
                if chosen is None:
                    continue
                _, share = chosen
                mass = float(self._plan.weights[index]) * float(self._plan.rows[index]) * share
                if mass <= 0.0:
                    continue
                slot = keys.setdefault(key, {"heads": set(), "rates": set()})
                slot["rates"].add(entry.control_hz)
                for position, record in enumerate(self._prompt_records(index, entry)):
                    if self.embodiment_of(index, position) == key:
                        slot["heads"].add((record["head"], record["tail"]))
        summary = {}
        for key, slot in sorted(keys.items()):
            frames, arm_draws = self._realised_frames(key)
            logger.info("trained frames of %s over %d arm-draws: %s", key, arm_draws,
                        " ".join(f"{name}={share:.4f}" for name, share in frames.items()))
            summary[key] = {
                "frames": frames,
                "frame_arm_draws": arm_draws,
                "prompt_heads": [list(pair) for pair in sorted(slot["heads"])],
                "control_hz": sorted(slot["rates"], key=lambda hz: (hz is not None, hz or 0.0)),
            }
        return {
            "manifest_sha256": self.manifest.sha256,
            "keys": summary,
            "blocked_episodes": sum(len(blocked) for blocked in self._blocked),
        }

    def _realised_frames(self, key: str) -> tuple[dict[str, float], int]:
        """(share of arm-draws per frame spelling, arm-draws) over FRAME_DRAWS windows of key."""
        counts: dict[str, int] = {}
        total = 0
        for draw in self.key_windows(key, FRAME_DRAWS, seed=self.seed):
            rows = self.realised_frame_ref(draw)
            for arm in self.manifest.entries[draw.dataset_index].arms:
                spelling = conditioning.frame_spelling(int(rows[arm.slot]))
                counts[spelling] = counts.get(spelling, 0) + 1
                total += 1
        if total == 0:
            raise ValueError(
                f"key {key!r} realised no arm-draw over {FRAME_DRAWS} windows.\n"
                "Expected every drawn window to hold at least one arm.\n"
                "Check the key's directories in the manifest."
            )
        return {name: counts[name] / total for name in sorted(counts)}, total

    def sample_fields(self, draw: Draw) -> dict[str, torch.Tensor | str]:
        """One draw -> the robot half of a sample, without the chat half.

        Reads the window at t and, when the draw has history, the step p; places
        both sets of views; builds the state records and codes; normalises the
        targets through the draw's key and packs them. Every field is
        fixed-shape: an absent camera, a missing arm and the no-history form
        are masks and markers, never a different width. The key set is
        constants.batch_keys for this intrinsics mode, with train keys.
        """
        embodiments = self._require_embodiments("sample_fields")
        context = self._context(draw, prefix=True)
        entry, reader, targets = context.entry, context.reader, context.targets
        # The pictures at t and at p in one read, so each file is opened once a sample.
        drawn = self.history_draw(draw)
        earlier_step = self._earlier_step(draw, drawn)
        steps = (draw.start,) if earlier_step is None else (draw.start, earlier_step)
        pictures = reader.frames_at(draw.episode, steps)
        frames = pictures[0]
        if frozenset(frames) != context.present:
            raise ValueError(
                f"directory {entry.name!r} episode {draw.episode} step {draw.start}: the reader "
                f"decoded {sorted(frames)} and states {sorted(context.present)} present.\n"
                "Expected one statement of which cameras have a picture at a step.\n"
                "Fix the reader; frames decodes exactly the keys present_views names."
            )
        slots = {view.spec.key: view.spec.slot for view in entry.views if view.spec.key in frames}
        tiles = {key: tile for key, tile in self._tiles(entry).items() if key in frames}
        pixels, image_shapes, view_valid = self._place(frames, tiles, slots)

        key = targets.key
        emb = embodiments.entry(key)
        alpha = embodiments.band_alpha
        dims = self.loss_dims(targets)
        normalised, bands = [], []
        for arm, slot in enumerate(targets.slots):
            values, band = embodiment.normalise_actions(
                targets.chunk[arm], emb, slot, alpha=alpha, dims_valid=dims[arm].numpy(),
            )
            normalised.append(values)
            bands.append(band)
        # dim_valid is per sample (C7-3): the directory's trained columns read
        # on every step the sample keeps, so no unread value enters the flow
        # input and the mask is constant over the chunk, as at serve.
        actions, action_valid, action_loss_dims, dim_valid = pack_slots(
            torch.stack(normalised), slots=targets.slots, n_slots=self.n_slots,
            chunk_len=self.chunk, action_dim=self.action_dim,
            steps_valid=targets.steps_valid, dims_valid=dims, read=targets.read,
            in_band=torch.stack(bands),
        )
        body_dims = self.body_loss_dims(targets)
        body_values, body_band = embodiment.normalise_body_actions(
            targets.body, emb, alpha=alpha, dims_valid=body_dims.numpy())
        body_actions, body_action_valid, body_loss_dims, body_dim_valid = pack_body(
            body_values, steps_valid=targets.steps_valid[0], dims_valid=body_dims,
            read=targets.body_read, in_band=body_band,
        )

        uniforms = history.history_uniforms(self.stream_number(draw), seed=self.seed)
        earlier = self._earlier(draw, entry, reader, drawn,
                                None if earlier_step is None else pictures[1])
        slot_valid = torch.zeros(self.n_slots, dtype=torch.bool)
        slot_valid[list(targets.slots)] = True
        held, body_held = self._held_dims(entry, key)
        cond = conditioning.build_conditioning(
            earlier["state"], earlier["state_dims"] & held, context.now, context.now_dims & held,
            slot_valid=slot_valid, history=drawn.step is not None,
            frame_ref=targets.frame_ref,
            body_start=earlier["body"], body_start_dims=earlier["body_dims"] & body_held,
            body_now=context.body_now, body_now_dims=context.body_now_dims & body_held,
            body_present=emb.body.present,
        )
        check_body_masks(cond.body_state_code, body_action_valid, body_dim_valid)
        noise = dict(self.state_noise)
        sample: dict[str, torch.Tensor | str] = {
            "pixels": pixels,
            "image_shapes": image_shapes,
            "view_valid": view_valid,
            "history_pixels": earlier["pixels"],
            "history_image_shapes": earlier["image_shapes"],
            "history_view_valid": earlier["view_valid"],
            "prompt": self._prompt(draw.dataset_index, entry, draw.episode,
                                   camera_frame=bool(context.rotations), start=draw.start,
                                   wrap=draw.wrap),
            "embodiment": key,
            "frame_ref": cond.frame_ref,
            "slot_valid": cond.slot_valid,
            "state": cond.state,
            "state_code": cond.state_code,
            "state_dims": cond.state_dims,
            "body_state": cond.body_state,
            "body_state_code": cond.body_state_code,
            "body_state_dims": cond.body_state_dims,
            "history_valid": torch.tensor(drawn.step is not None),
            "dim_valid": dim_valid,
            "body_dim_valid": body_dim_valid,
            "actions": actions,
            "action_valid": action_valid,
            "action_loss_dims": action_loss_dims,
            "state_noise": history.state_noise(uniforms, cond.state_code, cond.state_dims,
                                               **noise),
            "body_actions": body_actions,
            "body_action_valid": body_action_valid,
            "body_loss_dims": body_loss_dims,
            "body_state_noise": history.body_state_noise(
                uniforms, cond.body_state_code, cond.body_state_dims, **noise),
        }
        # The calibration: the earlier view k takes the current view k's record,
        # bit for bit (no source varies inside an episode), valid where both are.
        calibration = self._calibration(draw.dataset_index, entry, draw.episode)
        values, valid = rays.calibration_for_views(
            {key_: calibration[key_] for key_ in slots}, slots=slots, n_views=self.n_views,
        )
        sample["intrinsics"], sample["intrinsics_valid"] = values, valid
        sample["history_intrinsics"] = values.clone()
        sample["history_intrinsics_valid"] = valid & earlier["view_valid"]
        # The camera's rotation, in the same per-view block as its calibration
        # and named apart from the suffix anchor; the earlier block states each
        # camera's pose at p.
        sample["camera_extrinsics"], sample["camera_extrinsics_valid"] = (
            conditioning.view_extrinsics(
                {key_: context.poses[key_] for key_ in slots},
                slots=slots, n_views=self.n_views,
            )
        )
        sample["history_camera_extrinsics"] = earlier["camera_extrinsics"]
        sample["history_camera_extrinsics_valid"] = earlier["camera_extrinsics_valid"]
        wanted = batch_keys(self._geometry(), train=True)
        if set(sample) != wanted:
            raise ValueError(
                f"the sample holds {sorted(set(sample) ^ wanted)} against the contract.\n"
                "Expected exactly constants.batch_keys for this intrinsics mode.\n"
                "Change the contract first, then the producer."
            )
        return sample

    def collate(self, samples: list[dict]) -> dict:
        """This dataset's batch assembly, which the loader takes as its collate_fn."""
        return collate_samples(samples)

    def collate_step(self, read: StepSamples, *, per_device: int, accumulation: int,
                     rank: int | None = None, world: int | None = None) -> list[dict]:
        """One step item's samples -> its micro-batches with their chat packs (collate_step)."""
        return collate_step(read, per_device=per_device, accumulation=accumulation,
                            chat=self.understanding, rank=rank, world=world)

    @property
    def chat_pack(self):
        """The chat corpus's PackConfig, or None when the run trains chat rows or no chat."""
        return None if self.understanding is None else self.understanding.pack

    def __getitem__(self, index: int | StepItem) -> dict | StepSamples:
        """One sample, ready for collate; or, for a StepItem, one step's samples.

        Adds the chat half under "chat" when the run co-trains, so a co-train
        batch carries robot rows and chat rows in the same micro-batch and the
        policy raises if either half is missing. A chat corpus that packs is
        read a whole step at a time: the loader
        passes a StepItem, each robot sample carries its draw number under
        draw_index, and collate_step builds the packs. An index alone is
        refused then, since a row drawn by index is not what the run trains.
        """
        if isinstance(index, StepItem):
            return self.step_samples(index)
        if self.chat_pack is not None:
            raise ValueError(
                f"draw {index} was read by index from a dataset whose chat corpus packs "
                f"{self.chat_pack.tokens} tokens a micro-batch.\n"
                "Expected a StepItem: a packing run is read a whole step at a time, and its "
                "chat half is a pack per micro-batch, not a row per sample.\n"
                "Read it through runner.build_loader, or state pack_tokens null."
            )
        sample = self.sample_fields(self.draw(index))
        if self.understanding is not None:
            sample[CHAT_KEY] = self.understanding[int(index) % len(self.understanding)]
        return sample

    def step_samples(self, item: StepItem) -> StepSamples:
        """A step item -> its robot samples in sampler order, each with its draw_index.

        The samples are exactly the ones __getitem__ draws for those indices,
        without a chat row: collate_step puts the step's packs beside them.
        """
        if self.chat_pack is None:
            raise ValueError(
                f"step {item.step} was read as a step item from a dataset whose chat corpus "
                "does not pack.\n"
                "Expected step items only when data.train.understanding.pack_tokens is "
                "stated: without it the loader reads one sample per index, as before.\n"
                "Read samples by index."
            )
        samples = []
        for index in item.indices:
            sample = self.sample_fields(self.draw(int(index)))
            sample[DRAW_INDEX_KEY] = int(index)
            samples.append(sample)
        return StepSamples(int(item.step), samples)

    def _geometry(self) -> tuple[str, ...]:
        """The geometry records every batch carries: all of GEOMETRY_KEYS (tokens)."""
        return tuple(GEOMETRY_KEYS)

    def _require_embodiments(self, what: str) -> Embodiments:
        if self.mode != "train" or self._embodiments is None:
            raise ValueError(
                f"{what} was asked of a dataset in mode {self.mode!r}.\n"
                "Expected train mode, whose samples are normalised through the "
                "embodiments file.\n"
                "Read physical targets with targets(); normalised output needs train mode."
            )
        return self._embodiments

    def _load_normaliser(self, normaliser: str | Path | None) -> Embodiments | None:
        """The embodiments file, required in train mode (G33, G39)."""
        if normaliser is None:
            if self.mode == "train":
                raise ValueError(
                    "data.train.normaliser is not set.\n"
                    "Expected the embodiments.json this run normalises through.\n"
                    "Set it to ${data.root}/prep_${data.corpus}/embodiments.json."
                )
            return None
        path = Path(normaliser)
        return Embodiments.load(path if path.is_absolute() else self.root / path)

    def _check_embodiments(self) -> None:
        """Every directory's keys, slots and client record against the embodiments file.

        R3 for a key the file does not hold, R15 for an arm slot outside its
        entry, and R4 for a directory whose datasets record no longer matches
        the manifest's own client inputs (another manifest's normaliser).
        """
        if self._embodiments is None:
            return
        for entry in self.manifest.entries:
            for key in entry.embodiment.keys:
                emb = self._embodiments.entry(key)
                for arm in entry.arms:
                    emb.slot_index(arm.slot)
            record = self._embodiments.record(entry.name)
            wanted = embodiment.record_digest(entry.client_record)
            if record.digest != wanted:
                raise ValueError(
                    f"{refusal_prefix(4)} directory {entry.name!r}: the embodiments file's "
                    f"record is {record.digest} and this manifest's entry gives {wanted}.\n"
                    "Expected the normaliser built from the manifest this run trains.\n"
                    "Rebuild the stats from this manifest, or train the manifest they came from."
                )
            self._check_parts_held(entry, record)
            # The six client fields leave the rotation convention out, so the
            # record digest cannot see a manifest whose convention moved after
            # the stats were measured; the record's drot can.
            drot = embodiment.DROT_RECORD[entry.rotation_delta_frame]
            if record.drot != drot:
                raise ValueError(
                    f"directory {entry.name!r}: the embodiments file "
                    f"states drot {record.drot!r} and the manifest's rotation_delta_frame "
                    f"{entry.rotation_delta_frame!r} gives {drot!r}.\n"
                    "Expected the stats measured under the convention this manifest trains.\n"
                    "Rebuild the stats from this manifest, or train the manifest they came from."
                )

    def _check_parts_held(self, entry: DatasetEntry, record) -> None:
        """The embodiments file and the manifest agree on what this directory reads.

        Every key of the directory has a body exactly when the manifest names
        where the body comes from (ABSENT is the key's, C7-2), and every
        joint, hand or body dimension the record trains or holds valid is one
        a layout of the manifest reads. Otherwise the normaliser describes a
        directory the loader cannot read, and its masks would train nothing
        where they claim to train something.
        """
        for key in entry.embodiment.keys:
            present = self._embodiments.entry(key).body.present
            if present != (entry.body is not None):
                raise ValueError(
                    f"directory {entry.name!r}: key {key!r} "
                    f"{'has' if present else 'has no'} body and the manifest states body "
                    f"{entry.body}.\n"
                    "Expected a body block exactly for a key with a body: ABSENT is the key's.\n"
                    "Rebuild the manifest and the stats from one config/embodiments.yaml."
                )
        spans = {"joints": (ACTION_JOINTS, STATE_JOINTS), "hand": (ACTION_HAND, STATE_HAND)}
        for arm in entry.arms:
            row = record.slot_row(arm.slot)
            for field, (action_span, state_span) in spans.items():
                part = None if arm.state is None else arm.state.part(field)
                read = 0 if part is None else part.count
                trained = np.flatnonzero(record.columns[row, action_span]).tolist()
                held = np.flatnonzero(record.state_dims[row, state_span]).tolist()
                beyond = [one for one in trained + held if one >= read]
                if beyond:
                    raise ValueError(
                        f"directory {entry.name!r} arm {arm.spec.name!r}: the embodiments "
                        f"record trains or holds {field} {sorted(set(beyond))} and the manifest "
                        f"reads {read} of them.\n"
                        f"Expected the record's {field} inside what the arm's {field} block "
                        "reads.\n"
                        "Rebuild the stats from this manifest, or train the manifest they came "
                        "from."
                    )
        if entry.body is not None:
            for said, flags, readable in (
                    ("trains", record.body_columns, entry.body.action_dims().numpy()),
                    ("holds", record.body_state_dims, entry.body.state_dims().numpy())):
                beyond = np.flatnonzero(flags & ~readable).tolist()
                if beyond:
                    raise ValueError(
                        f"directory {entry.name!r}: the embodiments record {said} body dims "
                        f"{beyond} and the manifest's body {entry.body} reads none of them.\n"
                        "Expected the record's body inside what the body block reads.\n"
                        "Rebuild the stats from this manifest, or train the manifest they came "
                        "from."
                    )

    def _episodes_of_key(self, index: int, key: str):
        """(episode draw weights of the key, the key's share of the directory's draws), or None."""
        entry = self.manifest.entries[index]
        if key not in entry.embodiment.keys:
            return None
        trajectory = self._trajectory[index]
        episodes = entry.episodes
        if self.sampling == "rows":
            # Every row is a window start once per wrap, so an episode is drawn by its length.
            base = torch.diff(self._episode_starts[index]).to(torch.float64)
        elif trajectory is None:
            base = torch.ones(episodes, dtype=torch.float64)
        else:
            cumulative = trajectory.cumulative
            base = torch.diff(cumulative, prepend=torch.zeros(1, dtype=torch.float64))
        if entry.embodiment.kind == "constant":
            chosen = base
        else:
            codes, _ = self._episode_keys[index]
            member = torch.as_tensor(
                codes == entry.embodiment.keys.index(key), dtype=torch.bool
            )
            chosen = torch.where(member, base, torch.zeros_like(base))
        total = float(base.sum())
        share = float(chosen.sum()) / total if total > 0.0 else 0.0
        if share <= 0.0:
            return None
        return chosen, share

    def _place(self, frames: dict, tiles: dict, slots: dict):
        pixels, image_shapes, view_valid = images.place_views(
            frames, tiles, slots=slots, n_views=self.n_views, cell=self.manifest.cell,
            short_side=self.manifest.short_side, token=self.manifest.token,
            resample=self.resample,
        )
        images.check_absence_agrees(image_shapes, view_valid)
        return pixels, image_shapes, view_valid

    def _held_dims(self, entry: DatasetEntry, key: str) -> tuple[torch.Tensor, torch.Tensor]:
        """The state dims the directory holds valid under a key: [n_slots, 29] and [8].

        embodiments.state_dims per arm slot and embodiments.body_state_dims
       : its dead readings, gated joints, a masked aperture and
        the key's absent dims. A read dimension outside them is masked, not
        fed, exactly as the server masks it, so train and serve read one rule.
        """
        embodiments = self._require_embodiments("_held_dims")
        held = torch.zeros((self.n_slots, STATE_DIM), dtype=torch.bool)
        for arm in entry.arms:
            held[arm.slot] = torch.as_tensor(
                embodiments.state_dims(entry.name, key, arm.slot), dtype=torch.bool)
        body = torch.as_tensor(embodiments.body_state_dims(entry.name, key), dtype=torch.bool)
        return held, body

    def _earlier_step(self, draw: Draw, drawn: HistoryDraw) -> int | None:
        """The history step a sample reads, held at least MIN_GAP rows before t, or None."""
        if drawn.step is None:
            return None
        step = int(drawn.step)
        if step > draw.start - history.MIN_GAP:
            raise ValueError(
                f"history step {step} for a draw at {draw.start}.\n"
                f"Expected at most {draw.start - history.MIN_GAP}, MIN_GAP rows back.\n"
                "Draw the step with history.draw_history_step."
            )
        return step

    def _earlier(self, draw: Draw, entry: DatasetEntry, reader: EpisodeReader,
                 drawn: HistoryDraw, frames: dict[str, torch.Tensor] | None
                 ) -> dict[str, torch.Tensor]:
        """The views, poses, arm and body records at the history step, or their absent form.

        The absent form is what a server builds for call 0: zero pixels, zero
        shapes, false flags, NaN geometry and no state records. Every read here
        is at p, which is at least MIN_GAP rows before t; frames are the
        pictures at p, read with the ones at t. The records are as read; the
        directory's masks are applied with the ones at t.
        """
        if drawn.step is None:
            pixels, image_shapes, view_valid = self._place({}, {}, {})
            poses, poses_valid = conditioning.view_extrinsics({}, slots={}, n_views=self.n_views)
            return {
                "pixels": pixels, "image_shapes": image_shapes, "view_valid": view_valid,
                "camera_extrinsics": poses, "camera_extrinsics_valid": poses_valid,
                "state": torch.full((self.n_slots, STATE_DIM), float("nan")),
                "state_dims": torch.zeros((self.n_slots, STATE_DIM), dtype=torch.bool),
                "body": torch.full((BODY_DIM,), float("nan")),
                "body_dims": torch.zeros(BODY_DIM, dtype=torch.bool),
            }
        step = self._earlier_step(draw, drawn)
        slots = {view.spec.key: view.spec.slot for view in entry.views if view.spec.key in frames}
        tiles = {key: tile for key, tile in self._tiles(entry).items() if key in frames}
        pixels, image_shapes, view_valid = self._place(frames, tiles, slots)
        rows = reader.rows(draw.episode, step, 1)
        offsets = self._base_offsets(draw.dataset_index, entry, draw.episode)
        state, state_dims, base_from_arm = self._arm_records(entry, rows, offsets)
        body, body_dims = self._body_record(entry, rows)
        poses = self._camera_rotations(draw.dataset_index, entry, draw.episode,
                                       base_from_arm, keys=set(slots))
        values, valid = conditioning.view_extrinsics(
            {key: poses[key] for key in slots}, slots=slots, n_views=self.n_views,
        )
        return {
            "pixels": pixels, "image_shapes": image_shapes, "view_valid": view_valid,
            "camera_extrinsics": values, "camera_extrinsics_valid": valid & view_valid,
            "state": state, "state_dims": state_dims, "body": body, "body_dims": body_dims,
        }

    def _context(self, draw: Draw, *, prefix: bool) -> _Context:
        """Everything one draw needs at t: the state, the reference, the poses, the targets."""
        entry = self.manifest.entries[draw.dataset_index]
        reader = self._reader(draw.dataset_index)
        # chunk + 1 rows: step j's joint, hand and body targets are the state
        # one row on, and the Euler conversion reads the orientation after the
        # last step.
        window = reader.rows(draw.episode, draw.start, self.chunk + 1)
        present = frozenset(reader.present_views(draw.episode, draw.start))
        offsets = self._base_offsets(draw.dataset_index, entry, draw.episode)
        now, now_dims, base_from_arm = self._arm_records(entry, window, offsets)
        body_now, body_now_dims = self._body_record(entry, window)

        # One mapping from view key to rotation feeds both the prefix record and
        # the chunk, so a camera cannot reach the model as one pose and turn a
        # chunk by another.
        also = set()
        if prefix:
            also = {view.spec.key for view in entry.views if view.spec.key in present}
        poses, rotations, frame_ref = self._drawn_frames(
            draw, entry, present, base_from_arm, also=also, count=True
        )
        targets = self._action_targets(draw, entry, window, rotations, frame_ref)
        return _Context(entry, reader, present, now, now_dims, body_now, body_now_dims, poses,
                        rotations, targets)

    def _drawn_frames(self, draw: Draw, entry: DatasetEntry, present: frozenset[str],
                      base_from_arm: dict[int, torch.Tensor], *, also: set[str],
                      count: bool):
        """The audit, the reference draw over what passed it, and the anchor rows.

        (poses by view key, rotations by action slot, frame_ref [n_slots]).
        Every posed camera with a picture at the step is audited first, and the
        draw is over the ones whose pose passed: a sample with any usable camera is in
        camera mode, and a
        slot reads the robot row only when no usable camera can serve it. The
        one path from a draw to its frame: targets and the trained stamp's
        realised frames both come through here.
        """
        candidates = {camera.key for camera in present_pool(self._posed(draw, entry), present)}
        poses = self._camera_rotations(
            draw.dataset_index, entry, draw.episode, base_from_arm, keys=candidates | set(also)
        )
        usable = frozenset(key for key in candidates if poses[key] is not None)
        drawn = self._reference(draw, entry, present, usable, count=count)
        rotations = {slot: poses[camera.key] for slot, camera in drawn.cameras.items()}
        failed = sorted(slot for slot, rotation in rotations.items() if rotation is None)
        if failed:
            raise RuntimeError(
                f"directory {entry.name!r}: slots {failed} drew a camera whose pose failed "
                "its audit.\nExpected the draw over the usable cameras only.\n"
                "Pass the audited keys to draw_reference as usable."
            )
        frame_ref = conditioning.frame_ref_rows(
            {slot: drawn.cameras[slot].slot for slot in rotations}, n_slots=self.n_slots,
        )
        return poses, rotations, frame_ref

    def realised_frame_ref(self, draw: Draw) -> torch.Tensor:
        """[n_slots] the anchor rows targets(draw) gives, without reading the chunk.

        A robot-frame directory reads nothing. A camera-frame one reads the
        presence of its cameras at the draw's step and, when a camera is
        mounted on an arm, that step's state row, which is everything the
        reference draw and the pose audit look at: a camera absent at the step,
        a stored pose that fails the audit, and a mounted camera whose arm pose
        was not read all leave the pool before the draw, as they train, and a
        slot no usable camera can serve comes back as the robot row.
        """
        entry = self.manifest.entries[draw.dataset_index]
        if conditioning.parse_frame_ref(entry.frame_ref)[0] != "camera":
            return conditioning.frame_ref_rows({}, n_slots=self.n_slots)
        reader = self._reader(draw.dataset_index)
        present = frozenset(reader.present_views(draw.episode, draw.start))
        base_from_arm: dict[int, torch.Tensor] = {}
        if any(view.extrinsics.kind == MOUNTED for view in entry.views):
            rows = reader.rows(draw.episode, draw.start, 1)
            offsets = self._base_offsets(draw.dataset_index, entry, draw.episode)
            _, _, base_from_arm = self._arm_records(entry, rows, offsets)
        return self._drawn_frames(draw, entry, present, base_from_arm, also=set(),
                                  count=False)[2]

    def _action_targets(self, draw: Draw, entry: DatasetEntry,
                        window: dict[str, torch.Tensor], rotations: dict[int, torch.Tensor],
                        frame_ref: torch.Tensor) -> Targets:
        """The v7 row per arm and the body's.

        The EE columns: unify, canonical gripper, spatial drot, drawn-frame
        rotate. Joints and hand: part_targets, the stored state at row t+j+1
        or the recorded command at row t+j, never turned by the frame. The
        body: body_targets. A step exists only when row t+j holds an action and
        its end row t+j+1 is inside the episode: every target of step j is the
        motion t+j -> t+j+1, so the last row of an episode is the end of a
        step and the start of none (a step every group cannot have would keep
        the EE and drop the joints from every window touching an episode's end).
        window holds chunk + 1 rows.
        """
        layout = entry.action
        arms = len(entry.arms)
        steps = self.chunk
        length = self._episode_length(draw)
        inside = torch.arange(steps + 1) + int(draw.start) < length
        exists = window[ROW_VALID][:steps] & inside[1:]
        steps_valid = exists[None, :].repeat(arms, 1)
        flat = window[ROW_ACTION][:steps]
        chunk = unify(flat, layout)
        ee_read = self._flat_read(layout, flat, gripper_trained=entry.gripper.method != "masked")

        gripper = entry.gripper
        for arm, fields in enumerate(layout.arms):
            if "gripper" not in fields:
                continue
            # The stride is the row width (block_columns), never a field's
            # stop: arm 1's gripper is flat column 39 (trap T1).
            column = block_columns(arm, GRIPPER).start
            if gripper.method == "masked":
                chunk[:, column] = 0.0
                continue
            canonical, _ = embodiment.canonical_gripper(
                chunk[:, column], gripper.raw_open, gripper.raw_closed
            )
            moved = (chunk[:, column] - gripper.raw_open) / (gripper.raw_closed - gripper.raw_open)
            outside = ((moved < 0.0) | (moved > 1.0)) & steps_valid[arm]
            if bool(outside.any()):
                first = entry.name not in self._gripper_clips
                self._gripper_clips[entry.name] = (
                    self._gripper_clips.get(entry.name, 0) + int(outside.sum())
                )
                if first:
                    # Counted per copy of the dataset, which in training is a
                    # loader worker the runner does not read, so the first clip
                    # of a directory in a worker is also said (V6 1.4).
                    logger.warning(
                        "directory %s: a gripper value outside raw_open %s .. raw_closed %s "
                        "was clipped to the canonical [0, 1]; gripper_clips() counts them",
                        entry.name, gripper.raw_open, gripper.raw_closed,
                    )
            chunk[:, column] = canonical

        if layout.rotation == "body_to_spatial":
            # CX 12: every body delta becomes spatial before any frame turns it,
            # in every mode, so the robot frame and each fallback read one target.
            orientation, read = self._orientations(entry, window[ROW_STATE][:steps],
                                                   inside[:steps])
            chunk = body_to_spatial(chunk, orientation)
            ee_read &= read
        elif layout.rotation == "euler_to_spatial":
            orientation, read = self._orientations(entry, window[ROW_STATE], inside)
            chunk = euler_to_spatial(chunk, orientation)
            ee_read &= read[:, :-1] & read[:, 1:]

        stored = _stored_rows(entry, window)
        parts, parts_read = part_targets(layout, stored, inside=inside)
        chunk = chunk + parts
        read = parts_read.clone()
        for arm, fields in enumerate(layout.arms):
            for name in fields:
                read[:, block_columns(arm, ARM_FIELDS[name])] = ee_read[arm][:, None]

        by_block = {arm: rotations[one.slot] for arm, one in enumerate(entry.arms)
                    if one.slot in rotations}
        if by_block:
            chunk = rotate_chunk(chunk, by_block, layout.rotation)
        body, body_read = body_targets(entry.body, stored, inside=inside)
        return Targets(
            chunk=blocks_of(chunk),
            steps_valid=steps_valid,
            dims_valid=_structural_dims(entry),
            slots=tuple(arm.slot for arm in entry.arms),
            key=self.embodiment_of(draw.dataset_index, draw.episode),
            frame_ref=frame_ref,
            directory=entry.name,
            read=blocks_of(read),
            body=body,
            body_read=body_read,
            body_dims=torch.zeros(BODY_DIM, dtype=torch.bool) if entry.body is None
            else entry.body.action_dims(),
        )

    @staticmethod
    def _flat_read(layout: ActionLayout, flat: torch.Tensor, *,
                   gripper_trained: bool) -> torch.Tensor:
        """[arms, K] bool: each arm's EE group read at each step from the flat action.

        The group is read where every flat column that carries a target holds
        a number a measurement could be. The EE is one group (the band's and
        the loss's), so one bad column takes the whole group at that step. A
        masked gripper and a drot the policy trains nothing from carry none,
        so whatever their flat columns hold (language_table's fabricated
        gripper) reads nothing either way.
        """
        trained = {"dpos": True, "drot": layout.rotation in ROTATION_TRAINED,
                   "gripper": gripper_trained}
        read = torch.ones((layout.n_arms, int(flat.shape[0])), dtype=torch.bool)
        for arm, fields in enumerate(layout.arms):
            for name, columns in fields.items():
                if not trained[name]:
                    continue
                picked = flat.index_select(-1, torch.as_tensor(columns, dtype=torch.long))
                read[arm] &= within_range(picked).all(dim=-1)
        return read

    def _orientations(self, entry: DatasetEntry, rows: torch.Tensor,
                      inside: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Each arm's R_base<-ee at each row, and the rows it was read at.

        From the manifest's rotation_source (Euler angles in the state column)
        where it states one, else from the arm's state layout; a row outside
        the episode, or one whose pose was not read, is unread and keeps the
        identity, which no valid step reads.
        """
        steps = int(rows.shape[0])
        orientation = torch.eye(3, dtype=torch.float64).repeat(steps, len(entry.arms), 1, 1)
        read = torch.zeros((len(entry.arms), steps), dtype=torch.bool)
        source = entry.rotation_source
        for arm, one in enumerate(entry.arms):
            if source is not None:
                angles = rows[:, source.at:source.at + 3].to(torch.float64)
                matrices = euler_matrix(angles, source.euler)
                measured = within_range(angles).all(dim=-1)
            elif one.state is not None and one.state.has_pose:
                poses = arm_poses(rows, one.state)
                matrices, measured = poses.rotation, poses.pose
            else:
                continue
            read[arm] = measured & inside[:steps]
            orientation[:, arm] = torch.where(read[arm][:, None, None], matrices,
                                              orientation[:, arm])
        return orientation, read

    def _episode_length(self, draw: Draw) -> int:
        starts = self._episode_starts[draw.dataset_index]
        return int(starts[draw.episode + 1]) - int(starts[draw.episode])

    def _arm_records(self, entry: DatasetEntry, rows: dict[str, torch.Tensor],
                     offsets: dict[int, torch.Tensor]):
        """A window's first row -> records [n_slots, 29], their dims, and each posed arm's raw R.

        The pose and aperture come from the state row, the joints and hand
        from their stored columns at the same row. An arm with no state layout
        is unknown on its own; a slot no arm fills is left for
        build_conditioning to mark absent. The raw R is what a mounted camera
        is composed with and what nothing normalises: the key's end-effector
        fix applies to the record's 6D in the normaliser only. It is there for
        every arm whose pose was read, whatever else of it was.
        """
        raw = rows[ROW_STATE][0]
        stored = {name: values[0] for name, values in _stored_rows(entry, rows).items()}
        state, dims = split_arms(raw, entry.arms, n_slots=self.n_slots,
                                 joints=stored.get(COLUMN_JOINTS), hand=stored.get(COLUMN_HAND),
                                 base_offsets=offsets)
        base_from_arm: dict[int, torch.Tensor] = {}
        for arm in entry.arms:
            if arm.state is None or not arm.state.has_pose:
                continue
            poses = arm_poses(raw[None], arm.state)
            if bool(poses.pose[0]):
                base_from_arm[arm.slot] = poses.rotation[0]
        return state, dims, base_from_arm

    def _body_record(self, entry: DatasetEntry,
                     rows: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """A window's first row -> the body record [BODY_DIM] float32 and its dims, as read."""
        stored = _stored_rows(entry, rows).get(COLUMN_BODY) if entry.body is not None else None
        record, dims = read_body(None if stored is None else stored[:1], entry.body, count=1)
        return record[0].to(torch.float32), dims[0]

    def _base_offsets(self, dataset_index: int, entry: DatasetEntry,
                      episode: int) -> dict[int, torch.Tensor]:
        """Per arm slot, the metres its stored position is off the robot base, for one episode."""
        if dataset_index not in self._offsets:
            tables = {}
            for arm in entry.arms:
                if arm.base_offset is None:
                    continue
                array = np.asarray(np.load(self.root / arm.base_offset), dtype=np.float64)
                if array.shape != (entry.episodes, 3) or not np.isfinite(array).all():
                    raise ValueError(
                        f"directory {entry.name!r} arm {arm.spec.name!r} base offsets in "
                        f"{arm.base_offset!r} are {array.shape} or not finite.\n"
                        f"Expected ({entry.episodes}, 3) finite metres, one row per episode.\n"
                        "Rebuild them with the LIBERO base-offset measurement."
                    )
                tables[arm.slot] = array
            self._offsets[dataset_index] = tables
        return {slot: torch.from_numpy(table[int(episode)])
                for slot, table in self._offsets[dataset_index].items()}

    def _read_episode_keys(self, entry: DatasetEntry):
        """A per-episode directory's key codes and source episode ids, length-checked (R16)."""
        spec = entry.embodiment
        if spec.kind == "constant":
            return None
        codes = np.asarray(np.load(self.root / spec.path))
        ids = np.asarray(np.load(self.root / spec.ids_path))
        for name, array in (("key", codes), ("episode id", ids)):
            if array.ndim != 1 or array.shape[0] != entry.episodes:
                raise ValueError(
                    f"{refusal_prefix(16)} directory {entry.name!r} has {entry.episodes} kept "
                    f"episodes and a {name} array of shape {array.shape}.\n"
                    "Expected one value per kept episode, in kept order.\n"
                    "Rebuild the arrays with tools/build_embodiment.py."
                )
        if codes.dtype != np.int8 or ids.dtype != np.int64:
            raise ValueError(
                f"directory {entry.name!r} key array is {codes.dtype} and id array {ids.dtype}.\n"
                "Expected int8 key indexes and int64 source episode ids.\n"
                "Rebuild the arrays with tools/build_embodiment.py."
            )
        if codes.min(initial=0) < 0 or codes.max(initial=0) >= len(spec.keys):
            raise ValueError(
                f"directory {entry.name!r} key array holds indexes outside 0.."
                f"{len(spec.keys) - 1}.\n"
                f"Expected indexes into its keys {list(spec.keys)}.\n"
                "Rebuild the arrays with tools/build_embodiment.py."
            )
        if len(np.unique(ids)) != ids.shape[0]:
            raise ValueError(
                f"{refusal_prefix(16)} directory {entry.name!r} names one source episode at "
                "two kept positions.\n"
                "Expected each kept position joined to its own source episode.\n"
                "Rebuild the arrays with tools/build_embodiment.py."
            )
        return codes, ids

    def _read_blocked(self, entry: DatasetEntry) -> frozenset[int]:
        """The kept positions whose draws always take the no-history form."""
        outside = [one for one in entry.history_blocked if not 0 <= one < entry.episodes]
        if outside:
            raise ValueError(
                f"directory {entry.name!r} blocks history at kept positions {outside}.\n"
                f"Expected positions in 0..{entry.episodes - 1}.\n"
                "Rebuild history.blocked from the blocked list through kept.jsonl."
            )
        return frozenset(entry.history_blocked)

    def _reader(self, dataset_index: int) -> EpisodeReader:
        if dataset_index not in self._readers:
            entry = self.manifest.entries[dataset_index]
            reader = self.open_reader(entry)
            _check_rotation_source(entry, reader)
            self._readers[dataset_index] = reader
        return self._readers[dataset_index]

    def _tiles(self, entry: DatasetEntry) -> dict[str, images.Tile]:
        """The recorded grid of each view, as the tile the loader writes.

        The grid comes from the manifest rather than from snap_grid, and the
        manifest read refused any grid this tree's rule does not give its shape,
        so a run whose snapping rule moved is refused before it starts rather
        than spending other tokens under the same config.
        """
        token = self.manifest.token
        return {
            view.spec.key: images.Tile(
                view.grid[0], view.grid[1], view.grid[0] * token, view.grid[1] * token
            )
            for view in entry.views
        }

    def _posed(self, draw: Draw, entry: DatasetEntry) -> tuple[ReferenceCamera, ...]:
        """The cameras one directory's samples may be written against; none in robot mode."""
        # THE spelling trap: "camera:1" != "camera" is True, so a literal comparison sends
        # every PINNED corpus down the robot path and the chunk is never rotated into the
        # camera frame. Training then learns robot-frame actions while the manifest says
        # camera:N, and eval -- which resolves the slot correctly -- rotates anyway. That
        # silent mismatch scored 0.00% over 2258 evaluation episodes. Compare the PARSED
        # kind, never the string.
        if conditioning.parse_frame_ref(entry.frame_ref)[0] != "camera":
            return ()
        if draw.dataset_index not in self._cameras:
            self._cameras[draw.dataset_index] = entry.reference_cameras()
        return self._cameras[draw.dataset_index]

    def _reference(
        self, draw: Draw, entry: DatasetEntry, present: frozenset[str],
        usable: frozenset[str], *, count: bool = True,
    ) -> Reference:
        """The camera each action slot of one sample is written against.

        A pure function of the draw index, the run seed, which cameras the
        reader returned a picture for at that step and which of them passed
        the pose audit there, so the sample a resumed run rebuilds is the
        sample it would have drawn: presence is a column of the cache, the
        audit reads the stored poses and that step's state, and the row is what
        the index and the seed already fix. A directory in robot mode offers no
        camera and every slot comes back empty.

        present is the reader's own answer rather than the manifest's, so a
        camera whose join found no frame near enough to this row is out of the
        draw for this sample alone; usable takes out the cameras whose pose
        failed its audit the same way. count False leaves the counters alone,
        for a draw made to measure the frames rather than to train.
        """
        if self.frame_mode is not None:
            from .frame_modes import choose
            key = self.embodiment_of(draw.dataset_index, draw.episode)
            episode_id = self._prompt_records(draw.dataset_index, entry)[draw.episode]["episode"]
            frames = choose(self.frame_set[key], seed=self.seed, directory=entry.name,
                            episode_id=episode_id,
                            sample=self.stream_number(draw) if self.frame_draw == "sample" else None)
            posed = tuple(camera for camera in self._posed(draw, entry)
                          if f"camera:{camera.slot}" in frames)
            present_cameras = present_pool(posed, present)
            pool = {camera.slot: camera for camera in present_pool(present_cameras, usable)}
            selected = {}
            for arm, spelling in zip(entry.arms, frames):
                kind, slot = conditioning.parse_frame_ref(spelling)
                if kind == "camera":
                    if slot in pool:
                        selected[arm.spec.slot] = pool[slot]
            if count and len(present_cameras) != len(posed):
                self._count_absence(entry, posed=len(posed), left=len(present_cameras))
            if count and len(pool) != len(present_cameras):
                self._count_audit(entry, robot=not selected,
                                  arm_robot=bool(selected) and len(selected) != len(entry.arms))
            if count and selected and (
                len(present_cameras) != len(posed)
                and self._absences.get(entry.name, {}).get("restricted") == 1
                or len(pool) != len(present_cameras)
                and self._audits.get(entry.name, {}).get("cut") == 1
            ):
                logger.warning("directory %s: an unavailable selected camera leaves an arm "
                               "in the robot frame", entry.name)
            strategy = PER_ARM if len({camera.slot for camera in selected.values()}) > 1 else SHARED
            return Reference(strategy, selected)
        posed = self._posed(draw, entry)
        if not posed:
            return Reference(SHARED, {})
        pool = present_pool(posed, present)
        if count and len(pool) != len(posed):
            self._count_absence(entry, posed=len(posed), left=len(pool))
        reference = draw_reference(
            posed,
            tuple(arm.spec.slot for arm in entry.arms),
            self.stream_number(draw),
            seed=self.seed,
            common_base_frame=entry.common_base_frame,
            present=present,
            usable=usable,
        )
        kept = present_pool(pool, usable)
        if count and len(kept) != len(pool):
            served = [reference.camera_for(arm.spec.slot) is not None for arm in entry.arms]
            self._count_audit(entry, robot=not any(served), arm_robot=any(served)
                              and not all(served))
        return reference

    def _count_absence(self, entry: DatasetEntry, *, posed: int, left: int) -> None:
        """Record a draw whose reference pool was cut, and say so the first time.

        Two counts per directory: restricted is the draws taken over fewer
        cameras than the manifest poses, and robot_fallback is the ones that
        found none of them and put every slot in the robot frame. The second is
        the case that would otherwise fall through to whichever camera the
        uniform landed on, so it is counted rather than left implicit, and the
        first one in a worker is logged: a directory that keeps losing its whole
        pool is a question about the join's tolerance, not about the loader.
        """
        counts = self._absences.setdefault(
            entry.name, {"restricted": 0, "robot_fallback": 0}
        )
        counts["restricted"] += 1
        if left:
            return
        counts["robot_fallback"] += 1
        if counts["robot_fallback"] == 1:
            logger.warning(
                "directory %s: a draw opened at a step none of its %d posed "
                "cameras has a frame at, so that sample is in the robot frame",
                entry.name, posed,
            )

    def _count_audit(self, entry: DatasetEntry, *, robot: bool, arm_robot: bool) -> None:
        """Record a draw whose pool lost a camera with a picture to the pose audit.

        Three counts per directory: cut is the draws where at least one present
        camera's pose failed its audit at the step and the draw went on over
        the rest; robot the ones of those that came back with no camera at
        all, every slot in the robot frame; arm_robot the ones where one arm
        kept a camera and another is in the robot frame (PER_ARM over the arms
        it can serve). The first robot one in a worker is logged: a directory
        whose poses keep failing is a question about its extrinsics, not about
        the loader.
        """
        counts = self._audits.setdefault(entry.name, {"cut": 0, "robot": 0, "arm_robot": 0})
        counts["cut"] += 1
        counts["arm_robot"] += int(arm_robot)
        if not robot:
            return
        counts["robot"] += 1
        if counts["robot"] == 1:
            logger.warning(
                "directory %s: a draw opened at a step where no camera with a picture "
                "has a pose that passes its audit, so that sample is in the robot frame",
                entry.name,
            )

    def _camera_rotations(
        self,
        dataset_index: int,
        entry: DatasetEntry,
        episode: int,
        base_from_arm: dict[int, torch.Tensor],
        *,
        keys: set[str],
    ) -> dict[str, torch.Tensor | None]:
        """The named cameras' R_base_from_camera at the drawn step, or None each.

        One answer per key, so the pose the prefix states for a camera and the
        rotation a chunk written against that camera turned by are one audit.
        keys is what the sample needs an answer for and not simply every view:
        the cameras the reference draw may pick (audited before it draws) and
        the ones the prefix states; reading a rotation file nothing consumes
        would make a directory's unaudited extrinsics fail a run that never
        looks at them.

        A view the manifest gives no pose is None, and so is one whose stored
        pose is not a rotation: an extrinsic that failed its audit is unknown,
        and repairing it here would put a measured-looking number in the prefix
        and rotate a chunk by it. A mounted camera is composed with the arm it
        is bolted to at the step the window opens at, so the pose is the one
        that camera held then rather than the one it held at the episode's
        first frame; an arm whose own pose was not read gives its camera none.
        """
        posed = {view.spec.key: view for view in entry.views}
        found: dict[str, torch.Tensor | None] = {}
        for key in sorted(keys):
            view = posed[key]
            found[key] = None
            if view.extrinsics.kind == "none":
                continue
            table = self._rotation_table(dataset_index, entry, key)
            rotation = table.at(episode)
            if table.kind == MOUNTED:
                arm = base_from_arm.get(view.spec.arm_slot)
                if arm is None:
                    continue
                rotation = arm @ rotation
            if conditioning.is_rotation(rotation):
                found[key] = rotation.to(torch.float32)
        return found

    def _rotation_table(
        self, dataset_index: int, entry: DatasetEntry, key: str
    ) -> Rotations:
        """One camera's rotations, opened on first use and kept by view key."""
        if dataset_index not in self._rotations:
            self._rotations[dataset_index] = self._read_rotations(entry)
        return self._rotations[dataset_index][key]

    def _read_rotations(self, entry: DatasetEntry) -> dict[str, Rotations]:
        """Every camera's rotations of one directory, keyed by view key."""
        tables: dict[str, Rotations] = {}
        for view in entry.views:
            kind = view.extrinsics.kind
            if kind == "none":
                continue
            array = np.asarray(np.load(self.root / view.extrinsics.path))
            if kind == "constant" and array.shape == (3, 3):
                array = array.reshape(1, 3, 3)
            expected = (1 if kind == "constant" else entry.episodes, 3, 3)
            if tuple(array.shape) != expected:
                # Named here rather than left to a reshape: a constant camera's
                # file holding one 3x3 per kept episode is a stage writing the
                # episode layout for it, and numpy's reshape error says neither.
                raise ValueError(
                    f"directory {entry.name!r} camera {view.spec.key!r} has rotations "
                    f"of shape {tuple(array.shape)} in {view.extrinsics.path!r}.\n"
                    f"Expected {expected}, one 3x3 per episode in kept order, or one for "
                    "the whole directory under kind constant.\n"
                    "Rebuild the extrinsics file, or correct the entry's episode count."
                )
            tables[view.spec.key] = Rotations(kind, array)
        return tables

    def _calibration(
        self, dataset_index: int, entry: DatasetEntry, episode: int
    ) -> dict[str, Intrinsics | None]:
        """Each camera of one episode -> its calibration, at its own resolution.

        One entry per camera, None where that camera of that episode was never
        calibrated, which is what calibration_for_views turns into a false flag
        and a row of the unknown marker. The numbers are the source's own, and
        the rescale onto the realised tile happens where the rays are built, so
        the manifest keeps stating the resolution they were measured against.
        """
        found: dict[str, Intrinsics | None] = {}
        for view in entry.views:
            if view.intrinsics.kind == "none":
                found[view.spec.key] = None
            elif view.intrinsics.kind == "constant":
                found[view.spec.key] = view.intrinsics.constant
            else:
                table = self._read_calibrations(dataset_index, entry)[view.spec.key]
                found[view.spec.key] = rays.calibration_from_row(
                    table[episode], view.intrinsics.provenance
                )
        return found

    def _read_calibrations(
        self, dataset_index: int, entry: DatasetEntry
    ) -> dict[str, np.ndarray]:
        """One directory's per-episode calibration arrays, opened on first use."""
        if dataset_index in self._calibrations:
            return self._calibrations[dataset_index]
        tables: dict[str, np.ndarray] = {}
        width = len(rays.CALIBRATION_COLUMNS)
        for view in entry.views:
            if view.intrinsics.kind != "episode":
                continue
            array = np.load(self.root / view.intrinsics.path)
            if array.shape != (entry.episodes, width):
                raise ValueError(
                    f"directory {entry.name!r} camera {view.spec.key!r} has "
                    f"calibrations of shape {tuple(array.shape)} in "
                    f"{view.intrinsics.path!r}.\n"
                    f"Expected ({entry.episodes}, {width}), one row per episode.\n"
                    "Rebuild the intrinsics part of the cache."
                )
            tables[view.spec.key] = array
        self._calibrations[dataset_index] = tables
        return tables

    def _prompt_records(self, dataset_index: int, entry: DatasetEntry) -> tuple[dict, ...]:
        """One directory's v6 prompt records, read once and checked against the manifest.

        The records are keyed by kept position, which is what the draw hands
        over. Each states the text and its head, task and tail, the key and the
        frequency it was rendered for, and the source episode it came from, so
        a record written for another episode or another key is refused (R16)
        rather than trained under this one's name.
        """
        if dataset_index in self._prompts:
            return self._prompts[dataset_index]
        path = self.root / self.manifest.prompt_cache / f"{entry.name}.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
        if len(records) != entry.episodes:
            raise ValueError(
                f"directory {entry.name!r} has {entry.episodes} episodes and "
                f"{len(records)} rendered prompts in {path}.\n"
                "Expected one prompt per kept episode, in kept order.\n"
                "Re-render the prompt cache from this manifest."
            )
        ids = None if entry.embodiment.kind == "constant" else self._episode_keys[dataset_index][1]
        for position, record in enumerate(records):
            for field in PROMPT_FIELDS:
                _require(record, field, str(path))
            if record["position"] != position:
                raise ValueError(
                    f"the prompts in {path} are not in kept order.\n"
                    "Expected position 0 to n - 1, one per kept episode.\n"
                    "Re-render the prompt cache; the draw addresses it by position."
                )
            if record["text"] != record["head"] + record["task"] + record["tail"]:
                raise ValueError(
                    f"{path} position {position}: text is not head + task + tail.\n"
                    "Expected the record's text to be the three parts it states.\n"
                    "Re-render the prompt cache with this tree's tools/build_cache.py."
                )
            key = self.embodiment_of(dataset_index, position)
            if record["embodiment"] != key or (ids is not None
                                               and int(record["episode"]) != int(ids[position])):
                raise ValueError(
                    f"{refusal_prefix(16)} {path} position {position} was rendered for key "
                    f"{record['embodiment']!r} episode {record['episode']} and the manifest "
                    f"gives key {key!r}"
                    + ("" if ids is None else f" episode {int(ids[position])}") + ".\n"
                    "Expected the prompt and the embodiment arrays to name one episode.\n"
                    "Rebuild the embodiment arrays and the prompt cache from one kept.jsonl."
                )
            if record["control_hz"] != entry.control_hz:
                raise ValueError(
                    f"{path} position {position} was rendered at {record['control_hz']} Hz and "
                    f"the manifest states {entry.control_hz}.\n"
                    "Expected the frequency line of the directory's own entry.\n"
                    "Re-render the prompt cache from this manifest."
                )
            if int(record["tokens"]) > self.max_text_tokens:
                raise ValueError(
                    f"{path} position {position} holds {record['tokens']} text tokens.\n"
                    f"Expected at most max_text_tokens {self.max_text_tokens}; v6 drops nothing.\n"
                    "Shorten the instruction at the source, or leave the episode out."
                )
        self._prompts[dataset_index] = tuple(records)
        return self._prompts[dataset_index]

    def _prompt(
        self, dataset_index: int, entry: DatasetEntry, episode: int, *, camera_frame: bool,
        start: int = 0, wrap: int = 0,
    ) -> str:
        """The rendered text of one kept episode, from the cache stage's file.

        The camera clause names no camera, because the reference is drawn per
        sample and text cached per episode cannot name it. The robot clause
        does state the base frame, so this raises when a sample with any slot
        in a camera frame arrives under it: a sentence saying the wrong frame
        is one no mask retracts.

        A directory with a label pool (label_pools) trains the cached head and
        tail around the pool's phrasing for this draw: row start of the
        episode on wrap wrap of its directory reads labels[(start + wrap x
        rows) mod n] (_pool_label), so every row of a pass and every pass shows
        another phrasing until the pool is spent.
        """
        records = self._prompt_records(dataset_index, entry)
        if dataset_index not in self._identified:
            self._check_rows_identity(dataset_index, entry, records)
            self._identified.add(dataset_index)
        record = records[episode]
        if camera_frame and record["frame_ref"] == "robot":
            raise ValueError(
                f"directory {entry.name!r} episode {episode} drew a camera frame "
                "and its prompt was rendered for robot.\n"
                "Expected a camera clause; the robot one states the base frame in words.\n"
                "Re-render the prompt cache from this manifest's extrinsics."
            )
        if dataset_index in self._label_pools:
            label = self._pool_label(dataset_index, entry, episode, record, start, wrap)
            return str(record["head"]) + label + str(record["tail"])
        return str(record["text"])

    def _check_label_pools(self, config: dict | None) -> dict[int, tuple[Path, np.ndarray]]:
        """The label pools a group states: {index: (file, offsets)}.

        config is None (no pool) or {dir, sources}: every directory of a listed
        source trains its instruction from <root>/<dir>/<name>.jsonl, one line
        per kept position (LABEL_POOL_FIELDS), and <name>.offsets.npy, the
        episodes + 1 byte offsets of its lines ending at the file's size. Both
        are checked here, before any draw; a listed source with no directory,
        or a directory with no pool, is refused rather than trained on its
        cached instruction (an empty search is not absence). Each line is held
        to its position when a draw reads it (_pool_label).
        """
        if config is None:
            return {}
        # hydra hands the group's block over as a DictConfig / ListConfig: read it as plain values.
        if isinstance(config, Mapping):
            config = {str(key): value for key, value in config.items()}
        if not isinstance(config, dict) or sorted(config) != sorted(LABEL_POOL_KEYS):
            raise ValueError(
                f"data.train.label_pools is {config!r}.\n"
                f"Expected null or a mapping of exactly {LABEL_POOL_KEYS}.\n"
                "State the pools' directory under the root and the sources that train from them."
            )
        folder, sources = config["dir"], config["sources"]
        if isinstance(sources, Sequence) and not isinstance(sources, str):
            sources = list(sources)
        if not isinstance(folder, str) or not folder or Path(folder).is_absolute() \
                or ".." in Path(folder).parts:
            raise ValueError(
                f"data.train.label_pools.dir is {folder!r}.\n"
                "Expected a directory relative to the data root, inside it.\n"
                "Write the pools under the root, as the manifest's other files."
            )
        if isinstance(sources, str) or not isinstance(sources, Sequence) or not sources \
                or any(not isinstance(name, str) or not name for name in sources):
            raise ValueError(
                f"data.train.label_pools.sources is {sources!r}.\n"
                "Expected a non-empty list of manifest sources.\n"
                "Name the sources whose directories train from a label pool."
            )
        pools: dict[int, tuple[Path, np.ndarray]] = {}
        for source in sources:
            listed = [index for index, entry in enumerate(self.manifest.entries)
                      if entry.source == source]
            if not listed:
                raise ValueError(
                    f"data.train.label_pools names source {source!r} and the manifest holds no "
                    "directory of it.\n"
                    "Expected every listed source to train from its pools.\n"
                    "Drop the source from label_pools, or build the corpus with its directories."
                )
            for index in listed:
                entry = self.manifest.entries[index]
                path = self.root / folder / f"{entry.name}.jsonl"
                offsets_path = self.root / folder / f"{entry.name}.offsets.npy"
                if not path.is_file() or not offsets_path.is_file():
                    raise ValueError(
                        f"directory {entry.name!r} of source {source!r} has no label pool: "
                        f"{path} or {offsets_path} is absent.\n"
                        "Expected the pool and its offsets for every directory of a listed "
                        "source.\n"
                        "Run the label builder for this directory, or drop the source."
                    )
                offsets = np.asarray(np.load(offsets_path), dtype=np.int64)
                size = path.stat().st_size
                if offsets.shape != (entry.episodes + 1,) or int(offsets[0]) != 0 \
                        or int(offsets[-1]) != size or bool(np.any(np.diff(offsets) <= 0)):
                    raise ValueError(
                        f"directory {entry.name!r} has {entry.episodes} episodes and the label "
                        f"pool offsets {offsets_path} hold shape {tuple(offsets.shape)}, first "
                        f"{int(offsets[0]) if offsets.size else None}, last "
                        f"{int(offsets[-1]) if offsets.size else None} for a file of {size} B.\n"
                        "Expected episodes + 1 rising offsets from 0 to the file's size, one "
                        "line per kept position.\n"
                        "Rebuild the pool and its offsets from this manifest's kept order."
                    )
                pools[index] = (path, offsets)
        logger.info("label pools: %d directories of %s train their instruction pass-indexed "
                    "from %s", len(pools), ",".join(sources), folder)
        return pools

    def _pool_label(self, dataset_index: int, entry: DatasetEntry, episode: int, record: dict,
                    start: int, wrap: int) -> str:
        """The phrasing one draw trains: labels[(start + wrap x rows) mod n] of its episode.

        rows is the episode's row count, so the rows of one wrap read
        consecutive labels and the next wrap goes on where it stopped. The
        line is read by its offsets and held to the draw: its position, the
        source episode the prompt cache names (R16), distinct non-empty
        labels with no trailing whitespace (the encoder refuses a prompt
        ending in it), and a token count per label within max_text_tokens.
        """
        path, offsets = self._label_pools[dataset_index]
        begin, end = int(offsets[episode]), int(offsets[episode + 1])
        with path.open("rb") as handle:
            handle.seek(begin)
            line = json.loads(handle.read(end - begin).decode("utf-8"))
        where = f"{path} position {episode}"
        if not isinstance(line, dict):
            raise ValueError(f"{where} is not a JSON object.\n"
                             "Expected one object per kept position.\n"
                             "Rebuild the pool and its offsets.")
        for field in LABEL_POOL_FIELDS:
            _require(line, field, where)
        if int(line["position"]) != int(episode) \
                or int(line["episode"]) != int(record["episode"]):
            raise ValueError(
                f"{refusal_prefix(16)} {where} states position {line['position']} source "
                f"episode {line['episode']}; the prompt cache gives source episode "
                f"{record['episode']} at position {episode}.\n"
                "Expected the pool and the prompt cache to name one episode at every position.\n"
                "Rebuild the pool from this manifest's prompt cache."
            )
        labels, tokens = line["labels"], line["tokens"]
        if not isinstance(labels, list) or not labels or not isinstance(tokens, list) \
                or len(tokens) != len(labels) or len(set(map(str, labels))) != len(labels):
            raise ValueError(
                f"{where} holds {len(labels) if isinstance(labels, list) else labels!r} labels "
                f"and {len(tokens) if isinstance(tokens, list) else tokens!r} token counts.\n"
                "Expected a non-empty list of distinct labels and one token count each.\n"
                "Rebuild the pool."
            )
        rows = int(self._episode_starts[dataset_index][episode + 1]
                   - self._episode_starts[dataset_index][episode])
        pick = (int(start) + int(wrap) * rows) % len(labels)
        label, count = labels[pick], tokens[pick]
        if not isinstance(label, str) or not label.strip() or label != label.rstrip():
            raise ValueError(
                f"{where} label {pick} is {label!r}.\n"
                "Expected a non-empty phrasing with no trailing whitespace: the prompt ends with "
                "it and the encoder refuses one ending in whitespace.\n"
                "Strip the label in the pool."
            )
        if isinstance(count, bool) or not isinstance(count, int) \
                or not 1 <= count <= self.max_text_tokens:
            raise ValueError(
                f"{where} label {pick} renders to {count!r} text tokens.\n"
                f"Expected 1 to max_text_tokens {self.max_text_tokens}; v6 drops nothing.\n"
                "Shorten the label, or leave it out of the pool."
            )
        return label

    def _check_rows_identity(self, dataset_index: int, entry: DatasetEntry,
                             records: tuple[dict, ...]) -> None:
        """R16 for every directory: each prompt names the source episode its rows hold.

        A constant-key directory (LIBERO, its test-phase leg) stores no ids
        array, so the arrow's own episode_index is the one witness of which
        source episode a kept position holds. Run once per directory, at its
        first sample, when its reader is open anyway.
        """
        held = self._reader(dataset_index).episode_ids()
        for position, record in enumerate(records):
            if int(record["episode"]) != int(held[position]):
                raise ValueError(
                    f"{refusal_prefix(16)} directory {entry.name!r} position {position} was "
                    f"rendered for source episode {record['episode']} and the cache's rows at "
                    f"that position are source episode {int(held[position])}.\n"
                    "Expected the prompt and the rows to name one episode at every kept "
                    "position.\n"
                    "Rebuild the arrow cache and the prompt cache from one kept.jsonl."
                )

    def _read_episode_starts(self, entry: DatasetEntry) -> torch.Tensor:
        array = np.load(self.root / entry.episode_starts_path)
        starts = torch.from_numpy(np.ascontiguousarray(array)).to(torch.int64)
        if starts.ndim != 1 or starts.numel() != entry.episodes + 1:
            raise ValueError(
                f"directory {entry.name!r} declares {entry.episodes} episodes and "
                f"its episode_starts holds {tuple(starts.shape)}.\n"
                "Expected a one-dimensional array of episodes + 1 boundaries.\n"
                "Rebuild episode_starts for this directory."
            )
        if int(starts[-1]) != entry.rows:
            raise ValueError(
                f"directory {entry.name!r} declares {entry.rows} rows and its "
                f"episode_starts ends at {int(starts[-1])}.\n"
                "Expected the last boundary to be the row count.\n"
                "Rebuild episode_starts and the entry's row count together."
            )
        return starts

    def _read_trajectory_weights(
        self, dataset_index: int, entry: DatasetEntry
    ) -> TrajectoryWeights | None:
        if entry.trajectory_weights_path is None:
            return None
        array = np.load(self.root / entry.trajectory_weights_path)
        weights = torch.from_numpy(np.ascontiguousarray(array)).to(torch.float64)
        if weights.ndim != 1 or weights.numel() != entry.episodes:
            raise ValueError(
                f"directory {entry.name!r} has {entry.episodes} episodes and "
                f"{tuple(weights.shape)} trajectory weights.\n"
                "Expected one weight per episode.\n"
                "Rebuild the weights, or drop trajectory_weights from this entry."
            )
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError(
                f"directory {entry.name!r} has a trajectory weight that is negative "
                "or not finite.\n"
                "Expected every weight to be finite and at or above 0.\n"
                "Rebuild the weights for this directory."
            )
        if float(weights.sum()) <= 0.0:
            raise ValueError(
                f"directory {entry.name!r} has trajectory weights summing to "
                f"{float(weights.sum())}.\n"
                "Expected a positive total; no episode of it could be drawn.\n"
                "Rebuild the weights, or drop trajectory_weights from this entry."
            )
        return TrajectoryWeights(dataset_index, torch.cumsum(weights, dim=0))

    def _check_canvas(self) -> None:
        if self.pixels_per_token != self.manifest.token:
            raise ValueError(
                f"the run reads {self.pixels_per_token} pixels per token and the "
                f"manifest snapped its grids against {self.manifest.token}.\n"
                "Expected the two to be equal.\n"
                "Set data.train.pixels_per_token to the manifest's canvas token size."
            )
        wanted = (
            self.max_grid[0] * self.manifest.token,
            self.max_grid[1] * self.manifest.token,
        )
        if wanted != self.manifest.cell:
            raise ValueError(
                f"the model's canvas is {wanted} pixels and the manifest's is "
                f"{self.manifest.cell}.\n"
                "Expected the same canvas; a grid is not resnapped to fit another one.\n"
                "Rebuild the manifest against this canvas, or use the model it was built for."
            )
        # The server snaps every live frame against the canvas's short side,
        # min(max_grid) * token (server.canvas_of). A corpus snapped against
        # another one would train each view on one token grid and serve it on
        # another, and nothing downstream compares the two.
        served = min(self.max_grid) * self.manifest.token
        if self.manifest.short_side != served:
            raise ValueError(
                f"the manifest snapped its grids against short side "
                f"{self.manifest.short_side} and the model's canvas has short side {served} "
                f"(min(max_grid) {min(self.max_grid)} x token {self.manifest.token}).\n"
                "Expected the two to be equal: the server snaps each frame against the "
                "canvas's short side, so another one trains a grid that is never served.\n"
                f"Rebuild the manifest with --canvas-short-side {served}, or use the model "
                "it was built for."
            )
        for entry in self.manifest.entries:
            for view in entry.views:
                if view.grid[0] > self.max_grid[0] or view.grid[1] > self.max_grid[1]:
                    raise ValueError(
                        f"directory {entry.name!r} view {view.spec.key!r} snapped to "
                        f"grid {view.grid} and the model's is {self.max_grid}.\n"
                        "Expected a grid inside the model's.\n"
                        "Rebuild the manifest against this canvas."
                    )

    def _check_slots(self) -> None:
        for entry in self.manifest.entries:
            for view in entry.views:
                if view.spec.slot >= self.n_views:
                    raise ValueError(
                        f"directory {entry.name!r} view {view.spec.key!r} takes camera "
                        f"slot {view.spec.slot} of {self.n_views}.\n"
                        "Expected a slot below the model's n_views.\n"
                        "Raise model.prefix.n_views, or renumber the view in the manifest."
                    )
            for arm in entry.arms:
                if arm.spec.slot >= self.n_slots:
                    raise ValueError(
                        f"directory {entry.name!r} arm {arm.spec.name!r} takes action "
                        f"slot {arm.spec.slot} of {self.n_slots}.\n"
                        "Expected a slot below the model's n_slots.\n"
                        "Raise model.action.n_slots, or renumber the arm in the manifest."
                    )

    def _check_action_columns(self) -> None:
        if self.action_dim != ACTION_DIM:
            old = " (the v6 row, ACTION_RECORD replaced it)" if self.action_dim in OLD_ACTION_DIMS \
                else ""
            raise ValueError(
                f"the model's action_dim is {self.action_dim}{old} and this loader packs "
                f"{ACTION_DIM} columns ({ACTION_RECORD}).\n"
                "Expected the same width; the columns are fixed once pretraining starts.\n"
                "Set model.action.action_dim to the packer's width."
            )
        check_gripper_columns(self.gripper_dims)

    def _check_prompt_stamp(self) -> None:
        stamp = self.manifest.prompt_stamp
        mismatched = [
            ("tokenizer_id", stamp.tokenizer_id, self.tokenizer_id),
            ("max_text_tokens", stamp.max_text_tokens, self.max_text_tokens),
            ("n_views", stamp.n_views, self.n_views),
        ]
        for field, found, wanted in mismatched:
            if found != wanted:
                raise ValueError(
                    f"the prompt cache was stamped with {field} {found!r} and the run "
                    f"asks for {wanted!r}.\n"
                    "Expected the two to agree; the cached text was fitted to the stamp.\n"
                    "Re-render the prompt cache, or run with the settings it was rendered for."
                )

    def _check_intrinsics_mode(self) -> None:
        """The mode the model declares against the one the corpus was built for.

        tokens is the one mode; a removed one is refused by name, and so is a
        corpus stamped for one, since its prefix layout is not this tree's.
        """
        rays.check_intrinsics_mode(self.intrinsics_mode, "model.prefix.intrinsics_mode")
        stamped = self.manifest.prompt_stamp.intrinsics_mode
        rays.check_intrinsics_mode(stamped, "the corpus's prompt stamp intrinsics_mode")
        if stamped != self.intrinsics_mode:
            raise ValueError(
                f"the model asks for intrinsics_mode {self.intrinsics_mode!r} and the "
                f"corpus was stamped for {stamped!r}.\n"
                "Expected the two to agree; each mode is its own prefix layout and its "
                "own pretrain.\n"
                "Set model.prefix.intrinsics_mode to the stamped mode, or rebuild the "
                "manifest for this one."
            )


def _read_entry(
    raw: dict, *, where: str, short_side: int, token: int, cell: tuple[int, int],
    pin_keeps_other_poses: bool = False,
) -> DatasetEntry:
    """One manifest entry -> a DatasetEntry, or a raise naming the directory."""
    name = str(_require(raw, "name", where))
    source = str(_require(raw, "source", where))
    stated_id = int(_require(raw, "source_id", where))
    if stated_id != source_id(source):
        raise ValueError(
            f"directory {name!r} states source {source!r} with id {stated_id}.\n"
            f"Expected id {source_id(source)}, the permanent number for that source.\n"
            "Rebuild the manifest; source ids come from registry.SOURCE_IDS."
        )
    weight = _require(raw, "weight", where)
    if weight is None or not np.isfinite(float(weight)) or float(weight) < 0.0:
        raise ValueError(
            f"directory {name!r} has weight {weight!r}.\n"
            "Expected a finite weight at or above 0; there is no default weight.\n"
            "Give it a weight in the weights stage; 0 removes it from the draw."
        )
    episodes = _positive(raw, "episodes", where)
    rows = _positive(raw, "rows", where)
    frame_ref = str(_require(raw, "frame_ref", where))
    try:
        frame_kind, frame_slot = conditioning.parse_frame_ref(frame_ref)
    except ValueError as exc:
        raise ValueError(
            f"directory {name!r} states frame_ref {frame_ref!r}.\n"
            "Expected 'robot', 'camera', or 'camera:N' pinning camera slot N.\n"
            "There is no third frame; fix the entry."
        ) from exc
    views = _read_views(raw, name=name, where=where, short_side=short_side, token=token,
                        cell=cell)
    arms = _read_arms(raw, name=name, source=source, where=where)
    block = _require(raw, "action", where)
    action = _read_action(block, name=name, arms=arms, where=where)
    body = _read_body(_require(raw, "body", where), name)
    columns = _require(raw, "columns", where)
    state_column = str(_require(columns, "state", where))
    stored = _read_stored(columns, _require(raw, "sidecar", where), name=name, arms=arms,
                          body=body, where=where)
    gripper = _read_action_gripper(_require(block, "gripper", where), name, where)
    rotation_source = _read_rotation_source(
        _require(block, "rotation_source", where), name=name, policy=action.rotation,
        arms=arms, action=action, state_column=state_column, where=where,
    )
    spec = _read_embodiment(_require(raw, "embodiment", where), name, where)
    control_hz = _require(raw, "control_hz", where)
    if control_hz is not None and (isinstance(control_hz, bool)
                                   or not math.isfinite(float(control_hz))
                                   or float(control_hz) <= 0.0):
        raise ValueError(
            f"directory {name!r} states control_hz {control_hz!r}.\n"
            "Expected a finite rate above 0, or null for a directory whose rate is unknown.\n"
            "Declare it with tools/declare_frequency.py."
        )
    blocked = _require(_require(raw, "history", where), "blocked", where)
    if not isinstance(blocked, list) or any(
            isinstance(one, bool) or not isinstance(one, int) for one in blocked):
        raise ValueError(
            f"directory {name!r} states history.blocked {blocked!r}.\n"
            "Expected a list of kept positions, [] where no episode is blocked.\n"
            "Rebuild the manifest with this tree's manifest stage."
        )
    shared_base = bool(raw.get("common_base_frame", False))
    held = {arm.spec.slot for arm in arms}
    for view in views:
        if view.spec.arm_slot is not None and view.spec.arm_slot not in held:
            raise ValueError(
                f"directory {name!r} mounts camera {view.spec.key!r} on action slot "
                f"{view.spec.arm_slot}, which it does not have.\n"
                f"Expected one of {sorted(held)}.\n"
                "Fix the view's arm_slot, or drop it for a camera on no arm."
            )
        if view.extrinsics.kind == MOUNTED and view.spec.arm_slot is None:
            raise ValueError(
                f"directory {name!r} camera {view.spec.key!r} states extrinsics kind "
                f"{MOUNTED!r} and no arm_slot.\n"
                "Expected the arm its offset is measured against; the pose is composed "
                "from that arm.\n"
                "Give the view an arm_slot, or state a kind that holds a base-frame pose."
            )
    if frame_kind == "camera":
        _check_camera_mode(name, views, action, arms, shared_base, pinned=frame_slot,
                           pin_keeps_other_poses=pin_keeps_other_poses)
    cache = _require(raw, "cache", where)
    return DatasetEntry(
        name=name,
        source=source,
        source_id=stated_id,
        path=str(_require(raw, "path", where)),
        episodes=episodes,
        rows=rows,
        views=views,
        arms=arms,
        action=action,
        state_column=state_column,
        action_column=str(_require(columns, "action", where)),
        valid_column=_optional_string(_require(block, "valid_column", where)),
        frame_ref=frame_ref,
        pin_keeps_other_poses=pin_keeps_other_poses,
        # Absent is "not stated", which keeps the shared strategy away from a
        # directory whose arms would share a rotation nobody checked applies.
        common_base_frame=shared_base,
        weight=float(weight),
        episode_starts_path=str(_require(raw, "episode_starts", where)),
        trajectory_weights_path=_optional_string(raw.get("trajectory_weights")),
        arrow_path=str(_require(cache, "arrow", where)),
        frames_path=str(_require(cache, "frames", where)),
        embodiment=spec,
        control_hz=None if control_hz is None else float(control_hz),
        control_hz_method=str(_require(raw, "control_hz_method", where)),
        control_hz_evidence=str(_require(raw, "control_hz_evidence", where)),
        gripper=gripper,
        rotation_delta_frame=str(block["rotation_delta_frame"]),
        rotation_source=rotation_source,
        history_blocked=tuple(int(one) for one in blocked),
        client_record=embodiment.record_from_manifest(raw),
        body=body,
        stored=stored,
    )


def _read_body(raw: dict | None, name: str) -> BodyLayout | None:
    """The entry's body block: None for a key with no body, else where each part reads."""
    if raw is None:
        return None
    if not isinstance(raw, dict) or sorted(raw) != ["base", "head", "torso"]:
        raise ValueError(
            f"directory {name!r} states body {raw!r}.\n"
            "Expected null for a fixed base, or exactly {base, torso, head}: the base's "
            "source and the torso and head joint counts.\n"
            "Rebuild the manifest with this tree's manifest stage."
        )
    try:
        return BodyLayout(base=str(raw["base"]), torso=raw["torso"], head=raw["head"])
    except ValueError as error:
        raise ValueError(f"directory {name!r}: {error}") from error


def _read_stored(columns: dict, sidecar: dict | None, *, name: str, arms: tuple,
                 body: BodyLayout | None, where: str) -> StoredParts | None:
    """The stored v7 columns the entry's layouts read, and the writer that stamped them.

    The columns block names each part column exactly when a layout reads it,
    in the one spelling every writer uses (neo_vla.constants COLUMN_*), and
    null otherwise; a command column exactly when a target names it. sidecar
    is null for an ingest output, or the stage-1 sidecar the cache stage
    merged, with the digests the reader holds the merged stamp to.
    """
    reads = {
        "joints": any(arm.state is not None and arm.state.joints is not None for arm in arms),
        "hand": any(arm.state is not None and arm.state.hand is not None for arm in arms),
        "joints_command": any(
            arm.state is not None and arm.state.joints is not None
            and arm.state.joints.target.startswith(TARGET_COMMAND_PREFIX) for arm in arms),
        "body": body is not None,
        "body_command": body is not None and body.base == "command",
    }
    widths = {}
    for field, (spelling, block) in COLUMN_FIELDS.items():
        stated = _require(columns, field, where)
        if reads[field] and stated != spelling:
            raise ValueError(
                f"directory {name!r} reads {field} and its columns block names {stated!r}.\n"
                f"Expected {spelling!r}, the one spelling of that stored v7 column.\n"
                "Rebuild the manifest with this tree's manifest stage."
            )
        if not reads[field] and stated is not None:
            raise ValueError(
                f"directory {name!r} names the column {stated!r} for {field} and no layout "
                "reads it.\n"
                f"Expected null: a stored {field} column nothing reads states nothing.\n"
                "Rebuild the manifest with this tree's manifest stage."
            )
        if reads[field]:
            widths[spelling] = block * (len(arms) if field in ARM_COLUMNS else 1)
    if sidecar is not None:
        missing = [key for key in ("path", *SIDECAR_PINS) if key not in sidecar]
        if not widths or missing:
            raise ValueError(
                f"directory {name!r} states sidecar {sidecar!r}"
                + (f" with no {missing}" if missing else " and reads no stored column") + ".\n"
                f"Expected a sidecar only where a layout reads its columns, with its path and "
                f"{list(SIDECAR_PINS)}.\n"
                "Rebuild the manifest with this tree's manifest stage."
            )
    if not widths:
        return None
    pins = None if sidecar is None else {pin: str(sidecar[pin]) for pin in SIDECAR_PINS}
    return StoredParts(columns=widths, origin="ingest" if sidecar is None else "sidecar",
                       entry=name, pins=pins)


def _read_part(raw: dict | None, field: str, name: str, arm: str,
               where: str) -> PartLayout | None:
    """One arm's joints or hand block: null, or its count, unit factor and target rule."""
    if raw is None:
        return None
    if not isinstance(raw, dict) or sorted(raw) != ["count", "target", "unit_factor"]:
        raise ValueError(
            f"directory {name!r} arm {arm!r} states {field} {raw!r}.\n"
            "Expected null, or exactly {count, unit_factor, target}.\n"
            "Rebuild the manifest with this tree's manifest stage."
        )
    return PartLayout(count=_require(raw, "count", where),
                      unit_factor=_require(raw, "unit_factor", where),
                      target=str(_require(raw, "target", where)))


def _read_embodiment(raw: dict, name: str, where: str) -> EmbodimentSpec:
    """The entry's embodiment block: one key, or a per-episode key array."""
    kind = _require(raw, "kind", where) if isinstance(raw, dict) else None
    if kind not in EMBODIMENT_KINDS:
        raise ValueError(
            f"directory {name!r} states embodiment {raw!r}.\n"
            f"Expected an object of kind {EMBODIMENT_KINDS}.\n"
            "Rebuild the manifest with tools/build_embodiment.py and the manifest stage."
        )
    if kind == "constant":
        return EmbodimentSpec("constant", (str(_require(raw, "key", where)),), None, None)
    keys = tuple(str(key) for key in _require(raw, "keys", where))
    if not keys or len(set(keys)) != len(keys):
        raise ValueError(
            f"directory {name!r} states per-episode keys {list(keys)}.\n"
            "Expected a non-empty list of distinct keys.\n"
            "Rebuild the arrays with tools/build_embodiment.py."
        )
    return EmbodimentSpec("episode", keys, str(_require(raw, "path", where)),
                          str(_require(raw, "ids_path", where)))


def _read_action_gripper(raw: dict, name: str, where: str) -> ActionGripper:
    """The action gripper's polarity, refused when a trained one has none."""
    gripper = ActionGripper(
        raw_open=float(_require(raw, "raw_open", where)),
        raw_closed=float(_require(raw, "raw_closed", where)),
        method=str(_require(raw, "method", where)),
    )
    if gripper.method not in embodiment.GRIPPER_METHODS:
        raise ValueError(
            f"directory {name!r} states gripper method {gripper.method!r}.\n"
            f"Expected one of {embodiment.GRIPPER_METHODS}.\n"
            "Measure it with tools/measure_gripper_polarity.py."
        )
    if gripper.method != "masked" and (
            not math.isfinite(gripper.raw_open) or not math.isfinite(gripper.raw_closed)
            or gripper.raw_open == gripper.raw_closed):
        raise ValueError(
            f"directory {name!r} states gripper raw_open {gripper.raw_open} and raw_closed "
            f"{gripper.raw_closed}.\n"
            "Expected two different finite values; their order is the polarity.\n"
            "Measure it with tools/measure_gripper_polarity.py, or state method masked."
        )
    return gripper


def _read_rotation_source(raw: dict | None, *, name: str, policy: str,
                          arms: tuple[ArmEntry, ...], action: ActionLayout, state_column: str,
                          where: str) -> RotationSource | None:
    """Where a converted rotation reads orientation, refused when it has nowhere to read.

    body_to_spatial and euler_to_spatial read the arm's orientation at every
    step: from the stated Euler columns, or, when the source is null, from
    each arm's state layout, which then needs a pose (a gripper spec is not
    needed: v7 reads the pose and the aperture apart). Only an arm that names
    drot converts; a joints-only arm reads none. Any other policy reads none
    and states none.
    """
    converts = policy in ("body_to_spatial", "euler_to_spatial")
    if raw is None:
        blind = [arm.spec.name for arm, fields in zip(arms, action.arms)
                 if "drot" in fields and (arm.state is None or not arm.state.has_pose)]
        if converts and blind:
            raise ValueError(
                f"directory {name!r} converts its rotation ({policy}) and arm {blind[0]!r} "
                "has no readable pose.\n"
                "Expected a state layout with a pose, or an action.rotation_source.\n"
                "State the Euler columns the orientation is read from."
            )
        return None
    if not converts:
        raise ValueError(
            f"directory {name!r} states a rotation_source under policy {policy}.\n"
            "Expected null: only a converted rotation reads the arm's orientation.\n"
            "Drop the rotation_source, or fix the rotation_delta_frame."
        )
    source = RotationSource(
        column=str(_require(raw, "column", where)),
        at=int(_require(raw, "at", where)),
        euler=str(_require(raw, "euler", where)),
    )
    if source.column != state_column or source.at < 0 or source.euler not in EULER_CONVENTIONS \
            or len(arms) != 1:
        raise ValueError(
            f"directory {name!r} states rotation_source {raw!r} for {len(arms)} arm(s).\n"
            f"Expected one arm, the entry's state column {state_column!r}, a column at or "
            f"above 0 and a reading in {EULER_CONVENTIONS}.\n"
            "Fix the layout's rotation_source."
        )
    return source


def _check_rotation_source(entry: DatasetEntry, reader: EpisodeReader) -> None:
    """Refuse, when a directory's reader opens, Euler columns past its state row.

    The manifest states the columns and the cache holds the row, so the two
    meet first here; a source past the row's width would otherwise surface as
    an IndexError inside a loader worker at the directory's first draw.
    """
    source = entry.rotation_source
    if source is None:
        return
    width = int(reader.rows(0, 0, 1)[ROW_STATE].shape[1])
    if source.at + 3 > width:
        raise ValueError(
            f"directory {entry.name!r}: rotation_source reads state columns "
            f"{source.at}..{source.at + 2} and the cache's state row holds {width}.\n"
            "Expected the three Euler columns inside the state row.\n"
            "Fix the layout's rotation_source.at, or rebuild the cache."
        )


def _read_views(
    raw: dict, *, name: str, where: str, short_side: int, token: int,
    cell: tuple[int, int]
) -> tuple[ViewEntry, ...]:
    entries = []
    slots: set[int] = set()
    for view in _require(raw, "views", where):
        mounted = _require(view, "arm_slot", where)
        spec = ViewSpec(
            key=str(_require(view, "key", where)),
            role=str(_require(view, "role", where)),
            slot=int(_require(view, "slot", where)),
            prompt_noun=str(_require(view, "prompt_noun", where)),
            arm_slot=None if mounted is None else int(mounted),
        )
        if spec.slot in slots:
            raise ValueError(
                f"directory {name!r} puts two views in camera slot {spec.slot}.\n"
                "Expected one view per slot.\n"
                "Renumber the views in this entry."
            )
        slots.add(spec.slot)
        shape = _read_pair(view, "shape", name, where)
        grid = _read_pair(view, "grid", name, where)
        images.check_fits(
            images.Tile(grid[0], grid[1], grid[0] * token, grid[1] * token), cell=cell
        )
        # The grid is what this tree's rule (NEO's smart_resize) gives the shape:
        # a manifest built under another rule is refused here, before a run,
        # rather than by the first frame of the view that check_snaps_to reads.
        ruled = images.snap_grid(shape, short_side=short_side, token=token, cell=cell)
        if (ruled.grid_h, ruled.grid_w) != tuple(grid):
            raise ValueError(
                f"directory {name!r} view {spec.key!r} records grid {list(grid)} for its "
                f"{shape[0]}x{shape[1]} frames, and this tree's rule gives "
                f"[{ruled.grid_h}, {ruled.grid_w}].\n"
                "Expected the grid NEO's smart_resize gives the shape on the manifest's "
                "canvas.\nRebuild the manifest with this tree's tools/build_manifest.py."
            )
        intrinsics = _read_intrinsics(_require(view, "intrinsics", where), name, where)
        extrinsics = _read_extrinsics(
            _require(view, "extrinsics", where), name, spec.key, where
        )
        entries.append(ViewEntry(spec, shape, grid, intrinsics, extrinsics))
    if not entries:
        raise ValueError(
            f"directory {name!r} lists no views.\n"
            "Expected at least one camera.\n"
            "Add the camera to this entry, or drop the directory from the manifest."
        )
    return tuple(entries)


def _read_arms(raw: dict, *, name: str, source: str, where: str) -> tuple[ArmEntry, ...]:
    """One entry's arms, in increasing slot order.

    Arm k of the list is block k of the unified chunk and names its own action
    slot, arms[k].slot, which need not be k: a single arm may sit at slot 1.
    Two arms on one slot, or a list out of slot order, is refused rather than
    reordered, because the order is also the action layout's order.
    """
    entries = []
    previous = -1
    for position, arm in enumerate(_require(raw, "arms", where)):
        spec = ArmSpec(slot=int(_require(arm, "slot", where)),
                       name=str(_require(arm, "name", where)))
        if spec.slot <= previous:
            raise ValueError(
                f"directory {name!r} lists arm {spec.name!r} at position {position} "
                f"with slot {spec.slot}, after slot {previous}.\n"
                "Expected the arms in increasing slot order, each on its own slot.\n"
                "Reorder or renumber the arms in this entry."
            )
        previous = spec.slot
        layout = _require(arm, "state", where)
        parts = {field: _read_part(_require(arm, field, where), field, name, spec.name, where)
                 for field in PART_FIELDS}
        offset = None
        if layout is not None:
            block = _require(layout, "base_offset", where)
            if block is not None:
                if _require(block, "kind", where) != "episode":
                    raise ValueError(
                        f"directory {name!r} arm {spec.name!r} states base_offset {block!r}.\n"
                        "Expected null, or kind 'episode' with a path to [episodes, 3] metres.\n"
                        "Rebuild the offsets with the LIBERO base-offset measurement."
                    )
                offset = str(_require(block, "path", where))
        try:
            state = _read_state_layout(layout, spec, source, where, **parts)
        except ValueError as error:
            raise ValueError(f"directory {name!r}: {error}") from error
        entries.append(ArmEntry(spec, state, offset))
    if not entries:
        raise ValueError(
            f"directory {name!r} lists no arms.\n"
            "Expected at least one arm.\n"
            "Add the arm to this entry, or drop the directory from the manifest."
        )
    return tuple(entries)


def _read_state_layout(
    layout: dict | None, spec: ArmSpec, source: str, where: str, *,
    joints: PartLayout | None = None, hand: PartLayout | None = None,
) -> ArmStateLayout | None:
    """One arm's state layout, or None where nothing of it was ever established.

    The pose fields are all stated or all null (a joints-only arm, BimanualYAM
    or Tien Kung); the joints and hand blocks sit beside the state block. An
    arm with parts and no state block reads its parts alone, and says so in
    its provenance. Checked here, so a malformed layout is refused at load.
    """
    if layout is None and joints is None and hand is None:
        return None

    def optional(field: str, kind):
        value = None if layout is None else _require(layout, field, where)
        return None if value is None else kind(value)

    gripper = None if layout is None else layout.get("gripper")
    built = ArmStateLayout(
        source=source,
        arm=spec.name,
        position_at=optional("position_at", int),
        rotation_at=optional("rotation_at", int),
        rotation_kind=optional("rotation_kind", str),
        position_unit=optional("position_unit", str),
        gripper=None if gripper is None else GripperSpec(
            column=int(_require(gripper, "column", where)),
            raw_open=float(_require(gripper, "raw_open", where)),
            raw_closed=float(_require(gripper, "raw_closed", where)),
        ),
        provenance=PARTS_ONLY if layout is None else str(_require(layout, "provenance", where)),
        joints=joints,
        hand=hand,
    )
    # The stored row's width is held to the layout when a row is read; here
    # every other rule is, so no row width is claimed.
    check_layout(built, raw_width=UNREAD_WIDTH)
    return built


def _read_action(block: dict, *, name: str, arms: tuple, where: str) -> ActionLayout:
    """One entry's action block: the layout and the policy its rotation convention gives.

    Each arm's joints and hand targets come from the same PartLayout its state
    reads (the manifest states one block per arm and field), so a state and a
    target cannot read one stored column two ways. A unit factor no arm reads
    is null.
    """
    for field, became in RETIRED_ACTION_FIELDS.items():
        if field in block:
            raise ValueError(
                f"directory {name!r} states action.{field}, a manifest schema 1 field.\n"
                f"It became {became}.\n"
                "Rebuild the manifest with this tree's manifest stage."
            )
    frame = str(_require(block, "rotation_delta_frame", where))
    if frame not in ROTATION_CHANNEL_POLICY:
        raise ValueError(
            f"directory {name!r} states rotation_delta_frame {frame!r}.\n"
            f"Expected one of {tuple(ROTATION_CHANNEL_POLICY)}.\n"
            "Measure the source's rotation convention, or state unresolved."
        )
    rotation = str(_require(block, "rotation", where))
    if rotation in RETIRED_POLICIES:
        raise ValueError(
            f"directory {name!r} states rotation policy {rotation!r}, which this tree retired.\n"
            f"It became {RETIRED_POLICIES[rotation]}.\n"
            "Rebuild the manifest with this tree's manifest stage."
        )
    gives = ROTATION_CHANNEL_POLICY[frame]
    if gives != rotation:
        raise ValueError(
            f"directory {name!r} pairs rotation_delta_frame {frame!r} with policy "
            f"{rotation!r}.\nExpected policy {gives!r}.\n"
            "Rebuild the manifest with this tree's manifest stage."
        )
    layouts = tuple(
        {field: tuple(int(column) for column in columns)
         for field, columns in arm.items()}
        for arm in _require(block, "arms", where)
    )
    if len(layouts) != len(arms):
        raise ValueError(
            f"directory {name!r} has {len(arms)} arms and {len(layouts)} action layouts.\n"
            "Expected one action layout per arm, in slot order.\n"
            "Fix the action block in this entry."
        )
    # Every layout in this tree writes null here, which is the layout table
    # saying the directory has no base. A stated one is a directory whose planar
    # columns had a slot of their own, and there is none now: slot 0 is the first
    # arm, so those columns would land on it.
    if block.get("base") is not None:
        raise ValueError(
            f"directory {name!r} states a mobile base block.\n"
            "Expected null: every action slot is an arm and slot 0 is the first "
            "of them, so the base columns have nowhere to land.\n"
            "Drop the base from the layout, or leave the directory out."
        )
    translation = _require(block, "translation_cm_per_unit", where)
    turned = _require(block, "rotation_rad_per_unit", where)
    parts = {field: tuple(None if arm.state is None else arm.state.part(field) for arm in arms)
             for field in PART_FIELDS}
    try:
        return ActionLayout(
            arms=layouts,
            flat_dim=_positive(block, "flat_dim", where),
            rotation=rotation,
            translation_cm_per_unit=None if translation is None
            else tuple(float(value) for value in translation),
            rotation_rad_per_unit=None if turned is None else float(turned),
            **{field: stated if any(part is not None for part in stated) else ()
               for field, stated in parts.items()},
        )
    except ValueError as error:
        raise ValueError(f"directory {name!r}: {error}") from error


def _check_camera_mode(
    name: str,
    views: tuple[ViewEntry, ...],
    action: ActionLayout,
    arms: tuple[ArmEntry, ...],
    common_base_frame: bool,
    pinned: int | None = None,
    pin_keeps_other_poses: bool = False,
) -> None:
    """Refuse a camera-frame directory that cannot say what the camera frame is."""
    eligible = {view.key for view in reference_views(
        tuple(view.spec for view in views)
    )}
    cameras = tuple(
        ReferenceCamera(view.spec.key, view.spec.slot, view.spec.arm_slot)
        for view in views
        if view.spec.key in eligible and view.extrinsics.kind != "none"
    )
    if not cameras:
        raise ValueError(
            f"directory {name!r} states frame_ref camera and has no camera with a pose.\n"
            "Expected a view whose role is in registry.ACTION_FRAME_ROLES and whose "
            "extrinsics kind is not none.\n"
            "State frame_ref robot, or give a camera its pose."
        )
    if pinned is not None:
        # Legacy pins require one posed view. Opt-in PINPOSE restricts only the reference
        # pool; every other view keeps its geometry in the prefix.
        slots = sorted(camera.slot for camera in cameras)
        invalid = pinned not in slots if pin_keeps_other_poses else slots != [pinned]
        if invalid:
            raise ValueError(
                f"directory {name!r} states frame_ref camera:{pinned} and its reference pool "
                f"is {slots}.\n"
                "Expected exactly that one slot to carry a pose, so the draw cannot pick "
                "another camera than the one eval is told to supply.\n"
                "Set the other cameras' extrinsics kind to none, or state plain 'camera'."
            )
        if pin_keeps_other_poses:
            cameras = tuple(camera for camera in cameras if camera.slot == pinned)
    arm_slots = tuple(arm.spec.slot for arm in arms)
    if not strategies_for(cameras, arm_slots, common_base_frame=common_base_frame):
        raise ValueError(
            f"directory {name!r} states frame_ref camera and offers no way to pair "
            f"its {len(arm_slots)} arms with its cameras.\n"
            "Expected common_base_frame true for a shared reference, or a camera "
            "mounted on every arm.\n"
            "State one of them, or state frame_ref robot."
        )


def _read_extrinsics(
    raw: dict, name: str, key: str, where: str
) -> ExtrinsicsEntry:
    """One view's pose block, refused rather than defaulted.

    A view whose block is absent is a manifest built before the pose moved onto
    the views, not a camera with no measured pose, and the two must not read the
    same: the second is a statement and the first is a stage that never ran.
    """
    if not isinstance(raw, dict):
        raise ValueError(
            f"directory {name!r} camera {key!r} states extrinsics {raw!r}.\n"
            "Expected the pose block, which says kind 'none' for a camera with no "
            "measured pose.\n"
            "Rebuild the manifest with this tree's manifest stage."
        )
    kind = str(_require(raw, "kind", where))
    if kind not in EXTRINSICS_KINDS:
        raise ValueError(
            f"directory {name!r} camera {key!r} states extrinsics kind {kind!r}.\n"
            f"Expected one of {EXTRINSICS_KINDS}.\n"
            "Fix the view's extrinsics block."
        )
    path = _optional_string(raw.get("path"))
    if kind != "none" and not path:
        raise ValueError(
            f"directory {name!r} camera {key!r} states extrinsics kind {kind!r} "
            "with no path.\n"
            "Expected a path to the rotations, relative to the corpus root.\n"
            "Give the path, or state kind none."
        )
    return ExtrinsicsEntry(kind=kind, path=path)


def _read_intrinsics(raw: dict, name: str, where: str) -> IntrinsicsEntry:
    """One view's calibration block, refused rather than defaulted.

    A view whose block is absent is a manifest built before the intrinsics
    stage, not a view with no camera calibration, and the two must not read the
    same: the second is a statement and the first is a stage that never ran.
    """
    if not isinstance(raw, dict):
        raise ValueError(
            f"directory {name!r} states intrinsics {raw!r}.\n"
            "Expected the calibration block, which says kind 'none' when a view has "
            "no calibration.\n"
            "Rebuild the manifest with this tree's manifest stage."
        )
    kind = str(_require(raw, "kind", where))
    if kind not in rays.INTRINSICS_KINDS:
        raise ValueError(
            f"directory {name!r} states intrinsics kind {kind!r}.\n"
            f"Expected one of {rays.INTRINSICS_KINDS}.\n"
            "Rebuild the manifest with this tree's manifest stage."
        )
    if kind == "none":
        return IntrinsicsEntry(kind=kind, provenance=None, constant=None, path=None)
    provenance = str(_require(raw, "provenance", where))
    if kind == "episode":
        path = str(_require(raw, "path", where))
        rays.check_provenance(provenance)
        return IntrinsicsEntry(kind=kind, provenance=provenance, constant=None, path=path)
    constant = Intrinsics(
        fx=float(_require(raw, "fx", where)),
        fy=float(_require(raw, "fy", where)),
        cx=float(_require(raw, "cx", where)),
        cy=float(_require(raw, "cy", where)),
        width=int(_require(raw, "width", where)),
        height=int(_require(raw, "height", where)),
        provenance=provenance,
    )
    rays.check_provenance(provenance)
    return IntrinsicsEntry(kind=kind, provenance=provenance, constant=constant, path=None)


def _read_stamp(raw: dict, where: str) -> Stamp:
    stamp = Stamp(
        template_name=str(_require(raw, "template_name", where)),
        template_version=str(_require(raw, "template_version", where)),
        template_digest=str(_require(raw, "template_digest", where)),
        tokenizer_id=str(_require(raw, "tokenizer_id", where)),
        max_text_tokens=int(_require(raw, "max_text_tokens", where)),
        prefix_layout=str(_require(raw, "prefix_layout", where)),
        n_views=int(_require(raw, "n_views", where)),
        intrinsics_mode=str(_require(raw, "intrinsics_mode", where)),
    )
    if stamp.prefix_layout != PREFIX_LAYOUT:
        raise ValueError(
            f"{where} names a prompt cache stamped for prefix layout "
            f"{stamp.prefix_layout!r}.\n"
            f"Expected {PREFIX_LAYOUT!r}; another layout describes a different prefix.\n"
            "Re-render the prompt cache against this tree's template."
        )
    # And the artifact behind it. The layout above is a string a tool wrote
    # into the stamp; the version is the artifact's own, and the two are the
    # same statement. Held together because they were once written by two
    # different hands: a v4 template stamped 'v5' passed the check above and
    # reached the model carrying neither the geometry wording nor its width.
    major = stamp.template_version.split(".")[0]
    if major != PREFIX_LAYOUT.lstrip("v"):
        raise ValueError(
            f"{where} names prefix layout {PREFIX_LAYOUT!r} and template version "
            f"{stamp.template_version!r}.\n"
            f"Expected a version {PREFIX_LAYOUT.lstrip('v')} artifact; the layout and "
            "the artifact that describes it are one statement.\n"
            "Rebuild the manifest against this tree's template."
        )
    return stamp


def _read_provenance(raw: dict, where: str) -> Provenance:
    return Provenance(
        schema_version=int(_require(raw, "schema_version", where)),
        tool=str(_require(raw, "tool", where)),
        tool_version=str(_require(raw, "tool_version", where)),
        git_commit=str(_require(raw, "git_commit", where)),
        inputs={str(key): str(value)
                for key, value in _require(raw, "inputs", where).items()},
        created=str(_require(raw, "created", where)),
    )


def _check_totals(manifest: Manifest, stated: dict, where: str) -> None:
    for field, value in manifest.totals().items():
        claimed = int(_require(stated, field, where))
        if claimed != value:
            raise ValueError(
                f"{where} states {claimed} {field} and its entries hold {value}.\n"
                "Expected the totals to be the entries' own sums.\n"
                "Rebuild the manifest; the totals are written from the entries."
            )


def _read_cell(canvas: dict, token: int, where: str) -> tuple[int, int]:
    cell = _require(canvas, "cell", where)
    if len(cell) != 2 or int(cell[0]) % token or int(cell[1]) % token:
        raise ValueError(
            f"{where} states canvas cell {list(cell)} against token {token}.\n"
            "Expected two sides, each a whole number of tokens.\n"
            "Fix the canvas in the manifest stage."
        )
    return int(cell[0]), int(cell[1])


def _read_pair(raw: dict, key: str, name: str, where: str) -> tuple[int, int]:
    value = _require(raw, key, where)
    if len(value) != 2 or int(value[0]) < 1 or int(value[1]) < 1:
        raise ValueError(
            f"directory {name!r} states {key} {list(value)}.\n"
            "Expected two sides, both at least 1.\n"
            "Fix the view in this entry."
        )
    return int(value[0]), int(value[1])


def _require(raw: dict, key: str, where: str) -> Any:
    """One field of one object, or a raise naming the field and the file."""
    if key not in raw:
        raise ValueError(
            f"{where} has an object with no {key!r} field.\n"
            "Expected the field.\n"
            "Rebuild the manifest with this tree's manifest stage."
        )
    return raw[key]


def _positive(raw: dict, key: str, where: str) -> int:
    value = int(_require(raw, key, where))
    if value < 1:
        raise ValueError(
            f"{where} states {key} {value}.\n"
            "Expected a whole number at or above 1.\n"
            "Rebuild the manifest; this field counts something that exists."
        )
    return value


def _optional_string(value: Any) -> str | None:
    """A field that may be null: its text, or None where the manifest says null."""
    return None if value is None else str(value)


def _stored_rows(entry: DatasetEntry, rows: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The stored v7 columns of a window the entry's layouts read, or a refusal naming one.

    A reader that returns a window without a column the layouts read is
    refused here, never read as "not recorded": CacheReader holds each one to
    its stamp when it opens, and another EpisodeReader must return them too.
    """
    names = () if entry.stored is None else tuple(entry.stored.columns)
    missing = sorted(set(names) - set(rows))
    if missing:
        raise ValueError(
            f"directory {entry.name!r}: the reader returned no {missing} for a window, and the "
            "directory's layouts read them.\n"
            "Expected every stored v7 column the manifest names, row-aligned with the state.\n"
            "Read the directory through CacheReader with the entry's StoredParts."
        )
    return {name: rows[name] for name in names}


def _structural_dims(entry: DatasetEntry) -> torch.Tensor:
    """[arms, ARM_DIM] bool: the columns a directory's layout, rotation policy and gripper leave."""
    return structural_dims(entry.action, gripper_trained=entry.gripper.method != "masked")


def collate_samples(samples: list[dict]) -> dict:
    """A list of samples -> one batch: torch's default, and the native chat half.

    Every robot field goes through default_collate. A chat half holds rows of
    their own lengths and images of their own sizes, which cannot be stacked,
    so collate_chat pads the rows to the batch's longest and lists the images.
    """
    if not samples or CHAT_KEY not in samples[0]:
        return default_collate(samples)
    robot = [
        {key: value for key, value in sample.items() if key != CHAT_KEY}
        for sample in samples
    ]
    batch = default_collate(robot)
    batch[CHAT_KEY] = collate_chat([sample[CHAT_KEY] for sample in samples])
    return batch


class StepSamples(NamedTuple):
    """What RobotDataset reads for one step item: the step and its robot samples."""

    step: int
    samples: list[dict]


def collate_step(read: StepSamples, *, per_device: int, accumulation: int, chat,
                 rank: int | None = None, world: int | None = None) -> list[dict]:
    """One step item -> its accumulation micro-batches, the chat micro-batches with their packs.

    Micro-batch m is samples m * per_device to (m + 1) * per_device, in
    sampler order: the samples today's loader puts in micro-batch
    step * accumulation + m. Each drops its draw_index and goes through
    collate_samples, the collate of every other robot batch, so the robot
    batches are today's byte for byte. The chat micro-batches are the
    corpus's chat_pack.PackConfig.chat_micro_batches: every one, or those
    pack_micro_batches names (shape B: 3 and 7 of 8), the rest robot rows
    alone. chat.step_packs draws one pack per chat micro-batch, the k-th
    beside the k-th of them. When every micro-batch packs, the chain is the
    step's own and its key the step's first draw number modulo the chat
    corpus's draws (chat_pack.step_key). When fewer do, PackConfig.chain_steps
    consecutive chat steps share one chain of at least accumulation packs
    (shape B: 4 steps, 8 packs; the final recipe, micro-batch 7 on every 2nd
    step: 8 chat steps, 8 packs), keyed by the group and this rank
    (chat_pack.group_key, which needs rank and world), chat step c
    (PackConfig.chat_step: the global step at pack_every_steps 1) taking slot
    c % chain_steps of it. A step the chat corpus's cadence leaves out
    (PackConfig.carries, pack_every_steps) draws no pack and carries robot rows
    alone on every micro-batch.
    """
    indices = tuple(int(sample[DRAW_INDEX_KEY]) for sample in read.samples)
    split_step_item(StepItem(read.step, indices), per_device, accumulation)
    carrying = chat.pack.step_micro_batches(read.step, accumulation)
    packs = {}
    if carrying:
        steps = chat.pack.chain_steps(accumulation)
        ordinal = chat.pack.chat_step(read.step)
        key = (step_key(indices[0], len(chat)) if steps == 1
               else group_key(ordinal, steps, rank=rank, world=world))
        packs = dict(zip(carrying, chat.step_packs(key, len(carrying), step=read.step,
                                                   slot=ordinal % steps, slots=steps)))
    batches = []
    for micro in range(int(accumulation)):
        robot = [
            {key: value for key, value in sample.items() if key != DRAW_INDEX_KEY}
            for sample in read.samples[micro * per_device:(micro + 1) * per_device]
        ]
        batch = collate_samples(robot)
        if micro in packs:
            batch[CHAT_KEY] = packs[micro]
        batches.append(batch)
    return batches
