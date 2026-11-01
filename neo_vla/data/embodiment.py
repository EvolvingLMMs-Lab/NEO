"""The one normaliser (v7): bounds per (key, slot, dim), the file holding them, its identities.

embodiments.json (neo_vla_embodiments_v2) states per embodiment key the arm
joints and hand it declares; per action slot the weighted q01 and q99 of each
range column of the 26-wide row and which columns are constant or absent; the
bounds of the 29-wide state record per dimension; the shared body block (torso
and head by q01/q99, the base by a symmetric scale fitted on moving rows, so a
still base is exactly 0), or that the key has none; a frame fix and a clause.
Per directory it states the client inputs and the masks the loader trains under
and the server rebuilds. The dataset, the policy, the server, the clients and
the tools normalise through the functions here and through nothing else.

Module scope imports the standard library, numpy and neo_vla.constants only, so
an eval client in a simulator environment with no torch can read the file and
call gripper_raw and record_inputs; the tensor functions import torch inside.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import (
    ACTION_DIM,
    ACTION_DPOS,
    ACTION_DROT,
    ACTION_EE,
    ACTION_HAND,
    ACTION_JOINTS,
    ACTION_RECORD,
    BODY_BASE,
    BODY_DIM,
    BODY_HEAD,
    BODY_POSTURE,
    BODY_RECORD,
    BODY_TORSO,
    EMBODIMENTS_FILE,
    EMBODIMENTS_SCHEMA,
    GRIPPER_COLUMN,
    OLD_EMBODIMENTS_SCHEMAS,
    RANGE_COLUMNS,
    STATE_APERTURE,
    STATE_DIM,
    STATE_HAND,
    STATE_JOINTS,
    STATE_POSITION,
    STATE_RECORD,
    STATE_ROTATION,
    refusal_prefix,
)

__all__ = [
    "EMBODIMENTS_FILE", "EMBODIMENTS_SCHEMA", "TRANSFORM", "QUANTILES", "OUT_OF_BAND",
    "BASE_TRANSFORM", "RANGE_COLUMNS", "GRIPPER_COLUMN", "ACTION_WIDTH", "ARM_DIM", "MASK",
    "DIM_NAMES", "STATE_DIM_NAMES", "BODY_DIM_NAMES", "STATE_RANGE", "ACTION_GROUPS",
    "BODY_GROUPS", "FIT_FIELDS", "BASE_SOURCES", "BASE_MOTION", "BASE_SCALE_FLOOR", "Entry",
    "BodyEntry", "DatasetRecord", "Embodiments", "FitReport", "RefusalKeyError",
    "entry_identity", "identity_differs", "record_inputs", "record_from_manifest",
    "without_checksums",
    "header_of", "normalise_actions", "denormalise", "normalise_state", "canonical_gripper",
    "gripper_raw", "fit_check", "require_fit", "DROT_RECORD", "UNTRAINED_DROT",
    "normalise_body_actions", "denormalise_body", "normalise_body_state", "fit_check_body",
    "BaseScale", "weighted_quantile", "base_moving", "fit_base_scale",
    "INHERITED_GROUPS_ALL", "inherited_groups", "inherited_columns",
]


def _columns(span: slice) -> tuple[int, ...]:
    return tuple(range(span.start, span.stop))


# The one transform, quantile pair, base rule and out-of-band rule this tree
# reads. The header states each, so a file built under another one is refused
# by name rather than read as if it were this one. The v6 whole-step band is
# named, because under it a joint outlier deleted the step's EE target.
TRANSFORM = "q01q99_linear"
QUANTILES = (1, 99)
OUT_OF_BAND = "loss_mask_group"
BASE_TRANSFORM = "symmetric_q99_moving"
RETIRED_RULES = {
    "loss_mask_step": "the v6 whole-step band, under which a joint or hand outlier took the "
                      "step's EE target out of the loss",
}

# One arm's action row (constants.ACTION_*): every column but the gripper is
# range-mapped per (key, slot, column); the gripper is the canonical 0-open
# 1-closed value mapped to 2 g - 1.
ACTION_WIDTH = ACTION_DIM
ARM_DIM = ACTION_DIM
MASK = tuple(column != GRIPPER_COLUMN for column in range(ACTION_DIM))
DIM_NAMES = (
    *(f"j{joint}" for joint in range(7)), "dx", "dy", "dz", "drot_x", "drot_y", "drot_z",
    "gripper", *(f"h{joint}" for joint in range(12)),
)
STATE_DIM_NAMES = (
    *(f"j{joint}" for joint in range(7)), "px", "py", "pz", "r00", "r10", "r20", "r01", "r11",
    "r21", "aperture", *(f"h{joint}" for joint in range(12)),
)
BODY_DIM_NAMES = ("base_dx", "base_dy", "base_dyaw", "torso0", "torso1", "torso2", "head_yaw",
                  "head_pitch")

# The state dimensions mapped per (key, slot, dim) by q01/q99: joints, position
# and hand. The 6D passes through (after the frame fix) and the aperture is 2a - 1.
STATE_RANGE = _columns(STATE_JOINTS) + _columns(STATE_POSITION) + _columns(STATE_HAND)

# The band groups: a step outside the band, or holding a non-finite value,
# masks only its own group at that step. EE is the 7 columns every benchmark
# executes (the gripper's finiteness counts there). FIT_FIELDS are the range
# columns of each field the R14 fit gate measures on its own.
ACTION_GROUPS = {"joints": ACTION_JOINTS, "ee": ACTION_EE, "hand": ACTION_HAND}
BODY_GROUPS = {"base": BODY_BASE, "posture": BODY_POSTURE}
FIT_FIELDS = {"joints": _columns(ACTION_JOINTS),
              "ee": _columns(ACTION_DPOS) + _columns(ACTION_DROT),
              "hand": _columns(ACTION_HAND)}
BODY_FIT_FIELDS = {name: _columns(span) for name, span in BODY_GROUPS.items()}

# A key whose action block names a source (action.from, V6 4.7 D6(i)) copies
# that source's rows of these ACTION_GROUPS (action.from_groups) and measures
# the others on its own rows. "ee" is ACTION_EE, the seven columns every
# benchmark executes (dpos, drot and the gripper, with gripper_valid), so a
# key that copies only "ee" keeps the pretrain key's executed scale and fits
# its own joints (a benchmark that never executes joints). A file written before
# from_groups copied every group.
INHERITED_GROUPS_ALL = tuple(sorted(ACTION_GROUPS))


def inherited_groups(raw: Any, where: str) -> tuple[str, ...]:
    """A stated from_groups -> its sorted tuple; refused unless a non-empty list of
    distinct ACTION_GROUPS names. None (not stated) is every group."""
    if raw is None:
        return INHERITED_GROUPS_ALL
    if not isinstance(raw, (list, tuple)) or not raw \
            or any(not isinstance(name, str) or name not in ACTION_GROUPS for name in raw) \
            or len(set(raw)) != len(raw):
        raise ValueError(
            f"{where} states action from_groups {raw!r}.\n"
            f"Expected a non-empty list of distinct names from {list(INHERITED_GROUPS_ALL)}: "
            "the action groups copied from action.from.\n"
            "Fix action_from_groups in config/embodiments.yaml and rebuild the stats."
        )
    return tuple(sorted(raw))


def inherited_columns(groups: tuple[str, ...] | None) -> tuple[int, ...]:
    """The action columns a key copies from its source: those of the named groups, all
    ACTION_DIM of them when groups is None (a file written before from_groups)."""
    names = INHERITED_GROUPS_ALL if groups is None else groups
    return tuple(sorted(column for name in names for column in _columns(ACTION_GROUPS[name])))


# The body base. What a key reads it from (config/embodiments.yaml v2), the
# census motion rule that picks the moving rows the symmetric scale is fitted
# on (a window of 3 steps moving more than 3 mm in the plane or 3e-3 rad in
# yaw; a command source moves where its command is non-zero), and the floor
# below which a fitted scale is jitter: such a dim is constant, not fitted.
BASE_SOURCES = ("odometry", "command", "imu_yaw", "still", "none")
BASE_MOTION = {"window_steps": 3, "planar_cm": 0.3, "dyaw_rad": 0.003, "command": "nonzero"}
BASE_SCALE_FLOOR = (0.1, 0.1, 0.001)
BASE_FITTED, BASE_CONSTANT, BASE_ABSENT = "fitted", "constant", "absent"

# The units a key may declare: joints are radians; a hand is radians or the
# dimensionless unit of a source that stores it normalised (Inspire).
JOINT_UNITS = ("rad",)
HAND_UNITS = ("rad", "unit")
MAX_TORSO = BODY_TORSO.stop - BODY_TORSO.start
MAX_HEAD = BODY_HEAD.stop - BODY_HEAD.start

# How a directory's gripper polarity was established. masked is a gripper the
# key never trains, so its column is invalid and decodes to nothing.
GRIPPER_METHODS = ("measured", "declared", "masked")

# The unnumbered refusal of CX 11: a range dimension with no finite, ordered
# bound pair maps every value to an edge, and a per-sample raise at train time
# would kill the run, so the file is refused at load instead.
DEGENERATE_RANGE = "DEGENERATE_RANGE"

# The fit gate (R14), per field: every valid range dimension's p99 |y| inside
# this band, and at most this share of the field's rows outside the alpha band.
FIT_P99_BAND = (0.5, 2.0)
FIT_FLAGGED_MAX = 0.01

# How far a stated end-effector frame fix may sit from a proper rotation.
FRAME_FIX_TOLERANCE = 1e-6

# The six datasets-record fields a client recomputes from its own manifest, and
# so the only ones a record's inputs compare.
RECORD_FIELDS = ("embodiment", "keys", "slots", "gripper", "unit_factors", "control_hz")
PER_EPISODE = "per_episode"

# The per-directory masks a datasets record carries beside its client inputs
#: what the loader trains under and the server rebuilds.
MASK_FIELDS = ("constant", "drot", "history_blocked", "columns", "state_dims", "body_columns",
               "body_state_dims")

# What a directory's drot is in its datasets record, by the manifest's
# rotation_delta_frame; masked and none carry no target.
DROT_RECORD = {"spatial": "spatial", "body": "body_to_spatial",
               "euler_diff": "euler_to_spatial", "none": "none", "unresolved": "masked"}
UNTRAINED_DROT = ("masked", "none")


class RefusalKeyError(KeyError):
    """A KeyError whose message reads as written, so it starts with its refusal number.

    KeyError quotes its argument when printed, which would put a quote before
    "R3" and hide the refusal from a search for its number.
    """

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


def _as_float(value: Any) -> Any:
    """Every number of a document as a float, so 1 and 1.0 compare alike.

    An identity is compared across writers (the stats tool, the runner, a test),
    and json keeps the int-or-float spelling each of them used, so a comparison
    of the raw spelling would refuse two equal files.
    """
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _as_float(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_float(item) for item in value]
    raise TypeError(
        f"an identity was asked to hold a {type(value).__name__}.\n"
        "Expected json values: numbers, strings, booleans, null, lists and objects.\n"
        "Convert the value to json before comparing it."
    )


def header_of(alpha: float) -> dict:
    """The normaliser header tools/build_stats.py writes and every entry identity holds.

    The rule fields (CHECKED_HEADER) are what the reader holds a file to; the
    prose fields say the same in words and are compared through the identity only.
    """
    return {
        "transform": TRANSFORM,
        "quantiles": list(QUANTILES),
        "base_transform": BASE_TRANSFORM,
        "base_motion": dict(BASE_MOTION),
        "base_floor": list(BASE_SCALE_FLOOR),
        "band_alpha": float(alpha),
        "out_of_band": OUT_OF_BAND,
        "groups": {"action": {name: list(_columns(span)) for name, span in ACTION_GROUPS.items()},
                   "body": {name: list(_columns(span)) for name, span in BODY_GROUPS.items()}},
        "layout": {"action": ACTION_RECORD, "state": STATE_RECORD, "body": BODY_RECORD},
        "domain": "joints rad absolute; dpos cm/step and drot rad/step (spatial rotation vector, "
                  "drawn frame) after unify; canonical gripper; hand in the key's unit "
                  "(RobotDataset.targets); body base cm/step and rad/step in the base frame, "
                  "torso and head absolute",
        "input_clamp": "range columns and every body column clamped to ±(1+2α), the gripper "
                       "to ±1",
        "non_finite": "the value is 0 and its group at that step is loss-invalid",
        "drot": "spatial",
        "gripper": "canonical 0 open..1 closed, clipped to [0,1], y=2g-1",
        "state": {"range": "joints, position and hand per (key,slot,dim) q01/q99; clip "
                           "±(1+2α) on both paths, counted; a constant dim maps to 0",
                  "rotation": "6d_cols01_passthrough", "aperture": "2a-1"},
        "body": {"posture": "torso and head per (key,dim) q01/q99; a constant dim maps to 0",
                 "base": "x/s with s the q99 of |x| over moving rows, so a still base is 0; "
                         "below the floor the dim is constant"},
        "constant_scan": "full",
    }


CHECKED_HEADER = ("transform", "quantiles", "base_transform", "base_motion", "base_floor",
                  "out_of_band", "groups", "layout")

# The fields an entry identity holds (and every v2 entry states).
ENTRY_FIELDS = ("slots", "joints", "hand", "action", "state", "body", "ee_frame_fix", "clause")
ACTION_FIELDS = ("low", "high", "constant", "absent", "mask", "gripper_valid")


def entry_identity(entry_document: dict, header: dict) -> dict:
    """One entry's identity: its numbers, declarations, clause and the normaliser header.

    The values themselves, numbers as floats, compared with ==. reference,
    stage, fit, action.from and action.from_groups are provenance and stay out,
    so an identical rebuild keeps the identity and an alpha, clause, joint, hand
    or body change moves it. A digest field an older writer stored is not read.
    An entry missing a field (a v6 one has no joints, hand or body) is refused
    naming the fields.
    """
    for where, document, names in (
            ("an embodiments entry", entry_document, ENTRY_FIELDS),
            ("an embodiments entry's action", entry_document.get("action"), ACTION_FIELDS)):
        missing = [name for name in names
                   if not isinstance(document, dict) or name not in document]
        if missing:
            raise ValueError(
                f"{where} has no {missing} field.\n"
                f"Expected a {EMBODIMENTS_SCHEMA} entry; a {OLD_EMBODIMENTS_SCHEMAS[-1]} (v6) "
                "entry has no joints, hand or body and 7-wide rows.\n"
                "Build the entry with this tree's tools/build_stats.py."
            )
    action = entry_document["action"]
    view = {
        "slots": entry_document["slots"],
        "joints": entry_document["joints"],
        "hand": entry_document["hand"],
        "action": {name: action[name] for name in ACTION_FIELDS},
        "state": entry_document["state"],
        "body": entry_document["body"],
        "ee_frame_fix": entry_document["ee_frame_fix"],
        "clause": entry_document["clause"],
        "normaliser": header,
    }
    return _as_float(view)


def identity_differs(found: dict, wanted: dict) -> list[str]:
    """The top-level fields of two identities that disagree, for a refusal to name."""
    return sorted(name for name in set(found) | set(wanted)
                  if found.get(name) != wanted.get(name))


def record_inputs(record_document: dict) -> dict:
    """A datasets record's six client-input fields and no others, numbers as floats."""
    missing = [name for name in RECORD_FIELDS if name not in record_document]
    if missing:
        raise ValueError(
            f"a datasets record has no {missing} field.\n"
            f"Expected all of {list(RECORD_FIELDS)}.\n"
            "Build the record with record_from_manifest."
        )
    return _as_float({name: record_document[name] for name in RECORD_FIELDS})


def record_from_manifest(entry_document: dict) -> dict:
    """The client-input fields of one manifest entry, as a datasets record states them.

    The server's record and the client's recomputation both come from here, so
    a changed gripper polarity, unit factor or frequency is a record mismatch
    before any simulator opens.
    """
    name = entry_document.get("name", "?")
    embodiment = _field(entry_document, "embodiment", f"manifest entry {name!r}")
    kind = embodiment.get("kind")
    if kind == "constant":
        label, keys = str(embodiment["key"]), [str(embodiment["key"])]
    elif kind == "episode":
        label, keys = PER_EPISODE, [str(key) for key in embodiment["keys"]]
    else:
        raise ValueError(
            f"manifest entry {name!r} states embodiment kind {kind!r}.\n"
            "Expected 'constant' or 'episode'.\n"
            "Rebuild the manifest with this tree's manifest stage."
        )
    action = _field(entry_document, "action", f"manifest entry {name!r}")
    gripper = _field(action, "gripper", f"manifest entry {name!r} action")
    control = _field(entry_document, "control_hz", f"manifest entry {name!r}")
    # A directory whose arms read no dpos or drot (joints-only: BimanualYAM, the
    # Tien Kung) states no unit factor in manifest schema 3; null is recorded
    # as null, so the client's recomputation still meets the server's record.
    translation = _field(action, "translation_cm_per_unit", f"manifest entry {name!r} action")
    rotation = _field(action, "rotation_rad_per_unit", f"manifest entry {name!r} action")
    return {
        "embodiment": label,
        "keys": keys,
        "slots": [int(arm["slot"]) for arm in entry_document["arms"]],
        "gripper": {"raw_open": float(gripper["raw_open"]),
                    "raw_closed": float(gripper["raw_closed"]),
                    "method": str(gripper["method"])},
        "unit_factors": {
            "translation_cm_per_unit": None if translation is None
            else [float(v) for v in translation],
            "rotation_rad_per_unit": None if rotation is None else float(rotation),
        },
        "control_hz": None if control is None else float(control),
    }


def _field(document: dict, key: str, where: str) -> Any:
    if not isinstance(document, dict) or key not in document:
        raise ValueError(
            f"{where} has no {key!r} field.\n"
            "Expected the field; the embodiments contract lists every one.\n"
            "Rebuild the file with this tree's tools."
        )
    return document[key]


@dataclass(frozen=True, eq=False)
class BodyEntry:
    """One key's body block, in the base frame and never turned by frame_ref.

    present False is a key with no body (ABSENT): every dimension is absent.
    The low and high arrays are [8] float64, NaN where absent; a fitted base
    dim holds -s and s, its symmetric scale, and a constant one 0 and 0 (still).
    The constant and absent arrays are [8] bool. base names what the key reads
    its base from (BASE_SOURCES); torso and head name the joints it declares.
    """

    present: bool
    base: str | None
    torso: tuple[str, ...]
    head: tuple[str, ...]
    action_low: np.ndarray
    action_high: np.ndarray
    action_constant: np.ndarray
    action_absent: np.ndarray
    state_low: np.ndarray
    state_high: np.ndarray
    state_constant: np.ndarray
    state_absent: np.ndarray
    base_scale_rows: int

    def decodable(self) -> np.ndarray:
        """[8] bool: the body columns this key trains and decodes (neither constant nor absent)."""
        return ~self.action_constant & ~self.action_absent


@dataclass(frozen=True, eq=False)
class Entry:
    """One embodiment key: per-slot bounds and flags, its declarations, clause and identity.

    low, high are [n_slots, 26] float64 (NaN where absent), constant and absent
    [n_slots, 26] bool, the gripper column holding 0 and 1. state_low and
    state_high are [n_slots, 29], NaN where absent and on the 6D and aperture,
    which are not range-mapped; state_constant and state_absent are [n_slots, 29].
    joints and hand are the key's declarations or None. ee_frame_fix is R_c,
    applied as R @ R_c to the state 6D only. identity is entry_identity of
    the entry as stored, what two files' entries are compared by.
    """

    key: str
    slots: tuple[int, ...]
    joints: dict | None
    hand: dict | None
    low: np.ndarray
    high: np.ndarray
    constant: np.ndarray
    absent: np.ndarray
    mask: tuple[bool, ...]
    gripper_valid: tuple[bool, ...]
    action_from: str | None
    state_low: np.ndarray
    state_high: np.ndarray
    state_constant: np.ndarray
    state_absent: np.ndarray
    body: BodyEntry
    ee_frame_fix: np.ndarray | None
    clause: str
    stage: str
    identity: dict
    # The groups copied from action_from (inherited_groups); None when action_from is.
    action_from_groups: tuple[str, ...] | None = None

    def slot_index(self, slot: int) -> int:
        """The row of this entry that one action slot reads, or R15."""
        if int(slot) not in self.slots:
            raise ValueError(
                f"{refusal_prefix(15)} key {self.key!r} covers slots {list(self.slots)} "
                f"and was asked for slot {int(slot)}.\n"
                "Expected the directory's arms[k].slot to be one of the entry's slots.\n"
                "Fix the arm's slot in the manifest, or rebuild the entry with that slot."
            )
        return self.slots.index(int(slot))

    def decodable(self, slot: int) -> np.ndarray:
        """[26] bool: the columns this key trains and decodes on one slot.

        Every range column (joints, dpos, drot, hand) that is neither constant
        nor absent, and the gripper where some reference directory trains it.
        """
        row = self.slot_index(slot)
        out = np.asarray(MASK, dtype=bool) & ~self.constant[row] & ~self.absent[row]
        out[GRIPPER_COLUMN] = bool(self.gripper_valid[row])
        return out

    def state_valid(self, slot: int) -> np.ndarray:
        """[29] bool: the state dimensions this key can hold valid on one slot (not absent)."""
        return ~self.state_absent[self.slot_index(slot)]

    def _state_block(self, span: slice, bounds: np.ndarray) -> np.ndarray | None:
        block = bounds[:, span]
        return None if bool(self.state_absent[:, span].all()) else block

    @property
    def position_low(self) -> np.ndarray | None:
        """[n_slots, 3] metres, or None for a key with no position on any slot."""
        return self._state_block(STATE_POSITION, self.state_low)

    @property
    def position_high(self) -> np.ndarray | None:
        return self._state_block(STATE_POSITION, self.state_high)

    @property
    def joint_low(self) -> np.ndarray | None:
        """[n_slots, 7] rad, NaN on absent joints, or None for a key with no joint state."""
        return self._state_block(STATE_JOINTS, self.state_low)

    @property
    def joint_high(self) -> np.ndarray | None:
        return self._state_block(STATE_JOINTS, self.state_high)

    @property
    def hand_low(self) -> np.ndarray | None:
        """[n_slots, 12] in the key's hand unit, NaN on absent joints, or None with no hand."""
        return self._state_block(STATE_HAND, self.state_low)

    @property
    def hand_high(self) -> np.ndarray | None:
        return self._state_block(STATE_HAND, self.state_high)


@dataclass(frozen=True, eq=False)
class DatasetRecord:
    """One directory's client inputs and the masks it trains under.

    The six inputs (record_inputs) are what a client recomputes. constant is the
    directory's own constant action columns per slot; columns [n_slots, 26]
    and state_dims [n_slots, 29] are the action columns it trains and the
    state dimensions it holds valid, after its constants, dead readings, gated
    joints, a masked drot or gripper; body_columns and body_state_dims are the
    body's [8], all False when no key of it has a body.
    """

    directory: str
    embodiment: str
    keys: tuple[str, ...]
    slots: tuple[int, ...]
    gripper: dict
    unit_factors: dict | None
    control_hz: float | None
    inputs: dict
    constant: tuple[tuple[bool, ...], ...]
    drot: str
    history_blocked: int
    columns: np.ndarray
    state_dims: np.ndarray
    body_columns: np.ndarray
    body_state_dims: np.ndarray

    def slot_row(self, slot: int) -> int:
        """The row of this record's masks one arm slot reads, or R15."""
        if int(slot) not in self.slots:
            raise ValueError(
                f"{refusal_prefix(15)} directory {self.directory!r} packs its arms into slots "
                f"{list(self.slots)} and was asked for slot {int(slot)}.\n"
                "Expected one of the slots the directory's arms fill.\n"
                "Drive the slots this directory packed."
            )
        return self.slots.index(int(slot))


@dataclass(frozen=True)
class FitReport:
    """What a fit check measured: per-column p99 |y|, the worst field's share, the verdict.

    fields holds per field (joints, ee, hand; base, posture for the body) its
    rows, flagged share, the columns measured and its own verdict.
    """

    p99_abs_y: list
    flagged_share: float
    ok: bool
    rows: int
    fields: dict = field(default_factory=dict)


class Embodiments:
    """An embodiments.json, checked on the way in and read by key and by directory."""

    def __init__(self, document: dict, *, entries: dict[str, Entry],
                 datasets: dict[str, DatasetRecord]):
        self._document = document
        self.schema = str(document["schema"])
        self.header = dict(document["normaliser"])
        self.parent = document.get("parent")
        self.provenance = dict(document["provenance"])
        self.entries = entries
        self.datasets = datasets
        self.trained = document.get("trained")

    @property
    def band_alpha(self) -> float:
        """The outlier band's alpha: flagged outside [lo - a span, hi + a span]."""
        return float(self.header["band_alpha"])

    @classmethod
    def load(cls, path) -> Embodiments:
        """Read and check one embodiments.json; R1 when the file is not there."""
        where = Path(path)
        if not where.is_file():
            raise FileNotFoundError(
                f"{refusal_prefix(1)} there is no embodiments file at {where}.\n"
                f"Expected the {EMBODIMENTS_FILE} the stats stage or the runner wrote.\n"
                "Build it with tools/build_stats.py, or point the config at it."
            )
        document = json.loads(where.read_bytes().decode("utf-8"),
                              parse_constant=_refuse_constant)
        return cls.from_document(document, where=str(where))

    @classmethod
    def from_document(cls, document: dict, *, where: str) -> Embodiments:
        """Check a parsed document and build the reader over it.

        Refuses another schema (a v6 one by name), another normaliser rule or
        layout, a band alpha at or below zero, a bound that cannot map
        (DEGENERATE_RANGE, CX 11), a declaration its flags contradict, a record
        mask wider than its keys, and an end-effector fix that is not a proper
        rotation. The checksum fields an older writer stored are neither read
        nor kept, so a copy this reader writes holds none (without_checksums).
        """
        schema = _field(document, "schema", where)
        if schema in OLD_EMBODIMENTS_SCHEMAS:
            raise ValueError(
                f"{where} states schema {schema!r}, the v6 normaliser file (19-wide actions, "
                "10-wide state, no body).\n"
                f"Expected {EMBODIMENTS_SCHEMA!r}: the v7 layout reads no v6 file.\n"
                "Rebuild the file with this tree's tools/build_stats.py."
            )
        if schema != EMBODIMENTS_SCHEMA:
            raise ValueError(
                f"{where} states schema {schema!r}.\n"
                f"Expected {EMBODIMENTS_SCHEMA!r}.\n"
                "Rebuild the file with this tree's tools/build_stats.py."
            )
        header = _check_header(_field(document, "normaliser", where), where)
        _field(document, "provenance", where)
        _field(document, "parent", where)
        entries = {
            str(key): _read_entry(str(key), raw, header, where)
            for key, raw in sorted(_field(document, "embodiments", where).items())
        }
        datasets = {
            str(name): _read_record(str(name), raw, entries, where)
            for name, raw in sorted(_field(document, "datasets", where).items())
        }
        trained = _field(document, "trained", where)
        if trained is not None and not isinstance(trained, dict):
            raise ValueError(
                f"{where} states trained {trained!r}.\n"
                "Expected null in a corpus copy, or the runner's object in a checkpoint copy.\n"
                "Rebuild the file; the runner fills trained."
            )
        return cls(without_checksums(document), entries=entries, datasets=datasets)

    def document(self) -> dict:
        """A copy of the document this reader was built from."""
        return copy.deepcopy(self._document)

    def dumps(self) -> bytes:
        """The bytes a copy of this file is written as."""
        return _dumps(self._document)

    def entry(self, key: str) -> Entry:
        """One key's entry, or R3 for a key this file does not hold."""
        if key not in self.entries:
            raise RefusalKeyError(
                f"{refusal_prefix(3)} key {key!r} is not in this embodiments file.\n"
                f"Expected one of {sorted(self.entries)}.\n"
                "Serve a checkpoint that trained this embodiment, or fix the key."
            )
        return self.entries[key]

    def record(self, directory: str) -> DatasetRecord:
        """One directory's record; a directory with none could not be served."""
        if directory not in self.datasets:
            raise RefusalKeyError(
                f"directory {directory!r} has no datasets record in this embodiments file.\n"
                "Expected one record per manifest directory.\n"
                "Rebuild the file from the manifest this run trains."
            )
        return self.datasets[directory]

    def _directory(self, directory: str, key: str) -> tuple[DatasetRecord, Entry]:
        record = self.record(directory)
        entry = self.entry(key)
        if key not in record.keys:
            raise ValueError(
                f"{refusal_prefix(4)} directory {directory!r} is a record of keys "
                f"{list(record.keys)} and was asked for key {key!r}.\n"
                "Expected a key the directory trained (for a per-episode directory, one of "
                "its keys).\n"
                "Name the directory recorded for this embodiment."
            )
        return record, entry

    def columns(self, directory: str, key: str, slot: int) -> np.ndarray:
        """[26] bool: the action columns a directory trained at one slot under one key.

        The directory's own mask (its constants, dead readings, gated joints, a
        masked drot or gripper) AND the key's decodable columns:
        the loader's dim_valid before the per-sample read, and the server's for
        a null request, so train and serve start from one rule.
        """
        record, entry = self._directory(directory, key)
        return record.columns[record.slot_row(slot)] & entry.decodable(slot)

    def state_dims(self, directory: str, key: str, slot: int) -> np.ndarray:
        """[29] bool: the state dims a directory holds valid at one slot under one key."""
        record, entry = self._directory(directory, key)
        return record.state_dims[record.slot_row(slot)] & entry.state_valid(slot)

    def body_columns(self, directory: str, key: str) -> np.ndarray:
        """[8] bool: the body columns a directory trained under one key; none without a body."""
        record, entry = self._directory(directory, key)
        return record.body_columns & entry.body.decodable()

    def body_state_dims(self, directory: str, key: str) -> np.ndarray:
        """[8] bool: the body state dims a directory holds valid under one key."""
        record, entry = self._directory(directory, key)
        return record.body_state_dims & ~entry.body.state_absent

    def meta(self) -> dict:
        """The stamp a checkpoint's meta.json carries: the keys of the file beside it."""
        return {"keys": sorted(self.entries)}

    def with_trained(self, trained: dict, *, parent: str | None) -> Embodiments:
        """This file with a trained section and a parent, as a checkpoint step writes it."""
        document = self.document()
        document["trained"] = copy.deepcopy(trained)
        document["parent"] = parent
        return Embodiments.from_document(document, where="with_trained")


def _dumps(document: dict) -> bytes:
    text = json.dumps(document, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


# The checksum fields earlier writers of this file stored: an entry's and a
# record's own, the manifest's and the lineage's in provenance and trained.
OBSOLETE_FIELDS = ("digest", "record_digest", "manifest_sha256", "source_digest",
                   "lineage_sha256")


def without_checksums(value: Any, key: str | None = None) -> Any:
    """A copy of a document without the checksums earlier writers stored.

    The OBSOLETE_FIELDS are dropped, a provenance inputs map of path to checksum
    becomes the sorted list of its paths, and a parent stated as a checksum is
    null. Every other value is copied unchanged.
    """
    if isinstance(value, dict):
        if key == "inputs":
            return sorted(value)
        return {name: without_checksums(item, name) for name, item in value.items()
                if name not in OBSOLETE_FIELDS}
    if isinstance(value, list):
        return [without_checksums(item) for item in value]
    if key == "parent" and isinstance(value, str) and value.startswith("sha256:"):
        return None
    return value


def _refuse_constant(name: str) -> None:
    raise ValueError(
        f"an embodiments file holds {name}.\n"
        "Expected null for an absent bound; json has no NaN or infinity.\n"
        "Rebuild the file with this tree's tools/build_stats.py."
    )


def _check_header(header: dict, where: str) -> dict:
    """The normaliser header, refused unless it names this tree's rules and layout."""
    wanted = header_of(1.0)
    for name in CHECKED_HEADER:
        found = _field(header, name, where)
        if name == "out_of_band" and found in RETIRED_RULES:
            raise ValueError(
                f"{where} states normaliser out_of_band {found!r}, {RETIRED_RULES[found]}.\n"
                f"Expected {OUT_OF_BAND!r}: a step outside the band masks only its own group.\n"
                "Rebuild the file with this tree's tools/build_stats.py."
            )
        try:
            same = _as_float(found) == _as_float(wanted[name])
        except TypeError:
            same = False
        if not same:
            raise ValueError(
                f"{where} states normaliser {name} {found!r}.\n"
                f"Expected {wanted[name]!r}; this tree reads no other.\n"
                "Rebuild the file with this tree's tools/build_stats.py."
            )
    alpha = _field(header, "band_alpha", where)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) \
            or not math.isfinite(float(alpha)) or float(alpha) <= 0.0:
        raise ValueError(
            f"{where} states band_alpha {alpha!r}.\n"
            "Expected a finite alpha above 0; the band is [lo - a span, hi + a span].\n"
            "Rebuild the file with a positive --band-alpha."
        )
    return dict(header)


def _matrix(raw: Any, rows: int, cols: int, name: str, where: str, *,
            nullable: bool) -> np.ndarray:
    """A [rows][cols] json array -> float64, NaN where null is allowed."""
    if not isinstance(raw, list) or len(raw) != rows \
            or any(not isinstance(row, list) or len(row) != cols for row in raw):
        raise ValueError(
            f"{where} {name} is not [{rows}][{cols}].\n"
            "Expected one row per slot of the entry, as wide as the v7 record.\n"
            "Rebuild the file with this tree's tools/build_stats.py."
        )
    out = np.full((rows, cols), np.nan, dtype=np.float64)
    for i, row in enumerate(raw):
        for j, value in enumerate(row):
            if value is None and nullable:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"{where} {name}[{i}][{j}] is {value!r}.\n"
                    "Expected a number" + (" or null" if nullable else "") + ".\n"
                    "Rebuild the file with this tree's tools/build_stats.py."
                )
            out[i, j] = float(value)
    return out


def _flags(raw: Any, rows: int, cols: int, name: str, where: str) -> np.ndarray:
    if not isinstance(raw, list) or len(raw) != rows or any(
            not isinstance(row, list) or len(row) != cols
            or any(not isinstance(flag, bool) for flag in row) for row in raw):
        raise ValueError(
            f"{where} {name} is not [{rows}][{cols}] booleans.\n"
            "Expected one row of flags per slot, as wide as the v7 record.\n"
            "Rebuild the file with this tree's tools/build_stats.py."
        )
    return np.asarray(raw, dtype=bool).reshape(rows, cols)


def _degenerate(key: str, slot: int | None, dim: str, low: float, high: float,
                what: str) -> ValueError:
    at = "" if slot is None else f" slot {slot}"
    return ValueError(
        f"{DEGENERATE_RANGE}: key {key!r}{at} {what} {dim} has lo {low!r} and hi {high!r}.\n"
        "Expected finite bounds with lo < hi on every range dimension that is neither "
        "constant nor absent, and on every state position axis.\n"
        "Declare the dimension constant or absent for the key, or rebuild the stats."
    )


def _range_dim(key: str, slot: int | None, name: str, what: str, lo: float, hi: float,
               constant: bool, absent: bool, where: str) -> None:
    """One q01/q99 dimension's bounds against its flags (C10, CX 11)."""
    if absent:
        if constant or not (math.isnan(lo) and math.isnan(hi)):
            raise ValueError(
                f"{where} {what} {name} is absent and states bounds {lo!r} to {hi!r} "
                f"(constant {constant}).\n"
                "Expected null bounds and constant false on an absent dimension.\n"
                "Rebuild the file with this tree's tools/build_stats.py."
            )
        return
    if constant:
        if not (math.isfinite(lo) and lo == hi):
            raise _degenerate(key, slot, name, lo, hi, f"constant {what}")
        return
    if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
        raise _degenerate(key, slot, name, lo, hi, what)


def _declaration(raw: Any, what: str, where: str) -> dict | None:
    """A key's joints or hand declaration: null, or count, names and unit (spec 8.3)."""
    if raw is None:
        return None
    widest, names_field, units = {"joints": (7, "names", JOINT_UNITS),
                                  "hand": (12, "order", HAND_UNITS)}[what]
    required = {"count", names_field, "unit"}
    allowed = required | ({"polarity_note"} if what == "hand" else set())
    good = isinstance(raw, dict) and required <= set(raw) and not set(raw) - allowed
    if good:
        count, names = raw["count"], raw[names_field]
        good = not isinstance(count, bool) and isinstance(count, int) \
            and 1 <= count <= widest and isinstance(names, list) and len(names) == count \
            and all(isinstance(one, str) and one for one in names) and raw["unit"] in units
    if not good:
        raise ValueError(
            f"{where} declares {what} {raw!r}.\n"
            f"Expected null, or {{count: 1..{widest}, {names_field}: count names, unit: one of "
            f"{list(units)}}}.\n"
            "Take it from config/embodiments.yaml (neo_vla_embodiment_keys_v2)."
        )
    return dict(raw)


def _beyond(declared: dict | None, span: slice) -> list[int]:
    """The record columns of span a declaration leaves out: all of them for null."""
    count = 0 if declared is None else int(declared["count"])
    return list(range(span.start + count, span.stop))


def _read_slots(raw: dict, where: str) -> tuple[int, ...]:
    slots_raw = _field(raw, "slots", where)
    if not isinstance(slots_raw, list) or not slots_raw \
            or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in slots_raw) \
            or len(set(slots_raw)) != len(slots_raw):
        raise ValueError(
            f"{where} states slots {slots_raw!r}.\n"
            "Expected a non-empty list of distinct action slots.\n"
            "Take them from the reference directories' arms[k].slot."
        )
    return tuple(int(s) for s in slots_raw)


def _read_action(key: str, raw: dict, slots: tuple[int, ...], joints: dict | None,
                 hand: dict | None, where: str):
    n = len(slots)
    action = _field(raw, "action", where)
    low = _matrix(_field(action, "low", where), n, ACTION_DIM, "action.low", where,
                  nullable=True)
    high = _matrix(_field(action, "high", where), n, ACTION_DIM, "action.high", where,
                   nullable=True)
    constant = _flags(_field(action, "constant", where), n, ACTION_DIM, "action.constant", where)
    absent = _flags(_field(action, "absent", where), n, ACTION_DIM, "action.absent", where)
    mask = tuple(_field(action, "mask", where))
    if mask != MASK:
        raise ValueError(
            f"{where} action.mask is {list(mask)}.\n"
            f"Expected {list(MASK)}: every column range-mapped but the gripper "
            f"({GRIPPER_COLUMN}).\n"
            "Rebuild the file with this tree's tools/build_stats.py."
        )
    gripper_valid = _field(action, "gripper_valid", where)
    if not isinstance(gripper_valid, list) or len(gripper_valid) != n \
            or any(not isinstance(flag, bool) for flag in gripper_valid):
        raise ValueError(
            f"{where} action.gripper_valid is {gripper_valid!r}.\n"
            f"Expected {n} booleans, one per slot.\n"
            "Rebuild the file with this tree's tools/build_stats.py."
        )
    undeclared = {"joints": _beyond(joints, ACTION_JOINTS), "hand": _beyond(hand, ACTION_HAND)}
    for row, slot in enumerate(slots):
        lo, hi = float(low[row, GRIPPER_COLUMN]), float(high[row, GRIPPER_COLUMN])
        if (lo, hi) != (0.0, 1.0):
            raise ValueError(
                f"{where} slot {slot} gripper bounds are {lo!r} to {hi!r}.\n"
                "Expected 0 and 1: the gripper is the canonical 0-open 1-closed value.\n"
                "Rebuild the file with this tree's tools/build_stats.py."
            )
        if bool(absent[row, GRIPPER_COLUMN]) or bool(constant[row, GRIPPER_COLUMN]):
            raise ValueError(
                f"{where} slot {slot} gripper is stated absent "
                f"{bool(absent[row, GRIPPER_COLUMN])} and constant "
                f"{bool(constant[row, GRIPPER_COLUMN])}.\n"
                "Expected both false: whether a key trains its gripper is gripper_valid alone.\n"
                "Rebuild the file with this tree's tools/build_stats.py."
            )
        for column in RANGE_COLUMNS:
            _range_dim(key, slot, DIM_NAMES[column], "action dim", float(low[row, column]),
                       float(high[row, column]), bool(constant[row, column]),
                       bool(absent[row, column]), f"{where} slot {slot}")
        for what, columns in undeclared.items():
            trained = [DIM_NAMES[column] for column in columns if not absent[row, column]]
            if trained:
                count = _declared(joints if what == "joints" else hand)
                raise ValueError(
                    f"{where} slot {slot} states action columns {trained} present, beyond "
                    f"the {what} the key declares ({count}).\n"
                    "Expected every column past the declared count absent: a 6-DoF arm's j6 "
                    "and a gripper key's hand are invalid, never 0.\n"
                    "Fix the key's declaration in config/embodiments.yaml, or rebuild the stats."
                )
    return action, low, high, constant, absent, mask, gripper_valid


def _declared(declared: dict | None) -> str:
    return "none" if declared is None else f"{declared['count']}"


def _read_state(key: str, raw: dict, slots: tuple[int, ...], joints: dict | None,
                hand: dict | None, where: str):
    """The state block: per-slot bounds of position, joints and hand, and [29] flags."""
    n = len(slots)
    state = _field(raw, "state", where)
    where_state = f"{where} state"
    blocks = {
        "position": (STATE_POSITION, "position_low", "position_high"),
        "joint": (STATE_JOINTS, "joint_low", "joint_high"),
        "hand": (STATE_HAND, "hand_low", "hand_high"),
    }
    low = np.full((n, STATE_DIM), np.nan, dtype=np.float64)
    high = np.full((n, STATE_DIM), np.nan, dtype=np.float64)
    for span, low_name, high_name in blocks.values():
        width = span.stop - span.start
        low[:, span] = _matrix(_field(state, low_name, where_state), n, width,
                               f"state.{low_name}", where, nullable=True)
        high[:, span] = _matrix(_field(state, high_name, where_state), n, width,
                                f"state.{high_name}", where, nullable=True)
    constant = _flags(_field(state, "constant", where_state), n, STATE_DIM, "state.constant",
                      where)
    absent = _flags(_field(state, "absent", where_state), n, STATE_DIM, "state.absent", where)
    undeclared = {"joints": _beyond(joints, STATE_JOINTS), "hand": _beyond(hand, STATE_HAND)}
    for row, slot in enumerate(slots):
        at = f"{where} slot {slot}"
        for name, span in (("position", STATE_POSITION), ("rotation", STATE_ROTATION)):
            flags = absent[row, span]
            if bool(flags.any()) and not bool(flags.all()):
                raise ValueError(
                    f"{at} states the state {name} absent on {flags.astype(int).tolist()}.\n"
                    f"Expected the {name} all absent or all present: it is one group (C7-6).\n"
                    "Rebuild the file with this tree's tools/build_stats.py."
                )
        fixed = [STATE_DIM_NAMES[dim] for dim in (*_columns(STATE_POSITION),
                                                  *_columns(STATE_ROTATION), STATE_APERTURE)
                 if constant[row, dim]]
        if fixed:
            raise ValueError(
                f"{at} states state dims {fixed} constant.\n"
                "Expected position, the 6D and the aperture never constant: a position axis "
                "needs lo < hi, and the 6D and aperture are not range-mapped.\n"
                "Rebuild the file with this tree's tools/build_stats.py."
            )
        for dim in range(STATE_DIM):
            if dim in STATE_RANGE:
                what = "state position axis" if dim in _columns(STATE_POSITION) else "state dim"
                name = "xyz"[dim - STATE_POSITION.start] if dim in _columns(STATE_POSITION) \
                    else STATE_DIM_NAMES[dim]
                _range_dim(key, slot, name, what, float(low[row, dim]), float(high[row, dim]),
                           bool(constant[row, dim]), bool(absent[row, dim]), at)
        for what, columns in undeclared.items():
            held = [STATE_DIM_NAMES[dim] for dim in columns if not absent[row, dim]]
            if held:
                count = _declared(joints if what == "joints" else hand)
                raise ValueError(
                    f"{at} states state dims {held} present, beyond the {what} the key "
                    f"declares ({count}).\n"
                    "Expected every dimension past the declared count absent.\n"
                    "Fix the key's declaration in config/embodiments.yaml, or rebuild the stats."
                )
    return low, high, constant, absent


def _vector(raw: Any, name: str, where: str) -> np.ndarray:
    """A [8] json list of numbers or nulls -> float64, NaN for null."""
    return _matrix([raw] if isinstance(raw, list) else raw, 1, BODY_DIM, name, where,
                   nullable=True)[0]


def _flag_vector(raw: Any, name: str, where: str) -> np.ndarray:
    return _flags([raw] if isinstance(raw, list) else raw, 1, BODY_DIM, name, where)[0]


def _names(raw: Any, widest: int, what: str, where: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not 1 <= len(raw) <= widest \
            or any(not isinstance(one, str) or not one for one in raw):
        raise ValueError(
            f"{where} body {what} is {raw!r}.\n"
            f"Expected null or 1 to {widest} joint names.\n"
            "Take them from config/embodiments.yaml (neo_vla_embodiment_keys_v2)."
        )
    return tuple(raw)


ABSENT_BODY_FLAGS = np.ones(BODY_DIM, dtype=bool)
BODY_FIELDS = ("declared", "base", "torso", "head", "action_low", "action_high",
               "action_constant", "action_absent", "state_low", "state_high", "state_constant",
               "state_absent", "base_scale_rows")


def _absent_body() -> BodyEntry:
    nan = np.full(BODY_DIM, np.nan)
    return BodyEntry(
        present=False, base=None, torso=(), head=(), action_low=nan.copy(),
        action_high=nan.copy(), action_constant=~ABSENT_BODY_FLAGS.copy(),
        action_absent=ABSENT_BODY_FLAGS.copy(), state_low=nan.copy(), state_high=nan.copy(),
        state_constant=~ABSENT_BODY_FLAGS.copy(), state_absent=ABSENT_BODY_FLAGS.copy(),
        base_scale_rows=0,
    )


def _read_body(key: str, raw: dict, where: str) -> BodyEntry:
    """The body block: {"declared": "absent"}, or the present body's declaration and bounds."""
    body = _field(raw, "body", where)
    at = f"{where} body"
    declared = body.get("declared") if isinstance(body, dict) else None
    if declared == "absent":
        if set(body) != {"declared"}:
            raise ValueError(
                f"{at} is declared absent and states {sorted(set(body) - {'declared'})}.\n"
                "Expected {\"declared\": \"absent\"} alone: a key with no body has no bounds.\n"
                "Rebuild the file with this tree's tools/build_stats.py."
            )
        return _absent_body()
    if declared != "present" or set(body) != set(BODY_FIELDS):
        raise ValueError(
            f"{at} is {body!r}.\n"
            f"Expected {{\"declared\": \"absent\"}}, or declared \"present\" with exactly "
            f"{list(BODY_FIELDS)}.\n"
            "Rebuild the file with this tree's tools/build_stats.py."
        )
    base = body["base"]
    if base not in BASE_SOURCES:
        raise ValueError(
            f"{at} reads its base from {base!r}.\n"
            f"Expected one of {list(BASE_SOURCES)}.\n"
            "Take it from config/embodiments.yaml (neo_vla_embodiment_keys_v2)."
        )
    torso = _names(body["torso"], MAX_TORSO, "torso", where)
    head = _names(body["head"], MAX_HEAD, "head", where)
    rows = body["base_scale_rows"]
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
        raise ValueError(
            f"{at} states base_scale_rows {rows!r}.\n"
            "Expected the count of moving rows the base scale was fitted on, 0 or more.\n"
            "Rebuild the file with this tree's tools/build_stats.py."
        )
    arrays = {}
    for side in ("action", "state"):
        arrays[f"{side}_low"] = _vector(body[f"{side}_low"], f"body.{side}_low", where)
        arrays[f"{side}_high"] = _vector(body[f"{side}_high"], f"body.{side}_high", where)
        arrays[f"{side}_constant"] = _flag_vector(body[f"{side}_constant"],
                                                  f"body.{side}_constant", where)
        arrays[f"{side}_absent"] = _flag_vector(body[f"{side}_absent"], f"body.{side}_absent",
                                                where)
        _check_body_side(key, side, base, torso, head, rows, arrays, at)
    return BodyEntry(present=True, base=str(base), torso=torso, head=head,
                     base_scale_rows=int(rows), **arrays)


def _check_body_side(key: str, side: str, base: str, torso: tuple, head: tuple, rows: int,
                     arrays: dict, at: str) -> None:
    low, high = arrays[f"{side}_low"], arrays[f"{side}_high"]
    constant, absent = arrays[f"{side}_constant"], arrays[f"{side}_absent"]
    for dim in _columns(BODY_BASE):
        name, lo, hi = BODY_DIM_NAMES[dim], float(low[dim]), float(high[dim])
        if absent[dim] or constant[dim]:
            _range_dim(key, None, name, f"body {side} dim", lo, hi, bool(constant[dim]),
                       bool(absent[dim]), at)
            if constant[dim] and lo != 0.0:
                raise ValueError(
                    f"{at} {side} {name} is constant at {lo!r}.\n"
                    "Expected a constant base dim at 0: it is a base that does not move.\n"
                    "Rebuild the file with this tree's tools/build_stats.py."
                )
            continue
        if not (math.isfinite(lo) and math.isfinite(hi) and hi > 0.0):
            raise _degenerate(key, None, name, lo, hi, f"body {side} base scale")
        if lo != -hi:
            raise ValueError(
                f"{at} {side} {name} has lo {lo!r} and hi {hi!r}.\n"
                f"Expected -s and s: the base is mapped by the symmetric scale s "
                f"({BASE_TRANSFORM}), so a still base is exactly 0.\n"
                "Rebuild the file with this tree's tools/build_stats.py."
            )
        if side == "action" and rows <= 0:
            raise ValueError(
                f"{at} fits the action {name} scale {hi!r} on base_scale_rows {rows}.\n"
                "Expected the moving rows it was fitted on, at least one; a scale over no "
                "moving rows is a still base, which is constant.\n"
                "Rebuild the file with this tree's tools/build_stats.py."
            )
    for dim in _columns(BODY_POSTURE):
        _range_dim(key, None, BODY_DIM_NAMES[dim], f"body {side} dim", float(low[dim]),
                   float(high[dim]), bool(constant[dim]), bool(absent[dim]), at)
    required = {
        "none": _columns(BODY_BASE),
        "imu_yaw": (BODY_BASE.start, BODY_BASE.start + 1),
        "command": _columns(BODY_BASE) if side == "state" else (),
    }.get(base, ())
    required += tuple(range(BODY_TORSO.start + len(torso), BODY_TORSO.stop))
    required += tuple(range(BODY_HEAD.start + len(head), BODY_HEAD.stop))
    held = [BODY_DIM_NAMES[dim] for dim in required if not absent[dim]]
    if held:
        raise ValueError(
            f"{at} states {side} dims {held} present under base {base!r}, torso "
            f"{list(torso)} and head {list(head)}.\n"
            "Expected them absent: a base source none has no base, imu_yaw only the yaw, a "
            "command no base state (D5), and a torso or head only its declared joints.\n"
            "Fix the key's body in config/embodiments.yaml, or rebuild the stats."
        )
    if base == "still":
        fitted = [BODY_DIM_NAMES[dim] for dim in _columns(BODY_BASE)
                  if not (absent[dim] or constant[dim])]
        if fitted:
            raise ValueError(
                f"{at} reads a still base and fits {side} scales on {fitted}.\n"
                "Expected a still base constant at 0 or absent.\n"
                "Fix the key's body in config/embodiments.yaml, or rebuild the stats."
            )


def _read_entry(key: str, raw: dict, header: dict, file: str) -> Entry:
    where = f"{file} embodiments[{key!r}]"
    slots = _read_slots(raw, where)
    joints = _declaration(_field(raw, "joints", where), "joints", where)
    hand = _declaration(_field(raw, "hand", where), "hand", where)
    action, low, high, constant, absent, mask, gripper_valid = _read_action(
        key, raw, slots, joints, hand, where)
    state_low, state_high, state_constant, state_absent = _read_state(
        key, raw, slots, joints, hand, where)
    body = _read_body(key, raw, where)
    fix_raw = _field(raw, "ee_frame_fix", where)
    fix = None
    if fix_raw is not None:
        fix = _matrix(fix_raw, 3, 3, "ee_frame_fix", where, nullable=False)
        gram = fix.T @ fix - np.eye(3)
        if float(np.abs(gram).max()) > FRAME_FIX_TOLERANCE \
                or abs(float(np.linalg.det(fix)) - 1.0) > FRAME_FIX_TOLERANCE:
            raise ValueError(
                f"{where} ee_frame_fix {fix_raw!r} is not a proper rotation.\n"
                "Expected an orthonormal 3x3 with determinant 1.\n"
                "State R_c exactly, e.g. Rz(+90) = [[0,-1,0],[1,0,0],[0,0,1]]."
            )
    clause = _field(raw, "clause", where)
    if not isinstance(clause, str) or not clause.strip():
        raise ValueError(
            f"{where} states clause {clause!r}.\n"
            "Expected the embodiment sentence the prompt carries for this key.\n"
            "Take it from config/embodiments.yaml."
        )
    action_from = _field(action, "from", where)
    from_groups = action.get("from_groups")
    if action_from is None and from_groups is not None:
        raise ValueError(
            f"{where} states action from_groups {from_groups!r} and no action.from.\n"
            "Expected from_groups only beside the source key it copies from.\n"
            "Rebuild the file with this tree's tools/build_stats.py."
        )
    return Entry(
        key=key, slots=slots, joints=joints, hand=hand, low=low, high=high, constant=constant,
        absent=absent, mask=mask, gripper_valid=tuple(bool(flag) for flag in gripper_valid),
        action_from=None if action_from is None else str(action_from),
        state_low=state_low, state_high=state_high, state_constant=state_constant,
        state_absent=state_absent, body=body, ee_frame_fix=fix, clause=clause,
        stage=str(_field(raw, "stage", where)), identity=entry_identity(raw, header),
        action_from_groups=None if action_from is None else inherited_groups(from_groups, where),
    )


def _read_record(name: str, raw: dict, entries: dict[str, Entry], file: str) -> DatasetRecord:
    where = f"{file} datasets[{name!r}]"
    for field_name in (*RECORD_FIELDS, *MASK_FIELDS):
        _field(raw, field_name, where)
    keys = tuple(str(key) for key in raw["keys"])
    for key in keys:
        if key not in entries:
            raise ValueError(
                f"{refusal_prefix(3)} {where} names key {key!r}, which the file does not hold.\n"
                f"Expected one of {sorted(entries)}.\n"
                "Rebuild the file from the manifest this run trains."
            )
        for slot in raw["slots"]:
            entries[key].slot_index(int(slot))
    embodiment = str(raw["embodiment"])
    if embodiment != PER_EPISODE and keys != (embodiment,):
        raise ValueError(
            f"{where} states embodiment {embodiment!r} with keys {list(keys)}.\n"
            f"Expected keys [{embodiment!r}], or embodiment {PER_EPISODE!r}.\n"
            "Rebuild the file with this tree's tools/build_stats.py."
        )
    gripper = dict(raw["gripper"])
    if gripper.get("method") not in GRIPPER_METHODS:
        raise ValueError(
            f"{where} states gripper method {gripper.get('method')!r}.\n"
            f"Expected one of {GRIPPER_METHODS}.\n"
            "Rebuild the file with this tree's tools/build_stats.py."
        )
    slots = tuple(int(slot) for slot in raw["slots"])
    n = len(slots)
    constant = _flags(raw["constant"], n, ACTION_DIM, "constant", where)
    drot = raw["drot"]
    if drot not in set(DROT_RECORD.values()):
        raise ValueError(
            f"{where} states drot {drot!r}.\n"
            f"Expected one of {sorted(set(DROT_RECORD.values()))} (DROT_RECORD).\n"
            "Rebuild the file with this tree's tools/build_stats.py."
        )
    blocked = raw["history_blocked"]
    if isinstance(blocked, bool) or not isinstance(blocked, int) or blocked < 0:
        raise ValueError(
            f"{where} states history_blocked {blocked!r}.\n"
            "Expected the count of the directory's blocked positions, 0 or more.\n"
            "Rebuild the file with this tree's tools/build_stats.py."
        )
    record = DatasetRecord(
        directory=name, embodiment=embodiment, keys=keys, slots=slots, gripper=gripper,
        unit_factors=raw["unit_factors"],
        control_hz=None if raw["control_hz"] is None else float(raw["control_hz"]),
        inputs=record_inputs(raw), constant=tuple(tuple(bool(flag) for flag in row)
                                        for row in constant.tolist()),
        drot=str(drot), history_blocked=int(blocked),
        columns=_flags(raw["columns"], n, ACTION_DIM, "columns", where),
        state_dims=_flags(raw["state_dims"], n, STATE_DIM, "state_dims", where),
        body_columns=_flag_vector(raw["body_columns"], "body_columns", where),
        body_state_dims=_flag_vector(raw["body_state_dims"], "body_state_dims", where),
    )
    _check_masks(record, [entries[key] for key in keys], constant, where)
    return record


def _refuse_mask(where: str, said: str, expected: str) -> ValueError:
    return ValueError(
        f"{where} {said}.\n"
        f"Expected {expected}.\n"
        "Rebuild the file with this tree's tools/build_stats.py; the masks are what the "
        "loader trains under and the server rebuilds."
    )


def _check_masks(record: DatasetRecord, entries: list[Entry], constant: np.ndarray,
                 where: str) -> None:
    """A record's masks inside its keys' entries and its own constants, drot and gripper."""
    masked_gripper = record.gripper.get("method") == "masked"
    for arm, slot in enumerate(record.slots):
        at = f"{where} slot {slot}"
        decodable = np.any([entry.decodable(slot) for entry in entries], axis=0)
        holdable = np.any([entry.state_valid(slot) for entry in entries], axis=0)
        columns, dims = record.columns[arm], record.state_dims[arm]
        names = [DIM_NAMES[c] for c in np.flatnonzero(columns & ~decodable)]
        if names:
            raise _refuse_mask(at, f"trains columns {names} its keys cannot decode",
                               "the columns a key decodes (neither constant nor absent)")
        names = [DIM_NAMES[c] for c in np.flatnonzero(columns & constant[arm])]
        if names:
            raise _refuse_mask(at, f"trains columns {names} it states constant",
                               "a directory's constant columns outside its trained columns")
        if record.drot in UNTRAINED_DROT and bool(columns[ACTION_DROT].any()):
            raise _refuse_mask(at, f"trains drot under drot {record.drot!r}",
                               "no drot column trained where the drot carries no target")
        if masked_gripper and bool(columns[GRIPPER_COLUMN] or dims[STATE_APERTURE]):
            raise _refuse_mask(at, "masks its gripper and trains its column or aperture",
                               "the gripper column and the aperture invalid under a masked "
                               "gripper")
        names = [STATE_DIM_NAMES[d] for d in np.flatnonzero(dims & ~holdable)]
        if names:
            raise _refuse_mask(at, f"holds state dims {names} its keys state absent",
                               "the state dims a key holds")
        for group, span in (("position", STATE_POSITION), ("rotation", STATE_ROTATION)):
            if bool(dims[span].any()) and not bool(dims[span].all()):
                raise _refuse_mask(at, f"holds part of the state {group} "
                                       f"{dims[span].astype(int).tolist()}",
                                   f"the {group} all valid or all invalid (C7-6)")
    present = [entry.body for entry in entries if entry.body.present]
    if not present:
        if bool(record.body_columns.any() or record.body_state_dims.any()):
            raise _refuse_mask(where, "holds body columns or body state dims under keys with "
                                      "no body", "both body masks all false for an absent body")
        return
    decodable = np.any([body.decodable() for body in present], axis=0)
    holdable = np.any([~body.state_absent for body in present], axis=0)
    names = [BODY_DIM_NAMES[d] for d in np.flatnonzero(record.body_columns & ~decodable)]
    if names:
        raise _refuse_mask(where, f"trains body columns {names} its keys cannot decode",
                           "the body columns a key decodes")
    names = [BODY_DIM_NAMES[d] for d in np.flatnonzero(record.body_state_dims & ~holdable)]
    if names:
        raise _refuse_mask(where, f"holds body state dims {names} its keys state absent",
                           "the body state dims a key holds")
    if bool(record.body_state_dims.all()):
        raise _refuse_mask(where, "holds all 8 body state dims valid",
                           "at most 7: a measured body with all 8 valid, a still base and a "
                           "mid-range posture would equal the no-history input")


def _torch():
    import torch
    return torch


def _is_tensor(value: Any) -> bool:
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(value, torch.Tensor)


def _dims(flags: Any, width: int, name: str) -> np.ndarray:
    array = np.asarray(flags.cpu() if _is_tensor(flags) else flags)
    if array.shape != (width,) or array.dtype != np.bool_:
        raise ValueError(
            f"{name} is {array.shape} {array.dtype}, expected ({width},) bool.\n"
            "One flag per physical column.\n"
            "Pass the sample's dim_valid for this slot (or the body's)."
        )
    return array


def _tensor_of(x, width: int, what: str):
    torch = _torch()
    values = torch.as_tensor(x, dtype=torch.float64)
    if values.dim() < 1 or int(values.shape[-1]) != width:
        raise ValueError(
            f"{what} are {tuple(values.shape)}, expected [..., {width}].\n"
            f"The v7 record is {width} wide ({ACTION_RECORD}, {STATE_RECORD}, {BODY_RECORD}).\n"
            "Pass one slot of the loader's record, or the body's."
        )
    return values


def _map(column, lo: float, hi: float, *, symmetric: bool):
    """Physical -> model units: x / s for a symmetric base dim, else the q01/q99 line."""
    if symmetric:
        return column / hi
    return 2.0 * (column - lo) / (hi - lo) - 1.0


def _unmap(column, lo: float, hi: float, *, symmetric: bool):
    if symmetric:
        return column * hi
    return (column + 1.0) / 2.0 * (hi - lo) + lo


def _bounds_or_raise(key: str, slot: int | None, name: str, lo: float, hi: float, what: str):
    if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
        raise _degenerate(key, slot, name, lo, hi, what)


def _range_columns(values, valid: np.ndarray, columns, low, high, names, *, key, slot,
                   alpha: float, what: str, symmetric: tuple[int, ...] = ()):
    """Map the valid range columns; returns (y, ok) with ok per column (finite and in band)."""
    torch = _torch()
    bound = 1.0 + 2.0 * float(alpha)
    y = torch.zeros_like(values)
    ok = torch.ones(values.shape, dtype=torch.bool, device=values.device)
    for column in columns:
        if not valid[column]:
            continue
        lo, hi = float(low[column]), float(high[column])
        _bounds_or_raise(key, slot, names[column], lo, hi, what)
        span = hi - lo
        values_column = values[..., column]
        finite = torch.isfinite(values_column)
        inside = (values_column >= lo - alpha * span) & (values_column <= hi + alpha * span)
        ok[..., column] = finite & inside
        scaled = _map(values_column, lo, hi, symmetric=column in symmetric)
        y[..., column] = torch.where(finite, scaled, torch.zeros_like(values_column)) \
            .clamp(-bound, bound)
    return y, ok


def _group_flags(ok, groups: dict[str, slice]):
    """Per column, the flag of its group: every column of the group ok."""
    torch = _torch()
    in_band = torch.ones_like(ok)
    for span in groups.values():
        flag = ok[..., span].all(dim=-1, keepdim=True)
        in_band[..., span] = flag.expand_as(ok[..., span])
    return in_band


def normalise_actions(x, entry: Entry, slot: int, *, alpha: float, dims_valid):
    """Physical targets [..., K, 26] -> (y float32 [..., K, 26], in_band bool [..., K, 26]).

    Valid range columns (joints, dpos, drot, hand: RANGE_COLUMNS, T2) map to
    y = 2 (x - lo) / (hi - lo) - 1 per (key, slot, column), clamped to
    +-(1 + 2 alpha); the gripper column (g_c) maps to 2 g_c - 1 in [-1, 1].
    in_band holds, per column, its group's flag (joints, EE, hand): a group is
    in band at a step when each valid range column in it lies inside
    [lo - alpha span, hi + alpha span] and each valid column in it is finite.
    A non-finite value becomes 0 and takes only its own group at that step out
    of band. Invalid columns are 0.
    """
    torch = _torch()
    row = entry.slot_index(slot)
    values = _tensor_of(x, ACTION_DIM, "targets")
    valid = _dims(dims_valid, ACTION_DIM, "dims_valid")
    wrong = valid & ~entry.decodable(slot)
    if wrong.any():
        names = [DIM_NAMES[column] for column in np.flatnonzero(wrong)]
        raise ValueError(
            f"key {entry.key!r} slot {slot} columns {names} are marked valid and the entry "
            "cannot decode them (constant, absent or a masked gripper).\n"
            "Expected dims_valid inside the entry's decodable columns.\n"
            "Clear the columns in the sample's dim_valid."
        )
    y, ok = _range_columns(values, valid, RANGE_COLUMNS, entry.low[row], entry.high[row],
                           DIM_NAMES, key=entry.key, slot=int(slot), alpha=alpha,
                           what="action dim")
    if valid[GRIPPER_COLUMN]:
        column = values[..., GRIPPER_COLUMN]
        finite = torch.isfinite(column)
        ok[..., GRIPPER_COLUMN] = finite
        mapped = torch.where(finite, 2.0 * column - 1.0, torch.zeros_like(column))
        y[..., GRIPPER_COLUMN] = mapped.clamp(-1.0, 1.0)
    return y.to(torch.float32), _group_flags(ok, ACTION_GROUPS)


def denormalise(y, entry: Entry, slot: int, *, alpha: float):
    """Model output [..., K, 26] -> (x float32 [..., K, 26], emitted [26] bool, clips).

    Range columns are clipped to +-(1 + 2 alpha) and mapped back; a constant
    column returns lo; an absent column is NaN and not emitted; the gripper
    returns g_c = clip((y + 1) / 2, 0, 1) where the key trains it (CX 05).
    clips counts the values each clip moved, over emitted columns.
    """
    torch = _torch()
    row = entry.slot_index(slot)
    values = _tensor_of(y, ACTION_DIM, "model outputs")
    bound = 1.0 + 2.0 * float(alpha)
    x = torch.full_like(values, float("nan"))
    emitted = np.zeros(ACTION_DIM, dtype=bool)
    clips = {"range": 0, "gripper": 0}
    for column in RANGE_COLUMNS:
        if bool(entry.absent[row, column]):
            continue
        emitted[column] = True
        lo, hi = float(entry.low[row, column]), float(entry.high[row, column])
        if bool(entry.constant[row, column]):
            x[..., column] = lo
            continue
        values_column = values[..., column]
        clips["range"] += int((values_column.abs() > bound).sum())
        x[..., column] = _unmap(values_column.clamp(-bound, bound), lo, hi, symmetric=False)
    if entry.gripper_valid[row]:
        emitted[GRIPPER_COLUMN] = True
        g = (values[..., GRIPPER_COLUMN] + 1.0) / 2.0
        clips["gripper"] += int(((g < 0.0) | (g > 1.0)).sum())
        x[..., GRIPPER_COLUMN] = g.clamp(0.0, 1.0)
    return x.to(torch.float32), emitted, clips


def _refuse_partial(values, key: str, slot: int | None, absent: np.ndarray, names, what: str,
                    groups: tuple = ()) -> None:
    """A finite value on a dimension the key has no range for, or a split atomic group."""
    torch = _torch()
    finite = torch.isfinite(values)
    given = [names[dim] for dim in np.flatnonzero(absent) if bool(finite[..., int(dim)].any())]
    at = "" if slot is None else f" slot {slot}"
    if given:
        raise ValueError(
            f"key {key!r}{at} has no range for {what} dims {given} and was given measured "
            "values there.\n"
            "Expected NaN on every dimension the key's entry states absent: the model never "
            "saw a measured value there.\n"
            "Mark the dimension invalid (NaN) in the record, or rebuild the stats with it."
        )
    for name, span in groups:
        part = finite[..., span]
        if bool((part.any(dim=-1) & ~part.all(dim=-1)).any()):
            raise ValueError(
                f"key {key!r}{at} was given a {what} record with part of its {name} finite.\n"
                f"Expected the {name} all finite or all NaN: it is one group (C7-6).\n"
                "Build the record with neo_vla.data.state, which writes the group whole."
            )


def normalise_state(record, entry: Entry, slot: int, *, alpha: float):
    """Physical state records [..., 29] -> (model-input y float32 [..., 29], clipped).

    Joints, position and hand map per (key, slot, dim) to 2 (v - lo) / (hi -
    lo) - 1, clipped to +-(1 + 2 alpha) with the clipped values counted; a
    constant dim maps to 0. The 6D passes through, or becomes the 6D of
    R @ R_c under the key's end-effector fix; the aperture a becomes 2 a - 1.
    NaN stays NaN (an invalid dimension), so the mask is isfinite(y). Refuses a
    finite value where the key's entry states the dimension absent, and a
    record holding part of its position or part of its 6D.
    """
    values = _tensor_of(record, STATE_DIM, "state records")
    row = entry.slot_index(slot)
    _refuse_partial(values, entry.key, int(slot), entry.state_absent[row], STATE_DIM_NAMES,
                    "state", (("position", STATE_POSITION), ("6D", STATE_ROTATION)))
    out, over = _state_input(values, entry, slot, alpha=alpha)
    return out, int(over.sum())


def _state_input(values, entry: Entry, slot: int, *, alpha: float):
    """normalise_state's arithmetic: (y float32 [..., 29], clipped-range mask [..., 29]).

    Every bound, and the frame fix, is built on the record's own device: the
    policy calls this on the batch's device and the server on its own, and a
    CPU bound against a CUDA record is an error on every measured state. Holds
    no host synchronisation, so it runs on any device the record is on.
    """
    torch = _torch()
    from .state import ee_rotation_6d, rotation_from_6d

    row = entry.slot_index(slot)
    device = values.device
    bound = 1.0 + 2.0 * float(alpha)
    ranged = np.zeros(STATE_DIM, dtype=bool)
    ranged[list(STATE_RANGE)] = True
    ranged &= ~entry.state_absent[row]
    constant = entry.state_constant[row] & ranged
    lo = np.where(ranged, entry.state_low[row], 0.0)
    span = np.where(ranged & ~constant, entry.state_high[row] - entry.state_low[row], 1.0)
    lo_t = torch.as_tensor(lo, dtype=torch.float64, device=device)
    span_t = torch.as_tensor(span, dtype=torch.float64, device=device)
    ranged_t = torch.as_tensor(ranged, device=device)
    constant_t = torch.as_tensor(constant, device=device)
    finite = torch.isfinite(values)
    scaled = 2.0 * (values - lo_t) / span_t - 1.0
    over = (scaled.abs() > bound) & finite & ranged_t & ~constant_t
    mapped = torch.where(constant_t, torch.zeros_like(scaled), scaled.clamp(-bound, bound))
    nan = torch.full_like(values, float("nan"))
    out = torch.where(ranged_t & finite, mapped, nan)
    six = values[..., STATE_ROTATION]
    if entry.ee_frame_fix is not None:
        fix = torch.as_tensor(entry.ee_frame_fix, dtype=torch.float64, device=device)
        six = ee_rotation_6d(rotation_from_6d(six) @ fix)
    # A 6D or aperture the key states absent stays NaN even when the caller
    # skipped normalise_state's refusal: the model never saw one there.
    rotation_ok = finite[..., STATE_ROTATION].all(dim=-1, keepdim=True)
    if bool(entry.state_absent[row, STATE_ROTATION].any()):
        rotation_ok = torch.zeros_like(rotation_ok)
    out[..., STATE_ROTATION] = torch.where(rotation_ok, six, nan[..., STATE_ROTATION])
    aperture_ok = finite[..., STATE_APERTURE]
    if bool(entry.state_absent[row, STATE_APERTURE]):
        aperture_ok = torch.zeros_like(aperture_ok)
    aperture = values[..., STATE_APERTURE]
    out[..., STATE_APERTURE] = torch.where(aperture_ok, (2.0 * aperture - 1.0).clamp(-1.0, 1.0),
                                           nan[..., STATE_APERTURE])
    return out.to(torch.float32), over


def canonical_gripper(g, raw_open: float, raw_closed: float):
    """Raw gripper values -> (g_c clipped to [0, 1], how many values the clip moved).

    g_c = (g - raw_open) / (raw_closed - raw_open), so 0 is open and 1 closed
    whatever the recording's polarity. Takes a float, a numpy array or a tensor
    and returns the same kind; a NaN stays NaN and is not counted.
    """
    opened, closed = float(raw_open), float(raw_closed)
    if not (math.isfinite(opened) and math.isfinite(closed)) or opened == closed:
        raise ValueError(
            f"gripper raw_open {raw_open!r} and raw_closed {raw_closed!r}.\n"
            "Expected two different finite values; their order is the polarity.\n"
            "Measure the polarity with tools/measure_gripper_polarity.py."
        )
    span = closed - opened
    if _is_tensor(g):
        value = (g.to(g.dtype if g.is_floating_point() else _torch().float64) - opened) / span
        moved = int(((value < 0.0) | (value > 1.0)).sum())
        return value.clamp(0.0, 1.0), moved
    if isinstance(g, np.ndarray):
        value = (g.astype(np.float64) - opened) / span
        moved = int(np.count_nonzero((value < 0.0) | (value > 1.0)))
        return np.clip(value, 0.0, 1.0), moved
    value = (float(g) - opened) / span
    if math.isnan(value):
        return value, 0
    clipped = min(max(value, 0.0), 1.0)
    return clipped, int(clipped != value)


def gripper_raw(g_c, record):
    """Canonical g_c -> the recording's raw gripper value, the only inverse there is.

    record is a DatasetRecord or its json dict; the server's copy of it is what
    the client holds against its own manifest (R4) before it sends a command.
    """
    gripper = record.gripper if isinstance(record, DatasetRecord) else record["gripper"]
    opened, closed = float(gripper["raw_open"]), float(gripper["raw_closed"])
    return opened + g_c * (closed - opened)


def _rows(targets, width: int, steps_valid, what: str, key: str, slot) -> tuple:
    values = np.asarray(targets.cpu() if _is_tensor(targets) else targets, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != width:
        raise ValueError(
            f"targets are {values.shape}, expected [..., {width}].\n"
            f"Pass the loader's targets for one {what}.\n"
            "The fit gate measures one record under one entry."
        )
    flat = values.reshape(-1, width)
    keep = np.ones(flat.shape[0], dtype=bool)
    if steps_valid is not None:
        keep = np.asarray(steps_valid.cpu() if _is_tensor(steps_valid) else steps_valid,
                          dtype=bool).reshape(-1)
    if not keep.any():
        at = "" if slot is None else f" slot {slot}"
        raise ValueError(
            f"fit_check for key {key!r}{at} was given 0 valid {what}-rows.\n"
            "Expected at least one; a gate over nothing passes on no evidence.\n"
            "Draw windows of this key before checking its fit."
        )
    return flat, keep


def _fit_field(flat, rows, columns, low, high, *, alpha: float, symmetric: tuple = ()):
    """One field's verdict over its rows: per-column p99 |y| and the flagged share."""
    flagged = np.zeros(int(rows.sum()), dtype=bool)
    p99 = {}
    ok = True
    for column in columns:
        lo, hi = float(low[column]), float(high[column])
        span = hi - lo
        values = flat[rows, column]
        finite = np.isfinite(values)
        flagged |= ~finite | (values < lo - alpha * span) | (values > hi + alpha * span)
        scaled = np.abs(_map(values[finite], lo, hi, symmetric=column in symmetric))
        p99[column] = float(np.percentile(scaled, 99)) if scaled.size else None
        if p99[column] is None or not FIT_P99_BAND[0] <= p99[column] <= FIT_P99_BAND[1]:
            ok = False
    share = float(flagged.mean())
    if share > FIT_FLAGGED_MAX:
        ok = False
    return p99, share, ok


def _fit_report(flat, fields: dict, width: int, low, high, *, alpha, symmetric=()) -> FitReport:
    p99_all: list = [None] * width
    report = {}
    for name, (columns, rows) in fields.items():
        p99, share, ok = _fit_field(flat, rows, columns, low, high, alpha=alpha,
                                    symmetric=symmetric)
        for column, value in p99.items():
            p99_all[column] = value
        report[name] = {"columns": list(columns), "rows": int(rows.sum()),
                        "flagged_share": share, "ok": ok}
    worst = max(one["flagged_share"] for one in report.values())
    rows = int(max(one["rows"] for one in report.values()))
    return FitReport(p99_abs_y=p99_all, flagged_share=worst,
                     ok=all(one["ok"] for one in report.values()), rows=rows, fields=report)


def fit_check(targets, entry: Entry, slot: int, *, alpha: float, dims_valid,
              steps_valid=None) -> FitReport:
    """Whether an arm's own targets fit a key's entry (R14), per field.

    targets are physical [..., 26]; steps_valid [...] picks the arm-rows that
    count. Each field (joints, EE range, hand) with a valid range column is
    judged on its own: every valid range column's p99 |y| inside FIT_P99_BAND,
    and at most FIT_FLAGGED_MAX of the rows with a value of that field outside
    the alpha band or non-finite. The gripper is exempt. Refuses n = 0 rows
    and a call with no valid range column (a gate over nothing).
    """
    row = entry.slot_index(slot)
    flat, keep = _rows(targets, ACTION_DIM, steps_valid, "arm", entry.key, int(slot))
    valid = _dims(dims_valid, ACTION_DIM, "dims_valid")
    for column in RANGE_COLUMNS:
        if valid[column]:
            _bounds_or_raise(entry.key, int(slot), DIM_NAMES[column],
                             float(entry.low[row, column]), float(entry.high[row, column]),
                             "action dim")
    fields = {name: (tuple(c for c in columns if valid[c]), keep)
              for name, columns in FIT_FIELDS.items()}
    fields = {name: value for name, value in fields.items() if value[0]}
    if not fields:
        raise ValueError(
            f"fit_check for key {entry.key!r} slot {slot} was given no valid range column.\n"
            "Expected at least one joint, EE or hand column; the gripper alone is exempt, so a "
            "gate over it passes on no evidence.\n"
            "Leave a key whose draws train no range column out of the fit gate."
        )
    return _fit_report(flat, fields, ACTION_DIM, entry.low[row], entry.high[row], alpha=alpha)


def require_fit(report: FitReport, *, where: str, run_frames: set[str] | None = None,
                lineage_frames: set[str] | None = None) -> None:
    """Refuse (R14) a record whose targets do not fit the entry it would train under."""
    if report.ok:
        return
    failed = sorted(name for name, one in report.fields.items() if not one["ok"])
    # Only start-up supplies inherited-key context; stored corpus checks stay strict.
    waived = {}
    if (run_frames is not None and lineage_frames is not None
            and run_frames <= lineage_frames):
        for name in failed:
            one = report.fields[name]
            values = {c: report.p99_abs_y[c] for c in one["columns"]}
            low = {c: v for c, v in values.items()
                   if v is not None and v < FIT_P99_BAND[0]}
            if (low and one["flagged_share"] <= FIT_FLAGGED_MAX
                    and all(v is not None and math.isfinite(v) and v <= FIT_P99_BAND[1]
                            for v in values.values())):
                waived[name] = low
        if failed and len(waived) == len(failed) and report.flagged_share <= FIT_FLAGGED_MAX:
            for name, low in waived.items():
                logging.getLogger(__name__).warning(
                    "R14 inherited-key waiver: %s field %s columns %s values %s; "
                    "run frames %s; lineage frames %s",
                    where, name, list(low), list(low.values()),
                    sorted(run_frames), sorted(lineage_frames))
            return
    context = (f"\nInherited key: run frames {sorted(run_frames)}; lineage frames "
               f"{sorted(lineage_frames) if lineage_frames is not None else 'missing'}."
               if run_frames is not None else "")
    raise ValueError(
        f"{refusal_prefix(14)} {where} fails the fit gate over {report.rows} rows in fields "
        f"{failed or ['all']}: p99 |y| {report.p99_abs_y}, worst flagged share "
        f"{report.flagged_share:.4f}.\n"
        f"Expected p99 |y| in {list(FIT_P99_BAND)} on every valid range column and at most "
        f"{FIT_FLAGGED_MAX} of each field's rows outside the band.\n"
        "Measure the key over this record's frame policies (V6 4.7(b)), or use its own key."
        + context
    )


# The body: its maps, the base's symmetric scale on moving rows, its fit gate.
# The base columns map by x / s, so a still base is exactly 0 on every path.

BASE_SYMMETRIC = _columns(BODY_BASE)


def _refuse_body_columns(entry: Entry, valid: np.ndarray) -> None:
    wrong = valid & ~entry.body.decodable()
    if wrong.any():
        names = [BODY_DIM_NAMES[dim] for dim in np.flatnonzero(wrong)]
        state = "with no body" if not entry.body.present else "whose body cannot decode them"
        raise ValueError(
            f"key {entry.key!r} ({state}) was given body columns {names} as valid.\n"
            "Expected body dims_valid inside the body's decodable columns (neither constant "
            "nor absent; none for a key with no body).\n"
            "Clear the columns in the sample's body_dim_valid."
        )


def normalise_body_actions(x, entry: Entry, *, alpha: float, dims_valid):
    """Physical body targets [..., K, 8] -> (y float32 [..., K, 8], in_band bool [..., K, 8]).

    The base maps by its symmetric scale, y = x / s, so a still base is exactly
    0; torso and head map by q01/q99 per (key, dim). Every valid column is
    clamped to +-(1 + 2 alpha). in_band holds per column its group's flag
    (base, posture), as normalise_actions does for the arm. Invalid columns are
    0; a key with no body decodes nothing, so a valid column is refused.
    """
    torch = _torch()
    body = entry.body
    values = _tensor_of(x, BODY_DIM, "body targets")
    valid = _dims(dims_valid, BODY_DIM, "body dims_valid")
    _refuse_body_columns(entry, valid)
    y, ok = _range_columns(values, valid, range(BODY_DIM), body.action_low, body.action_high,
                           BODY_DIM_NAMES, key=entry.key, slot=None, alpha=alpha,
                           what="body dim", symmetric=BASE_SYMMETRIC)
    return y.to(torch.float32), _group_flags(ok, BODY_GROUPS)


def denormalise_body(y, entry: Entry, *, alpha: float):
    """Body model output [..., K, 8] -> (x float32 [..., K, 8], emitted [8] bool, clips).

    Columns are clipped to +-(1 + 2 alpha) and mapped back (the base as y * s);
    a constant column returns its value (a still base, 0); an absent column,
    and every column of a key with no body, is NaN and not emitted. clips
    counts the values the clip moved, over emitted columns.
    """
    torch = _torch()
    body = entry.body
    values = _tensor_of(y, BODY_DIM, "body model outputs")
    bound = 1.0 + 2.0 * float(alpha)
    x = torch.full_like(values, float("nan"))
    emitted = np.zeros(BODY_DIM, dtype=bool)
    clips = {"range": 0}
    for dim in range(BODY_DIM):
        if bool(body.action_absent[dim]):
            continue
        emitted[dim] = True
        lo, hi = float(body.action_low[dim]), float(body.action_high[dim])
        if bool(body.action_constant[dim]):
            x[..., dim] = lo
            continue
        column = values[..., dim]
        clips["range"] += int((column.abs() > bound).sum())
        x[..., dim] = _unmap(column.clamp(-bound, bound), lo, hi,
                             symmetric=dim in BASE_SYMMETRIC)
    return x.to(torch.float32), emitted, clips


def normalise_body_state(record, entry: Entry, *, alpha: float):
    """Physical body records [..., 8] -> (model-input y float32 [..., 8], clipped).

    The base (the last step's motion) maps by its symmetric scale and torso
    and head by q01/q99, clipped to +-(1 + 2 alpha) and counted; a constant dim
    maps to 0, so a still base reads 0. NaN stays NaN, so the mask is
    isfinite(y). Refuses a finite value on a dim the key's body states absent,
    every dim of a key with no body included.
    """
    values = _tensor_of(record, BODY_DIM, "body state records")
    _refuse_partial(values, entry.key, None, entry.body.state_absent, BODY_DIM_NAMES,
                    "body state")
    out, over = _body_state_input(values, entry, alpha=alpha)
    return out, int(over.sum())


def _body_state_input(values, entry: Entry, *, alpha: float):
    """normalise_body_state's arithmetic, on the record's device and without host sync."""
    torch = _torch()
    body = entry.body
    device = values.device
    bound = 1.0 + 2.0 * float(alpha)
    present = ~body.state_absent
    constant = body.state_constant & present
    mapped_dims = present & ~constant
    symmetric = np.zeros(BODY_DIM, dtype=bool)
    symmetric[BODY_BASE] = True
    lo = np.where(mapped_dims, body.state_low, 0.0)
    hi = np.where(mapped_dims, body.state_high, 1.0)
    span = np.where(mapped_dims, hi - lo, 1.0)

    def on_device(array, dtype=torch.float64):
        return torch.as_tensor(array, dtype=dtype, device=device)

    finite = torch.isfinite(values)
    scaled = torch.where(on_device(symmetric, torch.bool), values / on_device(hi),
                         2.0 * (values - on_device(lo)) / on_device(span) - 1.0)
    present_t, constant_t = on_device(present, torch.bool), on_device(constant, torch.bool)
    over = (scaled.abs() > bound) & finite & present_t & ~constant_t
    mapped = torch.where(constant_t, torch.zeros_like(scaled), scaled.clamp(-bound, bound))
    out = torch.where(present_t & finite, mapped, torch.full_like(values, float("nan")))
    return out.to(torch.float32), over


def weighted_quantile(values, weights, q: float) -> float:
    """The lowest value whose normalised cumulative weight reaches q; NaN for no values.

    The estimator every bound in the file is measured with (tools/build_stats,
    g25_ana.py wquant), here so the base scale and the tools share one.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.shape != weights.shape or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError(
            f"weighted_quantile was given {values.size} values and {weights.size} weights, "
            "or a weight that is negative or not finite.\n"
            "Expected one finite, non-negative weight per value.\n"
            "Pass the rows' weights as the stats stage computes them."
        )
    if values.size == 0:
        return float("nan")
    if float(weights.sum()) <= 0.0:
        raise ValueError(
            f"weighted_quantile was given {values.size} values whose weights sum to 0.\n"
            "Expected a positive total weight; a quantile of nothing is not a bound.\n"
            "Leave rows of zero weight out of the call."
        )
    order = np.argsort(values, kind="stable")
    ordered, cumulative = values[order], np.cumsum(weights[order])
    cumulative = cumulative / cumulative[-1]
    return float(ordered[min(int(np.searchsorted(cumulative, q, side="left")),
                             ordered.size - 1)])


def base_moving(steps, *, command: bool = False) -> np.ndarray:
    """Which base steps move, by the census motion rule: [..., N, 3] -> [..., N] bool.

    steps are one sequence's per-step base values along the second-to-last
    axis (dx, dy in cm per step and dyaw in rad per step, after held odometry
    was resampled). A command source moves where any of its commands is not 0.
    Otherwise a step moves when the window of BASE_MOTION's window_steps steps
    centred on it (moved inside the sequence at its ends) carries the base more
    than planar_cm in the plane or turns it more than dyaw_rad (sums of the
    per-step values). A step with no finite value never moves.
    """
    values = np.asarray(steps, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] != BODY_BASE.stop - BODY_BASE.start:
        raise ValueError(
            f"base steps are {values.shape}, expected [..., N, 3].\n"
            "One sequence of base_dx, base_dy (cm/step) and base_dyaw (rad/step).\n"
            "Pass the body record's base columns, steps along the second-to-last axis."
        )
    finite = np.isfinite(values)
    recorded = finite.any(axis=-1)
    clean = np.where(finite, values, 0.0)
    if command:
        return (np.abs(clean) > 0.0).any(axis=-1) & recorded
    count = values.shape[-2]
    window = int(BASE_MOTION["window_steps"])
    starts = np.clip(np.arange(count) - window // 2, 0, max(count - window, 0))
    ends = np.minimum(starts + window, count)
    total = np.concatenate([np.zeros_like(clean[..., :1, :]), np.cumsum(clean, axis=-2)],
                           axis=-2)
    sums = total[..., ends, :] - total[..., starts, :]
    planar = np.hypot(sums[..., 0], sums[..., 1])
    turned = np.abs(sums[..., 2])
    return ((planar > BASE_MOTION["planar_cm"]) | (turned > BASE_MOTION["dyaw_rad"])) & recorded


@dataclass(frozen=True)
class BaseScale:
    """A key's fitted base, per dim: its scale s (None unless fitted) and its status.

    status is fitted (s = weighted q99 of |x| over moving rows, at or above
    its floor), constant (recorded, but never moving, or below the floor: a
    still base, 0) or absent (never recorded). rows counts the moving rows read.
    """

    scale: tuple
    status: tuple
    rows: int

    def bounds(self) -> dict:
        """The base dims of a body block: low, high, constant and absent lists of 3."""
        low, high, constant, absent = [], [], [], []
        for scale, status in zip(self.scale, self.status):
            low.append(-scale if status == BASE_FITTED else 0.0 if status == BASE_CONSTANT
                       else None)
            high.append(scale if status == BASE_FITTED else 0.0 if status == BASE_CONSTANT
                        else None)
            constant.append(status == BASE_CONSTANT)
            absent.append(status == BASE_ABSENT)
        return {"low": low, "high": high, "constant": constant, "absent": absent}


def fit_base_scale(steps, moving, weights=None) -> BaseScale:
    """The symmetric base scale of one key from its pooled steps [N, 3] (spec 4).

    moving [N] are the rows base_moving marked (or a command's non-zero rows);
    weights [N] what each row weighs (equal when None). Per dim, s is the
    weighted q99 of |x| over the moving rows where x is finite, so the still
    majority cannot shrink it; a dim recorded but never moving, or whose s is
    below BASE_SCALE_FLOOR (jitter), is constant at 0; a dim never recorded is
    absent.
    """
    values = np.asarray(steps, dtype=np.float64)
    width = BODY_BASE.stop - BODY_BASE.start
    moves = np.asarray(moving, dtype=bool).reshape(-1)
    if values.ndim != 2 or values.shape[1] != width or moves.shape != (values.shape[0],):
        raise ValueError(
            f"fit_base_scale was given steps {values.shape} and moving {moves.shape}.\n"
            f"Expected steps [N, {width}] and one moving flag per step.\n"
            "Pool the key's base steps and mark them with base_moving."
        )
    weight = np.ones(values.shape[0]) if weights is None \
        else np.asarray(weights, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    scale, status = [], []
    for dim in range(width):
        use = moves & finite[:, dim] & (weight > 0)
        if not finite[:, dim].any():
            scale.append(None), status.append(BASE_ABSENT)
            continue
        fitted = weighted_quantile(np.abs(values[use, dim]), weight[use], 0.99) \
            if use.any() else float("nan")
        if not (math.isfinite(fitted) and fitted >= BASE_SCALE_FLOOR[dim]):
            scale.append(None), status.append(BASE_CONSTANT)
            continue
        scale.append(float(fitted)), status.append(BASE_FITTED)
    rows = int((moves & finite.any(axis=1)).sum())
    return BaseScale(scale=tuple(scale), status=tuple(status), rows=rows)


def fit_check_body(targets, entry: Entry, *, alpha: float, dims_valid, steps_valid=None,
                   moving=None) -> FitReport:
    """Whether a key's own body targets fit its body entry (R14), per field.

    targets are physical [..., 8]; steps_valid [...] picks the rows that count
    and moving [...] which of them move the base (base_moving). The posture
    field is judged on every kept row, the base field only on kept rows that
    move, as its scale was fitted, so the still majority neither passes nor
    fails it. Refuses n = 0 rows, a base judged with no moving rows or no
    moving flags, and a call with no valid body column.
    """
    body = entry.body
    flat, keep = _rows(targets, BODY_DIM, steps_valid, "body", entry.key, None)
    valid = _dims(dims_valid, BODY_DIM, "body dims_valid")
    _refuse_body_columns(entry, valid)
    fields = {}
    base = tuple(dim for dim in BODY_FIT_FIELDS["base"] if valid[dim])
    if base:
        moves = None if moving is None else np.asarray(
            moving.cpu() if _is_tensor(moving) else moving, dtype=bool).reshape(-1)
        if moves is None or moves.shape != keep.shape or not (keep & moves).any():
            counted = None if moves is None or moves.shape != keep.shape \
                else int((keep & moves).sum())
            raise ValueError(
                f"fit_check_body for key {entry.key!r} judges base columns "
                f"{[BODY_DIM_NAMES[dim] for dim in base]} on {counted} moving rows.\n"
                "Expected moving flags for every row (base_moving) and at least one kept row "
                "that moves: the base scale was fitted on moving rows only.\n"
                "Draw windows of this key that move its base, or leave the base out."
            )
        fields["base"] = (base, keep & moves)
    posture = tuple(dim for dim in BODY_FIT_FIELDS["posture"] if valid[dim])
    if posture:
        fields["posture"] = (posture, keep)
    if not fields:
        raise ValueError(
            f"fit_check_body for key {entry.key!r} was given no valid body column.\n"
            "Expected at least one base or posture column; a gate over nothing passes on no "
            "evidence.\n"
            "Leave a key whose draws train no body column out of the body fit gate."
        )
    return _fit_report(flat, fields, BODY_DIM, body.action_low, body.action_high, alpha=alpha,
                       symmetric=BASE_SYMMETRIC)
