"""The one thing no tool can discover: what a directory's columns mean.

A layout file names each directory's source, its cameras, where each arm's pose
sits in the raw state vector and which flat columns hold its action. That
knowledge is established when a dataset is added and reviewed as a file; the
stages read it and never infer it, so a wrong column is a diff rather than a
plausible tensor.

Resolution is source defaults, then a per-dataset override, then the name.
A directory no rule claims raises, and so does one whose rule requires an entry
it has none of: a default would hand an unrecognised corpus another corpus's
state layout, which is the one failure that produces numbers nothing
downstream can question.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from neo_vla.constants import (
    ARM_JOINTS,
    COLUMN_BODY,
    COLUMN_BODY_COMMAND,
    COLUMN_HAND,
    COLUMN_JOINTS,
    COLUMN_JOINTS_COMMAND,
    HAND_JOINTS,
)
from neo_vla.data import registry
from neo_vla.data.actions import ROTATION_CHANNEL_POLICY
from neo_vla.data.body import BASE_SOURCES, HEAD_JOINTS, TORSO_JOINTS
from neo_vla.data.state import PART_FIELDS, parse_target

LAYOUT_SCHEMA_VERSION = 1

# Blocks a per-directory entry replaces whole rather than merging into. Merging
# a half-stated state block onto another source's would produce a layout no one
# wrote. joints, hand, body and stored are the v7 blocks (V7_BLOCKS).
BLOCKS = ("views", "state", "action", "columns", "extrinsics", "metadata_gates",
          "frame_ref", "trajectory_weights", "prompt", "frequency", "embodiment",
          "gripper", "joints", "hand", "body", "stored")

# The v7 blocks. joints and hand name, per arm slot, how many
# of the field's columns that arm fills, in which stored column and unit, and
# the target rule; body states whether the key has one and what it records;
# stored says where the v7 columns live: in the directory's own parquet under
# the v7 ingest stamp, or in the stage-1 sidecar the cache stage merges.
V7_BLOCKS = ("joints", "hand", "body", "stored")
STORED_ORIGINS = ("ingest", "sidecar")

# Model units per stored unit, by the unit a layout states: joints are in
# radians (a source in degrees says deg and is converted by its factor), a hand
# is radians or the dimensionless unit of a source that stores it normalised.
PART_UNITS = {"joints": {"rad": 1.0, "deg": math.pi / 180.0},
              "hand": {"rad": 1.0, "deg": math.pi / 180.0, "unit": 1.0}}
PART_COLUMN = {"joints": COLUMN_JOINTS, "hand": COLUMN_HAND}
PART_WIDEST = {"joints": ARM_JOINTS, "hand": HAND_JOINTS}
PART_NAMES = {"joints": "names", "hand": "order"}

# How a layout's body says a part records nothing: the converters write the
# word none, the key table null.
BODY_NONE = ("none", None)

# The three v6 policy blocks. Each is a statement a later
# stage acts on -- declare_frequency, build_embodiment, measure_gripper_polarity
# -- and a layout without one resolves it to None, which those stages refuse by
# name rather than defaulting: a default frequency or key would be a claim
# about a recording nobody made.
POLICY_BLOCKS = ("frequency", "embodiment", "gripper")

# The two stored columns every stage reads by name. A source that spells them
# otherwise says so in its own block; there is no default, because a column
# that is not there reads as a dataset with no state at all. The v7 columns a
# columns block may also name have one spelling each (neo_vla.constants), and
# the part blocks are what decide whether they are read.
COLUMN_FIELDS = ("state", "action")
STORED_COLUMN_FIELDS = ("joints", "hand", "body")
STORED_SPELLING = {"joints": COLUMN_JOINTS, "hand": COLUMN_HAND, "body": COLUMN_BODY}

# What a directory contributes to the rendered text when its layout says
# nothing: no robot description and no frame sentence.
NO_PROMPT = {"embodiment_clauses": [], "frame_clause": {"camera": "", "robot": ""}}

# Extrinsics kinds that give a rotation to express a chunk in a camera frame.
# A mounted camera stores its offset from the arm it is bolted to rather than a
# pose; the loader composes the pose, so it counts the same here.
CAMERA_GEOMETRY_KINDS = ("episode", "constant", "mounted")

# The suffixes an assign rule may remove to find a directory's dataset entry,
# spelled with runs of N where the directory carries digits. Only a shard
# suffix qualifies, because only shards of one dataset share one measurement.
# registry.SHARD_SUFFIXES also lists _unified, which is a different action
# convention rather than another spelling of the same one, so a rule naming it
# would give a dataset the factors measured for a different recording.
NORMALISED_SUFFIXES = ("_shardNNofNN",)


@dataclass(frozen=True)
class Layout:
    """One directory's established meaning, as the layout file states it."""

    directory: str
    name: str
    source: str
    views: tuple[dict[str, Any], ...]
    state: dict[str, Any] | None
    action: dict[str, Any]
    columns: dict[str, str]
    extrinsics: dict[str, Any]
    metadata_gates: tuple[dict[str, Any], ...]
    frame_ref: str | None
    prompt: dict[str, Any]
    trajectory_weights: str | None
    frequency: dict[str, Any] | None = None
    embodiment: dict[str, Any] | None = None
    gripper: dict[str, Any] | None = None
    joints: dict[str, Any] | None = None
    hand: dict[str, Any] | None = None
    body: dict[str, Any] | None = None
    stored: str | None = None
    v7: tuple[str, ...] = ()


def load(path: Path) -> dict[str, Any]:
    """Read a layout file and refuse a version this tool does not know."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    version = document.get("schema_version")
    if version != LAYOUT_SCHEMA_VERSION:
        raise ValueError(
            f"layout schema_version is {version!r}\n"
            f"expected {LAYOUT_SCHEMA_VERSION}\n"
            f"rewrite the layout file against the current fields"
        )
    return document


def _rule_for(document: dict[str, Any], directory: str) -> dict[str, Any]:
    matching = [
        rule for rule in document["assign"] if directory.startswith(rule["prefix"])
    ]
    if not matching:
        raise KeyError(
            f"no layout rule matches {directory}\n"
            f"expected a prefix in the layout file's assign list\n"
            f"add the directory's source there before scanning it"
        )
    return max(matching, key=lambda rule: len(rule["prefix"]))


def name_of(directory: str, parts: int) -> str:
    """The canonical name of a directory: its last `parts` path components.

    RH20T repeats thirty basenames across a hundred and sixty-six directories,
    so one component is not a name there. Two are, and the count is stated per
    rule rather than guessed per source.
    """
    pieces = [piece for piece in directory.split("/") if piece]
    return "_".join(pieces[-max(1, parts):])


@lru_cache(maxsize=None)
def _shard_pattern(suffix: str) -> re.Pattern[str]:
    """The regex one shard suffix spells, its runs of N standing for digits."""
    if suffix not in NORMALISED_SUFFIXES:
        raise ValueError(
            f"shard_suffix is {suffix!r}\n"
            f"expected one of {list(NORMALISED_SUFFIXES)}\n"
            f"a suffix marking another action convention is another dataset"
        )
    return re.compile("(?:%s)$" % re.sub("N+", r"\\d+", re.escape(suffix)))


def dataset_key(directory: str, rule: dict[str, Any]) -> str:
    """A directory -> the datasets entry it is looked up by.

    A rule naming no shard suffix is looked up by the directory itself. One
    naming a shard suffix has that suffix removed from the last path component,
    so the shards of a dataset share the one entry its factors were measured
    for rather than repeating that measurement once per shard.
    """
    suffix = rule.get("shard_suffix")
    if suffix is None:
        return directory
    head, separator, last = directory.rpartition("/")
    return head + separator + _shard_pattern(suffix).sub("", last, count=1)


def _override_for(document: dict[str, Any], directory: str, rule: dict[str, Any]) -> dict[str, Any]:
    """The entry governing one directory, refusing an excluded or absent one."""
    key = dataset_key(directory, rule)
    reason = document.get("excluded", {}).get(key)
    if reason is not None:
        raise KeyError(
            f"{directory} resolves to the excluded dataset {key}\n"
            f"held back because {reason}\n"
            f"drop the directory from the list, or establish its unit factors"
        )
    override = document.get("datasets", {}).get(key, {})
    if not override and rule.get("entry_required"):
        raise KeyError(
            f"{directory} resolves to dataset {key}, which this layout does not state\n"
            f"expected an entry under datasets, or one under excluded saying why not\n"
            f"measure the dataset's unit factors and write its entry"
        )
    return override


def resolve(document: dict[str, Any], directory: str) -> Layout:
    """A root-relative directory -> the layout that governs it."""
    rule = _rule_for(document, directory)
    source = rule["source"]
    if source not in document["sources"]:
        raise KeyError(
            f"layout rule for {directory} names source {source!r}\n"
            f"expected one of {sorted(document['sources'])}\n"
            f"add the source block or fix the rule"
        )
    merged: dict[str, Any] = dict(document["sources"][source])
    override = _override_for(document, directory, rule)
    for block in BLOCKS:
        if block in override:
            merged[block] = override[block]
    if "state" not in merged:
        raise KeyError(
            f"source {source!r} states no state block\n"
            f"expected a state layout or an explicit null\n"
            f"write null when the columns were never established"
        )
    missing = [field for field in COLUMN_FIELDS
               if field not in merged.get("columns", {})]
    if missing:
        raise KeyError(
            f"source {source!r} names no {missing[0]} column\n"
            f"expected a columns block naming {list(COLUMN_FIELDS)}\n"
            f"name the stored columns; nothing here assumes them"
        )
    return Layout(
        directory=directory,
        name=override.get("name") or name_of(directory, int(rule.get("name_parts", 1))),
        source=source,
        views=tuple(merged["views"]),
        state=merged["state"],
        action=merged["action"],
        columns={field: str(merged["columns"][field])
                 for field in (*COLUMN_FIELDS, *STORED_COLUMN_FIELDS)
                 if merged["columns"].get(field) is not None},
        extrinsics=merged.get("extrinsics", {}),
        metadata_gates=tuple(merged.get("metadata_gates", ())),
        frame_ref=merged.get("frame_ref"),
        prompt=merged.get("prompt", NO_PROMPT),
        trajectory_weights=override.get("trajectory_weights"),
        **{block: merged.get(block) for block in POLICY_BLOCKS},
        **{block: merged.get(block) for block in V7_BLOCKS},
        v7=tuple(block for block in V7_BLOCKS if block in merged),
    )


def claim(claimed: dict[str, str], layout: Layout) -> str:
    """Record that a layout's name is this directory's, refusing a second claimant.

    A stage keys its output by layout name, so two directories under one name
    would leave one silently overwriting the other.
    """
    holder = claimed.setdefault(layout.name, layout.directory)
    if holder != layout.directory:
        raise ValueError(
            f"name {layout.name} is claimed by {holder} and {layout.directory}\n"
            f"expected one directory per name\n"
            f"raise name_parts in the layout rule or set an explicit name"
        )
    return layout.name


def policy_block(layout: Layout, block: str) -> dict[str, Any]:
    """One of the v6 policy blocks of a layout, refusing a layout without it."""
    if block not in POLICY_BLOCKS:
        raise KeyError(
            f"{block!r} is not a policy block\n"
            f"expected one of {list(POLICY_BLOCKS)}\n"
            f"ask for a block the layout file states"
        )
    found = getattr(layout, block)
    if not isinstance(found, dict):
        raise KeyError(
            f"{layout.name} ({layout.directory}) states no {block} block\n"
            f"expected a layout file with a {block} policy for source {layout.source!r} or "
            f"this dataset (config/layouts/gd7.json carries oxe3src_gd6.json's)\n"
            f"write the {block} policy into the layout file"
        )
    return dict(found)


def arm_count(layout: Layout) -> int:
    """How many arms the action layout fills, which is the slot count minus one."""
    return len(layout.action["arms"])


def _refuse_v7(layout: Layout, block: str, said: str, expected: str) -> ValueError:
    return ValueError(
        f"{layout.name} ({layout.directory}) {block} block {said}\n"
        f"expected {expected}\n"
        f"fix the {block} block in the layout file"
    )


def require_v7(layout: Layout) -> None:
    """Refuse a layout that states no v7 blocks: a v6 layout is refused by name.

    joints, hand and body must each be stated, null or a statement: a v6
    layout says nothing about them, and reading its silence as "not recorded"
    would build a v7 corpus whose joints are masked for no recorded reason.
    """
    missing = [block for block in ("joints", "hand", "body") if block not in layout.v7]
    if missing:
        raise KeyError(
            f"{layout.name} ({layout.directory}) states no {missing} block\n"
            f"expected the v7 blocks joints, hand and body (each null or a statement), "
            f"as tools/make_layout_v7.py and the v7 converters write them\n"
            f"a v6 layout says nothing about joints, hand or body; regenerate it in v7"
        )


def arm_parts(layout: Layout, field: str) -> list[dict[str, Any] | None]:
    """Per arm, in arm order, the manifest's {count, unit_factor, target} for joints or hand.

    None for an arm the block does not name, and for every arm when the block
    is null. The block names its stored column (one spelling), its unit (the
    factor is the unit's, never assumed), its target rule
    and per arm slot the count it fills with that many names.
    """
    if field not in PART_FIELDS:
        raise KeyError(
            f"{field!r} is not a part field\n"
            f"expected one of {list(PART_FIELDS)}\n"
            f"ask for joints or hand"
        )
    require_v7(layout)
    block = layout.joints if field == "joints" else layout.hand
    arms = arm_count(layout)
    if block is None:
        return [None] * arms
    if not isinstance(block, dict):
        raise _refuse_v7(layout, field, f"is {block!r}", "null or an object")
    if block.get("column") != PART_COLUMN[field]:
        raise _refuse_v7(layout, field, f"names column {block.get('column')!r}",
                         f"{PART_COLUMN[field]!r}, the one spelling of the stored column")
    units = PART_UNITS[field]
    if block.get("unit") not in units:
        raise _refuse_v7(layout, field, f"states unit {block.get('unit')!r}",
                         f"one of {sorted(units)}")
    target = str(block.get("target"))
    try:
        parse_target(target, field)
    except ValueError as error:
        raise _refuse_v7(layout, field, f"states target {target!r}",
                         str(error).splitlines()[1]) from error
    if not str(block.get("provenance") or "").strip():
        raise _refuse_v7(layout, field, "states no provenance",
                         "the sentence saying what established the columns")
    found: list[dict[str, Any] | None] = [None] * arms
    stated = block.get("arms")
    if not isinstance(stated, list) or not stated:
        raise _refuse_v7(layout, field, f"names arms {stated!r}",
                         "a list of {slot, count, names} per arm that records the field")
    names = PART_NAMES[field]
    for arm in stated:
        slot, count = arm.get("slot"), arm.get("count")
        if isinstance(slot, bool) or not isinstance(slot, int) or not 0 <= slot < arms \
                or found[slot] is not None:
            raise _refuse_v7(layout, field, f"names arm slot {slot!r}",
                             f"a distinct slot of the layout's {arms} arms")
        if isinstance(count, bool) or not isinstance(count, int) \
                or not 1 <= count <= PART_WIDEST[field]:
            raise _refuse_v7(layout, field, f"states slot {slot} count {count!r}",
                             f"1 to {PART_WIDEST[field]}")
        listed = arm.get(names)
        if not isinstance(listed, list) or len(listed) != count \
                or not all(isinstance(one, str) and one for one in listed):
            raise _refuse_v7(layout, field, f"states slot {slot} {names} {listed!r}",
                             f"{count} names, one per column in the stored order")
        found[slot] = {"count": count, "unit_factor": units[block["unit"]], "target": target}
    return found


def _body_count(layout: Layout, block: dict[str, Any], part: str, widest: int) -> int:
    value = block.get(part)
    if value in BODY_NONE:
        return 0
    if not isinstance(value, list) or not 1 <= len(value) <= widest \
            or not all(isinstance(one, str) and one for one in value):
        raise _refuse_v7(layout, "body", f"states {part} {value!r}",
                         f"'none' or 1 to {widest} joint names")
    return len(value)


def body_block(layout: Layout) -> dict[str, Any] | None:
    """The manifest's body block, {base, torso, head} counts, or None for a key with none.

    declared absent is a fixed base (the key's body is ABSENT); declared
    present names where the base reading comes from (data.body.BASE_SOURCES),
    the torso and head joints it records ('none' for a part it does not) and
    the stored column; a commanded base names the command column too. A
    present body that records nothing is one nobody measured (UNKNOWN), and
    still reads the stored column, all NaN.
    """
    require_v7(layout)
    block = layout.body
    if not isinstance(block, dict) or block.get("declared") not in ("absent", "present"):
        raise _refuse_v7(layout, "body", f"is {block!r}",
                         "{declared: absent} or {declared: present, base, torso, head, column}")
    if block["declared"] == "absent":
        return None
    base = block.get("base")
    if base not in BASE_SOURCES:
        raise _refuse_v7(layout, "body", f"reads its base from {base!r}",
                         f"one of {list(BASE_SOURCES)}")
    if block.get("column") != COLUMN_BODY:
        raise _refuse_v7(layout, "body", f"names column {block.get('column')!r}",
                         f"{COLUMN_BODY!r}, the one spelling of the stored column")
    command = block.get("command_column")
    if (base == "command") != (command is not None) or command not in (None, COLUMN_BODY_COMMAND):
        raise _refuse_v7(layout, "body", f"reads base {base!r} with command column {command!r}",
                         f"{COLUMN_BODY_COMMAND!r} exactly when the base is a command")
    if not str(block.get("provenance") or "").strip():
        raise _refuse_v7(layout, "body", "states no provenance",
                         "the sentence saying what established the body")
    return {"base": str(base), "torso": _body_count(layout, block, "torso", TORSO_JOINTS),
            "head": _body_count(layout, block, "head", HEAD_JOINTS)}


def stored_origin(layout: Layout) -> str | None:
    """Where a layout's v7 columns live: ingest, sidecar, or None when it reads none.

    A layout that reads joints, hand or a body must say which writer stamped
    them; the manifest stage then holds the directory to that stamp, so a v6
    output or a missing sidecar is refused rather than read as not recorded.
    """
    reads = any(part is not None for field in PART_FIELDS for part in arm_parts(layout, field))
    reads = reads or body_block(layout) is not None
    if not reads:
        if layout.stored is not None:
            raise _refuse_v7(layout, "stored", f"is {layout.stored!r} and no block reads a "
                             "stored column", "null or no stored block")
        return None
    if layout.stored not in STORED_ORIGINS:
        raise _refuse_v7(layout, "stored", f"is {layout.stored!r}",
                         f"one of {list(STORED_ORIGINS)}: the v7 ingest parquet or the "
                         "stage-1 sidecar")
    return str(layout.stored)


def manifest_columns(layout: Layout) -> dict[str, str | None]:
    """The manifest's columns block: state, action, and each v7 column a layout reads.

    A v7 column is named exactly when a part, a command target or the body
    reads it, and null otherwise (dataset._read_stored holds the same rule).
    A layout that spells one of them in its own columns block must use the
    one spelling.
    """
    for field, spelling in STORED_SPELLING.items():
        named = layout.columns.get(field)
        if named is not None and named != spelling:
            raise _refuse_v7(layout, "columns", f"names {field} {named!r}", repr(spelling))
    parts = {field: arm_parts(layout, field) for field in PART_FIELDS}
    body = body_block(layout)

    def reads(field: str, command: bool = False) -> bool:
        return any(part is not None and (not command or part["target"].startswith("command:"))
                   for part in parts[field])

    return {
        "state": layout.columns["state"], "action": layout.columns["action"],
        "joints": COLUMN_JOINTS if reads("joints") else None,
        "hand": COLUMN_HAND if reads("hand") else None,
        "joints_command": COLUMN_JOINTS_COMMAND if reads("joints", command=True) else None,
        "body": COLUMN_BODY if body is not None else None,
        "body_command": COLUMN_BODY_COMMAND if body is not None and body["base"] == "command"
        else None,
    }


def rotation_policy(layout: Layout) -> str:
    """What the loader does with this directory's rotation columns.

    The table is neo_vla.data.actions.ROTATION_CHANNEL_POLICY, so the manifest
    stage and the loader read one statement of it. A convention the table does
    not name is refused here, by name: there is no policy to fall back to, and
    the loader would refuse the word later with nothing to say where it came
    from.
    """
    convention = layout.action.get("rotation_delta_frame", "unresolved")
    if convention not in ROTATION_CHANNEL_POLICY:
        raise ValueError(
            f"{layout.name} ({layout.directory}) states rotation_delta_frame {convention!r}\n"
            f"expected one of {sorted(ROTATION_CHANNEL_POLICY)}\n"
            f"state the convention the measurement found, or unresolved to mask the rotation"
        )
    return ROTATION_CHANNEL_POLICY[convention]


def view_extrinsics(layout: Layout, key: str) -> dict[str, Any]:
    """Where one camera's pose comes from, or that it has none.

    The layout keys the poses by view key, because a directory's cameras are
    calibrated one at a time and a source may publish a pose for one of them
    and not the next. A camera the block does not name has no measured pose and
    cannot be drawn as a reference.
    """
    return dict(layout.extrinsics.get(key, {"kind": "none"}))


def has_camera_geometry(layout: Layout) -> bool:
    """Whether this layout alone allows a camera-frame sample.

    The half of the camera-mode decision that needs no camera roles: the layout
    does not state robot frame and some camera has a rotation per episode or per
    corpus. Every rotation convention has a camera-frame form: the translation turns
    with the frame, and the rotation is carried
    to the base frame first or is masked, so the convention only has to be one
    the table names. build_manifest adds the other half, that a camera with a
    pose is eligible by its role.
    """
    rotation_policy(layout)
    kinds = {
        view_extrinsics(layout, view["key"]).get("kind") for view in layout.views
    }
    return layout.frame_ref != "robot" and bool(kinds & set(CAMERA_GEOMETRY_KINDS))


def check_role(role: str, where: str) -> str:
    """Raise unless the role is one the loader's registry knows."""
    if role not in registry.VIEW_ROLES:
        raise ValueError(
            f"{where} has camera role {role!r}\n"
            f"expected one of {list(registry.VIEW_ROLES)}\n"
            f"use unknown when the role was never established"
        )
    return role
