"""Stage 9: the one file the dataset opens to learn what exists (manifest schema 3).

Folds the earlier stages into one document and holds no numbers of its own:
every field traces to the stage that produced it through the provenance header.
Per entry: the embodiment key or per-episode key arrays, control_hz, the
gripper record, the rotation convention and its source, the RH20T blocked
history; schema 3 adds per arm the joints and hand blocks, the body block, the
stored v7 columns and the stage-1 sidecar they come from. A directory whose
key declares joints, hand or a body must carry them under a v7 stamp (its
ingest meta/info.json, or a current sidecar): a v6 output is refused by its
missing stamp (R22), never read as "not recorded". Two keys
sharing a clause refuse (R9).

Usage:
    python tools/build_manifest.py --root /corpus --kept metadata/C/kept.jsonl \
        --scale metadata/C/scale.json --weights metadata/C/weights.json \
        --inventory metadata/C/inventory.jsonl --layouts L --corpus C \
        --frequency metadata/C/frequency.json --embodiment metadata/C/embodiment.json \
        --gripper metadata/C/gripper.json --prompt-template config/prompt_template/policy.yaml \
        --prompt-cache processed/C/prompt ... --out manifests/C.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import key_table
import layouts as layout_table
import stage
import unify
from corpus_stamps import (
    check_sidecar_stamp,
    ingest_stamp,
    read_sidecar_stamp,
    refuse_stamp,
    sidecar_rows,
)

from neo_vla.constants import MANIFEST_SCHEMA_VERSION, PROCESSED_DIR, refusal_prefix
from neo_vla.data.state import PART_FIELDS
from neo_vla.data import rays, registry
from neo_vla.data.conditioning import parse_frame_ref
from neo_vla.data.images import check_fits, snap_grid
from neo_vla.data.prompt import check_mode_agrees, load_template, stamp_of

logger = logging.getLogger("build_manifest")

REPO = Path(__file__).resolve().parent.parent

# A view with no calibration, written rather than left out. The loader reads
# this as a view that emits no rays, which is a statement; a missing field
# would be a manifest built before the intrinsics stage existed. Every block
# names the same four fields so that a reader asking which file a view's
# calibration came from gets an answer from every view.
NO_INTRINSICS = {"kind": "none", "provenance": None, "path": None, "path_source": None}

# The committed RH20T blocked-history list, which every manifest holding one of its
# directories must be built with (require_listing).
HISTORY_LIST = REPO / "tools" / "lists" / "rh20t_history_blocked_20260930.json"

# Conventions whose drot the loader converts from the arm's Euler or body
# rotation; a source whose arm has no state layout must
# name where that rotation is read (action.rotation_source).
NEEDS_ROTATION = ("euler_diff", "body")

# The stage outputs schema 2 folds in, per directory, by layout name.
V6_INPUTS = ("frequency", "embodiment", "gripper")

# Extrinsics sources whose rotations come from a file rather than from the
# episode metadata. The entry names that file so the cache stage rereads what
# this stage recorded, and this stage records it as an input for the same reason.
FILE_SOURCES = ("sidecar", "stage")


class Retired(argparse.Action):
    """A flag this stage no longer takes, refused by name with what replaced it."""

    def __call__(self, parser, namespace, values, option_string=None):
        parser.error(f"{option_string} is retired: {self.help}")


def repo_or_root(root: Path, name: str) -> Path:
    """A repository config path (config/..., tools/lists/...) or one under the root."""
    path = Path(name)
    if path.is_absolute():
        return path
    if (REPO / path).exists():
        return REPO / path
    return stage.under_root(root, name)


def frame_shape(columns: dict[str, Any], key: str) -> tuple[int, int]:
    """A view column's stored frame size, from the shape info.json records."""
    shape = columns[key]["shape"]
    if len(shape) < 2:
        raise ValueError(
            f"view {key} has shape {shape}\n"
            f"expected at least height and width\n"
            f"check the directory's meta/info.json feature block"
        )
    return int(shape[0]), int(shape[1])


def role_of(view: dict[str, Any], metadata: list[dict[str, Any]], where: str) -> str:
    """A view's camera role, resolved from metadata when the layout defers to it."""
    rule = view.get("role_from_metadata")
    if rule is None:
        return view["role"]
    seen = {bool(record.get(rule["key"])) for record in metadata}
    if len(seen) != 1:
        if "mixed" not in rule:
            raise ValueError(
                f"{where} camera role varies within the directory on {rule['key']}\n"
                f"expected one role for every kept episode\n"
                f"split the directory or give the rule a mixed role"
            )
        return rule["mixed"]
    return rule["true"] if seen.pop() else rule["false"]


def reference_slots(views: list[dict[str, Any]], eligible: frozenset) -> list[int]:
    """The slots of the cameras a sample may express its actions in, in order.

    A camera qualifies on two counts: its role admits it, and this stage gave
    it a pose. Several is what the loader draws between per sample, so this
    counts them rather than settling on one.
    """
    return sorted(
        int(view["slot"]) for view in views
        if view["role"] in eligible and view["extrinsics"]["kind"] != "none"
    )


@dataclass(frozen=True)
class Calibration:
    """The intrinsics stage's records, and the file each view's came from.

    records is keyed by directory and view, then by the episode index the
    source wrote; origin is the root-relative intrinsics file, which the entry
    carries so the cache stage reads the same records this stage judged.
    """

    records: dict[tuple[str, str], dict[int, dict[str, Any]]]
    origin: dict[tuple[str, str], str]


def read_intrinsics(paths: dict[str, Path]) -> Calibration:
    """The intrinsics stage's records, keyed by directory and view, then episode.

    Several files because build_intrinsics reads one source per run. A view
    whose records come from two of them raises: the entry names the file the
    cache stage re-reads, and two files give it no single answer.
    """
    records: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    origin: dict[tuple[str, str], str] = {}
    for name in sorted(paths):
        _, lines = stage.read_jsonl(paths[name])
        for record in lines:
            key = (str(record["dataset"]), str(record["view"]))
            if origin.setdefault(key, name) != name:
                raise ValueError(
                    f"{key[0]} view {key[1]} is calibrated by {origin[key]} and {name}\n"
                    f"expected one intrinsics file per directory and view\n"
                    f"build that source's calibration into a single file"
                )
            episodes = records.setdefault(key, {})
            episode = int(record["episode"])
            if episode in episodes:
                raise ValueError(
                    f"{key[0]} episode {episode} view {key[1]} has two calibrations\n"
                    f"expected one record per directory, episode and view\n"
                    f"fix the join in build_intrinsics for this source"
                )
            episodes[episode] = record
    return Calibration(records=records, origin=origin)


def intrinsics_of(
    found: dict[int, dict[str, Any]],
    episodes: list[int],
    *,
    where: str,
    slot: int,
    path: str,
    path_source: str | None,
) -> dict[str, Any]:
    """One view's calibration block: constant, one per episode, or none.

    Constant only when every kept episode of the view carries the same six
    numbers. One episode short of that is the per-episode form, because a
    directory-wide calibration would state a measurement for the episodes that
    have none, and those are exactly the ones the ray channel must stay quiet
    about.

    Either form names the file its records came from. That is what the cache
    stage rereads and what the verify stage collects the files to check from, so
    a constant view that named none would be checked by one stage of two.
    """
    records = [found[episode] for episode in episodes if episode in found]
    if not records:
        return dict(NO_INTRINSICS)
    if path_source is None:
        raise ValueError(
            f"{where} has calibration records and names no source file\n"
            f"expected the intrinsics file its records came from\n"
            f"pass that file to --intrinsics"
        )
    stated = sorted({str(record["provenance"]) for record in records})
    if len(stated) != 1:
        raise ValueError(
            f"{where} states provenance {stated}\n"
            f"expected one origin for the view's calibration\n"
            f"split the directory, or build its calibration from one tree"
        )
    misplaced = sorted({int(record["slot"]) for record in records} - {slot})
    if misplaced:
        raise ValueError(
            f"{where} is camera slot {slot} and its records name {misplaced}\n"
            f"expected the layout's slot\n"
            f"rebuild the intrinsics file against this layout"
        )
    distinct = {
        tuple(record[column] for column in rays.CALIBRATION_COLUMNS)
        for record in records
    }
    if len(distinct) == 1 and len(records) == len(episodes):
        numbers = distinct.pop()
        return {"kind": "constant", "provenance": stated[0], "path": None,
                "path_source": path_source,
                **dict(zip(rays.CALIBRATION_COLUMNS, numbers, strict=True))}
    return {"kind": "episode", "provenance": stated[0], "path": path,
            "path_source": path_source}


def build_views(
    layout: layout_table.Layout,
    columns: dict[str, Any],
    metadata: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    calibration: Calibration,
    episodes: list[int],
) -> list[dict[str, Any]]:
    """One directory's cameras, each with the grid its frames snap to."""
    built = []
    for view in layout.views:
        shape = frame_shape(columns, view["key"])
        tile = snap_grid(
            shape, short_side=args.canvas_short_side, token=args.canvas_token,
            cell=(args.canvas_cell[0], args.canvas_cell[1]),
        )
        try:
            check_fits(tile, cell=(args.canvas_cell[0], args.canvas_cell[1]))
        except ValueError as error:
            raise ValueError(
                f"{layout.name} view {view['key']} does not fit the canvas\n"
                f"{error}"
            ) from error
        role = layout_table.check_role(
            role_of(view, metadata, layout.name), f"{layout.name} view {view['key']}"
        )
        if int(view["slot"]) >= args.n_views:
            raise ValueError(
                f"{layout.name} view {view['key']} claims slot {view['slot']}\n"
                f"expected a slot below n_views {args.n_views}\n"
                f"renumber the layout's camera slots"
            )
        slot = int(view["slot"])
        key = (layout.directory, view["key"])
        mounted = view.get("arm_slot")
        built.append({
            "key": view["key"],
            "role": role,
            "slot": slot,
            "arm_slot": None if mounted is None else int(mounted),
            "prompt_noun": view.get("prompt_noun", "") if role != "unknown" else "",
            "shape": [shape[0], shape[1]],
            "grid": [tile.grid_h, tile.grid_w],
            "extrinsics": extrinsics_of(layout, view, args.cache_dir, slot),
            "intrinsics": intrinsics_of(
                calibration.records.get(key, {}), episodes,
                where=f"{layout.name} view {view['key']}", slot=slot,
                path=f"{args.cache_dir}/intrinsics/{layout.name}.{slot}.npy",
                path_source=calibration.origin.get(key),
            ),
        })
    return built


def arm_slot(arm: dict[str, Any], index: int, where: str) -> int:
    """The action slot one arm fills, which is its position in the list.

    Arm identity is positional -- the model reads arm row k for slot k -- so
    the slot a layout may state is a restatement of where the arm already is,
    and one that disagrees is a layout written against the numbering the
    reserved base slot used to force (index + 1).
    """
    stated = int(arm.get("slot", index))
    if stated != index:
        raise ValueError(
            f"{where} states arm slot {stated} at position {index}\n"
            f"expected slot {index}: arm k fills slot k and there is no base slot\n"
            f"renumber the layout's arms from zero"
        )
    return index


# The four pose fields of a state arm: all stated, or all null for a joints-only
# arm (BimanualYAM, Tien Kung), whose layout writes them null.
POSE_FIELDS = ("position_at", "rotation_at", "rotation_kind", "position_unit")


def state_block(arm: dict[str, Any], base_offset: dict[str, Any] | None,
                where: str) -> dict[str, Any]:
    """One arm's manifest state block: its pose (or none), gripper, provenance, offset."""
    pose = [arm.get(field) for field in POSE_FIELDS]
    if any(value is None for value in pose) and not all(value is None for value in pose):
        raise ValueError(
            f"{where} states {dict(zip(POSE_FIELDS, pose))}\n"
            f"expected all four pose fields, or all null for a joints-only arm\n"
            f"fix the layout's state arm"
        )
    posed = pose[0] is not None
    return {
        "position_at": int(arm["position_at"]) if posed else None,
        "rotation_at": int(arm["rotation_at"]) if posed else None,
        "rotation_kind": arm["rotation_kind"] if posed else None,
        "position_unit": arm["position_unit"] if posed else None,
        "gripper": arm.get("gripper"),
        "provenance": arm["provenance"],
        # Metres subtracted from the position per kept episode (LIBERO F3,
        # the LIBERO base-offset measurement); null for every base-frame source.
        "base_offset": base_offset if posed else None,
    }


def build_arms(layout: layout_table.Layout,
               base_offset: dict[str, Any] | None = None,
               parts: dict[str, list[dict[str, Any] | None]] | None = None,
               ) -> list[dict[str, Any]]:
    """One directory's arms: the state layout that was established and the v7 parts.

    An arm whose whole state layout is unestablished carries a null state; its
    joints and hand blocks (layouts.arm_parts, one per arm) sit beside it, so a
    source with no established pose can still train the joints its sidecar
    holds. parts None states no part on any arm.
    """
    arms = layout_table.arm_count(layout)
    parts = parts or {field: [None] * arms for field in PART_FIELDS}
    built = []
    if layout.state is None:
        stated = [None] * arms
        names = [arm.get("name", "single") for arm in layout.action["arms"]]
        where = [f"{layout.name} action arm {index}" for index in range(arms)]
        slots = [arm_slot(arm, index, where[index])
                 for index, arm in enumerate(layout.action["arms"])]
    else:
        stated = list(layout.state["arms"])
        names = [arm["name"] for arm in stated]
        where = [f"{layout.name} state arm {arm['name']!r}" for arm in stated]
        slots = [arm_slot(arm, index, where[index]) for index, arm in enumerate(stated)]
    if len(stated) != arms:
        raise ValueError(
            f"{layout.name} states {len(stated)} state arms and {arms} action arms\n"
            f"expected one state arm per action arm\n"
            f"fix the layout's arms"
        )
    for index, arm in enumerate(stated):
        built.append({
            "slot": slots[index], "name": names[index],
            "state": None if arm is None else state_block(arm, base_offset, where[index]),
            **{field: parts[field][index] for field in PART_FIELDS},
        })
    return built


def base_offset_of(name: str, source: str, offsets: dict[str, Any]) -> dict[str, Any] | None:
    """The base offset block --base-offsets states for a directory (LIBERO F3), or None.

    A LIBERO-family directory (stage.is_libero_family) without one is refused
    here, before its cache and prompt are built: verify_corpus check (x) and the
    launchers' corpus gate would refuse the finished corpus anyway, after the
    build. Every other source is in its robot's base frame and states none.
    """
    block = offsets.get(name)
    if block is None and stage.is_libero_family({"source": source, "name": name}):
        raise ValueError(
            f"LIBERO F3: {name} is a LIBERO-family directory (source {source}) and "
            f"--base-offsets states no base offset for it\nExpected every arm of a LIBERO "
            f"directory to state the per-episode base offset the loader subtracts; without "
            f"it the corpus trains on world-frame positions.\nMeasure the LIBERO base offsets "
            f"and pass --base-offsets.")
    return block


def build_action(layout: layout_table.Layout, gripper: dict[str, Any]) -> dict[str, Any]:
    """One directory's action block: layout, unit factors, rotation delta frame, gripper.

    Both unit factors are measurements the layout file states, and a layout
    missing either one has no unified action and cannot enter a manifest. The
    gripper record is measure_gripper_polarity's; the normaliser is not here.
    """
    convention = layout.action.get("rotation_delta_frame", "unresolved")
    policy = layout_table.rotation_policy(layout)
    arms = [{field: list(arm[field]) for field in ("dpos", "drot", "gripper")
             if arm.get(field) is not None} for arm in layout.action["arms"]]
    reads = {field: any(field in arm for arm in arms) for field in ("dpos", "drot")}
    # A joints-only source (no arm reads dpos or drot) states no factor, and
    # the loader refuses one stated for a field nobody reads; every other
    # source needs both, measured.
    if (reads["dpos"] or reads["drot"]) and not unify.factors_stated(layout.action):
        raise ValueError(
            f"{layout.name} states no measured unit factor\n"
            f"expected translation_cm_per_unit and rotation_rad_per_unit\n"
            f"measure them for this dataset or leave the dataset out"
        )
    source = layout.action.get("rotation_source")
    posed = layout.state is not None and any(
        arm.get("position_at") is not None for arm in layout.state["arms"])
    if convention in NEEDS_ROTATION and source is None and not posed:
        raise ValueError(
            f"{layout.name} has rotation_delta_frame {convention!r} and no state layout\n"
            f"expected action.rotation_source naming where the rotation is read\n"
            f"state the column, offset and Euler convention in the layout"
        )
    return {
        "flat_dim": int(layout.action["flat_dim"]),
        "translation_cm_per_unit": [
            float(value) for value in unify.translation_factor(layout.action)
        ] if reads["dpos"] else None,
        "rotation_rad_per_unit": float(layout.action["rotation_rad_per_unit"])
        if reads["drot"] else None,
        "rotation": policy,
        "rotation_delta_frame": convention,
        "rotation_source": source,
        # The source's own per-row flag. Without it in the manifest the loader
        # reads a row the recording marks as holding no action as a target.
        "valid_column": layout.action.get("valid_column"),
        "arms": arms,
        "base": layout.action.get("base"),
        "gripper": gripper_record(gripper),
    }


def gripper_record(gripper: dict[str, Any]) -> dict[str, Any]:
    """measure_gripper_polarity's record as the manifest states it.

    A masked gripper has no raw stroke; the record still states two numbers,
    because every client recomputes its record inputs from the manifest, so it
    states the identity pair (raw_open 0, raw_closed 1) that the method
    "masked" says is never used: the gripper dim is invalid for the directory.
    """
    if gripper["method"] == "masked":
        return {"raw_open": 0.0, "raw_closed": 1.0, "method": "masked"}
    return {"raw_open": float(gripper["raw_open"]), "raw_closed": float(gripper["raw_closed"]),
            "method": gripper["method"]}


def embodiment_block(stated: dict[str, Any], name: str) -> dict[str, Any]:
    """The manifest's embodiment field from build_embodiment's record."""
    block = stated["embodiment"]
    if block["kind"] == "constant":
        return {"kind": "constant", "key": block["key"]}
    if block["kind"] == "episode":
        return {"kind": "episode", "keys": list(block["keys"]), "path": block["path"],
                "ids_path": block["ids_path"]}
    raise ValueError(
        f"{name} embodiment kind {block['kind']!r}\n"
        f"expected constant or episode\n"
        f"re-run build_embodiment"
    )


def keys_of(block: dict[str, Any]) -> list[str]:
    return [block["key"]] if block["kind"] == "constant" else list(block["keys"])


def check_clauses(keys_table: dict[str, Any], used: set[str]) -> dict[str, str]:
    """Each used key's clause, refusing a key without one and two keys sharing one (R9)."""
    clauses: dict[str, str] = {}
    seen: dict[str, str] = {}
    for key in sorted(used):
        spec = keys_table.get(key)
        if not isinstance(spec, dict) or not spec.get("clause"):
            raise ValueError(
                f"key {key} has no clause in config/embodiments.yaml\n"
                f"expected one clause per key\n"
                f"add the key's clause"
            )
        clause = str(spec["clause"])
        if clause in seen:
            raise ValueError(
                f"{refusal_prefix(9)} keys {seen[clause]} and {key} share the clause "
                f"{clause!r}\nExpected one clause per key: under per-embodiment targets the "
                f"clause names the normaliser.\nGive each key its own clause.")
        seen[clause] = key
        clauses[key] = clause
    return clauses


def require_listing(listing: dict[str, Any] | None, names: set[str],
                    committed: Path = HISTORY_LIST) -> None:
    """Refuse a manifest holding a directory of the committed blocked-history list without it.

    The listed RH20T episodes train in the no-history form, and build_manifest
    applies the list only when --history-blocked names it; this refuses a manifest
    without it at build time, before a cache is built from it.
    """
    if listing is not None:
        return
    rows = json.loads(committed.read_text(encoding="utf-8"))["episodes"]
    held = sorted({row["dir"] for row in rows} & names)
    if held:
        raise ValueError(
            f"the kept set holds {len(held)} directories of {committed.name} (first "
            f"{held[0]}) and --history-blocked is not given\n"
            f"expected the committed list wherever its directories are kept\n"
            f"pass --history-blocked tools/lists/{committed.name}")


def blocked_positions(listing: dict[str, Any] | None, kept_of: dict[str, list[int]],
                      name_of: dict[str, str]) -> dict[str, list[int]]:
    """RH20T episodes whose draws take the no-history form, as kept positions per directory.

    Joined through the kept file exactly as R16 joins keys: (dir, meta
    episode_index) -> the kept position whose episode is that index. A listed
    pair the kept file does not hold refuses rather than dropping out.
    """
    if listing is None:
        return {}
    found: dict[str, list[int]] = {}
    directory_of = {name: directory for directory, name in name_of.items()}
    for row in listing["episodes"]:
        directory = directory_of.get(row["dir"])
        if directory is None:
            raise ValueError(
                f"the blocked-history list names {row['dir']}, which this manifest lacks\n"
                f"expected every listed directory among the kept ones\n"
                f"pass --history-blocked only for the corpus the list was measured on"
            )
        episodes = kept_of[directory]
        index = int(row["episode_index"])
        if index not in episodes:
            raise ValueError(
                f"{row['dir']} episode_index {index} is not a kept episode\n"
                f"expected the listed episode among {len(episodes)} kept ones\n"
                f"rebuild the list against this kept file"
            )
        found.setdefault(name_of[directory], []).append(episodes.index(index))
    return {name: sorted(positions) for name, positions in found.items()}


def build_columns(layout: layout_table.Layout) -> dict[str, str | None]:
    """Which stored column holds the pose, the action, and each v7 column a layout reads.

    Written into the manifest rather than assumed by the loader, so a source
    that names its columns otherwise is a layout entry and not a branch in the
    reader. The v7 columns have one spelling each and are null where no part,
    command target or body reads them (layouts.manifest_columns).
    """
    return layout_table.manifest_columns(layout)


def check_keys(name: str, keys: list[str], table: dict[str, Any],
               parts: dict[str, list[dict[str, Any] | None]],
               body: dict[str, Any] | None, origin: str | None) -> None:
    """A directory's v7 blocks against what each of its keys declares.

    A key that declares joints, hand or a body needs the directory's v7
    columns under a stamp (refused, never masked); a layout may
    not read a part its key does not declare; and the body block is the
    key's, or the part of it the directory records: absent for both, or the
    key's base source (or none) and at most its torso and head counts. A
    directory of a key whose head one variant does not record (RoboCOIN Leju
    V26, head identically 0: spec 2.3) states head 0; it never states more.
    """
    for key in keys:
        spec = table[key]
        declared = {"joints": key_table.joints_of(key, spec), "hand": key_table.hand_of(key, spec)}
        body_declared = key_table.body_counts(key_table.body_of(key, spec))
        if key_table.declares_stored(spec, key) and origin is None:
            raise refuse_stamp(
                name, f"is a directory of key {key}, which declares joints, hand or a body, and "
                      "its layout states no stored v7 columns",
                "the directory's v7 columns under their writer's stamp (stored ingest or sidecar)",
                "re-emit the directory in v7 or build its stage-1 sidecar, then state stored")
        for field in PART_FIELDS:
            if declared[field] is None and any(part is not None for part in parts[field]):
                raise ValueError(
                    f"{name} reads {field} and its key {key} declares none\n"
                    f"expected {field} only on a key that declares them "
                    f"(config/embodiments.yaml)\n"
                    f"declare the key's {field}, or drop the layout's {field} block")
        if not body_within(body, body_declared):
            raise ValueError(
                f"{name} states body {body} and its key {key} declares {body_declared}\n"
                f"expected the key's body: absent for both, or its base (or none) and at most "
                f"its torso and head counts\n"
                f"fix the layout's body block or the key's body in config/embodiments.yaml")


def body_within(body: dict[str, Any] | None, declared: dict[str, Any] | None) -> bool:
    """Whether a directory's body block is its key's, or the part of it the directory records."""
    if body is None or declared is None:
        return body is None and declared is None
    return (body["base"] in (declared["base"], "none")
            and int(body["torso"]) <= int(declared["torso"])
            and int(body["head"]) <= int(declared["head"]))


def sidecar_block(root: Path, folder: str, name: str, kept: list[tuple[int, int]],
                  columns: list[str]) -> dict[str, Any]:
    """The manifest's sidecar block for a directory whose v7 columns are a stage-1 sidecar.

    The sidecar is metadata/<corpus>/sidecars/<entry name>.parquet (never beside
    the source). Its stamp must be this format and writer version, the v7
    ingest layout and record stamps, this entry; it must have written every
    column the layouts read, and hold every kept episode with its rows (R16).
    The block names the sidecar's path.
    """
    path = stage.under_root(root, Path(folder) / f"{name}.parquet")
    relative = stage.relative(root, path)
    if not path.is_file():
        raise refuse_stamp(name, f"states its v7 columns are a sidecar and {relative} is not "
                                 "there", "the stage-1 sidecar of every sidecar directory",
                           "build it with the sidecar writer, then rebuild the manifest")
    stamp = read_sidecar_stamp(path)
    check_sidecar_stamp(stamp, name, columns, relative)
    rows = sidecar_rows(stamp, relative)
    for episode, length in kept:
        if rows.get(int(episode)) != int(length):
            held = rows.get(int(episode))
            raise ValueError(
                f"{refusal_prefix(16)} {name}: kept episode {episode} has {length} rows and "
                f"its sidecar {relative} holds "
                f"{'no such episode' if held is None else f'{held} rows of it'}\n"
                f"expected every kept episode with the directory's rows\n"
                f"rebuild the sidecar against the directory as it is now")
    return {"path": relative}


def extrinsics_of(
    layout: layout_table.Layout, view: dict[str, Any], cache_dir: str, slot: int
) -> dict[str, Any]:
    """Where the loader reads one camera's rotations, and where they came from.

    One array per camera rather than one per directory, because a source may
    publish a pose for one of its cameras and not the next, and the reference
    is drawn between the ones it did. The layout's own source, key and frame
    stay beside the path as the record of what was read to build that array,
    which is what the cache stage needs and the loader never parses.
    """
    spec = layout_table.view_extrinsics(layout, view["key"])
    if spec.get("kind", "none") == "none":
        return {"kind": "none", "path": None}
    if spec.get("source") in FILE_SOURCES:
        spec["path_source"] = spec.pop("path")
    spec["path"] = f"{cache_dir}/extrinsics/{layout.name}.{slot}.npy"
    return spec


def source_id_of(layout: layout_table.Layout) -> int:
    """The layout source's permanent id; a source the registry never numbered is refused by name."""
    try:
        return registry.source_id(layout.source)
    except ValueError as error:
        raise ValueError(f"layout {layout.name} ({layout.directory}): {error}") from None


def frame_ref_of(layout: layout_table.Layout, views: list[dict],
                 eligible: frozenset) -> str:
    """Camera frame only when a reference camera, geometry and a policy all exist.

    The geometry and the policy are layouts.has_camera_geometry, which the stats
    stage reads too; the reference camera is the half that needs the roles this
    stage has already resolved and the poses it has already placed.
    """
    if not layout_table.has_camera_geometry(layout):
        return "robot"
    return "camera" if reference_slots(views, eligible) else "robot"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--kept", required=True)
    parser.add_argument("--stats", action=Retired,
                        help="the normaliser is embodiments.json, named by the data config "
                             "and measured after this stage (V6 C15)")
    parser.add_argument("--scale", required=True, help="audit only; never a model input")
    parser.add_argument("--frequency", required=True, help="declare_frequency's output")
    parser.add_argument("--embodiment", required=True, help="build_embodiment's output")
    parser.add_argument("--gripper", required=True, help="measure_gripper_polarity's output")
    parser.add_argument("--embodiments", default="config/embodiments.yaml",
                        help="the key table (neo_vla_embodiment_keys_v2): clauses and the "
                             "joints, hand and body each key declares")
    parser.add_argument("--sidecars",
                        help="where the stage-1 sidecars are, metadata/<corpus>/sidecars under the "
                             "root; needed when a layout states stored sidecar")
    parser.add_argument("--history-blocked",
                        help="tools/lists/rh20t_history_blocked_*.json, for the pretrain corpus")
    parser.add_argument("--base-offsets",
                        help="measure_libero_base's output: per directory, the per-episode "
                             "base translation the loader subtracts (LIBERO F3)")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--intrinsics", action="append", default=[],
                        help="an intrinsics stage output; repeat once per source")
    parser.add_argument("--layouts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--schema-version", type=int, default=MANIFEST_SCHEMA_VERSION,
                        choices=[MANIFEST_SCHEMA_VERSION])
    parser.add_argument("--canvas-short-side", type=int, default=256)
    parser.add_argument("--canvas-token", type=int, default=32)
    parser.add_argument("--canvas-cell", type=int, nargs=2, default=[256, 352])
    parser.add_argument("--n-views", type=int, default=3)
    parser.add_argument("--cache-dir", help=f"the caches' folder; default {PROCESSED_DIR}/<corpus>")
    parser.add_argument("--prompt-template", required=True)
    parser.add_argument("--prompt-cache", required=True)
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--max-text-tokens", type=int, required=True)
    parser.add_argument("--intrinsics-mode", default="tokens",
                        choices=list(rays.INTRINSICS_MODES),
                        help="how the calibration reaches the model this corpus is for; "
                             "tokens, the one mode")
    args = parser.parse_args(argv)
    if args.cache_dir is None:
        args.cache_dir = f"{PROCESSED_DIR}/{args.corpus}"

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    root = stage.resolve_root(args.root)
    paths = {
        "inventory": stage.under_root(root, args.inventory),
        "kept": stage.under_root(root, args.kept),
        "scale": stage.under_root(root, args.scale),
        "frequency": stage.under_root(root, args.frequency),
        "embodiment": stage.under_root(root, args.embodiment),
        "gripper": stage.under_root(root, args.gripper),
        "weights": stage.under_root(root, args.weights),
        "layouts": stage.under_root(root, args.layouts),
        "prompt_template": stage.under_root(root, args.prompt_template),
    }
    # Keyed by the path the entries carry, so an entry's path_source and the
    # header's inputs name one string.
    intrinsics = {}
    for name in args.intrinsics:
        resolved = stage.under_root(root, name)
        intrinsics[stage.relative(root, resolved)] = resolved
    paths.update(intrinsics)
    out = stage.under_root(root, args.out)

    _, inventory = stage.read_jsonl(paths["inventory"])
    _, kept = stage.read_jsonl(paths["kept"])
    scale = stage.read_json(paths["scale"])["datasets"]
    v6 = {name: stage.read_json(paths[name])["datasets"] for name in V6_INPUTS}
    # Repository configs are not under the root, so the header records them by
    # name in its settings rather than among the root-relative inputs.
    configs: dict[str, Any] = {}
    table_path = repo_or_root(root, args.embodiments)
    keys_table = key_table.load(table_path)["keys"]
    configs["embodiments"] = {"path": args.embodiments}
    listing = None
    if args.history_blocked:
        listing_path = repo_or_root(root, args.history_blocked)
        listing = json.loads(listing_path.read_text(encoding="utf-8"))
        configs["history_blocked"] = {"path": args.history_blocked}
    offsets = {}
    if args.base_offsets:
        paths["base_offsets"] = stage.under_root(root, args.base_offsets)
        offsets = stage.read_json(paths["base_offsets"])["datasets"]
    weights = stage.read_json(paths["weights"])
    document = layout_table.load(paths["layouts"])
    calibration = read_intrinsics(intrinsics)

    columns_of = {record["dataset"]: record["columns"] for record in inventory}
    rows_of = {(record["dataset"], int(record["episode"])): int(record["rows"])
               for record in inventory}
    metadata_of: dict[str, list[dict[str, Any]]] = {}
    for record in inventory:
        metadata_of.setdefault(record["dataset"], []).append(record["metadata"])
    # In kept order, which is the order the cache stage writes every per-episode
    # artifact in and the position the loader reads them back by.
    kept_of: dict[str, list[int]] = {}
    for record in kept:
        if record["kind"] == "kept":
            kept_of.setdefault(record["dataset"], []).append(int(record["episode"]))
    directories = sorted(kept_of)
    stage.refuse_write_into_input(out, [root / name for name in directories])

    entries = []
    seen: dict[str, str] = {}
    used_keys: set[str] = set()
    stores: dict[str, int] = {}
    for directory in directories:
        layout = layout_table.resolve(document, directory)
        if layout.name in seen:
            raise ValueError(
                f"name {layout.name} is claimed by {seen[layout.name]} and {directory}\n"
                f"expected one directory per name\n"
                f"raise name_parts in the layout rule or set an explicit name"
            )
        seen[layout.name] = directory
        entry_weight = weights["datasets"].get(layout.name)
        if entry_weight is None:
            raise KeyError(
                f"{layout.name} has no weight\n"
                f"expected an entry in the weights stage output\n"
                f"re-run build_weights over the same kept file"
            )
        if not (entry_weight["weight"] >= 0.0):
            raise ValueError(
                f"{layout.name} has weight {entry_weight['weight']}\n"
                f"expected a finite weight of at least zero\n"
                f"fix the shares and re-run build_weights"
            )
        stated = {}
        for name in V6_INPUTS:
            record = v6[name].get(layout.name)
            if record is None or record.get("directory", directory) != directory:
                raise KeyError(
                    f"{layout.name} ({directory}) has no {name} record\n"
                    f"expected one per kept directory in --{name}\n"
                    f"re-run that stage over the same kept file"
                )
            stated[name] = record
        views = build_views(layout, columns_of[directory], metadata_of[directory], args,
                            calibration=calibration, episodes=kept_of[directory])
        action = build_action(layout, stated["gripper"])
        embodiment = embodiment_block(stated["embodiment"], layout.name)
        parts = {field: layout_table.arm_parts(layout, field) for field in PART_FIELDS}
        body = layout_table.body_block(layout)
        origin = layout_table.stored_origin(layout)
        columns = build_columns(layout)
        unknown = sorted(set(keys_of(embodiment)) - set(keys_table))
        if unknown:
            raise KeyError(
                f"{layout.name} holds keys {unknown}, which config/embodiments.yaml lacks\n"
                f"expected every key a directory holds in the key table\n"
                f"add the key and its declarations")
        check_keys(layout.name, keys_of(embodiment), keys_table, parts, body, origin)
        sidecar = None
        if origin == "ingest":
            ingest_stamp(stage.read_json(root / directory / "meta" / "info.json"), layout.name)
        elif origin == "sidecar":
            if args.sidecars is None:
                raise ValueError(
                    f"{layout.name} states its v7 columns are a sidecar and --sidecars is not "
                    f"given\nexpected the sidecar folder, metadata/<corpus>/sidecars under the "
                    f"root\npass --sidecars")
            read = [name for field, name in columns.items()
                    if field not in ("state", "action") and name is not None]
            sidecar = sidecar_block(
                root, args.sidecars, layout.name,
                [(episode, rows_of[(directory, episode)]) for episode in kept_of[directory]],
                read)
        stores[origin or "none"] = stores.get(origin or "none", 0) + 1
        if stated["embodiment"]["episodes"] != len(kept_of[directory]):
            raise ValueError(
                f"{refusal_prefix(16)} {layout.name}: build_embodiment recorded "
                f"{stated['embodiment']['episodes']} episodes and the kept file holds "
                f"{len(kept_of[directory])}\nExpected one key per kept episode.\n"
                f"Re-run build_embodiment over this kept file.")
        used_keys.update(keys_of(embodiment))
        frequency = stated["frequency"]
        entries.append({
            "name": layout.name,
            "source": layout.source,
            "source_id": source_id_of(layout),
            "path": directory,
            "episodes": entry_weight["episodes"],
            "rows": entry_weight["rows"],
            "episode_starts": f"{args.cache_dir}/episode_starts/{layout.name}.npy",
            "weight": entry_weight["weight"],
            "trajectory_weights": entry_weight["trajectory_weights"],
            "views": views,
            "arms": build_arms(layout, base_offset_of(layout.name, layout.source, offsets),
                               parts),
            "action": action,
            "columns": columns,
            "body": body,
            "sidecar": sidecar,
            "scale_cm_per_step": scale[layout.name]["scale_cm_per_step"],
            "frame_ref": frame_ref_of(layout, views, registry.ACTION_FRAME_ROLES),
            "common_base_frame": bool(layout.action.get("common_base_frame", False)),
            "embodiment": embodiment,
            "control_hz": frequency["control_hz"],
            "control_hz_method": frequency["control_hz_method"],
            "control_hz_evidence": frequency["control_hz_evidence"],
            "history": {"blocked": []},
            # The frame sentence stays the layout's; the embodiment clause is the
            # key's own (config/embodiments.yaml), one per key this entry holds.
            "prompt": {"frame_clause": dict(layout.prompt["frame_clause"]),
                       "embodiment_clauses": keys_of(embodiment)},
            "cache": {"arrow": f"{args.cache_dir}/arrow/{layout.name}.arrow",
                      "frames": f"{args.cache_dir}/frames/{layout.name}.json"},
        })

    # The key's clause replaces the source's, so every OXE and DROID directory
    # stops calling its robot a Franka Panda (V6 1.8); R9 over the keys used.
    clauses = check_clauses(keys_table, used_keys)
    for entry in entries:
        entry["prompt"]["embodiment_clauses"] = {
            key: clauses[key] for key in entry["prompt"]["embodiment_clauses"]}
    # Official builds take frame sentences from the same authority as the
    # embodiment sentence. Layout copies are assertions, never overrides.
    from neo_vla.data.prompt.embodiments import TABLE, canonical_prompt, validate_manifest_prompts

    if table_path.resolve() == TABLE.resolve():
        validate_manifest_prompts({"datasets": entries})
        for entry in entries:
            entry["prompt"] = canonical_prompt(entry)
    require_listing(listing, {entry["name"] for entry in entries})
    blocked = blocked_positions(listing, kept_of,
                                {entry["path"]: entry["name"] for entry in entries})
    for entry in entries:
        entry["history"]["blocked"] = blocked.get(entry["name"], [])
    if listing is not None:
        logger.info("history blocked: %d episodes in %d directories (list states %d)",
                    sum(len(v) for v in blocked.values()), len(blocked),
                    listing["header"]["count"])

    # After the entries, because it is the layout's own extrinsics path that
    # names them: the pose files the cache stage writes from are inputs of the
    # manifest too.
    for entry in entries:
        for view in entry["views"]:
            named = view["extrinsics"].get("path_source")
            if named:
                paths[named] = stage.under_root(root, named)

    # The layout is the artifact's own, read off the file that states it, never
    # a default: it stamps the cache the loader then refuses by name. The mode and
    # the artifact are held together here too, because they say the same thing
    # -- what stands between <cam_k> and that view's picture -- and only the
    # artifact's name and version go into the stamp.
    template = load_template(paths["prompt_template"])
    check_mode_agrees(template, args.intrinsics_mode)

    payload = {
        "schema_version": args.schema_version,
        "corpus": args.corpus,
        "canvas": {"short_side": args.canvas_short_side, "token": args.canvas_token,
                   "cell": list(args.canvas_cell)},
        "prompt": {
            "template": stage.relative(root, paths["prompt_template"]),
            "cache": args.prompt_cache,
            "stamp": asdict(stamp_of(
                template,
                tokenizer_id=args.tokenizer_id,
                max_text_tokens=args.max_text_tokens,
                prefix_layout=template.prefix_layout,
                n_views=args.n_views,
                intrinsics_mode=args.intrinsics_mode,
            )),
        },
        "totals": {
            "directories": len(entries),
            "episodes": sum(entry["episodes"] for entry in entries),
            "rows": sum(entry["rows"] for entry in entries),
        },
        "provenance": stage.header("build_manifest", root=root, inputs=paths, tree=REPO,
                                   extra={"corpus": args.corpus, "n_views": args.n_views,
                                          **configs}),
        "datasets": entries,
    }
    written = stage.write_json(out, payload)
    camera = sum(1 for entry in entries
                 if parse_frame_ref(entry["frame_ref"])[0] == "camera")
    kinds: Counter = Counter(view["intrinsics"]["kind"]
                             for entry in entries for view in entry["views"])
    logger.info("%d directories, %d episodes, %d rows", payload["totals"]["directories"],
                payload["totals"]["episodes"], payload["totals"]["rows"])
    logger.info("%d in camera frame, %d in robot frame", camera, len(entries) - camera)
    logger.info("v7 columns by writer (of %d directories): %s", len(entries),
                dict(sorted(stores.items())))
    drawn: Counter = Counter(
        len(reference_slots(entry["views"], registry.ACTION_FRAME_ROLES))
        for entry in entries if parse_frame_ref(entry["frame_ref"])[0] == "camera"
    )
    logger.info("reference cameras per camera-frame directory: %s",
                dict(sorted(drawn.items())))
    logger.info("views by calibration: %s", dict(sorted(kinds.items())))
    logger.info("%s %s", "wrote" if written else "unchanged", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
