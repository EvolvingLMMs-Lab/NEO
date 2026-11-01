"""Stage 2: what is wrong with each episode. It drops nothing.

One record per inventory record, in the same order, holding a named defect list
and a pass flag. Reporting is separate from filtering so that a defect which
turns out to be common is a decision about the corpus rather than a silent loss
of part of it.

The defect names cover the classes that have actually cost this corpus a run:
unreadable and truncated episodes, an action column that is a copy of the
state, zero rows presented as valid, missing instruction text, a quaternion
that is not a rotation, and an extrinsic that is not one either.

Usage:
    python tools/validate_episodes.py --root /corpus --inventory metadata/C/inventory.jsonl \
        --layouts config/layouts/action_pretrain.json --out metadata/C/validation.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import layouts as layout_table
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import stage

# The bound is imported rather than restated: layouts already pulls neo_vla in,
# so the module is loaded by the time this line runs and it costs nothing. One
# number, in one place, is what makes it auditable.
from neo_vla.data.state import ROTATION_WIDTHS
from neo_vla.model.tokenizers import RECORD_LIMIT

logger = logging.getLogger("validate_episodes")

REPO = Path(__file__).resolve().parent.parent


def out_of_range(values: np.ndarray) -> np.ndarray:
    """Elementwise: is this a number NO measurement could be?

    Not merely non-finite. RH20T ships garbled telemetry as a finite number of
    impossible size -- state = -9.648338e37 with the frames either side
    continuous to a tenth of a centimetre -- and an isfinite count called the
    file that carried it clean. It went on to make the loss nan at step 9,920,
    through the backward pass only, with a healthy forward loss.
    """
    return ~np.isfinite(values) | (np.abs(values) > RECORD_LIMIT)


# The blocks a v7 layout states and a v6 one does not (layouts.require_v7).
V7_STATED = ("joints", "hand", "body")


def read_columns_of(layout: layout_table.Layout, field: str, width: int) -> np.ndarray:
    """Which columns of the state or the action a later stage reads, as a [width] mask.

    In a v7 output NaN means not recorded, never 0. A v7
    layout names what is read: each arm's position, rotation and gripper columns
    of the state, and each arm's dpos, drot and gripper columns of the action; a
    gripper whose method is masked is never read. A joints-only key (BimanualYAM)
    states its pose null and writes NaN there, and a hand key (robocoin_hands)
    masks its gripper and writes NaN there. A v6 layout states nothing about
    unrecorded columns, and an action with a base block is not one this reads,
    so every column of those counts as read.
    """
    read = np.ones(width, dtype=bool)
    if not all(block in layout.v7 for block in V7_STATED):
        return read
    masked = (layout.gripper or {}).get("method") == "masked"
    named: set[int] = set()
    if field == "action":
        if layout.action.get("base") is not None:
            return read
        parts = ("dpos", "drot") if masked else ("dpos", "drot", "gripper")
        for arm in layout.action["arms"]:
            for part in parts:
                named.update(int(column) for column in arm.get(part) or ())
    else:
        for arm in (layout.state or {}).get("arms", ()):
            if arm.get("position_at") is not None:
                start = int(arm["position_at"])
                named.update(range(start, start + 3))
            if arm.get("rotation_kind") is not None:
                start = int(arm["rotation_at"])
                named.update(range(start, start + ROTATION_WIDTHS[arm["rotation_kind"]]))
            if arm.get("gripper") is not None and not masked:
                named.add(int(arm["gripper"]["column"]))
    read[:] = False
    read[[column for column in named if column < width]] = True
    return read


def declared_unrecorded(layout: layout_table.Layout, field: str, width: int,
                        metadata: dict[str, Any]) -> np.ndarray:
    """The [width] columns this episode itself states it does not record, as a mask.

    The ingest writer names, per field, the arm slots an episode does not record
    (episodes.jsonl ingest_unrecorded; the v7 ingest writer holds those columns to
    NaN): the RoboMIND Tien Kung gello's dead follower closure. Of these fields
    only the gripper sits in the state and action columns read here. A v6 layout
    states nothing, and an action with a base block is read whole, as in
    read_columns_of.
    """
    declared = np.zeros(width, dtype=bool)
    slots = {int(slot) for slot in
             ((metadata.get("ingest_unrecorded") or {}).get("gripper") or {})}
    if not slots or not all(block in layout.v7 for block in V7_STATED):
        return declared
    if field == "action" and layout.action.get("base") is not None:
        return declared
    arms = (layout.state or {}).get("arms", ())
    names = {arm["name"] for index, arm in enumerate(arms)
             if int(arm.get("slot", index)) in slots}
    if field == "action":
        columns = [int(column) for arm in layout.action["arms"] if arm["name"] in names
                   for column in arm.get("gripper") or ()]
    else:
        columns = [int(arm["gripper"]["column"]) for arm in arms
                   if arm["name"] in names and arm.get("gripper") is not None]
    declared[[column for column in columns if column < width]] = True
    return declared


def not_recorded(values: np.ndarray, read: np.ndarray) -> np.ndarray:
    """The [width] columns that hold the not-recorded statement: unread and NaN on every row.

    A column NaN on some rows only is a broken record, not an unrecorded one,
    and an infinity is never a statement, so both stay defects.
    """
    return ~read & np.isnan(values).all(axis=0)


DEFECTS = (
    "data_file_missing",
    "episode_unreadable",
    "rows_mismatch",
    "frames_missing",
    "episode_too_short",
    "instruction_missing",
    "state_layout_unknown",
    "state_width_mismatch",
    "quaternion_not_unit",
    "gripper_polarity_unknown",
    "gripper_constant",
    "action_width_mismatch",
    "action_equals_state",
    "fabricated_zero_rows",
    "action_column_constant",
    "not_finite",
    "record_out_of_range",
    "extrinsics_missing",
    "extrinsics_not_rotation",
    "calibration_unusable",
    "rotation_convention_unresolved",
)

# A rotation delta whose convention was never established has no camera-frame
# form; the sample can still train in robot frame, so this is a defect to
# report and not automatically a reason to drop.
UNRESOLVED_FRAMES = ("euler_diff", "unresolved")


def _defect(name: str, **detail: Any) -> dict[str, Any]:
    if name not in DEFECTS:
        raise KeyError(
            f"defect {name!r} is not named in DEFECTS\n"
            f"expected one of the names this stage records\n"
            f"add the name before reporting it, so filter can key on it"
        )
    return {"name": name, **detail}


def read_columns(path: Path, wanted: list[str], max_rows: int) -> dict[str, np.ndarray]:
    """The numeric columns of one episode, as float arrays of shape [rows, width]."""
    table = pq.read_table(path, columns=[c for c in wanted if c])
    columns = {}
    for name in table.schema.names:
        values = table.column(name).to_pylist()
        if max_rows:
            values = values[:max_rows]
        array = np.asarray(values, dtype=np.float64)
        columns[name] = array if array.ndim == 2 else array.reshape(len(values), -1)
    return columns


def rotation_of(flat: list[float]) -> np.ndarray | None:
    """A stored 4x4 or 3x3 pose -> its rotation block, or None when it is neither."""
    values = np.asarray(flat, dtype=np.float64)
    if values.size == 16:
        return values.reshape(4, 4)[:3, :3]
    if values.size == 9:
        return values.reshape(3, 3)
    return None


def rotation_error(matrix: np.ndarray) -> float:
    """How far a matrix is from being a rotation, in the same units as its entries."""
    orthonormal = float(np.abs(matrix @ matrix.T - np.eye(3)).max())
    handedness = float(abs(np.linalg.det(matrix) - 1.0))
    return max(orthonormal, handedness)


def load_sidecar(path: Path, index_key: str) -> dict[int, dict[str, Any]]:
    """A per-episode geometry sidecar, keyed by the field it says it is keyed by."""
    records: dict[int, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records[int(record[index_key])] = record
    return records


def check_metadata_gates(metadata: dict[str, Any], gates: tuple) -> list[dict[str, Any]]:
    found = []
    for gate in gates:
        key = gate["key"]
        if key not in metadata:
            found.append(_defect(gate["defect"], gate=key, reason="absent"))
        elif metadata[key] != gate["equals"]:
            found.append(_defect(gate["defect"], gate=key, value=metadata[key]))
    return found


def check_state(state: np.ndarray, layout: layout_table.Layout, tol: float) -> list[dict]:
    """State width, quaternion norm and gripper motion, against the layout."""
    found: list[dict[str, Any]] = []
    if layout.state is None:
        return [_defect("state_layout_unknown", source=layout.source)]
    width = int(layout.state["width"])
    if state.shape[1] != width:
        return [_defect("state_width_mismatch", got=state.shape[1], expected=width)]
    for arm in layout.state["arms"]:
        # A joints-only arm states no rotation (rotation_kind null): no quaternion.
        kind = arm["rotation_kind"]
        if kind is not None and kind.startswith("quat"):
            start = int(arm["rotation_at"])
            norms = np.linalg.norm(state[:, start:start + 4], axis=1)
            worst = float(np.abs(norms - 1.0).max())
            if worst > tol:
                found.append(_defect("quaternion_not_unit", arm=arm["name"], error=worst))
        gripper = arm.get("gripper")
        if gripper is None:
            found.append(_defect("gripper_polarity_unknown", arm=arm["name"]))
            continue
        column = state[:, int(gripper["column"])]
        if float(column.max() - column.min()) == 0.0:
            found.append(_defect("gripper_constant", arm=arm["name"], value=float(column[0])))
    return found


def check_action(
    action: np.ndarray,
    state: np.ndarray | None,
    valid: np.ndarray | None,
    layout: layout_table.Layout,
    zero_fraction: float,
) -> list[dict[str, Any]]:
    """Width, the state copy, the fabricated zeros and the dead columns."""
    found: list[dict[str, Any]] = []
    expected = int(layout.action["flat_dim"])
    if action.shape[1] != expected:
        return [_defect("action_width_mismatch", got=action.shape[1], expected=expected)]
    if state is not None and state.shape == action.shape:
        same = float((action == state).all(axis=1).mean())
        if same > 0.0:
            found.append(_defect("action_equals_state", fraction=same))
    zero_rows = (action == 0.0).all(axis=1)
    if valid is not None:
        zero_rows = zero_rows & (valid.reshape(-1) != 0)
    fraction = float(zero_rows.mean())
    if fraction > zero_fraction:
        found.append(_defect("fabricated_zero_rows", fraction=fraction))
    constant = [
        index for index in range(action.shape[1])
        if float(action[:, index].max() - action[:, index].min()) == 0.0
    ]
    if constant:
        found.append(_defect("action_column_constant", columns=constant))
    if layout.action.get("rotation_delta_frame") in UNRESOLVED_FRAMES:
        found.append(
            _defect("rotation_convention_unresolved",
                    frame=layout.action.get("rotation_delta_frame"))
        )
    return found


def check_extrinsics(
    metadata: dict[str, Any],
    sidecar: dict[int, dict[str, Any]] | None,
    episode: int,
    spec: dict[str, Any],
    tol: float,
    view: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One camera's recorded pose for this episode, and whether it is a rotation.

    A source whose poses come from build_extrinsics is not read here. That
    stage runs after this one and reports its own defect per record, so
    repeating the audit would mean reading a file this run may not have yet
    and counting the same refusal twice.
    """
    if spec.get("kind") != "episode":
        return [], {"present": False, "kind": spec.get("kind", "none")}
    if spec["source"] == "stage":
        return [], {"kind": "episode", "source": "stage"}
    refuse = spec.get("refuse_when")
    # The episode's own metadata can say the stored pose does not describe it,
    # as RH20T's does for an in-hand recording. build_cache reads that flag and
    # hands the loader an unknown pose, so counting the episode as covered here
    # would report a pose the corpus never uses.
    if refuse is not None and bool(metadata.get(refuse["key"])) == bool(refuse["equals"]):
        return [], {"present": False, "refused": refuse["key"]}
    if spec["source"] == "metadata":
        raw = metadata.get(spec["key"])
    else:
        record = (sidecar or {}).get(episode)
        raw = record.get(spec["key"]) if record else None
    # A source that publishes every camera's pose under one field names the
    # camera inside it, so one field serves several views. DROID does this.
    if "member" in spec:
        raw = raw.get(spec["member"]) if isinstance(raw, dict) else None
    if raw is None:
        return (
            [_defect("extrinsics_missing", view=view, key=spec["key"])],
            {"present": False},
        )
    matrix = rotation_of(raw)
    if matrix is None:
        return (
            [_defect("extrinsics_not_rotation", view=view, reason="shape",
                     size=len(raw))],
            {"present": True, "rotation_ok": False},
        )
    error = rotation_error(matrix)
    if error > tol:
        return (
            [_defect("extrinsics_not_rotation", view=view, error=error)],
            {"present": True, "rotation_ok": False},
        )
    return [], {"present": True, "rotation_ok": True}


def sidecar_path(layout: layout_table.Layout, spec: dict[str, Any]) -> str | None:
    """The geometry sidecar one view's spec reads, or None when it reads none."""
    if spec.get("kind") != "episode" or spec.get("source") != "sidecar":
        return None
    return str(spec["path"]).replace("{name}", layout.name)


def check_views(
    metadata: dict[str, Any],
    episode: int,
    layout: layout_table.Layout,
    sidecars: dict[str, dict[int, dict[str, Any]]],
    tol: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Every camera of one directory, each against the pose spec it was given.

    The layout keys its poses by view, because a source may publish a pose for
    one camera and not the next. A view whose spec is kind none is a camera
    that publishes no pose, which is most of OXE and DROID's wrist: no defect,
    and coverage that says so per camera rather than for the directory.
    """
    found: list[dict[str, Any]] = []
    coverage: dict[str, dict[str, Any]] = {}
    for view in layout.views:
        key = str(view["key"])
        spec = layout_table.view_extrinsics(layout, key)
        named = sidecar_path(layout, spec)
        defects, seen = check_extrinsics(
            metadata, sidecars.get(named) if named else None, episode, spec, tol, key,
        )
        found.extend(defects)
        coverage[key] = seen
    return found, coverage


def validate_record(
    record: dict[str, Any],
    layout: layout_table.Layout,
    sidecars: dict[str, dict[int, dict[str, Any]]],
    root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """One inventory record -> its defects, its coverage per camera and pass."""
    found: list[dict[str, Any]] = []
    rows_on_disk = None

    if not any(task.strip() for task in record["tasks"]):
        found.append(_defect("instruction_missing"))
    if record["rows"] < args.min_rows:
        found.append(_defect("episode_too_short", rows=record["rows"]))
    for key, video in sorted(record["videos"].items()):
        # Measured here rather than taken from the inventory: a stream that was
        # truncated after the scan is exactly what this defect name is for.
        size = (root / video["path"]).stat().st_size if (
            video["path"] and (root / video["path"]).is_file()) else 0
        if size == 0:
            found.append(_defect("frames_missing", view=key, bytes=size))

    found.extend(check_metadata_gates(record["metadata"], layout.metadata_gates))
    extrinsic_defects, coverage = check_views(
        record["metadata"], record["episode"], layout, sidecars,
        args.rotation_tolerance,
    )
    found.extend(extrinsic_defects)

    if record["data_file"] is None:
        found.append(_defect("data_file_missing"))
    else:
        path = root / record["data_file"]
        wanted = [layout.columns["state"], layout.columns["action"],
                  layout.action.get("valid_column")]
        try:
            columns = read_columns(path, wanted, args.max_rows)
        except (OSError, pa.ArrowInvalid) as error:
            found.append(_defect("episode_unreadable", error=str(error)[:200]))
            columns = None
        if columns is not None:
            action = columns.get(layout.columns["action"])
            state = columns.get(layout.columns["state"])
            valid = columns.get(layout.action.get("valid_column") or "")
            rows_on_disk = int(action.shape[0]) if action is not None else 0
            if not args.max_rows and rows_on_disk != record["rows"]:
                found.append(
                    _defect("rows_mismatch", declared=record["rows"], on_disk=rows_on_disk)
                )
            # A v7 column the layout does not read, or one this episode declares
            # it does not record (declared_unrecorded), that is NaN on every row
            # is the not-recorded statement, not a defect.
            absent = {
                name: not_recorded(column, read_columns_of(layout, name, column.shape[1])
                                   & ~declared_unrecorded(layout, name, column.shape[1],
                                                          record["metadata"]))
                for name, column in (("action", action), ("state", state))
                if column is not None
            }
            non_finite = sum(
                int((~np.isfinite(column[:, ~absent[name]])).sum())
                for name, column in (("action", action), ("state", state))
                if column is not None
            )
            if non_finite:
                found.append(_defect("not_finite", count=non_finite))
            # Reported per column and per axis with the worst value, so the
            # defect names what to take back to the source rather than only
            # saying that something was wrong somewhere in the episode.
            for name, column in (("action", action), ("state", state)):
                if column is None:
                    continue
                outside = out_of_range(column)
                outside[:, absent[name]] = False
                if not outside.any():
                    continue
                axes = sorted({int(axis) for axis in np.where(outside)[1]})
                worst = column[outside]
                peak = worst[np.argmax(np.abs(np.nan_to_num(worst)))]
                found.append(_defect(
                    "record_out_of_range", column=name, axes=axes,
                    count=int(outside.sum()), worst=float(peak), limit=RECORD_LIMIT,
                ))
            if state is not None:
                found.extend(check_state(state, layout, args.quaternion_tolerance))
            if action is not None:
                found.extend(
                    check_action(action, state, valid, layout, args.zero_row_fraction)
                )

    return {
        "dataset": record["dataset"],
        "episode": record["episode"],
        "name": layout.name,
        "source": layout.source,
        "rows": record["rows"],
        "rows_on_disk": rows_on_disk,
        "extrinsics": coverage,
        "defects": found,
        "pass": not found,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--layouts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-rows", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=0,
                        help="read at most this many rows per episode; 0 reads all")
    parser.add_argument("--quaternion-tolerance", type=float, default=1e-3)
    parser.add_argument("--rotation-tolerance", type=float, default=1e-4)
    parser.add_argument("--zero-row-fraction", type=float, default=0.0,
                        help="report zero action rows above this share of the episode")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    root = stage.resolve_root(args.root)
    inventory_path = stage.under_root(root, args.inventory)
    layouts_path = stage.under_root(root, args.layouts)
    out = stage.under_root(root, args.out)

    _, records = stage.read_jsonl(inventory_path)
    document = layout_table.load(layouts_path)
    directories = sorted({record["dataset"] for record in records})
    stage.refuse_write_into_input(out, [root / name for name in directories])

    resolved = {name: layout_table.resolve(document, name) for name in directories}
    # Keyed by the file rather than by the directory: two cameras of one
    # directory are often calibrated into one sidecar, and two directories can
    # resolve to one layout name, so the path is what is read at most once.
    sidecars: dict[str, dict[int, dict[str, Any]]] = {}
    for layout in resolved.values():
        for view in layout.views:
            spec = layout_table.view_extrinsics(layout, view["key"])
            named = sidecar_path(layout, spec)
            if named is None or named in sidecars:
                continue
            sidecars[named] = load_sidecar(
                stage.under_root(root, named),
                spec.get("index_key", "episode_index"),
            )

    out_records = [
        validate_record(record, resolved[record["dataset"]], sidecars, root, args)
        for record in records
    ]

    counts: Counter = Counter()
    for record in out_records:
        for defect in record["defects"]:
            counts[defect["name"]] += 1
    head = stage.header(
        "validate_episodes",
        root=root,
        inputs={"inventory": inventory_path, "layouts": layouts_path},
        tree=REPO,
        extra={
            "min_rows": args.min_rows, "max_rows": args.max_rows,
            "quaternion_tolerance": args.quaternion_tolerance,
            "rotation_tolerance": args.rotation_tolerance,
            "zero_row_fraction": args.zero_row_fraction,
            "defect_names": list(DEFECTS),
        },
    )
    written = stage.write_jsonl(out, head, out_records)
    passed = sum(1 for record in out_records if record["pass"])
    logger.info("%d episodes, %d clean, %d with defects",
                len(out_records), passed, len(out_records) - passed)
    for name, count in sorted(counts.items()):
        logger.info("defect %s: %d", name, count)
    logger.info("%s %s", "wrote" if written else "unchanged", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
