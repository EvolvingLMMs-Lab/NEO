"""The units the stats and scale stages both measure in: centimetres, radians.

A source's raw action columns become one block per arm, seven wide, holding a
per-step translation in centimetres, a per-step rotation in radians and the
gripper as stored. The factors come from the layout file, never from the data:
a dataset whose translation unit was never established has no unified form and
this module says so rather than assuming the file is already in centimetres.

The scale and gripper stages measure in those units. The normaliser is
measured on the loader's own targets (build_stats, V6 1.1), so nothing here
maps a block into a model domain: the translation curve is gone (V6 1.7).

Both measuring stages sample episodes rather than rows, because rows inside an
episode are strongly correlated and a row-level draw reports an error bar
orders of magnitude too small.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow.parquet as pq

# Rotation deltas this module can put in radians by scaling. Anything else
# needs a log map, which belongs with the loader's action packer.
ROTATION_KINDS = ("axis_angle", "none")

# Per-arm block width: translation, rotation, gripper.
ARM_WIDTH = 7

# Axes a translation factor states, one measurement each: the dpos columns
# 0:3 of an arm block (neo_vla.data.actions.DPOS, stated here so this module
# imports neither torch nor the loader).
AXES = 3


def translation_factor(action: dict[str, Any]) -> np.ndarray | None:
    """The per-axis centimetres one stored translation unit is worth, or None.

    None says the layout states no factor, which is what keeps a unit-free
    dataset out of the stats, the scale and the manifest. A factor of one is a
    measurement like any other and is written out as three numbers, so a
    dataset already in centimetres and one measured to be is read the same way.
    """
    declared = action.get("translation_cm_per_unit")
    if declared is None:
        return None
    values = np.asarray(declared, dtype=np.float64)
    if values.shape != (AXES,):
        raise ValueError(
            f"translation_cm_per_unit is {declared!r}\n"
            f"expected {AXES} numbers, one per translation axis\n"
            f"measure the factor for each axis of this dataset"
        )
    return values


def factors_stated(action: dict[str, Any]) -> bool:
    """Whether both measured unit factors are there, so the actions unify."""
    return translation_factor(action) is not None and "rotation_rad_per_unit" in action


def unified_arms(raw: np.ndarray, action: dict[str, Any]) -> list[np.ndarray]:
    """A raw action window -> one [rows, 7] block per arm, in cm and radians."""
    kind = action.get("rotation_delta_kind")
    if kind not in ROTATION_KINDS:
        raise ValueError(
            f"rotation_delta_kind is {kind!r}\n"
            f"expected one of {ROTATION_KINDS}\n"
            f"establish the source's rotation encoding in the layout file"
        )
    factor = translation_factor(action)
    if factor is None:
        raise ValueError(
            "translation_cm_per_unit is not stated\n"
            "expected the centimetres one stored unit is worth\n"
            "measure it for this dataset or leave the dataset out"
        )
    if "rotation_rad_per_unit" not in action:
        raise ValueError(
            "rotation_rad_per_unit is not stated\n"
            "expected the radians one stored rotation unit is worth\n"
            "measure it for this dataset or leave the dataset out"
        )
    radians = float(action["rotation_rad_per_unit"])
    blocks = []
    for arm in action["arms"]:
        block = np.zeros((raw.shape[0], ARM_WIDTH), dtype=np.float64)
        block[:, 0:3] = raw[:, list(arm["dpos"])] * factor
        block[:, 3:6] = raw[:, list(arm["drot"])] * radians
        block[:, 6] = raw[:, arm["gripper"][0]]
        blocks.append(block)
    return blocks


def read_action(path: Path, column: str, max_rows: int) -> np.ndarray:
    """One episode's action column as [rows, width]."""
    table = pq.read_table(path, columns=[column])
    values = table.column(column).to_pylist()
    if max_rows:
        values = values[:max_rows]
    array = np.asarray(values, dtype=np.float64)
    return array if array.ndim == 2 else array.reshape(len(values), -1)


def sample(episodes: list[dict[str, Any]], limit: int, seed: str) -> list[dict[str, Any]]:
    """A seeded draw of at most `limit` episodes, in a stable order."""
    if not limit or len(episodes) <= limit:
        return episodes
    drawn = random.Random(seed).sample(range(len(episodes)), limit)
    return [episodes[index] for index in sorted(drawn)]


def stable_seed(text: str) -> int:
    """A seed that is the same in every process, which hash() is not."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def kept_by_dataset(
    kept: list[dict[str, Any]], inventory: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """The kept episodes of each directory, carrying their data file paths."""
    files = {(item["dataset"], item["episode"]): item["data_file"] for item in inventory}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in kept:
        if record["kind"] != "kept":
            continue
        path = files.get((record["dataset"], record["episode"]))
        if path is None:
            continue
        grouped.setdefault(record["dataset"], []).append({**record, "data_file": path})
    return grouped


def iter_blocks(
    root: Path,
    episodes: list[dict[str, Any]],
    layout: Any,
    max_rows: int,
) -> Iterator[tuple[dict[str, Any], list[np.ndarray]]]:
    """Each drawn episode with its per-arm unified blocks."""
    for episode in episodes:
        raw = read_action(
            root / episode["data_file"], layout.columns["action"], max_rows
        )
        yield episode, unified_arms(raw, layout.action)
