"""Rule 2: find unreliable joint and hand channels; nothing is removed from the data.

Dead channel (per dataset directory): a joint or hand column whose finite
reading is constant over the whole episode in at least 90% of the directory's
episodes that read it at all.

Wrapping joint (per embodiment): a joint with any step |q[t+1] - q[t]| > pi
between consecutive finite readings of one episode. Steps across an episode
boundary are not counted. Counts are summed per (embodiment key, arm slot)
over the directories that hold that key; any such step masks the joint for the
key, and each directory masks the joints gated for any key it holds.

Readings are the stored joint and hand columns of each directory's Arrow cache
(all rows, times the manifest's unit factor), as the manifest names them.

Usage:
    python tools/filter_channels.py --root $NEO_VLA_DATA_ROOT \
        --manifest manifests/action_pretrain.json --out channels.json
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import filter_common as common
import numpy as np

from neo_vla.constants import (
    ACTION_HAND,
    ACTION_JOINTS,
    ARM_JOINTS,
    HAND_JOINTS,
    MANIFEST_SCHEMA_VERSION,
)
from neo_vla.data.embodiment import DIM_NAMES

DEAD_SHARE = 0.9
WRAP_STEP = math.pi


def episode_dead(block: np.ndarray, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per column of [rows, width]: episodes constant over their whole length, episodes read."""
    width = block.shape[1]
    if block.shape[0] == 0 or len(starts) < 2:
        return np.zeros(width, np.int64), np.zeros(width, np.int64)
    finite = np.isfinite(block)
    bounds = starts[:-1][np.diff(starts) > 0]
    low = np.minimum.reduceat(np.where(finite, block, np.inf), bounds, axis=0)
    high = np.maximum.reduceat(np.where(finite, block, -np.inf), bounds, axis=0)
    counted = np.add.reduceat(finite.astype(np.int64), bounds, axis=0)
    read = counted > 0
    return (read & (low == high)).sum(axis=0), read.sum(axis=0)


def is_dead(dead: int, read: int) -> bool:
    """The dead-channel decision for one column of one directory."""
    return bool(read) and dead >= DEAD_SHARE * read


def wrap_counts(block: np.ndarray, starts: np.ndarray) -> tuple[np.ndarray, int]:
    """Per column, consecutive in-episode steps with |q[r+1] - q[r]| > pi; and steps checked."""
    if block.shape[0] < 2:
        return np.zeros(block.shape[1], np.int64), 0
    step = np.abs(block[1:] - block[:-1])
    inside = np.ones(block.shape[0] - 1, dtype=bool)
    inner = starts[1:-1]
    inside[inner[(inner > 0) & (inner < block.shape[0])] - 1] = False
    both = np.isfinite(step) & inside[:, None]
    return (both & (step > WRAP_STEP)).sum(axis=0).astype(np.int64), int(both.any(axis=1).sum())


def episode_keys(root: Path, entry: dict[str, Any], episodes: int) -> list[str]:
    """The embodiment key of every episode of a manifest entry."""
    block = entry.get("embodiment")
    if not isinstance(block, dict):
        raise ValueError(f"{entry['name']} has no embodiment block")
    if block["kind"] == "constant":
        return [str(block["key"])] * episodes
    if block["kind"] == "episode":
        indices = np.load(common.under_root(root, block["path"]), allow_pickle=False)
        if indices.shape[0] != episodes:
            raise ValueError(f"{entry['name']}: {indices.shape[0]} episode keys "
                             f"for {episodes} episodes")
        return [str(block["keys"][int(index)]) for index in indices]
    raise ValueError(f"{entry['name']}: embodiment kind {block['kind']!r}; "
                     "expected constant or episode")


def scan_entry(root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    """One directory: dead counts per joint/hand column and wrap counts per joint, per slot."""
    starts = np.load(common.under_root(root, entry["episode_starts"]),
                     allow_pickle=False).astype(np.int64)
    stored = {part: entry["columns"].get(part) for part in ("joints", "hand")}
    names = sorted({column for column in stored.values() if column is not None})
    table = (common.read_table(common.under_root(root, entry["cache"]["arrow"]), names)
             if names else None)
    slots = [int(arm["slot"]) for arm in entry["arms"]]
    dead: dict[str, dict[str, list[int]]] = {}
    wraps: dict[int, np.ndarray] = {}
    checked = 0
    for index in range(len(entry["action"]["arms"])):
        arm = entry["arms"][index]
        slot = slots[index]
        for part, span, width in (("joints", ACTION_JOINTS, ARM_JOINTS),
                                  ("hand", ACTION_HAND, HAND_JOINTS)):
            if arm.get(part) is None:
                continue
            if stored[part] is None:
                raise ValueError(f"{entry['name']} arm {index} states {part} and the "
                                 f"manifest names no stored {part} column")
            count = int(arm[part]["count"])
            readings = common.matrix(table, stored[part])[:, index * width:index * width + count]
            readings = readings * float(arm[part]["unit_factor"])
            dead_episodes, read_episodes = episode_dead(readings, starts)
            for offset in range(count):
                dead.setdefault(str(slot), {})[DIM_NAMES[span.start + offset]] = [
                    int(dead_episodes[offset]), int(read_episodes[offset])]
            if part == "joints":
                counted, steps = wrap_counts(readings, starts)
                wraps[slot] = np.zeros(ARM_JOINTS, dtype=np.int64)
                wraps[slot][:count] = counted
                checked = max(checked, steps)
    return {"starts": starts, "slots": slots, "dead": dead, "wraps": wraps, "checked": checked}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, help="the data root the manifest paths hang off")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True, help="new JSON output")
    args = parser.parse_args(argv)

    root = common.resolve_root(args.root)
    manifest_path = common.cli_path(args.manifest)
    out = common.new_output(args.out, [manifest_path])
    manifest = common.read_json(manifest_path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"manifest schema_version {manifest.get('schema_version')!r}; "
                         f"expected {MANIFEST_SCHEMA_VERSION}")

    datasets: dict[str, Any] = {}
    members: dict[str, list[str]] = {}
    held_keys: dict[str, set[str]] = {}
    scans: dict[str, dict[str, Any]] = {}
    for entry in manifest["datasets"]:
        scan = scan_entry(root, entry)
        episodes = len(scan["starts"]) - 1
        keys = episode_keys(root, entry, episodes)
        weights = (np.load(common.under_root(root, entry["trajectory_weights"]),
                           allow_pickle=False).astype(np.float64)
                   if entry.get("trajectory_weights") else np.ones(episodes))
        array = np.asarray(keys)
        for key in sorted(set(keys)):
            if weights[np.nonzero(array == key)[0]].sum() > 0:
                members.setdefault(key, []).append(entry["name"])
        held_keys[entry["name"]] = set(keys)
        scans[entry["name"]] = scan

    gates: dict[str, dict[str, Any]] = {}
    for key, names in sorted(members.items()):
        slots = sorted({slot for name in names for slot in scans[name]["slots"]})
        for slot in slots:
            counts = np.zeros(ARM_JOINTS, dtype=np.int64)
            checked = 0
            for name in names:
                scan = scans[name]
                if slot not in scan["slots"]:
                    continue
                counts += scan["wraps"].get(slot, np.zeros(ARM_JOINTS, dtype=np.int64))
                checked += scan["checked"]
            joints = [int(joint) for joint in np.flatnonzero(counts)]
            if joints:
                gates.setdefault(key, {})[str(slot)] = {
                    "joints": [DIM_NAMES[joint] for joint in joints],
                    "steps": [int(counts[joint]) for joint in joints],
                    "checked_steps": checked}

    for name, scan in scans.items():
        masked: dict[str, list[str]] = {}
        for slot in scan["slots"]:
            joints = sorted({joint for key in held_keys[name]
                             for joint in gates.get(key, {}).get(str(slot), {}).get("joints", [])},
                            key=DIM_NAMES.index)
            if joints:
                masked[str(slot)] = joints
        datasets[name] = {
            "dead_counts": scan["dead"],
            "dead_dims": {slot: [column for column, (dead, read) in columns.items()
                                 if is_dead(dead, read)]
                          for slot, columns in scan["dead"].items()
                          if any(is_dead(dead, read) for dead, read in columns.values())},
            "wrap_steps": {str(slot): {DIM_NAMES[joint]: int(count)
                                       for joint, count in enumerate(counts) if count}
                           for slot, counts in scan["wraps"].items() if counts.any()},
            "wrap_checked_steps": scan["checked"],
            "wrap_masked": masked,
        }

    common.write_json(out, {
        "provenance": common.header(
            "filter_channels", inputs=[common.display(root, manifest_path)],
            settings={"dead_share": DEAD_SHARE, "wrap_step_rad": WRAP_STEP}),
        "datasets": datasets,
        "keys": gates,
    })
    dead = sum(len(names) for record in datasets.values()
               for names in record["dead_dims"].values())
    print(f"{len(datasets)} directories, {dead} dead channels, "
          f"{len(gates)} keys with wrapping joints; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
