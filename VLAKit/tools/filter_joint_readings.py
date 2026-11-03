"""Rule 1: exclude implausible joint readings; keep every episode.

Within each episode, a joint reading is excluded (written as NaN) when it lies
beyond +-2 pi, or more than pi from the median of the finite readings in the
5-row window centred on it. The window never crosses an episode boundary.
Non-finite readings are NaN on output; rows and episodes are never removed.

Usage:
    python tools/filter_joint_readings.py --values joints.npy \
        --episode-starts episode_starts.npy --out joints_filtered.npz
    python tools/filter_joint_readings.py --values episode_000000.parquet \
        --column observation.joints --episode-column episode_index --out filtered.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import filter_common as common
import numpy as np


def filter_joints(values: np.ndarray, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[rows, joints] radians and episode starts -> (filtered float32, excluded bool)."""
    joints = common.finite(values)
    excluded = np.zeros(joints.shape, dtype=bool)
    for start, stop in zip(starts[:-1], starts[1:]):
        if stop > start:
            excluded[start:stop] = common.implausible_joints(joints[start:stop])
    joints[excluded] = np.nan
    return joints, excluded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--values", required=True,
                        help=".npy [rows, joints], or an Arrow/Feather/Parquet file (radians)")
    parser.add_argument("--column", help="the joint column of a table file")
    episodes = parser.add_mutually_exclusive_group(required=True)
    episodes.add_argument("--episode-starts",
                          help=".npy of episode start rows, [episodes + 1], 0 .. rows")
    episodes.add_argument("--episode-column",
                          help="a per-row episode column of the same table file")
    parser.add_argument("--out", required=True, help="new .npz output")
    args = parser.parse_args(argv)

    source = Path(args.values)
    out = common.new_output(args.out, [source])
    values = common.read_values(source, args.column)
    if args.episode_starts:
        starts = np.load(args.episode_starts, allow_pickle=False)
    else:
        if source.suffix == ".npy":
            raise ValueError("--episode-column needs a table file; pass --episode-starts with .npy")
        table = common.read_table(source, [args.episode_column])
        starts = common.starts_from_column(common.matrix(table, args.episode_column)[:, 0])
    starts = common.check_starts(starts, values.shape[0])

    joints, excluded = filter_joints(values, starts)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as stream:
        np.savez(stream, values=joints, excluded=excluded, episode_starts=starts)
    print(json.dumps({
        "rows": int(values.shape[0]), "episodes": int(len(starts) - 1),
        "excluded_per_column": excluded.sum(axis=0).astype(int).tolist(),
        "ceiling_rad": common.JOINT_CEILING_RAD, "spike_rad": common.SPIKE_RAD,
        "window": common.JOINT_WINDOW, "out": str(out),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
