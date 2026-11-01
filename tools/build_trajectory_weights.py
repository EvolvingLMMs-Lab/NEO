"""Stage 6a: one weight per kept episode, so an episode can leave the mixture
without renumbering the kept set.

Writes an array per directory, one everywhere and zero where validation found
a defect that leaves the episode with no bytes to read. Dropping such an
episode in the filter stage is the cleaner answer, but every kept position
after it renumbers, and the caches, the episode starts and the row offsets are
all keyed on that numbering. A zero weight moves no position.

The index hashes every array it wrote, so an edited one is not trusted.

Usage:
    python tools/build_trajectory_weights.py --root /corpus \
        --validation prep/validation.jsonl --kept prep/kept.jsonl \
        --out-dir prep/trajectory --out prep/trajectory.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import stage
from filter_episodes import UNREADABLE, unreadable_defects

logger = logging.getLogger("build_trajectory_weights")

REPO = Path(__file__).resolve().parent.parent


def stream_records(path: Path) -> Iterator[dict[str, Any]]:
    """The records of one jsonl stage output, a record at a time.

    Both inputs run to hundreds of megabytes and nothing here needs two records
    at once, so they are read as a stream rather than held whole.
    """
    with open(path, encoding="utf-8") as handle:
        if not handle.readline():
            raise ValueError(
                f"{path} is empty\n"
                f"expected a provenance header on line 1\n"
                f"re-run the stage that writes it"
            )
        for line in handle:
            if line.strip():
                yield json.loads(line)


def unreadable_episodes(validation: Path) -> dict[tuple[str, int], list[str]]:
    """Every validated episode with no bytes to read, and the defects saying so."""
    found: dict[tuple[str, int], list[str]] = {}
    for record in stream_records(validation):
        names = unreadable_defects(record)
        if names:
            found[(record["dataset"], record["episode"])] = names
    return found


class Tally:
    """One directory's kept positions: how many, and which ones to zero.

    last and total are kept beside the count so the positions can be checked
    for gaps and repeats without holding one entry per episode: a run of
    0..count-1 taken once has last equal to count - 1 and total equal to their
    sum, and no other set of that size has both.
    """

    def __init__(self, directory: str) -> None:
        self.directory = directory
        self.count = 0
        self.last = -1
        self.total = 0
        self.zeroed: dict[int, list[str]] = {}

    def add(self, position: int) -> None:
        self.count += 1
        self.last = max(self.last, position)
        self.total += position

    def check(self, name: str) -> None:
        expected = self.count * (self.count - 1) // 2
        if self.last != self.count - 1 or self.total != expected:
            raise ValueError(
                f"directory {name!r} keeps {self.count} episodes and its kept "
                f"positions end at {self.last} and sum to {self.total}\n"
                f"expected the positions 0 to {self.count - 1}, each once\n"
                f"re-run filter_episodes; it numbers the kept records"
            )

    def weights(self, name: str) -> np.ndarray:
        """This directory's array: one per kept position, zero where unreadable."""
        self.check(name)
        array = np.ones(self.count, dtype=np.float64)
        for position in self.zeroed:
            array[position] = 0.0
        if float(array.sum()) <= 0.0:
            raise ValueError(
                f"every kept episode of directory {name!r} has no bytes to read\n"
                f"expected at least one readable episode\n"
                f"leave the directory out of the scan rather than weighting it to zero"
            )
        return array


def tally_kept(kept: Path, unreadable: dict[tuple[str, int], list[str]]) -> dict[str, Tally]:
    """Per dataset name, its kept positions and the ones with nothing to read."""
    tallies: dict[str, Tally] = {}
    for record in stream_records(kept):
        if record["kind"] != "kept":
            continue
        tally = tallies.setdefault(record["name"], Tally(record["dataset"]))
        tally.add(record["kept_position"])
        names = unreadable.get((record["dataset"], record["episode"]))
        if names:
            tally.zeroed[record["kept_position"]] = names
    return tallies


def write_array(path: Path, array: np.ndarray) -> None:
    """One directory's weights, replaced whole so no reader sees half a file."""
    temporary = path.with_name(path.name + ".partial")
    with open(temporary, "wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--kept", required=True)
    parser.add_argument("--out-dir", required=True,
                        help="where the per-directory arrays go")
    parser.add_argument("--out", required=True, help="the index naming them")
    parser.add_argument("--suffix", default=".npy",
                        help="must match build_weights --trajectory-weights-suffix")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    root = stage.resolve_root(args.root)
    validation_path = stage.under_root(root, args.validation)
    kept_path = stage.under_root(root, args.kept)
    out_dir = stage.under_root(root, args.out_dir)
    out = stage.under_root(root, args.out)

    unreadable = unreadable_episodes(validation_path)
    tallies = tally_kept(kept_path, unreadable)
    if not tallies:
        raise ValueError(
            f"{kept_path} holds no kept records\n"
            f"expected at least one kept episode to weight\n"
            f"re-run filter_episodes over a validation file with survivors"
        )
    stage.refuse_write_into_input(out, [root / tally.directory for tally in tallies.values()])
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, str] = {}
    directories: dict[str, Any] = {}
    zeroed_total = 0
    for name in sorted(tallies):
        tally = tallies[name]
        path = out_dir / f"{name}{args.suffix}"
        write_array(path, tally.weights(name))
        key = stage.relative(root, path)
        artifacts[key] = stage.sha256_of(path)
        zeroed_total += len(tally.zeroed)
        directories[name] = {
            "directory": tally.directory,
            "episodes": tally.count,
            "path": key,
            "zeroed": {str(position): defects
                       for position, defects in sorted(tally.zeroed.items())},
        }

    payload = {
        "provenance": stage.header(
            "build_trajectory_weights", root=root,
            inputs={"validation": validation_path, "kept": kept_path}, tree=REPO,
            extra={"unreadable_defects": list(UNREADABLE), "suffix": args.suffix},
        ),
        "artifacts": artifacts,
        "directories": directories,
        "totals": {
            "directories": len(directories),
            "episodes": sum(tally.count for tally in tallies.values()),
            "zeroed": zeroed_total,
        },
    }
    written = stage.write_json(out, payload)
    logger.info("%d directories weighted, %d episodes given weight zero",
                len(directories), zeroed_total)
    for name in sorted(directories):
        for position, defects in sorted(directories[name]["zeroed"].items()):
            logger.warning("%s kept position %s weighted zero: %s",
                           name, position, ", ".join(defects))
    logger.info("%s %s", "wrote" if written else "unchanged", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
