"""Stage 6: the mixture. One weight per directory and the share it expects.

Sampling mass is weight times rows, so a weight is only meaningful beside a row
count. This stage takes the share each source should take of the mixture and
solves for the per-directory weights that deliver it over the kept rows, then
writes the expected share back out so the verify stage can draw against it.

A source with no declared share is refused rather than defaulted. A trajectory
weight file that was asked for and is not there is refused too: the previous
table keyed its files by the absolute directory they were built from, so moving
the corpus left every lookup missing and every directory silently uniform.

Usage:
    python tools/build_weights.py --root /corpus --kept metadata/C/kept.jsonl \
        --share agibot=0.49 --share oxe=0.49 --share rh20t=0.02 --out metadata/C/weights.json
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import stage

logger = logging.getLogger("build_weights")

REPO = Path(__file__).resolve().parent.parent

# How the mass of one source is divided among its directories. "rows" gives
# every directory the same weight, so a long directory takes proportionally
# more; "equal" gives every directory the same mass whatever its length.
WITHIN = ("rows", "equal")


def parse_shares(pairs: list[str]) -> dict[str, float]:
    """The source=fraction arguments, checked for sign and total."""
    shares: dict[str, float] = {}
    for pair in pairs:
        name, _, value = pair.partition("=")
        if not value:
            raise ValueError(
                f"share {pair!r} has no value\n"
                f"expected source=fraction, as in rh20t=0.02\n"
                f"pass one --share per source"
            )
        shares[name] = float(value)
    if not shares:
        raise ValueError(
            "no shares given\n"
            "expected at least one --share source=fraction\n"
            "a source with no declared share cannot enter the mixture"
        )
    total = sum(shares.values())
    if any(value < 0 for value in shares.values()) or abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"shares sum to {total}\n"
            f"expected non-negative shares summing to 1\n"
            f"fix the --share arguments"
        )
    return shares


def totals(kept: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per directory: its source, its kept episode count and its kept rows."""
    counts: dict[str, dict[str, Any]] = {}
    for record in kept:
        if record["kind"] != "kept":
            continue
        entry = counts.setdefault(
            record["dataset"],
            {"source": record["source"], "name": record["name"], "episodes": 0, "rows": 0},
        )
        entry["episodes"] += 1
        entry["rows"] += record["rows"]
    return counts


def solve(counts: dict[str, dict], shares: dict[str, float], within: str) -> dict[str, float]:
    """The per-directory weights that make each source take its declared share."""
    by_source: dict[str, list[str]] = {}
    for directory, entry in counts.items():
        by_source.setdefault(entry["source"], []).append(directory)
    missing = sorted(set(by_source) - set(shares))
    if missing:
        raise KeyError(
            f"sources {missing} have kept rows and no declared share\n"
            f"expected a --share for every source in the corpus\n"
            f"declare it or leave those directories out of the scan"
        )
    weights: dict[str, float] = {}
    for source, directories in by_source.items():
        rows = sum(counts[directory]["rows"] for directory in directories)
        if rows == 0:
            raise ValueError(
                f"source {source} has no kept rows\n"
                f"expected at least one kept episode\n"
                f"loosen the filter rules or drop the source"
            )
        for directory in directories:
            if within == "rows":
                weights[directory] = shares[source] / rows
            else:
                own = counts[directory]["rows"]
                weights[directory] = shares[source] / len(directories) / max(own, 1)
    largest = max(weights.values())
    return {directory: value / largest for directory, value in weights.items()}


def realised(counts: dict[str, dict], weights: dict[str, float]) -> dict[str, Any]:
    """The share each directory and each source takes under these weights."""
    mass = {
        directory: weights[directory] * counts[directory]["rows"] for directory in counts
    }
    total = sum(mass.values())
    by_dataset = {
        counts[directory]["name"]: round(value / total, 8) for directory, value in mass.items()
    }
    by_source: dict[str, float] = {}
    for directory, value in mass.items():
        source = counts[directory]["source"]
        by_source[source] = by_source.get(source, 0.0) + value / total
    return {"by_dataset": by_dataset,
            "by_source": {key: round(value, 8) for key, value in by_source.items()}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--kept", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--share", action="append", default=[], help="source=fraction")
    parser.add_argument("--within", choices=WITHIN, default="rows")
    parser.add_argument("--trajectory-weights-dir",
                        help="a directory holding one weight file per dataset name")
    parser.add_argument("--trajectory-weights-suffix", default=".npy")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    root = stage.resolve_root(args.root)
    kept_path = stage.under_root(root, args.kept)
    out = stage.under_root(root, args.out)

    _, kept = stage.read_jsonl(kept_path)
    counts = totals(kept)
    stage.refuse_write_into_input(out, [root / name for name in counts])
    shares = parse_shares(args.share)
    weights = solve(counts, shares, args.within)

    trajectory: dict[str, str | None] = {}
    for directory, entry in counts.items():
        if args.trajectory_weights_dir is None:
            trajectory[directory] = None
            continue
        filename = entry["name"] + args.trajectory_weights_suffix
        path = stage.under_root(root, Path(args.trajectory_weights_dir) / filename)
        if not path.is_file():
            raise FileNotFoundError(
                f"no trajectory weight file at {path}\n"
                f"expected one named for every kept directory\n"
                f"build it or drop --trajectory-weights-dir"
            )
        trajectory[directory] = stage.relative(root, path)

    datasets = {
        counts[directory]["name"]: {
            "directory": directory,
            "source": counts[directory]["source"],
            "episodes": counts[directory]["episodes"],
            "rows": counts[directory]["rows"],
            "weight": round(weights[directory], 12),
            "trajectory_weights": trajectory[directory],
        }
        for directory in sorted(counts)
    }
    inputs = {"kept": kept_path}
    if args.trajectory_weights_dir is not None:
        inputs["trajectory_weights"] = stage.under_root(root, args.trajectory_weights_dir)
    payload = {
        "provenance": stage.header(
            "build_weights", root=root, inputs=inputs, tree=REPO,
            extra={"within": args.within, "targets": shares},
        ),
        "targets": shares,
        "within": args.within,
        "expected_share": realised(counts, weights),
        "datasets": datasets,
    }
    written = stage.write_json(out, payload)
    logger.info("%d directories weighted", len(datasets))
    for source, share in sorted(payload["expected_share"]["by_source"].items()):
        logger.info("expected share %s %.4f against target %.4f", source, share, shares[source])
    logger.info("%s %s", "wrote" if written else "unchanged", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
