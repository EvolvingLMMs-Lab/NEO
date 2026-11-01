"""Stage: each directory's control frequency, the number section 2 of the prompt states.

control_hz is the nominal control rate of the recorded stream as the source
declares it: an env or converter constant, the RH20T grid spacing, the AgiBot
gates or the OXE published sheet (V6 section 2, C14). Where a directory stores
a clock, its median row rate must lie within the policy's tolerance of that
nominal or the directory states NA; the measurement is kept as evidence either
way. A published family missing from the table, an info.json fps that
disagrees with the directory name or the env constant, and a measurement over
zero rows refuse rather than fall back to NA.

Usage:
    python tools/declare_frequency.py --root /corpus --kept prep/kept.jsonl \
        --layouts config/layouts/gd7.json --published config/rates/oxe_published.json \
        --scale prep/scale.json --out prep/frequency.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import layouts as layout_table
import numpy as np
import pyarrow.parquet as pq
import stage

logger = logging.getLogger("declare_frequency")

REPO = Path(__file__).resolve().parent.parent

# Where each policy's number comes from, and the method word the manifest
# records for it (V6 section 2, manifest field control_hz_method).
METHODS = {
    "agibot_gates": "measured",
    "droid_nominal": "metadata",
    "rh20t_grid": "measured",
    "published": "published",
    "env_constant": "metadata",
    "info_fps": "metadata",
    "none": "na",
}
POLICIES = tuple(METHODS)

# How far a stored clock may sit from the nominal before the directory says NA
# (C14). The AgiBot gates are the converter's own per-episode rate
# measurement, 99.68% of them within 1% of 30 [V6 section 2], so a directory
# whose median misses by more than 1% is a recording that is not the one the
# nominal describes.
DEFAULT_TOLERANCE = 0.10
POLICY_TOLERANCE = {"agibot_gates": 0.01}

# The column a clock policy reads its stored timestamps from, per row.
CLOCK_COLUMN = {"droid_nominal": "source_timestamp_ms"}

# Episodes a clock is measured over. The DROID measurement behind V6 used 164.
DEFAULT_CLOCK_EPISODES = 200

# The QC band on hz x scale_cm_per_step. A directory outside it keeps its
# value and records a warning: the published rate is available, and the miss
# comes from the action values (V6 section 2, D2-A).
SPEED_BAND_CM_S = (5.0, 45.0)

# The RH20T grid names its rate in the directory: rh20t_grid_k10_... is 10 Hz.
GRID_RATE = re.compile(r"(?:^|_)k(\d+)(?:_|$)")

# A published family is looked up with the shard suffix removed.
SHARD = re.compile(r"_shard\d+of\d+$")


def refuse(message: str, expected: str, fix: str) -> ValueError:
    return ValueError(f"{message}\nexpected {expected}\n{fix}")


def info_of(root: Path, directory: str) -> dict[str, Any]:
    """A directory's meta/info.json, or an empty dict when it has none."""
    path = root / directory / "meta" / "info.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def gates_of(root: Path, directory: str) -> dict[str, Any]:
    """The AgiBot gates' per-episode rates: count, median, tails, share near the median."""
    path = root / directory / "meta" / "agibot_gates.jsonl"
    if not path.exists():
        raise refuse(f"{directory} has no meta/agibot_gates.jsonl",
                     "the converter's per-episode gate file for an agibot_gates policy",
                     "re-run convert_agibot for the directory, or change its policy")
    rates = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line).get("hz")
            if isinstance(value, (int, float)) and np.isfinite(value):
                rates.append(float(value))
    if not rates:
        raise refuse(f"{directory} gates hold 0 episodes with a finite hz (n = 0)",
                     "at least one measured rate", "a frequency measured over nothing "
                     "is not a measurement; re-run convert_agibot")
    values = np.asarray(rates)
    return {"n": len(rates), "median": float(np.median(values)),
            "p1": float(np.percentile(values, 1)), "p99": float(np.percentile(values, 99))}


def clock_of(root: Path, directory: str, column: str, episodes: int) -> dict[str, Any]:
    """The median interval between consecutive stored rows, over evenly spaced episodes.

    One lerobot data file holds one episode, so a difference is taken inside a
    file and never across two.
    """
    files = sorted((root / directory / "data").rglob("*.parquet"))
    if episodes and len(files) > episodes:
        files = [files[index] for index in np.linspace(0, len(files) - 1, episodes).astype(int)]
    intervals = []
    for path in files:
        table = pq.read_table(path, columns=[column])
        stamps = np.asarray(table.column(column).to_numpy(), dtype=np.float64)
        if stamps.size >= 2:
            intervals.append(np.diff(stamps))
    if not intervals or sum(block.size for block in intervals) == 0:
        raise refuse(f"{directory} gives 0 intervals of {column} over {len(files)} files (n = 0)",
                     "at least one pair of consecutive stored rows",
                     "check the data files, or state the directory's policy as none")
    joined = np.concatenate(intervals)
    median = float(np.median(joined))
    if not median > 0:
        raise refuse(f"{directory} median {column} interval is {median}",
                     "a positive interval", "the stored clock is not monotonic; check it")
    return {"n_intervals": int(joined.size), "n_episodes": len(intervals),
            "median_interval": median}


def within(measured: float, nominal: float, tolerance: float) -> bool:
    return abs(measured / nominal - 1.0) <= tolerance


def nominal_of(block: dict[str, Any], where: str) -> float:
    value = block.get("hz")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        raise refuse(f"{where} policy {block.get('from')!r} states hz {value!r}",
                     "a positive nominal rate in the layout block",
                     "write the source's nominal rate into the layout's frequency block")
    return float(value)


def family_of(directory: str, block: dict[str, Any]) -> str:
    """The published-sheet family: stated in the block, else the basename unsharded."""
    stated = block.get("family")
    if stated:
        return str(stated)
    return SHARD.sub("", directory.rstrip("/").rsplit("/", 1)[-1])


def declare(root: Path, directory: str, layout: layout_table.Layout,
            published: dict[str, Any] | None, args: argparse.Namespace) -> dict[str, Any]:
    """One directory's control_hz, method, evidence and warnings."""
    block = layout_table.policy_block(layout, "frequency")
    policy = block.get("from")
    if policy not in POLICIES:
        raise refuse(f"{layout.name} frequency policy is {policy!r}",
                     f"one of {list(POLICIES)}", "fix the layout's frequency block")
    tolerance = float(block.get("tolerance", POLICY_TOLERANCE.get(policy, DEFAULT_TOLERANCE)))
    info = info_of(root, directory)
    record: dict[str, Any] = {"directory": directory, "policy": policy, "measured_hz": None,
                              "tolerance": None, "warnings": []}
    value: float | None
    if policy == "agibot_gates":
        value = nominal_of(block, layout.name)
        gates = gates_of(root, directory)
        record.update(measured_hz=gates["median"], tolerance=tolerance)
        evidence = (f"meta/agibot_gates.jsonl hz median {gates['median']:.4f} over "
                    f"{gates['n']} episodes (p1 {gates['p1']:.4f}, p99 {gates['p99']:.4f}); "
                    f"nominal {value:g}")
        if not within(gates["median"], value, tolerance):
            evidence += f"; outside {tolerance:.0%} of the nominal, so NA"
            value = None
    elif policy == "droid_nominal":
        value = nominal_of(block, layout.name)
        column = str(block.get("timestamp_column", CLOCK_COLUMN[policy]))
        clock = clock_of(root, directory, column, args.clock_episodes)
        measured = 1000.0 / clock["median_interval"]
        record.update(measured_hz=measured, tolerance=tolerance)
        evidence = (f"nominal {value:g} ({block.get('evidence', 'converter constant')}); stored "
                    f"clock {column} median interval {clock['median_interval']:.2f} ms = "
                    f"{measured:.3f} Hz over {clock['n_intervals']} intervals in "
                    f"{clock['n_episodes']} episodes")
        if not within(measured, value, tolerance):
            evidence += f"; outside {tolerance:.0%} of the nominal, so NA"
            value = None
    elif policy == "rh20t_grid":
        found = GRID_RATE.search(directory.rstrip("/").rsplit("/", 1)[-1])
        if found is None:
            raise refuse(f"{directory} names no grid rate (k<N>)",
                         "an RH20T grid directory named rh20t_grid_k<N>_...",
                         "check the directory list, or change its policy")
        value = float(found.group(1))
        fps = info.get("fps")
        if fps is None or float(fps) != value:
            raise refuse(f"{directory} is named k{found.group(1)} and its info.json fps is {fps}",
                         "the grid rate in the name to equal info.json fps",
                         "rebuild the grid directory; the name or the metadata is wrong")
        evidence = (f"directory name k{found.group(1)} = info.json fps {fps}; rows are 1/K s "
                    f"apart by construction (build_rh20t_grid)")
    elif policy == "published":
        if published is None:
            raise refuse(f"{layout.name} has a published policy and no --published table",
                         "config/rates/oxe_published.json", "pass --published")
        family = family_of(directory, block)
        rates = published["rates"]
        if family not in rates:
            raise refuse(f"{directory} is published family {family!r}, which the table lacks",
                         f"a rate for every published family (the table has {len(rates)})",
                         "add the family's published rate; a missing one is never NA")
        value = float(rates[family])
        evidence = (f"published sheet {published.get('source')} fetched "
                    f"{published.get('fetched')}, family {family}")
        if "fps" in info:
            evidence += f"; info.json fps {info['fps']} is the converter's default, not a clock"
    elif policy == "env_constant":
        value = nominal_of(block, layout.name)
        if not block.get("evidence"):
            raise refuse(f"{layout.name} env_constant states no evidence",
                         "the code site the constant is read from", "add evidence to the block")
        evidence = f"env constant {value:g}: {block['evidence']}"
    elif policy == "info_fps":
        fps = info.get("fps")
        if fps is None:
            raise refuse(f"{directory} has no info.json fps", "meta/info.json with fps",
                         "state another policy for the directory")
        value = float(fps)
        if "hz" in block and float(block["hz"]) != value:
            raise refuse(f"{directory} info.json fps {fps} disagrees with the env constant "
                         f"{block['hz']}", "the two to agree",
                         "fix the converter's fps or the layout's env constant")
        evidence = f"meta/info.json fps {fps}"
    else:
        if not block.get("evidence"):
            raise refuse(f"{layout.name} frequency policy none states no evidence",
                         "the reason the rate is not available", "add evidence to the block")
        value = None
        evidence = f"NA: {block['evidence']}"
    record["control_hz"] = value
    record["control_hz_method"] = "na" if value is None else METHODS[policy]
    record["control_hz_evidence"] = evidence
    return record


def speed_check(record: dict[str, Any], scale: float | None) -> None:
    """hz x scale_cm_per_step, recorded, and a warning outside the band."""
    record["q99_speed_cm_s"] = None
    if record["control_hz"] is None or scale is None:
        return
    speed = float(record["control_hz"]) * float(scale)
    record["q99_speed_cm_s"] = round(speed, 4)
    low, high = SPEED_BAND_CM_S
    if not low <= speed <= high:
        warning = (f"speed check: {record['control_hz']:g} Hz x {scale:.4f} cm/step = "
                   f"{speed:.1f} cm/s is outside [{low:g}, {high:g}]; value kept")
        record["warnings"].append(warning)
        record["control_hz_evidence"] += "; " + warning


def directories_of(root: Path, args: argparse.Namespace) -> tuple[list[str], dict[str, Path]]:
    """The directories to declare, from the kept file or a manifest, and that input."""
    if bool(args.kept) == bool(args.manifest):
        raise refuse("pass exactly one of --kept and --manifest", "one directory list",
                     "--kept before a manifest exists, --manifest to re-declare a corpus")
    if args.kept:
        path = stage.under_root(root, args.kept)
        _, kept = stage.read_jsonl(path)
        found = sorted({record["dataset"] for record in kept if record["kind"] == "kept"})
        return found, {"kept": path}
    path = stage.under_root(root, args.manifest)
    found = sorted(entry["path"] for entry in stage.read_json(path)["datasets"])
    return found, {"manifest": path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--kept", help="the filter stage's kept file (directory list)")
    parser.add_argument("--manifest", help="a manifest, as the directory list instead")
    parser.add_argument("--layouts", required=True, help="a v6 layout file")
    parser.add_argument("--published", help="config/rates/oxe_published.json")
    parser.add_argument("--scale", help="measure_scale's output, for the speed check")
    parser.add_argument("--clock-episodes", type=int, default=DEFAULT_CLOCK_EPISODES)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    root = stage.resolve_root(args.root)
    directories, inputs = directories_of(root, args)
    if not directories:
        raise refuse("0 directories to declare (n = 0)", "at least one kept directory",
                      "check the kept file or the manifest")
    inputs["layouts"] = stage.under_root(root, args.layouts)
    document = layout_table.load(inputs["layouts"])
    published = None
    if args.published:
        # The table is a repository config; it is hashed from wherever it is.
        published_path = Path(args.published)
        if not published_path.is_absolute():
            published_path = (REPO / published_path) if (REPO / published_path).exists() \
                else stage.under_root(root, args.published)
        published = json.loads(published_path.read_text(encoding="utf-8"))
    scale = {}
    if args.scale:
        inputs["scale"] = stage.under_root(root, args.scale)
        scale = stage.read_json(inputs["scale"])["datasets"]
    out = stage.under_root(root, args.out)
    stage.refuse_write_into_input(out, [root / name for name in directories])

    datasets: dict[str, Any] = {}
    claimed: dict[str, str] = {}
    for directory in directories:
        layout = layout_table.resolve(document, directory)
        record = declare(root, directory, layout, published, args)
        speed_check(record, (scale.get(layout.name) or {}).get("scale_cm_per_step"))
        datasets[layout_table.claim(claimed, layout)] = record
    valued = sum(1 for record in datasets.values() if record["control_hz"] is not None)
    methods = Counter(record["control_hz_method"] for record in datasets.values())
    warned = sum(1 for record in datasets.values() if record["warnings"])
    payload = {
        "provenance": stage.header("declare_frequency", root=root, inputs=inputs, tree=REPO,
                                   extra={"clock_episodes": args.clock_episodes,
                                          "published": args.published}),
        "totals": {"directories": len(datasets), "with_value": valued,
                   "na": len(datasets) - valued, "by_method": dict(sorted(methods.items())),
                   "warnings": warned},
        "datasets": datasets,
    }
    written = stage.write_json(out, payload)
    logger.info("%d of %d directories state a frequency, %d state NA", valued, len(datasets),
                len(datasets) - valued)
    logger.info("by method: %s; %d with a speed-check warning", dict(sorted(methods.items())),
                warned)
    logger.info("%s %s", "wrote" if written else "unchanged", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
