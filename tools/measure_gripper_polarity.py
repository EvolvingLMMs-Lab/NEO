"""Stage: which raw action-gripper value means open and which closed, per directory.

g_c = (g - raw_open) / (raw_closed - raw_open), 0 open and 1 closed (V6 1.4).
Polarity is measured where the state records an aperture (|corr| >= 0.5, its
sign says which end is closed), else the layout's declared semantics decide;
neither, or a confident measurement contradicting the declaration, refuses.
The raw ends are always the layout's declared stroke, never a directory's own
extremes (they would stretch a gripper that barely moved to the full stroke).
Under NARROW_SPAN of its stroke observed, a directory is flagged; under
MIN_DECLARED_COVERAGE by all its directories, a declaration is refused.

Usage:
    python tools/measure_gripper_polarity.py --root /corpus --inventory metadata/C/inventory.jsonl \
        --kept metadata/C/kept.jsonl --layouts config/layouts/action_pretrain.json \
        --out metadata/C/gripper.json
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import layouts as layout_table
import numpy as np
import stage
import unify

logger = logging.getLogger("measure_gripper_polarity")

REPO = Path(__file__).resolve().parent.parent

METHODS = ("measured", "declared", "masked")

# The least |corr| a measurement decides polarity at (V6 5.1-18). The weakest
# measured directory group is libero_v5a at -0.713.
MIN_CORRELATION = 0.5

# Declared semantics (embodiment_spec.json gripper_semantics [W]) -> whether the
# high raw value is the open end, or None for a gripper that is masked.
SEMANTICS = {
    "binary_open_is_1": True,
    "continuous_width": True,
    "open_is_high": True,
    "binary_closed_is_1": False,
    "closed_is_high": False,
    "constant_fabricated": None,
    "state_leakage": None,
}

DEFAULT_MAX_EPISODES = 200

# How far outside a declared range an observed raw value may sit, as a share of
# the span, before the declaration is refused as not this directory's.
RANGE_TOLERANCE = 0.02

# A directory whose sampled values span less than this share of its declared
# stroke is flagged narrow. It is not refused: the declared ends map it, so a
# gripper that barely moved keeps its place on the stroke. On the real stage-1
# corpus (569 directories measured) 7 are flagged: agibot task_729 (0.16),
# RH20T k06 (0.43) and k11 (0.0118), and four one-episode k14 directories
# whose gripper never moves (span 0).
NARROW_SPAN = 0.5

# A declaration (one source's or one dataset's gripper block) whose directories
# together observe less than this share of the declared stroke is refused: a
# stroke in metres declared as [0, 1] observes about 0.08 of it. On the real
# stage-1 corpus the least covered of the 11 declarations
# observes 0.9996 (dobbe); every other one observes the whole stroke.
MIN_DECLARED_COVERAGE = 0.25


class Refusal(ValueError):
    """One directory's (or one declaration's) refusal, with the reason it is counted under."""

    def __init__(self, reason: str, message: str, expected: str, fix: str) -> None:
        super().__init__(f"{message}\nexpected {expected}\n{fix}")
        self.reason = reason


def canonical_aperture(state: np.ndarray, gripper: dict[str, Any]) -> np.ndarray:
    """The state's aperture in its canonical form, 0 open and 1 closed, unclipped."""
    open_, closed = float(gripper["raw_open"]), float(gripper["raw_closed"])
    return (state[:, int(gripper["column"])] - open_) / (closed - open_)


def correlation(action: np.ndarray, aperture: np.ndarray) -> float | None:
    """Pearson correlation over rows where both are finite, or None when either is still."""
    both = np.isfinite(action) & np.isfinite(aperture)
    if both.sum() < 2:
        return None
    a, b = action[both], aperture[both]
    if a.std() <= 1e-12 or b.std() <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def arm_columns(layout: layout_table.Layout) -> list[tuple[int, dict[str, Any] | None]]:
    """Per arm: the action gripper column and the state aperture spec, if the state has one."""
    states = (layout.state or {}).get("arms", []) if layout.state else []
    found = []
    for index, arm in enumerate(layout.action["arms"]):
        spec = states[index].get("gripper") if index < len(states) else None
        found.append((int(arm["gripper"][0]), spec))
    return found


def gripper_slots(layout: layout_table.Layout) -> set[int]:
    """The arm slots whose action gripper the layout reads, paired as arm_columns pairs them."""
    states = (layout.state or {}).get("arms", []) if layout.state else []
    return {int(states[index].get("slot", index)) if index < len(states) else index
            for index, _ in enumerate(layout.action["arms"])}


def states_no_gripper(episode: dict[str, Any], slots: set[int]) -> bool:
    """Whether the episode states every gripper slot unrecorded (gripper_unrecorded, from main)."""
    return bool(slots) and slots <= set(episode.get("gripper_unrecorded", ()))


def unrecorded_grippers(inventory: list[dict[str, Any]]) -> dict[tuple[str, int], list[int]]:
    """(directory, episode) -> the gripper slots its metadata states unrecorded, if any."""
    found = {}
    for item in inventory:
        stated = ((item.get("metadata") or {}).get("ingest_unrecorded") or {}).get("gripper")
        if stated:
            found[(item["dataset"], item["episode"])] = sorted(int(slot) for slot in stated)
    return found


def check_block(layout: layout_table.Layout, block: dict[str, Any]) -> tuple[float, float] | None:
    """Refuse a gripper block that cannot say what the stroke means; return its range.

    method is one of METHODS. masked states why (evidence) and no polarity;
    measured needs a state aperture to measure against; declared needs its
    semantics; both state the source's raw stroke as range [low, high].
    """
    name = layout.name
    method = block.get("method")
    if method not in METHODS:
        raise Refusal("block", f"{name} states gripper method {method!r}", f"one of {METHODS}",
                      "fix the layout's gripper block")
    semantics = block.get("semantics")
    if semantics is not None and semantics not in SEMANTICS:
        raise Refusal("block", f"{name} declares gripper semantics {semantics!r}",
                      f"one of {sorted(SEMANTICS)}", "fix the layout's gripper block")
    masking = semantics is not None and SEMANTICS[semantics] is None
    if method == "masked":
        if semantics is not None and not masking:
            raise Refusal("block", f"{name} masks its gripper and declares polarity "
                          f"{semantics!r}", "a masked gripper with no polarity, or a trained "
                          "one", "drop the semantics, or train the gripper")
        if not str(block.get("evidence") or "").strip():
            raise Refusal("block", f"{name} masks its gripper and states no evidence",
                          "a written reason the gripper is not trained",
                          "write why into the gripper block's evidence")
        return None
    if masking:
        raise Refusal("block", f"{name} states method {method!r} with semantics "
                      f"{semantics!r}, which masks the gripper", "method masked for a "
                      "masking semantics", "state method masked, or fix the semantics")
    if method == "declared" and semantics is None:
        raise Refusal("block", f"{name} states method declared and no semantics",
                      "the declared polarity", "declare the gripper semantics")
    if method == "measured" and all(spec is None for _, spec in arm_columns(layout)):
        raise Refusal("block", f"{name} states method measured and its state layout has no "
                      "gripper aperture", "a state aperture to measure against",
                      "state method declared with its semantics, or add the aperture")
    if "range" not in block:
        raise Refusal("block", f"{name} states no gripper range",
                      "the source's raw stroke [low, high] (a directory's own extremes would "
                      "stretch a gripper that barely moved to the full stroke)",
                      "declare the source's gripper range in the layout, or mask the gripper")
    stated = block["range"]
    if not isinstance(stated, (list, tuple)) or len(stated) != 2:
        raise Refusal("block", f"{name} declares gripper range {stated!r}",
                      "two ends, [low, high]", "fix the layout's gripper range")
    low, high = (float(value) for value in stated)
    if not (np.isfinite(low) and np.isfinite(high) and high > low):
        raise Refusal("block", f"{name} declares gripper range {stated!r}",
                      "two finite ends, low < high", "fix the layout's gripper range")
    return low, high


def read_rows(root: Path, episodes: list[dict[str, Any]], layout: layout_table.Layout,
              max_rows: int) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Action, state (when the layout has one) and validity rows over the drawn episodes."""
    actions, states, valid = [], [], []
    for episode in episodes:
        path = root / episode["data_file"]
        action = unify.read_action(path, layout.columns["action"], max_rows)
        actions.append(action)
        mask = np.ones(action.shape[0], dtype=bool)
        column = layout.action.get("valid_column")
        if column:
            flags = unify.read_action(path, column, max_rows)
            mask &= flags.reshape(action.shape[0], -1)[:, 0] > 0.5
        valid.append(mask)
        if layout.state is not None:
            states.append(unify.read_action(path, layout.columns["state"], max_rows))
    return (np.concatenate(actions), np.concatenate(states) if states else None,
            np.concatenate(valid))


def decide(name: str, block: dict[str, Any], corrs: list[float | None]) -> tuple[str, bool]:
    """The method and whether the high raw value is the open end, for a block check_block passed."""
    semantics = block.get("semantics")
    declared = None if semantics is None else SEMANTICS[semantics]
    confident = [value for value in corrs if value is not None and abs(value) >= MIN_CORRELATION]
    if confident:
        # A positive correlation with the aperture (1 closed) makes the high end closed.
        # An arm whose gripper or aperture never moves in the sample (None) or moves
        # without a clear relation says nothing; arms that do measure must agree.
        signs = {value < 0 for value in confident}
        if len(signs) != 1:
            raise Refusal("arms_disagree", f"{name} arms measure polarity {corrs}",
                          "every confidently measured arm of one directory with one sign",
                          "check the arms' state apertures; one directory has one gripper map")
        open_is_high = signs.pop()
        if declared is not None and declared != open_is_high:
            raise Refusal("contradiction", f"{name} measures corr {confident} with the "
                          f"aperture and declares {semantics!r}",
                          "a declaration the measurement agrees with",
                          "fix the declaration, or the state aperture's raw_open/raw_closed")
        return "measured", open_is_high
    if declared is None:
        raise Refusal("no_polarity", f"{name} measures |corr| < {MIN_CORRELATION} ({corrs}) "
                      "and declares no semantics", "a measured polarity or a declaration",
                      "declare the gripper semantics in the layout, or mask the gripper")
    return "declared", declared


def within(name: str, stroke: tuple[float, float], seen: tuple[float, float]) -> None:
    """Refuse a directory whose raw values leave its declared stroke (RANGE_TOLERANCE of it)."""
    low, high = stroke
    slack = RANGE_TOLERANCE * (high - low)
    if seen[0] < low - slack or seen[1] > high + slack:
        raise Refusal("outside_range", f"{name} action gripper spans [{seen[0]:.4f}, "
                      f"{seen[1]:.4f}] outside its declared range [{low}, {high}]",
                      "every raw value inside the declared stroke",
                      "fix the range, or check the directory's gripper column")


def declaration_of(layout: layout_table.Layout, block: dict[str, Any]) -> str:
    """The gripper block a directory resolves to: its source and the block itself.

    Directories whose blocks are equal, value for value, share one declaration.
    """
    return f"{layout.source}:{json.dumps(block, sort_keys=True, separators=(',', ':'))}"


def measure(root: Path, directory: str, episodes: list[dict[str, Any]],
            layout: layout_table.Layout, args: argparse.Namespace) -> dict[str, Any]:
    block = layout_table.policy_block(layout, "gripper")
    stroke = check_block(layout, block)
    # An episode that states its gripper unrecorded on every gripper arm (the ingest
    # writer's ingest_unrecorded: the Tien Kung gello's dead follower closure, NaN on
    # every row) has nothing to measure and leaves the sample; a directory made only
    # of such episodes records no gripper, which is what masked states.
    slots = gripper_slots(layout) if stroke is not None else set()
    measurable = [episode for episode in episodes if not states_no_gripper(episode, slots)]
    drawn = unify.sample(measurable, args.max_episodes, f"{args.seed}:{layout.name}")
    record: dict[str, Any] = {"directory": directory, "semantics": block.get("semantics"),
                              "episodes": len(drawn), "episodes_available": len(episodes),
                              "declaration": declaration_of(layout, block)}
    if len(measurable) < len(episodes):
        record["unrecorded_episodes"] = len(episodes) - len(measurable)
    if stroke is None or not measurable:
        record.update(raw_open=None, raw_closed=None, method="masked", corr=[], rows=0)
        return record
    action, state, valid = read_rows(root, drawn, layout, args.max_rows)
    corrs: list[float | None] = []
    raw = []
    for column, spec in arm_columns(layout):
        values = action[valid, column]
        raw.append(values[np.isfinite(values)])
        if spec is None or state is None:
            corrs.append(None)
            continue
        corrs.append(correlation(values, canonical_aperture(state[valid], spec)))
    joined = np.concatenate(raw)
    if joined.size == 0:
        raise Refusal("no_rows", f"{layout.name} has 0 valid finite gripper rows over "
                      f"{len(drawn)} episodes (n = 0)", "rows to measure the gripper on",
                      "check the kept episodes, or mask the gripper")
    method, open_is_high = decide(layout.name, block, corrs)
    seen = (float(joined.min()), float(joined.max()))
    within(layout.name, stroke, seen)
    low, high = stroke
    share = (seen[1] - seen[0]) / (high - low)
    record.update(raw_open=high if open_is_high else low, raw_closed=low if open_is_high else high,
                  method=method, corr=corrs, rows=int(joined.size), range=[low, high],
                  observed=list(seen), span_share=round(share, 6),
                  narrow=bool(share < NARROW_SPAN),
                  distinct_values=int(min(np.unique(joined).size, 1000)))
    return record


def coverage(datasets: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per declaration: its directories, the union of what they observe, its share of the range."""
    found: dict[str, dict[str, Any]] = {}
    for name, record in sorted(datasets.items()):
        if record["method"] == "masked":
            continue
        entry = found.setdefault(record["declaration"], {
            "range": record["range"], "directories": 0, "observed": [np.inf, -np.inf],
            "first": name})
        entry["directories"] += 1
        entry["observed"] = [min(entry["observed"][0], record["observed"][0]),
                             max(entry["observed"][1], record["observed"][1])]
    for entry in found.values():
        low, high = entry["range"]
        entry["coverage"] = round((entry["observed"][1] - entry["observed"][0]) / (high - low), 6)
    return found


# The layouts and arguments a worker measures with, set before the pool forks.
_JOB: dict[str, Any] = {}


def _measure_one(item: tuple[str, str]) -> tuple[str, dict[str, Any] | None, str | None,
                                                 str | None]:
    """(name, record, refusal reason, refusal text) for one directory."""
    directory, name = item
    try:
        record = measure(_JOB["root"], directory, _JOB["grouped"][directory],
                         _JOB["layouts"][directory], _JOB["args"])
    except Refusal as refusal:
        return name, None, refusal.reason, f"{directory}: {refusal}"
    return name, record, None, None


def measure_all(root: Path, grouped: dict[str, list[dict[str, Any]]],
                document: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Every directory's record, the declarations' coverage, or one refusal naming them all."""
    layouts, claimed, names = {}, {}, {}
    for directory in sorted(grouped):
        layout = layout_table.resolve(document, directory)
        names[directory] = layout_table.claim(claimed, layout)
        layouts[directory] = layout
    _JOB.update(root=root, grouped=grouped, layouts=layouts, args=args)
    items = [(directory, names[directory]) for directory in sorted(grouped)]
    try:
        if args.workers > 1:
            with multiprocessing.get_context("fork").Pool(args.workers) as pool:
                results = pool.map(_measure_one, items, chunksize=1)
        else:
            results = [_measure_one(item) for item in items]
    finally:
        _JOB.clear()
    datasets = {name: record for name, record, _, _ in results if record is not None}
    refused = [(reason, text) for _, record, reason, text in results if record is None]
    declarations = coverage(datasets)
    for key, entry in sorted(declarations.items()):
        if entry["coverage"] < MIN_DECLARED_COVERAGE:
            refused.append(("coverage", (
                f"declaration {key} ({entry['directories']} directories, first "
                f"{entry['first']}): together they observe [{entry['observed'][0]:.4f}, "
                f"{entry['observed'][1]:.4f}], {entry['coverage']:.4f} of the declared range "
                f"{entry['range']}\nexpected at least {MIN_DECLARED_COVERAGE} of the stroke "
                f"observed somewhere under one declaration\nthe range is not this data's "
                f"stroke (a unit mismatch?); fix the layout's gripper range")))
    if refused:
        reasons = Counter(reason for reason, _ in refused)
        listed = "\n".join(text.replace("\n", " | ") for _, text in refused[:8])
        raise ValueError(
            f"{len(refused)} refusals over {len(items)} directories (by reason: "
            f"{dict(sorted(reasons.items()))}); {len(datasets)} measured\n{listed}\n"
            f"expected every directory measured or declared, or masked with evidence\n"
            f"fix each named directory's gripper block in the layout")
    for entry in declarations.values():
        entry.pop("first")
    return {"datasets": datasets, "declarations": declarations}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--kept", required=True)
    parser.add_argument("--layouts", required=True, help="a v6 layout file")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-episodes", type=int, default=DEFAULT_MAX_EPISODES)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", default="gripper")
    parser.add_argument("--workers", type=int, default=1,
                        help="directories measured in parallel; the output does not depend on it")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    root = stage.resolve_root(args.root)
    inputs = {"inventory": stage.under_root(root, args.inventory),
              "kept": stage.under_root(root, args.kept),
              "layouts": stage.under_root(root, args.layouts)}
    _, inventory = stage.read_jsonl(inputs["inventory"])
    _, kept = stage.read_jsonl(inputs["kept"])
    document = layout_table.load(inputs["layouts"])
    grouped = unify.kept_by_dataset(kept, inventory)
    stated = unrecorded_grippers(inventory)
    for name, episodes in grouped.items():
        for episode in episodes:
            if (name, episode["episode"]) in stated:
                episode["gripper_unrecorded"] = stated[(name, episode["episode"])]
    del inventory, kept, stated
    if not grouped:
        raise ValueError("0 kept directories (n = 0)\nexpected at least one\n"
                         "check the kept file")
    out = stage.under_root(root, args.out)
    stage.refuse_write_into_input(out, [root / name for name in grouped])

    measured = measure_all(root, grouped, document, args)
    datasets = measured["datasets"]
    methods = Counter(record["method"] for record in datasets.values())
    narrow = sorted(name for name, record in datasets.items() if record.get("narrow"))
    payload = {
        "provenance": stage.header("measure_gripper_polarity", root=root, inputs=inputs,
                                   tree=REPO, extra={"max_episodes": args.max_episodes,
                                                     "max_rows": args.max_rows,
                                                     "seed": args.seed,
                                                     "min_correlation": MIN_CORRELATION,
                                                     "narrow_span": NARROW_SPAN,
                                                     "min_declared_coverage":
                                                         MIN_DECLARED_COVERAGE}),
        "totals": {"directories": len(datasets), "narrow": len(narrow),
                   **{method: methods.get(method, 0) for method in METHODS}},
        "declarations": measured["declarations"],
        "datasets": datasets,
    }
    written = stage.write_json(out, payload)
    logger.info("%d directories: %s", len(datasets),
                ", ".join(f"{methods.get(method, 0)} {method}" for method in METHODS))
    if narrow:
        logger.info("%d directories observe under %g of their declared stroke (mapped by the "
                    "declared ends, not their own): %s%s", len(narrow), NARROW_SPAN,
                    ", ".join(narrow[:8]), " ..." if len(narrow) > 8 else "")
    for key, entry in sorted(measured["declarations"].items()):
        logger.info("declaration %s: %d directories observe %.4f of range %s", key,
                    entry["directories"], entry["coverage"], entry["range"])
    logger.info("%s %s", "wrote" if written else "unchanged", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
