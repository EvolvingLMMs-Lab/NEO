"""Rule 4 (part 2): drop corrupt episodes and record the rule that fired.

Reads validate_episodes.py output and writes one kept record per surviving
episode and one dropped record per removal, naming the first listed rule the
episode's defects match and the value that fired it. Kept plus dropped must
equal validated. Defects not named by a rule are reported and kept.

Kept records carry each episode's position in its directory's kept list, the
order processed arrays are indexed by after filtering.

Usage:
    python tools/filter_episodes.py --root $NEO_VLA_DATA_ROOT \
        --validation filtered/validation.jsonl --out filtered/kept.jsonl \
        --drop data_file_missing --drop episode_unreadable --drop rows_mismatch \
        --drop frames_missing --drop episode_too_short --drop state_width_mismatch \
        --drop action_width_mismatch --drop not_finite --drop quaternion_not_unit \
        --drop action_equals_state
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import filter_common as common

# How many unreadable kept episodes the refusal lists.
LISTED_UNREADABLE = 10

# Defects that leave no bytes to read. A kept episode with one of these would
# fail when the loader opens it, so keeping one is refused unless asked for.
UNREADABLE = (
    "data_file_missing",
    "episode_unreadable",
    "rows_mismatch",
    "frames_missing",
)


def unreadable_defects(record: dict[str, Any]) -> list[str]:
    """The names among one episode's defects that leave nothing to read."""
    return sorted({defect["name"] for defect in record["defects"]
                   if defect["name"] in UNREADABLE})


def read_rules(listed: list[str], rules_file: Path | None) -> list[str]:
    """The defect names that drop an episode, in the order they are checked."""
    names = list(listed)
    if rules_file is not None:
        names.extend(
            line.strip() for line in rules_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    return names


def first_firing(record: dict[str, Any], rules: list[str]) -> dict[str, Any] | None:
    """The first rule this episode's defects match, with the defect that fired it."""
    by_name = {defect["name"]: defect for defect in record["defects"]}
    for rule in rules:
        if rule in by_name:
            return by_name[rule]
    return None


def filter_records(records: list[dict[str, Any]], rules: list[str]
                   ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validation records -> kept and dropped records, and what the kept set carries."""
    out_records: list[dict[str, Any]] = []
    reasons: Counter = Counter()
    position: Counter = Counter()
    carried: Counter = Counter()
    failing = 0
    unreadable: list[dict[str, Any]] = []
    for record in records:
        fired = first_firing(record, rules)
        if fired is None:
            out_records.append({
                "kind": "kept",
                "dataset": record["dataset"],
                "name": record["name"],
                "source": record["source"],
                "episode": record["episode"],
                "kept_position": position[record["dataset"]],
                "rows": record["rows"],
                "extrinsics": record["extrinsics"],
                "defects": [defect["name"] for defect in record["defects"]],
            })
            if not record["pass"]:
                failing += 1
            for defect in {defect["name"] for defect in record["defects"]}:
                carried[defect] += 1
            names = unreadable_defects(record)
            if names:
                unreadable.append({
                    "dataset": record["dataset"],
                    "episode": record["episode"],
                    "kept_position": position[record["dataset"]],
                    "defects": names,
                })
            position[record["dataset"]] += 1
        else:
            out_records.append({
                "kind": "dropped",
                "dataset": record["dataset"],
                "name": record["name"],
                "source": record["source"],
                "episode": record["episode"],
                "rows": record["rows"],
                "rule": fired["name"],
                "value": {key: value for key, value in fired.items() if key != "name"},
            })
            reasons[fired["name"]] += 1

    kept = sum(1 for record in out_records if record["kind"] == "kept")
    dropped = len(out_records) - kept
    if kept + dropped != len(records):
        raise AssertionError(f"kept {kept} plus dropped {dropped} is not {len(records)} validated")
    summary = {"validated": len(records), "kept": kept, "dropped": dropped,
               "dropped_by_rule": dict(sorted(reasons.items())),
               "kept_failing_validation": failing,
               "kept_with_defect": dict(sorted(carried.items())),
               "kept_unreadable": unreadable}
    return out_records, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--validation", required=True, help="validate_episodes.py output")
    parser.add_argument("--out", required=True, help="new JSONL output")
    parser.add_argument("--drop", action="append", default=[],
                        help="a defect name that removes an episode")
    parser.add_argument("--rules-file", help="a file of defect names, one per line")
    parser.add_argument("--keep-unreadable", action="store_true",
                        help="keep episodes with no bytes to read (listed in the header)")
    args = parser.parse_args(argv)

    root = common.resolve_root(args.root)
    validation_path = common.cli_path(args.validation)
    rules_path = common.cli_path(args.rules_file) if args.rules_file else None
    _, records = common.read_jsonl(validation_path)
    directories = sorted({record["dataset"] for record in records})
    out = common.new_output(args.out, [root / name for name in directories])
    rules = read_rules(args.drop, rules_path)

    out_records, summary = filter_records(records, rules)
    unreadable = summary["kept_unreadable"]
    if unreadable and not args.keep_unreadable:
        listed = "\n".join(
            f"  {item['dataset']} episode {item['episode']} at kept position "
            f"{item['kept_position']}: {', '.join(item['defects'])}"
            for item in unreadable[:LISTED_UNREADABLE])
        raise ValueError(f"{len(unreadable)} kept episodes have no bytes to read:\n{listed}\n"
                         "name the defect in --drop, or pass --keep-unreadable")

    inputs = [common.display(root, validation_path)]
    if rules_path is not None:
        inputs.append(common.display(root, rules_path))
    head = common.header("filter_episodes", inputs=inputs,
                         settings={"rules": rules, "keep_unreadable": bool(args.keep_unreadable),
                                   **summary})
    common.write_jsonl(out, head, out_records)
    print(f"{summary['validated']} validated, {summary['kept']} kept, "
          f"{summary['dropped']} dropped; wrote {out}")
    for name, count in summary["dropped_by_rule"].items():
        print(f"  {name}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
