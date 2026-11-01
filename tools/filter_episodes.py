"""Stage 3: what survives, and a named reason for everything that does not.

Reads validation.jsonl and writes one kept record per surviving episode plus one
dropped record per removal, naming the rule that fired and the value that fired
it. Kept plus dropped equals validated, checked before the file is written: a
rule that drops without recording is a defect in this tool, not in the corpus.

The kept records carry the position of each episode in its directory's kept
list. That position, not the episode index, is what every later stage keys on,
because a filtered directory renumbers everything after its first drop.

Usage:
    python tools/filter_episodes.py --root /corpus --validation prep/validation.jsonl \
        --drop episode_unreadable --drop frames_missing --out prep/kept.jsonl
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import stage

logger = logging.getLogger("filter_episodes")

# How many unreadable episodes the refusal names before it stops listing.
LISTED_UNREADABLE = 10

REPO = Path(__file__).resolve().parent.parent

# The defects that mean the episode's bytes are not there. Nothing later can
# recover from one: the loader opens the file, raises, and takes the worker
# and then the collective down with it. Every other defect describes content
# a run may still reasonably train on, so only these refuse on their own.
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--drop", action="append", default=[],
                        help="a defect name that removes an episode")
    parser.add_argument("--rules-file", help="a file of defect names, one per line")
    parser.add_argument("--keep-unreadable", action="store_true",
                        help="keep episodes with no bytes to read, having given "
                             "them a trajectory weight of zero")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    root = stage.resolve_root(args.root)
    validation_path = stage.under_root(root, args.validation)
    out = stage.under_root(root, args.out)

    _, records = stage.read_jsonl(validation_path)
    directories = sorted({record["dataset"] for record in records})
    stage.refuse_write_into_input(out, [root / name for name in directories])
    rules = read_rules(
        args.drop,
        stage.under_root(root, args.rules_file) if args.rules_file else None,
    )

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
            # What the kept set carries, counted here rather than left for a
            # reader of the corpus: a kept episode that validation marked
            # failed is a decision, and a decision nobody states is a defect.
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
        raise AssertionError(
            f"kept {kept} plus dropped {dropped} is not {len(records)} validated\n"
            f"expected one output record per validation record\n"
            f"this is a defect in filter_episodes, not in the corpus"
        )

    if unreadable and not args.keep_unreadable:
        listed = "\n".join(
            f"  {item['dataset']} episode {item['episode']} at kept position "
            f"{item['kept_position']}: {', '.join(item['defects'])}"
            for item in unreadable[:LISTED_UNREADABLE]
        )
        raise ValueError(
            f"{len(unreadable)} kept episodes have no bytes to read:\n{listed}\n"
            f"expected every kept episode to be readable, since the loader raises "
            f"on one that is not and the raise ends the run\n"
            f"name the defect in --drop, or pass --keep-unreadable and give those "
            f"directories a trajectory weight of zero at the positions above"
        )

    head = stage.header(
        "filter_episodes",
        root=root,
        inputs={"validation": validation_path},
        tree=REPO,
        extra={"rules": rules,
               "keep_unreadable": bool(args.keep_unreadable),
               "kept_failing_validation": failing,
               "kept_with_defect": dict(sorted(carried.items())),
               "kept_unreadable": unreadable},
    )
    written = stage.write_jsonl(out, head, out_records)
    logger.info("%d validated, %d kept, %d dropped", len(records), kept, dropped)
    for name, count in sorted(reasons.items()):
        logger.info("rule %s dropped %d", name, count)
    logger.info("%d kept episodes validation marked failed", failing)
    for name, count in sorted(carried.items()):
        logger.info("kept carrying defect %s: %d", name, count)
    for item in unreadable:
        logger.warning("kept with nothing to read: %s episode %d at kept position %d, %s",
                       item["dataset"], item["episode"], item["kept_position"],
                       ", ".join(item["defects"]))
    logger.info("%s %s", "wrote" if written else "unchanged", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
