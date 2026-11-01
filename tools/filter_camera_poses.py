"""Rule 5 (apply): mark a camera pose unknown where the projection score failed.

Reads one or more qc_extrinsics.py outputs and, for every episode whose stored
pose scored fail, writes the unknown marker (a NaN 3x3 rotation) in place of
that episode's row of the camera's per-episode rotation array. Episodes are
never removed; ok and inconclusive poses are unchanged. Inconclusive means no
measurement, not a pass.

Only a census is accepted: a run with --control and without --sample, at
threshold 2.5, whose verdict counts match its records and whose every verdict
follows from its score. Only cameras whose manifest extrinsics are per episode
from a separate pose file (kind episode, source stage) can be gated.

Rows of the rotation array follow the directory's kept list (kept_position in
filter_episodes.py output). Arrays holding at least one failed pose are written
under --out-dir at their manifest-relative paths (other arrays are unchanged
and not copied), with pose_qc_summary.json beside them.

Usage:
    python tools/filter_camera_poses.py --root $NEO_VLA_DATA_ROOT \
        --manifest manifests/action_pretrain.json --kept filtered/kept.jsonl \
        --pose-qc filtered/qc_extrinsics_agibot.jsonl --out-dir filtered/poses
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import filter_common as common
import numpy as np

QC_TOOL = "qc_extrinsics"
VERDICTS = ("ok", "fail", "inconclusive")
UNKNOWN_ON = ("fail",)
THRESHOLD = 2.5
SETTINGS = ("sample", "threshold", "episodes", "episodes_read", "verdicts",
            "control_verdicts", "win_rate")
RECORD_KEYS = ("dataset", "view", "episode", "verdict", "score")
GATED_KIND = (("kind", "episode"), ("source", "stage"))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError(f"repeated keys {sorted({k for k in keys if keys.count(k) > 1})}")
    return dict(pairs)


def _refuse(where: str, problem: str, expected: str) -> ValueError:
    return ValueError(f"{where}: {problem}; expected {expected}")


def check_header(where: str, head: Any, verdicts: Counter, records: int) -> None:
    """Refuse a header that is not a qc_extrinsics census with a wrong-pose control."""
    if not isinstance(head, dict) or head.get("tool") != QC_TOOL:
        raise _refuse(where, f"header tool {head.get('tool') if isinstance(head, dict) else None!r}",
                      f"a {QC_TOOL} output")
    settings = head.get("settings")
    if not isinstance(settings, dict):
        raise _refuse(where, "no header settings", f"settings holding {list(SETTINGS)}")
    missing = [key for key in SETTINGS if key not in settings]
    if missing:
        raise _refuse(where, f"header settings lack {missing}", f"settings holding {list(SETTINGS)}")
    if type(settings["sample"]) is not int or settings["sample"] != 0:
        raise _refuse(where, f"sample is {settings['sample']!r}", "sample 0 (a census)")
    if not _is_number(settings["threshold"]) or settings["threshold"] != THRESHOLD:
        raise _refuse(where, f"threshold is {settings['threshold']!r}", f"{THRESHOLD}")
    counted = (settings["episodes"], settings["episodes_read"])
    if any(type(value) is not int for value in counted) or counted != (records, records):
        raise _refuse(where, f"header states {counted[0]!r} scored of {counted[1]!r} read "
                             f"and the file holds {records}", "every episode read scored")
    win, controls = settings["win_rate"], settings["control_verdicts"]
    if (not _is_number(win) or not math.isfinite(win) or not 0.0 <= win <= 1.0
            or not isinstance(controls, dict) or not controls):
        raise _refuse(where, f"win_rate {win!r}, control_verdicts {controls!r}",
                      "a --control run")
    if not isinstance(settings["verdicts"], dict) or settings["verdicts"] != dict(verdicts):
        raise _refuse(where, f"header counts {settings['verdicts']!r}, records "
                             f"{dict(sorted(verdicts.items()))}", "matching verdict counts")


def check_record(where: str, line: int, record: Any) -> tuple[tuple[str, str], int, str]:
    """One QC record -> its (directory, view), episode index and verdict."""
    if not isinstance(record, dict):
        raise _refuse(where, f"line {line} is not an object", "one record per line")
    missing = [key for key in RECORD_KEYS if key not in record]
    if missing:
        raise _refuse(where, f"line {line} lacks {missing}", f"{list(RECORD_KEYS)}")
    dataset, view, episode = record["dataset"], record["view"], record["episode"]
    verdict, score = record["verdict"], record["score"]
    if (not isinstance(dataset, str) or not dataset or dataset == "."
            or Path(dataset).is_absolute() or ".." in Path(dataset).parts
            or Path(dataset).as_posix() != dataset):
        raise _refuse(where, f"line {line} names directory {dataset!r}",
                      "a normalised root-relative directory")
    if not isinstance(view, str) or not view:
        raise _refuse(where, f"line {line} names view {view!r}", "a view key")
    if type(episode) is not int or episode < 0:
        raise _refuse(where, f"line {line} names episode {episode!r}",
                      "a non-negative integer episode index")
    if verdict not in VERDICTS:
        raise _refuse(where, f"line {line} has verdict {verdict!r}", f"one of {list(VERDICTS)}")
    if verdict == "inconclusive":
        if score is not None:
            raise _refuse(where, f"line {line} is inconclusive with score {score!r}", "no score")
    elif (not _is_number(score) or not math.isfinite(score)
          or (score >= THRESHOLD) != (verdict == "ok")):
        raise _refuse(where, f"line {line} has verdict {verdict!r} and score {score!r}",
                      f"a finite score giving that verdict at {THRESHOLD}")
    return (dataset, view), episode, verdict


def read_pose_qc(paths: list[Path]) -> tuple[dict[tuple[str, str], dict[int, str]], list[dict]]:
    """QC files -> verdict per (directory, view) and episode, and per-file counts."""
    verdicts: dict[tuple[str, str], dict[int, str]] = {}
    files = []
    for path in paths:
        where = str(path)
        first, _, rest = path.read_text(encoding="utf-8").partition("\n")
        try:
            head = json.loads(first, object_pairs_hook=_unique_keys)["provenance"]
            records = [json.loads(line, object_pairs_hook=_unique_keys)
                       for line in rest.split("\n") if line.strip()]
        except (ValueError, KeyError, TypeError) as error:
            raise _refuse(where, f"does not parse ({error})",
                          "a provenance header and one JSON record per line") from error
        own: Counter = Counter()
        for line, record in enumerate(records, start=2):
            key, episode, verdict = check_record(where, line, record)
            held = verdicts.setdefault(key, {})
            if episode in held:
                raise _refuse(where, f"line {line} repeats {key[0]} {key[1]} episode {episode}",
                              "one verdict per directory, view and episode")
            held[episode] = verdict
            own[verdict] += 1
        check_header(where, head, own, len(records))
        files.append({"path": where, "verdicts": dict(sorted(own.items())),
                      "win_rate": head["settings"]["win_rate"],
                      "control_verdicts": head["settings"]["control_verdicts"]})
    return verdicts, files


def failed_poses(verdicts: dict[tuple[str, str], dict[int, str]]
                 ) -> dict[tuple[str, str], frozenset[int]]:
    """The episodes whose pose becomes unknown, per (directory, view)."""
    failed = {key: frozenset(episode for episode, verdict in held.items()
                             if verdict in UNKNOWN_ON)
              for key, held in verdicts.items()}
    return {key: episodes for key, episodes in failed.items() if episodes}


def gated(spec: dict[str, Any]) -> bool:
    """Whether a view's extrinsics are per episode from a separate pose file."""
    return all(spec.get(field) == value for field, value in GATED_KIND)


def mark_unknown(rotations: np.ndarray, kept: list[int], failed: frozenset[int]) -> np.ndarray:
    """A copy of [kept, 3, 3] rotations with failed episodes' rows set to NaN."""
    if rotations.shape != (len(kept), 3, 3):
        raise ValueError(f"rotations are {rotations.shape}; expected ({len(kept)}, 3, 3), "
                         "one per kept episode")
    marked = rotations.copy()
    for position, episode in enumerate(kept):
        if episode in failed:
            marked[position] = np.nan
    return marked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--kept", required=True, help="filter_episodes.py output")
    parser.add_argument("--pose-qc", action="append", required=True,
                        help="a qc_extrinsics.py census; repeat per file")
    parser.add_argument("--out-dir", required=True, help="a new directory for corrected arrays")
    args = parser.parse_args(argv)

    root = common.resolve_root(args.root)
    manifest_path = common.cli_path(args.manifest)
    kept_path = common.cli_path(args.kept)
    qc_paths = [common.cli_path(path) for path in args.pose_qc]
    out_dir = common.new_output(args.out_dir, [manifest_path, kept_path, *qc_paths])
    manifest = common.read_json(manifest_path)
    _, kept_records = common.read_jsonl(kept_path)

    verdicts, files = read_pose_qc(qc_paths)
    failed = failed_poses(verdicts)
    entries = {entry["path"]: entry for entry in manifest["datasets"]}
    kept: dict[str, list[dict[str, Any]]] = {}
    for record in kept_records:
        if record["kind"] == "kept":
            kept.setdefault(record["dataset"], []).append(record)
    order = {dataset: [record["episode"] for record in
                       sorted(records, key=lambda record: record["kept_position"])]
             for dataset, records in kept.items()}

    plan = []
    for (dataset, view_key), episodes in sorted(failed.items()):
        entry = entries.get(dataset)
        if entry is None:
            continue
        views = {view["key"]: view for view in entry["views"]}
        view = views.get(view_key)
        if view is None or not gated(view["extrinsics"]):
            raise ValueError(f"{dataset} view {view_key!r} has {len(episodes)} failed poses and "
                             f"extrinsics {None if view is None else view['extrinsics']}; only "
                             "kind episode, source stage views can be gated")
        plan.append((entry, view, episodes))

    marked_counts: dict[str, int] = {}
    kept_by_verdict: Counter = Counter()
    for entry in entries.values():
        for view in entry["views"]:
            if gated(view["extrinsics"]):
                held = verdicts.get((entry["path"], view["key"]), {})
                for episode in order.get(entry["path"], []):
                    kept_by_verdict[held.get(episode, "unscored")] += 1
    for entry, view, episodes in plan:
        relative = view["extrinsics"]["path"]
        rotations = np.load(common.under_root(root, relative), allow_pickle=False)
        marked = mark_unknown(rotations, order.get(entry["path"], []), episodes)
        target = out_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            np.save(stream, marked, allow_pickle=False)
        marked_counts[f"{entry['path']}:{view['key']}"] = int(
            np.isnan(marked).all(axis=(1, 2)).sum() - np.isnan(rotations).all(axis=(1, 2)).sum())

    common.write_json(out_dir / "pose_qc_summary.json", {
        "provenance": common.header(
            "filter_camera_poses",
            inputs=[common.display(root, path) for path in (manifest_path, kept_path, *qc_paths)],
            settings={"threshold": THRESHOLD, "unknown_on": list(UNKNOWN_ON)}),
        "files": files,
        "kept_episodes_by_verdict": dict(sorted(kept_by_verdict.items())),
        "poses_marked_unknown": marked_counts,
    })
    print(f"{sum(marked_counts.values())} poses marked unknown in {len(plan)} arrays; "
          f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
