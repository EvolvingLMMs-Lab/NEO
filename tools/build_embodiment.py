"""Stage: each directory's normaliser key, and per-episode keys joined by identity (R16).

A layout names a directory's key, or says it is per episode (RH20T, meta
rh20t_robot). Kept position k joins its source episode by id, kept[k].episode,
never meta line k: a positional join mislabels 7,485 of 32,323 RH20T episodes
[V6 1.8]. A per-episode directory gets int8 key indices and int64 source ids in
kept order, and the join is refused (R16, CX 09) on a duplicate id, an id meta
lacks, an unknown robot, a length that disagrees, or an arrow or prompt id that
is not the kept id. verify_corpus check (vii) reruns check_identity.

Usage:
    python tools/build_embodiment.py --root /corpus --kept prep/kept.jsonl \
        --layouts config/layouts/gd7.json --embodiments config/embodiments.yaml \
        --arrays cache/embodiment --out prep/embodiment.json [--manifest manifests/gd5.json]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import layouts as layout_table
import numpy as np
import stage
import yaml
from pyarrow import feather

from neo_vla.constants import refusal_prefix

logger = logging.getLogger("build_embodiment")

REPO = Path(__file__).resolve().parent.parent

# The two forms an embodiment block takes in a v6 layout,
# and the two a manifest entry records.
LAYOUT_KINDS = ("constant", "rh20t_meta")

# Where the per-episode robot string sits in a directory's meta.
ROBOT_FIELD = "rh20t_robot"

# Key indices are stored as int8, so a directory can hold at most this many.
MAX_KEYS = 127


def refused(where: str, what: str, expected: str, fix: str) -> ValueError:
    """An R16 refusal in the repository's three-line form."""
    return ValueError(f"{refusal_prefix(16)} {where}: {what}\nExpected {expected}.\n{fix}")


def read_embodiments(path: Path) -> dict[str, Any]:
    """config/embodiments.yaml: its keys and the RH20T robot-string table."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    for field in ("keys", "rh20t_robot_keys"):
        if not isinstance(document.get(field), dict):
            raise ValueError(
                f"{path} has no {field} table\n"
                f"expected config/embodiments.yaml with that table\n"
                f"write the {field} block"
            )
    unknown = sorted(set(document["rh20t_robot_keys"].values()) - set(document["keys"]))
    if unknown:
        raise ValueError(
            f"{path} maps RH20T robots to keys it does not define: {unknown}\n"
            f"expected every rh20t_robot_keys value under keys\n"
            f"add the key or fix the mapping"
        )
    return document


def read_meta(root: Path, directory: str) -> dict[int, dict[str, Any]]:
    """meta/episodes.jsonl by episode_index, refusing an index written twice."""
    path = root / directory / "meta" / "episodes.jsonl"
    found: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        index = int(record["episode_index"])
        if index in found:
            raise refused(directory, f"meta/episodes.jsonl states episode_index {index} twice",
                          "one meta line per source episode", "rebuild the directory's meta")
        found[index] = record
    return found


def check_identity(
    where: str,
    *,
    ids: Sequence[int],
    meta: dict[int, dict[str, Any]],
    robot_keys: dict[str, str],
    kept_ids: Sequence[int] | None = None,
    rows: Sequence[int] | None = None,
    starts: np.ndarray | None = None,
    arrow_ids: Sequence[int] | None = None,
    prompt_ids: Sequence[int] | None = None,
) -> list[str]:
    """The key of each position, after every identity check R16 names.

    ids is the per-position source episode id under test: the kept file's at
    build time, the stored ids array at verify time. Each other argument is a
    witness of the same positions read another way, checked when given.
    """
    ids = [int(value) for value in ids]
    seen: dict[int, int] = {}
    for position, value in enumerate(ids):
        if value in seen:
            raise refused(where, f"source episode {value} is at kept positions {seen[value]} "
                          f"and {position}", "each source episode at one position",
                          "rebuild the kept file; an episode kept twice is a join defect")
        seen[value] = position
    witnesses = {"the kept file": kept_ids, "the arrow cache at episode_starts[k]": arrow_ids,
                 "the prompt cache": prompt_ids}
    for name, witness in witnesses.items():
        if witness is None:
            continue
        witness = [int(value) for value in witness]
        if len(witness) != len(ids):
            raise refused(where, f"{name} holds {len(witness)} positions and the ids {len(ids)}",
                          "one position per kept episode in both",
                          "rebuild the stage that wrote the shorter one")
        for position, (value, other) in enumerate(zip(ids, witness)):
            if value != other:
                raise refused(where, f"position {position} is source episode {value} and "
                              f"{name} says {other}", "the same source episode at every "
                              "position, whatever reads it", "rebuild the embodiment arrays "
                              "and the cache from one kept file")
    if starts is not None and len(starts) != len(ids) + 1:
        raise refused(where, f"episode_starts holds {len(starts) - 1} episodes for "
                      f"{len(ids)} positions", "one boundary per kept episode, plus one",
                      "rebuild the cache from this kept file")
    keys = []
    for position, value in enumerate(ids):
        record = meta.get(value)
        if record is None:
            raise refused(where, f"position {position} is source episode {value}, which "
                          f"meta/episodes.jsonl does not hold ({len(meta)} lines)",
                          "every kept episode in the directory's meta", "rebuild the kept "
                          "file against this directory, or restore its meta")
        robot = record.get(ROBOT_FIELD)
        if robot not in robot_keys:
            raise refused(where, f"source episode {value} has {ROBOT_FIELD} {robot!r}",
                          f"one of {sorted(robot_keys)}", "add the robot to "
                          "config/embodiments.yaml rh20t_robot_keys, or fix the meta")
        length = record.get("length")
        lengths = {"the kept file's rows": None if rows is None else int(rows[position]),
                   "diff(episode_starts)": None if starts is None
                   else int(starts[position + 1] - starts[position])}
        for name, stated in lengths.items():
            if stated is not None and length is not None and stated != int(length):
                raise refused(where, f"source episode {value} has meta length {length} and "
                              f"{name} {stated} at position {position}",
                              "the same row count on both sides", "rebuild the cache from "
                              "this kept file; the join is not the one meta describes")
        keys.append(robot_keys[robot])
    return keys


def npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def kept_of(kept: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """The kept records of each directory in kept order, refusing a gap."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in kept:
        if record["kind"] == "kept":
            grouped.setdefault(record["dataset"], []).append(record)
    for directory, records in grouped.items():
        stated = [record.get("kept_position") for record in records]
        if all(value is not None for value in stated) and stated != list(range(len(records))):
            raise refused(directory, f"kept_position runs {stated[:5]}...",
                          "0, 1, 2, ... in file order", "rebuild the kept file")
    return grouped


def cache_witnesses(root: Path, entry: dict[str, Any] | None,
                    prompt_cache: str | None) -> dict[str, Any]:
    """episode_starts, the arrow episode ids at each start and the prompt ids, if built."""
    if entry is None:
        return {}
    starts = np.load(root / entry["episode_starts"]).astype(np.int64)
    table = feather.read_table(root / entry["cache"]["arrow"], columns=["episode_index"],
                               memory_map=True)
    column = np.asarray(table.column("episode_index").to_numpy(), dtype=np.int64)
    found = {"starts": starts, "arrow_ids": column[starts[:-1]].tolist()}
    if prompt_cache is not None:
        path = root / prompt_cache / f"{entry['name']}.jsonl"
        if path.exists():
            found["prompt_ids"] = [
                int(json.loads(line)["episode"])
                for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return found


def per_episode(root: Path, directory: str, name: str, records: list[dict[str, Any]],
                document: dict[str, Any], witnesses: dict[str, Any],
                arrays: str) -> tuple[dict[str, Any], dict[str, bytes]]:
    """One RH20T directory's manifest block and the two arrays behind it."""
    ids = [int(record["episode"]) for record in records]
    rows = None
    if all("rows" in record for record in records):
        rows = [int(record["rows"]) for record in records]
    keys = check_identity(
        directory, ids=ids, meta=read_meta(root, directory),
        robot_keys=document["rh20t_robot_keys"], rows=rows,
        starts=witnesses.get("starts"), arrow_ids=witnesses.get("arrow_ids"),
        prompt_ids=witnesses.get("prompt_ids"))
    names = sorted(set(keys))
    if len(names) > MAX_KEYS:
        raise refused(directory, f"{len(names)} keys", f"at most {MAX_KEYS} (int8 indices)",
                      "split the directory")
    index = np.asarray([names.index(key) for key in keys], dtype=np.int8)
    block = {"kind": "episode", "keys": names, "path": f"{arrays}/{name}.npy",
             "ids_path": f"{arrays}/{name}.ids.npy",
             "counts": {key: keys.count(key) for key in names}}
    return block, {block["path"]: npy_bytes(index),
                   block["ids_path"]: npy_bytes(np.asarray(ids, dtype=np.int64))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--kept", required=True)
    parser.add_argument("--layouts", required=True, help="a v6 layout file")
    parser.add_argument("--embodiments", default="config/embodiments.yaml",
                        help="the key table; a repository path or one under the root")
    parser.add_argument("--arrays", required=True,
                        help="root-relative directory the per-episode arrays go to")
    parser.add_argument("--manifest", help="a built manifest whose cache witnesses the join")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    root = stage.resolve_root(args.root)
    inputs = {"kept": stage.under_root(root, args.kept),
              "layouts": stage.under_root(root, args.layouts)}
    table_path = Path(args.embodiments)
    if not table_path.is_absolute():
        table_path = REPO / table_path if (REPO / table_path).exists() \
            else stage.under_root(root, args.embodiments)
    document = read_embodiments(table_path)
    layouts = layout_table.load(inputs["layouts"])
    _, kept = stage.read_jsonl(inputs["kept"])
    grouped = kept_of(kept)
    if not grouped:
        raise ValueError("0 kept directories (n = 0)\nexpected at least one\n"
                         "check the kept file")
    entries: dict[str, dict[str, Any]] = {}
    prompt_cache = None
    if args.manifest:
        inputs["manifest"] = stage.under_root(root, args.manifest)
        manifest = stage.read_json(inputs["manifest"])
        entries = {entry["path"]: entry for entry in manifest["datasets"]}
        prompt_cache = manifest.get("prompt", {}).get("cache")
    out = stage.under_root(root, args.out)
    stage.refuse_write_into_input(out, [root / name for name in grouped])

    datasets: dict[str, Any] = {}
    claimed: dict[str, str] = {}
    artifacts: dict[str, str] = {}
    episodes: Counter = Counter()
    directories: Counter = Counter()
    witnessed = 0
    for directory in sorted(grouped):
        layout = layout_table.resolve(layouts, directory)
        name = layout_table.claim(claimed, layout)
        block = layout_table.policy_block(layout, "embodiment")
        kind = block.get("kind")
        records = grouped[directory]
        if kind == "constant":
            key = block.get("key")
            if key not in document["keys"]:
                raise ValueError(
                    f"{layout.name} names key {key!r}, which config/embodiments.yaml lacks\n"
                    f"expected one of {sorted(document['keys'])}\n"
                    f"add the key or fix the layout's embodiment block")
            embodiment: dict[str, Any] = {"kind": "constant", "key": key}
            episodes[key] += len(records)
            directories[key] += 1
        elif kind == "rh20t_meta":
            witnesses = cache_witnesses(root, entries.get(directory), prompt_cache)
            witnessed += bool(witnesses)
            embodiment, written = per_episode(root, directory, layout.name, records, document,
                                              witnesses, args.arrays)
            for path, payload in written.items():
                _, digest = write_npy(stage.under_root(root, path), payload)
                artifacts[path] = digest
            for key, count in embodiment["counts"].items():
                episodes[key] += count
                directories[key] += 1
        else:
            raise ValueError(
                f"{layout.name} embodiment kind is {kind!r}\n"
                f"expected one of {list(LAYOUT_KINDS)}\n"
                f"fix the layout's embodiment block")
        datasets[name] = {
            "directory": directory, "episodes": len(records), "embodiment": embodiment}
    per_episode_dirs = sum(1 for record in datasets.values()
                           if record["embodiment"]["kind"] == "episode")
    payload = {
        "provenance": stage.header("build_embodiment", root=root,
                                   inputs={**inputs, "embodiments": table_path}
                                   if table_path.is_relative_to(root) else inputs,
                                   tree=REPO, extra={"embodiments": str(args.embodiments),
                                                     "arrays": args.arrays}),
        "totals": {"directories": len(datasets), "per_episode": per_episode_dirs,
                   "constant": len(datasets) - per_episode_dirs,
                   "witnessed_by_cache": witnessed,
                   "episodes": sum(episodes.values())},
        "keys": {key: {"episodes": episodes[key], "directories": directories[key]}
                 for key in sorted(episodes)},
        "artifacts": artifacts,
        "datasets": datasets,
    }
    written = stage.write_json(out, payload)
    logger.info("%d directories: %d constant, %d per episode (%d of them joined against "
                "the cache)", len(datasets), len(datasets) - per_episode_dirs,
                per_episode_dirs, witnessed)
    for key in sorted(episodes):
        logger.info("key %s: %d episodes in %d directories", key, episodes[key],
                    directories[key])
    logger.info("%s %s", "wrote" if written else "unchanged", out)
    return 0


def write_npy(path: Path, payload: bytes) -> tuple[bool, str]:
    """Write an array's bytes unless the same bytes are there; return its sha256."""
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists() and path.read_bytes() == payload:
        return False, digest
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return True, digest


if __name__ == "__main__":
    sys.exit(main())
