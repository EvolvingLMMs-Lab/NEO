"""Stage 1: what exists, as one record per episode, judged in no way.

Reads each directory's meta/info.json and meta/episodes.jsonl and lists the
parquet and video files beside them. A video of no bytes is listed as no video,
because a name for it would read downstream as a picture and open as an error.
Frame counts are the counts the metadata declares; whether the files agree with
that declaration is stage 2's question, which is why nothing here opens a
parquet.

Every path in the output is relative to the root, and the records are sorted by
directory and episode, so a rerun over unchanged data writes the same bytes.

Usage:
    python tools/scan_source.py --root /corpus --dirs-file dirs.txt \
        --out prep/inventory.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import stage

logger = logging.getLogger("scan_source")

REPO = Path(__file__).resolve().parent.parent

# LeRobot writes one chunk directory per this many episodes, and the count is
# in info.json rather than derivable from the tree.
CHUNK_KEY = "chunks_size"


def read_dir_list(root: Path, listed: list[str], list_file: Path | None) -> list[Path]:
    """The directories to scan, absolute and checked against the root."""
    names = list(listed)
    if list_file is not None:
        names.extend(
            line.strip() for line in list_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    if not names:
        raise ValueError(
            "no directories to scan\n"
            "expected at least one --dir or a non-empty --dirs-file\n"
            "pass the LeRobot directories that make up the corpus"
        )
    return [stage.under_root(root, name) for name in names]


def directory_inputs(
    root: Path, directories: list[Path], list_file: Path | None
) -> dict[str, Path]:
    """What a stage reading a directory list records as its inputs.

    The list itself is one of them. Removing a directory from it leaves every
    other recorded input hashing to what it did, so a stage that named only the
    directories would be declared current and the next stage would read an
    inventory still naming what was dropped.
    """
    inputs = {stage.relative(root, path / "meta"): path / "meta" for path in directories}
    if list_file is not None:
        inputs[stage.relative(root, list_file)] = list_file
    return inputs


def episode_files(directory: Path, subdir: str, suffix: str) -> dict[int, Path]:
    """Every episode file under one subtree, keyed by the index in its name."""
    found: dict[int, Path] = {}
    base = directory / subdir
    if not base.is_dir():
        return found
    for path in sorted(base.rglob(f"episode_*{suffix}")):
        index = int(path.stem.split("_")[-1])
        found[index] = path
    return found


def video_files(directory: Path, keys: list[str]) -> dict[str, dict[int, Path]]:
    """Per video column, the episode files stored for it."""
    per_key: dict[str, dict[int, Path]] = {}
    for key in keys:
        found: dict[int, Path] = {}
        for chunk in sorted((directory / "videos").glob("chunk-*")):
            column = chunk / key
            if not column.is_dir():
                continue
            for path in sorted(column.glob("episode_*.mp4")):
                found[int(path.stem.split("_")[-1])] = path
        per_key[key] = found
    return per_key


def columns_of(info: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The stored columns, their dtypes and their shapes, as info.json states."""
    return {
        name: {"dtype": feature["dtype"], "shape": list(feature.get("shape") or [])}
        for name, feature in sorted(info["features"].items())
    }


def stat_size(path: Path) -> int | None:
    """The file's size, or None when the name resolves to nothing.

    A dangling symlink is a name with no file behind it, which is an absence
    and not an error: video_files and episode_files glob names, and a glob
    returns a broken link like any other. Raising here ends the scan with a
    traceback, and this is the stage whose whole job is to state what is
    there -- so the one that meets the defect is the one that cannot report
    it. Anything that is not absence still raises: a permission or an IO
    error is a fact about the machine, not about the corpus.
    """
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return None


def video_record(root: Path, path: Path | None) -> dict[str, Any]:
    """What one episode holds for one camera: a file and its size, or neither.

    A file of no bytes is recorded as no file. Every reader of this inventory
    tells a null apart from a path, and none of them can tell a path apart
    from a path to nothing: an empty name would be copied into the frame index
    and answered as present by every check that asks only whether it resolves.
    Recording the absence is the statement that is true, and stage 2 still
    names the defect, because it measures the size rather than reading it here.
    """
    if path is None:
        return {"path": None, "bytes": 0}
    size = stat_size(path)
    if not size:
        return {"path": None, "bytes": 0}
    return {"path": stage.relative(root, path), "bytes": size}


def scan_directory(root: Path, directory: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """One LeRobot directory -> its episode records, and the videos of no bytes.

    The empty ones are handed back rather than only dropped, so that a source
    losing pictures is a number this stage states and a list a reader can go
    and look at.
    """
    info = json.loads((directory / "meta" / "info.json").read_text(encoding="utf-8"))
    columns = columns_of(info)
    video_keys = [name for name, spec in columns.items() if spec["dtype"] == "video"]
    parquets = episode_files(directory, "data", ".parquet")
    videos = video_files(directory, video_keys)
    relative_dir = stage.relative(root, directory)

    records = []
    empty: list[str] = []
    with open(directory / "meta" / "episodes.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            episode = json.loads(line)
            index = int(episode["episode_index"])
            data_file = parquets.get(index)
            data_size = stat_size(data_file) if data_file else None
            if data_file is not None and data_size is None:
                empty.append(stage.relative(root, data_file))
                data_file = None
            record: dict[str, Any] = {
                "dataset": relative_dir,
                "episode": index,
                "rows": int(episode["length"]),
                "tasks": list(episode.get("tasks") or []),
                "fps": info["fps"],
                "robot_type": info["robot_type"],
                "chunks_size": info[CHUNK_KEY],
                "columns": columns,
                "data_file": stage.relative(root, data_file) if data_file else None,
                "data_bytes": data_size or 0,
                "videos": {},
                "metadata": {
                    key: value for key, value in sorted(episode.items())
                    if key not in ("episode_index", "length", "tasks")
                },
            }
            for key in video_keys:
                path = videos[key].get(index)
                record["videos"][key] = video_record(root, path)
                if path is not None and record["videos"][key]["path"] is None:
                    empty.append(stage.relative(root, path))
            records.append(record)
    records.sort(key=lambda item: item["episode"])
    return records, sorted(empty)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, help="the one corpus root")
    parser.add_argument("--dir", action="append", default=[], help="a directory to scan")
    parser.add_argument("--dirs-file", help="a file of directories, one per line")
    parser.add_argument("--out", required=True, help="where inventory.jsonl goes")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    root = stage.resolve_root(args.root)
    dirs_file = stage.under_root(root, args.dirs_file) if args.dirs_file else None
    directories = read_dir_list(root, args.dir, dirs_file)
    out = stage.under_root(root, args.out)
    stage.refuse_write_into_input(out, directories)

    records: list[dict[str, Any]] = []
    empty: list[str] = []
    for directory in sorted(directories):
        found, none_held = scan_directory(root, directory)
        records.extend(found)
        empty.extend(none_held)
    records.sort(key=lambda item: (item["dataset"], item["episode"]))
    for name in empty:
        logger.warning("%s holds no bytes; recorded as no file for its camera", name)

    head = stage.header(
        "scan_source",
        root=root,
        inputs=directory_inputs(root, directories, dirs_file),
        tree=REPO,
        extra={"directories": len(directories), "empty_videos": empty},
    )
    written = stage.write_jsonl(out, head, records)
    logger.info(
        "%d directories, %d episodes, %d rows, %d videos of no bytes",
        len(directories), len(records),
        sum(record["rows"] for record in records), len(empty),
    )
    logger.info("%s %s", "wrote" if written else "unchanged", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
