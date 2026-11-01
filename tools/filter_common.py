"""Shared helpers of the data-filter tools: paths, record files, readers, layouts.

Only input/output and the joint-reading rule live here. Every filter command
writes to an explicit new path and never changes its inputs.
"""
from __future__ import annotations

import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Rule 1: implausible joint readings.
#
# A reading beyond JOINT_CEILING_RAD cannot be a joint angle of these arms, and
# one more than SPIKE_RAD from the median of the JOINT_WINDOW in-episode rows
# centred on it is an isolated spike. A sustained jump moves the median with
# it, so only a reading that leaves and returns within two rows is flagged; a
# genuine wrap is left for the channel filter (rule 2).
JOINT_CEILING_RAD = 2.0 * math.pi
SPIKE_RAD = math.pi
JOINT_WINDOW = 5


def finite(values: np.ndarray) -> np.ndarray:
    """An array as float32 with every non-finite reading made NaN."""
    array = np.asarray(values, dtype=np.float64)
    array = np.where(np.isfinite(array), array, np.nan)
    return array.astype(np.float32)


def implausible_joints(values: np.ndarray) -> np.ndarray:
    """[rows, joints] of one episode -> [rows, joints] True where a reading cannot be an angle.

    A finite reading beyond JOINT_CEILING_RAD, or more than SPIKE_RAD from
    the median of the finite readings among the JOINT_WINDOW rows centred on
    it (rows of this episode only; the ceiling's readings left out). NaN is
    never implausible: it was not recorded.
    """
    array = np.asarray(values, dtype=np.float64)
    finite_values = np.isfinite(array)
    beyond = finite_values & (np.abs(np.where(finite_values, array, 0.0)) > JOINT_CEILING_RAD)
    kept = np.where(finite_values & ~beyond, array, np.nan)
    if kept.shape[0] == 0:
        return beyond
    half = JOINT_WINDOW // 2
    padded = np.pad(kept, ((half, half), (0, 0)), constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, JOINT_WINDOW, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(windows, axis=-1)
    spike = np.isfinite(kept) & (np.abs(np.where(np.isfinite(kept), kept - median, 0.0))
                                 > SPIKE_RAD)
    return beyond | spike


# ---------------------------------------------------------------------------
# Paths and outputs.

def resolve_root(root: str | os.PathLike) -> Path:
    """The data root as an absolute path; it must exist."""
    resolved = Path(root).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"root {resolved} is not a directory")
    return resolved


def under_root(root: Path, path: str | os.PathLike) -> Path:
    """A root-relative or absolute path -> an absolute path under the root.

    Symbolic links are not followed, so a root staged as links into other
    trees keeps its own paths.
    """
    candidate = Path(path)
    joined = Path(os.path.normpath(candidate if candidate.is_absolute() else root / candidate))
    if not joined.is_relative_to(root):
        raise ValueError(f"path {joined} is outside the root {root}")
    return joined


def relative(root: Path, path: Path) -> str:
    """An absolute path under the root -> the root-relative string records use."""
    return str(Path(os.path.normpath(path)).relative_to(root))


def cli_path(path: str | os.PathLike) -> Path:
    """A file named on the command line (relative to the working directory or absolute)."""
    return Path(path).expanduser().resolve()


def display(root: Path, path: Path) -> str:
    """How a header names an input: root-relative when under the root, else absolute."""
    normal = Path(os.path.normpath(path))
    return str(normal.relative_to(root)) if normal.is_relative_to(root) else str(normal)


def new_output(out: str | os.PathLike, inputs: Iterable[str | os.PathLike] = ()) -> Path:
    """Check an output path: it must not exist and must not lie inside an input directory.

    Filters write next to, never over, the data they read.
    """
    path = Path(out).expanduser()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"output {path} exists; pass a new path")
    target = path.resolve()
    for source in inputs:
        source = Path(source).resolve()
        if target == source or (source.is_dir() and target.is_relative_to(source)):
            raise ValueError(f"output {path} is inside input {source}; write elsewhere")
    return path


def header(tool: str, *, inputs: Iterable[str], settings: dict[str, Any] | None = None
           ) -> dict[str, Any]:
    """The first line of a record file: the tool, its inputs and its settings."""
    record: dict[str, Any] = {"tool": tool, "inputs": sorted(str(name) for name in inputs)}
    if settings:
        record["settings"] = settings
    return record


def dumps_line(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"


def dumps_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, indent=1, ensure_ascii=True) + "\n"


def write_jsonl(path: Path, head: dict[str, Any], records: Sequence[Any]) -> None:
    """Write a new JSONL file: the provenance header, then one record per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(dumps_line({"provenance": head}))
        for record in records:
            stream.write(dumps_line(record))


def write_json(path: Path, payload: Any) -> None:
    """Write a new JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(dumps_json(payload))


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """A JSONL record file -> its provenance header (or None) and its records."""
    head = None
    records = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            if number == 0 and isinstance(record, dict) and set(record) == {"provenance"}:
                head = record["provenance"]
                continue
            records.append(record)
    return head, records


# ---------------------------------------------------------------------------
# Array readers.

def matrix(table: Any, column: str) -> np.ndarray:
    """An Arrow column of fixed-width lists (or scalars) as float64 [rows, width]."""
    chunked = table.column(column).combine_chunks()
    rows = len(chunked)
    if hasattr(chunked.type, "value_type"):
        flat = np.asarray(chunked.flatten().to_numpy(zero_copy_only=False), dtype=np.float64)
        return flat.reshape(rows, -1) if rows else flat.reshape(0, 1)
    return np.asarray(chunked.to_numpy(zero_copy_only=False), dtype=np.float64).reshape(rows, 1)


def read_table(path: Path, columns: list[str]) -> Any:
    """Named columns of an Arrow IPC/Feather or Parquet file."""
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path, columns=columns)
    from pyarrow import feather
    return feather.read_table(path, columns=columns, memory_map=True)


def read_values(path: Path, column: str | None) -> np.ndarray:
    """A [rows, width] float64 array from .npy, or from one column of an Arrow/Parquet file."""
    if path.suffix == ".npy":
        array = np.load(path, allow_pickle=False).astype(np.float64)
        return array.reshape(len(array), -1)
    if column is None:
        raise ValueError(f"{path} is a table; name the column with --column")
    return matrix(read_table(path, [column]), column)


def starts_from_column(episodes: np.ndarray) -> np.ndarray:
    """Episode start offsets [episodes + 1] from a per-row episode column.

    Rows of one episode must be contiguous; an episode that reappears later is
    refused rather than merged.
    """
    episodes = np.asarray(episodes).reshape(-1)
    if episodes.size == 0:
        return np.zeros(1, dtype=np.int64)
    change = np.flatnonzero(episodes[1:] != episodes[:-1]) + 1
    starts = np.concatenate([[0], change, [episodes.size]]).astype(np.int64)
    firsts = episodes[starts[:-1]]
    if len(set(firsts.tolist())) != len(firsts):
        raise ValueError("episode rows are not contiguous; sort rows by episode first")
    return starts


def check_starts(starts: np.ndarray, rows: int) -> np.ndarray:
    """Episode start offsets: from 0 to rows, non-decreasing."""
    starts = np.asarray(starts, dtype=np.int64).reshape(-1)
    if (starts.size < 1 or starts[0] != 0 or starts[-1] != rows
            or (np.diff(starts) < 0).any()):
        raise ValueError(f"episode starts must run from 0 to {rows} without decreasing")
    return starts


# ---------------------------------------------------------------------------
# Episode directories (per-episode parquet and video, meta/info.json,
# meta/episodes.jsonl).

def read_dir_list(root: Path, listed: list[str], list_file: Path | None) -> list[Path]:
    """The directories to read, from --dir and a --dirs-file (one per line, # comments)."""
    names = list(listed)
    if list_file is not None:
        names.extend(
            line.strip() for line in list_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    if not names:
        raise ValueError("no directories given; pass --dir or --dirs-file")
    return [under_root(root, name) for name in names]


def episode_files(directory: Path, subdir: str, suffix: str) -> dict[int, Path]:
    """Every episode file under one subtree, keyed by the index in its name."""
    found: dict[int, Path] = {}
    base = directory / subdir
    if not base.is_dir():
        return found
    for path in sorted(base.rglob(f"episode_*{suffix}")):
        found[int(path.stem.split("_")[-1])] = path
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
    """The stored columns, their dtypes and shapes, as meta/info.json states them."""
    return {
        name: {"dtype": feature["dtype"], "shape": list(feature.get("shape") or [])}
        for name, feature in sorted(info["features"].items())
    }


def frame_shape(columns: dict[str, Any], key: str) -> tuple[int, int]:
    """A view column's stored frame size (height, width) from meta/info.json."""
    shape = columns[key]["shape"]
    if len(shape) < 2:
        raise ValueError(f"view {key} has shape {shape}; expected at least height and width")
    return int(shape[0]), int(shape[1])


def stat_size(path: Path) -> int | None:
    """The file's size, or None when the name resolves to nothing (a dangling link)."""
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return None


def directory_episodes(root: Path, directory: Path) -> list[dict[str, Any]]:
    """One episode directory -> one record per episode listed in meta/episodes.jsonl.

    rows is the listed length; a video of no bytes is recorded as no file, and
    a data file that resolves to nothing as none. Nothing is opened here.
    """
    info = json.loads((directory / "meta" / "info.json").read_text(encoding="utf-8"))
    columns = columns_of(info)
    video_keys = [name for name, spec in columns.items() if spec["dtype"] == "video"]
    parquets = episode_files(directory, "data", ".parquet")
    videos = video_files(directory, video_keys)
    name = relative(root, directory)
    records = []
    with open(directory / "meta" / "episodes.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            episode = json.loads(line)
            index = int(episode["episode_index"])
            data_file = parquets.get(index)
            data_size = stat_size(data_file) if data_file else None
            if data_file is not None and data_size is None:
                data_file = None
            record: dict[str, Any] = {
                "dataset": name,
                "episode": index,
                "rows": int(episode["length"]),
                "tasks": list(episode.get("tasks") or []),
                "data_file": relative(root, data_file) if data_file else None,
                "videos": {},
                "metadata": {key: value for key, value in sorted(episode.items())
                             if key not in ("episode_index", "length", "tasks")},
            }
            for key in video_keys:
                path = videos[key].get(index)
                size = stat_size(path) if path is not None else None
                record["videos"][key] = ({"path": relative(root, path), "bytes": size}
                                         if size else {"path": None, "bytes": 0})
            records.append(record)
    records.sort(key=lambda item: item["episode"])
    return records


# ---------------------------------------------------------------------------
# Layout files (config/layouts/*.json): what each directory's columns mean.
# Resolution is source defaults, then a per-dataset entry, then the name.

LAYOUT_SCHEMA_VERSION = 1
BLOCKS = ("views", "state", "action", "columns", "extrinsics", "metadata_gates",
          "frame_ref", "trajectory_weights", "prompt", "frequency", "embodiment",
          "gripper", "joints", "hand", "body", "stored")
V7_BLOCKS = ("joints", "hand", "body", "stored")
POLICY_BLOCKS = ("frequency", "embodiment", "gripper")
COLUMN_FIELDS = ("state", "action")
STORED_COLUMN_FIELDS = ("joints", "hand", "body")
NO_PROMPT = {"embodiment_clauses": [], "frame_clause": {"camera": "", "robot": ""}}
NORMALISED_SUFFIXES = ("_shardNNofNN",)


@dataclass(frozen=True)
class Layout:
    """One directory's column meaning, as the layout file states it."""

    directory: str
    name: str
    source: str
    views: tuple[dict[str, Any], ...]
    state: dict[str, Any] | None
    action: dict[str, Any]
    columns: dict[str, str]
    extrinsics: dict[str, Any]
    metadata_gates: tuple[dict[str, Any], ...]
    frame_ref: str | None
    prompt: dict[str, Any]
    trajectory_weights: str | None
    frequency: dict[str, Any] | None = None
    embodiment: dict[str, Any] | None = None
    gripper: dict[str, Any] | None = None
    joints: dict[str, Any] | None = None
    hand: dict[str, Any] | None = None
    body: dict[str, Any] | None = None
    stored: str | None = None
    v7: tuple[str, ...] = ()


def load_layouts(path: Path) -> dict[str, Any]:
    """Read a layout file of the supported schema version."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema_version") != LAYOUT_SCHEMA_VERSION:
        raise ValueError(f"layout schema_version is {document.get('schema_version')!r}; "
                         f"expected {LAYOUT_SCHEMA_VERSION}")
    return document


def _rule_for(document: dict[str, Any], directory: str) -> dict[str, Any]:
    matching = [rule for rule in document["assign"] if directory.startswith(rule["prefix"])]
    if not matching:
        raise KeyError(f"no layout rule matches {directory}")
    return max(matching, key=lambda rule: len(rule["prefix"]))


def name_of(directory: str, parts: int) -> str:
    """The canonical name of a directory: its last `parts` path components."""
    pieces = [piece for piece in directory.split("/") if piece]
    return "_".join(pieces[-max(1, parts):])


@lru_cache(maxsize=None)
def _shard_pattern(suffix: str) -> re.Pattern[str]:
    if suffix not in NORMALISED_SUFFIXES:
        raise ValueError(f"shard_suffix is {suffix!r}; expected one of {list(NORMALISED_SUFFIXES)}")
    return re.compile("(?:%s)$" % re.sub("N+", r"\\d+", re.escape(suffix)))


def dataset_key(directory: str, rule: dict[str, Any]) -> str:
    """A directory -> its datasets entry (shard suffix removed when the rule names one)."""
    suffix = rule.get("shard_suffix")
    if suffix is None:
        return directory
    head, separator, last = directory.rpartition("/")
    return head + separator + _shard_pattern(suffix).sub("", last, count=1)


def _override_for(document: dict[str, Any], directory: str, rule: dict[str, Any]) -> dict[str, Any]:
    key = dataset_key(directory, rule)
    reason = document.get("excluded", {}).get(key)
    if reason is not None:
        raise KeyError(f"{directory} resolves to the excluded dataset {key}: {reason}")
    override = document.get("datasets", {}).get(key, {})
    if not override and rule.get("entry_required"):
        raise KeyError(f"{directory} resolves to dataset {key}, which the layout does not state")
    return override


def resolve(document: dict[str, Any], directory: str) -> Layout:
    """A root-relative directory -> the layout that governs it."""
    rule = _rule_for(document, directory)
    source = rule["source"]
    if source not in document["sources"]:
        raise KeyError(f"layout rule for {directory} names unknown source {source!r}")
    merged: dict[str, Any] = dict(document["sources"][source])
    override = _override_for(document, directory, rule)
    for block in BLOCKS:
        if block in override:
            merged[block] = override[block]
    if "state" not in merged:
        raise KeyError(f"source {source!r} states no state block (write null if unknown)")
    missing = [field for field in COLUMN_FIELDS if field not in merged.get("columns", {})]
    if missing:
        raise KeyError(f"source {source!r} names no {missing[0]} column")
    return Layout(
        directory=directory,
        name=override.get("name") or name_of(directory, int(rule.get("name_parts", 1))),
        source=source,
        views=tuple(merged["views"]),
        state=merged["state"],
        action=merged["action"],
        columns={field: str(merged["columns"][field])
                 for field in (*COLUMN_FIELDS, *STORED_COLUMN_FIELDS)
                 if merged["columns"].get(field) is not None},
        extrinsics=merged.get("extrinsics", {}),
        metadata_gates=tuple(merged.get("metadata_gates", ())),
        frame_ref=merged.get("frame_ref"),
        prompt=merged.get("prompt", NO_PROMPT),
        trajectory_weights=override.get("trajectory_weights"),
        **{block: merged.get(block) for block in POLICY_BLOCKS},
        **{block: merged.get(block) for block in V7_BLOCKS},
        v7=tuple(block for block in V7_BLOCKS if block in merged),
    )


def view_extrinsics(layout: Layout, key: str) -> dict[str, Any]:
    """Where one camera's pose comes from, or kind none."""
    return dict(layout.extrinsics.get(key, {"kind": "none"}))
