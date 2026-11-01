"""The sidecar format, its stamp, the byte-checked writer and the one reader.

A sidecar is one parquet per stage-1 directory with one row per directory row,
episodes in ascending episode_index, each episode's rows in the order its own
parquet holds them. Its columns are episode_index and frame_index (copied from
the directory, so a reader can line the rows up), observation.joints [A x 7],
observation.hand [A x 12], observation.body [8] and, for a family whose joint
target is a command, action.joints [A x 7]. Every value is float32 and NaN
where the source recorded nothing; 0 is never written for a missing reading.

The key-value metadata under METADATA_KEY is the stamp: the format and ingest
layout, this tool's version, the v7 record stamps, the row count, the
episodes and their rows with the digest of their boundaries, the digest of the
directory's meta tree and the digest of every source value the rows were read
from. read_sidecar refuses a file whose stamp is not this one.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import stage

from neo_vla.constants import (
    ACTION_STAMP,
    ARM_JOINTS,
    BODY_DIM,
    BODY_STAMP,
    COLUMN_BODY,
    COLUMN_HAND,
    COLUMN_JOINTS,
    COLUMN_JOINTS_COMMAND,
    HAND_JOINTS,
    INGEST_LAYOUT,
    OLD_PREFIX_LAYOUTS,
    SIDECAR_DIR,
    STAGE1_CORPUS,
    STATE_STAMP,
    refusal_prefix,
)

# What this file kind is called and which writer wrote it. A reader compares
# both with its own, so a sidecar from another layout or an older writer is
# refused by name rather than read as one with nothing recorded.
FORMAT = "neo_vla_sidecar_v1"
TOOL = "sidecars"
TOOL_VERSION = "2.0.0"
METADATA_KEY = b"neo_vla_sidecar"

EPISODE_COLUMN = "episode_index"
FRAME_COLUMN = "frame_index"

# The observation columns every sidecar carries, and the command column only a
# family whose joint target is its command (DROID) carries.
OBSERVATION_COLUMNS = (COLUMN_JOINTS, COLUMN_HAND, COLUMN_BODY)
COMMAND_COLUMNS = (COLUMN_JOINTS_COMMAND,)

# Rows per row group. A reader of one episode reads a few groups at most, and
# zstd sees enough of a column to fold the long NaN runs to almost nothing.
ROW_GROUP_ROWS = 1 << 18
COMPRESSION = "zstd"

RECEIPTS = "_receipts"

# A joint reading that cannot be an angle of the arm is NaN: not recorded,
# never 0 (RH20T flexiv joint.npy holds finite garbage such as 1.88e38,
# -334142.25 and 6.408, one joint of one row at a time while the rest of the
# row reads normally, and each one stepped past pi twice, so without this rule
# the wrap gate masked flexiv j0-j5 for the whole key). Two rules:
# beyond JOINT_CEILING_RAD, which no joint of a stage-1 robot reaches (the
# rh20t_api URDF limits of flexiv, franka, kuka and ur5 lie within -3.15 and
# 4.63 rad); or more than SPIKE_RAD, the wrap gate's step, from the median of
# the JOINT_WINDOW in-episode rows centred on it. A joint moving one way keeps
# its own value as the median and a sustained jump moves the median with it,
# so only a reading that leaves and comes back within two rows is a spike. The
# URDF limits themselves are not the rule: flexiv joint4 reads above
# robot_pvt3.urdf's upper 1.95477 rad on 3,244,070 of 6,419,566 readings.
JOINT_CEILING_RAD = 2.0 * math.pi
SPIKE_RAD = math.pi
JOINT_WINDOW = 5


class Misaligned(ValueError):
    """A source that does not line up with the directory it is written for."""


def default_out(root: Path) -> Path:
    """Where the stage-1 sidecars go under a corpus root."""
    return root / f"prep_{STAGE1_CORPUS}" / SIDECAR_DIR


def sidecar_path(out: Path, name: str) -> Path:
    """The sidecar of one manifest entry."""
    return out / f"{name}.parquet"


def column_width(name: str, arms: int) -> int:
    """How many float32 values one row of a column holds."""
    if name in (COLUMN_JOINTS, COLUMN_JOINTS_COMMAND):
        return arms * ARM_JOINTS
    if name == COLUMN_HAND:
        return arms * HAND_JOINTS
    if name == COLUMN_BODY:
        return BODY_DIM
    raise KeyError(
        f"{name} is not a sidecar column\n"
        f"expected one of {list(OBSERVATION_COLUMNS + COMMAND_COLUMNS)}\n"
        "name a v7 stored column from neo_vla.constants"
    )


@dataclass(frozen=True)
class Episode:
    """One episode of a directory: its index, its rows and its parquet."""

    index: int
    rows: int
    data_file: Path
    record: dict[str, Any] = field(repr=False)


def directory_episodes(directory: Path) -> list[Episode]:
    """Every episode meta/episodes.jsonl lists, in ascending episode_index.

    The rows are the episode parquet's own row count, which is what the cache
    stage copies; a listed length that disagrees with it is refused, because
    then one of the two is not the directory the sidecar would be aligned to.
    """
    info = json.loads((directory / "meta" / "info.json").read_text(encoding="utf-8"))
    template = info["data_path"]
    chunk = int(info.get("chunks_size", 1000))
    episodes = []
    with open(directory / "meta" / "episodes.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            index = int(record["episode_index"])
            data_file = directory / template.format(
                episode_chunk=index // chunk, episode_index=index)
            rows = pq.ParquetFile(data_file).metadata.num_rows
            if "length" in record and int(record["length"]) != rows:
                raise Misaligned(
                    f"{data_file} holds {rows} rows, meta/episodes.jsonl says "
                    f"{record['length']}\n"
                    "expected the listed length to be the parquet's row count\n"
                    "repair the directory before writing its sidecar"
                )
            episodes.append(Episode(index, rows, data_file, record))
    episodes.sort(key=lambda episode: episode.index)
    indices = [episode.index for episode in episodes]
    if len(set(indices)) != len(indices):
        raise Misaligned(
            f"{directory} lists an episode index twice\n"
            "expected each episode_index once in meta/episodes.jsonl\n"
            "repair the directory before writing its sidecar"
        )
    return episodes


def episode_starts_bytes(rows: Iterable[int]) -> bytes:
    """The episodes' row boundaries as the npy bytes build_cache writes."""
    counts = np.asarray(list(rows), dtype=np.int64)
    starts = np.zeros(counts.size + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])
    buffer = io.BytesIO()
    np.save(buffer, starts, allow_pickle=False)
    return buffer.getvalue()


def episode_starts_digest(rows: Iterable[int]) -> str:
    """The sha256 of the boundaries, as the stamp records them."""
    return hashlib.sha256(episode_starts_bytes(rows)).hexdigest()


class SourceDigest:
    """A running hash over every source value a sidecar was read from."""

    def __init__(self) -> None:
        self._hash = hashlib.sha256()

    def text(self, value: str) -> None:
        payload = value.encode("utf-8")
        self._hash.update(len(payload).to_bytes(8, "little"))
        self._hash.update(payload)

    def array(self, value: np.ndarray | None) -> None:
        if value is None:
            self.text("absent")
            return
        contiguous = np.ascontiguousarray(value)
        self.text(f"{contiguous.dtype.str}{list(contiguous.shape)}")
        self._hash.update(contiguous.tobytes())

    def hexdigest(self) -> str:
        return self._hash.hexdigest()


def empty_columns(rows: int, arms: int, command: bool) -> dict[str, np.ndarray]:
    """The data columns of a sidecar, every value NaN until a source fills it."""
    names = OBSERVATION_COLUMNS + (COMMAND_COLUMNS if command else ())
    return {name: np.full((rows, column_width(name, arms)), np.nan, dtype=np.float32)
            for name in names}


def finite(values: np.ndarray) -> np.ndarray:
    """A source array as float32 with every non-finite reading made NaN."""
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


def column_summary(values: np.ndarray) -> dict[str, int]:
    """How much of one column holds a reading: rows with any, and values."""
    present = np.isfinite(values)
    return {"finite_rows": int(present.any(axis=1).sum()),
            "finite_values": int(present.sum())}


def stamp_of(*, entry: dict[str, Any], family: str, arms: int,
             episodes: list[Episode], columns: dict[str, np.ndarray],
             described: dict[str, dict[str, Any]], source_digest: str,
             directory_digest: str, checks: dict[str, Any]) -> dict[str, Any]:
    """The metadata a sidecar carries, with every count taken from its columns."""
    rows = [episode.rows for episode in episodes]
    summary = {}
    for name, values in columns.items():
        if name not in described:
            raise KeyError(
                f"column {name} has no description\n"
                "expected a unit and a source for every column written\n"
                "describe it in the family's builder"
            )
        summary[name] = {"width": int(values.shape[1]), **described[name],
                         **column_summary(values)}
    return {
        "format": FORMAT,
        "ingest_layout": INGEST_LAYOUT,
        "tool": TOOL,
        "tool_version": TOOL_VERSION,
        "stamps": {"action": ACTION_STAMP, "state": STATE_STAMP, "body": BODY_STAMP},
        "entry": entry["name"],
        "directory": entry["path"],
        "source": entry.get("source"),
        "family": family,
        "arms": arms,
        "rows": int(sum(rows)),
        "episodes": [episode.index for episode in episodes],
        "episode_rows": rows,
        "episode_starts_sha256": episode_starts_digest(rows),
        "directory_meta_sha256": directory_digest,
        "source_sha256": source_digest,
        "columns": summary,
        "column_order": list(columns),
        "checks": checks,
    }


def table_of(episode_column: np.ndarray, frame_column: np.ndarray,
             columns: dict[str, np.ndarray], meta: dict[str, Any]) -> pa.Table:
    """The sidecar's arrow table with its stamp in the schema metadata."""
    arrays = {EPISODE_COLUMN: pa.array(episode_column, type=pa.int64()),
              FRAME_COLUMN: pa.array(frame_column, type=pa.int64())}
    for name, values in columns.items():
        flat = pa.array(np.ascontiguousarray(values, dtype=np.float32).reshape(-1),
                        type=pa.float32())
        arrays[name] = pa.FixedSizeListArray.from_arrays(flat, values.shape[1])
    table = pa.table(arrays)
    return table.replace_schema_metadata(
        {METADATA_KEY: json.dumps(meta, sort_keys=True).encode("utf-8")})


def write_sidecar(path: Path, table: pa.Table) -> dict[str, Any]:
    """Write one sidecar through a temporary, then read the bytes back.

    The filesystem under the corpus reports success and truncates at quota,
    so the file on disk is hashed and compared with the bytes produced before
    it replaces anything. Returns the byte count and the hash.
    """
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression=COMPRESSION, row_group_size=ROW_GROUP_ROWS)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    landed = stage.sha256_of(temporary)
    if landed != digest or temporary.stat().st_size != len(payload):
        temporary.unlink()
        raise OSError(
            f"{temporary} does not hold the {len(payload)} bytes written to it\n"
            "expected the file to read back as what was produced\n"
            "free space under the output root, then run this directory again"
        )
    os.replace(temporary, path)
    return {"bytes": len(payload), "sha256": digest}


def _refuse_stamp(path: Path, what: str, found: Any, expected: Any) -> None:
    raise ValueError(
        f"{refusal_prefix(22)} sidecar {path} states {what} {found!r}\n"
        f"expected {expected!r}, what this reader writes\n"
        "rebuild the sidecar with tools/sidecars/run.py"
    )


def check_stamp(path: Path, meta: dict[str, Any]) -> None:
    """Refuse a stamp that is not this format, layout, writer and record set."""
    if meta.get("format") != FORMAT:
        _refuse_stamp(path, "format", meta.get("format"), FORMAT)
    layout = meta.get("ingest_layout")
    if layout != INGEST_LAYOUT:
        old = " (a retired layout)" if layout in OLD_PREFIX_LAYOUTS else ""
        _refuse_stamp(path, f"ingest layout{old}", layout, INGEST_LAYOUT)
    if meta.get("tool_version") != TOOL_VERSION:
        _refuse_stamp(path, "tool version", meta.get("tool_version"), TOOL_VERSION)
    stamps = {"action": ACTION_STAMP, "state": STATE_STAMP, "body": BODY_STAMP}
    if meta.get("stamps") != stamps:
        _refuse_stamp(path, "record stamps", meta.get("stamps"), stamps)


def check_rows(path: Path, meta: dict[str, Any], table: pa.Table) -> None:
    """Refuse a sidecar whose rows are not the episodes its stamp lists."""
    rows = meta["episode_rows"]
    if len(rows) != len(meta["episodes"]):
        raise ValueError(
            f"{refusal_prefix(16)} {path} lists {len(meta['episodes'])} episodes "
            f"and {len(rows)} row counts\n"
            "expected one row count per episode\n"
            "rebuild the sidecar"
        )
    if episode_starts_digest(rows) != meta["episode_starts_sha256"]:
        raise ValueError(
            f"{refusal_prefix(16)} {path} episode boundaries do not hash to its stamp\n"
            f"expected sha256 {meta['episode_starts_sha256'][:12]}\n"
            "rebuild the sidecar"
        )
    if table.num_rows != meta["rows"] or sum(rows) != meta["rows"]:
        raise ValueError(
            f"{refusal_prefix(16)} {path} holds {table.num_rows} rows, its stamp "
            f"{meta['rows']} and its episodes {sum(rows)}\n"
            "expected the three counts to agree\n"
            "rebuild the sidecar; a short file is a truncated write"
        )
    expected = np.repeat(np.asarray(meta["episodes"], dtype=np.int64),
                         np.asarray(rows, dtype=np.int64))
    found = table.column(EPISODE_COLUMN).to_numpy()
    if not np.array_equal(found, expected):
        raise ValueError(
            f"{refusal_prefix(16)} {path} episode_index column does not follow "
            "its stamp's episodes and rows\n"
            "expected each listed episode's rows, in the listed order\n"
            "rebuild the sidecar"
        )
    order = meta.get("column_order", [])
    names = [EPISODE_COLUMN, FRAME_COLUMN, *order]
    if table.column_names != names or sorted(order) != sorted(meta["columns"]):
        raise ValueError(
            f"{refusal_prefix(22)} {path} holds columns {table.column_names}\n"
            f"expected {names}, the columns its stamp describes\n"
            "rebuild the sidecar"
        )


def read_stamp(path: Path) -> dict[str, Any]:
    """A sidecar's stamp, read from the parquet footer only."""
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    if METADATA_KEY not in metadata:
        raise ValueError(
            f"{refusal_prefix(22)} {path} carries no {METADATA_KEY.decode()} stamp\n"
            f"expected a sidecar written by tools/sidecars ({FORMAT})\n"
            "rebuild the sidecar with tools/sidecars/run.py"
        )
    return json.loads(metadata[METADATA_KEY])


def read_sidecar(path: Path) -> tuple[dict[str, Any], pa.Table]:
    """A checked sidecar: its stamp and its table, or a refusal."""
    meta = read_stamp(path)
    check_stamp(path, meta)
    table = pq.read_table(path)
    check_rows(path, meta, table)
    return meta, table


def kept_table(path: Path, meta: dict[str, Any], table: pa.Table,
               kept: list[tuple[int, int]]) -> pa.Table:
    """The sidecar rows of the kept episodes, in kept order.

    kept is (episode_index, rows) per kept episode, as the inventory states
    them. An episode the sidecar does not hold, or holds with other rows, is
    refused: merging it would shift every later row of the directory.
    """
    offsets = {}
    start = 0
    for index, rows in zip(meta["episodes"], meta["episode_rows"]):
        offsets[int(index)] = (start, int(rows))
        start += int(rows)
    pieces = []
    for index, rows in kept:
        if int(index) not in offsets:
            raise ValueError(
                f"{refusal_prefix(16)} {path} holds no episode {index}\n"
                "expected every kept episode of the directory\n"
                "rebuild the sidecar against the directory as it is now"
            )
        first, count = offsets[int(index)]
        if count != int(rows):
            raise ValueError(
                f"{refusal_prefix(16)} {path} episode {index} has {count} rows, "
                f"the inventory {rows}\n"
                "expected the directory's rows\n"
                "rebuild the sidecar against the directory as it is now"
            )
        pieces.append(table.slice(first, count))
    if not pieces:
        return table.slice(0, 0)
    return pa.concat_tables(pieces)


def receipt_path(out: Path, name: str) -> Path:
    return out / RECEIPTS / f"{name}.json"


def current(out: Path, name: str, commit: str | None = None) -> dict[str, Any] | None:
    """The receipt of a sidecar this version wrote, if it still holds.

    With commit, a sidecar written from another tree is not current either, so
    a code change between two runs rewrites what the first one wrote.
    """
    receipt = receipt_path(out, name)
    path = sidecar_path(out, name)
    if not receipt.exists() or not path.exists():
        return None
    try:
        record = json.loads(receipt.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if record.get("tool_version") != TOOL_VERSION or record.get("format") != FORMAT:
        return None
    if commit is not None and record.get("git_commit") != commit:
        return None
    if path.stat().st_size != record.get("bytes") or stage.sha256_of(path) != record.get("sha256"):
        return None
    return record


def write_receipt(out: Path, name: str, record: dict[str, Any]) -> None:
    path = receipt_path(out, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(stage.dumps_json(record), encoding="utf-8")
    os.replace(temporary, path)


def write_built(out: Path, entry: dict[str, Any], family: str, directory: Path,
                built: dict[str, Any], git_commit: str) -> dict[str, Any]:
    """Stamp, write, read back and receipt one directory's built columns.

    built is what a family's build returns: episodes, columns, episode and
    frame columns, descriptions, arms, the source digest and the checks. The
    sidecar is read back through read_sidecar before the receipt is written,
    so a receipt names only a sidecar a reader accepts.
    """
    meta = stamp_of(
        entry=entry, family=family, arms=built["arms"], episodes=built["episodes"],
        columns=built["columns"], described=built["described"],
        source_digest=built["source_digest"],
        directory_digest=stage.sha256_of_tree(directory / "meta"), checks=built["checks"])
    meta["git_commit"] = git_commit
    if "source_root" in built:
        meta["source_root"] = built["source_root"]
    table = table_of(built["episode_column"], built["frame_column"], built["columns"], meta)
    path = sidecar_path(out, entry["name"])
    written = write_sidecar(path, table)
    read_sidecar(path)
    receipt = {"format": FORMAT, "tool_version": TOOL_VERSION, "name": entry["name"],
               "family": family, "directory": entry["path"], "rows": meta["rows"],
               "episodes": len(meta["episodes"]), "columns": meta["columns"],
               "checks": meta["checks"], "git_commit": git_commit, **written}
    write_receipt(out, entry["name"], receipt)
    return receipt
