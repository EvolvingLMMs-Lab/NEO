"""Stage 10: the readable form the manifest already named.

Six parts, each written where the manifest says it goes: the episode
boundaries, the kept episodes' numeric columns as one Arrow file per
directory (the v7 columns merged row-aligned from a stage-1 sidecar where the
manifest names one, and the writer's checked stamp in the schema metadata), a
frame index naming each kept episode's picture files, the per-episode camera
geometry, and the rendered prompts with the stamp beside them.

Pictures stay where they were recorded. --shard K/N writes one part of the
manifest; merging the shards' indexes gives the index one process would have
written.

--pose-qc names a qc_extrinsics census (run with --control and without
--sample). An episode whose stored pose that census failed is written with the
unknown pose and stays in the kept list; ok and inconclusive leave the pose as
build_extrinsics audited it. Without the flag the outputs are what they were.

Usage:
    python tools/build_cache.py --root /corpus --manifest manifests/action_pretrain.json \
        --kept metadata/C/kept.jsonl \
        --inventory metadata/C/inventory.jsonl \
        --out processed/C/index.json
"""
from __future__ import annotations

import argparse
import dataclasses
import io
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq
import stage
import torch
from corpus_stamps import (
    check_sidecar_stamp,
    ingest_stamp,
    read_sidecar_stamp,
    refuse_stamp,
    sidecar_rows,
)
from transformers import AutoTokenizer

from neo_vla.data import rays
from neo_vla.data.conditioning import is_rotation
from neo_vla.data.reader import (
    EPISODE_ID_COLUMN,
    FRAME_VALID_SUFFIX,
    INGEST_STAMP_KEY,
    SIDECAR_STAMP_FIELDS,
    SIDECAR_STAMP_KEY,
    VIEW_FRAME_SUFFIX,
)
from neo_vla.constants import prefix_layout_of, refusal_prefix
from neo_vla.data.prompt import (
    RenderContext,
    Stamp,
    Template,
    fit,
    check_mode_agrees,
    load_template,
    stamp_of,
)
from neo_vla.data.prompt import stamp as stamp_module
from neo_vla.data.conditioning import parse_frame_ref

logger = logging.getLogger("build_cache")

REPO = Path(__file__).resolve().parent.parent

PARTS = ("episode_starts", "arrow", "frames", "extrinsics", "intrinsics", "prompt")

# Column dtypes that hold pictures rather than numbers. They are left out of
# the arrow file and named in the frame index instead.
PICTURE_DTYPES = ("image", "video")

# The per-camera columns the loader reads for every camera of a directory or
# for none of them: which frame of its own stream a camera contributes to a
# row, and whether that frame is close enough in time to be a picture of it.
# A directory that keeps one of these for some cameras only would pair one
# camera's instant with another's, so it is refused while it is still a cache
# nobody has trained on rather than at the first epoch.
PER_VIEW_SUFFIXES = (VIEW_FRAME_SUFFIX, FRAME_VALID_SUFFIX)

# What the extrinsics artifact holds, per kept episode: one 3x3 the loader
# reads without turning it. A camera fixed to the room stores the rotation into
# the robot base frame, which is what conditioning.extrinsics_6 reads; a camera
# bolted to an arm stores the rotation into that arm's frame, which the loader
# composes with the arm's own pose. A source storing either one inverted is
# transposed here rather than in the loader, so the loader knows one convention.
BASE_FROM_REFERENCE = "base_from_camera"
CAMERA_FROM_BASE = "camera_from_base"
END_EFFECTOR_FROM_CAMERA = "end_effector_from_camera"
UNTURNED_FRAMES = (BASE_FROM_REFERENCE, END_EFFECTOR_FROM_CAMERA)

# Uncompressed, so two builds of the same rows are the same bytes.
ARROW_COMPRESSION = "uncompressed"

# The v7 columns a manifest entry's columns block may name (dataset
# COLUMN_FIELDS), and the column a sidecar and a directory both carry to line
# their rows up beside the episode index.
STORED_FIELDS = ("joints", "hand", "joints_command", "body", "body_command")
FRAME_INDEX_COLUMN = "frame_index"

# Row groups reach the cache as record batches unless the table is made
# contiguous before it is written, and a memory-mapped table costs about 4.5
# KiB of private memory per batch in every dataloader worker for as long as the
# run holds it. DROID writes eight rows to a row group, so one of its
# directories arrives as tens of thousands of batches. Combining them needs one
# contiguous copy of the table, and this bounds that copy: the largest
# directory of the present corpus is 1.6 GiB, so all of them are combined.
COMBINE_CHUNKS_LIMIT = 2 * 1024 ** 3

# What a shard's own index is called. Each shard writes its own file, so two of
# them cannot race on one index or on the temporary it is replaced through.
SHARD_INDEX_NAME = "index_{index}_of_{count}.json"

# How the manifest's entries are divided between shards. rows puts the largest
# directory into the lightest shard, which keeps the shards within a fraction
# of a percent of each other; index is the round robin the fleet script splits
# its directory list with, kept for comparing one against the other.
SHARD_KEYS = ("rows", "index")

# The projection QC a --pose-qc file must come from (tools/qc_extrinsics.py),
# the verdicts it writes, and the ones that make a stored pose unknown. Only a
# failed score is evidence that the pose is wrong: inconclusive means nothing
# was measured, so that pose is kept as build_extrinsics audited it and is
# never counted as a pass. The threshold is the only one a gating file may
# state, and the header keys are the ones its settings must hold: a census
# (sample 0, every episode read scored), a wrong-pose control run (a win rate
# and control verdicts), and verdict counts the records must reproduce.
# No source-level acceptance number (win rate, fail share) is stated anywhere;
# none is applied here and a source is not a projection-QC pass on this file.
POSE_QC_TOOL = "qc_extrinsics"
POSE_QC_VERDICTS = ("ok", "fail", "inconclusive")
POSE_QC_UNKNOWN = ("fail",)
POSE_QC_THRESHOLD = 2.5
POSE_QC_SETTINGS = ("sample", "threshold", "episodes", "episodes_read", "verdicts",
                    "control_verdicts", "win_rate")
POSE_QC_RECORD_KEYS = ("dataset", "view", "episode", "verdict", "score")
POSE_QC_KINDS = (("kind", "episode"), ("source", "stage"))


def write_bytes_if_changed(path: Path, payload: bytes) -> bool:
    """Write a binary artifact only when its bytes differ from what is there."""
    if path.exists() and path.read_bytes() == payload:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return True


def write_artifact(path: Path, payload: bytes) -> bool:
    """Write one artifact, read it back, and return whether it moved.

    The filesystem under this pipeline reports success and truncates at quota,
    so what landed is read back and compared with the bytes the stage produced.
    """
    changed = write_bytes_if_changed(path, payload)
    if path.read_bytes() != payload:
        raise OSError(
            f"{path} does not hold the {len(payload)} bytes written to it\n"
            f"expected the file to read back as what the stage produced\n"
            f"free space under the root, then run this stage again"
        )
    return changed


def parse_shard(text: str) -> tuple[int, int]:
    """A K/N argument -> which shard this process is and how many there are."""
    index, _, count = text.partition("/")
    if not (index.isdigit() and count.isdigit()) or not 0 <= int(index) < int(count):
        raise ValueError(
            f"--shard is {text!r}\n"
            f"expected K/N, with N shards and K below N\n"
            f"pass --shard 0/8 for the first of eight shards"
        )
    return int(index), int(count)


def shard_assignment(entries: list[dict[str, Any]], count: int, key: str) -> list[int]:
    """Which shard writes each manifest entry, in the manifest's own order.

    Computed from the manifest alone, so a shard and the merge that checks it
    reach the same answer without either being told. Ordering by rows and
    filling the lightest shard first is what keeps the shards even: the arrow
    part is almost all of the bytes and one directory can hold a fortieth of
    the rows, which a round robin cannot spread.
    """
    if key == "index":
        return [position % count for position in range(len(entries))]
    loads = [0] * count
    assigned = [0] * len(entries)
    heaviest = sorted(
        range(len(entries)),
        key=lambda position: (-entries[position]["rows"], entries[position]["path"]),
    )
    for position in heaviest:
        chosen = min(range(count), key=lambda shard: (loads[shard], shard))
        assigned[position] = chosen
        loads[chosen] += entries[position]["rows"]
    return assigned


def planned_artifacts(
    root: Path, manifest: dict[str, Any], parts: list[str]
) -> list[list[str]]:
    """The artifacts each manifest entry writes, in the manifest's own order.

    The conditions the writing loop applies, stated once, so a shard can say at
    the end that it wrote what it was given and the merge can say that the
    shards together wrote what the manifest names. A part this build does not
    write names nothing here either.
    """
    planned = []
    for entry in manifest["datasets"]:
        names: list[str] = []
        if "episode_starts" in parts:
            names.append(entry["episode_starts"])
        if "arrow" in parts:
            names.append(entry["cache"]["arrow"])
        if "frames" in parts:
            names.append(entry["cache"]["frames"])
        if "extrinsics" in parts:
            names += [view["extrinsics"]["path"] for view in entry["views"]
                      if view["extrinsics"]["kind"] != "none"]
        if "intrinsics" in parts:
            names += [view["intrinsics"]["path"] for view in entry["views"]
                      if view["intrinsics"]["kind"] == "episode"]
        if "prompt" in parts:
            names.append(prompt_artifact(root, manifest, f"{entry['name']}.jsonl"))
        planned.append(names)
    return planned


def prompt_artifact(root: Path, manifest: dict[str, Any], name: str) -> str:
    """One file of the prompt cache, named the way the index records it."""
    return stage.relative(
        root, stage.under_root(root, Path(manifest["prompt"]["cache"]) / name))


def current_shard(
    out: Path, root: Path, settings: dict[str, Any], block: dict[str, Any]
) -> bool:
    """Whether a shard's own index still describes what it would write now.

    The index is written last, so a shard killed part way through has none and
    runs again. What it recorded is checked rather than trusted: the version,
    the inputs it read, the settings and the entries it ran with, and every
    artifact it claims.
    """
    if not out.is_file():
        return False
    payload = stage.read_json(out)
    head = payload.get("provenance", {})
    if head.get("tool") != "build_cache" or head.get("tool_version") != stage.TOOL_VERSION:
        return False
    if head.get("settings") != settings or payload.get("shard") != block:
        return False
    if stage.check_inputs_match(head, root):
        return False
    return not stage.check_artifacts_match(payload, root)


def episode_starts(rows: list[int]) -> bytes:
    """The kept episodes' row boundaries, as the npy the manifest names."""
    starts = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(np.asarray(rows, dtype=np.int64), out=starts[1:])
    buffer = io.BytesIO()
    np.save(buffer, starts, allow_pickle=False)
    return buffer.getvalue()


def check_per_view_kept(kept: list[str], views: list[str], where: str) -> None:
    """Refuse a column set that states a per-camera column for some cameras.

    Everything a source parquet holds that is not a picture is copied, so the
    two ways to reach this are a drop list naming one camera's column and a
    directory whose episodes were not all written by the same run. Either one
    reaches the loader as a directory reading one camera at the window's own
    step and the rest at their own, which shows up in no count.
    """
    for suffix in PER_VIEW_SUFFIXES:
        stated = [key for key in views if f"{key}{suffix}" in kept]
        if stated and len(stated) != len(views):
            missing = sorted(set(views) - set(stated))
            raise ValueError(
                f"the cache for {where} would state {suffix!r} for "
                f"{len(stated)} of its {len(views)} cameras\n"
                f"expected the column for every camera or for none\n"
                f"drop it for {missing} as well, or rebuild the directory whole"
            )


def stored_columns(entry: dict[str, Any]) -> list[str]:
    """The v7 columns a manifest entry's layouts read, by the columns block."""
    columns = entry["columns"]
    return [columns[field] for field in STORED_FIELDS if columns.get(field) is not None]


def stored_parts(root: Path, entry: dict[str, Any],
                 episodes: list[dict[str, Any]]) -> tuple[pa.Table | None, dict[bytes, bytes]]:
    """What a directory's v7 columns add to its arrow file: the sidecar rows and the stamp.

    An entry reading no v7 column adds nothing. An ingest output's columns are
    in its own parquet; its meta/info.json stamp is checked and copied
    under INGEST_STAMP_KEY. A sidecar entry's stamp is checked, every kept
    episode must be in it with its rows (R16), and its rows of the kept
    episodes, in kept order, are returned with the episode and frame index to
    line up against the directory.
    """
    names = stored_columns(entry)
    if not names:
        return None, {}
    sidecar = entry.get("sidecar")
    if sidecar is None:
        info = stage.read_json(root / entry["path"] / "meta" / "info.json")
        stamp = ingest_stamp(info, entry["name"])
        return None, {INGEST_STAMP_KEY: json.dumps(stamp, sort_keys=True).encode("utf-8")}
    path = stage.under_root(root, sidecar["path"])
    where = str(sidecar["path"])
    if not path.is_file():
        raise refuse_stamp(entry["name"], f"names sidecar {where} and it is not there",
                           "the sidecar the manifest names",
                           "build it with tools/sidecars/run.py, then rebuild the manifest")
    stamp = read_sidecar_stamp(path)
    check_sidecar_stamp(stamp, entry["name"], names, where)
    rows = sidecar_rows(stamp, where)
    offsets, start = {}, 0
    for episode in stamp["episodes"]:
        offsets[int(episode)] = start
        start += rows[int(episode)]
    table = pq.read_table(path, columns=[EPISODE_ID_COLUMN, FRAME_INDEX_COLUMN, *names])
    if table.num_rows != start:
        raise ValueError(
            f"{refusal_prefix(16)} {where} holds {table.num_rows} rows and its stamp lists "
            f"{start}\nexpected the stamp's rows (a short file is a truncated write)\n"
            f"rebuild the sidecar")
    pieces = []
    for episode in episodes:
        index, length = int(episode["episode"]), int(episode["rows"])
        if rows.get(index) != length:
            held = rows.get(index)
            raise ValueError(
                f"{refusal_prefix(16)} {entry['name']}: kept episode {index} has {length} rows "
                f"and its sidecar {where} holds "
                f"{'no such episode' if held is None else f'{held} rows of it'}\n"
                f"expected every kept episode with the directory's rows\n"
                f"rebuild the sidecar against the directory as it is now")
        pieces.append(table.slice(offsets[index], length))
    kept = pa.concat_tables(pieces) if pieces else table.slice(0, 0)
    # Only the writer's own fields are carried into the cache; anything else an
    # older writer stated is not.
    stamp = {key: value for key, value in stamp.items() if key in SIDECAR_STAMP_FIELDS}
    return kept, {SIDECAR_STAMP_KEY: json.dumps(stamp, sort_keys=True).encode("utf-8")}


def merge_stored(combined: pa.Table, stored: pa.Table | None, names: list[str],
                 where: str) -> pa.Table:
    """The directory's kept rows with the sidecar's v7 columns beside them, or its own checked.

    A sidecar's rows must be the directory's, row for row: the same episode
    index and frame index; a directory that already stores a v7 column a
    sidecar would add is refused, since two writers would then claim it. An
    ingest output must hold every v7 column the manifest names.
    """
    if stored is None:
        missing = [name for name in names if name not in combined.schema.names]
        if missing:
            raise refuse_stamp(where, f"holds no {missing}, which its manifest entry reads",
                               "every v7 column the layouts read, under the v7 ingest stamp",
                               "re-emit the directory with the v7 writer")
        return combined
    if stored.num_rows != combined.num_rows:
        raise ValueError(
            f"{refusal_prefix(16)} {where}: the sidecar gives {stored.num_rows} rows for "
            f"{combined.num_rows} kept rows\nexpected one sidecar row per directory row\n"
            f"rebuild the sidecar against the directory as it is now")
    for column in (EPISODE_ID_COLUMN, FRAME_INDEX_COLUMN):
        if column not in combined.schema.names:
            continue
        if not stored.column(column).equals(combined.column(column).cast(pa.int64())):
            raise ValueError(
                f"{refusal_prefix(16)} {where}: the sidecar's {column} is not the directory's "
                f"row for row\nexpected the sidecar's rows in the directory's order\n"
                f"rebuild the sidecar against the directory as it is now")
    clash = [name for name in names if name in combined.schema.names]
    if clash:
        raise refuse_stamp(where, f"stores {clash} itself and names a sidecar for them",
                           "one writer per v7 column", "state stored ingest, or drop the columns")
    for name in names:
        combined = combined.append_column(name, stored.column(name))
    return combined


def arrow_table(root: Path, files: list[str], drop: list[str],
                views: list[str], where: str, *, stored: pa.Table | None = None,
                names: tuple[str, ...] = (), metadata: dict[bytes, bytes] | None = None) -> bytes:
    """The kept episodes' numeric columns, in kept order, as one Arrow file.

    The picture columns are among the dropped ones: copying them here would
    duplicate the corpus, and the frame index already names the file each one
    is stored in. Everything else a source parquet holds is copied as it was
    written, which is how the per-camera frame index and its flag reach the
    cache without this stage knowing either by name. The v7 columns the
    manifest names come from the parquet (ingest) or the sidecar rows given
    (stored_parts), and metadata is the checked stamp the loader reads.

    The rows are written contiguously rather than in the row groups they were
    read in, which is what the loader pays for: see COMBINE_CHUNKS_LIMIT.
    """
    tables = []
    for name in files:
        table = pq.read_table(root / name)
        keep = [column for column in table.schema.names if column not in drop]
        check_per_view_kept(keep, views, where)
        tables.append(table.select(keep))
    combined = pa.concat_tables(tables)
    if names:
        combined = merge_stored(combined, stored, list(names), where)
    if metadata:
        combined = combined.replace_schema_metadata(
            {**(combined.schema.metadata or {}), **metadata})
    if combined.nbytes <= COMBINE_CHUNKS_LIMIT:
        combined = combined.combine_chunks()
        # The row groups the copy was made from are not read again. Releasing
        # them here holds this stage's peak where it was before the copy.
        tables.clear()
    else:
        logger.info("%s is %.2f GiB and keeps the chunking it was read in",
                    where, combined.nbytes / 1024 ** 3)
    buffer = io.BytesIO()
    feather.write_feather(combined, buffer, compression=ARROW_COMPRESSION)
    return buffer.getvalue()


def picture_columns(episodes: list[dict[str, Any]]) -> dict[str, str]:
    """Each camera column of a directory and the dtype its frames are stored as."""
    if not episodes:
        return {}
    return {
        name: spec["dtype"]
        for name, spec in sorted(episodes[0]["columns"].items())
        if spec["dtype"] in PICTURE_DTYPES
    }


def readable_bytes(path: Path) -> int | None:
    """How many bytes one frame file holds, or None when there is no file.

    One call rather than an existence test and a size: the two would disagree
    for a file removed between them, and the size is what says whether anything
    can be decoded. An empty file is what a re-encode leaves behind when it
    could not read its source, and it passes every test that only asks whether
    a name resolves.
    """
    try:
        return path.stat().st_size
    except OSError:
        return None


def absent_frame_files(root: Path, views: dict[str, list[Any]], where: str) -> int:
    """How many episode cameras have no frame file, refusing a name with none.

    The two cases are told apart on purpose. A camera the scan found no file
    for is recorded as absent, which the loader and the verify stage both read,
    and is counted here so the build can say how many there were. A name the
    scan did hand over that nothing can be read at is refused instead, naming
    the camera and the kept position: writing it puts a path no decoder can
    open into the cache, which the verify stage then has to find again over
    every name in the corpus. A file of no bytes is refused along with one
    that is gone, because the two reach a reader the same way.
    """
    absent = 0
    checked: set[str] = set()
    for key, files in sorted(views.items()):
        for position, name in enumerate(files):
            if name is None:
                absent += 1
                continue
            if name in checked:
                continue
            checked.add(name)
            held = readable_bytes(stage.under_root(root, name))
            if held is None or held == 0:
                raise FileNotFoundError(
                    f"{where} names {name} for camera {key} at kept position "
                    f"{position} and it holds "
                    f"{'no file' if held is None else 'no bytes'}\n"
                    f"expected every frame file the scan named to be a file a "
                    f"decoder can open\n"
                    f"encode that episode again, or scan the source so the cache "
                    f"records the camera as absent"
                )
    return absent


def frame_index(root: Path, entry: dict[str, Any],
                episodes: list[dict[str, Any]]) -> tuple[dict[str, Any], int]:
    """Per camera, the file holding each kept episode's pictures, in kept order.

    One kind for the whole directory. A stream and an image column are read
    differently but are addressed the same way, by kept position, so a mixed
    directory would need two kinds in one index and is refused here instead.

    Returned with the count of episode cameras the scan found no file for, so
    that a directory losing pictures is a number the build states rather than
    something only the loader would ever meet.
    """
    stored = picture_columns(episodes)
    wanted = [view["key"] for view in entry["views"]]
    missing = [key for key in wanted if key not in stored]
    if missing:
        raise KeyError(
            f"{entry['name']} names camera {missing[0]} and the directory stores no "
            f"such column\n"
            f"expected one of {sorted(stored)}\n"
            f"fix the layout's view keys"
        )
    kinds = {stored[key] for key in wanted}
    if len(kinds) != 1:
        raise ValueError(
            f"{entry['name']} stores its cameras as {sorted(kinds)}\n"
            f"expected one storage kind for the whole directory\n"
            f"split the directory, or convert the odd camera"
        )
    dtype = kinds.pop()
    if dtype == "video":
        views = {key: [episode["videos"][key]["path"] for episode in episodes]
                 for key in wanted}
        kind = "mp4"
    else:
        views = {key: [episode["data_file"] for episode in episodes]
                 for key in wanted}
        kind = "parquet"
    return {"kind": kind, "views": views}, absent_frame_files(
        root, views, entry["name"])


def rotation_from(flat: list[float] | None, frame: str) -> np.ndarray:
    """One stored rotation, the way round the loader reads it, or the marker."""
    if flat is None:
        return np.full((3, 3), np.nan)
    values = np.asarray(flat, dtype=np.float32)
    matrix = values.reshape(4, 4)[:3, :3] if values.size == 16 else values.reshape(3, 3)
    if frame == CAMERA_FROM_BASE:
        return matrix.T
    if frame in UNTURNED_FRAMES:
        return matrix
    raise ValueError(
        f"extrinsics frame is {frame!r}\n"
        f"expected {CAMERA_FROM_BASE!r} or one of {list(UNTURNED_FRAMES)}\n"
        f"state which way round the stored rotation is in the layout file"
    )


def stage_extrinsics(path: Path) -> dict[tuple[str, str], dict[int, dict[str, Any]]]:
    """One extrinsics stage output, keyed by directory and camera, then episode.

    Read once for the whole run, like the intrinsics beside it: the file covers
    every directory of its source and reading it per directory would open a
    hundred and fifty thousand records once per entry. The camera is part of
    the key because a source publishes a pose for several of its cameras, and
    two of them under one key would be settled by file order.
    """
    _, records = stage.read_jsonl(path)
    grouped: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    for record in records:
        key = (record["dataset"], record["view"])
        grouped.setdefault(key, {})[int(record["episode"])] = record
    return grouped


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """A json object whose keys are each stated once; json.loads keeps the last."""
    held: dict[str, Any] = {}
    for key, value in pairs:
        if key in held:
            raise ValueError(f"key {key!r} is stated twice in one object")
        held[key] = value
    return held


def _is_number(value: Any) -> bool:
    """A json number, not a bool (True == 1 in Python, and a flag is not a count)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _refuse_pose_qc(where: str, problem: str, expected: str) -> ValueError:
    return ValueError(
        f"--pose-qc {where}: {problem}\n"
        f"expected {expected}\n"
        f"re-run qc_extrinsics --control without --sample, or drop --pose-qc"
    )


def check_pose_qc_header(where: str, head: Any, verdicts: Counter, records: int) -> None:
    """Refuse a qc_extrinsics header that is not a gating census with a control.

    stage.header puts a stage's extra keys under settings. Every key named in
    POSE_QC_SETTINGS has to be there: an absent one is refused rather than
    defaulted, so a file from another run shape cannot pass for a census.
    """
    if not isinstance(head, dict) or head.get("tool") != POSE_QC_TOOL:
        raise _refuse_pose_qc(where, f"the header names tool "
                              f"{head.get('tool') if isinstance(head, dict) else None!r}",
                              f"a {POSE_QC_TOOL} output")
    if head.get("tool_version") != stage.TOOL_VERSION:
        raise _refuse_pose_qc(where, f"tool_version is {head.get('tool_version')!r}",
                              f"{stage.TOOL_VERSION!r}, the version this stage reads")
    settings = head.get("settings")
    if not isinstance(settings, dict):
        raise _refuse_pose_qc(where, "the header states no settings",
                              f"settings holding {list(POSE_QC_SETTINGS)}")
    missing = [key for key in POSE_QC_SETTINGS if key not in settings]
    if missing:
        raise _refuse_pose_qc(where, f"the header settings hold no {missing}",
                              f"settings holding {list(POSE_QC_SETTINGS)}")
    if type(settings["sample"]) is not int or settings["sample"] != 0:
        raise _refuse_pose_qc(where, f"sample is {settings['sample']!r}",
                              "sample 0: a census, not a --sample run (evidence only)")
    threshold = settings["threshold"]
    if not _is_number(threshold) or threshold != POSE_QC_THRESHOLD:
        raise _refuse_pose_qc(where, f"threshold is {threshold!r}",
                              f"the figure's threshold {POSE_QC_THRESHOLD}")
    counted = (settings["episodes"], settings["episodes_read"])
    if any(type(value) is not int for value in counted) or counted != (records, records):
        raise _refuse_pose_qc(where, f"the header states {counted[0]!r} episodes scored of "
                                     f"{counted[1]!r} read and the file holds {records}",
                              "every episode read scored and written, one record each")
    win = settings["win_rate"]
    controls = settings["control_verdicts"]
    if (not _is_number(win) or not np.isfinite(win) or not 0.0 <= win <= 1.0
            or not isinstance(controls, dict) or not controls):
        raise _refuse_pose_qc(where, f"win_rate is {win!r} and control_verdicts "
                                     f"{controls!r}",
                              "a --control run: a win rate in [0, 1] and control verdicts")
    stated = settings["verdicts"]
    if not isinstance(stated, dict) or stated != dict(verdicts):
        raise _refuse_pose_qc(where, f"the header counts {stated!r} and the records "
                                     f"{dict(sorted(verdicts.items()))}",
                              "the header's verdict counts to be the records' own")


def check_pose_qc_record(where: str, line: int, record: Any) -> tuple[tuple[str, str], int, str]:
    """One qc record -> its (directory, camera) key, its episode and its verdict.

    The key is the one build_cache looks a stage pose up by: the directory as
    stage.relative wrote it, the layout's view key and the episode index the
    source wrote. Nothing is coerced: a bool, a string or a float episode, a
    path that is not root-relative and normalised, a verdict outside
    POSE_QC_VERDICTS and a score that does not give its own verdict are refused.
    """
    if not isinstance(record, dict):
        raise _refuse_pose_qc(where, f"line {line} is not an object", "one record per line")
    missing = [key for key in POSE_QC_RECORD_KEYS if key not in record]
    if missing:
        raise _refuse_pose_qc(where, f"line {line} holds no {missing}",
                              f"every record holding {list(POSE_QC_RECORD_KEYS)}")
    dataset, view, episode = record["dataset"], record["view"], record["episode"]
    verdict, score = record["verdict"], record["score"]
    if (not isinstance(dataset, str) or not dataset or dataset == "."
            or Path(dataset).is_absolute() or ".." in Path(dataset).parts
            or Path(dataset).as_posix() != dataset):
        raise _refuse_pose_qc(where, f"line {line} names directory {dataset!r}",
                              "a normalised root-relative directory, as stage.relative writes")
    if not isinstance(view, str) or not view:
        raise _refuse_pose_qc(where, f"line {line} names view {view!r}", "a layout view key")
    if type(episode) is not int or episode < 0:
        raise _refuse_pose_qc(where, f"line {line} names episode {episode!r}",
                              "the source's own episode index, a non-negative integer")
    if verdict not in POSE_QC_VERDICTS:
        raise _refuse_pose_qc(where, f"line {line} has verdict {verdict!r}",
                              f"one of {list(POSE_QC_VERDICTS)}")
    if verdict == "inconclusive":
        if score is not None:
            raise _refuse_pose_qc(where, f"line {line} is inconclusive with score {score!r}",
                                  "no score on an inconclusive record")
    elif (not _is_number(score) or not np.isfinite(score)
          or (score >= POSE_QC_THRESHOLD) != (verdict == "ok")):
        raise _refuse_pose_qc(where, f"line {line} has verdict {verdict!r} and score {score!r}",
                              f"a finite score that gives that verdict at {POSE_QC_THRESHOLD}")
    return (dataset, view), episode, verdict


def read_pose_qc(
    root: Path, paths: list[Path]
) -> tuple[dict[tuple[str, str], dict[int, str]], dict[str, Any]]:
    """The --pose-qc files -> each (directory, camera)'s verdict per episode, and a summary.

    Each file is read once, and the settings and the index name it with the
    verdict counts parsed from it, the evidence that was applied. The parse is
    stage.read_jsonl's (header on line 1, one record per
    non-blank line after it) with repeated object keys refused, since json.loads
    would silently keep the last. An episode stated twice, within one file or
    across files, is refused rather than settled by order. Records for a
    directory the manifest does not hold are kept in the counts and change
    nothing.
    """
    verdicts: dict[tuple[str, str], dict[int, str]] = {}
    counts: Counter = Counter()
    files, controls = [], []
    for path in paths:
        where = stage.relative(root, path)
        data = path.read_bytes()
        first, _, rest = data.decode("utf-8").partition("\n")
        try:
            head = json.loads(first, object_pairs_hook=_unique_keys)["provenance"]
            records = [json.loads(line, object_pairs_hook=_unique_keys)
                       for line in rest.split("\n") if line.strip()]
        except (ValueError, KeyError, TypeError) as error:
            raise _refuse_pose_qc(where, f"does not parse ({error})",
                                  "a provenance header and one json record per line, "
                                  "each key stated once") from error
        own: Counter = Counter()
        for line, record in enumerate(records, start=2):
            key, episode, verdict = check_pose_qc_record(where, line, record)
            held = verdicts.setdefault(key, {})
            if episode in held:
                raise _refuse_pose_qc(where, f"line {line} states {key[0]} {key[1]} episode "
                                             f"{episode} again",
                                      "one verdict per directory, camera and episode over "
                                      "every --pose-qc file")
            held[episode] = verdict
            own[verdict] += 1
        check_pose_qc_header(where, head, own, len(records))
        counts.update(own)
        files.append({"path": where, "verdicts": dict(sorted(own.items()))})
        controls.append({"path": where, "win_rate": head["settings"]["win_rate"],
                         "control_verdicts": head["settings"]["control_verdicts"]})
    summary = {
        "files": files,
        "threshold": POSE_QC_THRESHOLD,
        "unknown_on": list(POSE_QC_UNKNOWN),
        "verdicts": dict(sorted(counts.items())),
        "control": controls,
        "source_acceptance": "UNKNOWN: no win-rate or fail-share acceptance is set",
    }
    return verdicts, summary


def failed_poses(
    verdicts: dict[tuple[str, str], dict[int, str]]
) -> dict[tuple[str, str], frozenset[int]]:
    """The episodes whose stored pose becomes unknown, per (directory, camera)."""
    failed = {key: frozenset(episode for episode, verdict in held.items()
                             if verdict in POSE_QC_UNKNOWN)
              for key, held in verdicts.items()}
    return {key: episodes for key, episodes in failed.items() if episodes}


def check_pose_qc_views(entries: list[dict[str, Any]],
                        failed: dict[tuple[str, str], frozenset[int]]) -> None:
    """Refuse a failure on a manifest directory's camera this consumer cannot apply.

    Checked over the whole manifest before any entry is written, so every shard
    refuses the same file. A failure is applied only to a view whose extrinsics
    are kind episode from a stage file (the pose qc_extrinsics scored); one on a
    constant, mounted, sidecar or metadata view, or on a camera the entry does
    not have, would otherwise be dropped without a word. A directory the
    manifest does not hold is not this build's and is passed over.
    """
    specs = {(entry["path"], view["key"]): view["extrinsics"]
             for entry in entries for view in entry["views"]}
    held = {entry["path"] for entry in entries}
    for (dataset, key), episodes in sorted(failed.items()):
        if dataset not in held:
            continue
        spec = specs.get((dataset, key))
        if spec is None or any(spec.get(field) != value for field, value in POSE_QC_KINDS):
            raise ValueError(
                f"--pose-qc fails {len(episodes)} poses of {dataset} view {key!r}, which is "
                f"{'not a view of that entry' if spec is None else dict(spec)}\n"
                f"expected pose QC only on views of extrinsics kind episode, source stage\n"
                f"drop --pose-qc for this source or fix the layout"
            )


def pose_qc_applies(view: dict[str, Any]) -> bool:
    """Whether pose QC scores this view's pose: extrinsics kind episode from a stage file.

    The kept counts are taken over these views only. A sidecar, metadata or
    constant view can share a scored camera's key, and the census can hold ok
    or inconclusive records under its directory (only a failure there is
    refused), but none of them is a pose qc_extrinsics measured.
    """
    return all(view["extrinsics"].get(field) == value for field, value in POSE_QC_KINDS)


def kept_by_verdict(held: dict[int, str], episodes: list[dict[str, Any]]) -> Counter:
    """How many of one camera's kept episodes each verdict covers; unscored is not_scored."""
    return Counter(held.get(episode["episode"], "not_scored") for episode in episodes)


def cache_settings(parts: list[str], dropped: list[str],
                   pose_qc: dict[str, Any] | None) -> dict[str, Any]:
    """What the index records as this build's settings, and current_shard compares.

    The QC files and their verdict counts are added only when --pose-qc is
    given, so a build without it records the settings it always did, and a
    build with it is stale as soon as the QC verdicts it applied change.
    """
    settings: dict[str, Any] = {"parts": parts, "dropped_columns": dropped}
    if pose_qc is not None:
        settings["pose_qc"] = pose_qc["files"]
    return settings


def extrinsic_rotations(
    root: Path, entry: dict[str, Any], view: dict[str, Any],
    episodes: list[dict[str, Any]], stated: dict[str, dict[int, dict[str, Any]]],
    *, failed: frozenset[int] = frozenset(),
) -> np.ndarray:
    """One camera's rotations over the kept episodes, as one float32 array.

    The array is laid out by position in the kept list, because a filtered
    directory renumbers everything after its first drop and the loader reads
    this file by position. The lookup into the records is by the episode index
    the source itself wrote, which is what those records are keyed by.

    An episode whose pose failed projection QC (qc_extrinsics verdict in
    POSE_QC_UNKNOWN) gets the unknown marker; it stays in the kept list.
    """
    spec = view["extrinsics"]
    if failed and any(spec.get(field) != value for field, value in POSE_QC_KINDS):
        raise ValueError(
            f"{entry['name']}: view {view['key']!r} has extrinsics kind {spec['kind']!r} "
            f"source {spec.get('source')!r} and a pose QC failure\n"
            f"expected pose QC only on kind episode, source stage views "
            f"(qc_extrinsics scores the AgiBot head)\n"
            f"drop --pose-qc for this source or fix the layout")
    records: dict[int, dict[str, Any]] = {}
    if spec["source"] == "sidecar":
        records = {
            int(record[spec.get("index_key", "episode_index")]): record
            for record in (json.loads(line) for line in
                           (root / spec["path_source"]).read_text(
                               encoding="utf-8").splitlines() if line.strip())
        }
    elif spec["source"] == "stage":
        records = stated.get((entry["path"], view["key"]), {})
    refuse = spec.get("refuse_when")
    rotations = np.zeros((len(episodes), 3, 3), dtype=np.float32)
    for position, episode in enumerate(episodes):
        if spec["source"] == "metadata":
            stored = episode["metadata"].get(spec["key"])
        else:
            record = records.get(episode["episode"])
            stored = record.get(spec["key"]) if record else None
        # A source that publishes every camera's pose under one field names
        # the camera inside it, so one field serves several views.
        if "member" in spec:
            stored = stored.get(spec["member"]) if isinstance(stored, dict) else None
        if refuse is not None and refused_episode(episode, refuse):
            stored = None
        if episode["episode"] in failed:
            stored = None
        rotations[position] = rotation_from(stored, spec["frame"])
    return rotations


def extrinsics_payload(view: dict[str, Any], rotations: np.ndarray, name: str) -> np.ndarray:
    """The array a view's extrinsics file holds: one 3x3 per kept episode, or one for a constant.

    A constant camera's file is the one rotation, shape (1, 3, 3), which is what the loader
    reads for kind constant; one row per episode was refused by the
    loader's shape check. Every kept episode must carry that same rotation (NaN rows, the
    unknown pose, included bit for bit), or the view is not constant and is refused by name.
    """
    if view["extrinsics"]["kind"] != "constant":
        return rotations
    if len(rotations) == 0:
        raise ValueError(
            f"{name}: view {view['key']!r} states a constant camera pose and keeps no episode.\n"
            "Expected at least one kept episode to read the rotation from.\n"
            "Check the directory's kept list.")
    differs = [position for position in range(1, len(rotations))
               if not np.array_equal(rotations[position], rotations[0], equal_nan=True)]
    if differs:
        raise ValueError(
            f"{name}: view {view['key']!r} states a constant camera pose and {len(differs)} "
            f"of its {len(rotations)} kept episodes carry another rotation (first at kept "
            f"position {differs[0]}).\n"
            "Expected one rotation for the directory.\n"
            "State extrinsics kind episode in the layout, or fix the source's pose.")
    return rotations[:1].copy()


def refused_episode(episode: dict[str, Any], refuse: dict[str, Any]) -> bool:
    """Whether this episode's own metadata says its stored pose does not apply.

    A directory-wide pose that holds for most of its episodes and not the rest
    is refused per episode rather than per directory, so the episodes it does
    describe keep it and the others reach the model as unknown extrinsics
    instead of being dropped from the corpus.
    """
    return bool(episode["metadata"].get(refuse["key"])) == bool(refuse["equals"])


def npy_bytes(array: np.ndarray) -> bytes:
    """One array as the .npy payload the manifest names."""
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def intrinsics_records(path: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    """One intrinsics file, keyed by directory, episode index and view."""
    _, records = stage.read_jsonl(path)
    return {(record["dataset"], int(record["episode"]), record["view"]): record
            for record in records}


def calibrations(
    entry: dict[str, Any], view: dict[str, Any],
    episodes: list[dict[str, Any]], records: dict[tuple[str, int, str], dict[str, Any]],
) -> np.ndarray:
    """One view's calibration per kept episode, in kept order, as one array.

    Keyed by position, like the extrinsics beside it: a filtered directory
    renumbers everything after its first drop. An episode with no record is a
    row of the unknown marker rather than a row of zeros, because a zero focal
    would be read as a camera rather than as the absence of one.
    """
    table = np.full((len(episodes), len(rays.CALIBRATION_COLUMNS)), np.nan, dtype=np.float64)
    for position, episode in enumerate(episodes):
        record = records.get((entry["path"], episode["episode"], view["key"]))
        if record is None:
            continue
        table[position] = [float(record[column]) for column in rays.CALIBRATION_COLUMNS]
    return table


def episode_frames(
    entry: dict[str, Any], count: int, rotations: dict[str, np.ndarray]
) -> list[str]:
    """The frame each kept episode may be drawn in, by the loader's own audit.

    A directory is in camera frame, but an episode of it whose cameras all fail
    the audit can only be drawn in robot frame, and the robot clause states the
    base frame in words that no mask retracts. One camera surviving is enough
    for the camera clause, which names no camera and defers to each arm's own
    anchor. is_rotation is the loader's audit, called here on the same arrays
    the loader reads.

    A mounted camera's array holds the offset from its arm rather than a pose,
    and the loader composes the pose from it. An offset that is a rotation
    composes to one, so the audit reads the same either way.
    """
    # Compare the parsed kind: "camera:1" != "camera" is True, and returning
    # ["robot"] * count here would label every episode of a camera:N corpus robot-frame,
    # so the loader would never rotate its chunks -- the bug that scored 0.00% over 2258
    # episodes, moved one file upstream into the cache.
    if parse_frame_ref(entry["frame_ref"])[0] != "camera" or not rotations:
        return ["robot"] * count
    survives = np.zeros(count, dtype=bool)
    for view in entry["views"]:
        table = rotations.get(view["key"])
        if table is None:
            continue
        if view["extrinsics"]["kind"] == "constant":
            table = np.repeat(table[:1], count, axis=0)
        for position, rotation in enumerate(table):
            survives[position] |= is_rotation(torch.from_numpy(rotation))
    return ["camera" if seen else "robot" for seen in survives]


def instruction_of(tasks: list[str]) -> str:
    """An episode's task list -> the instruction it was recorded under.

    The first task that says something, not the first task. A source may write
    an empty string ahead of the real one: 55 of 99 episodes of one OXE dataset
    record ["", "pick the cube"], and taking position zero rendered a prompt
    with no task and stopped the run ten stages in. An episode with nothing at
    all in its list is a defect validate_episodes already names, so by the time
    a list reaches here it holds a task; returning empty keeps that raise where
    it belongs rather than moving it.
    """
    return next((task for task in tasks if task.strip()), "")


def context_of(entry: dict[str, Any], instruction: str, frame: str, key: str) -> RenderContext:
    """What one episode of one directory contributes to the rendered text.

    The frequency is the directory's control_hz (None renders NA) and the
    embodiment clause is the episode's key's own.
    """
    prompt = entry["prompt"]
    clauses = prompt["embodiment_clauses"]
    if key not in clauses:
        raise ValueError(
            f"{entry['name']} has an episode of key {key} and no clause for it\n"
            f"expected one clause per key the entry holds, {sorted(clauses)}\n"
            f"rebuild the manifest"
        )
    return RenderContext(
        task=instruction,
        embodiment_clauses=(clauses[key],),
        frame_clause=prompt["frame_clause"][frame],
        frequency=entry["control_hz"],
    )


def episode_keys(root: Path, entry: dict[str, Any], count: int) -> list[str]:
    """The normaliser key of each kept position, from the manifest's embodiment block."""
    block = entry["embodiment"]
    if block["kind"] == "constant":
        return [str(block["key"])] * count
    indices = np.load(stage.under_root(root, block["path"]))
    if indices.shape[0] != count:
        raise ValueError(
            f"{refusal_prefix(16)} {entry['name']}: the key array holds "
            f"{indices.shape[0]} positions for {count} kept episodes\n"
            f"Expected one key per kept episode, in kept order.\n"
            f"Rebuild the arrays with build_embodiment over this kept file.")
    return [str(block["keys"][int(index)]) for index in indices]


def build_prompts(
    template: Template,
    entry: dict[str, Any],
    episodes: list[dict[str, Any]],
    tokenizer: Any,
    max_tokens: int,
    rotations: dict[str, np.ndarray],
    keys: list[str],
) -> tuple[bytes, tuple[int, str, int]]:
    """The rendered prompts of one directory, and its longest (tokens, dir, episode).

    Each record carries the frame its clause was rendered for, which the loader
    compares with the frame the sample resolves to, and the key, control_hz,
    head, task and tail the runner and the server read (C14). Nothing is
    dropped under v6: a record over the budget refuses, naming its directory
    and episode (G5).
    """
    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    frames = episode_frames(entry, len(episodes), rotations)
    lines = []
    longest = (0, entry["name"], -1)
    for position, episode in enumerate(episodes):
        context = context_of(entry, instruction_of(episode["tasks"]), frames[position],
                             keys[position])
        try:
            rendered = fit(template, context, max_text_tokens=max_tokens, count=count)
        except ValueError as error:
            raise ValueError(f"{entry['name']} episode {episode['episode']} (kept position "
                             f"{position}): {error}") from error
        if rendered.head + rendered.task + rendered.tail != rendered.text:
            raise ValueError(
                f"{entry['name']} episode {episode['episode']}: head + task + tail is not the "
                f"rendered text\nexpected the renderer's own split\n"
                f"check neo_vla.data.prompt.render")
        tokens = count(rendered.text)
        longest = max(longest, (tokens, entry["name"], int(episode["episode"])))
        lines.append(stage.dumps_line(
            {"position": position, "episode": episode["episode"],
             "frame_ref": frames[position], "embodiment": keys[position],
             "control_hz": entry["control_hz"], "tokens": tokens, "head": rendered.head,
             "task": rendered.task, "tail": rendered.tail, "text": rendered.text}
        ))
    return "".join(lines).encode("utf-8"), longest


def shard_plan(
    entries: list[dict[str, Any]], shard: tuple[int, int], out: Path, args: Any
) -> tuple[list[int], dict[str, Any]]:
    """The entries one shard writes, and what its index records about the split.

    The library versions are recorded because the arrow part's bytes are what
    the writer emits: two shards on two versions of pyarrow each produce a file
    that is correct on its own and differs in its bytes, and nothing else in
    this pipeline compares them.
    """
    index, count = shard
    named = SHARD_INDEX_NAME.format(index=index, count=count)
    if out.name != named:
        raise ValueError(
            f"--out names {out.name} for shard {index} of {count}\n"
            f"expected a file called {named}\n"
            f"give each shard its own --out, or they overwrite one index"
        )
    assigned = shard_assignment(entries, count, args.shard_by)
    mine = [position for position, owner in enumerate(assigned) if owner == index]
    block = {
        "index": index,
        "count": count,
        "key": args.shard_by,
        "entries": [entries[position]["path"] for position in mine],
        "libraries": {"numpy": np.__version__, "pyarrow": pa.__version__},
        "tokenizer": args.tokenizer,
    }
    return mine, block


def check_shard_wrote(
    root: Path, manifest: dict[str, Any], parts: list[str],
    mine: list[int], index: int, artifacts: list[str],
) -> None:
    """Refuse a shard that wrote anything other than the entries it was given.

    The merge is held to the same plan, so checking it here means a writer and
    a plan that have drifted apart are named by the run that introduced the
    drift rather than by a merge hours later.
    """
    planned = planned_artifacts(root, manifest, parts)
    expected = {name for position in mine for name in planned[position]}
    if index == 0 and "prompt" in parts:
        expected.add(prompt_artifact(root, manifest, "stamp.json"))
    if set(artifacts) == expected:
        return
    stray = sorted(set(artifacts) - expected)
    absent = sorted(expected - set(artifacts))
    raise ValueError(
        f"shard {index} wrote {len(stray)} artifacts it was not given "
        f"and left out {len(absent)}\n"
        f"expected the entries the split gave it, {stray[:1] or absent[:1]} among them\n"
        f"the writing loop and planned_artifacts disagree: fix them together"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--kept", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--out", required=True, help="where the cache index goes")
    parser.add_argument("--parts", default=",".join(PARTS))
    parser.add_argument("--drop-column", action="append", default=[])
    parser.add_argument("--tokenizer", help="a tokeniser directory, for the prompt part")
    parser.add_argument("--shard", help="K/N: write only the entries of shard K of N")
    parser.add_argument("--shard-by", default=SHARD_KEYS[0],
                        help=f"how the entries are divided, one of {list(SHARD_KEYS)}")
    parser.add_argument("--pose-qc", action="append", default=[],
                        help="a qc_extrinsics census (--control, no --sample) whose failed "
                             "poses become unknown; the episodes are kept")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    parts = [part.strip() for part in args.parts.split(",") if part.strip()]
    unknown = sorted(set(parts) - set(PARTS))
    if unknown:
        raise ValueError(
            f"unknown cache parts {unknown}\n"
            f"expected a comma-separated subset of {list(PARTS)}\n"
            f"fix --parts"
        )
    if args.shard_by not in SHARD_KEYS:
        raise ValueError(
            f"--shard-by is {args.shard_by!r}\n"
            f"expected one of {list(SHARD_KEYS)}\n"
            f"fix --shard-by"
        )
    shard = parse_shard(args.shard) if args.shard else None
    root = stage.resolve_root(args.root)
    manifest_path = stage.under_root(root, args.manifest)
    kept_path = stage.under_root(root, args.kept)
    inventory_path = stage.under_root(root, args.inventory)
    out = stage.under_root(root, args.out)
    qc_paths = [stage.under_root(root, path) for path in args.pose_qc]

    manifest = stage.read_json(manifest_path)
    from neo_vla.data.prompt.embodiments import canonical_prompt, validate_manifest_prompts

    validate_manifest_prompts(manifest)
    manifest = {**manifest, "datasets": [
        {**entry, "prompt": canonical_prompt(entry)} for entry in manifest["datasets"]]}
    stage.refuse_write_into_input(out, [root / entry["path"] for entry in manifest["datasets"]])

    # The QC files are read whole before the shard's currency check and before
    # any entry is stepped over, so every shard refuses the same bad file and a
    # shard cannot read an absent failure as an ok pose.
    qc_verdicts: dict[tuple[str, str], dict[int, str]] = {}
    qc_failed: dict[tuple[str, str], frozenset[int]] = {}
    qc_summary: dict[str, Any] | None = None
    if qc_paths:
        qc_verdicts, qc_summary = read_pose_qc(root, qc_paths)
        qc_failed = failed_poses(qc_verdicts)
        check_pose_qc_views(manifest["datasets"], qc_failed)
    qc_views = {view for _, view in qc_verdicts}
    qc_kept: Counter = Counter()

    settings = cache_settings(parts, args.drop_column, qc_summary)
    entries = manifest["datasets"]
    mine = list(range(len(entries)))
    block: dict[str, Any] | None = None
    if shard is not None:
        mine, block = shard_plan(entries, shard, out, args)
        if current_shard(out, root, settings, block):
            logger.info("shard %d of %d is current, %d directories",
                        shard[0], shard[1], len(mine))
            return 0

    _, kept = stage.read_jsonl(kept_path)
    _, inventory = stage.read_jsonl(inventory_path)

    by_key = {(item["dataset"], item["episode"]): item for item in inventory}
    per_directory: dict[str, list[dict[str, Any]]] = {}
    for record in kept:
        if record["kind"] == "kept":
            per_directory.setdefault(record["dataset"], []).append(
                by_key[(record["dataset"], record["episode"])]
            )

    tokenizer = None
    template = None
    # The stamp's own fields, an earlier label of this layout read as this one;
    # extra fields an older manifest states are not read.
    stated = manifest["prompt"]["stamp"]
    declared = Stamp(**{field.name: prefix_layout_of(stated[field.name])
                        if field.name == "prefix_layout" else stated[field.name]
                        for field in dataclasses.fields(Stamp)})
    if "prompt" in parts:
        if args.tokenizer is None:
            raise ValueError(
                "the prompt part needs a tokeniser\n"
                "expected --tokenizer pointing at the run's tokeniser directory\n"
                "pass it or drop prompt from --parts"
            )
        # The same call the encoder makes. Loaded another way it is another
        # tokeniser, and the budget this stage fits the text to is then not the
        # budget the encoder measures it against.
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.tokenizer), local_files_only=True, trust_remote_code=False
        )
        template = load_template(root / manifest["prompt"]["template"])
        check_mode_agrees(template, declared.intrinsics_mode)
        # The artifact on disk has to be the one the manifest was stamped
        # against, or the cache would carry text from a different layout.
        stamp_module.check(
            stamp_of(template, tokenizer_id=declared.tokenizer_id,
                     max_text_tokens=declared.max_text_tokens,
                     prefix_layout=declared.prefix_layout, n_views=declared.n_views,
                     intrinsics_mode=declared.intrinsics_mode),
            declared,
        )

    # Read once per file rather than once per directory: the largest of them
    # holds four hundred thousand records and every directory names the same one.
    # Only for the views that need an array; a constant view names its file too,
    # and its six numbers are already in the manifest.
    sources: dict[str, dict[tuple[str, int, str], dict[str, Any]]] = {}
    if "intrinsics" in parts:
        for entry in manifest["datasets"]:
            for view in entry["views"]:
                if view["intrinsics"]["kind"] != "episode":
                    continue
                named = view["intrinsics"]["path_source"]
                if named not in sources:
                    sources[named] = intrinsics_records(stage.under_root(root, named))

    poses: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for entry in manifest["datasets"]:
        for view in entry["views"]:
            spec = view["extrinsics"]
            if spec.get("source") != "stage" or spec["path_source"] in poses:
                continue
            poses[spec["path_source"]] = stage_extrinsics(
                stage.under_root(root, spec["path_source"]))

    artifacts: list[str] = []
    written = 0
    longest = (0, "", -1)
    absent_total = 0
    stores = {"ingest": 0, "sidecar": 0, "none": 0}
    chosen = set(mine)
    for position, entry in enumerate(manifest["datasets"]):
        # The entries of another shard are stepped over here rather than
        # filtered out above: everything above is read for the whole manifest
        # in both cases, so a shard reads the poses and the calibrations the
        # unsharded run reads and cannot be handed an absent one as an unknown.
        if position not in chosen:
            continue
        episodes = per_directory.get(entry["path"], [])
        # Read once: the rotations are what the extrinsics artifact holds and
        # what the frame clause is rendered from, and one of the sidecars they
        # come from holds a hundred and fifty thousand records.
        rotations: dict[str, np.ndarray] = {}
        if {"extrinsics", "prompt"} & set(parts):
            for view in entry["views"]:
                if view["extrinsics"]["kind"] == "none":
                    continue
                pair = (entry["path"], view["key"])
                rotations[view["key"]] = extrinsic_rotations(
                    root, entry, view, episodes,
                    poses.get(view["extrinsics"].get("path_source"), {}),
                    failed=qc_failed.get(pair, frozenset()))
                # Counted per kept episode of a view pose QC applies to (kind
                # episode, source stage), so an inconclusive or unscored pose
                # is a number the index states and never a pass. A sidecar,
                # metadata or constant view sharing a scored key is not counted,
                # whatever the census holds under its directory.
                applies = pose_qc_applies(view)
                if applies and pair in qc_verdicts:
                    qc_kept.update(kept_by_verdict(qc_verdicts[pair], episodes))
                elif applies and view["key"] in qc_views:
                    qc_kept["not_scored"] += len(episodes)
        if "episode_starts" in parts:
            path = stage.under_root(root, entry["episode_starts"])
            written += write_artifact(
                path, episode_starts([episode["rows"] for episode in episodes]))
            artifacts.append(entry["episode_starts"])
        if "arrow" in parts:
            path = stage.under_root(root, entry["cache"]["arrow"])
            dropped = args.drop_column + sorted(picture_columns(episodes))
            stored, metadata = stored_parts(root, entry, episodes)
            names = tuple(stored_columns(entry))
            stores[("sidecar" if stored is not None else "ingest") if names else "none"] += 1
            written += write_artifact(
                path, arrow_table(root, [episode["data_file"] for episode in episodes],
                                  dropped, [view["key"] for view in entry["views"]],
                                  entry["path"], stored=stored, names=names,
                                  metadata=metadata))
            artifacts.append(entry["cache"]["arrow"])
        if "frames" in parts:
            path = stage.under_root(root, entry["cache"]["frames"])
            index, absent = frame_index(root, entry, episodes)
            if absent:
                # Named as it is written rather than only counted at the end: a
                # directory that lost pictures is the one thing a reader of
                # this log needs to be able to go and look at.
                logger.warning("%s has no frame file for %d of its episode "
                               "cameras", entry["name"], absent)
            absent_total += absent
            written += write_artifact(path, stage.dumps_json(index).encode("utf-8"))
            artifacts.append(entry["cache"]["frames"])
        if "extrinsics" in parts:
            for view in entry["views"]:
                if view["key"] not in rotations:
                    continue
                named = view["extrinsics"]["path"]
                path = stage.under_root(root, named)
                written += write_artifact(path, npy_bytes(extrinsics_payload(
                    view, rotations[view["key"]], entry["name"])))
                artifacts.append(named)
        if "intrinsics" in parts:
            for view in entry["views"]:
                if view["intrinsics"]["kind"] != "episode":
                    continue
                named = view["intrinsics"]["path"]
                path = stage.under_root(root, named)
                written += write_artifact(path, npy_bytes(calibrations(
                    entry, view, episodes,
                    sources[view["intrinsics"]["path_source"]])))
                artifacts.append(named)
        if "prompt" in parts:
            path = stage.under_root(
                root, Path(manifest["prompt"]["cache"]) / f"{entry['name']}.jsonl")
            payload, found = build_prompts(
                template, entry, episodes, tokenizer, declared.max_text_tokens,
                rotations, episode_keys(root, entry, len(episodes)))
            longest = max(longest, found)
            written += write_artifact(path, payload)
            artifacts.append(stage.relative(root, path))

    # One writer for the stamp. Its first write is not through a temporary and
    # its name is the same for every shard, so shards past the first check it
    # and leave it to shard zero, which writes the bytes one process would.
    if "prompt" in parts and (shard is None or shard[0] == 0):
        stamp_path = stage.under_root(
            root, Path(manifest["prompt"]["cache"]) / "stamp.json")
        write_artifact(
            stamp_path,
            Path(stamp_module.write(declared, stamp_path.parent)).read_bytes())
        artifacts.append(stage.relative(root, stamp_path))

    if shard is not None:
        check_shard_wrote(root, manifest, parts, mine, shard[0], artifacts)

    inputs = {"manifest": manifest_path, "kept": kept_path, "inventory": inventory_path}
    inputs.update({f"pose_qc_{number}": path for number, path in enumerate(qc_paths)})
    payload = {
        "provenance": stage.header(
            "build_cache", root=root,
            inputs=inputs,
            tree=REPO, extra=settings,
        ),
        "artifacts": sorted(set(artifacts)),
    }
    if block is not None:
        payload["shard"] = block
    if qc_summary is not None:
        payload["pose_qc"] = {**qc_summary, "kept": dict(sorted(qc_kept.items()))}
    index_written = stage.write_json(out, payload)
    logger.info("%d artifacts, %d rewritten, %d episode cameras with no frame file",
                len(set(artifacts)), written, absent_total)
    if "arrow" in parts:
        logger.info("arrow files by v7 writer (of %d): %s", sum(stores.values()), stores)
    if "prompt" in parts:
        # G5: the longest record of the build, against the budget it had to fit.
        logger.info("longest prompt %d of %d tokens: %s episode %d", longest[0],
                    declared.max_text_tokens, longest[1], longest[2])
    if qc_summary is not None:
        # Kept episodes by verdict: fail made unknown and kept; inconclusive and
        # not_scored keep the audited pose and are not a pass.
        logger.info("pose QC over kept episodes of scored views: %s",
                    dict(sorted(qc_kept.items())))
    logger.info("%s %s", "wrote" if index_written else "unchanged", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
