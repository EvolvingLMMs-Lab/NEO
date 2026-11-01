"""Reading one directory: the arrow table and the frame files a manifest names.

The cache stage writes two things per directory: one arrow file holding the
numeric columns of the kept episodes concatenated in kept order, and one frame
index naming, per camera, the file each kept episode's pictures live in. This
module opens those two and nothing else, so a window is addressed by kept
position rather than by the episode index the source happened to write.

Frames arrive either as an mp4 per episode or as an image column inside the
episode's own parquet. Both name one file per kept episode, so the two differ
only in how a single frame is pulled out of that file. A directory whose
cameras were recorded on separate clocks states, per camera, which frame of
that camera's own file a row reads and whether it has one. The v7 joints, hand
and body columns are read beside the state, under their writer's stamp.
"""
from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.parquet as pq
import torch
from av.video.reformatter import VideoReformatter
from torchvision.io import ImageReadMode, decode_image

from ..constants import (
    ACTION_RECORD,
    ACTION_STAMP,
    BODY_RECORD,
    BODY_STAMP,
    COLUMN_BODY,
    COLUMN_BODY_COMMAND,
    COLUMN_HAND,
    COLUMN_JOINTS,
    COLUMN_JOINTS_COMMAND,
    INGEST_LAYOUT,
    INGEST_LAYOUT_FIELD,
    OLD_PREFIX_LAYOUTS,
    STATE_RECORD,
    STATE_STAMP,
    refusal_prefix,
)

logger = logging.getLogger(__name__)

# A node-local copy of the corpus, named by the absolute path it shadows: a file
# at /a/b is at <root>/a/b. Unset means every read goes to the shared
# mount, which is what this module did before the variable existed.
MIRROR_ROOT_VAR = "NEO_VLA_VIDEO_MIRROR"

# One warning per process. A worker holds a reader per directory, and every
# one announcing the same absent mirror is thousands of lines saying one thing.
_warned_absent_mirror = False

# How many bytes of timestamp tables one process keeps, and the variable that
# moves it. Sixty-four mebibytes holds around 860 of this corpus's tables,
# which is 380 samples' worth of the files one worker opens; a node runs eight
# ranks of eight workers, so these tables are at most 4.3 GB of a cgroup that
# allows 2,061 GB. A larger budget buys little: over the draw stream one worker
# really follows, sixteen times this raises the hit rate from 0.2% to 3.5%,
# against the 66% an unbounded one reaches by holding 36.7 GB.
TIMESTAMP_BUDGET_VAR = "NEO_VLA_TIMESTAMP_CACHE_BYTES"
TIMESTAMP_BUDGET = 64 * 1024 * 1024

# What one cached timestamp costs: the integer, the tuple's slot for it and
# what the allocator rounds those up to. Measured at 43.8 resident bytes a
# frame over 40 agibot files and at 39.0 over 6,000 tables of one length, so
# the charge below is above the truth and a full store is never larger than
# its budget says. A table of 1,831 frames, this corpus's mean, is 88 KB here.
STAMP_BYTES = 48
TABLE_BYTES = 64

# What one entry costs a store before its value: the key, which is a corpus
# path of about 118 characters, the pair the value is held as, and the slot the
# dictionary keeps both in. Measured at 289.8 bytes an entry over 55,509 of
# this corpus's paths, three times over. Anything held is charged at least
# this, so a caller that undercounts its value cannot fill a store for free.
ENTRY_BYTES = 320

# The same for one remembered mirror decision: a path string, the Path object
# it resolved to and the dictionary's own slot, measured at 467 resident bytes
# over 200,000 of this corpus's paths. Bounded on a budget of its own because
# it is keyed by the same files and would otherwise grow along with them; eight
# mebibytes is around 16,000 decisions and 0.5 GB across a node's workers.
MIRROR_BUDGET = 8 * 1024 * 1024
MIRROR_ENTRY_BYTES = 512

# How a directory stores its pictures. Both kinds name one file per kept
# episode; mp4 holds the whole episode as a stream and parquet holds it as one
# encoded image per row.
FRAME_KINDS = ("mp4", "parquet")

# What an eight-bit picture divides by. Frames leave here in 0..1, which is
# what images.resize_frame reads; the scale to the model's own range is that
# module's, and happens once.
PIXEL_MAX = 255.0

# The numeric column a directory names its per-camera frame index in, one per
# camera key. A source whose cameras were recorded on one clock has none, and
# every camera is read at the window's own step. The RH20T join has one clock
# per camera, so a row of a joined episode names the frame each camera
# contributes to it and reading them all at the same step would pair one
# camera's instant with another's seconds away.
VIEW_FRAME_SUFFIX = ".source_frame"

# The column a directory names, per camera, the rows that camera has a frame
# for. A false flag is that camera absent from that step, which is what the
# join writes where its nearest frame is further from the row's own time than
# the tolerance it recorded. The nearest frame would be a picture of another
# instant and a black tile would be a picture of nothing, so neither is placed.
FRAME_VALID_SUFFIX = ".frame_valid"

# The column LeRobot writes on every row: the source episode the row was
# recorded in. It is what a kept position of the cache says it holds, so R16
# holds the directory's other per-position records (the prompt, the embodiment
# ids) against it, and a cache without it is refused rather than trusted.
EPISODE_ID_COLUMN = "episode_index"

# The stored v7 columns a window may carry beside the state and the action:
# arm joints and hand per arm block, the body, and the two command columns a
# family whose target is its command stores. NaN in them
# is a value not recorded, never 0, so a window pads them with NaN past the end
# of its episode where the state and the action pad with 0.
PART_COLUMNS = (COLUMN_JOINTS, COLUMN_HAND, COLUMN_BODY, COLUMN_JOINTS_COMMAND,
                COLUMN_BODY_COMMAND)

# Where a directory's stored v7 columns came from, and the arrow schema
# metadata key the cache stage copies that writer's stamp under: an ingest output's
# meta/info.json fields, or a stage-1 sidecar's own
# footer stamp, which the cache stage merged row-aligned into the arrow. A
# column without its stamp is refused, never read as "not recorded".
PART_ORIGINS = ("ingest", "sidecar")
INGEST_STAMP_KEY = b"neo_vla_ingest"
SIDECAR_STAMP_KEY = b"neo_vla_sidecar"

# The sidecar writer this reader takes (the writer's FORMAT and
# TOOL_VERSION; the stage tool's 2.0.0): any other is a sidecar from another
# layout or an older writer, refused by name.
SIDECAR_FORMAT = "neo_vla_sidecar_v1"
SIDECAR_TOOL_VERSION = "2.0.0"
# The fields of a sidecar's stamp the cache stage copies into the arrow: what
# the sidecar writer states (tools/sidecars/common.stamp_of and write_built).
SIDECAR_STAMP_FIELDS = ("format", "ingest_layout", "tool", "tool_version", "stamps", "entry",
                        "directory", "source", "family", "arms", "rows", "episodes",
                        "episode_rows", "columns", "column_order", "checks", "git_commit",
                        "source_root")

# The ingest stamp's fields, as the v7 writer states them in meta/info.json.
INGEST_STAMP = {INGEST_LAYOUT_FIELD: INGEST_LAYOUT, "action_record": ACTION_RECORD,
                "state_record": STATE_RECORD, "body_record": BODY_RECORD}


@dataclass(frozen=True)
class StoredParts:
    """The stored v7 columns one directory's layouts read, and who wrote them.

    columns  column name -> values per row (arms x 7, arms x 12, or 8)
    origin   a member of PART_ORIGINS
    entry    the manifest entry's name, which a sidecar's stamp names too
    """

    columns: dict[str, int]
    origin: str
    entry: str

    def __post_init__(self) -> None:
        unknown = sorted(set(self.columns) - set(PART_COLUMNS))
        if unknown:
            raise ValueError(
                f"directory {self.entry!r} names stored columns {unknown}.\n"
                f"Expected a subset of {list(PART_COLUMNS)}, one spelling for every writer.\n"
                "Name the v7 column neo_vla.constants spells."
            )
        if self.origin not in PART_ORIGINS:
            raise ValueError(
                f"directory {self.entry!r} states its stored columns came from "
                f"{self.origin!r}.\nExpected one of {PART_ORIGINS}.\n"
                "A sidecar entry names its sidecar; every other directory is an ingest output."
            )
        for name, width in self.columns.items():
            if isinstance(width, bool) or not isinstance(width, int) or width < 1:
                raise ValueError(
                    f"directory {self.entry!r} column {name!r} is {width!r} wide.\n"
                    "Expected a positive number of values per row.\n"
                    "Derive the width from the directory's arms."
                )


def _refuse_stamp(where: str, said: str, expected) -> ValueError:
    return ValueError(
        f"{refusal_prefix(22)} the cache at {where} {said}.\n"
        f"Expected {expected}.\n"
        "Rebuild the cache from a v7 ingest output or a current sidecar; a stored column "
        "without its writer's stamp is refused, never read as not recorded."
    )


def _metadata_stamp(table: pa.Table, key: bytes, where: str) -> dict:
    """One stamp from the arrow schema metadata, or R22 when it is not there."""
    metadata = table.schema.metadata or {}
    if key not in metadata:
        raise _refuse_stamp(where, f"carries no {key.decode()} stamp",
                            f"the stamp the cache stage copies under {key.decode()!r}")
    try:
        stamp = json.loads(metadata[key])
    except json.JSONDecodeError as error:
        raise _refuse_stamp(where, f"carries a {key.decode()} stamp that is not json "
                                   f"({error.msg})", "a json object") from error
    if not isinstance(stamp, dict):
        raise _refuse_stamp(where, f"carries a {key.decode()} stamp {stamp!r}", "a json object")
    return stamp


def check_ingest_stamp(table: pa.Table, where: str) -> dict:
    """Refuse a cache whose ingest stamp is missing or is not the v7 one."""
    stamp = _metadata_stamp(table, INGEST_STAMP_KEY, where)
    layout = stamp.get(INGEST_LAYOUT_FIELD)
    if layout in OLD_PREFIX_LAYOUTS:
        raise _refuse_stamp(where, f"states {INGEST_LAYOUT_FIELD} {layout!r}, a retired "
                                   "layout's ingest output", f"{INGEST_LAYOUT!r}")
    for field, expected in INGEST_STAMP.items():
        if stamp.get(field) != expected:
            raise _refuse_stamp(where, f"states ingest {field} {stamp.get(field)!r}",
                                f"{expected!r}")
    return stamp


def check_sidecar_stamp(table: pa.Table, parts: StoredParts, where: str) -> dict:
    """Refuse a sidecar stamp from another writer, layout, record set or directory.

    The stamp is the sidecar's footer, carried into the arrow by the cache
    stage. It must name this format, the v7 ingest layout, this writer's
    version, the three record stamps and this entry; and it must have written
    every column the layouts read. Its episodes and rows are held to the
    cache's by check_sidecar_episodes.
    """
    stamp = _metadata_stamp(table, SIDECAR_STAMP_KEY, where)
    layout = stamp.get("ingest_layout")
    if layout in OLD_PREFIX_LAYOUTS:
        raise _refuse_stamp(where, f"carries a sidecar of ingest layout {layout!r}, a retired "
                                   "layout", f"{INGEST_LAYOUT!r}")
    records = {"action": ACTION_STAMP, "state": STATE_STAMP, "body": BODY_STAMP}
    for field, expected in (("format", SIDECAR_FORMAT), ("ingest_layout", INGEST_LAYOUT),
                            ("tool_version", SIDECAR_TOOL_VERSION), ("stamps", records),
                            ("entry", parts.entry)):
        if stamp.get(field) != expected:
            raise _refuse_stamp(where, f"carries a sidecar stamp whose {field} is "
                                       f"{stamp.get(field)!r}", f"{expected!r}")
    written = set(stamp.get("column_order") or ())
    missing = sorted(set(parts.columns) - written)
    if missing:
        raise _refuse_stamp(where, f"reads {missing} and its sidecar wrote "
                                   f"{sorted(written)}", "every column the layouts read")
    return stamp


def check_sidecar_episodes(stamp: dict, ids: np.ndarray, starts: torch.Tensor,
                           where: str) -> None:
    """Refuse (R16) a sidecar that does not hold every kept episode with its rows.

    ids are the source episode of each kept position and starts the kept
    boundaries. A sidecar built before a directory changed lists another row
    count, or no entry, for some episode; merging it would shift every later
    row of the directory onto the wrong state.
    """
    listed = stamp.get("episodes")
    counts = stamp.get("episode_rows")
    if not isinstance(listed, list) or not isinstance(counts, list) \
            or len(listed) != len(counts):
        said = (f"{len(listed)} episodes and {len(counts)} row counts"
                if isinstance(listed, list) and isinstance(counts, list)
                else f"episodes {listed!r} and row counts {counts!r}")
        raise ValueError(
            f"{refusal_prefix(16)} the cache at {where} carries a sidecar stamp listing "
            f"{said}.\n"
            "Expected one row count per listed episode.\n"
            "Rebuild the sidecar with the sidecar writer and the cache from it."
        )
    rows = {int(episode): int(count) for episode, count in zip(listed, counts)}
    lengths = torch.diff(starts.to(torch.int64)).tolist()
    for position, (episode, length) in enumerate(zip(ids.tolist(), lengths)):
        held = rows.get(int(episode))
        if held != int(length):
            said = "holds no episode" if held is None else f"holds {held} rows of episode"
            raise ValueError(
                f"{refusal_prefix(16)} the cache at {where}: kept position {position} is source "
                f"episode {int(episode)} of {int(length)} rows and its sidecar {said} "
                f"{int(episode)}.\n"
                "Expected the sidecar to hold every kept episode with the cache's rows.\n"
                "Rebuild the sidecar against the directory as it is now, then the cache."
            )


def per_view_columns(
    table: pa.Table,
    view_keys: tuple[str, ...],
    *,
    suffix: str,
    holds: Callable[[pa.DataType], bool],
    stored: str,
    where: str,
) -> dict[str, str]:
    """The column each camera states a per-row quantity in, for all or none.

    A directory that names it for some cameras and not the rest would read
    those at the window's own step and the rest at their own, which pairs one
    camera's instant with another's and shows up in no count. The arrow type is
    checked here as well: a frame index written as a float reads back through
    int() as a neighbouring frame, and a flag written as a number reads back
    through bool() as true for every row it was ever false at. A null reads
    back as row zero and as false, so a column holding one is refused too.
    """
    stated = [key for key in view_keys if f"{key}{suffix}" in table.schema.names]
    if stated and len(stated) != len(view_keys):
        raise ValueError(
            f"the cache at {where} names {suffix!r} for {len(stated)} of its "
            f"{len(view_keys)} cameras.\n"
            "Expected the column for every camera or for none.\n"
            "Rebuild the cache from a directory that states one per camera."
        )
    columns = {key: f"{key}{suffix}" for key in stated}
    for column in columns.values():
        found = table.schema.field(column).type
        if not holds(found):
            raise ValueError(
                f"the cache at {where} holds {column!r} as {found}.\n"
                f"Expected {stored}.\n"
                "Rebuild the cache from a directory that writes that type."
            )
        empty = int(table.column(column).null_count)
        if empty:
            raise ValueError(
                f"the cache at {where} leaves {column!r} null in {empty} rows.\n"
                "Expected one value per row.\n"
                "Rebuild the cache from a directory that states every row."
            )
    return columns


def read_frame_index(path: Path, *, episodes: int) -> dict[str, object]:
    """The frame index of one directory, checked against its episode count.

    Every camera names one file per kept episode, in kept order. A shorter list
    would silently pair a window with another episode's pictures, which is the
    one error in this module that produces a plausible sample.
    """
    index = json.loads(path.read_text(encoding="utf-8"))
    kind = str(index["kind"])
    if kind not in FRAME_KINDS:
        raise ValueError(
            f"{path} states frame kind {kind!r}.\n"
            f"Expected one of {FRAME_KINDS}.\n"
            "Rebuild the cache with this tree's cache stage."
        )
    for key, files in sorted(index["views"].items()):
        if len(files) != episodes:
            raise ValueError(
                f"{path} lists {len(files)} files for {key!r} and the manifest "
                f"declares {episodes} episodes.\n"
                "Expected one file per kept episode, in kept order.\n"
                "Rebuild the cache and the manifest from the same kept list."
            )
    return index


def held_bytes(path: Path) -> str:
    """How many bytes one file holds, said in words, or why that cannot be asked."""
    try:
        return f"{path.stat().st_size} bytes"
    except OSError as error:
        return f"no size: {error.strerror}"


def decode_image_bytes(payload: bytes) -> torch.Tensor:
    """One encoded picture -> [3, height, width] float in 0..1."""
    buffer = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    return decode_image(buffer, mode=ImageReadMode.RGB).float() / PIXEL_MAX


def indexed_timestamps(stream: av.video.stream.VideoStream) -> tuple[int, ...] | None:
    """One stream's own sample table as a timestamp table, or None.

    Opening an mp4 makes libavformat read the sample table into an index entry
    per frame, which costs the header rather than the stream. Those entries are
    timed by decoding time, so they are presentation times only where nothing
    is reordered, and a stream that reorders is refused here and scanned
    instead. An index shorter than the stream's own frame count describes only
    the frames it lists and is refused for the same reason.
    """
    if int(stream.codec_context.has_b_frames) != 0:
        return None
    entries = stream.index_entries
    if not entries or len(entries) != int(stream.frames):
        return None
    return tuple(sorted(entry.timestamp for entry in entries))


def stream_timestamps(container: av.container.InputContainer, path: Path) -> tuple[int, ...]:
    """Every frame's presentation timestamp in an open file's stream, in display order.

    Taken from the container's own sample table where that table is complete
    and nothing is reordered, and by demultiplexing the stream where it is not.
    Either way the timestamps are the ones the packets carry: the average rate
    a container reports is an estimate, and one stream of this corpus reports
    27100/3379 for a recording whose timestamps step by exactly an eighth of a
    second, which drifts a whole frame inside 250 of them.
    """
    stream = container.streams.video[0]
    indexed = indexed_timestamps(stream)
    stamps = indexed if indexed is not None else sorted(
        packet.pts for packet in container.demux(stream) if packet.pts is not None
    )
    if not stamps:
        raise ValueError(
            f"{path} carries no frame timestamps.\n"
            "Expected a stream whose packets are timed.\n"
            "Re-encode the episode, or cache its frames as images."
        )
    return tuple(stamps)


def frame_timestamps(path: Path) -> tuple[int, ...]:
    """Every frame's presentation timestamp in one file's stream, from an opening of its own."""
    with av.open(str(path)) as container:
        return stream_timestamps(container, path)


class FileCache:
    """What a process keeps per frame file, bounded by the bytes it holds.

    One store for a process rather than one per reader: a worker holds a reader
    for every directory its own draws reach, and this corpus has 569 of them,
    so a cap stated per reader would be that many times what it said. The least
    recently used entry goes first, and an entry that is gone costs the read
    that built it and nothing else, because what a key holds is a function of
    that key and of the file it names.
    """

    def __init__(self, budget: int):
        self.budget = int(budget)
        # Oldest first, so evicting is popping the front. The values carry
        # their own charge: what a table costs depends on how long it is, and
        # the store cannot ask the value again once it has dropped it.
        self.entries: OrderedDict[str, tuple[object, int]] = OrderedDict()
        self.held = 0
        self.hits = 0
        self.misses = 0
        self.evicted = 0

    def take(self, key: str) -> object | None:
        """What is held for one key, or None, counted either way.

        Taken out and put back rather than moved to the end: the two orderings
        are the same one, and this way nothing raises if the key is dropped
        between the lookup and the reordering.
        """
        held = self.entries.pop(key, None)
        if held is None:
            self.misses += 1
            return None
        self.hits += 1
        self.entries[key] = held
        return held[0]

    def put(self, key: str, value: object, cost: int) -> None:
        """Hold one value, dropping the oldest entries until it fits."""
        cost = max(int(cost), ENTRY_BYTES)
        older = self.entries.pop(key, None)
        if older is not None:
            self.held -= older[1]
        if cost > self.budget:
            return
        # Stopping at an empty store rather than trusting the running total: if
        # the two ever disagree, popping from nothing raises where the worst
        # that should happen is a store holding less than it could. A store
        # with no entries holds nothing, so the total is taken from it again.
        while self.held + cost > self.budget and self.entries:
            self.held -= self.entries.popitem(last=False)[1][1]
            self.evicted += 1
        if not self.entries:
            self.held = 0
        self.entries[key] = (value, cost)
        self.held += cost

    def clear(self) -> None:
        """Drop everything, leaving the counts alone."""
        self.entries.clear()
        self.held = 0


def cache_key(path: Path) -> str:
    """The name a file is held under: absolute, with . and .. taken out.

    A relative path names one file from one directory and another file from the
    next, so a store keyed by the string as written can hand back the first
    file's answer for the second. Symlinks are left alone: two names for one
    file cost two entries, which is memory rather than a wrong answer, and
    following them is a call on the shared mount for every frame read.
    """
    return os.path.abspath(path)


def cache_budget(name: str, default: int) -> int:
    """One store's budget in bytes, from the environment or the default."""
    stated = os.environ.get(name)
    if stated is None:
        return default
    try:
        budget = int(stated)
    except ValueError:
        budget = 0
    if budget <= 0:
        raise ValueError(
            f"{name} is {stated!r}.\n"
            "Expected a positive number of bytes.\n"
            f"Unset it for the default of {default} bytes, or state a size."
        )
    return budget


# One of each per process. A dataloader worker is forked from the process that
# built the dataset, so each worker ends up with its own store and its own
# budget rather than sharing one; nothing here is written by two processes, and
# there is no lock for a fork to inherit while it is held.
TIMESTAMPS = FileCache(cache_budget(TIMESTAMP_BUDGET_VAR, TIMESTAMP_BUDGET))
MIRRORS = FileCache(MIRROR_BUDGET)


def timestamps_of(path: Path) -> tuple[int, ...]:
    """One stream's timestamp table, from the process's store or from the file.

    The table a dropped entry is rebuilt from is the same table: it is read out
    of the packets of that one file, so which of the two a caller was handed
    can only be seen in how long the call took.
    """
    key = cache_key(path)
    held = TIMESTAMPS.take(key)
    if held is not None:
        return held
    stamps = frame_timestamps(path)
    TIMESTAMPS.put(key, stamps, TABLE_BYTES + len(stamps) * STAMP_BYTES)
    return stamps


def held_timestamps(container: av.container.InputContainer, path: Path) -> tuple[int, ...]:
    """timestamps_of for a file the caller has open: the store's table, or this opening's."""
    key = cache_key(path)
    held = TIMESTAMPS.take(key)
    if held is not None:
        return held
    stamps = stream_timestamps(container, path)
    TIMESTAMPS.put(key, stamps, TABLE_BYTES + len(stamps) * STAMP_BYTES)
    return stamps


# The converter from a decoded picture to rgb24, one per process, made in the
# process that uses it so a forked worker never inherits its parent's. A
# frame's own to_ndarray makes a converter, and a scaling context, for every
# frame; this one keeps its context and rebuilds it when the input changes. One
# thread: a worker converts one small picture at a time, and the automatic
# count starts a pool sized to the node's cores for each context it builds.
_converter: tuple[int, VideoReformatter] | None = None
CONVERTER_THREADS = 1


def rgb_array(frame: av.VideoFrame) -> np.ndarray:
    """One decoded picture -> [height, width, 3] uint8, as frame.to_ndarray(format='rgb24')."""
    global _converter
    if _converter is None or _converter[0] != os.getpid():
        _converter = (os.getpid(), VideoReformatter())
    return _converter[1].reformat(frame, format="rgb24", threads=CONVERTER_THREADS).to_ndarray()


def check_frame_index(path: Path, index: int, timestamps: tuple[int, ...]) -> None:
    if not 0 <= index < len(timestamps):
        raise IndexError(
            f"frame index {index} of {len(timestamps)} in {path}.\n"
            "Expected an index inside the stream.\n"
            "Rebuild the cache: the row count and the stream disagree."
        )


def decode_video_frame(path: Path, index: int, timestamps: tuple[int, ...]) -> torch.Tensor:
    """Frame `index` of one episode's stream -> [3, height, width] float."""
    check_frame_index(path, index, timestamps)
    return decode_video_frames(path, (index,), timestamps)[0]


def decode_video_frames(path: Path, indices: Sequence[int],
                        timestamps: tuple[int, ...] | None = None) -> list[torch.Tensor]:
    """Frames `indices` of one episode's stream, all from one opening of its file.

    timestamps is the stream's table when the caller holds it; without it the
    table is the process store's, or this same opening's, which the store then
    keeps. Each frame seeks to the keyframe at or before its own timestamp and
    decodes forward to it, so the cost is one group of pictures rather than the
    whole episode, and the frame returned is the one whose timestamp was asked
    for rather than the one nearest an estimate. A seek flushes the decoder, so
    a second frame decodes as it would from a fresh opening. The decoder runs on
    the calling thread: a pool per opened file is a pool per frame.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if timestamps is None:
            timestamps = held_timestamps(container, path)
        for index in indices:
            check_frame_index(path, index, timestamps)
        stream.codec_context.thread_type = "NONE"
        stream.codec_context.thread_count = 1
        return [_decode_at(container, stream, path, index, timestamps[index])
                for index in indices]


def _decode_at(container: av.container.InputContainer, stream: av.video.stream.VideoStream,
               path: Path, index: int, target: int) -> torch.Tensor:
    container.seek(target, stream=stream, backward=True, any_frame=False)
    for frame in container.decode(stream):
        if frame.pts == target:
            # from_numpy shares the converted frame's buffer and float() copies
            # it, so nothing aliases the decoder's memory once this returns.
            picture = torch.from_numpy(rgb_array(frame))
            return picture.permute(2, 0, 1).float() / PIXEL_MAX
        if frame.pts is not None and frame.pts > target:
            break
    raise ValueError(
        f"{path} decodes no frame at timestamp {target}, index {index}.\n"
        "Expected the timestamp the packet headers listed.\n"
        "Re-encode the episode, or cache its frames as images."
    )


class CacheReader:
    """One directory's cache, addressed by kept position and frame index.

    arrow           the numeric columns of every kept episode, in kept order
    frames          the frame index, one file per camera per kept episode
    episode_starts  the row boundaries of those episodes, episodes + 1 long
    parts           the stored v7 columns the directory's layouts read and the
                    stamp they carry (StoredParts), or None for a directory
                    whose layouts read none; checked when the reader opens

    Holds the arrow table memory-mapped and opens a frame file per call. A
    dataloader worker pays for the directories its own draws reach, which is
    why the dataset opens these lazily. A stream's timestamp table and the copy
    of a file this node opens it from are kept in the process's own stores,
    which readers share and which are bounded: a worker of this corpus reaches
    463,920 files over a run and holding every table would be 36.7 GB of them.
    """

    def __init__(
        self,
        *,
        root: Path,
        arrow_path: str,
        frames_path: str,
        episode_starts: torch.Tensor,
        state_column: str,
        action_column: str,
        valid_column: str | None,
        view_keys: tuple[str, ...],
        parts: StoredParts | None = None,
    ):
        self.root = Path(root)
        self.arrow_path = str(arrow_path)
        self.state_column = str(state_column)
        self.action_column = str(action_column)
        self.valid_column = None if valid_column is None else str(valid_column)
        self.starts = episode_starts.to(torch.int64)
        self.table = feather.read_table(self.root / arrow_path, memory_map=True)
        self.frames_index = read_frame_index(
            self.root / frames_path, episodes=int(self.starts.numel()) - 1
        )
        self.kind = str(self.frames_index["kind"])
        self.files: dict[str, list[str | None]] = {
            key: list(files) for key, files in self.frames_index["views"].items()
        }
        missing = sorted(set(view_keys) - set(self.files))
        if missing:
            raise ValueError(
                f"the frame index at {frames_path} has no camera {missing[0]!r}.\n"
                f"Expected the manifest's cameras {sorted(view_keys)}.\n"
                "Rebuild the cache from the manifest that names these cameras."
            )
        self.view_keys = tuple(view_keys)
        self._mirror_root = os.environ.get(MIRROR_ROOT_VAR) or None
        self.mirror_hits = 0
        self.mirror_misses = 0
        self.view_frames = per_view_columns(
            self.table, self.view_keys, suffix=VIEW_FRAME_SUFFIX,
            holds=pa.types.is_integer, stored="an integer column", where=arrow_path,
        )
        self.view_flags = per_view_columns(
            self.table, self.view_keys, suffix=FRAME_VALID_SUFFIX,
            holds=pa.types.is_boolean, stored="a boolean column", where=arrow_path,
        )
        self._columns: dict[str, np.ndarray] = {}
        self._episode_ids: np.ndarray | None = None
        named = [self.state_column, self.action_column]
        if self.valid_column is not None:
            named.append(self.valid_column)
        for name in named:
            if name not in self.table.schema.names:
                raise ValueError(
                    f"the cache at {arrow_path} holds no column {name!r}.\n"
                    f"Expected the manifest's columns; it has "
                    f"{self.table.schema.names[:4]}.\n"
                    "Rebuild the cache, or correct the entry's columns block."
                )
        if int(self.starts[-1]) != self.table.num_rows:
            raise ValueError(
                f"the cache at {arrow_path} holds {self.table.num_rows} rows and "
                f"its boundaries end at {int(self.starts[-1])}.\n"
                "Expected the arrow file and episode_starts to cover the same rows.\n"
                "Rebuild both from the same kept list."
            )
        self.parts = parts
        self.part_columns: dict[str, int] = {}
        if parts is not None and parts.columns:
            self._check_parts(parts)
            self.part_columns = dict(parts.columns)

    def _check_parts(self, parts: StoredParts) -> None:
        """Refuse stored v7 columns that are missing, mistyped, null or unstamped.

        Each column the layouts read must be in the arrow as a list of floats
        with no null cell (a fixed-size list of the expected width when the
        type states one), and the arrow must carry its writer's stamp: the v7
        ingest stamp, or the sidecar's, which also has to hold every kept
        episode with the cache's own rows.
        """
        where = self.arrow_path
        for name, width in sorted(parts.columns.items()):
            if name not in self.table.schema.names:
                raise _refuse_stamp(where, f"holds no column {name!r}, which the directory's "
                                           "layouts read", "the stored v7 column of every "
                                                           "declared part")
            found = self.table.schema.field(name).type
            listed = pa.types.is_list(found) or pa.types.is_large_list(found) \
                or pa.types.is_fixed_size_list(found)
            if not listed or not pa.types.is_floating(found.value_type):
                raise ValueError(
                    f"the cache at {where} holds {name!r} as {found}.\n"
                    f"Expected a list of floats, {width} per row.\n"
                    "Rebuild the cache from a writer that stores the v7 column."
                )
            if pa.types.is_fixed_size_list(found) and int(found.list_size) != int(width):
                raise ValueError(
                    f"the cache at {where} holds {name!r} {found.list_size} wide.\n"
                    f"Expected {width}: one block per arm of the directory, in arm order.\n"
                    "Rebuild the cache, or correct the entry's arms."
                )
            empty = int(self.table.column(name).null_count)
            if empty:
                raise ValueError(
                    f"the cache at {where} leaves {name!r} null in {empty} rows.\n"
                    "Expected a value on every row, NaN where nothing was recorded.\n"
                    "Rebuild the cache from a writer that states every row."
                )
        if parts.origin == "ingest":
            check_ingest_stamp(self.table, where)
            return
        stamp = check_sidecar_stamp(self.table, parts, where)
        check_sidecar_episodes(stamp, self.episode_ids(), self.starts, where)

    def bounds(self, episode: int) -> tuple[int, int]:
        """The first and last row of one kept episode, in the arrow table."""
        if not 0 <= episode < int(self.starts.numel()) - 1:
            raise IndexError(
                f"episode {episode} of {int(self.starts.numel()) - 1} kept.\n"
                "Expected a kept position, not the index the source wrote.\n"
                "The draw gives kept positions; pass one of those."
            )
        return int(self.starts[episode]), int(self.starts[episode + 1])

    def episode_ids(self) -> np.ndarray:
        """[episodes] int64: the source episode each kept position was cut from.

        Read out of EPISODE_ID_COLUMN at every episode's first and last row,
        which must agree, or the boundaries put one source episode's rows under
        another's position. Every other per-position record of the directory is
        joined by kept position with nothing in it naming the episode, so this
        is what R16 holds them against.
        """
        if self._episode_ids is None:
            self._episode_ids = self._read_episode_ids()
        return self._episode_ids

    def _read_episode_ids(self) -> np.ndarray:
        where = self.arrow_path
        if EPISODE_ID_COLUMN not in self.table.schema.names:
            raise ValueError(
                f"{refusal_prefix(16)} the cache at {where} holds no "
                f"{EPISODE_ID_COLUMN!r} column.\n"
                "Expected the source episode id LeRobot writes on every row, which is how "
                "a kept position names the episode it holds.\n"
                "Rebuild the cache from parquet that carries it."
            )
        column = self.table.column(EPISODE_ID_COLUMN)
        if not pa.types.is_integer(column.type) or column.null_count:
            raise ValueError(
                f"{refusal_prefix(16)} the cache at {where} holds {EPISODE_ID_COLUMN!r} as "
                f"{column.type} with {column.null_count} nulls.\n"
                "Expected an integer on every row.\n"
                "Rebuild the cache from parquet that writes one."
            )
        starts = self.starts.numpy()
        first = np.asarray(column.take(pa.array(starts[:-1])).to_numpy(), dtype=np.int64)
        last = np.asarray(column.take(pa.array(starts[1:] - 1)).to_numpy(), dtype=np.int64)
        split = np.flatnonzero(first != last)
        if split.size:
            position = int(split[0])
            raise ValueError(
                f"{refusal_prefix(16)} the cache at {where}: kept position {position} "
                f"opens on source episode {int(first[position])} and closes on "
                f"{int(last[position])}.\n"
                "Expected one source episode between two boundaries.\n"
                "Rebuild episode_starts and the arrow file from the same kept list."
            )
        return first

    def rows(self, episode: int, start: int, length: int) -> dict[str, torch.Tensor]:
        """The numeric columns of one window, padded at the episode's end.

        valid says a step holds an action: it is false past the last frame of
        the episode, and false again where the source's own validity column
        says the row records no action. A window running past the last frame is
        never redrawn: a redraw reaching back to the mixture moves mass towards
        the directories with the longest episodes. Each stored v7 column the
        layouts read comes back under its own name, [length, width], NaN past
        the episode's end.
        """
        first, last = self.bounds(episode)
        span = last - first
        if not 0 <= start < span:
            raise IndexError(
                f"start {start} in an episode of {span} rows.\n"
                "Expected a start inside the episode.\n"
                "The draw takes the start from the episode's own length."
            )
        take = min(int(length), span - start)
        window = self.table.slice(first + start, take)
        valid = torch.zeros(int(length), dtype=torch.bool)
        valid[:take] = True
        if self.valid_column is not None:
            valid[:take] &= self._recorded(window)
        rows = {
            "state": _pad_rows(_matrix(window.column(self.state_column)), int(length)),
            "action": _pad_rows(_matrix(window.column(self.action_column)), int(length)),
            "valid": valid,
        }
        for name, width in self.part_columns.items():
            values = _matrix(window.column(name))
            if int(values.shape[1]) != width:
                raise ValueError(
                    f"the cache at {self.arrow_path} holds {int(values.shape[1])} values of "
                    f"{name!r} on row {first + start}.\n"
                    f"Expected {width}: one block per arm of the directory, in arm order.\n"
                    "Rebuild the cache, or correct the entry's arms."
                )
            rows[name] = _pad_rows(values, int(length), fill=float("nan"))
        return rows

    def _recorded(self, window: pa.Table) -> torch.Tensor:
        """The source's own per-row action flag, as one bool per window row."""
        flags = _matrix(window.column(self.valid_column))
        if int(flags.shape[1]) != 1:
            raise ValueError(
                f"column {self.valid_column!r} is {int(flags.shape[1])} wide.\n"
                "Expected one flag per row.\n"
                "Name a single-column flag, or drop valid_column from the layout."
            )
        return flags.reshape(-1) != 0.0

    def present_views(self, episode: int, step: int) -> frozenset[str]:
        """Which cameras have a frame at one step, without decoding any of them.

        The one producer of that statement: frames decodes these keys and no
        others, so a caller that needs to know before it has the pictures, or
        without paying for them, asks here rather than reading the flag column
        a second way. A directory that states no flag returns every camera.
        """
        first, last = self.bounds(episode)
        if not 0 <= step < last - first:
            raise IndexError(
                f"step {step} in an episode of {last - first} rows.\n"
                "Expected a step inside the episode.\n"
                "Read the frame of the window's first step."
            )
        return frozenset(
            key for key in self.view_keys if self._view_present(key, first + step)
        )

    def frames(self, episode: int, step: int) -> dict[str, torch.Tensor]:
        """One step's pictures, keyed by the manifest's camera keys.

        A camera the directory states has no frame at this row is left out
        rather than given the frame beside it. The caller places what it is
        handed and marks the rest of its slots absent, so that camera reaches
        the sample the way a directory that never had it does.
        """
        return self.frames_at(episode, (step,))[0]

    def frames_at(self, episode: int, steps: Sequence[int]) -> list[dict[str, torch.Tensor]]:
        """Several steps' pictures of one episode: per step, what frames returns for it.

        A sample reads its step t and its history step p together, so each
        camera's file is opened once for both, and a parquet holding several
        cameras is opened once for all of them, rather than once per camera per
        step. Every step is checked against the episode before anything is read.
        """
        steps = [int(step) for step in steps]
        present = [self.present_views(episode, step) for step in steps]
        first, _ = self.bounds(episode)
        # file -> (camera, position in steps, frame of that camera's own file)
        reads: dict[str, list[tuple[str, int, int]]] = {}
        for key in self.view_keys:
            path = self.files[key][episode]
            if path is None:
                raise ValueError(
                    f"kept episode {episode} has no file for camera {key!r}.\n"
                    "Expected one file per kept episode; the filter stage keeps "
                    "only episodes whose frames are there.\n"
                    "Drop the episode in the filter stage and rebuild the cache."
                )
            for position, step in enumerate(steps):
                if key in present[position]:
                    frame = self._view_frame(key, first + step, step)
                    reads.setdefault(path, []).append((key, position, frame))
        decoded: list[dict[str, torch.Tensor]] = [{} for _ in steps]
        for path, wanted in reads.items():
            for (key, position, _), picture in zip(wanted, self._file_frames(self.root / path,
                                                                             wanted)):
                decoded[position][key] = picture
        return [{key: one[key] for key in self.view_keys if key in one} for one in decoded]

    def _view_frame(self, key: str, row: int, step: int) -> int:
        """Which frame of one camera's own file a row of this cache reads.

        The window's own step, for a directory whose cameras were recorded on
        one clock, and the frame that camera contributed to this row for one
        that states a frame index per camera.
        """
        column = self.view_frames.get(key)
        if column is None:
            return step
        return int(self._column(column)[row])

    def _view_present(self, key: str, row: int) -> bool:
        """Whether one camera has a frame at one row of this cache.

        True throughout for a directory that states nothing, which is every
        directory whose cameras share a clock: there a row is a frame of each
        of them. A directory built over cameras on separate clocks states one
        flag per camera per row, and a false one means the nearest frame that
        camera has is further away in time than the tolerance the join kept.
        """
        column = self.view_flags.get(key)
        if column is None:
            return True
        return bool(self._column(column)[row])

    def _column(self, name: str) -> np.ndarray:
        """One per-camera column of this directory, read once and kept.

        Taken whole rather than a row at a time because every sample this
        directory serves reads one row of it: the arrow file is memory-mapped,
        so what this holds is one array per camera per column, which is four
        bytes a row for an index and one for a flag.
        """
        if name not in self._columns:
            self._columns[name] = np.asarray(self.table.column(name).to_numpy())
        return self._columns[name]

    def _local(self, path: Path) -> Path:
        """The copy of one file to open: this node's own when it holds a whole one.

        A miss is counted and answered with the shared path, so a node whose
        mirror is half populated trains correctly and only more slowly. The
        first is logged, because a mirror nobody notices is missing is a mirror
        that quietly buys nothing.

        The decision is remembered per file in a store this process bounds, so
        a file whose decision has been dropped is decided again and counted
        again: the counters below say how many decisions went each way, not how
        many distinct files did.
        """
        if self._mirror_root is None:
            return path
        # Under the root as well as the file: the store belongs to the process
        # and the answer does not, so a reader holding one root must not be
        # handed the copy another root chose. A path holds no null byte.
        key = f"{self._mirror_root}\0{cache_key(path)}"
        known = MIRRORS.take(key)
        if known is not None:
            return known
        shared = path.resolve()
        chosen = path
        try:
            candidate = Path(self._mirror_root) / str(shared).lstrip("/")
            if candidate.stat().st_size == shared.stat().st_size:
                chosen = candidate
                self.mirror_hits += 1
            else:
                self.mirror_misses += 1
        except OSError:
            self.mirror_misses += 1
        global _warned_absent_mirror
        if chosen is path and not _warned_absent_mirror:
            _warned_absent_mirror = True
            logger.warning("no whole mirrored copy of %s under %s; reading the "
                           "shared mount", shared, self._mirror_root)
        MIRRORS.put(key, chosen, MIRROR_ENTRY_BYTES)
        return chosen

    def _file_frames(self, path: Path, wanted: list[tuple[str, int, int]]) -> list[torch.Tensor]:
        """The pictures one episode file holds for (camera, position, frame) reads, in order.

        A decoder that cannot read the file is answered with the file's own
        size, because what it raises describes the bytes it was handed and not
        the file they came from: an empty file reports a missing header, which
        reads as a damaged video rather than as no video at all, and sends
        whoever meets it looking for a codec. A file that is not there keeps
        its own error, which the callers of this module tell apart already.
        """
        path = self._local(path)
        try:
            if self.kind != "mp4":
                cells = _image_cells(path, [(key, frame) for key, _, frame in wanted])
                return [decode_image_bytes(payload) for payload in cells]
            return decode_video_frames(path, [frame for _, _, frame in wanted])
        except FileNotFoundError:
            raise
        except (av.FFmpegError, OSError, RuntimeError, ValueError) as error:
            cameras = list(dict.fromkeys(repr(key) for key, _, _ in wanted))
            frames = list(dict.fromkeys(str(frame) for _, _, frame in wanted))
            raise ValueError(
                f"{path} does not read as camera{'s' * (len(cameras) > 1)} "
                f"{', '.join(cameras)} at step{'s' * (len(frames) > 1)} {', '.join(frames)}: "
                f"{str(error).splitlines()[0]}\n"
                f"Expected a file a decoder can open; it holds {held_bytes(path)}.\n"
                "Re-encode that episode, or record the camera as having no file "
                "and rebuild the cache."
            ) from error


def _matrix(column: pa.ChunkedArray) -> torch.Tensor:
    """One numeric column of a window -> [rows, width] float32.

    A list of floats, or a plain float, flag or small integer column, with no
    null anywhere is read from arrow's own buffers: the numbers the python path
    gives, 2.5-5.6 times faster a call on the mini corpus (0.02-0.07 ms against
    0.11-0.17, three passes of 14,800 column reads over 3,400 windows with the
    two paths alternated), and rows of more than one width are refused rather
    than reshaped. Anything else takes the values through python, the
    definition both are tested on.
    """
    array = _numeric_rows(column)
    if array is None:
        array = np.asarray(column.to_pylist(), dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
    return torch.from_numpy(array)


# The integers a float64 holds exactly. A python int reaches float32 through a
# float64, an int64 array straight; the two round alike below this.
EXACT_INTEGER = 2 ** 53


def _numeric_rows(column: pa.ChunkedArray) -> np.ndarray | None:
    """[rows, width] float32 copied out of the column's buffers, or None for the python path."""
    kind = column.type
    rows = len(column)
    if rows == 0 or column.null_count:
        return None
    if pa.types.is_floating(kind) or pa.types.is_boolean(kind) or pa.types.is_integer(kind):
        values = column.to_numpy()
        if pa.types.is_integer(kind) and int(np.abs(values).max()) >= EXACT_INTEGER:
            return None
        return values.astype(np.float32).reshape(rows, 1)
    listed = pa.types.is_list(kind) or pa.types.is_large_list(kind) \
        or pa.types.is_fixed_size_list(kind)
    if not listed or not pa.types.is_floating(kind.value_type):
        return None
    if pa.types.is_fixed_size_list(kind):
        width = int(kind.list_size)
    else:
        widths = pc.list_value_length(column).to_numpy()
        if int(widths.min()) != int(widths.max()):
            raise ValueError(
                f"a window of column type {kind} holds rows of {int(widths.min())} to "
                f"{int(widths.max())} values.\n"
                "Expected one width on every row of a column.\n"
                "Rebuild the cache from a writer that states every row in full."
            )
        width = int(widths[0])
    parts = [chunk.flatten() for chunk in column.chunks if len(chunk)]
    if any(part.null_count for part in parts):
        return None
    flat = parts[0].to_numpy(zero_copy_only=False) if len(parts) == 1 \
        else np.concatenate([part.to_numpy(zero_copy_only=False) for part in parts])
    return flat.astype(np.float32).reshape(rows, width)


def _pad_rows(window: torch.Tensor, length: int, fill: float = 0.0) -> torch.Tensor:
    """A short window -> `length` rows, the tail fill (0, marked in valid; NaN for a part)."""
    if int(window.shape[0]) == length:
        return window
    padded = window.new_full((length, int(window.shape[1])), fill)
    padded[: int(window.shape[0])] = window
    return padded


def _image_cells(path: Path, cells: Sequence[tuple[str, int]]) -> list[bytes]:
    """The encoded pictures at (column, row) cells of one episode's parquet, in order.

    Opens the file once and reads each row group holding a wanted row once,
    with every wanted column of that group, so a sample's cameras at t and at p
    cost one opening and one read per group rather than one of each per camera
    per step. A frame still costs one group of an episode instead of every
    picture in it, and the bytes are the ones a read of that one column returns.
    """
    with pq.ParquetFile(path) as handle:
        counts = [handle.metadata.row_group(group).num_rows
                  for group in range(handle.num_row_groups)]
        located = []
        columns: dict[int, list[str]] = {}
        for key, step in cells:
            offset = 0
            for group, count in enumerate(counts):
                if step < offset + count:
                    break
                offset += count
            else:
                raise IndexError(
                    f"{path} holds {offset} rows and row {step} was asked for.\n"
                    "Expected the episode's own file to hold every row of that episode.\n"
                    "Rebuild the cache: the frame index and the row counts disagree."
                )
            located.append((key, step, group, offset))
            if key not in columns.setdefault(group, []):
                columns[group].append(key)
        groups = {group: handle.read_row_group(group, columns=keys)
                  for group, keys in columns.items()}
    payloads = []
    for key, step, group, offset in located:
        payload = groups[group].column(key)[step - offset].as_py()
        if isinstance(payload, dict):
            payload = payload["bytes"]
        if not isinstance(payload, (bytes, bytearray)):
            raise ValueError(
                f"{path} column {key!r} row {step} holds {type(payload).__name__}.\n"
                "Expected encoded image bytes.\n"
                "Cache this camera as video, or store the picture as bytes."
            )
        payloads.append(bytes(payload))
    return payloads
