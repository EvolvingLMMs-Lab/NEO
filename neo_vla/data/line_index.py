"""Where the records NEO's trainer reads start in each MT_Und annotation file.

NEO reads an entry's file line by line from line 0: len = ceil(length *
repeat_time) positions an epoch, position p being line p mod L, the file
cycled only when repeat_time is above one (neo_train_plus
multimodal_dataset.py:164, :1127-1171), entries weighted by len. A byte-offset
draw weighs a record by the line before it and reaches lines NEO never read:
4193 of 4531 MT_Und entries have repeat_time below one, so NEO reads 30.5M of
their 182M lines. This builds, once, the byte offset of each position NEO reads.

On disk: offsets.npy, every entry's line starts end to end as uint64, and
index.json, each entry's run and the size and mtime of its file. index.json is
written last, so a build that stopped part way is never read as a finished one.
"""
from __future__ import annotations

import json
import math
import os
from multiprocessing import get_context
from pathlib import Path

import numpy as np

INDEX_FORMAT = 1
INDEX_FILE = "index.json"
OFFSETS_FILE = "offsets.npy"
# Bytes read at a time while looking for line ends.
SCAN_CHUNK = 1 << 26


def entry_need(entry: dict) -> int:
    """NEO's len for an entry: ceil(length * repeat_time) line positions an epoch."""
    return math.ceil(float(entry["length"]) * float(entry["repeat_time"]))


def scan_line_starts(path: str | Path, need: int, whole: bool) -> tuple[np.ndarray, bool]:
    """A file -> (the byte offset each line starts at, whether the file ended).

    A line is what readline returns: it ends after a newline or at the end of
    the file, so a last line with no newline counts and a trailing newline adds
    none. Stops once need starts are found unless whole is set.
    """
    found: list[np.ndarray] = []
    total = 0
    with open(path, "rb", buffering=0) as handle:
        size = os.fstat(handle.fileno()).st_size
        if size > 0:
            found.append(np.zeros(1, dtype=np.uint64))
            total = 1
        base = 0
        while (whole or total < need) and base < size:
            chunk = handle.read(SCAN_CHUNK)
            if not chunk:
                break
            ends = np.flatnonzero(np.frombuffer(chunk, dtype=np.uint8) == 10)
            starts = (ends + base + 1).astype(np.uint64)
            starts = starts[starts < size]
            found.append(starts)
            total += int(starts.shape[0])
            base += len(chunk)
    starts = np.concatenate(found) if found else np.zeros(0, dtype=np.uint64)
    ended = whole or total < need
    return (starts if whole else starts[:need]), ended


def scan_entry(job: tuple[str, str, int, bool]) -> tuple[str, np.ndarray, bool, int, int]:
    """One registry entry -> (key, line starts, ended, size, mtime_ns), for a pool."""
    key, annotation, need, whole = job
    before = os.stat(annotation)
    starts, ended = scan_line_starts(annotation, need, whole)
    after = os.stat(annotation)
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"{annotation} changed while it was read; build the index again")
    return key, starts, ended, int(after.st_size), int(after.st_mtime_ns)


def build_line_index(
    registry: str | Path, out: str | Path, *, workers: int = 8, log=print
) -> dict:
    """Read every entry of a registry once and write its index under out.

    Only the positions NEO reads are indexed: the first need lines of a file
    whose repeat_time is at most one, every line of one that is cycled.
    """
    listed = json.loads(Path(registry).read_text(encoding="utf-8"))
    jobs = []
    for key, entry in listed.items():
        need = entry_need(entry)
        if need > 0:
            jobs.append((key, str(entry["annotation"]), need, float(entry["repeat_time"]) > 1.0))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runs: dict[str, np.ndarray] = {}
    records: dict[str, dict] = {}
    if int(workers) > 1:
        pool = get_context("fork").Pool(int(workers))
        scanned = pool.imap_unordered(scan_entry, jobs, chunksize=1)
    else:
        pool, scanned = None, map(scan_entry, jobs)
    try:
        for done, (key, starts, ended, size, mtime_ns) in enumerate(scanned, start=1):
            entry = listed[key]
            need = entry_need(entry)
            count = int(starts.shape[0])
            cycles = math.ceil(float(entry["repeat_time"]))
            records[key] = {
                "annotation": str(entry["annotation"]), "size": size, "mtime_ns": mtime_ns,
                "need": need, "count": count, "used": min(need, cycles * count),
                "ended": bool(ended),
            }
            runs[key] = starts
            if done % 200 == 0 or done == len(jobs):
                log(f"line index: {done} of {len(jobs)} entries read")
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    start = 0
    for key in listed:
        if key in records:
            records[key]["start"] = start
            start += records[key]["count"]
    offsets = np.concatenate([runs[key] for key in listed if key in runs]) if runs else (
        np.zeros(0, dtype=np.uint64))
    scratch = out / (OFFSETS_FILE + ".partial.npy")
    np.save(scratch, offsets.astype(np.uint64))
    os.replace(scratch, out / OFFSETS_FILE)
    index = {"format": INDEX_FORMAT, "registry": str(registry), "offsets": int(offsets.shape[0]),
             "entries": records}
    partial = out / (INDEX_FILE + ".partial")
    partial.write_text(json.dumps(index), encoding="utf-8")
    os.replace(partial, out / INDEX_FILE)
    return index


class LineIndex:
    """A built index, read back: an entry's run of line starts and how to draw from it."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        path = self.directory / INDEX_FILE
        if not path.is_file():
            raise ValueError(
                f"{path} does not exist.\n"
                "Expected the MT_Und line index: where each record NEO reads starts.\n"
                "Build it once with tools/build_mtund_line_index.py --registry <registry> "
                "--out <directory>."
            )
        index = json.loads(path.read_text(encoding="utf-8"))
        if index.get("format") != INDEX_FORMAT:
            raise ValueError(
                f"{path} is format {index.get('format')!r}, expected {INDEX_FORMAT}.\n"
                "Build the index again with this tree's tools/build_mtund_line_index.py."
            )
        self.records: dict[str, dict] = index["entries"]
        self.total = int(index["offsets"])
        # Opened on first use inside the worker: a memory map pickled into a
        # spawned worker is a copy, and one opened here is shared by a fork.
        self.offsets: np.ndarray | None = None

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["offsets"] = None
        return state

    def check(self, key: str, entry: dict) -> dict:
        """The index's record for a registry entry, or a raise when it is stale."""
        record = self.records.get(key)
        if record is None:
            raise ValueError(
                f"the line index in {self.directory} has no entry {key!r}.\n"
                "Expected one per registry entry with positive length and repeat_time.\n"
                "Build the index from the registry this run reads."
            )
        found = os.stat(entry["annotation"])
        stated = (str(entry["annotation"]), entry_need(entry), int(found.st_size),
                  int(found.st_mtime_ns))
        held = (record["annotation"], record["need"], record["size"], record["mtime_ns"])
        if stated != held:
            raise ValueError(
                f"the line index entry {key!r} was built for {held} (path, need, size, "
                f"mtime_ns); the registry and the file now say {stated}.\n"
                "Expected them equal: an offset into a changed file lands mid-record.\n"
                "Build the index again."
            )
        return record

    def offset(self, record: dict, position: int) -> int:
        """Line position p of an entry -> the byte its line starts at (line p mod count)."""
        if self.offsets is None:
            self.offsets = np.load(self.directory / OFFSETS_FILE, mmap_mode="r")
            if int(self.offsets.shape[0]) != self.total:
                raise ValueError(
                    f"{self.directory / OFFSETS_FILE} holds {self.offsets.shape[0]} offsets; "
                    f"index.json says {self.total}.\nBuild the index again."
                )
        return int(self.offsets[record["start"] + position % record["count"]])
