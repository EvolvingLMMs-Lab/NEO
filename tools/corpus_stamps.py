"""The v7 stamps on stored corpus artefacts, as the manifest and cache stages hold them.

An ingest output (new-data parquet) states in its meta/info.json that it is
the v7 layout and which record stamps it was written under; a stage-1 sidecar
carries its writer's stamp in its parquet footer. Both stages read them here,
so a v6 output, an unstamped one or a sidecar from another writer is refused
by one rule (R22) and never read as "not recorded". The cache
stage copies the checked stamps into the arrow file for the loader
(neo_vla.data.reader checks them again when it opens the cache).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from neo_vla.constants import (
    ACTION_STAMP,
    BODY_STAMP,
    INGEST_LAYOUT,
    INGEST_LAYOUT_FIELD,
    OLD_PREFIX_LAYOUTS,
    STATE_STAMP,
    refusal_prefix,
)
from neo_vla.data.reader import (
    INGEST_STAMP,
    SIDECAR_FORMAT,
    SIDECAR_PINS,
    SIDECAR_STAMP_KEY,
    SIDECAR_TOOL_VERSION,
)

# The record names an ingest output's meta/info.json carries beside its
# ingest_layout, flat (the reader's INGEST_STAMP, which the cache stage copies).
# One spelling: an older nested ingest_records form is refused by name. The
# sidecar footer's stamps are its own block.
RECORD_FIELDS = ("action_record", "state_record", "body_record")
NESTED_RECORDS = "ingest_records"
RECORD_STAMPS = {"action": ACTION_STAMP, "state": STATE_STAMP, "body": BODY_STAMP}


def refuse_stamp(name: str, said: str, expected: str, fix: str) -> ValueError:
    """R22: a directory whose v7 columns carry no current v7 stamp."""
    return ValueError(
        f"{refusal_prefix(22)} {name} {said}\n"
        f"expected {expected}\n"
        f"{fix}"
    )


def ingest_stamp(info: dict[str, Any], name: str) -> dict[str, Any]:
    """An ingest output's v7 stamp from its meta/info.json, in the form the cache copies.

    ingest_layout must be v7 (a v6 output is named), and the three record
    names this tree's, stated flat; the nested ingest_records
    spelling is refused by name. Returns the reader's form,
    neo_vla.data.reader.INGEST_STAMP.
    """
    layout = info.get(INGEST_LAYOUT_FIELD)
    if layout in OLD_PREFIX_LAYOUTS:
        raise refuse_stamp(name, f"meta/info.json states {INGEST_LAYOUT_FIELD} {layout!r}, a v6 "
                                 "ingest output", f"{INGEST_LAYOUT!r}",
                           "re-emit the directory in v7 (python -m tools.ingest reemit)")
    if layout != INGEST_LAYOUT:
        raise refuse_stamp(name, f"meta/info.json states {INGEST_LAYOUT_FIELD} {layout!r}",
                           f"{INGEST_LAYOUT!r}, the v7 writer's stamp",
                           "convert or re-emit the directory with the v7 writer")
    if NESTED_RECORDS in info:
        raise refuse_stamp(name, f"meta/info.json states {NESTED_RECORDS}, the record stamps "
                                 "nested as v7-wp7 wrote them before integration",
                           f"{list(RECORD_FIELDS)} flat",
                           "re-emit the directory with this tree's v7 writer")
    flat = {field: info[field] for field in RECORD_FIELDS if field in info}
    if not flat:
        raise refuse_stamp(name, "meta/info.json states no record stamps",
                           f"{list(RECORD_FIELDS)}",
                           "convert or re-emit the directory with the v7 writer")
    wanted = {field: INGEST_STAMP[field] for field in RECORD_FIELDS}
    if flat != wanted:
        raise refuse_stamp(name, f"meta/info.json states records {flat}", f"{wanted}",
                           "re-emit the directory with this tree's v7 writer")
    return dict(INGEST_STAMP)


def read_sidecar_stamp(path: Path) -> dict[str, Any]:
    """A sidecar's footer stamp, read without its rows; R22 when there is none."""
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    if SIDECAR_STAMP_KEY not in metadata:
        raise refuse_stamp(str(path), f"carries no {SIDECAR_STAMP_KEY.decode()} stamp",
                           f"a sidecar written by tools/sidecars ({SIDECAR_FORMAT})",
                           "rebuild the sidecar with the sidecar writer")
    return json.loads(metadata[SIDECAR_STAMP_KEY])


def check_sidecar_stamp(stamp: dict[str, Any], name: str, columns: list[str],
                        where: str) -> dict[str, str]:
    """Refuse a sidecar stamp from another writer, layout or entry; return its pins.

    The stamp must name this format, the v7 ingest layout (v6 named), the
    sidecar writer's version the reader takes (SIDECAR_TOOL_VERSION, not the
    stage tool's own constant), the three record stamps and this entry, and
    must have written every column the layouts read.
    """
    if stamp.get("ingest_layout") in OLD_PREFIX_LAYOUTS:
        raise refuse_stamp(where, f"states ingest_layout {stamp.get('ingest_layout')!r}, a v6 "
                                  "sidecar", f"{INGEST_LAYOUT!r}", "rebuild the sidecar")
    expected = {"format": SIDECAR_FORMAT, "ingest_layout": INGEST_LAYOUT,
                "tool_version": SIDECAR_TOOL_VERSION, "entry": name, "stamps": RECORD_STAMPS}
    for field, value in expected.items():
        if stamp.get(field) != value:
            raise refuse_stamp(where, f"states {field} {stamp.get(field)!r}", f"{value!r}",
                               "rebuild the sidecar with the sidecar writer")
    missing = sorted(set(columns) - set(stamp.get("column_order") or ()))
    if missing:
        raise refuse_stamp(where, f"wrote {stamp.get('column_order')} and the layouts read "
                                  f"{missing}", "every column the layouts read",
                           "rebuild the sidecar for this layout")
    pins = {pin: stamp.get(pin) for pin in SIDECAR_PINS}
    if not all(isinstance(value, str) and value for value in pins.values()):
        raise refuse_stamp(where, f"states pins {pins}", f"{list(SIDECAR_PINS)} as digests",
                           "rebuild the sidecar with the sidecar writer")
    return pins


def sidecar_rows(stamp: dict[str, Any], where: str) -> dict[int, int]:
    """The sidecar's episodes and their rows, as its stamp lists them; R16 when malformed."""
    listed, counts = stamp.get("episodes"), stamp.get("episode_rows")
    if not isinstance(listed, list) or not isinstance(counts, list) or len(listed) != len(counts):
        raise ValueError(
            f"{refusal_prefix(16)} {where} lists episodes {type(listed).__name__} and row "
            f"counts {type(counts).__name__} of different lengths\n"
            f"expected one row count per listed episode\nrebuild the sidecar")
    return {int(episode): int(count) for episode, count in zip(listed, counts)}
