"""Rule 5 (intrinsics): check each calibration record against its frames. It drops nothing.

One output record per calibration record, in the same order, with a named
defect list, a named property list and a pass flag. Checks: required fields
and finite values, positive frame size and focal lengths, pixel aspect against
the median of the same directory and view, principal point near the centre
and inside the frame, horizontal and vertical field of view, the calibration's
aspect against the frame size stored in meta/info.json, duplicate records, and
a provenance the loader accepts.

Input records are JSONL, one per (dataset, episode, view), with fx, fy, cx,
cy, width, height (the raster the calibration was measured against) and
provenance; a leading {"provenance": ...} header line is accepted.

Usage:
    python tools/validate_intrinsics.py --root $NEO_VLA_DATA_ROOT \
        --intrinsics metadata/action_pretrain/intrinsics_agibot.jsonl \
        --out filtered/intrinsics_validation.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import filter_common as common

from neo_vla.data import rays

# Default bounds: the principal point within 10% of each side from the centre
# (calibrations measured against another frame size sit far outside it), and
# the field-of-view range of the cameras these datasets use.
PRINCIPAL_POINT_MARGIN = 0.10
FIELD_OF_VIEW_DEGREES = {"horizontal": (40.0, 140.0), "vertical": (30.0, 120.0)}

DEFECTS = (
    "field_missing",
    "value_not_finite",
    "size_not_positive",
    "focal_not_positive",
    "pixel_aspect_disagrees_with_view",
    "principal_point_off_centre",
    "principal_point_outside_frame",
    "field_of_view_out_of_range",
    "stored_frame_aspect_disagrees",
    "duplicate_record",
    "provenance_not_allowed",
)

# What is true of a record's view rather than wrong with the record. A view
# whose episodes hold several calibrations decides the form of the manifest
# block, per episode rather than one per view, and every one of those
# calibrations can still be right. It is stated per record because the set of
# views that state it is the answer to which directories can take one block.
PROPERTIES = (
    "view_has_many_calibrations",
)

# How far a record's pixel aspect may sit from the one the rest of its view
# states. Wide enough for the sub-percent spread the vendors record, narrow
# enough for the quarter a calibration fitted against another frame size is out.
ASPECT_TOLERANCE = 0.03

# The numbers every check reads, and what names one calibration in the file.
# Two records sharing KEY calibrate the same camera of the same step twice, and
# two differing in NUMBERS are two calibrations of that directory's view.
NUMBERS = ("fx", "fy", "cx", "cy", "width", "height")
KEY = ("dataset", "episode", "view")


def _defect(name: str, **detail: Any) -> dict[str, Any]:
    if name not in DEFECTS:
        raise KeyError(
            f"defect {name!r} is not named in DEFECTS\n"
            f"expected one of the names this stage records\n"
            f"add the name before reporting it, so a reader can key on it"
        )
    return {"name": name, **detail}


def _property(name: str, **detail: Any) -> dict[str, Any]:
    if name not in PROPERTIES:
        raise KeyError(
            f"property {name!r} is not named in PROPERTIES\n"
            f"expected one of the names this stage records\n"
            f"add the name before reporting it, so a reader can key on it"
        )
    return {"name": name, **detail}


def field_of_view(focal: float, extent: int) -> float:
    """The angle, in degrees, a pinhole of this focal sees across this many pixels."""
    return math.degrees(2.0 * math.atan(extent / (2.0 * focal)))


def check_readable(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Whether the record states the numbers the later checks read."""
    found = []
    for name in (*KEY, *NUMBERS, "provenance"):
        if name not in record:
            found.append(_defect("field_missing", field=name))
    for name in NUMBERS:
        value = record.get(name)
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            found.append(_defect("value_not_finite", field=name, value=str(value)))
    return found


def optical_aspect(record: dict[str, Any]) -> float | None:
    """The width over height of the field this calibration sees, or None.

    Reported rather than judged. fx over fy is not one wherever a frame is
    stored at an aspect its optics do not have, so the ratio alone says
    nothing about whether a calibration is sound. What it does say is what shape the camera's field is, and that
    is worth having in the file beside the frame it was measured against.
    """
    fx, fy = float(record["fx"]), float(record["fy"])
    if fx <= 0.0 or fy <= 0.0 or int(record["width"]) <= 0 or int(record["height"]) <= 0:
        return None
    return (record["width"] / fx) / (record["height"] / fy)


def check_pixel_aspect(
    record: dict[str, Any], typical: float | None, tolerance: float,
) -> list[dict[str, Any]]:
    """Whether one pixel is the shape the rest of this view's pixels are.

    fx over fy carries the ratio of the two scales the frame was last resized
    by, so a calibration fitted against a differently shaped frame brings that
    shape here. It is compared with the view's own median rather than with one,
    because a source may deliver a frame whose pixels are not square and nothing
    on disk declares which does. A view whose records are all wrong agrees with
    itself and is not findable this way; only a measurement finds that.
    """
    fx, fy = float(record["fx"]), float(record["fy"])
    if fy == 0.0 or not typical or int(record["width"]) <= 0:
        return []
    ratio = fx / fy
    if abs(ratio / typical - 1.0) <= tolerance:
        return []
    return [_defect(
        "pixel_aspect_disagrees_with_view", fx_over_fy=round(ratio, 6),
        view_median=round(typical, 6),
        frame_aspect=round(int(record["height"]) / int(record["width"]), 6),
        tolerance=tolerance,
    )]


def check_stored_frame(
    record: dict[str, Any], shape: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    """Whether the loader could put this calibration on the frames it will read.

    Against the frame size stored in meta/info.json, within half a pixel per
    side (rays.frame_band). A calibration measured at another aspect does not
    describe the stored frames.
    """
    if shape is None or int(record["height"]) <= 0:
        return []
    low, high = rays.frame_band(shape)
    aspect = int(record["width"]) / int(record["height"])
    if low <= aspect <= high:
        return []
    return [_defect(
        "stored_frame_aspect_disagrees",
        calibration=[record["width"], record["height"]],
        frame=[shape[1], shape[0]], aspect=round(aspect, 4),
        low=round(low, 4), high=round(high, 4),
    )]


def check_principal_point(record: dict[str, Any], margin: float) -> list[dict[str, Any]]:
    """Whether the principal point sits near the centre, and inside the frame.

    Two statements from one pair of numbers. Far from the centre is a lens the
    frame was cropped off, and exactly at the centre over a whole dataset is the
    fingerprint of a principal point assumed rather than measured, which this
    reports as neither: the defect fires only on the far case, and the offsets
    are in the detail so the exact case is visible in the file.
    """
    found = []
    for name, extent in (("cx", record["width"]), ("cy", record["height"])):
        value = float(record[name])
        offset = value - (extent - 1) / 2.0
        if not 0.0 <= value <= extent - 1:
            found.append(_defect("principal_point_outside_frame", field=name,
                                 value=value, extent=extent))
        elif abs(offset) > margin * extent:
            found.append(_defect("principal_point_off_centre", field=name,
                                 offset_pixels=round(offset, 3),
                                 offset_share=round(offset / extent, 6),
                                 margin=margin))
    return found


def check_field_of_view(
    record: dict[str, Any], bounds: dict[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    """Whether the two angles this calibration sees are angles a camera has."""
    found = []
    for name, focal, extent in (
        ("horizontal", record["fx"], record["width"]),
        ("vertical", record["fy"], record["height"]),
    ):
        low, high = bounds[name]
        if float(focal) <= 0.0:
            continue
        angle = field_of_view(float(focal), int(extent))
        if not low <= angle <= high:
            found.append(_defect("field_of_view_out_of_range", axis=name,
                                 degrees=round(angle, 3), low=low, high=high))
    return found


def check_size(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Whether the resolution and the focals are quantities at all."""
    found = []
    if int(record["width"]) <= 0 or int(record["height"]) <= 0:
        found.append(_defect("size_not_positive", width=record["width"],
                             height=record["height"]))
    if float(record["fx"]) <= 0.0 or float(record["fy"]) <= 0.0:
        found.append(_defect("focal_not_positive", fx=record["fx"], fy=record["fy"]))
    return found


def check_provenance(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Whether the loader would read this calibration or refuse it."""
    if record["provenance"] in rays.PROVENANCE_ALLOWED:
        return []
    return [_defect("provenance_not_allowed", provenance=record["provenance"],
                    allowed=list(rays.PROVENANCE_ALLOWED))]


def readable(record: dict[str, Any]) -> bool:
    """Whether a record states the fields the passes over the whole file read."""
    return all(name in record for name in (*KEY, *NUMBERS))


def view_aspects(records: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    """Each directory and view -> the pixel aspect its own records state."""
    per: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        if readable(record) and float(record["fy"]) != 0.0:
            per[(record["dataset"], record["view"])].append(
                float(record["fx"]) / float(record["fy"])
            )
    return {key: statistics.median(values) for key, values in per.items()}


def stored_shapes(
    root: Path, records: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], tuple[int, int]], dict[str, Path]]:
    """Each directory and view -> the size its frames are stored at, height first.

    A directory the root does not hold contributes no shape, so a file can be
    validated away from the corpus it was built against. How many views were
    read this way is in the header, because a check that could not be made and
    a check that passed must not read the same.
    """
    wanted = sorted({(record["dataset"], record["view"])
                     for record in records if readable(record)})
    shapes: dict[tuple[str, str], tuple[int, int]] = {}
    inputs: dict[str, Path] = {}
    columns: dict[str, dict[str, Any] | None] = {}
    for dataset, view in wanted:
        if dataset not in columns:
            path = root / dataset / "meta" / "info.json"
            columns[dataset] = None
            if path.is_file():
                inputs[common.relative(root, path)] = path
                columns[dataset] = common.columns_of(
                    json.loads(path.read_text(encoding="utf-8"))
                )
        stored = columns[dataset]
        if stored is None:
            continue
        if view not in stored:
            raise KeyError(
                f"{dataset} stores no column {view!r}\n"
                f"expected one of {sorted(stored)[:6]}\n"
                f"check that the calibration records belong to this data root"
            )
        shapes[(dataset, view)] = common.frame_shape(stored, view)
    return shapes, inputs


def calibrations_per_view(records: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    """How many distinct calibrations each directory holds for each of its views.

    A manifest states one calibration per view per directory. Where this is
    above one, that shape cannot express what the source recorded, and the
    entry needs the per-episode form the extrinsics already have.
    """
    distinct: dict[tuple[str, str], set[tuple]] = defaultdict(set)
    for record in records:
        if not all(name in record for name in (*KEY, *NUMBERS)):
            continue
        distinct[(record["dataset"], record["view"])].add(
            tuple(record[name] for name in NUMBERS)
        )
    return {key: len(values) for key, values in distinct.items()}


@dataclass(frozen=True)
class Thresholds:
    """The bounds every check compares against, stated once for the whole run."""

    aspect: float
    centre: float
    degrees: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class Corpus:
    """What the whole file, and the corpus it names, say about one record."""

    aspects: dict[tuple[str, str], float]
    frames: dict[tuple[str, str], tuple[int, int]]
    per_view: dict[tuple[str, str], int]


def validate_record(
    record: dict[str, Any], corpus: Corpus, seen: Counter, limits: Thresholds,
) -> dict[str, Any]:
    """One calibration record -> what is wrong with it, what is true of its view."""
    defects = check_readable(record)
    properties: list[dict[str, Any]] = []
    aspect = None
    if not defects:
        view = (record["dataset"], record["view"])
        defects.extend(check_size(record))
        aspect = optical_aspect(record)
        defects.extend(check_pixel_aspect(record, corpus.aspects.get(view), limits.aspect))
        defects.extend(check_principal_point(record, limits.centre))
        defects.extend(check_field_of_view(record, limits.degrees))
        defects.extend(check_stored_frame(record, corpus.frames.get(view)))
        defects.extend(check_provenance(record))
        key = tuple(record[name] for name in KEY)
        seen[key] += 1
        if seen[key] > 1:
            defects.append(_defect("duplicate_record", seen=seen[key]))
        count = corpus.per_view[view]
        if count > 1:
            properties.append(_property("view_has_many_calibrations", distinct=count))
    return {
        "dataset": record.get("dataset"),
        "episode": record.get("episode"),
        "view": record.get("view"),
        "optical_aspect": None if aspect is None else round(aspect, 6),
        "pass": not defects,
        "defects": defects,
        "properties": properties,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, help="the data root holding the directories")
    parser.add_argument("--intrinsics", action="append", required=True,
                        help="a calibration record file; repeat once per source")
    parser.add_argument("--out", required=True, help="new JSONL output")
    parser.add_argument("--aspect-tolerance", type=float, default=ASPECT_TOLERANCE,
                        help="how far a pixel aspect may sit from its view's median")
    parser.add_argument("--centre-margin", type=float, default=PRINCIPAL_POINT_MARGIN,
                        help="how far the principal point may sit from the centre, "
                             "as a share of the side")
    horizontal = FIELD_OF_VIEW_DEGREES["horizontal"]
    vertical = FIELD_OF_VIEW_DEGREES["vertical"]
    parser.add_argument("--horizontal-low", type=float, default=horizontal[0])
    parser.add_argument("--horizontal-high", type=float, default=horizontal[1])
    parser.add_argument("--vertical-low", type=float, default=vertical[0])
    parser.add_argument("--vertical-high", type=float, default=vertical[1])
    args = parser.parse_args(argv)

    root = common.resolve_root(args.root)
    sources = {common.display(root, common.cli_path(name)): common.cli_path(name)
               for name in args.intrinsics}
    out = common.new_output(args.out, list(sources.values()))

    # Concatenated in file-name order; one directory's views come from one source.
    records = [record for name in sorted(sources)
               for record in common.read_jsonl(sources[name])[1]]
    shapes, read = stored_shapes(root, records)
    corpus = Corpus(aspects=view_aspects(records), frames=shapes,
                    per_view=calibrations_per_view(records))
    limits = Thresholds(
        aspect=args.aspect_tolerance, centre=args.centre_margin,
        degrees={"horizontal": (args.horizontal_low, args.horizontal_high),
                 "vertical": (args.vertical_low, args.vertical_high)},
    )
    seen: Counter = Counter()
    out_records = [validate_record(record, corpus, seen, limits) for record in records]

    counts: dict[str, Counter] = {"defects": Counter(), "properties": Counter()}
    for record in out_records:
        for field, found in counts.items():
            for item in record[field]:
                found[item["name"]] += 1
    head = common.header(
        "validate_intrinsics",
        inputs=[*sources, *read],
        settings={
            "aspect_tolerance": args.aspect_tolerance,
            "centre_margin": args.centre_margin,
            "horizontal_degrees": [args.horizontal_low, args.horizontal_high],
            "vertical_degrees": [args.vertical_low, args.vertical_high],
            "views": len(corpus.per_view),
            "views_with_stored_frames": len(shapes),
            "defect_names": list(DEFECTS),
            "property_names": list(PROPERTIES),
            "records": len(out_records),
            "defect_counts": dict(sorted(counts["defects"].items())),
            "property_counts": dict(sorted(counts["properties"].items())),
        },
    )
    common.write_jsonl(out, head, out_records)
    passed = sum(1 for record in out_records if record["pass"])
    print(f"{len(out_records)} calibrations, {passed} without defects; wrote {out}")
    for name, count in sorted(counts["defects"].items()):
        print(f"  {name}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
