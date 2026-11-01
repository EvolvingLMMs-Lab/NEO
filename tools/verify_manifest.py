"""Stage 11: check a manifest against the data it points at, not against itself.

Every path resolves under the root and exists, every frame file the index names
is a file holding bytes, every hash the header recorded still matches, every
recorded grid is what the snapping rule produces under the manifest's own
canvas, every episode boundary array agrees with the row count beside it, every
calibration describes a picture the cache actually holds, and the declared
mixture is the one the weights and rows produce. Exits non-zero on the first
disagreement it finds, after reporting all of them.

Checks that need the loader are named and counted as skipped rather than
quietly passed; --require-loader turns a skipped check into a failure.

Usage:
    python tools/verify_manifest.py --root /corpus --manifest manifests/oxe3src.json
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import stat
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import stage
import torch

from neo_vla.data import (
    Draw,
    RobotDataset,
    images,
    mixture,
    rays,
    reader,
    registry,
)
from neo_vla.constants import (
    ACTION_DIM,
    BODY_DIM,
    BODY_NOISE_WIDTH,
    GEOMETRY_KEYS,
    GRIPPER_COLUMN,
    HISTORY_PREFIX,
    MANIFEST_SCHEMA_VERSION,
    N_ROLES,
    NOISE_WIDTH,
    OLD_MANIFEST_SCHEMA_VERSIONS,
    PREFIX_LAYOUT,
    ROLE_NOW,
    ROLE_START,
    STATE_ABSENT,
    STATE_DIM,
    STATE_MEASURED,
    STATE_NO_HISTORY,
    TEMPLATE_MAJOR,
    batch_keys,
)
from neo_vla.data.conditioning import parse_frame_ref
from neo_vla.data.history import noise_dims
from neo_vla.data.images import snap_grid
from neo_vla.data.prompt import check_mode_agrees, load_template
from neo_vla.model.tokenizers import EXTRINSICS_DIM, INTRINSICS_DIM

logger = logging.getLogger("verify_manifest")

REPO = Path(__file__).resolve().parent.parent

# How many frame files one thread stats before it hands a result back. One
# stat of a cold name on the corpus filesystem measured 3.7 ms, nearly all of
# it waiting, so a batch this size is seconds of work against a queueing cost
# of microseconds, and still reports progress often enough to watch.
STAT_BATCH = 256

# Threads statting frame files at once. Well past the core count because the
# work is waiting on the filesystem rather than computing, and stopping there
# because the filesystem does not go faster: measured over the corpus, 32
# threads held about 13,700 stats a second and 64 and 96 were both slower.
# What the whole pass costs is the filesystems mood rather than the checks:
# the same 1,808,137 names took 176.5 s on one run and 332.1 s on another, and
# in the second the pass that also weighs each file came in under the pass that
# only asked whether it was there, at 332.1 s against 353.9 s.
STAT_THREADS = 32

# How many faulty files the report names one by one before it says only how
# many there are. A tree that assembled short is missing them by the thousand,
# and one line each would bury every other finding in the same report.
NAMED_FAULTS = 20

# The fields every schema-3 entry states: the schema-2
# ones, the body and the sidecar; per arm the joints and hand blocks; the
# columns block naming each v7 column (null where none is read). A manifest
# naming a normaliser was built by a retired stage.
ENTRY_FIELDS = ("embodiment", "control_hz", "control_hz_method", "control_hz_evidence",
                "history", "body", "sidecar")
ACTION_FIELDS = ("gripper", "rotation_delta_frame", "rotation_source")
ARM_FIELDS = ("slot", "name", "state", "joints", "hand")
COLUMN_FIELDS = ("state", "action", "joints", "hand", "joints_command", "body", "body_command")
RETIRED_ACTION_FIELDS = ("normaliser",)

# The bound on a packed action: range dims are clamped to +-(1 + 2 alpha) at
# pack time, the gripper to [-1, 1]. Float slack on the compare.
BOUND_SLACK = 1e-5


def geometry_of(intrinsics_mode: str) -> tuple[str, ...]:
    """The geometry records the one mode reads: all four (tokens).

    A stamp naming a removed mode is refused by name.
    """
    rays.check_intrinsics_mode(intrinsics_mode, "the prompt stamp's intrinsics_mode")
    return tuple(GEOMETRY_KEYS)


def sample_shapes(canvas: tuple[int, int], n_views: int, n_slots: int,
                  chunk: int, action_dim: int,
                  intrinsics_mode: str) -> dict[str, tuple[int, ...]]:
    """The per-sample shapes of the v7 batch, before collate adds B.

    Built from the run's own numbers rather than written out. Every view and
    geometry record has a history_ twin laid out the same way; prompt and
    embodiment are strings and carry no shape.
    """
    views = {"pixels": (n_views, 3, canvas[0], canvas[1]),
             "image_shapes": (n_views, 2), "view_valid": (n_views,)}
    width = {"intrinsics": INTRINSICS_DIM, "camera_extrinsics": EXTRINSICS_DIM}
    geometry = {key: ((n_views, width[key]) if key in width else (n_views,))
                for key in geometry_of(intrinsics_mode)}
    shapes = {**views, **geometry}
    shapes.update({HISTORY_PREFIX + key: shape for key, shape in list(shapes.items())})
    shapes.update({
        "history_valid": (),
        "frame_ref": (n_slots,),
        "slot_valid": (n_slots,),
        "state": (N_ROLES, n_slots, STATE_DIM),
        "state_code": (N_ROLES, n_slots),
        "state_dims": (N_ROLES, n_slots, STATE_DIM),
        "dim_valid": (n_slots, action_dim),
        "actions": (n_slots, chunk, action_dim),
        "action_valid": (n_slots, chunk),
        "action_loss_dims": (n_slots, chunk, action_dim),
        "state_noise": (N_ROLES, n_slots, NOISE_WIDTH),
        "body_state": (N_ROLES, BODY_DIM),
        "body_state_code": (N_ROLES,),
        "body_state_dims": (N_ROLES, BODY_DIM),
        "body_dim_valid": (BODY_DIM,),
        "body_actions": (chunk, BODY_DIM),
        "body_action_valid": (chunk,),
        "body_loss_dims": (chunk, BODY_DIM),
        "body_state_noise": (N_ROLES, BODY_NOISE_WIDTH),
    })
    return shapes


class Report:
    """The failures and the skipped checks of one verification pass."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.skipped: list[str] = []

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def skip(self, message: str) -> None:
        self.skipped.append(message)


def check_paths(root: Path, manifest: dict[str, Any], report: Report) -> None:
    """Every path the manifest names resolves under the root and exists."""
    cache = manifest["prompt"]["cache"]
    named = [manifest["prompt"]["template"]]
    for entry in manifest["datasets"]:
        named.extend([entry["path"], entry["episode_starts"],
                      entry["cache"]["arrow"], entry["cache"]["frames"],
                      f"{cache}/{entry['name']}.jsonl"])
        if entry["trajectory_weights"]:
            named.append(entry["trajectory_weights"])
        for view in entry["views"]:
            named.extend(view[block]["path"] for block in ("intrinsics", "extrinsics")
                         if view[block]["path"])
    for name in named:
        if Path(name).is_absolute():
            report.fail(f"path {name} is absolute; the manifest states root-relative paths")
            continue
        if not (root / name).exists():
            report.fail(f"path {name} does not exist under the root")


def check_inputs(root: Path, manifest: dict[str, Any], report: Report) -> None:
    """The header's recorded input hashes still describe the files on disk."""
    for line in stage.check_inputs_match(manifest["provenance"], root):
        report.fail(f"input hash: {line}")


def check_grids(manifest: dict[str, Any], report: Report) -> None:
    """Every recorded grid is what this run's snapping rule produces."""
    canvas = manifest["canvas"]
    for entry in manifest["datasets"]:
        for view in entry["views"]:
            tile = snap_grid(
                tuple(view["shape"]),
                short_side=canvas["short_side"], token=canvas["token"],
                cell=(canvas["cell"][0], canvas["cell"][1]),
            )
            if [tile.grid_h, tile.grid_w] != list(view["grid"]):
                report.fail(
                    f"{entry['name']} view {view['key']} records grid {view['grid']}, "
                    f"the rule gives [{tile.grid_h}, {tile.grid_w}]"
                )
            if tile.height > canvas["cell"][0] or tile.width > canvas["cell"][1]:
                report.fail(
                    f"{entry['name']} view {view['key']} needs "
                    f"{tile.height}x{tile.width}, the canvas is {canvas['cell']}"
                )


def check_entries(manifest: dict[str, Any], report: Report) -> None:
    """Names, sources, weights and camera slots, against the registry."""
    seen: set[str] = set()
    for entry in manifest["datasets"]:
        if entry["name"] in seen:
            report.fail(f"duplicate dataset name {entry['name']}")
        seen.add(entry["name"])
        if entry["source"] not in registry.SOURCE_IDS:
            report.fail(f"{entry['name']} names unregistered source {entry['source']}")
        elif registry.SOURCE_IDS[entry["source"]] != entry["source_id"]:
            report.fail(
                f"{entry['name']} records source_id {entry['source_id']}, "
                f"the registry says {registry.SOURCE_IDS[entry['source']]}"
            )
        weight = entry.get("weight")
        if weight is None or not np.isfinite(weight) or weight < 0:
            report.fail(f"{entry['name']} has weight {weight}")
        for view in entry["views"]:
            if view["role"] not in registry.VIEW_ROLES:
                report.fail(f"{entry['name']} view {view['key']} has role {view['role']}")
            if view["slot"] >= manifest["prompt"]["stamp"]["n_views"]:
                report.fail(f"{entry['name']} view {view['key']} claims slot {view['slot']}")
            mounted = view["arm_slot"]
            arms = [arm["slot"] for arm in entry["arms"]]
            if mounted is not None and mounted not in arms:
                report.fail(
                    f"{entry['name']} view {view['key']} is mounted on arm slot "
                    f"{mounted}, which this directory does not have"
                )
        try:
            parse_frame_ref(entry["frame_ref"])
        except ValueError:
            report.fail(f"{entry['name']} has frame_ref {entry['frame_ref']}")
        check_entry_fields(entry, report)


def check_entry_fields(entry: dict[str, Any], report: Report) -> None:
    """The schema-3 fields are there and say something; the retired one is gone."""
    name = entry["name"]
    for field in ENTRY_FIELDS:
        if field not in entry:
            report.fail(f"{name} has no {field} (manifest schema 3)")
    for field in ("control_hz_method", "control_hz_evidence"):
        if field in entry and not str(entry[field] or "").strip():
            report.fail(f"{name} states an empty {field}")
    for field in ACTION_FIELDS:
        if field not in entry["action"]:
            report.fail(f"{name} action has no {field}")
    for arm in entry.get("arms", []):
        missing = [field for field in ARM_FIELDS if field not in arm]
        if missing:
            report.fail(f"{name} arm {arm.get('name')!r} has no {missing} (manifest schema 3)")
    missing = [field for field in COLUMN_FIELDS if field not in entry.get("columns", {})]
    if missing:
        report.fail(f"{name} columns block has no {missing} (manifest schema 3)")
    for field in RETIRED_ACTION_FIELDS:
        if field in entry["action"]:
            report.fail(f"{name} action names a {field}; the data config names "
                        f"embodiments.json, not the manifest (V6 C15)")
    blocked = (entry.get("history") or {}).get("blocked")
    if not isinstance(blocked, list) or any(
            not 0 <= int(position) < entry["episodes"] for position in blocked):
        report.fail(f"{name} history.blocked is {blocked!r}; expected kept positions")


def check_embodiment_arrays(root: Path, manifest: dict[str, Any], report: Report) -> None:
    """Every per-episode key array and id array holds one value per kept episode."""
    for entry in manifest["datasets"]:
        block = entry.get("embodiment") or {}
        if block.get("kind") != "episode":
            continue
        for field in ("path", "ids_path"):
            path = root / block[field]
            if not path.exists():
                report.fail(f"{entry['name']} embodiment {field} {block[field]} is missing")
                continue
            count = np.load(io.BytesIO(path.read_bytes()), allow_pickle=False).shape[0]
            if count != entry["episodes"]:
                report.fail(f"{entry['name']} embodiment {field} holds {count} positions for "
                            f"{entry['episodes']} kept episodes (R16)")


def check_totals(manifest: dict[str, Any], report: Report) -> None:
    """The totals line agrees with the entries it summarises."""
    entries = manifest["datasets"]
    expected = {
        "directories": len(entries),
        "episodes": sum(entry["episodes"] for entry in entries),
        "rows": sum(entry["rows"] for entry in entries),
    }
    if expected != manifest["totals"]:
        report.fail(f"totals {manifest['totals']} against entries {expected}")


def check_episode_starts(root: Path, manifest: dict[str, Any], report: Report) -> None:
    """Each boundary array has one more entry than episodes and ends at rows."""
    for entry in manifest["datasets"]:
        path = root / entry["episode_starts"]
        if not path.exists():
            continue
        starts = np.load(io.BytesIO(path.read_bytes()), allow_pickle=False)
        if starts.dtype != np.int64:
            report.fail(f"{entry['name']} episode_starts dtype is {starts.dtype}, not int64")
        if starts.shape != (entry["episodes"] + 1,):
            report.fail(
                f"{entry['name']} episode_starts has {starts.shape[0]} entries, "
                f"expected {entry['episodes'] + 1}"
            )
            continue
        if int(starts[-1]) != entry["rows"]:
            report.fail(
                f"{entry['name']} episode_starts ends at {int(starts[-1])}, "
                f"the manifest says {entry['rows']} rows"
            )
        if np.any(np.diff(starts) < 0):
            report.fail(f"{entry['name']} episode_starts is not monotone")


def frame_name_fault(name: object) -> str | None:
    """Why one frame index entry cannot name a file under the root, or None.

    Settled without a filesystem call, because every one of these reads as
    present to a stat: an empty name is the root directory itself, and an
    absolute or climbing name is a file in another tree, which on a corpus
    being repointed is the one that would be checked where it used to live and
    read where it now does.
    """
    if not isinstance(name, str):
        return f"frame index names {name!r}, which is not a path"
    if not name:
        return "frame index names an empty path"
    if os.path.isabs(name):
        return f"frame file {name} is absolute; the index names paths under the root"
    if os.path.normpath(name).split(os.sep)[0] == os.pardir:
        return f"frame file {name} climbs out of the root"
    return None


HEADER_BYTES = 16

# A container states what it is in its first bytes. A name that resolves to
# bytes which are not that container is a fault no stat can see: one byte, or
# another file's content at the right size, passes every test that asks only
# how big the file is, and then raises out of a dataloader worker on the first
# draw that reads it.
PARQUET_MAGIC = b"PAR1"

# Every ISO base media first box, not only the ftyp that every mp4 in this
# corpus actually carries. Refusing a valid variant would stop a launch, and
# the wider set costs almost nothing to detection: four bytes of noise land in
# it about once in 700 million.
MP4_FIRST_BOXES = frozenset({b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide"})

# The kinds build_cache writes a frame index for, and the only ones this check
# claims a signature for. Anything else is left to the stat rather than
# refused on a guess.
HEADER_SUFFIXES = ("mp4", "parquet")


def header_fault(path: str, name: str) -> str | None:
    """What the first bytes say the file is not, or None when they agree with the name."""
    suffix = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if suffix not in HEADER_SUFFIXES:
        return None
    try:
        with open(path, "rb") as handle:
            head = handle.read(HEADER_BYTES)
    except OSError as error:
        return f"cannot be opened: {error.strerror or error}"
    if len(head) < HEADER_BYTES:
        return f"holds {len(head)} bytes, fewer than a header"
    if suffix == "parquet":
        if not head.startswith(PARQUET_MAGIC):
            return "does not begin as a parquet file"
    elif head[4:8] not in MP4_FIRST_BOXES:
        return "does not begin as an mp4 file"
    return None


def faulty_frames(root: Path, names: list[str]) -> tuple[list[tuple[str, str]], float]:
    """The names no frame file can be read at and why, and the seconds it took.

    One stat per name answers three of the questions a name raises: whether
    anything is there, whether it is a file rather than a directory, and
    whether it holds any bytes. A sixteen-byte read answers a fourth, which
    the stat cannot: whether those bytes are the container the name claims.
    Both are cheap for the same reason the pool is threads -- on this
    filesystem the call is almost all waiting, so the wall clock is the corpus
    divided by the pool rather than by a core count.
    """
    base = str(root)

    def faulty_in(batch: list[str]) -> list[tuple[str, str]]:
        """The names in one batch no frame file can be read at, each with why."""
        found: list[tuple[str, str]] = []
        for name in batch:
            try:
                stated = os.stat(os.path.join(base, name))
            except OSError:
                found.append((name, "is missing"))
                continue
            if not stat.S_ISREG(stated.st_mode):
                found.append((name, "is not a file"))
            elif stated.st_size == 0:
                found.append((name, "holds no bytes"))
            else:
                fault = header_fault(os.path.join(base, name), name)
                if fault is not None:
                    found.append((name, fault))
        return found

    batches = [names[at:at + STAT_BATCH] for at in range(0, len(names), STAT_BATCH)]
    faulty: list[tuple[str, str]] = []
    start = time.perf_counter()
    if batches:
        with ThreadPoolExecutor(max_workers=STAT_THREADS,
                                thread_name_prefix="stat") as pool:
            for done, found in enumerate(pool.map(faulty_in, batches), 1):
                faulty.extend(found)
                if done % 500 == 0:
                    logger.info("%d of %d frame files checked",
                                done * STAT_BATCH, len(names))
    return faulty, time.perf_counter() - start


def check_frame_cache(root: Path, manifest: dict[str, Any], sample: int,
                      report: Report) -> None:
    """The frame index names one file per kept episode, and each is a file with bytes.

    The check covers every name the manifest reaches, not a sample of them. A
    sampled check passes a tree that assembled short, and the run then dies on
    the first draw that reaches the gap, hours after this gate said the corpus
    was ready. Names repeat across directories, so they are gathered into one
    set first and each file is statted once. A name that cannot be a path under
    the root at all is reported here rather than statted, both because a stat
    would call several of them present and because a name of the wrong type
    would end the pass with a traceback instead of a report.

    How far the word readable reaches here is decided in faulty_frames.
    """
    named: dict[str, str] = {}
    for entry in manifest["datasets"]:
        path = root / entry["cache"]["frames"]
        if not path.exists():
            continue
        try:
            index = json.loads(path.read_text(encoding="utf-8"))
            kind = index["kind"]
        except (OSError, UnicodeDecodeError, ValueError, TypeError, KeyError) as error:
            # Every way an index can be unreadable, because this stage reports
            # its findings together and a raise here would lose the ones
            # already collected and name the parser instead of the corpus.
            report.fail(f"{entry['name']} frame index {entry['cache']['frames']} "
                        f"does not read: {str(error).splitlines()[0]}")
            continue
        if kind not in reader.FRAME_KINDS:
            report.fail(f"{entry['name']} frame index states kind {kind}")
            continue
        views = index.get("views")
        if not isinstance(views, dict):
            report.fail(f"{entry['name']} frame index states no cameras")
            continue
        absent_views = {view["key"] for view in entry["views"]} - set(views)
        if absent_views:
            report.fail(
                f"{entry['name']} frame index has no camera {sorted(absent_views)[0]}")
        for key, files in sorted(views.items()):
            if not isinstance(files, list):
                # A string here would be read as its own characters and a
                # number would end the pass, and either way the index does
                # not state a file per episode, which is the finding.
                report.fail(f"{entry['name']} frame index states {files!r} for "
                            f"{key}, not a list of files")
                continue
            if len(files) != entry["episodes"]:
                report.fail(
                    f"{entry['name']} frame index for {key} has {len(files)} entries, "
                    f"expected {entry['episodes']}"
                )
            # A null is refused in the first `sample` positions and stepped
            # over everywhere else. The split is deliberate and tested both
            # ways -- test_a_null_at_the_head_is_still_caught and
            # test_a_null_past_the_head_still_passes -- because when the
            # existence pass below grew to cover every name, the null half
            # kept the reach it already had.
            #
            # What that means, stated rather than left to be discovered: one
            # null is a failure at kept position 3 and a fact at position 5,
            # so the verdict turns on where the episode happens to sit and not
            # on anything about the null. Changing it is a decision and not a
            # fix. reader.py leaves an absent camera out of the sample, so the
            # loader handles a null wherever it falls, and this corpus's four
            # nulls sit at kept positions 414 and 87, where this never fires.
            for name in files[:sample]:
                if name is None:
                    report.fail(f"{entry['name']} frame file {name} is missing")
            for name in files:
                if name is None:
                    continue
                fault = frame_name_fault(name)
                if fault is not None:
                    report.fail(f"{entry['name']} {fault}")
                    continue
                named.setdefault(name, entry["name"])
    # A stat and sixteen bytes is where this check stops, and the line between
    # what it answers and what it does not is worth stating exactly.
    #
    # It answers: absent, not a file, empty, and not the container the name
    # claims -- which covers a file of one byte and a file holding another
    # file's content at the right size. It does NOT answer whether a file that
    # begins correctly is whole: a truncated mp4 with a valid header passes
    # here and needs a decode, which is a frame-file decode audit,
    # run on its own.
    #
    # The cost is measured, not assumed, because an earlier version of this
    # comment priced the wrong test. Decoding each file as the media it claims
    # to be is 3.55 ms a file, so 6,421 s over 1,808,137 names against a gate
    # that runs in about 928 s -- which is why no decode happens here. Reading
    # sixteen bytes is a different thing entirely: over two disjoint 20,000
    # name samples, run in both orders so cache warming cancels, presence
    # alone was 0.147 and 0.174 ms, presence and size 0.177 and 0.191 ms, and
    # with this read 0.226 and 0.255 ms. About 1.3x the stat, so 90 to 110 s
    # on the full pass.
    #
    # The reason not to rely on the re-encode path having checked already:
    # verify_reencode.py does decode every output it accepts and
    # assemble_reencoded_tree.py refuses one whose sidecar does not say so, but
    # that covers files that arrive the sanctioned way. The four zero-byte
    # files this gate was widened for were placed into the tree directly, which
    # is the case the argument from the re-encode path does not reach.
    faulty, seconds = faulty_frames(root, list(named))
    logger.info("%d frame files checked in %.1f s, %d cannot be read",
                len(named), seconds, len(faulty))
    for name, fault in faulty[:NAMED_FAULTS]:
        report.fail(f"{named[name]} frame file {name} {fault}")
    if len(faulty) > NAMED_FAULTS:
        report.fail(
            f"{len(faulty) - NAMED_FAULTS} further frame files the manifest "
            f"names are missing, are not files, hold no bytes, or do not "
            f"begin as the container they claim; "
            f"a frame-file audit lists them all"
        )


def check_prompt_stamp(root: Path, manifest: dict[str, Any], report: Report) -> None:
    """The stamp the manifest holds against the cache, and against the artifact.

    Two comparisons of different reach. The stamp beside the rendered text is
    the stamp the manifest holds, which catches a cache swapped under a
    manifest but not a pair that went stale together, because both are written
    from one object. So the four fields the template file itself determines are
    read back off that file, prefix_layout among them: a command-line default
    could let an artifact of one version reach a run under another's layout
    string. The other three
    describe the run rather than the artifact, and re-deriving them would
    compare the manifest with itself.
    """
    declared = manifest["prompt"]["stamp"]
    if declared.get("prefix_layout") != PREFIX_LAYOUT:
        report.fail(f"prompt stamp prefix_layout {declared.get('prefix_layout')!r}; this tree "
                    f"reads {PREFIX_LAYOUT!r} only, rebuild with neo_v{TEMPLATE_MAJOR}.yaml")
    stamp_path = root / manifest["prompt"]["cache"] / "stamp.json"
    if not stamp_path.exists():
        report.skip(f"prompt stamp: {stamp_path} has not been built")
    else:
        stored = json.loads(stamp_path.read_text(encoding="utf-8"))
        for field, value in sorted(declared.items()):
            if stored.get(field) != value:
                report.fail(f"prompt stamp {field}: {stored.get(field)!r} against {value!r}")
    named = manifest["prompt"]["template"]
    if not named:
        report.skip("prompt template: no template named, the stamp is not re-derived")
        return
    if not (root / named).exists():
        # A missing artifact is already a failure from check_paths; this only
        # says which check could not run because of it.
        report.skip(f"prompt template: {named} does not exist under the root")
        return
    try:
        template = load_template(root / named)
    except Exception as error:
        # Every way a template can be unreadable, because this stage reports
        # its findings and a raise here would lose the ones already collected.
        report.fail(f"prompt template {named} does not load: {str(error).splitlines()[0]}")
        return
    for field, found in (("template_name", template.name),
                         ("template_version", template.version),
                         ("template_digest", template.digest),
                         ("prefix_layout", template.prefix_layout)):
        if declared.get(field) != found:
            report.fail(
                f"prompt {field}: the manifest holds {declared.get(field)!r} and "
                f"{named} produces {found!r}; re-render with tools/build_cache.py"
            )
    # The artifact and the mode name the same thing and only the artifact is
    # hashed into the stamp, so nothing else in this pass can see them disagree.
    try:
        check_mode_agrees(template, declared.get("intrinsics_mode", ""))
    except ValueError as error:
        report.fail(f"prompt template {named}: {str(error).splitlines()[0]}")


def check_expected_share(root: Path, manifest: dict[str, Any],
                         weights_path: Path | None, report: Report) -> None:
    """The mass weight times rows produces is the share the weights stage expected.

    A staleness check: both sides are the same arithmetic over the same two
    numbers, so it catches a manifest rebuilt after its weights changed and
    nothing else. What the draw actually realises is check_realised_mixture.
    """
    mass = {entry["name"]: entry["weight"] * entry["rows"] for entry in manifest["datasets"]}
    total = sum(mass.values())
    if total <= 0:
        report.fail("every directory has zero mass")
        return
    if weights_path is None or not weights_path.exists():
        report.skip("mixture: no weights file given, share not compared")
        return
    expected = stage.read_json(weights_path)["expected_share"]["by_source"]
    by_source: dict[str, float] = {}
    for entry in manifest["datasets"]:
        by_source[entry["source"]] = by_source.get(entry["source"], 0.0) + (
            mass[entry["name"]] / total)
    for source, share in sorted(by_source.items()):
        if abs(share - expected.get(source, -1.0)) > 1e-6:
            report.fail(
                f"source {source} takes {share:.6f} of the mixture, "
                f"the weights stage expected {expected.get(source)}"
            )


def check_realised_mixture(dataset: RobotDataset, draws: int, seed: int,
                           report: Report) -> None:
    """The share a large draw realises is the share the plan declares.

    The declared share is what the mixture table says; this is what the sampler
    does. They agree only because nothing after the first level rejects a draw,
    so this is the check a rejection added later fails, and it is the reason
    the two are not one comparison of weight times rows.
    """
    plan = dataset.plan()
    declared = plan.by_source()
    realised = mixture.realised_mass(plan, draws, seed=seed)
    for source, share in sorted(declared.items()):
        # Four standard errors of a share at this draw count, and one draw, so
        # a source holding a thousandth of the mixture is still comparable.
        tolerance = 4.0 * float(np.sqrt(share * (1.0 - share) / draws)) + 1.0 / draws
        if abs(realised[source] - share) > tolerance:
            report.fail(
                f"source {source} realises {realised[source]:.6f} of {draws} draws, "
                f"the plan declares {share:.6f}"
            )


def calibrated_sizes(root: Path, entry: dict[str, Any], view: dict[str, Any],
                     report: Report) -> list[tuple[int, int]]:
    """Every pixel size this view's calibration was measured against.

    One for a constant view, and the distinct ones of the array for a
    per-episode view; an episode with no calibration contributes none. The
    array is read rather than trusted, because it is where a directory that
    swaps cameras between episodes keeps the size that has to be checked.
    """
    intrinsics = view["intrinsics"]
    if intrinsics["kind"] == "none":
        return []
    if intrinsics["kind"] == "constant":
        return [(int(intrinsics["width"]), int(intrinsics["height"]))]
    path = root / intrinsics["path"]
    if not path.exists():
        report.fail(f"{entry['name']} view {view['key']} names a calibration array "
                    f"{intrinsics['path']} that does not exist")
        return []
    array = np.load(io.BytesIO(path.read_bytes()), allow_pickle=False)
    if array.ndim != 2 or array.shape != (entry["episodes"], 6):
        report.fail(
            f"{entry['name']} view {view['key']} has calibrations of shape "
            f"{tuple(array.shape)}, expected ({entry['episodes']}, 6)"
        )
        return []
    sizes = array[:, 4:6]
    measured = sizes[np.isfinite(sizes).all(axis=1)]
    return sorted({(int(width), int(height)) for width, height in measured})


def check_intrinsics(root: Path, manifest: dict[str, Any], report: Report) -> None:
    """A calibration's provenance is allowed and its picture is the cached one.

    The second half is the check this stage exists for. A calibration measured
    against a frame of another aspect describes a picture the cache does not
    hold, and rescaling it onto the tile silently turns every ray by an angle
    nobody measured. Compared with the stored frame the entry records rather
    than with the grid that frame snaps to: the grid stands for anything within
    half a token of it, which on this corpus's tiles is an aspect up to 18% from
    the stored one.
    """
    for entry in manifest["datasets"]:
        for view in entry["views"]:
            intrinsics = view["intrinsics"]
            if intrinsics["kind"] not in rays.INTRINSICS_KINDS:
                report.fail(f"{entry['name']} view {view['key']} states intrinsics "
                            f"kind {intrinsics['kind']!r}")
                continue
            if intrinsics["kind"] == "none":
                continue
            if intrinsics["provenance"] not in rays.PROVENANCE_ALLOWED:
                report.fail(f"{entry['name']} view {view['key']} has intrinsics "
                            f"provenance {intrinsics['provenance']!r}")
            shape = view["shape"]
            low, high = rays.frame_band((shape[0], shape[1]))
            for width, height in calibrated_sizes(root, entry, view, report):
                if not low <= width / height <= high:
                    report.fail(
                        f"{entry['name']} view {view['key']} is calibrated at "
                        f"{width}x{height}, aspect {width / height:.4f}, and its "
                        f"frames are stored at {shape[1]}x{shape[0]}, which takes "
                        f"{low:.4f}..{high:.4f}"
                    )


def check_records(what: str, state: np.ndarray, dims: np.ndarray, code: np.ndarray,
                  findings: list[str]) -> None:
    """Per (role, arm) or role: under MEASURED the record is finite exactly where its mask
    is set and the mask holds a dim; under every other code it is NaN with no dim set."""
    measured = code == STATE_MEASURED
    finite = np.isfinite(state)
    if not (finite[measured] == dims[measured]).all() or not dims[measured].any(axis=-1).all():
        findings.append(f"{what} is not finite exactly where a MEASURED record's dims are set")
    if finite[~measured].any() or dims[~measured].any():
        findings.append(f"{what} holds a value or a dim under a code that is not MEASURED")


def check_codes(what: str, code: np.ndarray, present: np.ndarray, history: bool,
                findings: list[str]) -> None:
    """The codes' precedence: ABSENT exactly where nothing is present, NO_HISTORY only in the
    start role and exactly without history."""
    if (code[:, ~present] != STATE_ABSENT).any() or (code[:, present] == STATE_ABSENT).any():
        findings.append(f"ABSENT is not exactly the {what} the sample leaves out")
    if (code[ROLE_NOW] == STATE_NO_HISTORY).any():
        findings.append(f"NO_HISTORY in the now role of the {what}")
    start = code[ROLE_START][present]
    if history and (start == STATE_NO_HISTORY).any():
        findings.append(f"NO_HISTORY with history present in the {what}")
    if not history and (start != STATE_NO_HISTORY).any():
        findings.append(f"an existing {what} start code is not NO_HISTORY without history")


def check_packed(what: str, actions: np.ndarray, valid: np.ndarray, columns: np.ndarray,
                 loss: np.ndarray, alpha: float, findings: list[str]) -> None:
    """Packed actions: finite, inside +-(1 + 2 alpha), 0 off the kept steps and valid
    columns; the loss mask inside the keys and the sample's dim_valid."""
    live = valid[..., :, None] & columns[..., None, :]
    if (loss & ~live).any():
        findings.append(f"{what} loss dims outside action_valid and dim_valid")
    if not np.isfinite(actions).all():
        findings.append(f"{what} hold a non-finite value")
        return
    if np.abs(actions).max(initial=0.0) > 1 + 2 * alpha + BOUND_SLACK:
        findings.append(f"|{what}| reaches {np.abs(actions).max():.4f}, past 1 + 2 alpha")
    if (actions[~live] != 0).any():
        findings.append(f"{what} hold a value off the kept steps and valid columns")


def check_sample(name: str, sample: dict[str, Any], n_slots: int, alpha: float,
                 report: Report) -> None:
    """One drawn sample against the v7 agreements a shape check cannot see.

    Absence against shapes for both view blocks; the arm and body codes'
    precedence; each record finite exactly where its per-dim mask is set under
    MEASURED and empty otherwise; the body ABSENT in both roles exactly when
    it trains nothing; the history flag against its views; history intrinsics
    a copy of the current ones; the loss dims inside the keys and dim_valid;
    packed actions finite, bounded and 0 off what trains; noise only on the
    MEASURED records' valid dims.
    """
    try:
        images.check_absence_agrees(sample["image_shapes"], sample["view_valid"])
        images.check_absence_agrees(sample["history_image_shapes"],
                                    sample["history_view_valid"])
        code = np.asarray(sample["state_code"])
        dims = np.asarray(sample["state_dims"], dtype=bool)
        slot_valid = np.asarray(sample["slot_valid"], dtype=bool)
        history = bool(np.asarray(sample["history_valid"]))
        state = np.asarray(sample["state"], dtype=np.float64)
        body_code = np.asarray(sample["body_state_code"])
        body_dims = np.asarray(sample["body_state_dims"], dtype=bool)
        body = np.asarray(sample["body_state"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        report.fail(f"{name} sample: {str(error).splitlines()[0]}")
        return
    findings: list[str] = []
    check_codes("arm", code, slot_valid, history, findings)
    check_records("state", state, dims, code, findings)
    body_present = np.asarray([bool((body_code != STATE_ABSENT).all())])
    if (body_code == STATE_ABSENT).any() and not (body_code == STATE_ABSENT).all():
        findings.append("the body is ABSENT in one role only")
    check_codes("body", body_code[:, None], body_present, history, findings)
    check_records("body_state", body, body_dims, body_code, findings)
    if not body_present[0] and (np.asarray(sample["body_dim_valid"]).any()
                                or np.asarray(sample["body_action_valid"]).any()):
        findings.append("an ABSENT body has a valid action step or dim")
    if history != bool(np.asarray(sample["history_view_valid"]).any()):
        findings.append("history_valid disagrees with history_view_valid.any()")
    if "intrinsics" in sample and not np.array_equal(
            np.asarray(sample["history_intrinsics"]), np.asarray(sample["intrinsics"]),
            equal_nan=True):
        findings.append("history_intrinsics is not a copy of intrinsics")
    check_packed("actions", np.asarray(sample["actions"], dtype=np.float64),
                 np.asarray(sample["action_valid"], dtype=bool),
                 np.asarray(sample["dim_valid"], dtype=bool),
                 np.asarray(sample["action_loss_dims"], dtype=bool), alpha, findings)
    check_packed("body_actions", np.asarray(sample["body_actions"], dtype=np.float64),
                 np.asarray(sample["body_action_valid"], dtype=bool),
                 np.asarray(sample["body_dim_valid"], dtype=bool),
                 np.asarray(sample["body_loss_dims"], dtype=bool), alpha, findings)
    noise = np.asarray(sample["state_noise"], dtype=np.float64)
    # The loader's own map from record dims to noise columns (the noise has its
    # own order): joints and hand per dimension, the 6D whole.
    measured = code == STATE_MEASURED
    live = noise_dims(torch.as_tensor(dims)).numpy() & measured[..., None]
    if (noise[~live] != 0).any():
        findings.append("state_noise off a MEASURED record's valid dims")
    body_noise = np.asarray(sample["body_state_noise"], dtype=np.float64)
    if (body_noise[~(body_dims & (body_code == STATE_MEASURED)[:, None])] != 0).any():
        findings.append("body_state_noise off a MEASURED body's valid dims")
    for finding in findings:
        report.fail(f"{name}: {finding}")


def check_loader(root: Path, manifest_path: Path, manifest: dict[str, Any],
                 args: argparse.Namespace, report: Report) -> None:
    """Build the loader against this manifest and draw from every directory.

    Constructing RobotDataset runs the loader's own refusals over the manifest,
    which is the check that matters: a manifest this tool likes and the loader
    rejects is a manifest that fails at the start of a run instead of here.

    Two passes. The weighted one is what a run would draw; the second reads
    every directory's first window, because a directory holding a hundredth of
    the mixture is not reached by any draw count worth waiting for.
    """
    if not args.normaliser:
        report.skip("loader: no --normaliser, and train mode refuses a dataset without one")
        return
    normaliser = stage.under_root(root, args.normaliser)
    alpha = float(json.loads(normaliser.read_text(encoding="utf-8"))["normaliser"]["band_alpha"])
    canvas = manifest["canvas"]
    stamp = manifest["prompt"]["stamp"]
    expected_keys = batch_keys(geometry_of(str(stamp["intrinsics_mode"])), train=True)
    shapes = sample_shapes(
        (canvas["cell"][0], canvas["cell"][1]),
        int(stamp["n_views"]), int(args.n_slots), int(args.chunk),
        int(args.action_dim), str(stamp["intrinsics_mode"]),
    )
    try:
        dataset = RobotDataset(
            manifest=manifest_path, root=root,
            max_grid=(canvas["cell"][0] // canvas["token"],
                      canvas["cell"][1] // canvas["token"]),
            pixels_per_token=canvas["token"],
            n_views=manifest["prompt"]["stamp"]["n_views"],
            n_slots=args.n_slots, chunk=args.chunk, action_dim=args.action_dim,
            gripper_dims=tuple(args.gripper_dims),
            tokenizer_id=manifest["prompt"]["stamp"]["tokenizer_id"],
            max_text_tokens=manifest["prompt"]["stamp"]["max_text_tokens"],
            intrinsics_mode=manifest["prompt"]["stamp"]["intrinsics_mode"],
            pixel_convention="neo_native", epoch_samples=max(args.sample_draws, 1), seed=0,
            normaliser=str(normaliser), history_dropout=args.history_dropout,
            state_noise={"sigma": args.state_noise_sigma,
                         "rot_sigma_rad": args.state_noise_rot_sigma_rad,
                         "prob": args.state_noise_prob},
        )
    except FileNotFoundError as error:
        # A missing artifact is already a failure from check_paths; this only
        # says which check could not run because of it.
        report.skip(f"loader construction: {error}")
        return
    except (TypeError, ValueError) as error:
        # A manifest the loader refuses is the finding this pass exists to
        # report. It used to propagate, and a raise here loses every failure
        # collected before it -- the report is the output, not the exit code.
        report.fail(f"loader construction: {str(error).splitlines()[0]}")
        return
    check_realised_mixture(dataset, args.mixture_draws, args.mixture_seed, report)
    seen = set()
    for index in range(args.sample_draws):
        try:
            sample = dataset[index]
        except FileNotFoundError as error:
            report.skip(f"sample draw: {error}")
            return
        except (KeyError, TypeError, ValueError) as error:
            report.fail(f"sample draw: {str(error).splitlines()[0]}")
            return
        name = dataset.draw(index).name
        seen.add(name)
        keys = set(sample) - {"chat"}
        if keys != expected_keys:
            report.fail(f"{name} sample keys differ from batch_keys: missing "
                        f"{sorted(expected_keys - keys)}, extra {sorted(keys - expected_keys)}")
        for key, shape in shapes.items():
            if key not in sample:
                report.fail(f"sample field {key} is absent, the shape table says {shape}")
                continue
            got = tuple(sample[key].shape)
            if got != shape:
                report.fail(f"sample field {key} is {got}, the shape table says {shape}")
        check_sample(name, sample, args.n_slots, alpha, report)
    logger.info("%d of %d directories reached by %d weighted draws",
                len(seen), len(manifest["datasets"]), args.sample_draws)
    # The weighted draw cannot reach a directory holding a hundredth of the
    # mixture, so every directory is also read at its own first window.
    for index, entry in enumerate(manifest["datasets"]):
        # The draw index is the directory's own, which is a number the
        # reference draw folds on. Any of them reaches a real camera; what this
        # loop checks is that the directory's first window assembles at all.
        first = Draw(index, index, entry["name"], 0, 0)
        try:
            check_sample(entry["name"], dataset.sample_fields(first), args.n_slots, alpha,
                         report)
        except (OSError, ValueError, IndexError, KeyError) as error:
            report.fail(f"{entry['name']} first window: {str(error).splitlines()[0]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--weights", help="the weights stage output, for the share check")
    parser.add_argument("--schema-version", type=int, default=MANIFEST_SCHEMA_VERSION)
    parser.add_argument("--normaliser",
                        help="embodiments.json under the root; the loader's train mode needs it")
    parser.add_argument("--history-dropout", type=float, default=0.1)
    parser.add_argument("--state-noise-sigma", type=float, default=0.02)
    parser.add_argument("--state-noise-rot-sigma-rad", type=float, default=0.02)
    parser.add_argument("--state-noise-prob", type=float, default=0.8)
    parser.add_argument("--sample-frames", type=int, default=4,
                        help="frame index entries read for a null; every name "
                             "the index holds is checked for existence")
    parser.add_argument("--sample-draws", type=int, default=64)
    parser.add_argument("--mixture-draws", type=int, default=20_000,
                        help="first-level draws the realised share is measured over")
    parser.add_argument("--mixture-seed", type=int, default=0)
    # No default. It is model.action.n_slots, it moved with the base slot, and
    # a tool that guesses it verifies a corpus at a slot count no run uses.
    parser.add_argument("--n-slots", type=int, required=True)
    parser.add_argument("--chunk", type=int, default=30)
    parser.add_argument("--action-dim", type=int, default=ACTION_DIM)
    parser.add_argument("--gripper-dims", type=int, nargs="+", default=[GRIPPER_COLUMN])
    parser.add_argument("--require-loader", action="store_true",
                        help="fail rather than skip the checks the loader cannot run yet")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    root = stage.resolve_root(args.root)
    manifest_path = stage.under_root(root, args.manifest)
    manifest = stage.read_json(manifest_path)
    report = Report()

    if manifest["schema_version"] != args.schema_version:
        old = (" (a v6 manifest: 19-wide rows, no joints, hand or body)"
               if manifest["schema_version"] in OLD_MANIFEST_SCHEMA_VERSIONS else "")
        logger.error("manifest schema_version %s%s, this tool reads %s",
                     manifest["schema_version"], old, args.schema_version)
        return 1

    check_paths(root, manifest, report)
    check_inputs(root, manifest, report)
    check_grids(manifest, report)
    check_entries(manifest, report)
    check_totals(manifest, report)
    check_episode_starts(root, manifest, report)
    check_embodiment_arrays(root, manifest, report)
    check_frame_cache(root, manifest, args.sample_frames, report)
    check_prompt_stamp(root, manifest, report)
    check_expected_share(
        root, manifest,
        stage.under_root(root, args.weights) if args.weights else None, report)
    check_intrinsics(root, manifest, report)
    check_loader(root, manifest_path, manifest, args, report)

    for line in report.failures:
        logger.error("%s", line)
    for line in report.skipped:
        logger.warning("skipped %s", line)
    if args.require_loader and report.skipped:
        logger.error("%d checks were skipped and --require-loader is set",
                     len(report.skipped))
        return 1
    if report.failures:
        logger.error("%d failures", len(report.failures))
        return 1
    logger.info("manifest %s verified: %d directories, %d rows, %d checks skipped",
                manifest["corpus"], manifest["totals"]["directories"],
                manifest["totals"]["rows"], len(report.skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
