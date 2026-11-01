"""What every data-preparation stage shares: the root, the header, the refusal.

One root. Every path a stage writes is stated relative to it, so moving the
corpus changes one environment variable and nothing else. Every stage output
carries a provenance header (stage.header): the tool, its version, the commit
and the root-relative inputs it read.

Writing is idempotent by comparison, not by luck: a stage serialises its
payload without the timestamp, compares that with what is already on disk, and
leaves the file untouched when the two agree. So re-running a stage on
unchanged inputs leaves byte-identical output and a changed input is the only
thing that moves the file.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

# The layout of the stage outputs in this module's format. Bumped when a field
# moves; the verify stage refuses a version it does not know.
SCHEMA_VERSION = 1

# The version every stage output states. 2.0.0 is the v7 layout (joints, hand
# and body; manifest schema 3; embodiments v2): a header another version wrote
# is stale, so every output built under v6 (1.1.0) is redone, and the stage-1
# sidecars and the loader that reads them hold this same version.
TOOL_VERSION = "2.0.0"

# What hides a provenance header from a reader of a plain text listing. The
# stages that read a directory list already skip a line starting with it, so a
# listing carries the same header the json outputs do without changing them.
LISTING_COMMENT = "# "

# A LIBERO-family directory: its source or name begins with one of these
# (config/layouts/libero_v7.json gives libero/, libero_nom/ and the tpr6 legs
# source 'libero'). Its state position is world-frame, so every arm states the
# per-episode base offset the loader subtracts (LIBERO F3): build_manifest
# refuses to write one without, and verify_corpus check (x) refuses a manifest
# that holds one without.
LIBERO_FAMILY = ("libero",)


def is_libero_family(entry: dict[str, Any]) -> bool:
    """Whether a manifest entry (or any mapping with source and name) is LIBERO's."""
    return any(str(entry.get(field, "")).startswith(word)
               for field in ("source", "name") for word in LIBERO_FAMILY)


def git_commit(tree: Path) -> str:
    """The commit the tree is at, or "dirty" when no commit names it.

    "dirty" covers both an edited checkout and a directory that is not one:
    in either case there is no commit that reproduces what ran.
    """
    try:
        head = subprocess.run(
            ["git", "-C", str(tree), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return "dirty"
    if head.returncode != 0:
        return "dirty"
    status = subprocess.run(
        ["git", "-C", str(tree), "status", "--porcelain"],
        capture_output=True, text=True, check=False,
    )
    if status.returncode != 0 or status.stdout.strip():
        return "dirty"
    return head.stdout.strip()


def resolve_root(root: str | os.PathLike) -> Path:
    """The corpus root as an absolute path, refusing one that does not exist."""
    resolved = Path(root).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"root {resolved} is not a directory\n"
            f"expected the corpus root holding the source trees\n"
            f"pass --root to the directory the manifest's paths hang off"
        )
    return resolved


def under_root(root: Path, path: str | os.PathLike) -> Path:
    """A root-relative or absolute path -> the absolute path, checked.

    Symbolic links are not followed here. A corpus is normally staged as a tree
    of links into the trees it was downloaded to, and following them would put
    every directory outside the root it was staged under.
    """
    candidate = Path(path)
    joined = Path(os.path.normpath(candidate if candidate.is_absolute() else root / candidate))
    if not joined.is_relative_to(root):
        raise ValueError(
            f"path {joined} is outside the root\n"
            f"expected a path under {root}\n"
            f"move the file under the root or pass a different --root"
        )
    return joined


def relative(root: Path, path: Path) -> str:
    """An absolute path -> the string the stage output records for it."""
    return str(Path(os.path.normpath(path)).relative_to(root))


def refuse_write_into_input(out: Path, inputs: Iterable[Path]) -> None:
    """Raise when the output would land inside a tree the stage reads.

    The raw trees are read-only for this pipeline. A stage that writes into one
    changes the inputs its own header names, and the next run would read
    something other than what produced the data. Both sides are resolved
    through their links here, so a staged tree cannot be written into by way of
    the path it was downloaded to.
    """
    out_dir = out.resolve().parent
    for source in inputs:
        source_dir = Path(source).resolve()
        if not source_dir.is_dir():
            source_dir = source_dir.parent
        if out_dir == source_dir or out_dir.is_relative_to(source_dir):
            raise ValueError(
                f"output {out} is inside input {source_dir}\n"
                f"expected an output directory outside every input tree\n"
                f"pass --out under a preparation directory of the root"
            )


def header(
    tool: str,
    *,
    root: Path,
    inputs: dict[str, Path],
    tree: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The provenance header, with every input named by its root-relative path."""
    record = {
        "schema_version": SCHEMA_VERSION,
        "tool": tool,
        "tool_version": TOOL_VERSION,
        "git_commit": git_commit(tree),
        "inputs": sorted(relative(root, Path(path)) for path in inputs.values()),
        "created": None,
    }
    if extra:
        record["settings"] = extra
    return record


def dumps_json(payload: Any) -> str:
    """One serialisation for every stage, so two runs produce the same bytes."""
    return json.dumps(payload, sort_keys=True, indent=1, ensure_ascii=True) + "\n"


def dumps_line(payload: Any) -> str:
    """One record of a jsonl stage output."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"


def _stamped(payload: dict[str, Any], provenance_key: str) -> dict[str, Any]:
    stamped = dict(payload)
    stamped[provenance_key] = dict(payload[provenance_key])
    stamped[provenance_key]["created"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    return stamped


def _unchanged(
    path: Path, body: str, provenance_key: str, first_line_only: bool, prefix: str = ""
) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    try:
        if first_line_only:
            head, _, rest = text.partition("\n")
            if not head.startswith(prefix):
                return False
            existing = json.loads(head[len(prefix):])
            existing[provenance_key]["created"] = None
            return prefix + dumps_line(existing) + rest == body
        existing = json.loads(text)
        existing[provenance_key]["created"] = None
        return dumps_json(existing) == body
    except (json.JSONDecodeError, KeyError, TypeError):
        return False


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, payload: dict[str, Any], *, provenance_key: str = "provenance",
               exclusive: bool = False) -> bool:
    """Write one json stage output. False when the bytes on disk already match."""
    body = dumps_json(payload)
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(dumps_json(_stamped(payload, provenance_key)))
        return True
    if _unchanged(path, body, provenance_key, first_line_only=False):
        return False
    _write_atomic(path, dumps_json(_stamped(payload, provenance_key)))
    return True


def write_jsonl(path: Path, head: dict[str, Any], records: Sequence[Any], *,
                exclusive: bool = False) -> bool:
    """Write one jsonl stage output, header first. False when it already matches."""
    tail = "".join(dumps_line(record) for record in records)
    body = dumps_line({"provenance": head}) + tail
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(dumps_line(_stamped({"provenance": head}, "provenance")) + tail)
        return True
    if _unchanged(path, body, "provenance", first_line_only=True):
        return False
    stamped = _stamped({"provenance": head}, "provenance")
    _write_atomic(path, dumps_line(stamped) + tail)
    return True


def write_listing(path: Path, head: dict[str, Any], lines: Sequence[str]) -> bool:
    """Write one text listing, header first. False when it already matches.

    The listing a later stage takes as its directory list, so it is that
    stage's input: it has to be byte-stable on unchanged input for the same
    reason every other stage output is.
    """
    tail = "".join(f"{line}\n" for line in lines)
    body = LISTING_COMMENT + dumps_line({"provenance": head}) + tail
    if _unchanged(path, body, "provenance", True, LISTING_COMMENT):
        return False
    stamped = _stamped({"provenance": head}, "provenance")
    _write_atomic(path, LISTING_COMMENT + dumps_line(stamped) + tail)
    return True


def read_jsonl(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """A jsonl stage output -> its header and its records, in file order."""
    with open(path, encoding="utf-8") as handle:
        first = handle.readline()
        if not first:
            raise ValueError(
                f"{path} is empty\n"
                f"expected a provenance header on line 1\n"
                f"re-run the stage that writes it"
            )
        head = json.loads(first)["provenance"]
        records = [json.loads(line) for line in handle if line.strip()]
    return head, records


def read_json(path: Path) -> dict[str, Any]:
    """A json stage output, whole."""
    return json.loads(path.read_text(encoding="utf-8"))


def check_inputs_match(head: dict[str, Any], root: Path) -> list[str]:
    """Which of a header's recorded inputs are no longer under the root.

    The header names its inputs; an older header maps each to a checksum,
    which is not read. Content is not compared: after editing an input in
    place, run its stages again (prepare_data.py --force or --only).
    """
    return [f"{name}: missing" for name in sorted(head["inputs"]) if not (root / name).exists()]


def check_artifacts_match(payload: dict[str, Any], root: Path) -> list[str]:
    """Which of a stage output's own artifacts are no longer on disk.

    The inputs half says what the stage read; this says the answer is still
    there. A stage output that names no artifacts states nothing about any
    file and returns nothing. An older output maps each artifact to a
    checksum, which is not read.
    """
    return [f"{name}: missing" for name in sorted(payload.get("artifacts", ()))
            if not (root / name).exists()]
