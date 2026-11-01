"""Shared, read-only helpers for the data release scripts.

Nothing here writes under the data root, follows a symbolic link, or hashes a
payload. Walks start at the literal prefix of each allowlisted pattern, so a
release that names three directories never walks the rest of the corpus.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator

SOURCES_SCHEMA = "neo_vla_data_release_sources_v1"

# Always excluded, whatever the allowlist says: hidden files and directories,
# run and operations artefacts, and the preparation driver's plan, whose argv
# records the absolute corpus root.
DENY_PATTERNS = (
    ".*", "*/.*", "**/.*",
    "*__pycache__*", "*.pyc", "*.log", "*/logs/*", "*.pid", "*.lock", "*.tmp",
    "*.swp", "*~", "nohup.out", "*/nohup.out", "core", "core.*",
    "*/plan.json", "plan.json", "*.hydra*", "*/outputs/*", "*/checkpoints/*",
)

# Bounded look for internal paths and hosts in small text artefacts: the first
# READ_LIMIT bytes of files with these suffixes. A hit excludes the file.
TEXT_SUFFIXES = {".json", ".jsonl", ".yaml", ".yml", ".txt", ".md", ".csv", ".tsv"}
READ_LIMIT = 64 * 1024
INTERNAL_PATTERNS = (
    re.compile(r"(?<![\w.])/(?:mnt|home|Users|data|nfs|afs|cpfs|scratch|lustre|gpfs)/[^\s\"']+"),
    re.compile(r"\b(?:10|172|192)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
)

PLACEHOLDER = re.compile(r"^\s*$|<[^>]*>|\bTODO\b|\bTBD\b|\bFIXME\b", re.IGNORECASE)
REDISTRIBUTION_OK = ("permitted", "permitted_with_conditions")
REDISTRIBUTION_ALL = REDISTRIBUTION_OK + ("not_permitted", "unknown")


def refuse(message: str) -> SystemExit:
    return SystemExit(f"refused: {message}")


def resolve_root(value: str) -> Path:
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise refuse(f"--root {value} is not a directory")
    return root


def outside_root(root: Path, value: str, flag: str) -> Path:
    """An output path that must not land under the data root."""
    path = Path(value).expanduser().resolve()
    if path == root or path.is_relative_to(root):
        raise refuse(f"{flag} {value} is under --root; outputs are written outside the data")
    return path


def read_allowlist(path: Path) -> list[str]:
    patterns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pure = PurePosixPath(line)
        if pure.is_absolute() or ".." in pure.parts:
            raise refuse(f"allowlist pattern {line!r} must be relative to the root without '..'")
        patterns.append(line)
    if not patterns:
        raise refuse(f"allowlist {path} names no pattern")
    return patterns


def matches(relative: str, patterns: tuple[str, ...] | list[str]) -> bool:
    return any(fnmatch.fnmatchcase(relative, pattern) for pattern in patterns)


def denied(relative: str) -> bool:
    parts = relative.split("/")
    if any(part.startswith(".") for part in parts):
        return True
    return matches(relative, DENY_PATTERNS) or matches(parts[-1], DENY_PATTERNS)


def literal_prefix(pattern: str) -> str:
    """The leading path components of a glob that hold no wildcard."""
    parts = []
    for part in PurePosixPath(pattern).parts:
        if any(char in part for char in "*?["):
            break
        parts.append(part)
    return "/".join(parts)


@dataclass(frozen=True)
class Entry:
    relative: str
    kind: str          # "file", "symlink", "other"
    size: int
    target: str | None = None
    escapes: bool = False


def classify(root: Path, path: Path) -> Entry:
    relative = path.relative_to(root).as_posix()
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode):
        target = os.readlink(path)
        resolved = (path.parent / target).resolve()
        escapes = not (resolved == root or resolved.is_relative_to(root))
        # The target is recorded relative to the root when it stays inside;
        # an escaping target is only flagged, never written out in full.
        shown = resolved.relative_to(root).as_posix() if not escapes else None
        return Entry(relative, "symlink", 0, shown, escapes)
    if stat.S_ISREG(info.st_mode):
        return Entry(relative, "file", info.st_size)
    return Entry(relative, "other", 0)


def walk(root: Path, patterns: list[str], max_entries: int) -> Iterator[Entry]:
    """Every entry under the literal prefixes of the patterns, never following links."""
    seen: set[str] = set()
    starts = sorted({literal_prefix(pattern) for pattern in patterns})
    count = 0
    for start in starts:
        base = root / start if start else root
        if base.is_symlink() or not base.exists():
            if base.is_symlink():
                entry = classify(root, base)
                if entry.relative not in seen:
                    seen.add(entry.relative)
                    yield entry
            continue
        if base.is_file():
            candidates = [base]
            walker: Iterator = iter(())
        else:
            candidates = []
            walker = os.walk(base, followlinks=False)
        for path in candidates:
            entry = classify(root, path)
            if entry.relative not in seen:
                seen.add(entry.relative)
                yield entry
        for directory, dirnames, filenames in walker:
            here = Path(directory)
            # Directory links are reported, not entered.
            for name in list(dirnames):
                if (here / name).is_symlink():
                    dirnames.remove(name)
                    filenames.append(name)
            dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))
            for name in sorted(filenames):
                entry = classify(root, here / name)
                if entry.relative in seen:
                    continue
                seen.add(entry.relative)
                count += 1
                if count > max_entries:
                    raise refuse(f"more than {max_entries} entries under the allowlisted "
                                 f"prefixes; narrow the allowlist or raise --max-entries")
                yield entry


def internal_reference(root: Path, relative: str) -> str | None:
    """The first internal-looking path or private address in a small text head, or None."""
    path = root / relative
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return None
    with open(path, "rb") as handle:
        head = handle.read(READ_LIMIT).decode("utf-8", errors="replace")
    for pattern in INTERNAL_PATTERNS:
        found = pattern.search(head)
        if found:
            return found.group(0)[:80]
    return None


def load_sources(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != SOURCES_SCHEMA:
        raise refuse(f"{path} schema {document.get('schema')!r}, expected {SOURCES_SCHEMA!r}")
    if not isinstance(document.get("sources"), list) or not document["sources"]:
        raise refuse(f"{path} lists no sources")
    names = [entry.get("source") for entry in document["sources"]]
    if len(set(names)) != len(names):
        raise refuse(f"{path} names a source twice")
    return document


def source_problems(entry: dict) -> list[str]:
    """Why one source entry may not be uploaded; empty when it may."""
    problems = []
    if entry.get("approved_for_upload") is not True:
        problems.append("approved_for_upload is not true")
    for field in ("source", "upstream_name", "upstream_url", "license_id", "attribution",
                  "approved_by"):
        value = entry.get(field)
        if not isinstance(value, str) or PLACEHOLDER.search(value):
            problems.append(f"{field} is empty or a placeholder")
    if entry.get("redistribution") not in REDISTRIBUTION_ALL:
        problems.append(f"redistribution must be one of {REDISTRIBUTION_ALL}")
    elif entry["redistribution"] not in REDISTRIBUTION_OK:
        problems.append(f"redistribution is {entry['redistribution']!r}")
    if entry.get("redistribution") == "permitted_with_conditions":
        conditions = entry.get("conditions")
        if not isinstance(conditions, str) or PLACEHOLDER.search(conditions):
            problems.append("conditions must state the upstream conditions")
    evidence = entry.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        problems.append("evidence lists nothing")
    else:
        for index, item in enumerate(evidence):
            for field in ("kind", "reference", "checked_on", "checked_by"):
                value = item.get(field) if isinstance(item, dict) else None
                if not isinstance(value, str) or PLACEHOLDER.search(value):
                    problems.append(f"evidence[{index}].{field} is empty or a placeholder")
            checked = item.get("checked_on") if isinstance(item, dict) else None
            if isinstance(checked, str) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", checked):
                problems.append(f"evidence[{index}].checked_on is not YYYY-MM-DD")
    for field in ("paths", "license_files"):
        value = entry.get(field)
        if not isinstance(value, list) or not value or not all(
                isinstance(item, str) and item and not PLACEHOLDER.search(item) for item in value):
            problems.append(f"{field} must list root-relative patterns")
    return problems
