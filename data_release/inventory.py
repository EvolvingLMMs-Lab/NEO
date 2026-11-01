"""List what a release candidate holds, without reading payloads or changing anything.

Walks only the literal prefixes of the allowlisted patterns under --root, using
lstat: path, size, file or link, and whether a link escapes the root. Flags the
entries a release must not carry (hidden or operations artefacts, links,
escaping links, internal paths in small text heads). Writes one JSON report
outside the root; the root itself is recorded only as a label.

Usage:
    python3 data_release/inventory.py --root <DATA_ROOT> \
        --allowlist data_release/allowlist.example.txt --out <OUTSIDE>/inventory.json
"""
from __future__ import annotations

import argparse
import datetime
import json
from collections import Counter
from pathlib import Path

import release_lib as lib


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, help="the prepared data root (never written)")
    parser.add_argument("--allowlist", required=True,
                        help="root-relative glob patterns, one per line")
    parser.add_argument("--out", required=True, help="the report, outside --root")
    parser.add_argument("--root-label", default="<DATA_ROOT>",
                        help="what the report calls the root instead of its absolute path")
    parser.add_argument("--max-entries", type=int, default=200_000)
    parser.add_argument("--check-text", action="store_true",
                        help="look for internal paths in the first 64 KiB of small text files")
    args = parser.parse_args(argv)

    root = lib.resolve_root(args.root)
    out = lib.outside_root(root, args.out, "--out")
    patterns = lib.read_allowlist(Path(args.allowlist))

    entries, flags = [], Counter()
    for entry in lib.walk(root, patterns, args.max_entries):
        record = {"path": entry.relative, "kind": entry.kind, "bytes": entry.size,
                  "allowlisted": lib.matches(entry.relative, patterns)}
        notes = []
        if lib.denied(entry.relative):
            notes.append("hidden_or_ops_artefact")
        if entry.kind == "symlink":
            notes.append("symlink_escapes_root" if entry.escapes else "symlink")
            if entry.target is not None:
                record["target"] = entry.target
        if entry.kind == "other":
            notes.append("not_a_regular_file")
        if args.check_text and entry.kind == "file" and record["allowlisted"]:
            hit = lib.internal_reference(root, entry.relative)
            if hit:
                notes.append("internal_reference")
                record["internal_reference"] = hit
        record["flags"] = notes
        flags.update(notes)
        entries.append(record)

    allowlisted = [record for record in entries if record["allowlisted"]]
    report = {
        "schema": "neo_vla_data_release_inventory_v1",
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "root": args.root_label,
        "allowlist": patterns,
        "method": "lstat only; no payload read or hashed; links never followed",
        "totals": {"entries": len(entries), "allowlisted": len(allowlisted),
                   "allowlisted_bytes": sum(record["bytes"] for record in allowlisted),
                   "flags": dict(sorted(flags.items()))},
        "entries": entries,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"{len(entries)} entries, {len(allowlisted)} allowlisted, flags {dict(flags)}; "
          f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
