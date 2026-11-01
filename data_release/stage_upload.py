"""Plan a private dataset upload from an explicit source manifest and allowlist; upload only on --upload.

Default is a dry run: it walks the allowlisted prefixes under --root (lstat
only, links never followed), decides per file whether it may be released, and
writes a plan outside the root. Nothing is copied, hashed or changed. A file is
planned only when it is allowlisted, is a regular file, is not a hidden or
operations artefact, carries no internal path in its text head, and belongs to
exactly one source whose entry in the source manifest is approved with licence
evidence (or to the manifest's corpus-level `common_files`). The plan is
blocked when a required provenance or mask file, or a source's licence file,
is missing, or when a planned corpus manifest references a source that is not
approved.

--upload sends the planned files to a Hugging Face dataset repository that is
private and stays private: it creates the repository private, refuses one that
is not, and needs --confirm-repo-id equal to --repo-id. Authentication is the
huggingface_hub client's own; this script never reads or prints a token.

Usage:
    python3 data_release/stage_upload.py --root <DATA_ROOT> \
        --sources <OUTSIDE>/sources_manifest.json --allowlist <OUTSIDE>/allowlist.txt \
        --plan-out <OUTSIDE>/release_plan.json
    python3 data_release/stage_upload.py ... --upload --card <OUTSIDE>/README.md \
        --repo-id <org>/<name> --confirm-repo-id <org>/<name>
"""
from __future__ import annotations

import argparse
import datetime
import fnmatch
import json
import re
from collections import Counter
from pathlib import Path

import release_lib as lib

MANIFEST_PATTERN = "manifests/*.json"
CARD_PLACEHOLDER = "<FILL"


def plan_release(root: Path, sources: dict, patterns: list[str], max_entries: int,
                 check_text: bool) -> dict:
    approved, refused = {}, {}
    for entry in sources["sources"]:
        problems = lib.source_problems(entry)
        (refused if problems else approved)[entry.get("source")] = problems or entry
    common = list(sources.get("common_files") or [])

    planned, excluded, blockers = [], [], []
    for entry in lib.walk(root, patterns, max_entries):
        relative = entry.relative
        if not lib.matches(relative, patterns):
            continue

        def drop(reason: str, **detail) -> None:
            excluded.append({"path": relative, "reason": reason, **detail})

        if lib.denied(relative):
            drop("hidden_or_ops_artefact")
            continue
        if entry.kind == "symlink":
            if entry.escapes:
                blockers.append(f"{relative}: symbolic link escapes the root")
            drop("symlink_escapes_root" if entry.escapes else "symlink")
            continue
        if entry.kind != "file":
            drop("not_a_regular_file")
            continue
        if check_text:
            hit = lib.internal_reference(root, relative)
            if hit:
                drop("internal_reference")
                continue
        owners = sorted(name for name, item in approved.items()
                        if lib.matches(relative, item["paths"] + item["license_files"]))
        owners_refused = sorted(
            str(item.get("source")) for item in sources["sources"]
            if item.get("source") in refused
            and lib.matches(relative, [p for p in (item.get("paths") or []) if isinstance(p, str)]
                            + [p for p in (item.get("license_files") or [])
                               if isinstance(p, str)]))
        if owners_refused:
            drop("source_not_approved", sources=owners_refused)
            continue
        if len(owners) > 1:
            drop("ambiguous_source", sources=owners)
            continue
        if not owners and not lib.matches(relative, common):
            drop("no_source")
            continue
        planned.append({"path": relative, "bytes": entry.size,
                        "source": owners[0] if owners else "common"})

    paths = [item["path"] for item in planned]
    for pattern in sources.get("required_companions") or []:
        if not any(fnmatch.fnmatchcase(path, pattern) for path in paths):
            blockers.append(f"required provenance or mask file {pattern!r} is not planned")
    for name, entry in approved.items():
        for pattern in entry["license_files"]:
            if not any(fnmatch.fnmatchcase(path, pattern) for path in paths):
                blockers.append(f"{name}: licence file {pattern!r} is not planned")
    # A corpus manifest is released only when every directory it names is
    # covered by an approved source's paths. Checked per directory rather than
    # per manifest source word, because one word (oxe) spans datasets whose
    # upstream licences differ. A directory is covered when an approved
    # pattern matches its meta/info.json.
    covering = [pattern for entry in approved.values() for pattern in entry["paths"]]
    for path in paths:
        if fnmatch.fnmatchcase(path, MANIFEST_PATTERN):
            document = json.loads((root / path).read_text(encoding="utf-8"))
            uncovered = Counter(
                str(item.get("source")) for item in document.get("datasets", [])
                if not lib.matches(f"{item.get('path')}/meta/info.json", covering))
            if uncovered:
                blockers.append(
                    f"{path} names directories no approved source covers: "
                    + ", ".join(f"{name} ({count})" for name, count in sorted(uncovered.items()))
                    + "; derive a manifest restricted to approved sources first")
    if not planned:
        blockers.append("no file is planned")

    return {
        "approved_sources": sorted(approved),
        "refused_sources": {name: problems for name, problems in sorted(refused.items())},
        "planned": planned,
        "excluded": excluded,
        "blockers": blockers,
        "totals": {"planned": len(planned), "planned_bytes": sum(i["bytes"] for i in planned),
                   "excluded": dict(sorted(Counter(i["reason"] for i in excluded).items()))},
    }


def upload(root: Path, plan: dict, args: argparse.Namespace, sources_path: Path,
           plan_path: Path) -> None:
    """Send the planned files to a private dataset repository, in batched commits."""
    if args.repo_id != args.confirm_repo_id:
        raise lib.refuse("--confirm-repo-id must repeat --repo-id exactly")
    if not args.card:
        raise lib.refuse("--upload needs --card, the dataset card uploaded as README.md")
    card = Path(args.card).read_text(encoding="utf-8")
    if CARD_PLACEHOLDER in card:
        raise lib.refuse(f"the card still holds {CARD_PLACEHOLDER} placeholders")
    card_path = Path(args.card).expanduser().resolve()
    hit = lib.internal_reference(card_path.parent, card_path.name)
    if hit:
        raise lib.refuse(f"the card holds an internal path or address ({hit})")
    for item in plan["planned"]:
        payload = root / item["path"]
        resolved = payload.resolve(strict=True)
        linked = payload.is_symlink() or any(
            part.is_symlink() for part in payload.parents if part.is_relative_to(root))
        if linked or not resolved.is_relative_to(root) or not payload.is_file():
            raise lib.refuse(f"planned file changed or follows a symbolic link: {item['path']}")
        if payload.stat().st_size != item["bytes"]:
            raise lib.refuse(f"planned file size changed: {item['path']}")
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError as error:
        raise lib.refuse("--upload needs the huggingface_hub package") from error

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)
    if api.repo_info(args.repo_id, repo_type="dataset").private is not True:
        raise lib.refuse(f"{args.repo_id} exists and is not private; nothing was uploaded")

    extras = [("README.md", Path(args.card)), ("SOURCES.json", sources_path),
              ("RELEASE_PLAN.json", plan_path)]
    for name in ("corpus_summary.json", "sources_inventory.json", "UPSTREAM_TERMS.md",
                 "upstream_license_metadata.json", "release_readiness.json"):
        companion = Path(__file__).resolve().parent / name
        if companion.is_file():
            extras.append((name, companion))
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))
                  for name, path in extras]
    operations += [CommitOperationAdd(path_in_repo=item["path"],
                                      path_or_fileobj=str(root / item["path"]))
                   for item in plan["planned"]]
    for start in range(0, len(operations), args.batch_size):
        batch = operations[start:start + args.batch_size]
        api.create_commit(args.repo_id, operations=batch, repo_type="dataset",
                          commit_message=f"release files {start + 1}-{start + len(batch)}")
        print(f"committed {start + len(batch)} of {len(operations)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, help="the prepared data root (never written)")
    parser.add_argument("--sources", required=True, help="the filled source manifest")
    parser.add_argument("--allowlist", required=True, help="root-relative glob patterns")
    parser.add_argument("--plan-out", required=True, help="the plan, written outside --root")
    parser.add_argument("--root-label", default="NEO-VLA-Action")
    parser.add_argument("--max-entries", type=int, default=200_000)
    parser.add_argument("--no-check-text", action="store_true",
                        help="skip the internal-path look at small text heads")
    parser.add_argument("--upload", action="store_true",
                        help="upload the plan to a private dataset repository")
    parser.add_argument("--repo-id")
    parser.add_argument("--confirm-repo-id")
    parser.add_argument("--card", help="the filled dataset card, uploaded as README.md")
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args(argv)

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_. -]{0,127}", args.root_label):
        raise lib.refuse("--root-label must be a short release name, without path separators")
    root = lib.resolve_root(args.root)
    plan_path = lib.outside_root(root, args.plan_out, "--plan-out")
    sources_path = Path(args.sources).expanduser().resolve()
    sources = lib.load_sources(sources_path)
    patterns = lib.read_allowlist(Path(args.allowlist))
    if args.upload and not args.repo_id:
        raise lib.refuse("--upload needs --repo-id")
    hit = lib.internal_reference(sources_path.parent, sources_path.name)
    if hit:
        raise lib.refuse(f"the source manifest holds an internal path or address ({hit})")

    plan = plan_release(root, sources, patterns, args.max_entries, not args.no_check_text)
    document = {
        "schema": "neo_vla_data_release_plan_v1",
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "root": args.root_label,
        "release_name": sources.get("release_name"),
        "visibility": "private",
        "allowlist": patterns,
        "method": "lstat only; no payload copied or hashed; links never followed",
        **plan,
    }
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    print(f"planned {plan['totals']['planned']} files ({plan['totals']['planned_bytes']} bytes); "
          f"excluded {plan['totals']['excluded']}; approved sources {plan['approved_sources']}; "
          f"wrote {plan_path}")
    for name, problems in plan["refused_sources"].items():
        print(f"source {name} not approved: {'; '.join(problems)}")
    for blocker in plan["blockers"]:
        print(f"blocked: {blocker}")
    if plan["blockers"]:
        return 2
    if args.upload:
        upload(root, plan, args, sources_path, plan_path)
    else:
        print("dry run: nothing uploaded (pass --upload to send this plan to a private repository)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
