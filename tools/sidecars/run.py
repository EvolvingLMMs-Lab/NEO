"""Write the stage-1 sidecars: extract the RH20T archives, then build per directory.

Two commands. rh20t-extract streams the lowdim archives once each, in
parallel. build writes one sidecar per manifest entry of the chosen families,
largest first over a process pool, and is resumable: an entry whose receipt
matches the bytes on disk is skipped, and a directory whose source does not
line up with it is refused and named in the report, never written. --shard
K/N splits the entries by rows between nodes. A post-train corpus (no manifest
yet) names its directories by --layouts L --dirs D1,D2 instead (layout_entries).

Usage:
    python tools/sidecars/run.py rh20t-extract --root R --manifest M \
        --archive A1 --archive A2 ... [--out O]
    python tools/sidecars/run.py build --root R --manifest M --families agibot,oxe \
        --agibot-raw P --rh20t-source-root P [--droid-raw P] [--shard K/N] [--out O]
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path[:0] = [str(Path(__file__).resolve().parents[2]),
                    str(Path(__file__).resolve().parents[1])]

import layouts as layout_table  # noqa: E402
import stage  # noqa: E402

import neo_vla  # noqa: E402
from sidecars import agibot, common, droid, libero, oxe, rh20t  # noqa: E402

logger = logging.getLogger("sidecars")

REPO = Path(__file__).resolve().parents[2]

# Manifest source -> family. libero is the post-train LIBERO family.
FAMILIES = {"agibot": "agibot", "droid3cam": "droid", "rh20t": "rh20t", "oxe": "oxe",
            "libero": "libero"}
REPORTS = "_reports"
WORK = "_work"


def check_tree() -> None:
    """Refuse to run against a neo_vla that is not this tree's own."""
    library = Path(neo_vla.__file__).resolve().parent
    if library != REPO / "neo_vla":
        raise RuntimeError(
            f"neo_vla is imported from {library}\n"
            f"expected {REPO / 'neo_vla'}\n"
            "put this tree first on PYTHONPATH"
        )


def family_of(entry: dict[str, Any]) -> str:
    source = entry.get("source")
    if source not in FAMILIES:
        raise ValueError(
            f"entry {entry.get('name')} has source {source!r}\n"
            f"expected one of {sorted(FAMILIES)}\n"
            "add the family to tools/sidecars before writing its sidecars"
        )
    return FAMILIES[source]


def shard_of(entries: list[dict[str, Any]], shard: tuple[int, int]) -> list[dict[str, Any]]:
    """Shard K of N: the largest entry into the lightest shard, in turn."""
    index, count = shard
    loads = [0] * count
    mine = []
    for entry in sorted(entries, key=lambda item: (-int(item["rows"]), item["name"])):
        lightest = loads.index(min(loads))
        loads[lightest] += int(entry["rows"])
        if lightest == index:
            mine.append(entry)
    return mine


def parse_shard(text: str) -> tuple[int, int]:
    index, _, count = text.partition("/")
    if not (index.isdigit() and count.isdigit() and int(index) < int(count)):
        raise ValueError(
            f"--shard is {text!r}\n"
            "expected K/N with 0 <= K < N\n"
            "fix --shard"
        )
    return int(index), int(count)


def build_entry(job: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    """Write one entry's sidecar, or say why its directory was refused."""
    entry, settings = job
    out = Path(settings["out"])
    name = entry["name"]
    began = time.time()
    record = common.current(out, name, settings["git_commit"])
    if record is not None:
        return {"name": name, "family": record["family"], "status": "current",
                "rows": record["rows"], "seconds": 0.0}
    family = family_of(entry)
    directory = Path(settings["root"]) / entry["path"]
    try:
        if family == "agibot":
            built = agibot.build(entry, directory, Path(settings["agibot_raw"]))
        elif family == "droid":
            raw = settings.get("droid_raw")
            built = droid.build(entry, directory, Path(raw) if raw else None)
        elif family == "rh20t":
            built = rh20t.build(entry, directory, Path(settings["rh20t_work"]),
                                Path(settings["rh20t_source_root"]))
        elif family == "libero":
            built = libero.build(entry, directory)
        else:
            built = oxe.build(entry, directory)
        receipt = common.write_built(out, entry, family, built, settings["git_commit"])
    except (common.Misaligned, OSError, ValueError, KeyError, StopIteration) as error:
        return {"name": name, "family": family, "status": "refused",
                "error": f"{type(error).__name__}: {error}",
                "seconds": round(time.time() - began, 1)}
    return {"name": name, "family": family, "status": "written", "rows": receipt["rows"],
            "bytes": receipt["bytes"], "seconds": round(time.time() - began, 1)}


def read_entries(root: Path, manifest: Path) -> list[dict[str, Any]]:
    entries = stage.read_json(stage.under_root(root, manifest))["datasets"]
    for entry in entries:
        if entry["name"] != Path(entry["path"]).name:
            raise ValueError(
                f"entry {entry['name']} sits at {entry['path']}\n"
                "expected an entry named by its directory, as every stage-1 entry is\n"
                "check the manifest"
            )
    return entries


def layout_entries(root: Path, layouts: Path, dirs: list[str]) -> list[dict[str, Any]]:
    """The entries a layout gives the named directories, as a manifest would list them.

    A post-train corpus has no manifest before its sidecars (build_manifest reads them),
    so build --layouts L --dirs D1,D2,... (for example the LIBERO family with
    its own layout file and --out <root>/metadata/<corpus>/sidecars, which has no
    default: the action-pretraining corpus is never its target) names them here. name and source are
    the layout's (tools/layouts.resolve), path the directory and rows its episodes'
    parquet rows; the name must be the directory's own, as read_entries holds a manifest's.
    """
    document = layout_table.load(layouts)
    entries = []
    for directory in dirs:
        resolved = layout_table.resolve(document, directory)
        if resolved.name != Path(directory).name:
            raise ValueError(
                f"{layouts} names {directory} {resolved.name}\n"
                "expected an entry named by its directory\n"
                "check the layout's name_parts")
        rows = sum(episode.rows for episode in common.directory_episodes(root / directory))
        entries.append({"name": resolved.name, "path": directory, "source": resolved.source,
                        "rows": rows})
    return entries


def run_build(args: argparse.Namespace) -> int:
    root = stage.resolve_root(args.root)
    if (args.manifest is None) == (args.layouts is None):
        raise ValueError("build takes --manifest, or --layouts with --dirs\n"
                         "expected exactly one source of entries\n"
                         "pass one of them")
    if args.layouts is not None and (not args.dirs or not args.out):
        raise ValueError("--layouts needs --dirs and --out\n"
                         "expected the directories to write and where (no default: the "
                         "action-pretraining corpus's folder is never a post-train corpus's)\n"
                         "pass --dirs D1,D2,... and --out <root>/metadata/<corpus>/sidecars")
    out = Path(args.out) if args.out else common.default_out(root)
    families = [name.strip() for name in args.families.split(",") if name.strip()]
    unknown = sorted(set(families) - set(FAMILIES.values()))
    if unknown:
        raise ValueError(
            f"--families names {unknown}\n"
            f"expected a subset of {sorted(set(FAMILIES.values()))}\n"
            "fix --families"
        )
    listed = (read_entries(root, Path(args.manifest)) if args.manifest is not None else
              layout_entries(root, Path(args.layouts),
                             [name.strip() for name in args.dirs.split(",") if name.strip()]))
    entries = [entry for entry in listed if family_of(entry) in families]
    if args.only:
        names = {name.strip() for name in args.only.split(",") if name.strip()}
        absent = sorted(names - {entry["name"] for entry in entries})
        if absent:
            raise ValueError(
                f"--only names {absent}, which are not entries of {families}\n"
                "expected manifest entry names\n"
                "fix --only or --families"
            )
        entries = [entry for entry in entries if entry["name"] in names]
    if args.shard:
        entries = shard_of(entries, parse_shard(args.shard))
    if "agibot" in families and not args.agibot_raw:
        raise ValueError("agibot needs --agibot-raw\nexpected the proprio_stats root\n"
                         "pass --agibot-raw")
    if "rh20t" in families and not args.rh20t_source_root:
        raise ValueError("rh20t needs --rh20t-source-root\nexpected the root the grid's "
                         "v3 source directories hang off\npass --rh20t-source-root")
    for entry in entries:
        stage.refuse_write_into_input(common.sidecar_path(out, entry["name"]),
                                      [root / entry["path"]])
    settings = {"root": str(root), "out": str(out), "agibot_raw": args.agibot_raw,
                "droid_raw": args.droid_raw,
                "rh20t_work": args.rh20t_work or str(out / WORK / "rh20t"),
                "rh20t_source_root": args.rh20t_source_root,
                "git_commit": stage.git_commit(REPO)}
    logger.info("%d entries of %s (shard %s) -> %s, %d workers, tree %s",
                len(entries), families, args.shard or "all", out, args.workers,
                settings["git_commit"])
    if not entries:
        raise ValueError("no entries to write\nexpected at least one manifest entry "
                         "of the chosen families\ncheck --families and --shard")
    jobs = [(entry, settings) for entry in
            sorted(entries, key=lambda item: (-int(item["rows"]), item["name"]))]
    results = []
    with multiprocessing.get_context("fork").Pool(args.workers, maxtasksperchild=1) as pool:
        for result in pool.imap_unordered(build_entry, jobs):
            results.append(result)
            logger.info("%d/%d %s %s rows=%s %.1fs%s", len(results), len(jobs),
                        result["status"], result["name"], result.get("rows"),
                        result["seconds"],
                        f" {result['error']}" if "error" in result else "")
    label = "_".join(families) + (f"_shard{args.shard.replace('/', 'of')}" if args.shard else "")
    label += "_only" if args.only else ""
    report = out / REPORTS / f"build_{label}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(stage.dumps_json({
        "tool_version": common.TOOL_VERSION, "families": families, "shard": args.shard,
        "entries": len(jobs), "git_commit": settings["git_commit"],
        "status": {status: sum(r["status"] == status for r in results)
                   for status in ("written", "current", "refused")},
        "results": sorted(results, key=lambda item: item["name"]),
    }), encoding="utf-8")
    refused = [result for result in results if result["status"] == "refused"]
    logger.info("done: %d written, %d current, %d refused; report %s",
                sum(r["status"] == "written" for r in results),
                sum(r["status"] == "current" for r in results), len(refused), report)
    return 1 if refused else 0


def extract_one(job: tuple[str, list[str], str]) -> dict[str, Any]:
    archive, wanted, work = job
    path = Path(archive)
    began = time.time()
    index = rh20t.extracted(path, Path(work))
    status = "current"
    if index is None:
        index = rh20t.extract(path, set(wanted), Path(work))
        status = "extracted"
    return {"archive": archive, "status": status, "seconds": round(time.time() - began, 1),
            "scenes_wanted": index["scenes_wanted"],
            "scenes_with_joint": index["scenes_with_joint"],
            "scenes_with_tcp_base": index["scenes_with_tcp_base"],
            "unreadable": len(index["unreadable"])}


def run_extract(args: argparse.Namespace) -> int:
    root = stage.resolve_root(args.root)
    out = Path(args.out) if args.out else common.default_out(root)
    work = Path(args.rh20t_work) if args.rh20t_work else out / WORK / "rh20t"
    entries = [entry for entry in read_entries(root, Path(args.manifest))
               if family_of(entry) == "rh20t"]
    wanted = rh20t.wanted_scenes(root / entry["path"] for entry in entries)
    jobs = []
    for archive in args.archive:
        config = Path(archive).name.split(".", 1)[0]
        if config not in wanted:
            raise ValueError(
                f"{archive} is configuration {config}, which no grid directory uses\n"
                f"expected one of {sorted(wanted)}\n"
                "pass only the archives the manifest's grid directories need"
            )
        jobs.append((archive, sorted(wanted[config]), str(work)))
    missing = sorted(set(wanted) - {Path(a).name.split(".", 1)[0] for a in args.archive})
    logger.info("%d archives -> %s; scenes wanted per configuration %s; not passed %s",
                len(jobs), work, {k: len(v) for k, v in sorted(wanted.items())}, missing)
    with multiprocessing.get_context("fork").Pool(len(jobs)) as pool:
        results = pool.map(extract_one, jobs)
    for result in results:
        logger.info("%s", json.dumps(result, sort_keys=True))
    report = out / REPORTS / "rh20t_extract.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(stage.dumps_json({"results": results, "not_passed": missing}),
                      encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "rh20t-extract"):
        command = commands.add_parser(name)
        command.add_argument("--root", required=True)
        command.add_argument("--manifest", required=name == "rh20t-extract",
                             help="a manifest whose datasets name the stage-1 directories")
        command.add_argument("--out", help="default: <root>/metadata/action_pretrain/sidecars")
        command.add_argument("--rh20t-work", help="default: <out>/_work/rh20t")
    build = commands.choices["build"]
    build.add_argument("--families", required=True)
    build.add_argument("--workers", type=int, default=24)
    build.add_argument("--shard")
    build.add_argument("--only", help="comma-separated entry names, for a rerun of a few")
    build.add_argument("--agibot-raw")
    build.add_argument("--droid-raw")
    build.add_argument("--rh20t-source-root")
    build.add_argument("--layouts", help="a layout naming --dirs, in place of --manifest "
                                         "(a post-train corpus, which has no manifest yet)")
    build.add_argument("--dirs", help="comma-separated root-relative directories of --layouts")
    commands.choices["rh20t-extract"].add_argument("--archive", action="append", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s: %(message)s")
    check_tree()
    if args.command == "build":
        return run_build(args)
    return run_extract(args)


if __name__ == "__main__":
    sys.exit(main())
