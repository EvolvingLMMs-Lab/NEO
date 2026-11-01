"""Run the preparation stages in order, skipping the ones nothing changed for.

Each stage is a separate program and is run as one, so the pipeline has no
private path into any stage that the command line does not have. What makes it
resumable is the plan file: it records the exact arguments each stage ran with,
and a stage re-runs when its arguments or tool version differ, or when its
output, an input or an artifact it recorded is missing. Contents are not
compared: after editing an input in place, pass --force, or --only with the
stages that read it. A run that fails keeps the plan for the stages that did
run, and only those.

Verification always runs, because it has no output to compare and its whole job
is to disagree with what the earlier stages believe.

Usage:
    python tools/prepare_data.py --root /corpus --dirs-file dirs.txt \
        --layouts config/layouts/action_pretrain.json --corpus action_pretrain \
        --calib agibot=calib/agibot_intrinsics --calib-dirs agibot=agibot.txt \
        --share agibot=0.49 --share oxe=0.49 --share rh20t=0.02

Every corpus keeps its manifest at manifests/<corpus>.json, its stage records,
normaliser and sidecars under metadata/<corpus>/ and its caches under
processed/<corpus>/, unless --metadata-dir or --processed-dir says otherwise.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import stage

from neo_vla.constants import METADATA_DIR, PROCESSED_DIR

logger = logging.getLogger("prepare_data")

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent

# The stages a step runs, when its name is not the module's. A calibration
# source gets a step of its own for each, because either stage reads one
# source per run and each writes its own file.
INTRINSICS_STAGE = "build_intrinsics"
EXTRINSICS_STAGE = "build_extrinsics"

# The flags the per-directory and pooled normalisers took, refused by name: the
# normaliser is per embodiment key now, measured by build_stats on the manifest
# after the cache (V6 5.1-18), with no scope and no quantile to choose.
RETIRED_FLAGS = {
    "--stats-scope": "the normaliser is per embodiment key; there is no scope to choose",
    "--stats-quantile": "the transform is q01/q99 (build_stats --quantiles 1 99)",
    "--stats-max-episodes": "build_stats draws --stats-windows-per-stratum windows per stratum",
}

# Where the stages' repository configs live, relative to the repository.
DEFAULT_TEMPLATE = "config/prompt_template/policy.yaml"
DEFAULT_PUBLISHED = "config/rates/oxe_published.json"
DEFAULT_EMBODIMENTS = "config/embodiments.yaml"


@dataclass(frozen=True)
class Step:
    """One stage invocation: what to run, with what, and the file it writes.

    name is what --only selects and what the plan records, and it is the module
    for every stage but the per-source calibration ones, which run one module
    several times. output is None for a stage that writes nothing, which is
    what makes it run every time.
    """

    name: str
    module: str
    argv: list[str]
    output: str | None


def paired(pairs: list[str], flag: str, what: str) -> dict[str, list[str]]:
    """source=value, repeated, as the values each source states."""
    sources: dict[str, list[str]] = {}
    for pair in pairs:
        source, _, value = pair.partition("=")
        if not source or not value:
            raise ValueError(
                f"calibration source {pair!r} is not source={what}\n"
                f"expected a reader name, then {what} under the root\n"
                f"fix {flag}"
            )
        sources.setdefault(source, []).append(value)
    return sources


def intrinsics_steps(args: argparse.Namespace) -> list[Step]:
    """One build_intrinsics step per calibration source, in source order.

    Each names its own directories. A reader covers one source's directories
    and refuses another's, and the two OXE readers split a corpus the layout
    file spells with one source word, so the run's own directory list cannot
    stand in for theirs.
    """
    trees = paired(args.calib, "--calib", "a calibration tree")
    listed = paired(args.calib_dirs, "--calib-dirs", "a file of directories")
    missing = sorted(set(trees) - set(listed))
    if missing:
        raise ValueError(
            f"calibration source {missing[0]!r} names no directories\n"
            f"expected --calib-dirs {missing[0]}=<file>, the directories it covers\n"
            f"add it, or drop the source from --calib"
        )
    # The other way round is a misspelled source, which would otherwise read as
    # a corpus nobody calibrated rather than as the typo it is.
    stray = sorted(set(listed) - set(trees))
    if stray:
        raise ValueError(
            f"calibration source {stray[0]!r} names directories and no tree\n"
            f"expected --calib {stray[0]}=<tree>, the calibration it reads\n"
            f"add it, or drop the source from --calib-dirs"
        )
    steps = []
    for source in sorted(trees):
        if len(listed[source]) != 1:
            raise ValueError(
                f"calibration source {source!r} names {len(listed[source])} "
                f"directory files\nexpected one; the stage reads a single list\n"
                f"put its directories in one file"
            )
        argv = ["--root", str(args.root), "--source", source,
                "--layouts", args.layouts, "--dirs-file", listed[source][0],
                "--out", intrinsics_file(args, source)]
        for tree in trees[source]:
            argv += ["--calib", tree]
        steps.append(Step(name=f"{INTRINSICS_STAGE}:{source}", module=INTRINSICS_STAGE,
                          argv=argv, output=intrinsics_file(args, source)))
    return steps


def extrinsics_steps(args: argparse.Namespace) -> list[Step]:
    """One build_extrinsics step per source that publishes camera poses.

    Paired the way the intrinsics steps are and for the same reason: the stage
    reads the corpus rather than the earlier stages, so it needs its own list
    of directories and refuses another source's.
    """
    trees = paired(args.extrinsics, "--extrinsics", "a pose tree")
    listed = paired(args.extrinsics_dirs, "--extrinsics-dirs", "a file of directories")
    for missing, flag, other in ((sorted(set(trees) - set(listed)), "--extrinsics-dirs",
                                 "--extrinsics"),
                                (sorted(set(listed) - set(trees)), "--extrinsics",
                                 "--extrinsics-dirs")):
        if missing:
            raise ValueError(
                f"pose source {missing[0]!r} is named by {other} alone\n"
                f"expected {flag} {missing[0]}=<value> beside it\n"
                f"add it, or drop the source from {other}"
            )
    steps = []
    for source in sorted(trees):
        if len(listed[source]) != 1:
            raise ValueError(
                f"pose source {source!r} names {len(listed[source])} directory files\n"
                f"expected one; the stage reads a single list\n"
                f"put its directories in one file"
            )
        argv = ["--root", str(args.root), "--source", source,
                "--layouts", args.layouts, "--dirs-file", listed[source][0],
                "--out", extrinsics_file(args, source)]
        for tree in trees[source]:
            argv += ["--calib", tree]
        steps.append(Step(name=f"{EXTRINSICS_STAGE}:{source}", module=EXTRINSICS_STAGE,
                          argv=argv, output=extrinsics_file(args, source)))
    return steps


def intrinsics_file(args: argparse.Namespace, source: str) -> str:
    """Where one calibration source's records go."""
    return f"{args.metadata_dir}/intrinsics_{source}.jsonl"


def extrinsics_file(args: argparse.Namespace, source: str) -> str:
    """Where one pose source's records go."""
    return f"{args.metadata_dir}/extrinsics_{source}.jsonl"


def trajectory_arguments(args: argparse.Namespace) -> list[str]:
    """What build_weights is told about the per-episode weights, if anything.

    Empty when no directory is named, which leaves every directory uniform
    over its episodes. Naming one is what carries an episode's exclusion
    through the weights and the manifest and into the loader.
    """
    if not args.trajectory_weights_dir:
        return []
    return ["--trajectory-weights-dir", args.trajectory_weights_dir,
            "--trajectory-weights-suffix", args.trajectory_weights_suffix]


def stats_arguments(args: argparse.Namespace) -> list[str]:
    """What build_stats is told beyond the manifest: the lineage flags, when given."""
    extra: list[str] = []
    for spec in args.inherit_action:
        extra += ["--inherit-action", spec]
    if args.stats_from:
        extra += ["--from", args.stats_from]
    for spec in args.reuse:
        extra += ["--reuse", spec]
    if args.check_fit:
        extra.append("--check-fit")
    return extra


def plan_of(args: argparse.Namespace) -> list[Step]:
    """Every stage's arguments and the one file it writes, in run order.

    V6 5.1-18: scan, validate, filter, measure_scale, declare_frequency,
    build_weights (after the trajectory weights), calibration and extrinsics,
    build_embodiment, measure_gripper_polarity, build_manifest, build_cache,
    build_stats (the loader's targets over the built cache), verify_manifest.
    """
    prep = args.metadata_dir
    manifest = f"{args.manifest_dir}/{args.corpus}.json"
    normaliser = f"{prep}/embodiments.json"
    common = ["--root", str(args.root)]
    scan = common + ["--out", f"{prep}/inventory.jsonl"]
    if args.dirs_file:
        scan += ["--dirs-file", args.dirs_file]
    for directory in args.dir:
        scan += ["--dir", directory]
    calibrated = intrinsics_steps(args)
    intrinsics = [item for step in calibrated
                  for item in ("--intrinsics", step.output)]
    calibrated = calibrated + extrinsics_steps(args)
    kept = ["--kept", f"{prep}/kept.jsonl"]
    manifest_extra = []
    if args.history_blocked:
        manifest_extra += ["--history-blocked", args.history_blocked]
    if args.base_offsets:
        manifest_extra += ["--base-offsets", args.base_offsets]
    if args.sidecars:
        manifest_extra += ["--sidecars", args.sidecars]
    plan = {
        "scan_source": {"argv": scan, "output": f"{prep}/inventory.jsonl"},
        "validate_episodes": {
            "argv": common + ["--inventory", f"{prep}/inventory.jsonl",
                              "--layouts", args.layouts,
                              "--out", f"{prep}/validation.jsonl",
                              "--max-rows", str(args.validate_max_rows)],
            "output": f"{prep}/validation.jsonl"},
        "filter_episodes": {
            "argv": common + ["--validation", f"{prep}/validation.jsonl",
                              "--out", f"{prep}/kept.jsonl"]
            + [item for rule in args.drop for item in ("--drop", rule)],
            "output": f"{prep}/kept.jsonl"},
        "measure_scale": {
            "argv": common + ["--inventory", f"{prep}/inventory.jsonl", *kept,
                              "--layouts", args.layouts,
                              "--max-episodes", str(args.scale_max_episodes),
                              "--out", f"{prep}/scale.json"],
            "output": f"{prep}/scale.json"},
        "declare_frequency": {
            "argv": common + [*kept, "--layouts", args.layouts,
                              "--published", args.published,
                              "--scale", f"{prep}/scale.json",
                              "--out", f"{prep}/frequency.json"],
            "output": f"{prep}/frequency.json"},
        "build_weights": {
            "argv": common + [*kept, "--out", f"{prep}/weights.json"]
            + [item for share in args.share for item in ("--share", share)]
            + trajectory_arguments(args),
            "output": f"{prep}/weights.json"},
        "build_embodiment": {
            "argv": common + [*kept, "--layouts", args.layouts,
                              "--embodiments", args.embodiments,
                              "--arrays", f"{args.processed_dir}/embodiment",
                              "--out", f"{prep}/embodiment.json"],
            "output": f"{prep}/embodiment.json"},
        "measure_gripper_polarity": {
            "argv": common + ["--inventory", f"{prep}/inventory.jsonl", *kept,
                              "--layouts", args.layouts,
                              "--max-episodes", str(args.gripper_max_episodes),
                              "--out", f"{prep}/gripper.json"],
            "output": f"{prep}/gripper.json"},
        "build_manifest": {
            "argv": common + ["--inventory", f"{prep}/inventory.jsonl", *kept,
                              "--scale", f"{prep}/scale.json",
                              "--weights", f"{prep}/weights.json",
                              "--frequency", f"{prep}/frequency.json",
                              "--embodiment", f"{prep}/embodiment.json",
                              "--gripper", f"{prep}/gripper.json",
                              "--embodiments", args.embodiments,
                              "--layouts", args.layouts,
                              "--corpus", args.corpus,
                              "--cache-dir", args.processed_dir,
                              "--prompt-template", args.prompt_template,
                              "--prompt-cache", args.prompt_cache,
                              "--tokenizer-id", args.tokenizer_id,
                              "--max-text-tokens", str(args.max_text_tokens),
                              "--intrinsics-mode", args.intrinsics_mode,
                              "--out", manifest]
            + manifest_extra + intrinsics,
            "output": manifest},
        "build_cache": {
            "argv": common + ["--manifest", manifest, *kept,
                              "--inventory", f"{prep}/inventory.jsonl",
                              "--parts", args.parts,
                              "--out", f"{args.processed_dir}/index.json"]
            + (["--tokenizer", args.tokenizer] if args.tokenizer else []),
            "output": f"{args.processed_dir}/index.json"},
        "build_stats": {
            "argv": common + ["--manifest", manifest, "--out", normaliser,
                              "--cache-index", f"{args.processed_dir}/index.json",
                              "--embodiments", args.embodiments,
                              "--quantiles", "1", "99",
                              "--band-alpha", str(args.band_alpha),
                              "--windows-per-stratum", str(args.stats_windows_per_stratum),
                              "--frame-policies", args.stats_frame_policies,
                              "--constant-scan", "full",
                              "--workers", str(args.stats_workers),
                              "--n-slots", str(args.n_slots)]
            + stats_arguments(args),
            "output": normaliser},
        "verify_manifest": {
            "argv": common + ["--manifest", manifest,
                              "--n-slots", str(args.n_slots),
                              "--weights", f"{prep}/weights.json",
                              "--normaliser", normaliser],
            "output": None},
    }
    if intrinsics:
        # Reported before the manifest folds the numbers in, so a run's log says
        # what was wrong with each calibration whether or not one is refused.
        calibrated.append(Step(
            name="validate_intrinsics", module="validate_intrinsics",
            argv=common + ["--layouts", args.layouts,
                           "--out", f"{prep}/intrinsics_validation.jsonl"] + intrinsics,
            output=f"{prep}/intrinsics_validation.jsonl"))
    steps = [Step(name=name, module=name, argv=entry["argv"], output=entry["output"])
             for name, entry in plan.items()]
    if args.trajectory_weights_dir:
        # Before the weights, which refuse a named directory unless it holds
        # a file for every kept directory, and after the filter, whose kept
        # positions are what the arrays are indexed by.
        at = [step.name for step in steps].index("build_weights")
        steps.insert(at, Step(
            name="build_trajectory_weights", module="build_trajectory_weights",
            argv=common + ["--validation", f"{prep}/validation.jsonl", *kept,
                           "--out-dir", args.trajectory_weights_dir,
                           "--suffix", args.trajectory_weights_suffix,
                           "--out", f"{prep}/trajectory.json"],
            output=f"{prep}/trajectory.json"))
    # After the weights, because the calibration reads the corpus and not the
    # earlier stages, and before the embodiment and the manifest, which fold it in.
    at = [step.name for step in steps].index("build_embodiment")
    return steps[:at] + calibrated + steps[at:]


def child_environment() -> dict[str, str]:
    """This process's environment with the repository importable by the stages.

    A stage is run as a script, so its own directory is what python puts on the
    path and the library beside it is not. Naming the tree here rather than in
    each stage keeps the path a machine fact set once by the launcher.
    """
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    parts = [str(REPO)] + [part for part in existing.split(os.pathsep) if part]
    environment["PYTHONPATH"] = os.pathsep.join(parts)
    return environment


def stands(head: dict[str, Any], root: Path) -> bool:
    """Whether a recorded header still describes what its stage would write now.

    A header another tool version wrote does not: which inputs a stage
    records is part of what the version names, and an output
    written before a stage started recording one cannot be compared with what
    it would record today.
    """
    if head.get("tool_version") != stage.TOOL_VERSION:
        return False
    return not stage.check_inputs_match(head, root)


def is_current(root: Path, entry: dict[str, Any], recorded: dict[str, Any] | None) -> bool:
    """Whether a stage's output stands for the arguments, inputs and files it had.

    The last of those is about the files rather than about the record of them: a
    stage that names artifacts, as the cache index does, is out of date when one
    of them is gone. Without it the driver the docs name as the remedy for a
    damaged cache does nothing.
    """
    if entry["output"] is None or recorded is None:
        return False
    if recorded.get("argv") != entry["argv"]:
        return False
    output = root / entry["output"]
    if not output.exists():
        return False
    if output.suffix != ".json":
        return stands(stage.read_jsonl(output)[0], root)
    payload = stage.read_json(output)
    if not stands(payload["provenance"], root):
        return False
    return not stage.check_artifacts_match(payload, root)


class Retired(argparse.Action):
    """A flag this driver no longer takes, refused by name with what replaced it."""

    def __call__(self, parser, namespace, values, option_string=None):
        parser.error(f"{option_string} is retired: {self.help}")


def parser() -> argparse.ArgumentParser:
    """Every option, built apart from main so a plan can be read without running it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--layouts", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--dir", action="append", default=[])
    parser.add_argument("--dirs-file")
    parser.add_argument("--metadata-dir",
                        help=f"stage records and normaliser; default {METADATA_DIR}/<corpus>")
    parser.add_argument("--manifest-dir", default="manifests")
    parser.add_argument("--processed-dir",
                        help=f"numeric, frame, camera and prompt caches; default "
                             f"{PROCESSED_DIR}/<corpus>")
    parser.add_argument("--drop", action="append", default=[])
    parser.add_argument("--share", action="append", default=[])
    parser.add_argument("--trajectory-weights-dir",
                        help="where the per-episode weight arrays are built and "
                             "read from; without it every directory is uniform "
                             "over its episodes")
    parser.add_argument("--trajectory-weights-suffix", default=".npy")
    parser.add_argument("--calib", action="append", default=[],
                        help="source=tree, repeated; a source with no tree is "
                             "not calibrated and its views record no intrinsics")
    parser.add_argument("--calib-dirs", action="append", default=[],
                        help="source=file, one per --calib source: the "
                             "directories that source's reader covers")
    parser.add_argument("--extrinsics", action="append", default=[],
                        help="source=tree, repeated; a source with no tree "
                             "publishes no camera pose and trains in robot frame")
    parser.add_argument("--extrinsics-dirs", action="append", default=[],
                        help="source=file, one per --extrinsics source: the "
                             "directories that source's poses cover")
    parser.add_argument("--parts",
                        default="episode_starts,arrow,frames,extrinsics,intrinsics,prompt",
                        help="the manifest names them all, so omitting one fails verify")
    parser.add_argument("--tokenizer")
    parser.add_argument("--prompt-template", default=DEFAULT_TEMPLATE,
                        help="the prompt template; it stamps the cache's prefix_layout, so it "
                             "must be the one for --intrinsics-mode (tokens)")
    parser.add_argument("--prompt-cache", help="default <processed dir>/prompt")
    parser.add_argument("--tokenizer-id", default=os.environ.get("NEO_VLA_TOKENIZER_ID", "neo-2b"))
    parser.add_argument("--max-text-tokens", type=int, default=256)
    parser.add_argument("--n-slots", type=int, default=2,
                        help="model.action.n_slots, the slot count verify_manifest "
                             "draws at; every slot is an arm and arm k fills slot k")
    parser.add_argument("--intrinsics-mode", default="tokens",
                        help="the model.prefix.intrinsics_mode this corpus is built for; "
                             "build_manifest stamps it and the loader checks it")
    parser.add_argument("--validate-max-rows", type=int, default=0)
    parser.add_argument("--published", default=DEFAULT_PUBLISHED,
                        help="the OXE published-rate table declare_frequency reads")
    parser.add_argument("--embodiments", default=DEFAULT_EMBODIMENTS,
                        help="the key table (neo_vla_embodiment_keys_v2): clause, stage, "
                             "ee_frame_fix and the joints, hand and body per key")
    parser.add_argument("--sidecars",
                        help="the stage-1 sidecars' folder under the root (metadata/<corpus>/"
                             "sidecars), for a layout whose v7 columns are a sidecar's")
    parser.add_argument("--history-blocked",
                        help="the RH20T blocked-history list, for the pretrain corpus only")
    parser.add_argument("--base-offsets",
                        help="measure_libero_base's output, for a LIBERO corpus")
    parser.add_argument("--gripper-max-episodes", type=int, default=200)
    parser.add_argument("--band-alpha", type=float, default=1.0)
    parser.add_argument("--stats-windows-per-stratum", type=int, default=15000)
    parser.add_argument("--stats-frame-policies", default="as-drawn",
                        help="as-drawn for pretrain keys; the planned frame_ref values, "
                             "comma separated, for a finetune key measured fresh (D9)")
    parser.add_argument("--stats-workers", type=int, default=1)
    parser.add_argument("--inherit-action", action="append", default=[],
                        help="KEY=SRC, passed to build_stats with --stats-from")
    parser.add_argument("--stats-from", help="the lineage's embodiments.json")
    parser.add_argument("--reuse", action="append", default=[],
                        help="FILE#KEY, passed to build_stats")
    parser.add_argument("--check-fit", action="store_true",
                        help="build_stats holds every key to R14 (post-train and debug)")
    for flag, reason in RETIRED_FLAGS.items():
        parser.add_argument(flag, action=Retired, help=reason)
    parser.add_argument("--scale-max-episodes", type=int, default=400)
    parser.add_argument("--only", action="append", default=[], help="run only these stages")
    parser.add_argument("--force", action="store_true", help="run every stage again")
    return parser


def resolve_dirs(args: argparse.Namespace) -> argparse.Namespace:
    """Fill the corpus's default metadata, processed and prompt-cache directories."""
    if args.metadata_dir is None:
        args.metadata_dir = f"{METADATA_DIR}/{args.corpus}"
    if args.processed_dir is None:
        args.processed_dir = f"{PROCESSED_DIR}/{args.corpus}"
    if args.prompt_cache is None:
        args.prompt_cache = f"{args.processed_dir}/prompt"
    return args


def main(argv: list[str] | None = None) -> int:
    args = resolve_dirs(parser().parse_args(argv))

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    args.root = stage.resolve_root(args.root)
    plan = plan_of(args)
    plan_path = stage.under_root(args.root, f"{args.metadata_dir}/plan.json")
    previous = (stage.read_json(plan_path)["stages"]
                if plan_path.exists() else {})

    names = [step.name for step in plan]
    wanted = args.only or names
    unknown = sorted(set(wanted) - set(names))
    if unknown:
        raise ValueError(
            f"unknown stages {unknown}\n"
            f"expected names from {names}\n"
            f"fix --only, or name the calibration source in --calib"
        )

    status = 0
    ran: dict[str, dict[str, Any]] = dict(previous)
    for step in plan:
        entry = {"argv": step.argv, "output": step.output}
        if step.name not in wanted:
            logger.info("%s not selected", step.name)
            continue
        if not args.force and is_current(args.root, entry, previous.get(step.name)):
            logger.info("%s is current", step.name)
            continue
        logger.info("%s running", step.name)
        result = subprocess.run(
            [sys.executable, str(TOOLS / f"{step.module}.py"), *step.argv],
            check=False, env=child_environment(),
        )
        if result.returncode != 0:
            # Out of the plan rather than in it, so the next run repeats this
            # stage rather than trusting a partial output. The stages that did
            # run stay in, which is what makes a late failure cost one stage.
            ran.pop(step.name, None)
            logger.error("%s exited %d", step.name, result.returncode)
            status = result.returncode
            break
        ran[step.name] = entry

    stage.write_json(plan_path, {
        "provenance": stage.header("prepare_data", root=args.root,
                                   inputs={"layouts": stage.under_root(args.root, args.layouts)},
                                   tree=TOOLS.parent, extra={"corpus": args.corpus}),
        "stages": ran,
    })
    logger.info("plan at %s", plan_path)
    return status


if __name__ == "__main__":
    sys.exit(main())
