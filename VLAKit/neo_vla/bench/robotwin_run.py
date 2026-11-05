"""Run RoboTwin 2.0 tasks one at a time through RoboTwin's own eval_policy.py.

Start the server first (python -m neo_vla.bench.spec, then neo-vla-serve), then:

    python -m neo_vla.bench.robotwin_run --checkpoint STEP_DIR --protocol clean \\
        --robotwin-root /path/to/RoboTwin --robotwin-python /path/to/robotwin/bin/python \\
        --server http://127.0.0.1:8000 --results out/robotwin_clean

clean is demo_clean and randomized is demo_randomized, both with seen
instructions, as eval.json states them. Results go under RESULTS/PROTOCOL/:
each task writes tasks/<name>/ (the settings, the simulator log and the
per-episode record) and <name>.json; a task with a result file is skipped, so
rerunning the command resumes. summary.json holds every task's success rate
and their mean.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from .spec import EvalSpec

POLICY_NAME = "neo_policy"
PLUGIN = Path(__file__).resolve().parent / "robotwin_policy"
PLUGIN_FILES = ("deploy_policy.py", "deploy_policy.yml")
STEP_LIMITS = "task_config/_eval_step_limit.yml"
NOT_TASKS = ("_GLOBAL_CONFIGS", "__init__", "_base_task")
VLAKIT = Path(__file__).resolve().parents[2]


def install_plugin(root: Path) -> Path:
    """Copy the plug-in into the checkout's policy/neo_policy."""
    target = root / "policy" / POLICY_NAME
    target.mkdir(parents=True, exist_ok=True)
    for name in PLUGIN_FILES:
        shutil.copyfile(PLUGIN / name, target / name)
    (target / "__init__.py").touch()
    return target


def read_tasks(root: Path) -> list[str]:
    """The checkout's task modules, held against its step-limit file."""
    limits = set(yaml.safe_load((root / STEP_LIMITS).read_text(encoding="utf-8")))
    modules = {path.stem for path in (root / "envs").glob("*.py")} - set(NOT_TASKS)
    if limits != modules:
        raise ValueError(f"task modules and step limits disagree on "
                         f"{sorted(limits ^ modules)}")
    return sorted(modules)


def run_task(task: str, args, protocol: dict, results: Path) -> dict:
    work = results / "tasks" / task
    work.mkdir(parents=True, exist_ok=True)
    record = work / "episodes.json"
    settings = work / "settings.json"
    settings.write_text(json.dumps({
        "server": args.server,
        "eval_json": str(Path(args.checkpoint).resolve() / "eval.json"),
        "task": task,
        "horizon": args.horizon,
        "noise_seed": args.noise_seed,
        "quaternion_order": args.quaternion_order,
        "record": str(record.resolve()),
    }, indent=1) + "\n")
    command = [
        args.robotwin_python, "script/eval_policy.py",
        "--config", f"policy/{POLICY_NAME}/deploy_policy.yml",
        "--overrides",
        "--task_name", task,
        "--task_config", protocol["task_config"],
        "--ckpt_setting", args.label,
        "--seed", str(args.seed),
        "--instruction_type", protocol["instruction_type"],
        "--policy_name", POLICY_NAME,
        "--neo_settings", str(settings.resolve()),
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(VLAKIT)] + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else []))
    if args.renderer == "cpu":
        icd = Path(args.mesa_root) / "share/vulkan/icd.d/lvp_icd.x86_64.json"
        environment["VK_ICD_FILENAMES"] = str(icd)
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(Path(args.mesa_root) / "lib/deps"), environment.get("LD_LIBRARY_PATH", "")])
        environment["ROBOTWIN_RASTERIZE"] = "1"
    with (work / "log.txt").open("w", encoding="utf-8") as log:
        finished = subprocess.run(command, cwd=args.robotwin_root, stdout=log,
                                  stderr=subprocess.STDOUT, env=environment, check=False)
    if finished.returncode != 0:
        raise RuntimeError(f"task {task} exited {finished.returncode}; see {work / 'log.txt'}")
    episodes = [one for one in json.loads(record.read_text()) if one["calls"] > 0]
    if len(episodes) < args.episodes:
        raise RuntimeError(f"task {task} recorded {len(episodes)} of {args.episodes} episodes")
    successes = sum(bool(one["success"]) for one in episodes)
    return {"task": task, "episodes": len(episodes), "successes": successes,
            "success_rate": successes / len(episodes)}


def summarise(results: Path, tasks: list[str], args) -> dict:
    found = {}
    for task in tasks:
        path = results / f"{task}.json"
        if path.is_file():
            found[task] = json.loads(path.read_text())["success_rate"]
    summary = {
        "protocol": args.protocol,
        "horizon": args.horizon,
        "seed": args.seed,
        "noise_seed": args.noise_seed,
        "renderer": args.renderer,
        "tasks_done": len(found),
        "tasks_expected": len(tasks),
        "success_rate": sum(found.values()) / len(found) if found else None,
        "tasks": found,
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="RoboTwin 2.0 through RoboTwin's eval_policy.py")
    parser.add_argument("--checkpoint", required=True, help="step directory holding eval.json")
    parser.add_argument("--protocol", required=True, help="clean or randomized")
    parser.add_argument("--robotwin-root", required=True, type=Path)
    parser.add_argument("--robotwin-python", required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--task", action="append", help="repeatable; default every task")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--noise-seed", type=int)
    parser.add_argument("--renderer", choices=("gpu", "cpu"), default="gpu",
                        help="gpu is ray tracing, the protocol's renderer")
    parser.add_argument("--mesa-root", help="software Vulkan driver, for --renderer cpu")
    parser.add_argument("--label", default="neo_vla")
    args = parser.parse_args(argv)

    spec = EvalSpec.load(args.checkpoint)
    bench = spec.benchmark("robotwin")
    if args.protocol not in bench["protocols"]:
        parser.error(f"--protocol must be one of {sorted(bench['protocols'])}")
    if args.renderer == "cpu" and not args.mesa_root:
        parser.error("--renderer cpu needs --mesa-root")
    protocol = bench["protocols"][args.protocol]
    args.horizon = bench["horizon"] if args.horizon is None else args.horizon
    args.seed = bench["seed"] if args.seed is None else args.seed
    args.noise_seed = bench.get("noise_seed") if args.noise_seed is None else args.noise_seed
    args.episodes = int(bench["episodes"])
    args.quaternion_order = bench.get("quaternion_order", "wxyz")

    every = read_tasks(args.robotwin_root)
    if args.task is None and len(every) != int(bench["tasks"]):
        parser.error(f"the checkout holds {len(every)} tasks, the benchmark {bench['tasks']}")
    tasks = every if args.task is None else args.task
    unknown = sorted(set(tasks) - set(every))
    if unknown:
        parser.error(f"not tasks of this checkout: {unknown}")

    install_plugin(args.robotwin_root)
    results = args.results / args.protocol
    results.mkdir(parents=True, exist_ok=True)
    (results / "run.json").write_text(json.dumps({
        "protocol": args.protocol, **protocol, "horizon": args.horizon, "seed": args.seed,
        "noise_seed": args.noise_seed, "renderer": args.renderer, "server": args.server,
        "checkpoint": str(args.checkpoint),
    }, indent=1) + "\n")
    for task in tasks:
        path = results / f"{task}.json"
        if path.is_file():
            continue
        found = run_task(task, args, protocol, results)
        path.write_text(json.dumps(found, indent=1) + "\n")
        print(f"{task}: {found['successes']}/{found['episodes']}", flush=True)
    summary = summarise(results, tasks, args)
    print(json.dumps({k: summary[k] for k in ("protocol", "tasks_done", "success_rate")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
