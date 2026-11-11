"""LIBERO, LIBERO-plus and LIBERO-PRO against a running neo-vla-serve.

The three share a robot, two cameras and an action layout, so they share this
client. They differ in where the tasks come from and in what the instruction
is: LIBERO-plus writes its perturbation into each task's language, and the tag
is stripped before the instruction is sent, as the benchmark's evaluation
protocol does.

    python -m neo_vla.bench.libero --benchmark libero --checkpoint STEP_DIR \\
        --server http://127.0.0.1:8000 --libero-root /path/to/LIBERO --out results/libero

The protocol (suites, step budgets, episodes, horizon, seeds) is the step's
eval.json; --horizon, --episodes, --suites and --tasks override it for a
shorter run, and the result files record what was used.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy

from ..serve import Client
from .client import BenchPolicy, Step, arm_state, extrinsics_6, noise_seed, quaternion_matrix
from .spec import EvalSpec

logger = logging.getLogger(__name__)

# Observation key and MuJoCo camera of each camera slot.
CAMERAS = (
    (0, "agentview_image", "agentview"),
    (1, "robot0_eye_in_hand_image", "robot0_eye_in_hand"),
)
# MuJoCo cameras look down -z with +y up; the model reads OpenCV axes.
MJ_TO_CV = numpy.diag([1.0, -1.0, -1.0])
BASE_BODY = "robot0_base"
# Six zero deltas and an open gripper while the scene settles.
DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
GRIPPER_COLUMN = 6

# LIBERO-plus derives a task's language from its file name, so the variant's
# identifier arrives as bare words at the end ("... basket view 0 0 100 2 4").
TAG_NUMBERED = ("view", "initstate", "noise", "table", "tb", "light", "add")
TAG_PLAIN = ("moved",)
TAG = re.compile(
    r"^\s*[\[(][^\])]*[\])]\s*|\s*[\[(][^\])]*[\])]\s*$"
    r"|\s+(?:(?:" + "|".join(TAG_NUMBERED) + r")(?:\s+-?[0-9]+)+"
    r"|" + "|".join(TAG_PLAIN) + r"|level[0-9]+|sample[0-9]+)\s*$"
)


def strip_tag(language: str) -> str:
    text = str(language).strip()
    while True:
        shorter = TAG.sub("", text).strip()
        if shorter == text:
            break
        text = shorter
    if not text:
        raise ValueError(f"the task language {language!r} is only a tag")
    return text


def check_strip(suite, bddl_root: str) -> None:
    """The tag must come off, and leave the name of a task the suite really holds.

    A pattern that matches nothing raises nothing on its own; a run would then
    send every instruction with its tag while reporting the tag stripped.
    """
    stripped, missing = 0, []
    listings: dict[str, set] = {}
    for index in range(suite.n_tasks):
        task = suite.get_task(index)
        if strip_tag(task.language) != str(task.language).strip():
            stripped += 1
        folder = os.path.join(bddl_root, task.problem_folder)
        listings.setdefault(folder, set(os.listdir(folder)))
        base, extension = os.path.splitext(task.bddl_file)
        nominal = strip_tag(base.replace("_", " ")).replace(" ", "_") + extension
        if nominal not in listings[folder]:
            missing.append(task.bddl_file)
    if not stripped:
        raise ValueError("the tag pattern changes none of this suite's instructions")
    if missing:
        raise ValueError(f"{len(missing)} tasks strip to a name the suite does not hold, "
                         f"first {missing[0]}")


def gripper_command(canonical: float) -> float:
    """0 open .. 1 closed -> robosuite's sign: +1 closes, -1 opens."""
    return 1.0 if float(canonical) >= 0.5 else -1.0


class LiberoEnv:
    """One task's simulator, reporting steps in model terms."""

    def __init__(self, environment, spec: EvalSpec, *, turn: str):
        self.environment = environment
        self.arm = spec.arms[0]
        self.turn = turn
        sim = environment.sim
        self.cameras = {slot: sim.model.camera_name2id(name) for slot, _, name in CAMERAS}
        self.base = sim.model.body_name2id(BASE_BODY)
        base = numpy.asarray(sim.data.body_xmat[self.base]).reshape(3, 3)
        # Poses are read in the world frame and sent as base frame.
        if not numpy.allclose(base, numpy.eye(3), atol=1e-5):
            raise ValueError(f"{BASE_BODY} is rotated in this scene")

    def frame(self, image) -> numpy.ndarray:
        image = numpy.asarray(image)
        # The offscreen buffer is bottom-up; the model reads upright frames.
        return numpy.ascontiguousarray(image[::-1, ::-1] if self.turn == "rotate180" else image)

    def observe(self, observation: dict) -> Step:
        sim = self.environment.sim
        base = numpy.asarray(sim.data.body_xpos[self.base], dtype=numpy.float64)
        state = arm_state(
            numpy.asarray(observation["robot0_eef_pos"], dtype=numpy.float64) - base,
            quaternion_matrix(observation["robot0_eef_quat"], "xyzw"),
            gripper_raw=float(numpy.asarray(observation["robot0_gripper_qpos"]).reshape(-1)[0]),
            raw_open=self.arm.raw_open,
            raw_closed=self.arm.raw_closed,
            joints=(numpy.asarray(observation["robot0_joint_pos"]).reshape(-1)
                    if self.arm.joints else None),
        )
        poses = {}
        for slot, identifier in self.cameras.items():
            world = numpy.asarray(sim.data.cam_xmat[identifier]).reshape(3, 3)
            poses[slot] = extrinsics_6(world @ MJ_TO_CV)
        images = {slot: self.frame(observation[key]) for slot, key, _ in CAMERAS}
        return Step(images=images, states=[state], poses=poses)

    def reset(self, init_state, settle_steps: int) -> Step:
        self.environment.reset()
        observation = self.environment.set_init_state(init_state)
        for _ in range(settle_steps):
            observation, _, _, _ = self.environment.step(DUMMY_ACTION)
        return self.observe(observation)

    def step(self, action) -> tuple[Step, bool]:
        values = [float(x) for x in numpy.asarray(action, dtype=numpy.float64).reshape(-1)]
        values[GRIPPER_COLUMN] = gripper_command(values[GRIPPER_COLUMN])
        observation, _, done, _ = self.environment.step(values)
        return self.observe(observation), bool(done)

    def close(self) -> None:
        self.environment.close()


def control_freq(environment) -> float:
    # LIBERO-plus wraps the robosuite env one level deeper.
    inner = environment if hasattr(environment, "control_freq") else environment.env
    return float(inner.control_freq)


def prepare_pro(clone: Path, data: Path, stage: Path, suites) -> Path:
    """Point a LIBERO-PRO checkout at its task files; returns the config directory.

    The fork's perturbed bddl and init-state files are distributed separately;
    they are linked into one staging tree and a config.yaml names it.
    """
    for category in ("bddl_files", "init_files"):
        (stage / category).mkdir(parents=True, exist_ok=True)
        for suite in suites:
            source = data / category / suite
            link = stage / category / suite
            if source.is_dir() and not link.exists():
                link.symlink_to(source.resolve(), target_is_directory=True)
    root = clone / "libero" / "libero"
    config = stage / "config"
    config.mkdir(exist_ok=True)
    (stage / "datasets").mkdir(exist_ok=True)
    paths = {"benchmark_root": str(root), "bddl_files": str(stage / "bddl_files"),
             "init_states": str(stage / "init_files"), "assets": str(root / "assets"),
             "datasets": str(stage / "datasets")}
    (config / "config.yaml").write_text(json.dumps(paths, indent=2) + "\n")
    return config


def pro_init_states(path: str):
    """LIBERO-PRO's init files are numpy pickles in a torch-style zip."""
    import pickle
    import zipfile
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if "archive/data.pkl" in names and not any("/data/" in name for name in names):
            return pickle.loads(archive.read("archive/data.pkl"))
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def task_path(out: Path, suite: str, index: int) -> Path:
    return out / suite / f"task_{index:04d}.json"


def run_task(env: LiberoEnv, policy: BenchPolicy, *, suite: str, index: int, instruction: str,
             init_states, episodes: int, max_steps: int, horizon: int, settle_steps: int,
             noise_base: int | None) -> list[dict]:
    results = []
    for episode in range(episodes):
        step = env.reset(init_states[episode % len(init_states)], settle_steps)
        policy.reset()
        pending: list = []
        executed, success = 0, False
        while executed < max_steps:
            if not pending:
                seed = None if noise_base is None else noise_seed(
                    noise_base, suite, index, episode, policy.calls)
                chunk = policy.chunk(step, instruction, seed=seed)
                pending = list(policy.decode(chunk[:, :horizon], step))
            step, success = env.step(pending.pop(0))
            policy.executed_rows(1)
            executed += 1
            if success:
                break
        results.append({"episode": episode, "success": bool(success), "steps": executed,
                        "calls": policy.calls})
        logger.info("%s task %d episode %d: %s in %d steps", suite, index, episode,
                    "success" if success else "failure", executed)
    return results


def open_suite(benchmark: str, name: str):
    from libero.libero import benchmark as registry
    from libero.libero import get_libero_path

    table = registry.get_benchmark_dict()
    if name not in table:
        raise ValueError(f"{name!r} is not a suite of this LIBERO tree")
    return table[name](), get_libero_path("bddl_files"), get_libero_path("init_states")


def summarise(out: Path, benchmark: str, suites: list[str], settings: dict) -> dict:
    found = {}
    for suite in suites:
        files = sorted((out / suite).glob("task_*.json"))
        episodes = [one for path in files for one in json.loads(path.read_text())["episodes"]]
        if episodes:
            found[suite] = {"tasks": len(files), "episodes": len(episodes),
                            "success_rate": 100.0 * sum(e["success"] for e in episodes)
                            / len(episodes)}
    rates = [one["success_rate"] for one in found.values()]
    summary = {"benchmark": benchmark, "settings": settings, "suites": found,
               "mean_success_rate": sum(rates) / len(rates) if rates else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="LIBERO, LIBERO-plus or LIBERO-PRO rollouts")
    parser.add_argument("--benchmark", required=True,
                        choices=("libero", "libero_plus", "libero_pro"))
    parser.add_argument("--checkpoint", required=True, help="step directory holding eval.json")
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--libero-root", required=True,
                        help="the benchmark's own checkout (LIBERO, LIBERO-plus or LIBERO-PRO)")
    parser.add_argument("--pro-data", type=Path,
                        help="LIBERO-PRO: directory holding its bddl_files and init_files")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--suites", nargs="*", help="default: every suite of the protocol")
    parser.add_argument("--tasks", type=int, nargs="*", help="task indices; default all")
    parser.add_argument("--episodes", type=int, help="episodes per task")
    parser.add_argument("--horizon", type=int, help="rows executed per chunk")
    parser.add_argument("--no-noise-seed", action="store_true",
                        help="let the server draw its own sampling noise")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    spec = EvalSpec.load(args.checkpoint)
    protocol = spec.benchmark(args.benchmark)
    suites = args.suites or list(protocol["suites"])
    horizon = int(args.horizon or protocol["horizon"])
    episodes = args.episodes or protocol.get("episodes")
    settle = int(protocol["settle_steps"])
    noise_base = None if args.no_noise_seed else protocol.get("noise_seed")
    root = Path(args.libero_root).resolve()
    os.environ["LIBERO_REPO_DIR"] = str(root)
    sys.path.insert(0, str(root))
    if args.benchmark == "libero_pro":
        if args.pro_data is None:
            parser.error("--pro-data is required for libero_pro")
        os.environ["LIBERO_CONFIG_PATH"] = str(
            prepare_pro(root, args.pro_data, args.out / "staged", suites))
    from libero.libero.envs import OffScreenRenderEnv

    settings = {"horizon": horizon, "episodes": episodes, "settle_steps": settle,
                "noise_seed": noise_base, "seed": protocol["seed"],
                "resolution": protocol["resolution"], "suites": suites,
                "strip_tag": bool(protocol.get("strip_tag", False))}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    client = Client(args.server, timeout=600)

    for suite_name in suites:
        suite, bddl_root, init_root = open_suite(args.benchmark, suite_name)
        if settings["strip_tag"]:
            check_strip(suite, bddl_root)
        units = protocol["suites"][suite_name]["units"]
        max_steps = int(protocol["suites"][suite_name]["max_steps"])
        policy = BenchPolicy(client, spec, units=units)
        indices = args.tasks if args.tasks is not None else range(suite.n_tasks)
        for index in indices:
            path = task_path(args.out, suite_name, index)
            if path.exists():
                continue
            task = suite.get_task(index)
            instruction = strip_tag(task.language) if settings["strip_tag"] else task.language
            if args.benchmark == "libero_pro":
                init_states = pro_init_states(
                    os.path.join(init_root, suite_name, task.init_states_file))
            else:
                init_states = suite.get_task_init_states(index)
            environment = OffScreenRenderEnv(
                bddl_file_name=os.path.join(bddl_root, task.problem_folder, task.bddl_file),
                camera_heights=int(protocol["resolution"]),
                camera_widths=int(protocol["resolution"]),
            )
            if abs(control_freq(environment) - spec.control_hz) > 1e-6:
                raise ValueError(f"the simulator runs at {control_freq(environment)} Hz, the "
                                 f"step expects {spec.control_hz} Hz")
            environment.seed(int(protocol["seed"]))
            env = LiberoEnv(environment, spec, turn=protocol["turn"])
            started = time.time()
            try:
                results = run_task(
                    env, policy, suite=suite_name, index=index, instruction=instruction,
                    init_states=init_states, episodes=int(episodes or len(init_states)),
                    max_steps=max_steps, horizon=horizon, settle_steps=settle,
                    noise_base=noise_base)
            finally:
                env.close()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "suite": suite_name, "task": index, "name": task.name,
                "language": task.language, "instruction": instruction,
                "horizon": horizon, "max_steps": max_steps,
                "seconds": round(time.time() - started, 1), "episodes": results,
            }, indent=2) + "\n")
    summary = summarise(args.out, args.benchmark, suites, settings)
    print(json.dumps({k: summary[k] for k in ("suites", "mean_success_rate")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
