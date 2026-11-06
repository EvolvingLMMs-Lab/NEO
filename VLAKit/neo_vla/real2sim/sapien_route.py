"""Configurable route: your URDF robot, measured scene and cameras, a reviewed task, SAPIEN rollouts.

Build stages exactly the reviewed specifications, the runner, checks.py and an optional reviewed
controller plugin (all hash-pinned), samples layouts and probes the simulator environment. Rollouts
run as separate processes in that environment with deadlines; verification repeats every check on
the persisted trace and episode files and requires the runner's own audit to be reproduced exactly.
"""
from __future__ import annotations

import json
import math
import os
import re
import socket
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from . import agent, checks
from . import spec as specmod
from .pipeline import Route, attempts_used, bounded_pool, outcome, write_audit
from .store import (PathError, contained, file_record, read_bounded, read_json, run_bounded, runs,
                    sha256_file, tail, utc_now, write_new, write_new_json)

PACKAGE = Path(__file__).resolve().parent
MAX_PLUGIN_BYTES = 1 << 20   # controller plugin source staged by a build
ATTEMPT_NAME = re.compile(r"(ep|q)(\d+)_a(\d+)")
SPEC_SCHEMA = "neo-vla-real2sim-rollout/1"
ATTEMPT_SCHEMA = "neo-vla-real2sim-attempt/1"
PROBE_TIMEOUT_S = 900
PROBE_OUTPUT_LIMIT_BYTES = 4 << 20   # a probe prints little; more output fails it
PROBE_REPORT_BYTES = 1 << 20
PROBE_SCHEMA = "neo-vla-real2sim-probe/1"
READBACK_RTOL, READBACK_ATOL = 1e-5, 1e-3   # live camera intrinsics vs calibration (reference readback rule)


def reject_duplicates(entries: list) -> list:
    """Fail every passing attempt whose trace or layout a passing attempt of another episode shares.

    Each key is grouped over the attempts that passed their own checks, so a rejection under one key
    cannot hide a duplicate under another. The content key compares the trace arrays themselves: the
    file bytes also carry zip member timestamps, which differ between identical trajectories written
    at different times. Returns the problems; the entries are updated in place.
    """
    problems = []
    candidates = [e for e in entries if e["status"] == "PASS"]
    for key in ("trace_sha256", "trace_content_sha256", "layout_sha256"):
        seen: dict = {}
        for e in candidates:
            if e.get(key):
                seen.setdefault(e[key], []).append(e)
        for group in seen.values():
            if len({e["index"] for e in group}) > 1:
                for e in group:
                    e["status"] = "FAIL"
                    e["failed"] = sorted(set(e["failed"]) | {f"unique_{key}"})
                problems.append(f"{key} shared by episodes {sorted(e['index'] for e in group)}")
    return problems


def execution_problems(res: dict, what: str) -> list:
    """Why a bounded child process did not finish cleanly; empty when it did.

    Clean means it started, exited by itself with status 0 within its deadline and output cap,
    both output streams were captured to the end, and no process of its group was left behind.
    The last point can only be observed on Linux (/proc and waitid WNOWAIT); where it cannot be
    observed it is not established, which counts as not clean. Nothing the process wrote can
    override these facts about how it ran.
    """
    problems = []
    if res.get("launch_error"):
        problems.append(f"{what} could not be started: {res['launch_error']}")
    if res.get("timed_out"):
        problems.append(f"{what} reached its deadline")
    if res.get("output_limit_reached"):
        problems.append(f"{what} exceeded the {res.get('output_limit_bytes')}-byte output limit on "
                        f"{res['output_limit_reached']}")
    if res.get("returncode") != 0:
        problems.append(f"{what} exited with status {res.get('returncode')!r}")
    if res.get("capture_complete") is not True:
        problems.append(f"{what}: its output was not captured to the end")
    cleanup = res.get("cleanup") or {}
    survivors = cleanup.get("survivors")
    stopped = res.get("timed_out") or res.get("output_limit_reached")   # signals were ours to send
    if cleanup.get("observed") is not True or survivors is None:
        problems.append(f"{what}: leftover processes could not be checked "
                        "(needs Linux /proc and waitid WNOWAIT)")
    elif survivors:
        problems.append(f"{what} left {len(survivors)} process(es) running even after cleanup")
    elif cleanup.get("signals") and not stopped:
        problems.append(f"{what} exited but left processes behind; cleanup sent "
                        f"{cleanup['signals']}")
    if cleanup.get("leader_not_reaped"):
        problems.append(f"{what} could not be reaped")
    if cleanup.get("error"):
        problems.append(f"{what}: cleanup failed: {cleanup['error']}")
    return problems


def probe_acceptance(res: dict, report_path: Path) -> tuple:
    """(report or None, problems). A probe passes only when its process ran cleanly (see
    execution_problems) and its own PROBE.json, read within a size bound and without following
    symlinks, carries the probe schema and ok true. A written success report never overrides a
    failed or unclean run."""
    problems = execution_problems(res, "the probe process")
    report = None
    if not report_path.is_file():
        problems.append("the probe wrote no PROBE.json")
    else:
        try:
            report = json.loads(read_bounded(report_path, PROBE_REPORT_BYTES))
        except (PathError, ValueError) as error:
            problems.append(f"PROBE.json is unreadable: {error}")
    if report is None:
        return None, problems
    if not isinstance(report, dict) or report.get("schema") != PROBE_SCHEMA:
        problems.append(f"PROBE.json is not a {PROBE_SCHEMA} report")
        return (report if isinstance(report, dict) else None), problems
    if report.get("ok") is not True:
        problems.append(f"the probe reported a failure: {report.get('error') or 'ok is not true'}")
    return report, problems


def _json_object(path: Path) -> tuple:
    """(object, problem) for a JSON file a child process may have written. A missing, unreadable
    or non-object file gives ({}, problem), so it fails a check instead of crashing a stage."""
    if not path.is_file():
        return {}, f"{path.name} missing"
    try:
        value = read_json(path)
    except (OSError, ValueError) as error:
        return {}, f"{path.name} unreadable: {error}"
    return (value, None) if isinstance(value, dict) else ({}, f"{path.name} is not a JSON object")


def _present(run: Path, rels: list) -> list:
    """The run-relative files among rels that are regular files reached without symlinks."""
    found = []
    for rel in rels:
        try:
            if contained(run, rel).is_file():
                found.append(rel)
        except PathError:
            continue
    return found


def _load(path: Path | None) -> tuple:
    if path is None or not path.is_file():
        return None, None
    data = path.read_bytes()
    return json.loads(data), data


class SapienRoute(Route):
    name = "sapien"

    # ------------------------------------------------------------------------- intake / review
    def intake(self, issues: specmod.Issues) -> dict:
        p = self.p.project
        self.urdf = None
        rel = ((p.data.get("inputs") or {}).get("robot") or {}).get("urdf")
        if isinstance(rel, str):
            try:
                path = contained(p.root, rel)
                if path.is_file():
                    self.urdf = specmod.inspect_urdf(path, p.root)
                    p.files.update(self.urdf["meshes"])
            except (ValueError, ET.ParseError, PathError) as error:
                issues.error("inputs.robot.urdf", str(error))
        self.scene = self.task = None
        try:
            self.scene, _ = _load(p.spec_path("scene"))
            self.task, _ = _load(p.spec_path("task"))
        except ValueError as error:
            issues.error("specs", f"specification is not valid JSON: {error}")
        if self.scene is not None:
            specmod.validate_scene(self.scene, p, self.urdf, issues)
        if self.task is not None:
            specmod.validate_task(self.task, self.scene, p, issues)
            if (self.task.get("controller") or {}).get("type") == "plugin":
                plugin = p.spec_path("plugin")
                if plugin is None or not plugin.is_file():
                    issues.request("specs.plugin", "the task uses a controller plugin that does not exist yet",
                                   ("review", "build"), "run `plan` to author one, or write it yourself")
                else:
                    source = plugin.read_text(errors="replace")
                    issues.item("plugin.source", "code_to_review",
                                {"path": p.data["specs"]["plugin"], "sha256": sha256_file(plugin)})
                    findings = specmod.screen_plugin(source)
                    if findings:
                        issues.item("plugin.static_findings", "code_to_review", findings[:50])
        return {"urdf": None if self.urdf is None else {"links": len(self.urdf["links"]),
                                                        "joints": len(self.urdf["joints"])},
                "scene_present": self.scene is not None, "task_present": self.task is not None}

    def artifacts(self) -> dict:
        p = self.p.project
        out = {}
        for key in ("scene", "task", "plugin"):
            path = p.spec_path(key)
            out[key] = sha256_file(path) if path is not None and path.is_file() else None
        return out

    def review(self) -> tuple:
        artifacts = self.artifacts()
        notes = specmod.Issues()
        for key, name in (("scene", "scene.json"), ("task", "task.json")):
            found = self._candidate_notes(name, artifacts[key])
            for q in found.get("open_questions", []):
                notes.item(f"{key}.{q.get('field')}", "agent_open_question", q.get("question"))
            for o in found.get("observations", []) + found.get("rationale", []):
                notes.item(key, "agent_note", o)
        items = {i["id"]: i for i in self.p.project.issues.review}
        for i in notes.review:
            items.setdefault(i["id"], i)
        return artifacts, list(items.values())

    def _candidate_notes(self, name: str, sha: str | None) -> dict:
        """Notes of the plan candidate the current specification was promoted from, if any."""
        if sha is None:
            return {}
        for run in reversed(runs(self.p.stage_dir("plan"))):
            cand = run / "candidates" / name
            if cand.is_file() and sha256_file(cand) == sha:
                notes = run / "candidates" / name.replace(".json", ".notes.json")
                return read_json(notes) if notes.is_file() else {}
        return {}

    # ------------------------------------------------------------------------- plan
    def plan(self, run: Path, author: set) -> dict:
        p = self.p.project
        config = self.p.agent_config()
        scene_path, task_path, plugin_path = p.spec_path("scene"), p.spec_path("task"), p.spec_path("plugin")
        need = {k for k, path in (("scene", scene_path), ("task", task_path))
                if k in author or path is None or not path.is_file()}
        candidates = run / "candidates"
        candidates.mkdir()
        proposal: dict = {"schema": "neo-vla-real2sim-proposal/1", "authored": {}, "jobs": []}
        if need and not config.get("enabled"):
            return outcome("HELD", [f"write {sorted(need)} yourself or enable the agent in project.json"],
                           [], ["nothing was authored"])
        if need or "plugin" in author:
            status = agent.auth_status(config["executable"], run)
            proposal["cli"] = status
            if status["version"]["returncode"] != 0 or status["auth_status"]["returncode"] != 0:
                write_new_json(run / "PROPOSAL.json", proposal)
                return outcome("FAILED", ["the Claude Code CLI is not available or not logged in: run "
                                          "`claude auth login`, then `claude auth status`"], ["PROPOSAL.json"])
        scene, scene_bytes = (None, None) if "scene" in need else _load(scene_path)
        if "scene" in need:
            record = agent.run_job(config, agent.scene_job(p, self.urdf), run / "agent", candidates)
            proposal["jobs"].append(_job_summary(record))
            if record["status"] != "PASSED":
                return self._plan_failed(run, proposal, record)
            scene_bytes = (candidates / "scene.json").read_bytes()
            scene = json.loads(scene_bytes)
            proposal["authored"]["scene"] = "agent"
        else:
            write_new(candidates / "scene.json", scene_bytes)
            proposal["authored"]["scene"] = "user"
        if "task" in need:
            record = agent.run_job(config, agent.task_job(p, scene, scene_bytes), run / "agent", candidates)
            proposal["jobs"].append(_job_summary(record))
            if record["status"] != "PASSED":
                return self._plan_failed(run, proposal, record)
            task_bytes = (candidates / "task.json").read_bytes()
            proposal["authored"]["task"] = "agent"
        else:
            _, task_bytes = _load(task_path)
            write_new(candidates / "task.json", task_bytes)
            proposal["authored"]["task"] = "user"
        task = json.loads(task_bytes)
        outputs = ["PROPOSAL.json", "candidates/scene.json", "candidates/task.json"]
        if (task.get("controller") or {}).get("type") == "plugin":
            if plugin_path is None:
                return outcome("FAILED", ["the task uses a plugin but specs.plugin is not declared"], [])
            if "plugin" in author or not plugin_path.is_file():
                record = agent.run_job(config, agent.controller_job(p, scene_bytes, task_bytes, plugin_path.name),
                                       run / "agent", candidates)
                proposal["jobs"].append(_job_summary(record))
                if record["status"] != "PASSED":
                    return self._plan_failed(run, proposal, record)
                proposal["authored"]["plugin"] = "agent"
                outputs.append(f"candidates/{plugin_path.name}")
        issues = specmod.Issues()
        specmod.validate_scene(scene, p, self.urdf, issues)
        specmod.validate_task(task, scene, p, issues)
        proposal["validation"] = issues.as_dict()
        write_new_json(run / "PROPOSAL.json", proposal)
        status = "FAILED" if issues.errors else "HELD" if issues.requests else "PASSED"
        reasons = [f"{e['field']}: {e['problem']}" for e in issues.errors] + \
                  [f"{r['field']}: {r['problem']}" for r in issues.requests]
        if status != "FAILED":
            reasons.append("candidates are not used until promoted (`review --promote`) and reviewed")
        return outcome(status, reasons, outputs, ["candidates are proposals; nothing is simulated or verified"])

    @staticmethod
    def _plan_failed(run: Path, proposal: dict, record: dict) -> dict:
        write_new_json(run / "PROPOSAL.json", proposal)
        return outcome("FAILED", [f"agent job {record['job']}: {r}" for r in record["reasons"][:10]],
                       ["PROPOSAL.json"], ["no candidate was accepted"])

    # ------------------------------------------------------------------------- build
    def _sim(self) -> dict:
        return dict(self.p.project.data.get("simulator") or {})

    def child_env(self) -> dict:
        sim = self._sim()
        env = dict(os.environ)
        env.update({"PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1"})
        env.update(sim.get("env") or {})
        if sim.get("renderer") == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        return env

    def build_inputs(self) -> dict:
        p, sim = self.p.project, self._sim()
        python = sim.get("python")
        return {"artifacts": self.artifacts(), "files": p.files,
                "simulator": {"python": python, "python_record": file_record(python) if python and
                              os.path.isfile(python) else None, "renderer": sim.get("renderer"),
                              "env": sim.get("env") or {},
                              "preload": {lib: file_record(lib) for lib in sim.get("preload_libraries") or []}},
                "collection": {k: p.data["collection"][k] for k in ("seed", "episodes", "qualify_episodes")},
                "runtime": {"runner.py": sha256_file(PACKAGE / "runner.py"),
                            "checks.py": sha256_file(PACKAGE / "checks.py")}}

    def resolved_scene(self) -> dict:
        p = self.p.project
        scene = specmod.plain(self.scene)
        scene["robot"]["urdf"] = str(contained(p.root, scene["robot"]["urdf"]))
        joints = self.urdf["joints"]
        scene["robot"]["joint_limits"] = {n: [joints[n]["lower"], joints[n]["upper"]] for n in scene["robot"]["arm_joints"]}
        for o in scene["objects"]:
            for key in ("collision", "visual"):
                if o["shape"].get(key):
                    o["shape"][key] = str(contained(p.root, o["shape"][key]))
            if o.get("texture"):
                o["texture"] = str(contained(p.root, o["texture"]))
        return scene

    def controller_problems(self, scene: dict, task: dict) -> list:
        c, robot = task["controller"], scene["robot"]
        vmax = float(robot["drive"]["max_joint_speed_rad_s"])
        limits = robot["joint_limits"]
        problems = []
        if c["type"] == "joint_waypoints":
            speed = min(vmax, float(c["max_joint_speed_rad_s"]))
            q = list(robot["init_arm_q"])
            for i, wp in enumerate(c["waypoints"]):
                for name, value in zip(robot["arm_joints"], wp["arm_q"]):
                    lo, hi = limits[name]
                    if not lo <= value <= hi:
                        problems.append(f"waypoint {i}: {name}={value} outside the URDF limits [{lo}, {hi}]")
                peak = 1.5 * max(abs(a - b) for a, b in zip(wp["arm_q"], q)) / float(wp["duration_s"])
                if peak > speed:
                    problems.append(f"waypoint {i}: peak joint speed {peak:.3g} rad/s exceeds {speed:.3g}")
                q = list(wp["arm_q"])
        for name, value in zip(robot["arm_joints"], robot["init_arm_q"]):
            lo, hi = limits[name]
            if not lo <= value <= hi:
                problems.append(f"init_arm_q: {name}={value} outside the URDF limits [{lo}, {hi}]")
        return problems

    def build(self, run: Path) -> dict:
        p, sim = self.p.project, self._sim()
        if self.scene is None or self.task is None or self.urdf is None:
            return outcome("FAILED", ["scene, task and URDF are required"], [])
        scene, task = self.resolved_scene(), specmod.plain(self.task)
        problems = self.controller_problems(scene, task)
        c = p.data["collection"]
        try:
            layouts = {"primary": specmod.sample_layouts(self.scene, c["seed"], c["episodes"], "primary"),
                       "qualify": specmod.sample_layouts(self.scene, c["seed"], c["qualify_episodes"], "qualify")}
        except ValueError as error:
            problems.append(str(error))
            layouts = {}
        if problems:
            return outcome("FAILED", problems, [])
        write_new_json(run / "layouts.json", layouts)
        runtime = run / "runtime"
        runtime.mkdir()
        staged = {}
        for name in ("runner.py", "checks.py"):
            write_new(runtime / name, (PACKAGE / name).read_bytes())
            staged[name] = sha256_file(runtime / name)
        plugin_sha = None
        if task["controller"]["type"] == "plugin":
            source = p.spec_path("plugin")
            if source is None:
                return outcome("FAILED", ["a plugin controller needs specs.plugin"], [])
            try:   # size checked before copying; a symlink or a file that grows is refused
                data = read_bounded(source, MAX_PLUGIN_BYTES)
            except PathError as error:
                return outcome("FAILED", [f"specs.plugin: {error}"], [])
            write_new(runtime / "controller_plugin.py", data)
            plugin_sha = sha256_file(runtime / "controller_plugin.py")
            if plugin_sha != self.artifacts()["plugin"]:
                return outcome("FAILED", ["plugin bytes changed while staging"], [])
            staged["controller_plugin.py"] = plugin_sha
        write_new_json(run / "RUNTIME.json", staged)
        probe = run_bounded([sim["python"], "-I", str(runtime / "runner.py"), "probe", "--out", str(run / "probe"),
                             "--renderer", sim["renderer"],
                             *sum((["--preload", lib] for lib in sim.get("preload_libraries") or []), [])],
                            cwd=run, env=self.child_env(), timeout_s=PROBE_TIMEOUT_S,
                            stdout=run / "probe.stdout.txt", stderr=run / "probe.stderr.txt",
                            output_limit_bytes=PROBE_OUTPUT_LIMIT_BYTES)
        report, probe_problems = probe_acceptance(probe, run / "probe" / "PROBE.json")
        build = {"schema": "neo-vla-real2sim-build/1", "scene": scene, "task": task,
                 "scene_reviewed": self.scene, "task_reviewed": self.task, "artifacts": self.artifacts(),
                 "urdf": {"links": self.urdf["links"], "joints": self.urdf["joints"], "meshes": self.urdf["meshes"]},
                 "runtime": staged, "plugin_sha256": plugin_sha, "probe": report, "probe_run": probe,
                 "probe_problems": probe_problems, "renderer": sim["renderer"],
                 "instruction": p.data["instruction"], "synthetic_fixture": p.synthetic,
                 "utc": utc_now()}
        write_new_json(run / "BUILD.json", build)
        outputs = (["BUILD.json", "layouts.json", "RUNTIME.json"] + [f"runtime/{n}" for n in staged]
                   + _present(run, ["probe.stdout.txt", "probe.stderr.txt", "probe/PROBE.json"]))
        not_established = ["task physics and rendering are UNVERIFIED until qualify",
                           "the probe only shows that the simulator imports and draws a frame"]
        if probe_problems:
            stderr = tail(run / "probe.stderr.txt", 800).strip()
            reasons = [f"simulator probe failed: {problem}" for problem in probe_problems]
            reasons += [f"probe stderr (end): {stderr}"] if stderr else []
            return outcome("FAILED", reasons, outputs, not_established)
        return outcome("PASSED", [], outputs, not_established, physics="UNVERIFIED", rendering="UNVERIFIED")

    # ------------------------------------------------------------------------- rollouts
    def _build(self, build: dict) -> tuple:
        run = build["dir"]   # absolute, resolved inside the workdir by the pipeline
        return run, read_json(run / "BUILD.json"), read_json(run / "layouts.json")

    def run_attempt(self, attempt: Path, pool: str, index: int, number: int, build: dict) -> dict:
        run, info, layouts = self._build(build)
        layout = layouts[pool][index]
        attempt.mkdir(parents=True, exist_ok=False)
        runtime = run / "runtime"
        spec = {"schema": SPEC_SCHEMA, "instruction": info["instruction"], "renderer": info["renderer"],
                "preload_libraries": self._sim().get("preload_libraries") or [],
                "episode": {"index": index, "pool": pool, "attempt": number, "layout": layout,
                            "layout_sha256": layout["layout_sha256"]},
                "scene": info["scene"], "task": info["task"], "scene_reviewed": info["scene_reviewed"],
                "task_reviewed": info["task_reviewed"], "scene_sha256": info["artifacts"]["scene"],
                "task_sha256": info["artifacts"]["task"],
                "runtime": {"runner_sha256": info["runtime"]["runner.py"], "checks_sha256": info["runtime"]["checks.py"],
                            "plugin_path": str(runtime / "controller_plugin.py") if info["plugin_sha256"] else None,
                            "plugin_sha256": info["plugin_sha256"]},
                "epoch_s": (1.9e9 if pool == "qualify" else 1.8e9) + index * 1e4}
        if info.get("synthetic_fixture") is True:   # the runner copies it into the episode manifest
            spec["synthetic_fixture"] = True
        write_new_json(attempt / "SPEC.json", spec)
        timeout = float(self.p.project.data["collection"]["attempt_timeout_s"])
        argv = [self._sim()["python"], "-I", str(runtime / "runner.py"), "rollout", "--spec",
                str(attempt / "SPEC.json"), "--out", str(attempt / "out")]
        write_new_json(attempt / "IDENTITY.json", {
            "schema": ATTEMPT_SCHEMA, "kind": "attempt_identity", "index": index, "pool": pool, "attempt": number,
            "layout_sha256": layout["layout_sha256"], "spec_sha256": sha256_file(attempt / "SPEC.json"),
            "build_run": run.name, "argv": argv, "timeout_s": timeout, "host": socket.gethostname(),
            "started_utc": utc_now()})
        res = run_bounded(argv, cwd=attempt, env=self.child_env(), timeout_s=timeout,
                          stdout=attempt / "stdout.txt", stderr=attempt / "stderr.txt")
        rr, _ = _json_object(attempt / "out" / "RUNNER_RESULT.json")
        # the process facts decide first; the runner's own success file cannot override them
        lifecycle = execution_problems(res, "the rollout process")
        accepted = res["returncode"] == 0 and rr.get("status") == "ACCEPTED_BY_RUNNER"
        status = ("TIMED_OUT" if res["timed_out"] else "OUTPUT_LIMIT" if res.get("output_limit_reached")
                  else "LAUNCH_ERROR" if res.get("launch_error")
                  else "RUNNER_ACCEPTED" if accepted and not lifecycle
                  else "UNCLEAN_EXIT" if accepted else "FAILED")
        result = {**res, "schema": ATTEMPT_SCHEMA, "kind": "attempt_result", "index": index, "pool": pool,
                  "attempt": number, "status": status, "lifecycle_problems": lifecycle,
                  "runner_status": rr.get("status"),
                  "runner_reasons": (rr.get("reasons") or [])[:20], "trace_sha256": rr.get("trace_sha256"),
                  "finished_utc": utc_now(),
                  "note": "driver view only; acceptance is decided by verify from the persisted files"}
        write_new_json(attempt / "RESULT.json", result)
        return result

    def qualify(self, run: Path, build: dict) -> dict:
        _, _, layouts = self._build(build)
        entries = []
        for index in range(len(layouts["qualify"])):
            attempt = run / "attempts" / f"q{index:03d}_a1"
            self.run_attempt(attempt, "qualify", index, 1, build)
            entries.append(self.verify_attempt(attempt, build))
        previews = run / "previews"
        for e in entries:
            for source in (Path(e["episode_dir"]) if e.get("episode_dir") else None,
                           run / "attempts" / e["attempt_id"] / "out" / "failures" / "raw_partial"):
                if source is not None and (source / "camera" / "color").is_dir():
                    for cam in sorted((source / "camera" / "color").iterdir()):
                        first = sorted(cam.glob("*.png"))[:1]
                        for f in first:
                            target = previews / f"{e['attempt_id']}_{cam.name}.png"
                            target.parent.mkdir(parents=True, exist_ok=True)
                            if not target.exists():
                                os.link(f, target)
                    break
        write_new_json(run / "QUALIFY.json", {"attempts": entries,
                                              "previews": sorted(x.name for x in previews.glob("*.png"))
                                              if previews.is_dir() else []})
        failed = [e for e in entries if e["status"] != "PASS"]
        reasons = [f"{e['attempt_id']}: {', '.join(e['failed'])}" for e in failed]
        return outcome("FAILED" if failed or not entries else "PASSED", reasons, ["QUALIFY.json"],
                       ["visual similarity to the real scene (compare previews/ with your camera images)",
                        "qualification rollouts are never counted as dataset episodes"])

    def targets(self, build: dict) -> list:
        return list(range(int(self.p.project.data["collection"]["episodes"])))

    def collect(self, namespace: Path, indices: list, workers: int, build: dict) -> dict:
        limit = int(self.p.project.data["collection"]["max_attempts_per_episode"])
        tasks, skipped = [], 0
        for index in indices:
            done = any(read_json(a / "RESULT.json").get("status") == "RUNNER_ACCEPTED"
                       for a in (namespace / "attempts").glob(f"ep{index:05d}_a*") if (a / "RESULT.json").is_file())
            used = attempts_used(namespace, index)
            if done or used >= limit:
                skipped += 1
                continue
            tasks.append((index, used + 1))

        def one(task: tuple) -> dict:
            index, number = task
            return self.run_attempt(namespace / "attempts" / f"ep{index:05d}_a{number}", "primary", index, number,
                                    build)

        results = bounded_pool(tasks, workers, one)
        return {"accepted": sum(r["status"] == "RUNNER_ACCEPTED" for r in results),
                "failed": sum(r["status"] != "RUNNER_ACCEPTED" for r in results), "skipped": skipped,
                "attempts": [{k: r[k] for k in ("index", "attempt", "status")} for r in results]}

    # ------------------------------------------------------------------------- verification
    def verify_attempt(self, attempt: Path, build: dict) -> dict:
        _, info, layouts = self._build(build)
        result: dict = {"attempt_id": attempt.name, "checks": {}, "split": "sampled"}
        checks_out = result["checks"]

        def check(name: str, problems: list) -> None:
            checks_out[name] = {"ok": not problems, "problems": problems[:20]}

        identity = read_json(attempt / "IDENTITY.json") if (attempt / "IDENTITY.json").is_file() else {}
        res = read_json(attempt / "RESULT.json") if (attempt / "RESULT.json").is_file() else {}
        spec_path = attempt / "SPEC.json"
        spec = read_json(spec_path) if spec_path.is_file() else {}
        spec_sha = sha256_file(spec_path) if spec_path.is_file() else None   # None fails below
        index, pool, number = identity.get("index"), identity.get("pool"), identity.get("attempt")
        named = ATTEMPT_NAME.fullmatch(attempt.name)
        if not identity and named:   # interrupted before IDENTITY.json: attribute it by its directory name
            index, number = int(named.group(2)), int(named.group(3))
            pool = "qualify" if named.group(1) == "q" else "primary"
        result.update(index=index, pool=pool, attempt=number)
        problems = []
        if not identity or not res or not spec:
            problems.append("IDENTITY.json, RESULT.json or SPEC.json missing")
        else:
            if (res.get("index"), res.get("pool"), res.get("attempt")) != (index, pool, identity.get("attempt")):
                problems.append("RESULT does not belong to this attempt")
            if identity.get("spec_sha256") != spec_sha:
                problems.append("SPEC.json changed after the attempt started")
        if res.get("status") == "INCOMPLETE":
            problems.append("the attempt was interrupted before it finished")
        elif res:   # the recorded process facts, re-checked here whatever status the driver wrote
            problems += execution_problems(res, "the rollout process")
        check("attempt_records", problems)
        out = attempt / "out"
        rr, rr_problem = _json_object(out / "RUNNER_RESULT.json")
        if rr.get("status") == "ACCEPTED_BY_RUNNER" and not rr.get("reasons"):
            problems = []
        elif rr_problem and (out / "RUNNER_RESULT.json").exists():
            problems = [rr_problem]
        else:
            first = (rr.get("reasons") or ["no result"])[0]
            problems = [f"runner status {rr.get('status')!r}: {first}"]
        check("runner_result", problems)
        problems = []
        layout = None
        if spec and pool in layouts and isinstance(index, int) and index < len(layouts[pool]):
            layout = layouts[pool][index]
            result["layout_sha256"] = layout["layout_sha256"]
            if spec.get("episode", {}).get("layout") != layout:
                problems.append("SPEC layout differs from the build layout")
        else:
            problems.append("attempt does not map to a build layout")
        for key, want in (("scene_sha256", info["artifacts"]["scene"]), ("task_sha256", info["artifacts"]["task"])):
            if spec.get(key) != want:
                problems.append(f"SPEC {key} differs from the reviewed build")
        runtime = spec.get("runtime") or {}
        if runtime.get("runner_sha256") != info["runtime"]["runner.py"] or \
                runtime.get("checks_sha256") != info["runtime"]["checks.py"] or \
                runtime.get("plugin_sha256") != info["plugin_sha256"]:
            problems.append("SPEC runtime hashes differ from the staged build")
        if rr and (rr.get("runner_sha256") != info["runtime"]["runner.py"] or
                   rr.get("checks_sha256") != info["runtime"]["checks.py"] or
                   rr.get("spec_sha256") != spec_sha):
            problems.append("the runner reports different runtime or spec bytes")
        if spec.get("scene") != info["scene"] or spec.get("task") != info["task"]:
            problems.append("SPEC scene/task differ from the build")
        if (spec.get("synthetic_fixture") is True) != (info.get("synthetic_fixture") is True):
            problems.append("SPEC synthetic_fixture flag differs from the build")
        check("provenance", problems)
        problems = []
        trace, meta_path = out / "trace.npz", out / "trace.json"
        audit = None
        if not trace.is_file() or not meta_path.is_file():
            problems.append("persisted trace missing")
        else:
            result["trace_sha256"] = sha256_file(trace)
            if rr.get("trace_sha256") != result["trace_sha256"]:
                problems.append("trace.npz differs from the runner's record")
            try:
                meta = read_json(meta_path)
                arrays = checks.load_npz(trace)
                result["trace_content_sha256"] = checks.content_sha256(arrays)   # arrays only
                audit = checks.audit_trace(info["task"], meta, arrays)
                if meta.get("physics_hz") != info["scene"]["physics"]["physics_hz"] or \
                        [o["name"] for o in meta.get("objects", [])] != [o["name"] for o in info["scene"]["objects"]]:
                    problems.append("trace metadata does not describe the built scene")
                if not audit["ok"]:
                    problems += audit["reasons"]
                if rr.get("audit") is not None and checks.jsonable(rr["audit"]) != audit:
                    problems.append("re-running the trace audit does not reproduce the runner's audit")
                if layout is not None:
                    problems += checks.initial_pose_problems(meta, arrays, layout)
            except (OSError, ValueError, KeyError, TypeError, IndexError, zipfile.BadZipFile) as error:
                problems.append(f"trace unreadable: {error!r}")
        check("trace_audit", problems)
        result["audit"] = None if audit is None else {k: audit.get(k) for k in ("ok", "success", "motion")}
        problems = []
        episode = out / "episode"
        if not (episode / "manifest.json").is_file():
            problems.append("no finalized episode")
        else:
            report = checks.raw_episode_report(episode)
            for cat, r in report["categories"].items():
                problems += [f"{cat}: {x}" for x in r["problems"][:3]]
            manifest, manifest_problem = _json_object(episode / "manifest.json")
            problems += [manifest_problem] if manifest_problem else []
            sim = manifest.get("simulation") or {}
            if sim.get("spec_sha256") != spec_sha or sim.get("trace_sha256") != result.get("trace_sha256"):
                problems.append("episode manifest does not name this spec and trace")
            if info.get("synthetic_fixture") is True and sim.get("synthetic_fixture") is not True:
                problems.append("the project is a synthetic fixture but the episode manifest does "
                                "not say so (simulation.synthetic_fixture)")
            if manifest.get("task") != info["instruction"]:
                problems.append("episode instruction differs from the project")
            if not (manifest.get("training_sync") or {}).get("task_success_verified"):
                problems.append("episode not finalized as verified by the runner")
            cams = [(c["name"], c["K"], c["width"], c["height"], c.get("every_n_rows", 1))
                    for c in info["scene"]["cameras"]]
            if [(c["name"], c["K"], c["width"], c["height"], c["every_n_rows"]) for c in manifest.get("cameras", [])] \
                    != cams:
                problems.append("episode cameras differ from the reviewed scene")
            result["episode_dir"] = str(episode)
        check("recording", problems)
        problems = []
        readback = rr.get("camera_readback") or {}
        for cam in info["scene"]["cameras"]:
            got = readback.get(cam["name"])
            if got is None:
                problems.append(f"{cam['name']}: no camera readback")
                continue
            k = cam["K"]
            want = [k[0][0], k[1][1], k[0][2], k[1][2]]
            if got.get("width_height") != [cam["width"], cam["height"]] or not all(
                    math.isclose(a, b, rel_tol=READBACK_RTOL, abs_tol=READBACK_ATOL)
                    for a, b in zip(got.get("fx_fy_cx_cy", []), want)) or len(got.get("fx_fy_cx_cy", [])) != 4:
                problems.append(f"{cam['name']}: rendered intrinsics differ from the calibration")
        device = str(rr.get("device") or "")
        if info["renderer"] == "cpu" and not any(x in device.lower() for x in ("llvmpipe", "lavapipe")):
            problems.append(f"CPU rendering reported device {device!r}")
        check("camera_readback", problems)
        result["failed"] = sorted(name for name, c in checks_out.items() if not c["ok"])
        result["status"] = "PASS" if not result["failed"] else "FAIL"
        return result

    def verify(self, run: Path, namespace: Path, build: dict) -> dict:
        entries = []
        base = namespace / "attempts"
        for attempt in sorted(base.iterdir()) if base.is_dir() else []:
            if attempt.is_dir() and (attempt / "RESULT.json").is_file():
                entries.append(self.verify_attempt(attempt, build))
        return write_audit(run, entries, self.targets(build), reject_duplicates(entries))

    # ------------------------------------------------------------------------- misc
    def probe(self, run: Path) -> dict:
        sim = self._sim()
        runtime = run / "runtime"
        runtime.mkdir()
        for name in ("runner.py", "checks.py"):
            write_new(runtime / name, (PACKAGE / name).read_bytes())
        res = run_bounded([sim["python"], "-I", str(runtime / "runner.py"), "probe", "--out", str(run / "probe"),
                           "--renderer", sim.get("renderer", "cpu"),
                           *sum((["--preload", lib] for lib in sim.get("preload_libraries") or []), [])],
                          cwd=run, env=self.child_env(), timeout_s=PROBE_TIMEOUT_S,
                          stdout=run / "stdout.txt", stderr=run / "stderr.txt",
                          output_limit_bytes=PROBE_OUTPUT_LIMIT_BYTES)
        report, problems = probe_acceptance(res, run / "probe" / "PROBE.json")
        if report is not None:
            self.p.log(json.dumps(report.get("versions"), indent=1))
        stderr = tail(run / "stderr.txt", 800).strip()
        reasons = problems + ([f"probe stderr (end): {stderr}"] if problems and stderr else [])
        outputs = ["runtime/runner.py", "runtime/checks.py"]
        outputs += _present(run, ["stdout.txt", "stderr.txt", "probe/PROBE.json"])
        return outcome("FAILED" if problems else "PASSED", reasons, outputs,
                       ["a probe is not a validation of any task's physics or rendering"],
                       probe_run=res)

    def diagnose_evidence(self, attempt: Path) -> dict:
        evidence = super().diagnose_evidence(attempt)
        if (attempt / "SPEC.json").is_file() and (attempt / "SPEC.json").stat().st_size < 2 << 20:
            evidence["SPEC.json"] = attempt / "SPEC.json"
        return evidence

    def disclosures(self) -> list:
        return ["Episodes come from a scripted controller in SAPIEN; quantities marked estimate in the reviewed "
                "scene are not calibrated values.",
                "The success contract and integrity tolerances were chosen and reviewed by the project author.",
                "Only rollouts that passed every check are exported (success-conditioned); failed attempts and "
                "their evidence stay in the workdir."]

    def conversion_settings(self) -> dict:
        return {"state": "measured TCP in the robot base frame: xyz cm, quaternion xyzw, aperture / stroke",
                "action": "next-minus-current commanded TCP: dxyz cm, spatial rotvec rad, next commanded "
                          "aperture / stroke",
                "image_sync": "latest capture at or before the row time (causal)", "control_hz": 30,
                "note": "convert with your own corpus tooling; this repository does not include corpus builders"}


def _job_summary(record: dict) -> dict:
    return {k: record.get(k) for k in ("job", "attempt", "status", "reasons", "requested_model", "reported_models",
                                      "auxiliary_models", "result", "returncode", "timed_out", "wall_s")}
