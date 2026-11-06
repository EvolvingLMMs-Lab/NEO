"""Command line: python -m neo_vla.real2sim <command> --project path/to/project.json

Exit codes: 0 passed (or export CONVERSION_READY), 2 held, partial or RAW_SCHEMA_VALID, 3 failed,
4 refused (a precondition is not met or another process holds the workdir).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from . import agent, templates
from .pipeline import Pipeline, Refused
from .store import LockedError, PathError, write_new_json


def _init(directory: str, route: str) -> int:
    root = Path(directory)
    if root.exists() and any(root.iterdir()):
        print(f"{root} exists and is not empty; choose a new directory")
        return 4
    for sub in ("specs", "reviews") if route == "sapien" else ("reviews",):
        (root / sub).mkdir(parents=True, exist_ok=True)
    write_new_json(root / "project.json", templates.project(route))
    print(f"wrote {root / 'project.json'}. Fill in the inputs, then run:\n"
          f"  python -m neo_vla.real2sim check --project {root / 'project.json'}")
    return 0


def _auth(executable: str) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        status = agent.auth_status(executable, Path(tmp))
    print(json.dumps(status, indent=1))
    if status["version"]["returncode"] != 0:
        print(f"{executable} is not runnable; install Claude Code first")
        return 3
    if status["auth_status"]["returncode"] != 0:
        print("not logged in: run `claude auth login`, then check with `claude auth status`")
        return 3
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m neo_vla.real2sim", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    init = sub.add_parser("init", help="write a project template into a new directory")
    init.add_argument("directory")
    init.add_argument("--route", choices=("sapien", "can_in_tape_reference"), default="sapien")
    auth = sub.add_parser("auth", help="show the Claude Code CLI version and login state")
    auth.add_argument("--executable", default="claude")

    def project(name: str, help_text: str) -> argparse.ArgumentParser:
        s = sub.add_parser(name, help=help_text)
        s.add_argument("--project", required=True, help="path of project.json")
        s.add_argument("--workdir", help="pipeline state directory (default: <project dir>/work)")
        return s

    project("check", "validate inputs and list requests")
    s = project("plan", "author candidate scene/task/plugin specifications (agent jobs when enabled)")
    s.add_argument("--author", default="", help="comma list of scene,task,plugin to author even if present")
    s = project("review", "show review items, promote candidates or record decisions")
    s.add_argument("--promote", nargs="+", choices=("scene", "task", "plugin"))
    s.add_argument("--reviewer")
    s.add_argument("--accept", nargs="*", default=[])
    s.add_argument("--reject", nargs="*", default=[])
    s.add_argument("--accept-all", action="store_true")
    s.add_argument("--notes", default="")
    s.add_argument("--rerun", action="store_true")
    project("probe", "check that the simulator environment imports SAPIEN and renders")
    for name, text in (("build", "stage the reviewed specifications and probe the simulator"),
                       ("qualify", "run and verify the qualification rollouts"),
                       ("verify", "check every collected attempt independently")):
        project(name, text).add_argument("--rerun", action="store_true")
    s = project("collect", "run bounded rollout attempts")
    s.add_argument("--indices", help="episode indices for this invocation, e.g. 0-9,15 (default: all)")
    s.add_argument("--workers", type=int)
    s = project("export", "export verified episodes and run the export checks")
    s.add_argument("--allow-partial", action="store_true")
    s = project("admit", "record that an exported set passed your training-side checks")
    s.add_argument("--export", required=True, help="export run id")
    s.add_argument("--evidence", required=True, help="report file from your corpus build")
    s.add_argument("--by", required=True)
    s.add_argument("--statement", required=True)
    project("status", "show the state of every stage")
    s = project("diagnose", "ask the agent for an advisory diagnosis of a failed attempt")
    s.add_argument("--attempt", required=True, help="attempt directory inside the workdir")
    return p


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    if a.cmd == "init":
        return _init(a.directory, a.route)
    if a.cmd == "auth":
        return _auth(a.executable)
    try:
        pipe = Pipeline(a.project, a.workdir, read_only=a.cmd == "status")
        if a.cmd == "check":
            return pipe.check()
        if a.cmd == "plan":
            return pipe.plan({x for x in a.author.split(",") if x})
        if a.cmd == "review":
            if a.promote:
                return pipe.promote(a.promote)
            if a.reviewer is not None:
                code = pipe.decide(a.reviewer, a.accept, a.reject, a.accept_all, a.notes)
                if code:
                    return code
                pipe = Pipeline(a.project, a.workdir)
            return pipe.review(a.rerun)
        if a.cmd in ("build", "qualify", "verify"):
            return getattr(pipe, a.cmd)(a.rerun)
        if a.cmd == "collect":
            return pipe.collect(a.indices, a.workers)
        if a.cmd == "export":
            return pipe.export(a.allow_partial)
        if a.cmd == "admit":
            return pipe.admit(a.export, a.evidence, a.by, a.statement)
        if a.cmd == "probe":
            return pipe.probe()
        if a.cmd == "diagnose":
            return pipe.diagnose(a.attempt)
        return pipe.status()
    except (Refused, LockedError, PathError, FileExistsError) as error:
        print(f"refused: {error}")
        return 4


if __name__ == "__main__":
    sys.exit(main())
