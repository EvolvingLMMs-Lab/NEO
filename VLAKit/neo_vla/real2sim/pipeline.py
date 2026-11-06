"""Stage workflow: check, plan, review, build, qualify, collect, verify, export, admit.

A stage is up to date only when its newest run for exactly the inputs it depends on now (file
hashes, upstream receipts, configuration) passed and the files its receipt lists still hash as
recorded. When that newest run failed or was interrupted, the stage is failed or held; an earlier
pass is never reused. Every run is a new, sequence-numbered directory under the workdir, nothing is
overwritten, and one process writes a workdir at a time. Run ids are names, never paths: their
directories are always resolved inside the workdir. `plan` only writes candidates; `review` pins the
exact specifications a build may use.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable

from . import __version__, agent, checks
from . import spec as specmod
from .store import (Lock, PathError, canonical_sha256, close_incomplete, file_record, latest, load_receipt, new_run,
                    read_json, run_dir, runs, utc_now, write_new, write_new_json, write_receipt)

EXIT = {"PASSED": 0, "CONVERSION_READY": 0, "TRAINING_ADMITTED": 0, "HELD": 2, "PARTIAL": 2,
        "RAW_SCHEMA_VALID": 2, "UNVERIFIED": 2, "INCOMPLETE": 2, "FAILED": 3}
INTAKE_REUSABLE = ("PASSED", "HELD", "FAILED")
EXPORT_SCHEMA = "neo-vla-real2sim-export/1"
NOT_MEASURED = [
    "visual similarity between rendered and real camera images (compare the qualification previews yourself)",
    "accuracy of estimated physical parameters (mass, friction, restitution, drive gains)",
    "sim-to-real transfer, policy performance or deployment reliability",
    "pixel content of frames beyond presence, PNG header and size",
]


class Refused(RuntimeError):
    """A precondition is not met (stage out of date, held, failed, locked or invalid request)."""


def outcome(status: str, reasons: Iterable[str] = (), outputs: Iterable[str] = (),
            not_established: Iterable[str] = (), **extra: Any) -> dict:
    return {"status": status, "reasons": list(reasons), "outputs": list(outputs),
            "not_established": list(not_established), **extra}


class Route:
    """Backend adapter API. A route turns reviewed inputs into verified raw episodes.

    build(run) / qualify(run) / verify(run) return an outcome(); collect(namespace, ...) adds attempt
    directories, each finished by its own RESULT.json. Routes report what they do not establish.
    """

    name = ""
    synthetic = False

    def __init__(self, pipeline: "Pipeline") -> None:
        self.p = pipeline

    def intake(self, issues: specmod.Issues) -> dict:
        return {}

    def review(self) -> tuple:
        return {}, []

    def build_inputs(self) -> dict:
        raise NotImplementedError

    def build(self, run: Path) -> dict:
        raise NotImplementedError

    def qualify(self, run: Path, build: dict) -> dict:
        raise NotImplementedError

    def collect(self, namespace: Path, indices: list, workers: int, build: dict) -> dict:
        raise NotImplementedError

    def targets(self, build: dict) -> list:
        raise NotImplementedError

    def verify(self, run: Path, namespace: Path, build: dict) -> dict:
        raise NotImplementedError

    def plan(self, run: Path, author: set) -> dict:
        return outcome("PASSED", ["this route has a fixed scene and task; nothing to author"])

    def probe(self, run: Path) -> dict:
        return outcome("PASSED", ["no simulator probe for this route"])

    def disclosures(self) -> list:
        return []

    def conversion_settings(self) -> dict:
        return {}

    def diagnose_evidence(self, attempt: Path) -> dict:
        evidence: dict = {}
        for name in ("IDENTITY.json", "RESULT.json", "out/RUNNER_RESULT.json", "out/trace.json"):
            if (attempt / name).is_file():
                evidence[name.replace("/", "_")] = attempt / name
        if (attempt / "stderr.txt").is_file():
            evidence["stderr_tail.txt"] = (attempt / "stderr.txt").read_bytes()[-20000:]
        return evidence


def make_route(pipeline: "Pipeline") -> Route:
    if pipeline.project.route == "sapien":
        from .sapien_route import SapienRoute

        return SapienRoute(pipeline)
    if pipeline.project.route == "can_in_tape_reference":
        from .reference import ReferenceRoute

        return ReferenceRoute(pipeline)
    return Route(pipeline)


def bounded_pool(tasks: list, workers: int, fn: Callable[[Any], Any]) -> list:
    """Run fn over tasks with at most `workers` concurrent child processes."""
    if workers <= 1:
        return [fn(t) for t in tasks]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, tasks))


def link_tree(source: Path, target: Path) -> dict:
    """Hard-link a finished episode into an export; returns {relative path: size}."""
    listing = {}
    for current, dirs, files in os.walk(source):
        dirs.sort()
        rel = Path(current).relative_to(source)
        (target / rel).mkdir(parents=True, exist_ok=rel != Path("."))
        for name in sorted(files):
            src = Path(current) / name
            if src.is_symlink():
                raise PathError(f"{src}: symlinks are not exported")
            os.link(src, target / rel / name)
            listing[(rel / name).as_posix()] = src.stat().st_size
    return listing


class Pipeline:
    def __init__(self, project_path: str | Path, workdir: str | Path | None = None, log: Callable = print,
                 read_only: bool = False) -> None:
        self.project = specmod.Project(Path(project_path))
        self.workdir = Path(workdir).resolve() if workdir else self.project.root / "work"
        self.log = log
        self.read_only = read_only
        self.route = make_route(self)
        # Route validation may add files (meshes, textures) to project.files, so it runs once, before
        # the intake fingerprint is computed.
        self.route_info = self.route.intake(self.project.issues) if not self.project.issues.errors else {}

    # ------------------------------------------------------------------------- plumbing
    def stage_dir(self, stage: str) -> Path:
        return self.workdir / "stages" / stage

    def agent_config(self) -> dict:
        return dict(self.project.data.get("agent") or {})

    def _close_incomplete(self) -> None:
        """Runs that ended without a receipt get an INCOMPLETE one in place (lock held)."""
        stages = self.workdir / "stages"
        if not stages.is_dir():
            return
        for stage in sorted(p for p in stages.iterdir() if p.is_dir()):
            for name in close_incomplete(stage):
                self.log(f"{stage.name}: run {name} ended without a receipt; recorded as INCOMPLETE")

    def _receipt_state(self, stage: str, inputs: Any) -> tuple:
        run, receipt, problems = latest(self.stage_dir(stage), inputs)
        if receipt is None:
            return None, None, "not run for the current inputs"
        if problems:
            return run, receipt, "latest run is not usable: " + "; ".join(problems[:3])
        return run, receipt, None

    def intake_inputs(self) -> dict:
        return {"stage": "intake", "version": __version__, "files": self.project.files,
                "workdir": str(self.workdir)}

    def intake(self) -> tuple:
        """Validate the project now; reuse a passed receipt for identical inputs."""
        inputs = self.intake_inputs()
        run, receipt, problem = self._receipt_state("intake", inputs)
        if receipt is not None and problem is None and receipt.get("status") in INTAKE_REUSABLE:
            return run, receipt
        if self.read_only:
            raise Refused("intake has not run for the current inputs; run `check`")
        issues = self.project.issues
        summary = {**self.project.summary(), **self.route_info}
        status = "FAILED" if issues.errors else "HELD" if issues.requests else "PASSED"
        run = new_run(self.stage_dir("intake"), inputs)
        write_new_json(run / "intake.json", summary)
        write_new_json(run / "requests.json", {"requests": issues.requests, "errors": issues.errors})
        receipt = write_receipt(run, stage="intake", status=status, inputs=inputs,
                                outputs=["intake.json", "requests.json"],
                                reasons=[f"{e['field']}: {e['problem']}" for e in issues.errors]
                                + [f"{r['field']}: {r['problem']}" for r in issues.requests],
                                not_established=["intake checks inputs and their provenance only"],
                                blocked_stages=issues.blocked())
        return run, receipt

    def require_intake(self, stage: str) -> tuple:
        run, receipt = self.intake()
        if receipt["status"] == "FAILED":
            raise Refused("the project has errors; run `check` and fix them first")
        blocked = set(receipt.get("blocked_stages") or [])
        if stage in blocked or "all" in blocked:
            raise Refused(f"open requests block {stage}; run `check` to list them")
        return run, receipt

    def accepted(self, stage: str) -> tuple:
        """The accepted receipt of a stage for the current inputs, or Refused with the reason."""
        inputs = self.inputs_for(stage)
        run, receipt, problem = self._receipt_state(stage, inputs)
        if problem:
            raise Refused(f"{stage}: {problem}; run `{stage}` first")
        if receipt["status"] != "PASSED":
            raise Refused(f"{stage} is {receipt['status']} for the current inputs: " +
                          "; ".join(receipt.get("reasons", [])[:3]))
        return run, receipt

    def inputs_for(self, stage: str) -> dict:
        if stage == "review":
            _, intake = self.require_intake("review")
            return {"stage": "review", "intake": intake["outputs_fingerprint"], "decision": self._decision_record()}
        if stage == "build":
            _, review = self.accepted("review")
            return {"stage": "build", "review": review["outputs_fingerprint"], "route": self.route.build_inputs()}
        if stage == "qualify":
            _, build = self.accepted("build")
            c = self.project.data["collection"]
            return {"stage": "qualify", "build": build["outputs_fingerprint"],
                    "qualify_episodes": c["qualify_episodes"], "attempt_timeout_s": c["attempt_timeout_s"]}
        if stage == "collect":
            _, qualify = self.accepted("qualify")
            c = self.project.data["collection"]
            return {"stage": "collect", "qualify": qualify["outputs_fingerprint"],
                    **{k: c[k] for k in ("seed", "episodes", "max_attempts_per_episode", "attempt_timeout_s")}}
        if stage == "verify":
            namespace = self.namespace()
            return {"stage": "verify", "collect": namespace.name, "attempts": attempt_records(namespace)}
        if stage == "export":
            _, verify = self.accepted_verify()
            return {"stage": "export", "verify": verify["outputs_fingerprint"],
                    "output": self.project.data.get("output")}
        raise ValueError(stage)

    def accepted_verify(self, allow_partial: bool = False) -> tuple:
        run, receipt, problem = self._receipt_state("verify", self.inputs_for("verify"))
        if problem:
            raise Refused(f"verify: {problem}; run `verify` first")
        if receipt["status"] == "PASSED" or (allow_partial and receipt["status"] == "PARTIAL"):
            return run, receipt
        raise Refused(f"verify is {receipt['status']}: " + "; ".join(receipt.get("reasons", [])[:3])
                      + ("" if allow_partial else " (use --allow-partial to export the verified subset)"))

    def namespace(self) -> Path:
        inputs = self.inputs_for("collect")
        return self.stage_dir("collect") / f"inputs-{canonical_sha256(inputs)[:16]}"

    def _finish(self, stage: str, run: Path, inputs: dict, result: dict) -> int:
        receipt = write_receipt(run, stage=stage, status=result["status"], inputs=inputs, outputs=result["outputs"],
                                reasons=result["reasons"], not_established=result["not_established"],
                                **{k: v for k, v in result.items()
                                   if k not in ("status", "outputs", "reasons", "not_established")})
        self.log(f"{stage}: {receipt['status']}  ({run})")
        for reason in receipt["reasons"][:20]:
            self.log(f"  - {reason}")
        for item in receipt["not_established"]:
            self.log(f"  not established: {item}")
        return EXIT.get(receipt["status"], 3)

    def _stage(self, stage: str, body: Callable[[Path], dict], rerun: bool) -> int:
        inputs = self.inputs_for(stage)
        run, receipt, problem = self._receipt_state(stage, inputs)
        if receipt is not None and problem is None and receipt["status"] == "PASSED" and not rerun:
            self.log(f"{stage}: up to date ({run.name}); use --rerun to run it again")
            return 0
        run = new_run(self.stage_dir(stage), inputs)
        return self._finish(stage, run, inputs, body(run))

    def locked(self, command: str, fn: Callable[[], int]) -> int:
        with Lock(self.workdir, command):
            self._close_incomplete()
            return fn()

    # ------------------------------------------------------------------------- commands
    def check(self) -> int:
        def body() -> int:
            run, receipt = self.intake()
            self.log(f"check: {receipt['status']}  ({run})")
            summary = read_json(run / "intake.json")
            for e in summary.get("errors", []):
                self.log(f"  error   {e['field']}: {e['problem']}")
            for r in summary.get("requests", []):
                self.log(f"  request {r['field']}: {r['problem']} -> {r['how_to_supply']} (blocks {r['blocks']})")
            return EXIT[receipt["status"]]

        return self.locked("check", body)

    def plan(self, author: set | None = None) -> int:
        def body() -> int:
            self.require_intake("plan")
            inputs = {"stage": "plan", "files": self.project.files, "agent": self.agent_config(),
                      "author": sorted(author or [])}
            run = new_run(self.stage_dir("plan"), inputs)
            return self._finish("plan", run, inputs, self.route.plan(run, set(author or [])))

        return self.locked("plan", body)

    def _review_state(self) -> tuple:
        artifacts, items = self.route.review()
        fp = canonical_sha256({"artifacts": artifacts, "items": specmod.review_digest(items)})
        return artifacts, items, fp, self.project.root / "reviews" / f"review-{fp[:12]}.json"

    def _decision_record(self) -> dict | None:
        _, _, _, path = self._review_state()
        return {"path": path.name, **file_record(path)} if path.is_file() else None

    def promote(self, names: list) -> int:
        def body() -> int:
            if self.project.issues.errors:
                raise Refused("the project has errors; run `check` and fix them first")
            for name in names:
                target = self.project.spec_path(name)
                if target is None:
                    raise Refused(f"specs.{name} is not declared in project.json")
                if target.exists():
                    raise Refused(f"{target} exists; edit it directly or move it away before promoting")
                candidate = {"scene": "scene.json", "task": "task.json"}.get(name, target.name)
                source = latest_candidate(self.stage_dir("plan"), candidate)
                if source is None:
                    raise Refused(f"no passed plan run has a candidate {candidate}")
                target.parent.mkdir(parents=True, exist_ok=True)
                write_new(target, source.read_bytes())
                self.log(f"promoted {source} -> {target}")
            return 0

        return self.locked("review", body)

    def decide(self, reviewer: str, accept: list, reject: list, accept_all: bool, notes: str) -> int:
        def body() -> int:
            self.require_intake("review")
            artifacts, items, fp, path = self._review_state()
            ids = {i["id"] for i in items}
            unknown = sorted((set(accept) | set(reject)) - ids)
            if unknown:
                raise Refused(f"unknown review item ids {unknown}")
            if not reviewer.strip():
                raise Refused("--reviewer is required")
            decisions = {i: "accept" for i in (ids if accept_all else accept)}
            decisions.update({i: "reject" for i in reject})
            write_new_json(path, {"schema": specmod.REVIEW_SCHEMA, "reviewer": reviewer.strip(), "utc": utc_now(),
                                  "artifacts": artifacts, "items_sha256": specmod.review_digest(items),
                                  "review_fingerprint": fp, "decisions": decisions, "notes": notes})
            self.log(f"review decision written: {path}")
            return 0

        return self.locked("review", body)

    def review(self, rerun: bool = False) -> int:
        def body() -> int:
            def run_review(run: Path) -> dict:
                artifacts, items, fp, path = self._review_state()
                write_new_json(run / "review.json", {"artifacts": artifacts, "items": items, "fingerprint": fp,
                                                     "decision_file": str(path)})
                if not path.is_file():
                    for i in items:
                        self.log(f"  [{i['id']}] {i['field']} ({i['kind']}): {json.dumps(i['detail'])[:160]}")
                    return outcome("HELD", [f"{len(items)} item(s) await a decision; record it with "
                                            f"`review --reviewer NAME --accept-all` or --accept/--reject IDS"],
                                   ["review.json"], ["a review records a person's decisions, not their correctness"])
                d = read_json(path)
                reasons = []
                if d.get("schema") != specmod.REVIEW_SCHEMA or d.get("artifacts") != artifacts or \
                        d.get("items_sha256") != specmod.review_digest(items) or not d.get("reviewer"):
                    reasons.append(f"{path.name} does not match the current artifacts and items")
                decisions = d.get("decisions") or {}
                missing = [i["id"] for i in items if i["id"] not in decisions]
                rejected = [i for i, v in decisions.items() if v != "accept"]
                if missing:
                    reasons.append(f"undecided items {missing[:10]}")
                if rejected:
                    reasons.append(f"rejected items {rejected[:10]}: change the inputs and review again")
                status = "FAILED" if rejected or (reasons and not missing) else "HELD" if missing else "PASSED"
                return outcome(status, reasons, ["review.json"],
                               ["a review records a person's decisions, not their correctness"],
                               reviewer=d.get("reviewer"), artifacts=artifacts)

            return self._stage("review", run_review, rerun)

        return self.locked("review", body)

    def build(self, rerun: bool = False) -> int:
        return self.locked("build", lambda: self._stage("build", self.route.build, rerun))

    def _build_receipt(self) -> dict:
        """The accepted build: its absolute directory inside the workdir, its run id and receipt."""
        run, receipt = self.accepted("build")
        return {"dir": run_dir(self.stage_dir("build"), run.name), "id": run.name, "receipt": receipt}

    def qualify(self, rerun: bool = False) -> int:
        def body() -> int:
            build = self._build_receipt()
            return self._stage("qualify", lambda run: self.route.qualify(run, build), rerun)

        return self.locked("qualify", body)

    def collect(self, indices: str | None, workers: int | None) -> int:
        def body() -> int:
            namespace = self.namespace()
            self.accepted("qualify")
            build = self._build_receipt()
            targets = self.route.targets(build)
            chosen = specmod.parse_indices(indices, max(targets) + 1) if indices else targets
            chosen = [i for i in chosen if i in set(targets)]
            namespace.mkdir(parents=True, exist_ok=True)
            write_inputs = namespace / "INPUTS.json"
            if not write_inputs.exists():
                write_new_json(write_inputs, self.inputs_for("collect"))
            for name in close_incomplete_attempts(namespace):
                self.log(f"collect: attempt {name} ended without a result; recorded as INCOMPLETE")
            n = workers or int(self.project.data["collection"]["workers"])
            summary = self.route.collect(namespace, chosen, n, build)
            previous = len(list((namespace / "invocations").glob("*.json"))) if (namespace / "invocations").is_dir() else 0
            log = namespace / "invocations" / f"{previous + 1:06d}-{utc_now().replace(':', '')}-{os.getpid()}.json"
            write_new_json(log, {"utc": utc_now(), "indices": chosen, "workers": n, **summary})
            self.log(f"collect: {summary.get('accepted', 0)} accepted by the runner, {summary.get('failed', 0)} "
                     f"failed, {summary.get('skipped', 0)} skipped (done or out of attempts); run `verify` next")
            return 0 if not summary.get("failed") else 2

        return self.locked("collect", body)

    def verify(self, rerun: bool = False) -> int:
        def body() -> int:
            build = self._build_receipt()
            namespace = self.namespace()
            return self._stage("verify", lambda run: self.route.verify(run, namespace, build), rerun)

        return self.locked("verify", body)

    def export(self, allow_partial: bool = False) -> int:
        def body() -> int:
            verify_run, verify = self.accepted_verify(allow_partial)
            inputs = {**self.inputs_for_export(verify), "allow_partial": allow_partial}
            run = new_run(self.stage_dir("export"), inputs)
            return self._finish("export", run, inputs, self._export(run, verify_run, allow_partial))

        return self.locked("export", body)

    def inputs_for_export(self, verify: dict) -> dict:
        return {"stage": "export", "verify": verify["outputs_fingerprint"], "output": self.project.data.get("output")}

    def admit(self, export_run: str, evidence: str, by: str, statement: str) -> int:
        def body() -> int:
            if self.project.synthetic:
                raise Refused("the project declares fixture.synthetic: true; a synthetic fixture "
                              "is never admitted for training")
            try:
                run = run_dir(self.stage_dir("export"), export_run)
            except PathError as error:
                raise Refused(f"--export expects an export run id: {error}") from error
            receipt, problems = load_receipt(run)
            if receipt is None or problems:
                raise Refused(f"export run {export_run} has no valid receipt: {problems}")
            if receipt["status"] != "CONVERSION_READY":
                raise Refused(f"export {export_run} is {receipt['status']}; only CONVERSION_READY exports can be "
                              "recorded as admitted")
            path = Path(evidence).resolve()
            if not path.is_file() or not by.strip() or not statement.strip():
                raise Refused("--evidence must be an existing file; --by and --statement are required")
            inputs = {"stage": "admit", "export": receipt["outputs_fingerprint"], "evidence": file_record(path)}
            out = new_run(self.stage_dir("admit"), inputs)
            write_new_json(out / "ADMISSION.json", {
                "schema": "neo-vla-real2sim-admission/1", "status": "TRAINING_ADMITTED", "export_run": export_run,
                "export_outputs_fingerprint": receipt["outputs_fingerprint"],
                "evidence": {"path": str(path), **file_record(path)}, "by": by.strip(), "statement": statement,
                "utc": utc_now(), "note": "recorded from your evidence; the pipeline did not run the training-side "
                                          "checks (corpus build, normalizer fit) itself"})
            return self._finish("admit", out, inputs,
                                outcome("TRAINING_ADMITTED", [], ["ADMISSION.json"],
                                        ["training-side checks were run by you, outside this pipeline"]))

        return self.locked("admit", body)

    def probe(self) -> int:
        def body() -> int:
            inputs = {"stage": "probe", "utc": utc_now()}
            run = new_run(self.stage_dir("probe"), inputs)
            return self._finish("probe", run, inputs, self.route.probe(run))

        return self.locked("probe", body)

    def diagnose(self, attempt: str) -> int:
        def body() -> int:
            config = self.agent_config()
            if not config.get("enabled"):
                raise Refused("the agent is disabled in project.json")
            path = Path(attempt).resolve()
            try:
                path.relative_to(self.workdir)
            except ValueError as error:
                raise Refused("the attempt must be inside this project's workdir") from error
            inputs = {"stage": "diagnose", "attempt": str(path), "utc": utc_now()}
            run = new_run(self.stage_dir("diagnose"), inputs)
            job = agent.diagnose_job(self.route.diagnose_evidence(path))
            record = agent.run_job(config, job, run / "agent", run / "candidates")
            write_new_json(run / "job.json", record)
            outputs = ["job.json"] + (["candidates/diagnosis.json"] if record["status"] == "PASSED" else [])
            return self._finish("diagnose", run, inputs,
                                outcome(record["status"], record["reasons"], outputs,
                                        ["a diagnosis is advisory and changes nothing"]))

        return self.locked("diagnose", body)

    def status(self) -> int:
        self.log(f"project {self.project.path}  route {self.project.route}  workdir {self.workdir}")
        for stage in ("intake", "review", "build", "qualify", "verify", "export"):
            try:
                inputs = self.intake_inputs() if stage == "intake" else (
                    self.inputs_for_export(self.accepted_verify(True)[1]) if stage == "export"
                    else self.inputs_for(stage))
            except (Refused, OSError, KeyError, ValueError) as error:
                self.log(f"  {stage:8} blocked: {error}")
                continue
            run, receipt, problem = self._receipt_state(stage, inputs)
            if stage == "export":
                receipts = [load_receipt(r)[0] for r in runs(self.stage_dir("export"))]
                receipts = [r for r in receipts if r and (r.get("inputs") or {}).get("verify") == inputs["verify"]]
                receipt, problem = (receipts[-1], None) if receipts else (None, "not run for the current inputs")
            self.log(f"  {stage:8} {problem or receipt['status']}" + (f"  ({run.name})" if run and not problem else ""))
        try:
            namespace = self.namespace()
            records = attempt_records(namespace)
            self.log(f"  collect  {len(records)} finished attempt(s) in {namespace.name}")
        except (Refused, OSError, KeyError, ValueError) as error:
            self.log(f"  collect  blocked: {error}")
        return 0

    # ------------------------------------------------------------------------- export
    def _export(self, run: Path, verify_run: Path, allow_partial: bool) -> dict:
        audit = read_json(verify_run / "AUDIT.json")
        build = self._build_receipt()
        targets = self.route.targets(build)
        reasons: list = []
        episodes = [e for e in audit["episodes"] if e["status"] == "PASS"]
        policy = (self.project.data.get("output") or {}).get("split_policy")
        groups, split_problems = assign_groups(episodes, policy, targets)
        reasons += split_problems
        if self.route.synthetic:
            reasons.append(f"route {self.route.name} produces synthetic fixtures; they are never exported as data")
            episodes = []
        if self.project.synthetic:
            reasons.append("the project declares fixture.synthetic: true; its episodes are never "
                           "exported as data")
            episodes = []
        synthetic = [e["index"] for e in episodes if is_synthetic(Path(e["episode_dir"]))]
        if synthetic:
            reasons.append(f"episodes {synthetic[:10]} are marked as synthetic test fixtures; nothing is exported")
            episodes = []
        dataset = run / "dataset"
        exported, checks_out = [], {}
        for e in episodes:
            group = groups.get(e["index"])
            if group is None:
                continue
            target = dataset / group / f"episode{e['index'] + 1}"
            try:
                listing = link_tree(Path(e["episode_dir"]), target)
            except OSError as error:
                write_new_json(run / "CHECKS.json", checks_out)
                write_new_json(run / "EXPORT_MANIFEST.json", {"schema": EXPORT_SCHEMA, "status": "FAILED",
                                                              "error": repr(error)})
                return outcome("FAILED", [f"could not hard-link {e['episode_dir']}: {error!r} (keep the "
                                          "workdir and the collected episodes on one filesystem)"],
                               ["EXPORT_MANIFEST.json", "CHECKS.json"])
            report = checks.raw_episode_report(target)
            checks_out[f"{group}/{target.name}"] = report
            manifest = read_json(target / "manifest.json")
            exported.append({"index": e["index"], "group": group, "split": e.get("split"),
                             "layout_sha256": e.get("layout_sha256"), "trace_sha256": e.get("trace_sha256"),
                             "input_sha256": manifest.get("input_sha256"), "path": f"dataset/{group}/{target.name}",
                             "files": len(listing), "bytes": sum(listing.values()),
                             "listing_sha256": canonical_sha256(listing),
                             "raw_schema_valid": report["raw_schema_valid"],
                             "conversion_checks_ok": report["conversion_checks_ok"]})
        write_new_json(run / "CHECKS.json", checks_out)
        complete = sorted(x["index"] for x in exported) == sorted(targets)
        if not complete:
            reasons.append(f"{len(exported)} of {len(targets)} target episodes exported"
                           + ("" if allow_partial else " (incomplete)"))
        raw_ok = bool(exported) and all(x["raw_schema_valid"] for x in exported)
        conversion_ok = raw_ok and all(x["conversion_checks_ok"] for x in exported) and not split_problems \
            and (complete or allow_partial) and not self.route.synthetic
        for key, report in checks_out.items():
            for cat, res in report["categories"].items():
                if not res["ok"]:
                    reasons.append(f"{key}: {cat}: {res['problems'][0]}")
        status = "CONVERSION_READY" if conversion_ok else "RAW_SCHEMA_VALID" if raw_ok else "HELD"
        if status == "HELD" and not reasons:
            reasons.append("no verified episode to export")
        if self.route.synthetic or synthetic or self.project.synthetic:
            status = "HELD"
        counts: dict = {}
        for x in exported:
            counts[x["group"]] = counts.get(x["group"], 0) + 1
        manifest = {
            "schema": EXPORT_SCHEMA, "status": status, "route": self.route.name, "project": self.project.data["name"],
            "instruction": self.project.data["instruction"], "utc": utc_now(),
            "levels": {"RAW_SCHEMA_VALID": "every exported episode passed the schema and numeric checks",
                       "CONVERSION_READY": "also causal, label and recording checks, a valid split policy and the "
                                           "full target count (or --allow-partial)",
                       "TRAINING_ADMITTED": "only recorded by `admit` with evidence from your corpus build"},
            "training_admission": "NOT_ESTABLISHED", "complete": complete, "allow_partial": allow_partial,
            "targets": len(targets), "counts": counts, "episodes": exported, "split_policy": policy,
            "raw_format": checks.RAW_FORMAT if self.route.name == "sapien" else checks.PIKA_FORMAT,
            "conversion_settings": self.route.conversion_settings(), "disclosures": self.route.disclosures(),
            "not_measured": NOT_MEASURED, "frames": "listed by path and size; pixel bytes are not hashed"}
        write_new_json(run / "EXPORT_MANIFEST.json", manifest)
        return outcome(status, reasons[:80], ["EXPORT_MANIFEST.json", "CHECKS.json"],
                       ["training admission (normalizer fit and corpus checks run by your own tooling)"],
                       counts=counts, complete=complete)


def assign_groups(episodes: list, policy: Any, targets: list) -> tuple:
    """Map every episode index to exactly one split group of the declared policy."""
    if not isinstance(policy, dict) or not isinstance(policy.get("groups"), dict) or not policy["groups"]:
        return {}, ["output.split_policy.groups is not declared"]
    problems, groups = [], {}
    limit = max(targets) + 1 if targets else 1
    for name, selector in policy["groups"].items():
        if not specmod.NAME.fullmatch(str(name)):
            problems.append(f"split group name {name!r} must be a lowercase identifier")
            continue
        if "indices" in selector:
            try:
                chosen = set(specmod.parse_indices(selector["indices"], limit))
            except ValueError as error:
                problems.append(f"group {name}: {error}")
                continue
            members = [e["index"] for e in episodes if e["index"] in chosen]
        elif "splits" in selector:
            members = [e["index"] for e in episodes if e.get("split") in set(selector["splits"])]
        else:
            problems.append(f"group {name}: use 'indices' or 'splits'")
            continue
        for index in members:
            if index in groups:
                problems.append(f"episode index {index} is in groups {groups[index]} and {name}")
            groups[index] = name
    missing = sorted(e["index"] for e in episodes if e["index"] not in groups)
    if missing:
        problems.append(f"episodes {missing[:10]} are not assigned to a split group")
    layouts: dict = {}
    for e in episodes:
        layouts.setdefault(e.get("layout_sha256"), []).append(e["index"])
    shared = [v for v in layouts.values() if len(v) > 1]
    if shared:
        problems.append(f"layouts shared by several exported episodes: {shared[:5]}")
    return groups, problems


def is_synthetic(episode: Path) -> bool:
    """Episodes written by test fixtures declare themselves; they are never exported as data."""
    try:
        manifest = read_json(episode / "manifest.json")
    except (OSError, ValueError):
        return False
    return manifest.get("domain") != "sim" or bool((manifest.get("simulation") or {}).get("synthetic_fixture"))


def attempt_records(namespace: Path) -> dict:
    """{attempt id: RESULT.json record} for finished attempts in a collection namespace."""
    out = {}
    base = namespace / "attempts"
    for attempt in sorted(base.iterdir()) if base.is_dir() else []:
        if attempt.is_dir() and (attempt / "RESULT.json").is_file():
            out[attempt.name] = file_record(attempt / "RESULT.json")
    return out


def attempts_used(namespace: Path, index: int) -> int:
    """Attempts already spent on an episode, interrupted ones included."""
    base, prefix = namespace / "attempts", f"ep{index:05d}_a"
    return sum(1 for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)) if base.is_dir() else 0


def close_incomplete_attempts(namespace: Path) -> list:
    """Attempts that ended without RESULT.json get an INCOMPLETE result in place: their files stay as
    evidence, they fail verification and they count toward the attempt budget (lock held)."""
    closed, base = [], namespace / "attempts"
    for attempt in sorted(base.iterdir()) if base.is_dir() else []:
        if attempt.is_dir() and not (attempt / "RESULT.json").exists():
            write_new_json(attempt / "RESULT.json", {
                "kind": "attempt_result", "status": "INCOMPLETE", "returncode": None, "timed_out": None,
                "finished_utc": utc_now(), "note": "the attempt ended without a result (interrupted); its files are "
                                                   "kept as evidence and it is not accepted"})
            closed.append(attempt.name)
    return closed


def latest_candidate(plan_dir: Path, name: str) -> Path | None:
    """Newest candidate from a plan run that did not fail (HELD candidates may still lack values)."""
    for run in reversed(runs(plan_dir)):
        receipt, problems = load_receipt(run)
        if receipt and not problems and receipt["status"] in ("PASSED", "HELD") and \
                (run / "candidates" / name).is_file():
            return run / "candidates" / name
    return None


def episode_status(entries: list) -> str:
    """Counting rule: one PASS attempt, and every other attempt of that episode finished as a failure."""
    passed = [e for e in entries if e["status"] == "PASS"]
    if len(passed) == 1 and all(e["status"] in ("PASS", "FAIL") for e in entries):
        return "PASS"
    if len(passed) > 1:
        return "CONFLICT"
    return "FAIL" if all(e["status"] == "FAIL" for e in entries) else "HOLD"


def write_audit(run: Path, entries: list, targets: list, problems: list, extra: dict | None = None) -> dict:
    """Group attempt checks per episode, write AUDIT.json and return the verify outcome."""
    by_index: dict = {}
    problems = list(problems)
    for e in entries:
        if not isinstance(e.get("index"), int):
            problems.append(f"attempt {e.get('attempt_id')} cannot be attributed to an episode")
            continue
        by_index.setdefault(e["index"], []).append(e)
    episodes = []
    for index in sorted(by_index):
        status = episode_status(by_index[index])
        chosen = next((e for e in by_index[index] if e["status"] == "PASS"), by_index[index][-1])
        episodes.append({**{k: chosen.get(k) for k in ("index", "split", "layout_sha256", "trace_sha256",
                                                     "episode_dir", "attempt")},
                         "status": status, "attempts": [{k: e.get(k) for k in ("attempt", "status", "failed")}
                                                        for e in by_index[index]]})
    passed = sorted(e["index"] for e in episodes if e["status"] == "PASS")
    audit = {"schema": "neo-vla-real2sim-audit/1", "utc": utc_now(), "targets": targets, "passed": passed,
             "episodes": episodes, "attempt_checks": entries, "problems": problems, **(extra or {})}
    write_new_json(run / "AUDIT.json", audit)
    missing = sorted(set(targets) - set(passed))
    reasons = list(problems) + ([f"{len(missing)} target episode(s) without a passing attempt: {missing[:12]}"]
                                if missing else [])
    status = "FAILED" if problems or not passed else "PASSED" if not missing else "PARTIAL"
    return outcome(status, reasons, ["AUDIT.json"], ["training readiness (see export)"],
                   passed=len(passed), targets=len(targets))
