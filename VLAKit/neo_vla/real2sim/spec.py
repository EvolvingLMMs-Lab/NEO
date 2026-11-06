"""Project, scene and task specifications: validation, provenance, requests and review items.

Every physical quantity is written as {"value", "unit", "provenance", "source"}. Geometry and
calibration must be measured or derived from measurements; images alone never count as measured
geometry, and dynamic properties are never taken from images. Missing or unusable inputs become
structured requests naming the stages they block; estimates and design choices become review items
that a person must decide before a build.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .store import PathError, canonical_sha256, contained, ensure_inside, file_record

PROJECT_SCHEMA = "neo-vla-real2sim-project/1"
SCENE_SCHEMA = "neo-vla-real2sim-scene/1"
TASK_SCHEMA = "neo-vla-real2sim-task/1"
REVIEW_SCHEMA = "neo-vla-real2sim-review/1"
PROVENANCE = ("measured", "derived", "estimate", "chosen", "unverified", "inferred_from_image")
FACTUAL = ("measured", "derived")
DYNAMIC = ("measured", "derived", "estimate")
NEEDS_REVIEW = ("estimate", "chosen", "unverified", "inferred_from_image")
IMAGE_USES = ("appearance_reference", "layout_reference", "scene_context")
ROUTES = ("sapien", "can_in_tape_reference")
EFFORTS = ("low", "medium", "high", "xhigh", "max")
MAX_ATTEMPTS = 3
CONTROLLERS = ("joint_waypoints", "pick_place", "plugin")
CONDITIONS = {
    "position_in_box": ("object", "center_m", "half_extents_m"),
    "axis_tilt_max": ("object", "axis", "max_deg"),
    "contact": ("object", "with"),
    "no_contact": ("object", "with"),
    "speed_max": ("object", "linear_m_s", "angular_rad_s"),
    "gripper_open_min": ("aperture_m",),
    "displacement_max": ("object", "max_m"),
    "phase_in": ("phases",),
}
RESERVED_NAMES = ("table", "robot", "world")
NAME = re.compile(r"[a-z][a-z0-9_]{0,47}")
AGENT_KEYS = {"enabled", "executable", "model", "effort", "timeout_s", "max_attempts"}
LAYOUT_SCHEME = "neo-vla-real2sim-layout/1"
MAX_DRAWS = 2000
MAX_JSON_BYTES = 5 << 20


class Issues:
    """Errors (invalid input), requests (missing input) and review items (needs a decision)."""

    def __init__(self) -> None:
        self.errors: list = []
        self.requests: list = []
        self.review: list = []

    def error(self, field: str, problem: str) -> None:
        self.errors.append({"field": field, "problem": problem})

    def request(self, field: str, problem: str, blocks: Sequence[str], how: str) -> None:
        self.requests.append({"id": f"request-{len(self.requests) + 1}", "field": field, "problem": problem,
                              "blocks": list(blocks), "how_to_supply": how})

    def item(self, field: str, kind: str, detail: Any) -> None:
        body = {"field": field, "kind": kind, "detail": detail}
        self.review.append({"id": canonical_sha256(body)[:12], **body})

    def blocked(self) -> list:
        return sorted({b for r in self.requests for b in r["blocks"]} | ({"all"} if self.errors else set()))

    def as_dict(self) -> dict:
        return {"errors": self.errors, "requests": self.requests, "review_items": self.review,
                "blocked_stages": self.blocked()}


# ----------------------------------------------------------------------------- small helpers
def _number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _shape_ok(value: Any, shape: Any) -> bool:
    if shape is None:
        return _number(value)
    if isinstance(shape, int):
        return isinstance(value, list) and len(value) == shape and all(_number(v) for v in value)
    rows, cols = shape
    return (isinstance(value, list) and len(value) == rows
            and all(isinstance(r, list) and len(r) == cols and all(_number(v) for v in r) for r in value))


def _flat(value: Any) -> list:
    return [float(x) for x in (sum(value, []) if value and isinstance(value[0], list) else value)] \
        if isinstance(value, list) else [float(value)]


def is_rigid(t: Any, tol: float = 1e-6) -> bool:
    if not _shape_ok(t, (4, 4)):
        return False
    if max(abs(t[3][j] - (0.0, 0.0, 0.0, 1.0)[j]) for j in range(4)) > tol:
        return False
    r = [[float(t[i][j]) for j in range(3)] for i in range(3)]
    for i in range(3):
        for j in range(3):
            if abs(sum(r[k][i] * r[k][j] for k in range(3)) - (1.0 if i == j else 0.0)) > tol:
                return False
    det = (r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
           + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]))
    return abs(det - 1.0) <= tol


def plain(value: Any) -> Any:
    """Strip quantity wrappers, keeping values only (for the simulator)."""
    if isinstance(value, dict):
        if "value" in value and "provenance" in value:
            return plain(value["value"])
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [plain(v) for v in value]
    return value


def _json_path(data: Any, path: str) -> Any:
    for part in path.split("."):
        if isinstance(data, list) and part.isdigit() and int(part) < len(data):
            data = data[int(part)]
        elif isinstance(data, dict) and part in data:
            data = data[part]
        else:
            raise KeyError(path)
    return data


class Context:
    """What quantity checks need to know about the project (measurements, calibration files)."""

    def __init__(self, root: Path, measurements: Mapping[str, dict], calibration: Mapping[str, Path]) -> None:
        self.root, self.measurements, self.calibration = root, measurements, calibration


def quantity(issues: Issues, ctx: Context, q: Any, field: str, *, shape: Any = None,
             need: Sequence[str] | None = FACTUAL, positive: bool = False, blocks: Sequence[str] = ("build",),
             dynamic: bool = False) -> Any:
    how = (f"set {field} to {{\"value\": ..., \"unit\": ..., \"provenance\": one of {list(need or PROVENANCE)}, "
           "\"source\": ...}")
    if q is None:
        issues.request(field, "missing", blocks, how)
        return None
    if not isinstance(q, dict) or "value" not in q or "provenance" not in q:
        issues.error(field, "expected a quantity object with value, unit, provenance and source")
        return None
    value, prov, source = q["value"], q["provenance"], q.get("source")
    if prov not in PROVENANCE:
        issues.error(field, f"provenance {prov!r} not in {list(PROVENANCE)}")
        return None
    if value is None:
        issues.request(field, "value not supplied yet", blocks, how)
        return None
    if not _shape_ok(value, shape):
        issues.error(field, f"value must be {'a finite number' if shape is None else f'finite numbers shaped {shape}'}")
        return None
    if positive and any(v <= 0.0 for v in _flat(value)):
        issues.error(field, "value must be positive")
        return None
    if dynamic and prov == "inferred_from_image":
        issues.request(field, "dynamic properties cannot be inferred from images", blocks,
                       "weigh or measure it, or declare a reviewed estimate")
        return None
    if need is not None and prov not in need:
        problem = ("images alone are not measured geometry" if prov == "inferred_from_image"
                   else f"provenance {prov!r} where {list(need)} is required")
        issues.request(field, problem, blocks, "measure it, or derive it from measurements, and cite the source")
        return None
    if prov in FACTUAL and not source:
        issues.error(field, "measured or derived values must name their source")
    if isinstance(source, str) and source.startswith("measurement:"):
        m = ctx.measurements.get(source.split(":", 1)[1])
        if m is None:
            issues.error(field, f"source {source} is not a declared measurement")
        elif _flat(m["value"]) != _flat(value) or m.get("unit") != q.get("unit"):
            issues.error(field, f"value/unit differ from {source} ({m['value']} {m.get('unit')})")
    elif isinstance(source, str) and source.startswith("calibration:"):
        ref, _, path = source.split(":", 1)[1].partition("#")
        file = ctx.calibration.get(ref)
        if file is None:
            issues.error(field, f"source {source} is not a declared calibration file")
        else:
            try:
                if file.stat().st_size > MAX_JSON_BYTES or not path:
                    raise ValueError("not machine-readable")
                got = _json_path(json.loads(file.read_text()), path)
                if _flat(got) != _flat(value):
                    issues.error(field, f"value differs from {source}")
            except (ValueError, KeyError, TypeError, OSError):
                issues.item(field, "transcribed_not_machine_checked", {"value": value, "source": source})
    if prov in NEEDS_REVIEW:
        issues.item(field, prov, {"value": value, "unit": q.get("unit"), "source": source})
    return value


def _file(issues: Issues, root: Path, rel: Any, field: str, blocks: Sequence[str], how: str,
          records: dict) -> Path | None:
    if not isinstance(rel, str) or not rel:
        issues.request(field, "path missing", blocks, how)
        return None
    try:
        path = contained(root, rel)
    except PathError as error:
        issues.error(field, str(error))
        return None
    if not path.is_file():
        issues.request(field, f"{rel} does not exist", blocks, how)
        return None
    records[rel] = file_record(path)
    return path


# ----------------------------------------------------------------------------- project
class Project:
    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        self.root = self.path.parent
        self.issues = Issues()
        self.files: dict = {}
        self.data: dict = {}
        self.ctx = Context(self.root, {}, {})
        try:
            raw = self.path.read_bytes()
            self.files["project.json"] = file_record(self.path)
            self.data = json.loads(raw)
        except (OSError, ValueError) as error:
            self.issues.error("project", f"cannot read {self.path}: {error}")
            return
        self._validate()

    @property
    def route(self) -> str:
        return self.data.get("route", "")

    @property
    def synthetic(self) -> bool:
        """True only when project.json declares fixture.synthetic: true (validated as a boolean)."""
        fixture = self.data.get("fixture")
        return isinstance(fixture, dict) and fixture.get("synthetic") is True

    def spec_path(self, key: str) -> Path | None:
        rel = (self.data.get("specs") or {}).get(key)
        if not isinstance(rel, str):
            return None
        try:
            return contained(self.root, rel)
        except PathError:
            return None

    def _validate(self) -> None:
        d, issues, root = self.data, self.issues, self.root
        if d.get("schema") != PROJECT_SCHEMA:
            issues.error("schema", f"expected {PROJECT_SCHEMA}")
        if not isinstance(d.get("name"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", d["name"]):
            issues.error("name", "use letters, digits, '.', '_' or '-' (at most 64 characters)")
        if d.get("route") not in ROUTES:
            issues.error("route", f"expected one of {list(ROUTES)}")
        if not isinstance(d.get("instruction"), str) or not 0 < len(d["instruction"].strip()) <= 500:
            issues.error("instruction", "a non-empty task instruction of at most 500 characters")
        fixture = d.get("fixture", {"synthetic": False})
        if not isinstance(fixture, dict) or not isinstance(fixture.get("synthetic"), bool):
            issues.error("fixture.synthetic", "true or false: whether this project is a synthetic "
                         "test fixture, whose episodes are marked as such and never exported")
        inputs = d.get("inputs") or {}
        for i, image in enumerate(inputs.get("images") or []):
            field = f"inputs.images[{i}]"
            if image.get("use") not in IMAGE_USES:
                issues.error(field, f"use must be one of {list(IMAGE_USES)}; images are not measurements")
            if not isinstance(image.get("shows"), str) or not image["shows"].strip():
                issues.error(field, "describe what the image shows")
            _file(issues, root, image.get("path"), field + ".path", ("plan",), "add the image file", self.files)
        for i, m in enumerate(inputs.get("measurements") or []):
            field = f"inputs.measurements[{i}]"
            if not isinstance(m.get("id"), str) or not NAME.fullmatch(m["id"]) or m["id"] in self.ctx.measurements:
                issues.error(field, "id must be a unique lowercase identifier")
                continue
            if m.get("value") is None:
                issues.request(field + ".value", "measurement not supplied yet", ("plan", "build"),
                               "measure it and record value, unit, method, by and date")
                continue
            if not (_number(m.get("value")) or _shape_ok(m.get("value"), len(m.get("value") or [0]))):
                issues.error(field, "value must be a number or a list of numbers")
                continue
            if m.get("provenance") not in FACTUAL:
                issues.error(field, "measurements are measured or derived; estimates belong in the specs")
                continue
            method = str(m.get("method", ""))
            if not method or not m.get("unit") or not m.get("by") or not m.get("date"):
                issues.error(field, "measurements need unit, method, by and date")
            if re.search(r"image|photo|picture", method, re.I) and (
                    m.get("provenance") == "measured" or not m.get("scale_reference")):
                issues.error(field, "an image-based measurement must be 'derived' and name its scale_reference")
            self.ctx.measurements[m["id"]] = m
        for i, c in enumerate(inputs.get("calibration") or []):
            field = f"inputs.calibration[{i}]"
            if not isinstance(c.get("id"), str) or not NAME.fullmatch(c["id"]) or c["id"] in self.ctx.calibration:
                issues.error(field, "id must be a unique lowercase identifier")
                continue
            if not c.get("kind") or not c.get("method"):
                issues.error(field, "calibration files need kind and method")
            path = _file(issues, root, c.get("path"), field + ".path", ("build",), "add the calibration file",
                         self.files)
            if path is not None:
                self.ctx.calibration[c["id"]] = path
        for i, a in enumerate(inputs.get("assets") or []):
            field = f"inputs.assets[{i}]"
            if a.get("role") not in ("collision", "visual", "texture", "robot"):
                issues.error(field, "role must be collision, visual, texture or robot")
            if not a.get("source") or not a.get("license"):
                issues.request(field, "asset source and license are required", ("build",),
                               "record where the asset comes from and the terms it is used under")
            _file(issues, root, a.get("path"), field + ".path", ("build",), "add the asset file", self.files)
        collection = d.get("collection") or {}
        for key, low, high in (("seed", 0, 2 ** 31), ("episodes", 1, 100000), ("qualify_episodes", 1, 20),
                               ("max_attempts_per_episode", 1, MAX_ATTEMPTS), ("workers", 1, 256)):
            v = collection.get(key)
            if not isinstance(v, int) or isinstance(v, bool) or not low <= v <= high:
                issues.error(f"collection.{key}", f"an integer in [{low}, {high}]")
        timeout = collection.get("attempt_timeout_s")
        if not _number(timeout) or timeout <= 0:
            issues.error("collection.attempt_timeout_s", "set an explicit per-attempt deadline in seconds")
        if not isinstance((d.get("output") or {}).get("split_policy"), dict):
            issues.request("output.split_policy", "no split policy declared", ("export",),
                           "declare which episodes form each split group (see the guide)")
        self._validate_agent(d.get("agent"))
        if d.get("route") == "sapien":
            self._validate_sapien(d)

    def _validate_agent(self, agent: Any) -> None:
        issues = self.issues
        if not isinstance(agent, dict):
            issues.error("agent", "an agent section is required (set enabled: false to work without one)")
            return
        unknown = set(agent) - AGENT_KEYS
        if unknown:
            issues.error("agent", f"unsupported keys {sorted(unknown)}; allowed: {sorted(AGENT_KEYS)}")
        if not isinstance(agent.get("enabled"), bool):
            issues.error("agent.enabled", "true or false")
        if not agent.get("enabled"):
            return
        if not isinstance(agent.get("executable"), str) or not agent["executable"]:
            issues.error("agent.executable", "the Claude Code CLI executable, for example 'claude'")
        if agent.get("model") is None:
            issues.request("agent.model", "no model pinned", ("plan", "diagnose"),
                           "set the full model id you want the jobs to use (no fallback is attempted)")
        elif not isinstance(agent["model"], str) or not re.fullmatch(r"[A-Za-z0-9._:\[\]-]{2,80}", agent["model"]):
            issues.error("agent.model", "a full model id such as the one `claude --model` accepts")
        if agent.get("effort") is not None and agent["effort"] not in EFFORTS:
            issues.error("agent.effort", f"one of {list(EFFORTS)} or null")
        if not _number(agent.get("timeout_s")) or agent["timeout_s"] <= 0:
            issues.error("agent.timeout_s", "an explicit per-job deadline in seconds")
        if not isinstance(agent.get("max_attempts"), int) or not 1 <= agent["max_attempts"] <= MAX_ATTEMPTS:
            issues.error("agent.max_attempts", f"an integer in [1, {MAX_ATTEMPTS}]")

    def _validate_sapien(self, d: dict) -> None:
        issues, root = self.issues, self.root
        robot = (d.get("inputs") or {}).get("robot") or {}
        _file(issues, root, robot.get("urdf"), "inputs.robot.urdf", ("plan", "build"), "add your robot URDF",
              self.files)
        if not robot.get("source") or not robot.get("license"):
            issues.request("inputs.robot", "robot description source and license are required", ("build",),
                           "record where the URDF and meshes come from and their terms")
        specs = d.get("specs") or {}
        for key in ("scene", "task"):
            rel = specs.get(key)
            if not isinstance(rel, str):
                issues.error(f"specs.{key}", "declare the path of the reviewed specification")
                continue
            path = self.spec_path(key)
            if path is None:
                issues.error(f"specs.{key}", "must be a plain relative path inside the project")
            elif path.is_file():
                self.files[rel] = file_record(path)
            else:
                issues.request(f"specs.{key}", f"{rel} does not exist yet", ("review", "build"),
                               "run `plan` and promote a candidate, or write it yourself")
        if specs.get("plugin") is not None:
            path = self.spec_path("plugin")
            if path is None:
                issues.error("specs.plugin", "must be a plain relative path inside the project")
            elif path.is_file():
                self.files[specs["plugin"]] = file_record(path)
        sim = d.get("simulator") or {}
        python = sim.get("python")
        if not isinstance(python, str) or not os.path.isabs(python) or not os.access(python, os.X_OK):
            issues.request("simulator.python", "no executable simulator interpreter", ("build",),
                           "absolute path of the Python that has sapien, numpy, Pillow and six "
                           "installed")
        if sim.get("renderer") not in ("cpu", "gpu"):
            issues.error("simulator.renderer", "cpu (lavapipe) or gpu")
        env = sim.get("env") or {}
        if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            issues.error("simulator.env", "a mapping of environment variable names to strings")
        for i, lib in enumerate(sim.get("preload_libraries") or []):
            if not isinstance(lib, str) or not os.path.isabs(lib) or not os.path.isfile(lib):
                issues.request(f"simulator.preload_libraries[{i}]", "library not found", ("build",),
                               "absolute path of an existing shared library, or remove the entry")
        if sim.get("renderer") == "cpu" and not os.path.isfile(str((env or {}).get("VK_ICD_FILENAMES", ""))):
            issues.request("simulator.env.VK_ICD_FILENAMES", "CPU rendering needs a lavapipe ICD file", ("build",),
                           "path of lvp_icd.x86_64.json from your Mesa lavapipe installation")

    def summary(self) -> dict:
        return {"path": str(self.path), "route": self.route, "files": self.files, **self.issues.as_dict()}


# ----------------------------------------------------------------------------- URDF
def inspect_urdf(urdf: Path, root: Path) -> dict:
    """Links, joints with limits and the mesh files of a URDF (relative mesh paths only)."""
    tree = ET.parse(str(urdf))
    robot = tree.getroot()
    if robot.tag != "robot":
        raise ValueError(f"{urdf.name}: root element is not <robot>")
    links = [link.get("name") for link in robot.findall("link")]
    joints = {}
    for j in robot.findall("joint"):
        limit, parent, child = j.find("limit"), j.find("parent"), j.find("child")
        entry = {"type": j.get("type"), "parent": parent.get("link") if parent is not None else None,
                 "child": child.get("link") if child is not None else None, "mimic": j.find("mimic") is not None}
        if limit is not None:
            for key in ("lower", "upper", "velocity", "effort"):
                if limit.get(key) is not None:
                    entry[key] = float(limit.get(key))
        joints[j.get("name")] = entry
    meshes = {}
    for mesh in robot.iter("mesh"):
        name = mesh.get("filename", "")
        if "://" in name or os.path.isabs(name):
            raise ValueError(f"mesh {name!r}: use a path relative to the URDF file (no package:// or absolute paths)")
        path = ensure_inside(urdf.parent / name, root)
        if not path.is_file():
            raise ValueError(f"mesh {name!r} not found")
        meshes[path.relative_to(root.resolve()).as_posix()] = file_record(path)
    return {"links": links, "joints": joints, "meshes": meshes, "urdf": file_record(urdf)}


# ----------------------------------------------------------------------------- scene and task
def _rigid_q(issues: Issues, ctx: Context, q: Any, field: str, need: Sequence[str] | None = FACTUAL) -> Any:
    value = quantity(issues, ctx, q, field, shape=(4, 4), need=need)
    if value is not None and not is_rigid(value):
        issues.error(field, "not a proper rigid transform (orthonormal rotation, last row 0 0 0 1)")
        return None
    return value


def footprint_radius(obj: Mapping[str, Any]) -> float | None:
    s = plain(obj.get("shape") or {})
    try:
        if s.get("type") == "box":
            return 0.5 * math.sqrt(sum(float(v) ** 2 for v in s["size_m"]))
        if s.get("type") == "cylinder":
            return math.hypot(float(s["radius_m"]), float(s["length_m"]) / 2.0)
        if s.get("type") == "sphere":
            return float(s["radius_m"])
        if s.get("type") == "mesh":
            return float(s["footprint_radius_m"])
    except (KeyError, TypeError, ValueError):
        return None
    return None


def validate_scene(scene: Any, project: Project, urdf: dict | None, issues: Issues) -> None:
    ctx, root = project.ctx, project.root
    if not isinstance(scene, dict) or scene.get("schema") != SCENE_SCHEMA:
        issues.error("scene", f"expected a {SCENE_SCHEMA} object")
        return
    physics = scene.get("physics") or {}
    hz = physics.get("physics_hz")
    if not isinstance(hz, int) or hz < 30 or hz % 30:
        issues.error("scene.physics.physics_hz", "an integer multiple of 30 (the recording rate)")
    if physics.get("control_hz") != 30:
        issues.error("scene.physics.control_hz", "must be 30 (raw episode format)")
    if not _shape_ok(physics.get("gravity_m_s2"), 3):
        issues.error("scene.physics.gravity_m_s2", "three numbers")
    table = scene.get("table")
    if not isinstance(table, dict):
        issues.request("scene.table", "a support surface is required", ("build",), "describe the measured table")
    else:
        quantity(issues, ctx, table.get("size_m"), "scene.table.size_m", shape=3, positive=True)
        quantity(issues, ctx, table.get("top_z_m"), "scene.table.top_z_m")
        quantity(issues, ctx, table.get("center_xy_m"), "scene.table.center_xy_m", shape=2)
        quantity(issues, ctx, table.get("yaw_rad"), "scene.table.yaw_rad")
        quantity(issues, ctx, table.get("friction"), "scene.table.friction", need=DYNAMIC, dynamic=True)
    robot = scene.get("robot") or {}
    rel = (project.data.get("inputs") or {}).get("robot", {}).get("urdf")
    if robot.get("urdf") != rel:
        issues.error("scene.robot.urdf", "must equal inputs.robot.urdf")
    _rigid_q(issues, ctx, robot.get("base_pose_W"), "scene.robot.base_pose_W")
    arm = robot.get("arm_joints")
    if not isinstance(arm, list) or not arm or len(set(arm)) != len(arm) or not all(isinstance(a, str) for a in arm):
        issues.error("scene.robot.arm_joints", "a non-empty list of distinct joint names")
        arm = []
    quantity(issues, ctx, robot.get("init_arm_q"), "scene.robot.init_arm_q", shape=len(arm) or None, need=None)
    _rigid_q(issues, ctx, robot.get("tcp_in_ee"), "scene.robot.tcp_in_ee")
    grip = robot.get("gripper") or {}
    gjoints = grip.get("joints")
    if not isinstance(gjoints, list) or not gjoints or not all(
            isinstance(g, dict) and isinstance(g.get("name"), str) and _number(g.get("scale")) and g["scale"] != 0
            for g in gjoints):
        issues.error("scene.robot.gripper.joints", "a list of {name, scale}: aperture = offset + sum(scale * q)")
        gjoints = []
    quantity(issues, ctx, grip.get("aperture_offset_m"), "scene.robot.gripper.aperture_offset_m")
    stroke = quantity(issues, ctx, grip.get("stroke_m"), "scene.robot.gripper.stroke_m", positive=True)
    init_ap = quantity(issues, ctx, grip.get("init_aperture_m"), "scene.robot.gripper.init_aperture_m", need=None)
    if stroke is not None and init_ap is not None and not 0.0 <= init_ap <= stroke:
        issues.error("scene.robot.gripper.init_aperture_m", "must lie within [0, stroke]")
    drive = robot.get("drive") or {}
    for group in ("arm", "gripper"):
        for key in ("stiffness", "damping", "force_limit"):
            quantity(issues, ctx, (drive.get(group) or {}).get(key), f"scene.robot.drive.{group}.{key}",
                     need=DYNAMIC, positive=True, dynamic=True)
    if not isinstance(drive.get("gravity_compensation"), bool):
        issues.error("scene.robot.drive.gravity_compensation", "true or false")
    quantity(issues, ctx, drive.get("max_joint_speed_rad_s"), "scene.robot.drive.max_joint_speed_rad_s",
             need=None, positive=True)
    follower = robot.get("follower")
    if follower is not None:
        alpha = quantity(issues, ctx, follower.get("alpha"), "scene.robot.follower.alpha", need=DYNAMIC, dynamic=True)
        if alpha is not None and not 0.0 < alpha <= 1.0:
            issues.error("scene.robot.follower.alpha", "in (0, 1]")
        if not isinstance(follower.get("delay_steps"), int) or follower["delay_steps"] < 0:
            issues.error("scene.robot.follower.delay_steps", "a non-negative integer")
    if urdf is not None:
        joints, links = urdf["joints"], urdf["links"]
        movable = {n for n, j in joints.items() if j["type"] not in ("fixed", None)}
        for name in arm:
            j = joints.get(name)
            if j is None or j["type"] not in ("revolute", "prismatic"):
                issues.error("scene.robot.arm_joints", f"{name} is not a revolute or prismatic URDF joint")
            elif "lower" not in j or "upper" not in j:
                issues.error("scene.robot.arm_joints", f"{name} has no position limits in the URDF")
        for g in gjoints:
            j = joints.get(g["name"])
            if j is None or j["type"] != "prismatic":
                issues.error("scene.robot.gripper.joints", f"{g['name']} is not a prismatic URDF joint")
        undeclared = sorted(movable - set(arm) - {g["name"] for g in gjoints})
        if undeclared:
            issues.error("scene.robot", f"movable URDF joints not declared as arm or gripper joints: {undeclared}")
        if robot.get("ee_link") not in links:
            issues.error("scene.robot.ee_link", "not a URDF link")
    names = set()
    objects = scene.get("objects")
    if not isinstance(objects, list) or not objects:
        issues.request("scene.objects", "no task objects", ("build",), "describe each object with measured geometry")
        objects = []
    for i, o in enumerate(objects):
        field = f"scene.objects[{i}]"
        name = o.get("name")
        if not isinstance(name, str) or not NAME.fullmatch(name) or name in names or name in RESERVED_NAMES or (
                urdf is not None and name in urdf["links"]):
            issues.error(field + ".name", "unique lowercase identifier, not a URDF link, 'table', 'robot' or 'world'")
        names.add(name)
        shape = o.get("shape") or {}
        kind = shape.get("type")
        if kind == "box":
            quantity(issues, ctx, shape.get("size_m"), field + ".shape.size_m", shape=3, positive=True)
        elif kind == "cylinder":
            quantity(issues, ctx, shape.get("radius_m"), field + ".shape.radius_m", positive=True)
            quantity(issues, ctx, shape.get("length_m"), field + ".shape.length_m", positive=True)
        elif kind == "sphere":
            quantity(issues, ctx, shape.get("radius_m"), field + ".shape.radius_m", positive=True)
        elif kind == "mesh":
            _file(issues, root, shape.get("collision"), field + ".shape.collision", ("build",),
                  "add the collision mesh", project.files)
            if shape.get("visual") is not None:
                _file(issues, root, shape.get("visual"), field + ".shape.visual", ("build",),
                      "add the visual mesh", project.files)
            quantity(issues, ctx, shape.get("scale"), field + ".shape.scale", shape=3, positive=True)
            quantity(issues, ctx, shape.get("footprint_radius_m"), field + ".shape.footprint_radius_m", positive=True)
            quantity(issues, ctx, shape.get("com_m"), field + ".shape.com_m", shape=3, need=DYNAMIC, dynamic=True)
            quantity(issues, ctx, shape.get("inertia_diag_kg_m2"), field + ".shape.inertia_diag_kg_m2", shape=3,
                     need=DYNAMIC, positive=True, dynamic=True)
            if shape.get("convex", "single") not in ("single", "multiple"):
                issues.error(field + ".shape.convex", "single or multiple")
        else:
            issues.error(field + ".shape.type", "box, cylinder, sphere or mesh")
        quantity(issues, ctx, o.get("mass_kg"), field + ".mass_kg", need=DYNAMIC, positive=True, dynamic=True)
        quantity(issues, ctx, o.get("friction"), field + ".friction", need=DYNAMIC, dynamic=True)
        rest = quantity(issues, ctx, o.get("restitution"), field + ".restitution", need=DYNAMIC, dynamic=True)
        if rest is not None and not 0.0 <= rest <= 1.0:
            issues.error(field + ".restitution", "in [0, 1]")
        _rigid_q(issues, ctx, o.get("pose_W"), field + ".pose_W", need=("measured", "derived", "chosen"))
        if not isinstance(o.get("dynamic", True), bool):
            issues.error(field + ".dynamic", "true or false")
        if o.get("texture") is not None:
            _file(issues, root, o.get("texture"), field + ".texture", ("build",), "add the texture", project.files)
    cams = scene.get("cameras")
    if not isinstance(cams, list) or not cams:
        issues.request("scene.cameras", "no cameras", ("build",), "describe each calibrated camera")
        cams = []
    seen = set()
    for i, c in enumerate(cams):
        field = f"scene.cameras[{i}]"
        if not isinstance(c.get("name"), str) or not NAME.fullmatch(c["name"]) or c["name"] in seen:
            issues.error(field + ".name", "unique lowercase identifier")
        seen.add(c.get("name"))
        mount = c.get("mount")
        if mount != "world" and not (isinstance(mount, str) and mount.startswith("link:")):
            issues.error(field + ".mount", "'world' or 'link:<URDF link name>'")
        elif urdf is not None and mount != "world" and mount.split(":", 1)[1] not in urdf["links"]:
            issues.error(field + ".mount", "mount link is not a URDF link")
        _rigid_q(issues, ctx, c.get("pose"), field + ".pose")
        k = quantity(issues, ctx, c.get("K"), field + ".K", shape=(3, 3))
        if k is not None and (k[0][0] <= 0 or k[1][1] <= 0 or k[2] != [0, 0, 1] or k[1][0] or k[0][1]):
            issues.error(field + ".K", "expected [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] with fx, fy > 0")
        if c.get("D") is not None:
            d = c["D"].get("value") if isinstance(c["D"], dict) else None
            quantity(issues, ctx, c["D"], field + ".D", shape=len(d) if isinstance(d, list) else 5)
        for key in ("width", "height"):
            if not isinstance(c.get(key), int) or not 16 <= c[key] <= 8192:
                issues.error(f"{field}.{key}", "an integer number of pixels")
        if not (_number(c.get("near_m")) and _number(c.get("far_m")) and 0 < c["near_m"] < c["far_m"]):
            issues.error(field, "near_m and far_m with 0 < near < far")
        if c.get("every_n_rows", 1) not in (1, 2, 3):
            issues.error(field + ".every_n_rows", "1, 2 or 3 (older images make rows invalid)")
    rand = scene.get("randomization") or {}
    for name, r in (rand.get("objects") or {}).items():
        if name not in names:
            issues.error(f"scene.randomization.objects.{name}", "unknown object")
        for key in ("dx_m", "dy_m", "dyaw_rad"):
            rng = r.get(key, [0.0, 0.0])
            if not _shape_ok(rng, 2) or rng[0] > rng[1]:
                issues.error(f"scene.randomization.objects.{name}.{key}", "[low, high]")
    if rand:
        issues.item("scene.randomization", "chosen", rand)


def validate_task(task: Any, scene: Any, project: Project, issues: Issues) -> None:
    if not isinstance(task, dict) or task.get("schema") != TASK_SCHEMA:
        issues.error("task", f"expected a {TASK_SCHEMA} object")
        return
    scene = scene if isinstance(scene, dict) else {}
    objects = {o.get("name"): o for o in scene.get("objects") or [] if isinstance(o, dict)}
    robot = plain(scene.get("robot") or {})
    stroke = (robot.get("gripper") or {}).get("stroke_m")
    joints = len(robot.get("arm_joints") or [])
    if task.get("instruction") != project.data.get("instruction"):
        issues.error("task.instruction", "must equal the project instruction")
    episode = task.get("episode") or {}
    if not _number(episode.get("max_duration_s")) or episode["max_duration_s"] <= 0:
        issues.error("task.episode.max_duration_s", "the episode budget in seconds")
    if not _number(episode.get("settle_s")) or episode["settle_s"] < 0:
        issues.error("task.episode.settle_s", "seconds the scene settles before the controller starts")
    c = task.get("controller") or {}
    kind = c.get("type")

    def num(key: str, low: float = 0.0, strict: bool = False) -> None:
        v = c.get(key)
        if not _number(v) or v < low or (strict and v == low):
            issues.error(f"task.controller.{key}", f"a number {'>' if strict else '>='} {low}")

    if kind == "joint_waypoints":
        wps = c.get("waypoints")
        if not isinstance(wps, list) or not wps:
            issues.error("task.controller.waypoints", "a non-empty list")
        for i, wp in enumerate(wps or []):
            if not _shape_ok(wp.get("arm_q"), joints) or not _number(wp.get("duration_s")) or wp["duration_s"] <= 0:
                issues.error(f"task.controller.waypoints[{i}]", "arm_q for every arm joint and duration_s > 0")
            g = wp.get("gripper_m")
            if g is not None and (not _number(g) or stroke is None or not 0 <= g <= stroke):
                issues.error(f"task.controller.waypoints[{i}].gripper_m", "within [0, stroke]")
        num("final_hold_s")
        num("max_joint_speed_rad_s", strict=True)
    elif kind == "pick_place":
        if c.get("object") not in objects or not objects[c.get("object")].get("dynamic", True):
            issues.error("task.controller.object", "a dynamic scene object")
        if not is_rigid(c.get("grasp_in_object")):
            issues.error("task.controller.grasp_in_object", "TCP pose in the object frame (4x4, TCP +z approaches)")
        place = c.get("place") or {}
        if place.get("relative_to", "world") not in set(objects) | {"world"}:
            issues.error("task.controller.place.relative_to", "a scene object or 'world'")
        if not is_rigid(place.get("object_in_reference")):
            issues.error("task.controller.place.object_in_reference", "target object pose in the reference (4x4)")
        for key in ("approach_distance_m", "retreat_height_m", "open_wait_s", "close_wait_s", "final_hold_s"):
            num(key)
        for key in ("lift_height_m", "max_joint_speed_rad_s", "segment_min_s"):
            num(key, strict=True)
        for key in ("gripper_open_m", "gripper_close_m"):
            if not _number(c.get(key)) or stroke is None or not 0 <= c[key] <= stroke:
                issues.error(f"task.controller.{key}", "within [0, stroke]")
        if not isinstance(c.get("cartesian_pieces"), int) or not 1 <= c["cartesian_pieces"] <= 100:
            issues.error("task.controller.cartesian_pieces", "an integer in [1, 100]")
        ik = c.get("ik") or {}
        for key in ("position_tol_m", "rotation_tol_rad", "max_joint_change_rad"):
            if not _number(ik.get(key)) or ik[key] <= 0:
                issues.error(f"task.controller.ik.{key}", "a positive number")
    elif kind == "plugin":
        if c.get("path") != (project.data.get("specs") or {}).get("plugin"):
            issues.error("task.controller.path", "must equal specs.plugin in the project")
        if not isinstance(c.get("class", "Controller"), str) or not c.get("class", "Controller").isidentifier():
            issues.error("task.controller.class", "a Python class name")
        if not isinstance(c.get("params", {}), dict):
            issues.error("task.controller.params", "an object")
    else:
        issues.error("task.controller.type", f"one of {list(CONTROLLERS)}")
    success = task.get("success") or {}
    if not _number(success.get("hold_s")) or success["hold_s"] <= 0:
        issues.error("task.success.hold_s", "seconds the conditions must hold at the end of the episode")
    conds = success.get("conditions")
    if not isinstance(conds, list) or not conds:
        issues.error("task.success.conditions", "a non-empty list")
    for i, cond in enumerate(conds or []):
        field = f"task.success.conditions[{i}]"
        need = CONDITIONS.get(cond.get("type"))
        if need is None:
            issues.error(field, f"type must be one of {sorted(CONDITIONS)}")
            continue
        missing = [k for k in need if k not in cond]
        if missing:
            issues.error(field, f"missing {missing}")
            continue
        if "object" in need and cond["object"] not in objects:
            issues.error(field, "unknown object")
        if "with" in need and cond["with"] not in set(objects) | {"table", "robot"}:
            issues.error(field, "with: a scene object, 'table' or 'robot'")
        if cond["type"] == "position_in_box":
            if cond.get("relative_to", "world") not in set(objects) | {"world"} or not _shape_ok(
                    cond["center_m"], 3) or not _shape_ok(cond["half_extents_m"], 3):
                issues.error(field, "center_m and half_extents_m (3 numbers) relative to an object or 'world'")
        if cond["type"] == "axis_tilt_max" and cond["axis"] not in ("x", "y", "z"):
            issues.error(field, "axis x, y or z")
        if cond["type"] == "phase_in" and not (isinstance(cond["phases"], list) and cond["phases"]):
            issues.error(field, "a non-empty list of phase names")
    if not isinstance(success.get("require_phases", []), list):
        issues.error("task.success.require_phases", "a list of phase names")
    integrity = task.get("integrity") or {}
    for key in ("max_teleport_residual_m", "max_teleport_residual_rad", "max_settle_drift_m", "contact_impulse_eps"):
        if not _number(integrity.get(key)) or integrity[key] <= 0:
            issues.error(f"task.integrity.{key}", "a positive number")
    for key in ("success", "integrity", "controller", "episode"):
        issues.item(f"task.{key}", "chosen", task.get(key))


# ----------------------------------------------------------------------------- plugins
PLUGIN_IMPORTS = {"math", "numpy", "dataclasses", "typing", "collections", "itertools", "functools",
                  "copy", "enum", "statistics", "__future__"}
PLUGIN_NAMES = {"eval", "exec", "compile", "open", "__import__", "globals", "locals", "vars", "input",
                "breakpoint", "exit", "quit"}


def screen_plugin(source: str) -> list:
    """Static findings for review. This is not a sandbox: reviewed plugins run with your permissions."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return [f"syntax error: {error}"]
    findings = []
    api = any(isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "API_VERSION" for t in n.targets)
              and isinstance(n.value, ast.Constant) and n.value.value == 1 for n in tree.body)
    if not api:
        findings.append("module does not set API_VERSION = 1")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in PLUGIN_IMPORTS:
                    findings.append(f"line {node.lineno}: imports {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in PLUGIN_IMPORTS:
                findings.append(f"line {node.lineno}: imports from {node.module}")
        elif isinstance(node, ast.Name) and node.id in PLUGIN_NAMES:
            findings.append(f"line {node.lineno}: uses {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr not in ("__init__",):
            findings.append(f"line {node.lineno}: accesses {node.attr}")
    return findings


# ----------------------------------------------------------------------------- layouts
def _u(*parts: Any) -> float:
    text = "|".join(str(p) for p in (LAYOUT_SCHEME, *parts))
    return (int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") >> 11) / float(1 << 53)


def _uniform(rng: Sequence[float], *parts: Any) -> float:
    lo, hi = float(rng[0]), float(rng[1])
    return lo + (hi - lo) * _u(*parts)


def _rz(angle: float) -> list:
    c, s = math.cos(angle), math.sin(angle)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def sample_layouts(scene: Mapping[str, Any], seed: int, count: int, pool: str) -> list:
    """Deterministic object layouts from the declared nominal poses and randomization ranges."""
    s = plain(scene)
    rand = s.get("randomization") or {}
    ranges, min_d = rand.get("objects") or {}, float(rand.get("min_center_distance_m", 0.0))
    table = s["table"]
    cx, cy = (float(v) for v in table["center_xy_m"])
    yaw, (length, width, _) = float(table["yaw_rad"]), table["size_m"]
    layouts = []
    for index in range(count):
        rejections: dict = {}
        for draw in range(MAX_DRAWS):
            poses, problems = {}, []
            for o in s["objects"]:
                t = [list(map(float, row)) for row in o["pose_W"]]
                r = ranges.get(o["name"]) or {}
                key = (seed, pool, index, draw, o["name"])
                dyaw = _uniform(r.get("dyaw_rad", [0, 0]), *key, "yaw")
                rz = _rz(dyaw)
                rot3 = [[sum(rz[i][k] * t[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
                p = [t[0][3] + _uniform(r.get("dx_m", [0, 0]), *key, "x"),
                     t[1][3] + _uniform(r.get("dy_m", [0, 0]), *key, "y"), t[2][3]]
                poses[o["name"]] = [[*rot3[i], p[i]] for i in range(3)] + [[0.0, 0.0, 0.0, 1.0]]
                dx, dy = p[0] - cx, p[1] - cy
                lx, ly = dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw)
                radius = footprint_radius(o) or 0.0
                if abs(lx) > length / 2 - radius or abs(ly) > width / 2 - radius:
                    problems.append(f"{o['name']}_off_table")
            names = list(poses)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = poses[names[i]], poses[names[j]]
                    if math.hypot(a[0][3] - b[0][3], a[1][3] - b[1][3]) < min_d:
                        problems.append("objects_too_close")
            if not problems:
                break
            for p in problems:
                rejections[p] = rejections.get(p, 0) + 1
        else:
            raise ValueError(f"no admissible layout for {pool} index {index} after {MAX_DRAWS} draws: {rejections}")
        layouts.append({"schema": LAYOUT_SCHEME, "index": index, "pool": pool, "seed": int(seed), "draw": draw,
                        "objects": poses, "layout_sha256": canonical_sha256(poses),
                        "status": "PROPOSED_LAYOUT_NOT_AN_EPISODE"})
    return layouts


def parse_indices(text: str, limit: int) -> list:
    """'0-9,15' -> [0..9, 15], each below limit."""
    out = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        a, b = int(lo), int(hi or lo)
        if a > b or a < 0 or b >= limit:
            raise ValueError(f"index range {part!r} outside 0..{limit - 1}")
        out.update(range(a, b + 1))
    return sorted(out)


def review_digest(items: Iterable[Mapping[str, Any]]) -> str:
    return canonical_sha256(sorted(i["id"] for i in items))
