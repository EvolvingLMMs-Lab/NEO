"""Can-in-tape regression reference: the authors' sim_twin task runtime with its original checks.

This route reproduces the can-in-tape simulation set-up of the NEO-VLA real-robot work (AgileX PiPER
arm with an AgileX Pika gripper, the measured 57 x 146 mm can and 107/76 x 48 mm tape ring, and
the calibrated rig of that work). It drives the runtime's own task CLI unchanged and verifies every
episode with the original acceptance contract (SuccessConfig and audit_trace below, verbatim) and
the original per-episode gates. The runtime workspace and its inputs are not distributed with this
repository; see the guide. Results of this route describe that one set-up and transfer to no other
scene.
"""
from __future__ import annotations

import ast
import json
import math
import os
import queue
import socket
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import checks
from . import spec as specmod
from .pipeline import Route, attempts_used, bounded_pool, outcome, write_audit
from .store import PathError, canonical_sha256, ensure_inside, file_record, read_json, run_bounded, sha256_file, \
    tail, utc_now, write_new_json

# Reviewed sources of the acceptance contract used for the published reference episodes.
REVIEWED_PINS = {
    "task3_success.py": "80753a05e5893416973501f6c42ecda578119ee8aaffc70bcbdc729e14142d7d",
    "task3_recorder_check.py": "c9a129beed035c911c31b0a506e00c7e411d0e7412acdb756f7e5cf5d36896b4",
    "task3_config.py": "0e244d771ad96d523e81e619d49f1ef61def6ab7126b235662d75e733f6f36de",
    "task3_collect.py": "f14e10c72dae43a4d5cb25d3ac54d5c6072d499c80d6626c570d8b88f13debff",
    "task3_layout.py": "2059e9ad225ea717c2436b13aa542f23bed57d1bd2283c4cf73950e62ad08500",
}
# Native controller revisions that produced accepted reference episodes (recorded, not required).
REVIEWED_NATIVES = {
    "82eb4dbf1ba321cbad968e9306cfc550d04c4bda01b056005d5d5d34b0acb39d",
    "a480226ca84e46ce608c4df1fcb4ae2f885911bd4dd1a14c9fa1501c98edb437",
    "00e343c2d1de0dc2e8581250e0ce797234f7d9b06a0567be8210c3147e7d6155",
    "36dd6c380989ad2d81bd3ca866f25d37a753770b510c31c3b358004d36883466",
}
REAL_PLAN_SHA256 = "8b45dc8b6950db1be91692faaf977e337d3dc9113192da0af85367c22e887f10"  # episode1 execution plan
AUDIT_USED = ("task3_success.py", "task3_config.py", "task3_layout.py", "task3_frames.py", "task3_native.py")
RUNTIME_FILES = (*REVIEWED_PINS, "task3_frames.py", "task3_native.py", "task3_assets.py", "task3_rigmeta.py",
                 "rig/rig.yaml", "rig/loader.py", "recorder/raw_episode.py")
RENDERERS = {"--cpu-test": "SAPIEN CPU lavapipe", "--gpu": "SAPIEN Vulkan"}
COUNTS = {"in_domain": 200, "ood": 200}
RESERVE = 60
TIMEOUT_TOOL_S = 1800
ATTEMPT_SCHEMA = "neo-vla-real2sim-reference-attempt/1"
MISSING = object()

# ----------------------------------------------------------------------------- task3_config (reviewed values)
TASK_NAME = "can_in_tape_roll"
INSTRUCTION = "put the can in the tape roll"
CAN_LENGTH_M = 0.146
CAN_DIAMETER_M = 0.057
RING_OUTER_DIAMETER_M = 0.107
RING_INNER_DIAMETER_M = 0.076
RING_HEIGHT_M = 0.048
R1_FINGER_OFFSET_M = 0.013250253377535708
SUCCESS_HOLD_S = 5.0
PHYSICS_HZ = 240
APPEARANCE_STRONG_KEYS = ("exposure_gain", "white_balance_rgb", "sensor_noise_scale", "light_intensity",
                          "light_color_rgb", "material_hue_shift_deg", "material_saturation", "material_value",
                          "table_contrast", "table_grain_std", "background_texture_scale", "background_grain_std")
APPEARANCE_GROUPS = {
    "postprocess": ("exposure_gain", "white_balance_rgb", "sensor_noise_scale"),
    "lights": ("light_intensity", "light_color_rgb"),
    "object_materials": ("material_hue_shift_deg", "material_saturation", "material_value"),
    "surfaces": ("table_contrast", "table_grain_std", "background_texture_scale", "background_grain_std"),
}
APPLIED_GROUPS = {"in_domain": ("postprocess", "lights", "object_materials"),
                  "ood": ("postprocess", "lights", "object_materials", "surfaces")}
ESTIMATES = {
    "can_mass_kg": (0.30, "sealed ~300 ml beverage can (photo); not weighed"),
    "can_friction": (0.4, "inherited legacy aluminium-can estimate"),
    "can_restitution": (0.02, "inherited legacy can estimate"),
    "ring_mass_kg": (0.12, "partly used 48 mm tape roll with cardboard core; not weighed"),
    "ring_friction": (0.5, "task-2 provisional tape friction"),
    "ring_restitution": (0.02, "estimate"),
}


def can_radius() -> float:
    return CAN_DIAMETER_M / 2.0


def ring_radii() -> tuple:
    return RING_INNER_DIAMETER_M / 2.0, RING_OUTER_DIAMETER_M / 2.0


def bore_radial_clearance_m() -> float:
    return ring_radii()[0] - can_radius()


@dataclass(frozen=True)
class SuccessConfig:
    """Original tolerances of the reference task (unchanged)."""
    upright_tol_deg: float = 10.0
    lid_up_required: bool = True
    bore_tol_m: float = 0.001
    bottom_z_band_m: tuple = (-0.002, 0.004)
    ring_max_shift_m: float = 0.005
    ring_max_tilt_deg: float = 3.0
    speed_max_m_s: float = 0.02
    angular_speed_max_rad_s: float = 0.10
    released_gap_m: float = CAN_DIAMETER_M - R1_FINGER_OFFSET_M + 0.004
    hold_s: float = SUCCESS_HOLD_S
    lift_min_m: float = 0.03
    grasp_max_slip_m: float = 0.010
    grasp_max_rotation_deg: float = 5.0
    tick_tol_s: float = 1e-9
    transform_tol: float = 1e-6

    def as_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------- task3_success (verbatim)
TRACE_KEYS = (
    "timestamp_s", "phase", "can_pose", "ring_pose", "flange_pose",
    "can_linear_velocity_m_s", "can_angular_velocity_rad_s",
    "ring_linear_velocity_m_s", "ring_angular_velocity_rad_s",
    "jaw_measured_gap_m", "jaw_command_gap_m",
    "contact_can_table", "contact_can_ring", "contact_can_robot", "contact_can_left_pad",
    "contact_can_right_pad", "contact_ring_robot", "contact_ring_table",
)
POST_RELEASE_PHASES = ("release", "withdraw", "hold")
CARRY_PHASE = "carry"
POSE_KEYS = ("can_pose", "ring_pose", "flange_pose")
VECTOR_KEYS = ("can_linear_velocity_m_s", "can_angular_velocity_rad_s", "ring_linear_velocity_m_s",
               "ring_angular_velocity_rad_s")
SCALAR_KEYS = ("jaw_measured_gap_m", "jaw_command_gap_m")
FLAG_KEYS = tuple(k for k in TRACE_KEYS if k.startswith("contact_"))


def _norm3(v: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v))


def _col(t: Sequence[Sequence[float]], j: int) -> list:
    return [float(t[i][j]) for i in range(3)]


def _pos(t: Sequence[Sequence[float]]) -> list:
    return _col(t, 3)


def can_axis_tilt_deg(can_pose: Sequence[Sequence[float]]) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(can_pose[2][0])))))


def can_bottom_center(can_pose: Sequence[Sequence[float]]) -> list:
    p, x = _pos(can_pose), _col(can_pose, 0)
    return [p[i] - CAN_LENGTH_M / 2.0 * x[i] for i in range(3)]


def ring_tilt_deg(ring_pose: Sequence[Sequence[float]]) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(ring_pose[2][2])))))


def tick_checks(row: Mapping[str, Any], ring_initial: Sequence[Sequence[float]],
                c: SuccessConfig = SuccessConfig()) -> dict:
    can, ring = row["can_pose"], row["ring_pose"]
    bottom = can_bottom_center(can)
    rp = _pos(ring)
    radial = math.hypot(bottom[0] - rp[0], bottom[1] - rp[1])
    top_point = [bottom[i] + RING_HEIGHT_M * float(can[i][0]) for i in range(3)]
    radial_top = math.hypot(top_point[0] - rp[0], top_point[1] - rp[1])
    tilt = can_axis_tilt_deg(can)
    limit = bore_radial_clearance_m() + c.bore_tol_m
    return {
        "upright": tilt <= c.upright_tol_deg,
        "lid_up": (float(can[2][0]) > 0.0) or not c.lid_up_required,
        "bottom_on_table_height": c.bottom_z_band_m[0] <= bottom[2] <= c.bottom_z_band_m[1],
        "table_contact": bool(row["contact_can_table"]),
        "inside_bore": radial <= limit and radial_top <= limit,
        "no_robot_contact": not row["contact_can_robot"] and not row["contact_ring_robot"],
        "jaws_open": float(row["jaw_measured_gap_m"]) >= c.released_gap_m,
        "released_phase": row["phase"] in POST_RELEASE_PHASES,
        "ring_in_place": math.dist(_pos(ring)[:2], _pos(ring_initial)[:2]) <= c.ring_max_shift_m
                         and ring_tilt_deg(ring) <= c.ring_max_tilt_deg,
        "settled": (_norm3(row["can_linear_velocity_m_s"]) <= c.speed_max_m_s
                    and _norm3(row["can_angular_velocity_rad_s"]) <= c.angular_speed_max_rad_s
                    and _norm3(row["ring_linear_velocity_m_s"]) <= c.speed_max_m_s
                    and _norm3(row["ring_angular_velocity_rad_s"]) <= c.angular_speed_max_rad_s),
        "_tilt_deg": tilt, "_bottom_z_m": bottom[2], "_radial_m": radial,
    }


def _rows(arrays: Mapping[str, Sequence[Any]]) -> list:
    missing = [k for k in TRACE_KEYS if k not in arrays]
    if missing:
        raise ValueError(f"trace lacks {missing}")
    n = len(arrays["timestamp_s"])
    if any(len(arrays[k]) != n for k in TRACE_KEYS):
        raise ValueError("trace arrays have different lengths")
    return [{k: arrays[k][i] for k in TRACE_KEYS} for i in range(n)]


def _relative(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> tuple:
    d = [float(b[i][3]) - float(a[i][3]) for i in range(3)]
    t = [sum(float(a[k][j]) * d[k] for k in range(3)) for j in range(3)]
    r = [[sum(float(a[k][i]) * float(b[k][j]) for k in range(3)) for j in range(3)] for i in range(3)]
    return t, r


def _angle_deg(r1: Sequence[Sequence[float]], r2: Sequence[Sequence[float]]) -> float:
    tr = sum(float(r1[k][i]) * float(r2[k][i]) for i in range(3) for k in range(3))
    return math.degrees(math.acos(max(-1.0, min(1.0, (tr - 1.0) / 2.0))))


def _finite(value: Any) -> bool:
    try:
        return all(_finite(v) for v in value)
    except TypeError:
        return math.isfinite(float(value))


def _proper_transform(t: Sequence[Sequence[float]], tol: float) -> bool:
    if len(t) != 4 or any(len(row) != 4 for row in t) or not _finite(t):
        return False
    if max(abs(float(t[3][j]) - (0.0, 0.0, 0.0, 1.0)[j]) for j in range(4)) > tol:
        return False
    r = [[float(t[i][j]) for j in range(3)] for i in range(3)]
    for i in range(3):
        for j in range(3):
            dot = sum(r[k][i] * r[k][j] for k in range(3))
            if abs(dot - (1.0 if i == j else 0.0)) > tol:
                return False
    det = (r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
           + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]))
    return abs(det - 1.0) <= tol


def validate_rows(rows: Sequence[Mapping[str, Any]], c: SuccessConfig) -> list:
    problems = []
    if len(rows) < 2:
        return ["trace has fewer than two ticks"]
    times = []
    for i, r in enumerate(rows):
        try:
            t = float(r["timestamp_s"])
        except (TypeError, ValueError):
            t = float("nan")
        if not math.isfinite(t):
            problems.append(f"tick {i}: non-finite timestamp")
        times.append(t)
        for key in POSE_KEYS:
            if not _proper_transform(r[key], c.transform_tol):
                problems.append(f"tick {i}: {key} is not a finite proper rigid transform")
        for key in VECTOR_KEYS:
            if len(r[key]) != 3 or not _finite(r[key]):
                problems.append(f"tick {i}: {key} non-finite or not a 3-vector")
        for key in SCALAR_KEYS:
            if not _finite(r[key]):
                problems.append(f"tick {i}: {key} non-finite")
        for key in FLAG_KEYS:
            if str(r[key]) not in ("True", "False"):
                problems.append(f"tick {i}: {key} is not a boolean flag")
        if len(problems) > 20:
            break
    step = 1.0 / PHYSICS_HZ
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        if not (math.isfinite(dt) and abs(dt - step) <= c.tick_tol_s):
            problems.append(f"tick {i}: timestamp step {dt!r} is not exactly 1/{PHYSICS_HZ} s (gap/missing/non-monotonic)")
            if len(problems) > 40:
                break
    return problems


def slip_excursion(carry_unsupported: Sequence[Mapping[str, Any]]) -> dict:
    if not carry_unsupported:
        return {"translation_m": None, "rotation_deg": None, "ticks": 0}
    t0, r0 = _relative(carry_unsupported[0]["flange_pose"], carry_unsupported[0]["can_pose"])
    max_t = max_r = 0.0
    for row in carry_unsupported:
        t, r = _relative(row["flange_pose"], row["can_pose"])
        max_t = max(max_t, math.dist(t, t0))
        max_r = max(max_r, _angle_deg(r, r0))
    return {"translation_m": max_t, "rotation_deg": max_r, "ticks": len(carry_unsupported)}


def audit_trace(arrays: Mapping[str, Sequence[Any]], c: SuccessConfig = SuccessConfig()) -> dict:
    rows = _rows(arrays)
    integrity = validate_rows(rows, c)
    if integrity:
        return {"ok": False, "reasons": ["invalid trace: " + "; ".join(integrity[:5])],
                "integrity_problems": integrity, "config": c.as_dict()}
    reasons = []
    ring_initial = rows[0]["ring_pose"]
    carry = [r for r in rows if r["phase"] == CARRY_PHASE]
    unsupported = [r for r in carry if not r["contact_can_table"]]
    free_lift = [r for r in unsupported if r["contact_can_left_pad"] and r["contact_can_right_pad"]
                 and min(float(r["can_pose"][2][3]) - can_radius(),
                         can_bottom_center(r["can_pose"])[2]) >= c.lift_min_m]
    bilateral_all = bool(unsupported) and all(r["contact_can_left_pad"] and r["contact_can_right_pad"]
                                              for r in unsupported)
    slip = slip_excursion(unsupported)
    if not free_lift:
        reasons.append("no free lift with bilateral pad contact and no table contact")
    if not bilateral_all:
        reasons.append("bilateral pad contact lost while can off the table during carry")
    if slip["translation_m"] is None or slip["translation_m"] > c.grasp_max_slip_m:
        reasons.append(f"grasp slip translation excursion {slip['translation_m']} m exceeds {c.grasp_max_slip_m}")
    if slip["rotation_deg"] is None or slip["rotation_deg"] > c.grasp_max_rotation_deg:
        reasons.append(f"grasp slip rotation excursion {slip['rotation_deg']} deg exceeds {c.grasp_max_rotation_deg}")
    if not any(r["phase"] == "release" for r in rows):
        reasons.append("no release phase")
    best = run = 0.0
    failing: dict = {}
    last = None
    prev_ok = False
    for i, r in enumerate(rows):
        dt = float(r["timestamp_s"]) - float(rows[i - 1]["timestamp_s"]) if i else 0.0
        checks_ = tick_checks(r, ring_initial, c)
        ok = all(v for k, v in checks_.items() if not k.startswith("_"))
        if ok:
            run = run + dt if prev_ok and abs(dt - 1.0 / PHYSICS_HZ) <= c.tick_tol_s else 0.0
            best = max(best, run)
            prev_ok = True
        else:
            run = 0.0
            prev_ok = False
            if r["phase"] in POST_RELEASE_PHASES:
                for k, v in checks_.items():
                    if not k.startswith("_") and not v:
                        failing[k] = failing.get(k, 0) + 1
        last = checks_
    ends_ok = last is not None and all(v for k, v in last.items() if not k.startswith("_"))
    if best + 1e-9 < c.hold_s or not ends_ok:
        reasons.append(f"continuous stable in-bore time {best:.3f}s < {c.hold_s}s or final tick fails")
    return {"ok": not reasons, "reasons": reasons, "free_lift_ticks": len(free_lift), "grasp_slip": slip,
            "best_hold_s": best, "final_checks": {k: (bool(v) if not k.startswith("_") else v)
                                                   for k, v in (last or {}).items()},
            "post_release_failing_tick_counts": failing, "config": c.as_dict()}


# ----------------------------------------------------------------------------- helpers
def jn(value: Any) -> Any:
    """The collector's own JSON serialisation (default=str)."""
    return json.loads(json.dumps(value, default=str, allow_nan=False))


def get(obj: Any, path: str) -> Any:
    for key in path.split("."):
        if isinstance(obj, dict) and key in obj:
            obj = obj[key]
        else:
            return MISSING
    return obj


class Check:
    def __init__(self) -> None:
        self.fail: list = []
        self.hold: list = []

    def need(self, obj: Any, path: str, where: str) -> Any:
        value = get(obj, path)
        if value is MISSING:
            self.hold.append(f"field {where}.{path} absent")
        return value

    def ok(self, condition: bool, message: str) -> bool:
        if not condition:
            self.fail.append(message)
        return bool(condition)

    def as_dict(self) -> dict:
        return {"ok": not self.fail and not self.hold, "status": "FAIL" if self.fail else "HOLD" if self.hold else "PASS",
                "problems": (self.fail + self.hold)[:20]}


def native_helpers(native: Path) -> dict:
    """_tf, _rotvec_matrix, CAN_VISUALS and RING_VISUALS of the runtime's native file, by AST."""
    import __future__

    import numpy as np

    tree = ast.parse(native.read_text())
    ns: dict = {"np": np, "math": math, "Optional": Any, "Sequence": Sequence, "Any": Any}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("_tf", "_rotvec_matrix"):
            code = compile(ast.Module(body=[node], type_ignores=[]), str(native), "exec",
                           flags=__future__.annotations.compiler_flag, dont_inherit=True)
            exec(code, ns)  # noqa: S102 - the user's own runtime source, as the original auditor did
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("CAN_VISUALS", "RING_VISUALS"):
            ns[node.targets[0].id] = ast.literal_eval(node.value)
    missing = {"_tf", "_rotvec_matrix", "CAN_VISUALS", "RING_VISUALS"} - set(ns)
    if missing:
        raise ValueError(f"{native.name} lacks {sorted(missing)}")
    return ns


class ReferenceRoute(Route):
    name = "can_in_tape_reference"

    # ------------------------------------------------------------------------- configuration
    def cfg(self) -> dict:
        return dict(self.p.project.data.get("reference") or {})

    def workspace(self) -> Path:
        return Path(self.cfg()["workspace"]).resolve()

    def intake(self, issues: specmod.Issues) -> dict:
        p, r = self.p.project, self.cfg()
        if p.data.get("instruction") != INSTRUCTION:
            issues.error("instruction", f"the reference task instruction is {INSTRUCTION!r}")
        if p.data["collection"].get("episodes") != sum(COUNTS.values()):
            issues.error("collection.episodes", "the reference manifest has 400 episodes (200 in-domain, 200 OOD)")
        ws = r.get("workspace")
        if not isinstance(ws, str) or not Path(ws).is_dir():
            issues.request("reference.workspace", "runtime workspace not found", ("build",),
                           "absolute path of the can-in-tape runtime workspace (see the guide)")
            return {}
        missing = [f for f in RUNTIME_FILES if not (Path(ws) / f).is_file()]
        if missing:
            issues.request("reference.workspace", f"workspace lacks {missing}", ("build",),
                           "use the complete runtime workspace")
        try:
            ensure_inside(self.p.workdir, ws)
        except PathError:
            issues.request("workdir", "the runtime writes episodes only inside its workspace", ("build",),
                           f"pass --workdir {ws}/real2sim_work/{p.data.get('name')}")
        launcher = r.get("launcher")
        if not isinstance(launcher, list) or not launcher or not all(isinstance(a, str) for a in launcher):
            issues.request("reference.launcher", "no launcher", ("build",),
                           "argv prefix that runs a Python script in the runtime environment, for example "
                           "[\"bash\", \"/path/to/sim_twin/scripts/cpu.sh\"]")
        if r.get("renderer") not in ("cpu", "gpu"):
            issues.error("reference.renderer", "cpu (validated) or gpu (not validated for accepted episodes)")
        for key in ("actuator_profile", "real_plan"):
            path = (r.get("inputs") or {}).get(key)
            if not isinstance(path, str) or not Path(path).is_file():
                issues.request(f"reference.inputs.{key}", "file not found", ("build",), "see the guide")
        assets = (r.get("inputs") or {}).get("assets")
        if not isinstance(assets, str) or not (Path(assets) / "receipt.json").is_file():
            issues.request("reference.inputs.assets", "photo-derived asset directory not found", ("build",),
                           "the directory built by `task3_collect.py assets`, holding receipt.json")
        if not isinstance(r.get("seed", 20261028), int):
            issues.error("reference.seed", "an integer")
        workers = p.data["collection"].get("workers", 1)
        cpu_lists = r.get("cpu_lists") or []
        if isinstance(workers, int) and workers > 1 and len(cpu_lists) < workers:
            issues.request("reference.cpu_lists", "one CPU list per worker is required", ("collect",),
                           "for example [\"0-7\", \"8-15\"]; passed to the launcher as SIM_TWIN_CPU_LIST")
        for name, (value, basis) in ESTIMATES.items():
            issues.item(f"reference.{name}", "estimate", {"value": value, "basis": basis})
        for text in self.disclosures():
            issues.item("reference.disclosure", "disclosure", text)
        return {"reference_workspace": ws}

    def review(self) -> tuple:
        try:
            artifacts = {"runtime": canonical_sha256(self.pins()), "inputs": canonical_sha256(self.inputs())}
        except (KeyError, OSError, TypeError):
            artifacts = {"runtime": None, "inputs": None}
        return artifacts, list({i["id"]: i for i in self.p.project.issues.review}.values())

    def native(self) -> dict:
        if getattr(self, "_native", None) is None:
            self._native = native_helpers(self.workspace() / "task3_native.py")
        return self._native

    def pins(self) -> dict:
        ws = self.cfg().get("workspace")
        ws = Path(ws).resolve() if isinstance(ws, str) else None
        return {f: sha256_file(ws / f) if ws is not None and (ws / f).is_file() else None for f in RUNTIME_FILES}

    def inputs(self) -> dict:
        r = self.cfg().get("inputs") or {}
        paths = {"actuator_profile": r.get("actuator_profile"), "real_plan": r.get("real_plan"),
                 "assets_receipt": str(Path(r["assets"]) / "receipt.json") if isinstance(r.get("assets"), str)
                 else None}
        return {k: file_record(p) if isinstance(p, str) and Path(p).is_file() else None for k, p in paths.items()}

    def build_inputs(self) -> dict:
        r = self.cfg()
        return {"pins": self.pins(), "inputs": self.inputs(), "launcher": r["launcher"], "renderer": r["renderer"],
                "seed": r.get("seed", 20261028)}

    def tool(self, run: Path, name: str, args: list, timeout: float = TIMEOUT_TOOL_S, env: dict | None = None) -> dict:
        argv = [*self.cfg()["launcher"], str(self.workspace() / "task3_collect.py"), *args]
        res = run_bounded(argv, cwd=self.workspace(), env=env, timeout_s=timeout, stdout=run / f"{name}.stdout.txt",
                          stderr=run / f"{name}.stderr.txt")
        return {"argv": argv, **res}

    # ------------------------------------------------------------------------- build
    def build(self, run: Path) -> dict:
        pins = self.pins()
        bad = [f"{f}: {pins.get(f)} != reviewed {want}" for f, want in REVIEWED_PINS.items() if pins.get(f) != want]
        if bad:
            return outcome("FAILED", ["runtime acceptance contract differs from the reviewed version"] + bad, [])
        r, inputs = self.cfg(), self.inputs()
        if any(v is None for v in inputs.values()):
            return outcome("FAILED", [f"missing reference inputs: {sorted(k for k, v in inputs.items() if v is None)}"],
                           [])
        if inputs["real_plan"]["sha256"] != REAL_PLAN_SHA256:
            return outcome("FAILED", ["real_plan is not the reviewed episode-1 execution plan"], [])
        seed = int(r.get("seed", 20261028))
        manifest_path = run / "layout_manifest.json"
        steps = {"manifest": self.tool(run, "manifest", ["manifest", "--seed", str(seed), "--n-id", "200",
                                                         "--n-ood", "200", "--reserve", str(RESERVE),
                                                         "--out", str(manifest_path)])}
        problems = []
        if steps["manifest"]["returncode"] != 0 or not manifest_path.is_file():
            problems.append(f"manifest command failed: {tail(run / 'manifest.stderr.txt', 600)}")
        else:
            steps["validate"] = self.tool(run, "validate", ["validate", "--manifest", str(manifest_path)])
            if steps["validate"]["returncode"] != 0:
                problems.append("runtime validate_manifest rejected the manifest")
            problems += self.manifest_problems(read_json(manifest_path), pins)
        steps["verify_assets"] = self.tool(run, "verify_assets", ["verify-assets", "--assets", r["inputs"]["assets"]])
        if steps["verify_assets"]["returncode"] != 0:
            problems.append("runtime verify-assets rejected the photo-derived assets")
        native = pins.get("task3_native.py")
        build = {"schema": "neo-vla-real2sim-reference-build/1", "pins": pins, "inputs": inputs, "seed": seed,
                 "native_sha256": native, "native_reviewed": native in REVIEWED_NATIVES, "steps": steps,
                 "renderer": r["renderer"], "utc": utc_now()}
        write_new_json(run / "BUILD.json", build)
        outputs = ["BUILD.json"] + (["layout_manifest.json"] if manifest_path.is_file() else [])
        return outcome("FAILED" if problems else "PASSED", problems, outputs,
                       ["physics and rendering are UNVERIFIED until qualify"], physics="UNVERIFIED",
                       rendering="UNVERIFIED", native_reviewed=build["native_reviewed"])

    @staticmethod
    def manifest_problems(m: dict, pins: dict) -> list:
        problems = []
        body = {k: v for k, v in m.items() if k != "manifest_sha256"}
        if m.get("manifest_sha256") != canonical_sha256(body):
            problems.append("manifest_sha256 does not match its body")
        eps = m.get("episodes") or []
        if [e.get("index") for e in eps] != list(range(len(eps))):
            problems.append("episode indices are not contiguous")
        if {s: sum(e.get("split") == s for e in eps) for s in COUNTS} != COUNTS:
            problems.append("manifest counts are not 200 in-domain + 200 OOD")
        layouts = [e.get("layout_sha256") for e in eps + list(m.get("reserve") or [])]
        if len(set(layouts)) != len(layouts):
            problems.append("duplicate layouts")
        if len({tuple(e["can"]["quat_xyzw_T"]) for e in eps}) != 1:
            problems.append("can heading is not fixed")
        identity = m.get("source_identity") or {}
        for name in AUDIT_USED:
            if identity.get(name) != pins.get(name):
                problems.append(f"manifest source_identity {name} differs from the workspace")
        return problems

    # ------------------------------------------------------------------------- rollouts
    def _build(self, build: dict) -> tuple:
        run = build["dir"]   # absolute, resolved inside the workdir by the pipeline
        return run, read_json(run / "BUILD.json"), read_json(run / "layout_manifest.json")

    def run_attempt(self, attempt: Path, index: int, number: int, pool: str, build: dict, cpu_list: str | None) -> dict:
        run, info, manifest = self._build(build)
        entry = manifest["episodes"][index]
        r = self.cfg()
        attempt.mkdir(parents=True, exist_ok=False)
        flag = "--cpu-test" if r["renderer"] == "cpu" else "--gpu"
        args = ["rollout", "--manifest", str(run / "layout_manifest.json"), "--index", str(index), "--attempt",
                str(number), "--actuator", r["inputs"]["actuator_profile"], "--assets", r["inputs"]["assets"],
                "--plan", r["inputs"]["real_plan"], "--visual-mode", "photo", "--out", str(attempt / "collector_out"),
                flag]
        argv = [*r["launcher"], str(self.workspace() / "task3_collect.py"), *args]
        env = dict(os.environ, SIM_TWIN_CPU_LIST=cpu_list) if cpu_list else None
        timeout = float(self.p.project.data["collection"]["attempt_timeout_s"])
        write_new_json(attempt / "IDENTITY.json", {
            "schema": ATTEMPT_SCHEMA, "kind": "attempt_identity", "index": index, "pool": pool,
            "split_index": entry["split_index"], "layout_sha256": entry["layout_sha256"], "attempt": number,
            "argv": argv, "launcher_length": len(r["launcher"]), "native_mode": r["renderer"],
            "source_pins_verified": info["pins"], "inputs": info["inputs"], "build_run": run.name,
            "timeout_s": timeout, "host": socket.gethostname(), "started_utc": utc_now()})
        res = run_bounded(argv, cwd=self.workspace(), env=env, timeout_s=timeout, stdout=attempt / "stdout.txt",
                          stderr=attempt / "stderr.txt")
        label = f"episode{index + 1}_attempt{number}"
        out = attempt / "collector_out"
        collector = out / "attempts" / f"{label}.json"
        trace = out / "physics" / f"{label}.npz"
        reasons, reaudit = [], {"ok": False}
        stored = read_json(collector) if collector.is_file() else {}
        if trace.is_file():
            try:
                arrays = {k: v.tolist() for k, v in checks.load_npz(trace).items()}
                redo = jn(audit_trace({k: arrays[k] for k in TRACE_KEYS}))
                reaudit = {"ok": bool(redo.get("ok")), "reasons": redo.get("reasons")}
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
                reaudit = {"ok": False, "error": repr(error)}
        if res["timed_out"]:
            reasons.append("deadline reached")
        if res.get("output_limit_reached"):
            reasons.append(f"output limit reached on {res['output_limit_reached']}")
        if res["returncode"] != 0:
            reasons.append(f"collector exit {res['returncode']}")
        if stored.get("accepted") is not True or stored.get("reasons") != []:
            reasons.append("collector did not accept the attempt")
        if not reaudit["ok"]:
            reasons.append("re-audit of the persisted trace rejects it")
        pins_after = self.pins()
        if pins_after != info["pins"]:
            reasons.append("runtime sources changed during the attempt")
        result = {"schema": ATTEMPT_SCHEMA, "kind": "attempt_result", "index": index, "split": entry["split"],
                  "attempt": number, "pool": pool, "status": "FAILED" if reasons else "ACCEPTED", "reasons": reasons,
                  "waited_exit_code": res["returncode"], "timed_out": res["timed_out"], "wall_s": res["wall_s"],
                  "trace_sha256": sha256_file(trace) if trace.is_file() else None,
                  "collector_result_sha256": sha256_file(collector) if collector.is_file() else None,
                  "reaudit": reaudit, "source_pins_after": pins_after, "native_mode": r["renderer"],
                  "finished_utc": utc_now()}
        write_new_json(attempt / "RESULT.json", result)
        return result

    def qualify(self, run: Path, build: dict) -> dict:
        _, info, manifest = self._build(build)
        r = self.cfg()
        first_ood = next(e["index"] for e in manifest["episodes"] if e["split"] == "ood")
        indices = [0, first_ood]
        preview = self.tool(run, "preview", ["preview", "--manifest", str(build["dir"] / "layout_manifest.json"),
                                             "--actuator", r["inputs"]["actuator_profile"], "--assets",
                                             r["inputs"]["assets"], "--visual-mode", "photo", "--out",
                                             str(run / "preview"),
                                             "--cpu-test" if r["renderer"] == "cpu" else "--gpu",
                                             *sum((["--index", str(i)] for i in indices), [])],
                            timeout=float(self.p.project.data["collection"]["attempt_timeout_s"]))
        entries = []
        for index in indices:
            attempt = run / "attempts" / f"ep{index:05d}_a0"
            self.run_attempt(attempt, index, 0, "qualify", build, None)
            entries.append(self.audit_attempt(attempt, build))
        g4 = self.apply_g4(entries, manifest)
        write_new_json(run / "QUALIFY.json", {"preview": preview, "attempts": entries, "g4": g4})
        failed = [e for e in entries if e["status"] != "PASS"]
        reasons = [f"index {e['index']}: {e['failed']}" for e in failed]
        if preview["returncode"] != 0:
            reasons.append("preview command failed")
        return outcome("FAILED" if reasons else "PASSED", reasons, ["QUALIFY.json"],
                       ["visual similarity of previews to the real cameras (compare them yourself)",
                        "qualification attempts are never counted as dataset episodes"])

    def targets(self, build: dict) -> list:
        return list(range(sum(COUNTS.values())))

    def collect(self, namespace: Path, indices: list, workers: int, build: dict) -> dict:
        limit = int(self.p.project.data["collection"]["max_attempts_per_episode"])
        cpu_lists = list(self.cfg().get("cpu_lists") or [])
        tasks, skipped = [], 0
        for index in indices:
            done = any(read_json(a / "RESULT.json").get("status") == "ACCEPTED"
                       for a in (namespace / "attempts").glob(f"ep{index:05d}_a*") if (a / "RESULT.json").is_file())
            used = attempts_used(namespace, index)
            if done or used >= limit:
                skipped += 1
                continue
            tasks.append((index, used))
        free: queue.Queue = queue.Queue()
        for cpu in (cpu_lists[:workers] if workers > 1 else cpu_lists[:1]) or [None]:
            free.put(cpu)

        def one(task: tuple) -> dict:
            index, number = task
            cpu = free.get()   # each running attempt holds its own CPU list
            try:
                return self.run_attempt(namespace / "attempts" / f"ep{index:05d}_a{number}", index, number,
                                        "primary", build, cpu)
            finally:
                free.put(cpu)

        results = bounded_pool(tasks, workers, one)
        return {"accepted": sum(r["status"] == "ACCEPTED" for r in results),
                "failed": sum(r["status"] != "ACCEPTED" for r in results), "skipped": skipped}

    # ------------------------------------------------------------------------- historical checks
    def audit_attempt(self, attempt: Path, build: dict) -> dict:
        run, info, m = self._build(build)
        identity = read_json(attempt / "IDENTITY.json") if (attempt / "IDENTITY.json").is_file() else {}
        rs = read_json(attempt / "RESULT.json") if (attempt / "RESULT.json").is_file() else {}
        index, number = identity.get("index"), identity.get("attempt")
        entry = m["episodes"][index] if isinstance(index, int) and index < len(m["episodes"]) else None
        out = attempt / "collector_out"
        label = f"episode{(index or 0) + 1}_attempt{number}"
        checks_: dict = {}
        result = {"attempt_id": attempt.name, "index": index, "attempt": number, "pool": identity.get("pool"),
                  "split": entry["split"] if entry else None, "checks": checks_}
        if entry is None:
            checks_["driver_attempt"] = {"ok": False, "status": "FAIL", "problems": ["attempt has no manifest slot"]}
            result.update(failed=["driver_attempt"], status="FAIL")
            return result
        res_path, trace = out / "attempts" / f"{label}.json", out / "physics" / f"{label}.npz"
        episode_dir, failure_dir = out / f"episode{index + 1}", out / "failures" / label
        c = Check()
        res = read_json(res_path) if res_path.is_file() else MISSING
        if res is MISSING:
            c.hold.append("collector attempt JSON absent")
            res = {}
        trace_sha = sha256_file(trace) if trace.is_file() else None
        res_sha = sha256_file(res_path) if res_path.is_file() else None
        result.update(layout_sha256=entry["layout_sha256"], trace_sha256=trace_sha)
        c.ok(res.get("accepted") is True, "collector accepted is not true")
        c.ok(res.get("reasons") == [], f"collector reasons {res.get('reasons')!r} != []")
        c.ok(not res.get("error"), f"collector error {res.get('error')!r}")
        c.ok((res.get("index"), res.get("attempt"), res.get("split")) == (index, number, entry["split"]),
             "collector index/attempt/split differ from the slot")
        c.ok(Path(str(res.get("trace"))).parts[-2:] == ("physics", f"{label}.npz"), "collector trace path is not this label")
        c.ok(trace_sha is not None and res.get("trace_sha256") == trace_sha, "collector trace_sha256 != sha256(npz)")
        c.ok(not failure_dir.exists(), f"failures/{label} present (failed or partial attempt)")
        checks_["collector_receipt"] = c.as_dict()
        a = Check()
        config = jn(SuccessConfig().as_dict())
        audit = a.need(res, "audit", "collector")
        if audit is not MISSING:
            a.ok(audit.get("ok") is True, "collector audit.ok is not true")
            a.ok(audit.get("reasons") == [], "collector audit.reasons not empty")
            a.ok(audit.get("config") == config, "audit.config != original SuccessConfig().as_dict()")
        if trace_sha is not None:
            try:
                arrays = {k: v.tolist() for k, v in checks.load_npz(trace).items()}
                missing = [k for k in TRACE_KEYS if k not in arrays]
                if missing:
                    a.hold.append(f"trace lacks {missing}")
                else:
                    redo = jn(audit_trace({k: arrays[k] for k in TRACE_KEYS}))
                    a.ok(redo.get("ok") is True, f"re-run audit_trace rejects: {redo.get('reasons')}")
                    a.ok(redo.get("config") == config, "re-run audit config != SuccessConfig")
                    if audit is not MISSING and redo != audit:
                        diff = sorted(k for k in set(redo) | set(audit) if redo.get(k) != audit.get(k))
                        a.fail.append(f"re-run audit_trace does not reproduce the stored audit (keys {diff})")
                    result["report"] = {k: redo.get(k) for k in ("best_hold_s", "free_lift_ticks", "grasp_slip")}
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
                a.fail.append(f"trace unreadable: {error!r}")
        checks_["trace_audit"] = a.as_dict()
        w = Check()
        wm = read_json(episode_dir / "manifest.json") if (episode_dir / "manifest.json").is_file() else MISSING
        if wm is MISSING:
            w.hold.append(f"{episode_dir.name}/manifest.json absent (writer not finalized)")
            wm = {}
        complete = episode_dir / "COMPLETE"
        w.ok(complete.is_file() and complete.read_text() == "sim episode\n", "COMPLETE is not the writer marker")
        w.ok(wm.get("domain") == "sim", f"writer domain {wm.get('domain')!r} != 'sim'")
        for name, digest in (wm.get("input_sha256") or {}).items():
            w.ok((episode_dir / name).is_file() and sha256_file(episode_dir / name) == digest, f"{name} sha256 changed")
        w.ok(set(wm.get("input_sha256") or {}) == {"arrays.npz", "camera_geometry.npz", "joint_states.npz"},
             "writer input_sha256 does not list arrays/camera_geometry/joint_states")
        sim = w.need(wm, "simulation", "writer")
        sim = sim if isinstance(sim, dict) else {}
        tr = w.need(sim, "task3_result", "writer.simulation")
        tr = tr if isinstance(tr, dict) else {}
        w.ok(tr.get("trace_sha256") == trace_sha and trace_sha is not None, "writer trace_sha256 != sha256(npz)")
        for key in ("audit", "recording", "visuals", "appearance", "camera_readback"):
            w.ok(key in tr and key in res and jn(tr[key]) == jn(res[key]), f"writer task3_result.{key} != collector {key}")
        checks_["writer_episode"] = w.as_dict()
        rc = Check()
        rec = rc.need(res, "recording", "collector")
        if isinstance(rec, dict):
            rc.ok(rec.get("ok") is True, "recording.ok is not true")
            rc.ok(rec.get("problem_count") == 0, f"recording.problem_count {rec.get('problem_count')} != 0")
        if (episode_dir / "manifest.json").is_file():
            raw = checks.raw_episode_report(episode_dir)
            for cat, rep in raw["categories"].items():
                rc.ok(rep["ok"], f"raw episode re-check {cat}: {(rep['problems'] or [''])[0]}")
            result["episode_dir"] = str(episode_dir)
        checks_["recording"] = rc.as_dict()
        p = Check()
        prov = p.need(sim, "task3", "writer.simulation")
        prov = prov if isinstance(prov, dict) else {}
        p.ok(prov.get("layout", MISSING) is not MISSING and jn(prov.get("layout")) == jn(entry),
             "provenance.layout != manifest entry")
        p.ok((prov.get("split"), prov.get("episode_seed"), prov.get("collection_seed")) ==
             (entry["split"], entry["episode_seed"], entry["collection_seed"]), "provenance split/seed differ")
        p.ok(prov.get("manifest_sha256") == m["manifest_sha256"], "provenance.manifest_sha256 != manifest_sha256")
        p.ok(prov.get("task") == TASK_NAME and prov.get("visual_mode") == "photo", "provenance task/visual_mode")
        ident = prov.get("source_identity") or {}
        for name in AUDIT_USED:
            p.ok(ident.get(name) == info["pins"].get(name), f"collection source {name} differs from the build")
        renderer = sim.get("renderer")
        result["renderer"] = renderer
        p.ok(renderer in RENDERERS.values(), f"renderer {renderer!r} not a task3_collect value")
        checks_["provenance"] = p.as_dict()
        d = Check()
        d.ok(bool(identity) and bool(rs), "IDENTITY.json or RESULT.json missing")
        d.ok((rs.get("kind"), rs.get("status"), rs.get("reasons")) == ("attempt_result", "ACCEPTED", []),
             "RESULT is not an ACCEPTED attempt_result")
        d.ok((rs.get("index"), rs.get("split"), rs.get("attempt")) == (index, entry["split"], number), "RESULT slot")
        d.ok((identity.get("split_index"), identity.get("layout_sha256")) == (entry["split_index"],
                                                                              entry["layout_sha256"]),
             "IDENTITY slot/layout mismatch")
        d.ok(rs.get("waited_exit_code") == 0 and rs.get("timed_out") is False, "child exit/timeout")
        d.ok(rs.get("trace_sha256") == trace_sha and rs.get("collector_result_sha256") == res_sha,
             "RESULT trace/collector sha256 != files")
        d.ok((rs.get("reaudit") or {}).get("ok") is True, "driver re-audit not ok")
        argv = [str(x) for x in identity.get("argv") or []]
        tail_argv = argv[int(identity.get("launcher_length", 1)) + 1:]   # after [*launcher, task3_collect.py]
        flags = [f for f in RENDERERS if f in tail_argv]
        d.ok(tail_argv[:1] == ["rollout"] and "--physics-only" not in tail_argv and
             tail_argv[tail_argv.index("--visual-mode") + 1:][:1] == ["photo"] if "--visual-mode" in tail_argv
             else False, "argv is not a photo-mode rollout")
        d.ok(tail_argv[tail_argv.index("--index") + 1:][:1] == [str(index)] if "--index" in tail_argv else False,
             "argv --index")
        d.ok(len(flags) == 1 and renderer == RENDERERS[flags[0]], f"renderer {renderer!r} disagrees with argv {flags}")
        d.ok(identity.get("source_pins_verified") == info["pins"] and rs.get("source_pins_after") == info["pins"],
             "source pins before/after != build pins")
        checks_["driver_attempt"] = d.as_dict()
        self.gates(checks_, result, entry, res, sim, prov, episode_dir, info)
        result["failed"] = sorted(k for k, v in checks_.items() if not v["ok"] and k != "G4_camera_hardware_construction")
        result["status"] = "PASS" if not result["failed"] else "FAIL"
        return result

    def gates(self, out: dict, result: dict, entry: dict, res: dict, sim: dict, prov: dict, episode_dir: Path,
              info: dict) -> None:
        og = sim.get("object_geometry", MISSING)
        cal = prov.get("calibration_actual", MISSING)
        split = entry["split"]
        g = Check()
        for obj in ("can", "ring"):
            actual = g.need(og, f"{obj}.initial_pose_T", "simulation.object_geometry")
            lay = g.need(prov, f"layout.{obj}.pose_T", "simulation.task3")
            if actual is not MISSING and lay is not MISSING:
                g.ok(actual == lay == entry[obj]["pose_T"], f"{obj} initial_pose_T != layout/manifest pose_T (exact)")
        out["G1_initial_geometry_source"] = g.as_dict()
        g = Check()
        ri, ro = ring_radii()
        for path, want in (("can.length_m", CAN_LENGTH_M), ("can.diameter_m", CAN_DIAMETER_M),
                           ("ring.inner_radius_m", ri), ("ring.outer_radius_m", ro), ("ring.height_m", RING_HEIGHT_M)):
            v = g.need(og, path, "simulation.object_geometry")
            if v is not MISSING:
                g.ok(v == want, f"{path} {v!r} != {want!r}")
        v = g.need(og, "can.inertial.native_readback_verified", "simulation.object_geometry")
        if v is not MISSING:
            g.ok(v is True, "can.inertial.native_readback_verified is not true")
        v = g.need(prov, "scene_objects", "simulation.task3")
        if v is not MISSING:
            g.ok(v == ["can", "tape_ring"], f"scene_objects {v!r}")
        out["G2_static_geometry"] = g.as_dict()
        g = Check()
        v = g.need(cal, "T_T_Cs", "simulation.task3.calibration_actual")
        if v is not MISSING:
            g.ok(v == entry["camera"]["T_T_Cs"], "calibration_actual.T_T_Cs != layout.camera.T_T_Cs (exact)")
        out["G3_side_extrinsic"] = g.as_dict()
        result["calibration"] = {k: get(cal, k) if get(cal, k) is not MISSING else None for k in ("side_K", "T_F_Cw")}
        g = Check()
        cr = res.get("camera_readback", MISSING)
        v = g.need(cr, "after_capture", "collector.camera_readback")
        if v is not MISSING:
            g.ok(v is True, "camera_readback.after_capture is not true")
        for role, cam in (("side", "pikaSideCamera"), ("wrist", "pikaGripperDepthCamera")):
            match = g.need(cr, f"{role}.intrinsics_match_rig", "collector.camera_readback")
            wh = g.need(cr, f"{role}.width_height", "collector.camera_readback")
            conf_path = episode_dir / f"camera/color/{cam}/config.json"
            conf = read_json(conf_path) if conf_path.is_file() else MISSING
            if conf is MISSING:
                g.hold.append(f"{cam}/config.json absent")
            if match is not MISSING:
                g.ok(match is True, f"{role}.intrinsics_match_rig is not true")
            if wh is not MISSING and conf is not MISSING:
                g.ok(wh == [conf.get("width"), conf.get("height")] and all(type(x) is int for x in wh),
                     f"{role}.width_height {wh} != resolution")
        out["G5_render_intrinsics"] = g.as_dict()
        g = Check()
        ap = res.get("appearance", MISSING)
        want_groups = list(APPLIED_GROUPS[split])
        v = g.need(ap, "schema", "collector.appearance")
        if v is not MISSING:
            g.ok(v == "task3_appearance_receipt_v1", f"appearance.schema {v!r}")
        groups = g.need(ap, "groups", "collector.appearance")
        if groups is not MISSING:
            g.ok(groups == entry["appearance"]["applied_groups"] == want_groups, "appearance.groups mismatch")
        for path in ("lights", "object_materials", "readback.light_colors", "readback.postprocess") + \
                (("surfaces",) if split == "ood" else ()):
            v = g.need(ap, path, "collector.appearance")
            if v is not MISSING:
                g.ok(v is not None, f"appearance.{path} is null")
        calls = get(ap, "object_materials")
        if isinstance(calls, list):
            native = self.native()
            expected = [[actor, spec[1]] for actor, specs in (("can", native["CAN_VISUALS"]),
                                                               ("ring", native["RING_VISUALS"])) for spec in specs]
            g.ok([[x.get("actor"), x.get("texture")] for x in calls] == expected,
                 "object_materials calls != 6 source part textures (can then ring, add order)")
            g.ok(all(isinstance(x.get("tinted_sha256"), str) and len(x["tinted_sha256"]) == 64
                     and get(x, "readback.texture") not in (MISSING, None, "") for x in calls),
                 "object_materials call lacks tinted_sha256 or readback.texture")
        v = g.need(ap, "not_applied", "collector.appearance")
        if v is not MISSING:
            applied = {k for grp in want_groups for k in APPEARANCE_GROUPS[grp]}
            g.ok(v == sorted(set(APPEARANCE_STRONG_KEYS) - applied), "appearance.not_applied differs")
        out["G6_appearance_applied"] = g.as_dict()
        g = Check()
        nt = g.need(res, "no_teleport", "collector")
        if nt is not MISSING:
            if not g.ok(isinstance(nt, dict), "no_teleport is null"):
                nt = {}
            names, unavailable, wrapped = (g.need(nt, k, "collector.no_teleport")
                                           for k in ("object_names", "unavailable", "wrapped"))
            if names is not MISSING:
                g.ok(names == ["can", "ring"], f"no_teleport.object_names {names}")
            if unavailable is not MISSING:
                g.ok(unavailable == [], f"no_teleport.unavailable {unavailable}")
            if wrapped is not MISSING:
                can = {x.split(".", 1)[1] for x in wrapped if str(x).startswith("can.")}
                ring = {x.split(".", 1)[1] for x in wrapped if str(x).startswith("ring.")}
                g.ok(bool(wrapped) and can == ring and len(can) + len(ring) == len(wrapped),
                     "no_teleport.wrapped empty or method sets differ between can and ring")
        out["G7_no_teleport_guard"] = g.as_dict()
        g = Check()
        vis = res.get("visuals", MISSING)
        mode, accepted, receipt = (g.need(vis, k, "collector.visuals")
                                   for k in ("mode", "accepted_for_episodes", "asset_receipt_sha256"))
        a_ok, a_sha = g.need(prov, "assets.ok", "simulation.task3"), g.need(prov, "assets.receipt_sha256",
                                                                            "simulation.task3")
        if mode is not MISSING:
            g.ok(mode == "photo", f"visuals.mode {mode!r}")
        if accepted is not MISSING:
            g.ok(accepted is True, "visuals.accepted_for_episodes is not true")
        if receipt is not MISSING and a_sha is not MISSING:
            g.ok(receipt == a_sha and isinstance(receipt, str), "visuals.asset_receipt_sha256 != provenance receipt")
        if a_ok is not MISSING:
            g.ok(a_ok is True, "provenance.assets.ok is not true")
        out["G8_visual_asset_identity"] = g.as_dict()
        out["G4_camera_hardware_construction"] = {"ok": True, "status": "PENDING", "problems": []}

    def apply_g4(self, entries: list, manifest: dict) -> dict:
        """ID: one identical nominal (side_K, T_F_Cw); OOD: exact native construction from that baseline."""
        name = "G4_camera_hardware_construction"
        try:
            import numpy as np

            native = self.native()
        except (ImportError, OSError, ValueError) as error:
            for e in entries:
                e["checks"][name] = {"ok": False, "status": "HOLD", "problems": [f"G4 needs NumPy and the native "
                                                                                 f"source: {error!r}"]}
                e["failed"] = sorted(set(e["failed"]) | {name})
                e["status"] = "FAIL"
            return {"baseline": None}
        by_index = {e["index"]: e for e in manifest["episodes"]}
        ids = [e for e in entries if e["split"] == "in_domain" and e.get("calibration", {}).get("side_K") is not None]
        values = {canonical_sha256(jn([e["calibration"]["side_K"], e["calibration"]["T_F_Cw"]])) for e in ids}
        qualified = [e for e in ids if e["status"] == "PASS"]
        baseline = (qualified[0]["calibration"] if qualified and len(values) == 1 else None)
        for e in entries:
            problems = []
            if e["split"] == "in_domain":
                if len(values) != 1:
                    problems.append(f"in_domain side_K/T_F_Cw not identical across {len(values)} variants")
            elif baseline is None:
                problems.append("no qualified in_domain baseline (side_K, T_F_Cw); OOD construction unresolved")
            elif e.get("calibration", {}).get("side_K") is not None:
                cam = by_index[e["index"]]["camera"]
                scale = float(cam.get("side_focal_scale", 1.0))
                k = np.asarray(baseline["side_K"], float)
                k[0, 0] *= scale
                k[1, 1] *= scale
                delta = cam.get("wrist_mount_delta", {"rotvec_rad": [0, 0, 0], "translation_m": [0, 0, 0]})
                wrist = np.asarray(baseline["T_F_Cw"], float) @ native["_tf"](
                    native["_rotvec_matrix"](delta["rotvec_rad"]), delta["translation_m"])
                if jn(k.tolist()) != jn(e["calibration"]["side_K"]):
                    problems.append("ood side_K != ID side_K scaled by the layout focal scale")
                if jn(wrist.tolist()) != jn(e["calibration"]["T_F_Cw"]):
                    problems.append("ood T_F_Cw != ID T_F_Cw composed with the layout mount delta")
            else:
                problems.append("calibration_actual missing")
            e["checks"][name] = {"ok": not problems, "status": "FAIL" if problems else "PASS", "problems": problems}
            if problems:
                e["failed"] = sorted(set(e["failed"]) | {name})
                e["status"] = "FAIL"
        return {"baseline": baseline, "in_domain_variants": len(values)}

    def verify(self, run: Path, namespace: Path, build: dict) -> dict:
        _, info, manifest = self._build(build)
        base = namespace / "attempts"
        entries = [self.audit_attempt(a, build) for a in sorted(base.iterdir()) if a.is_dir()
                   and (a / "RESULT.json").is_file()] if base.is_dir() else []
        g4 = self.apply_g4(entries, manifest)
        problems = []
        for key in ("layout_sha256", "trace_sha256"):
            seen: dict = {}
            for e in entries:
                if e["status"] == "PASS" and e.get(key):
                    seen.setdefault(e[key], set()).add(e["index"])
            problems += [f"{key} shared by indices {sorted(v)}" for v in seen.values() if len(v) > 1]
        if not info.get("native_reviewed"):
            problems.append("the runtime native controller is not one of the reviewed revisions")
        return write_audit(run, entries, self.targets(build), problems,
                           {"g4": g4, "native_sha256": info.get("native_sha256")})

    # ------------------------------------------------------------------------- export metadata
    def disclosures(self) -> list:
        return ["Physical and appearance parameters (can and ring mass, friction, restitution; appearance "
                "variations) are declared estimates, not calibrated hardware values.",
                "The scripted expert was revised after observing failures on layouts of this same manifest; "
                "accepted episodes are success-conditioned and are not an unbiased benchmark of one controller.",
                "SAPIEN CPU (lavapipe) rendering is the validated path; GPU rendering has not produced accepted "
                "reference episodes.",
                "Passing these checks establishes simulation acceptance only; training admission is a separate "
                "decision."]

    def conversion_settings(self) -> dict:
        return {"layout": "the Pika raw episode layout of the real recordings", "arm_action": "master",
                "gripper_action": "command", "grip_tighten": "off", "joints": "measured", "joint_target": "command",
                "image_sync": "dataset (latest image at or before the row time)", "frame_ref": "camera:1,robot",
                "fps": 30, "note": "settings used when the reference episodes were converted with the authors' "
                                   "real-data converter, which is not part of this repository"}

    def diagnose_evidence(self, attempt: Path) -> dict:
        evidence: dict = {}
        for name in ("IDENTITY.json", "RESULT.json"):
            if (attempt / name).is_file():
                evidence[name] = attempt / name
        for path in sorted((attempt / "collector_out").glob("*/*.json"))[:10]:
            if path.stat().st_size < 2 << 20:
                evidence[f"collector_{path.parent.name}_{path.name}"] = path
        if (attempt / "stderr.txt").is_file():
            evidence["stderr_tail.txt"] = (attempt / "stderr.txt").read_bytes()[-20000:]
        return evidence
