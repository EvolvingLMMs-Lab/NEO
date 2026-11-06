"""Independent checks on persisted simulator evidence. Standard library only.

The same functions run inside the rollout runner before it finalizes an episode, in the verify
stage on the persisted files, and in export. Nothing here imports a simulator, and nothing reads a
plan, a command target or a layout to decide physical success. Pickled arrays are refused.

This module is copied next to the runner when a build is staged, so it must not import anything
from its package.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import struct
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

RAW_FORMAT = "neo-vla-raw-episode/1"
PIKA_FORMAT = "pika-raw/1"
TRACE_FORMAT = "neo-vla-sim-trace/1"
CONTROL_HZ = 30
GRID_TOL_S = 1e-6          # raw writer: rows on a monotonic 30 Hz grid
MAX_IMAGE_AGE_S = 0.075    # raw writer camera_valid rule
MAX_RGB_SKEW_S = 0.060
TICK_TOL_S = 1e-9          # trace timestamps are tick / physics_hz
TRANSFORM_TOL = 1e-6
FLOAT32_TOL = 1e-4         # labels stored as float32
FLOAT64_TOL = 1e-6
MAX_PROBLEMS = 60


# ----------------------------------------------------------------------------- hashing
def canonical_sha256(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


# ----------------------------------------------------------------------------- .npy / .npz
_CODES = {"f8": "d", "f4": "f", "i8": "q", "i4": "i", "i2": "h", "i1": "b",
          "u8": "Q", "u4": "I", "u2": "H", "u1": "B", "b1": "?"}


class Array:
    """A decoded .npy array: dtype descriptor, shape and raw C-order bytes."""

    __slots__ = ("descr", "shape", "data")

    def __init__(self, descr: str, shape: tuple, data: bytes) -> None:
        self.descr, self.shape, self.data = descr, shape, data

    @property
    def size(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def itemsize(self) -> int:
        kind = self.descr[1:]
        if kind in _CODES:
            return struct.calcsize(_CODES[kind])
        if kind[:1] == "U":
            return 4 * int(kind[1:])
        if kind[:1] == "S":
            return int(kind[1:])
        raise ValueError(f"unsupported dtype {self.descr}")

    def flat(self) -> list:
        endian = ">" if self.descr[0] == ">" else "<"
        kind = self.descr[1:]
        if kind in _CODES:
            return list(struct.unpack(f"{endian}{self.size}{_CODES[kind]}", self.data))
        step = self.itemsize()
        chunks = [self.data[i * step:(i + 1) * step] for i in range(self.size)]
        if kind[:1] == "U":
            codec = "utf-32-be" if endian == ">" else "utf-32-le"
            return [c.decode(codec).rstrip("\x00") for c in chunks]
        return [c.rstrip(b"\x00").decode("utf-8", "replace") for c in chunks]

    def tolist(self) -> Any:
        values = self.flat()
        if not self.shape:
            return values[0]

        def nest(offset: int, dims: list) -> list:
            if len(dims) == 1:
                return values[offset:offset + dims[0]]
            step = 1
            for d in dims[1:]:
                step *= d
            return [nest(offset + i * step, dims[1:]) for i in range(dims[0])]

        return nest(0, list(self.shape))


def read_npy(raw: bytes) -> Array:
    if raw[:6] != b"\x93NUMPY":
        raise ValueError("not an .npy member")
    major = raw[6]
    if major == 1:
        (length,) = struct.unpack("<H", raw[8:10])
        start = 10
    elif major in (2, 3):
        (length,) = struct.unpack("<I", raw[8:12])
        start = 12
    else:
        raise ValueError(f"unsupported .npy version {major}")
    header = ast.literal_eval(raw[start:start + length].decode("latin1"))
    descr = header.get("descr")
    if not isinstance(descr, str) or "O" in descr or "V" in descr:
        raise ValueError(f"refusing dtype {descr!r} (object or structured arrays are not read)")
    if header.get("fortran_order"):
        raise ValueError("Fortran-ordered arrays are not supported")
    array = Array(descr, tuple(int(d) for d in header.get("shape", ())), raw[start + length:])
    if len(array.data) != array.itemsize() * array.size:
        raise ValueError(f"truncated array data for dtype {descr} shape {array.shape}")
    return array


def load_npz(path: str | Path, max_member_bytes: int = 2 << 30) -> dict:
    out: dict = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if not info.filename.endswith(".npy") or "/" in info.filename:
                raise ValueError(f"{path}: unexpected member {info.filename!r}")
            if info.file_size > max_member_bytes:
                raise ValueError(f"{path}: member {info.filename} exceeds {max_member_bytes} bytes")
            out[info.filename[:-4]] = read_npy(archive.read(info))
    return out


def content_sha256(arrays: Mapping[str, Array]) -> str:
    """Digest of array content independent of zip timestamps (dtype, shape and bytes)."""
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = arrays[key]
        digest.update(f"{key}|{value.descr}|{value.shape}|".encode())
        digest.update(value.data)
    return digest.hexdigest()


# ----------------------------------------------------------------------------- geometry
def finite(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def is_rigid(t: Any, tol: float = TRANSFORM_TOL) -> bool:
    if not isinstance(t, (list, tuple)) or len(t) != 4 or any(len(r) != 4 for r in t) or not finite(t):
        return False
    if max(abs(float(t[3][j]) - (0.0, 0.0, 0.0, 1.0)[j]) for j in range(4)) > tol:
        return False
    r = [[float(t[i][j]) for j in range(3)] for i in range(3)]
    for i in range(3):
        for j in range(3):
            dot = sum(r[k][i] * r[k][j] for k in range(3))
            if abs(dot - (1.0 if i == j else 0.0)) > tol:
                return False
    return abs(det3(r) - 1.0) <= tol


def det3(r: Sequence[Sequence[float]]) -> float:
    return (r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]))


def rot(t: Sequence[Sequence[float]]) -> list:
    return [[float(t[i][j]) for j in range(3)] for i in range(3)]


def pos(t: Sequence[Sequence[float]]) -> list:
    return [float(t[i][3]) for i in range(3)]


def matmul3(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def transpose3(a: Sequence[Sequence[float]]) -> list:
    return [[a[j][i] for j in range(3)] for i in range(3)]


def apply_inverse(t: Sequence[Sequence[float]], p: Sequence[float]) -> list:
    """Coordinates of world point p in the frame of rigid transform t."""
    d = [p[i] - float(t[i][3]) for i in range(3)]
    return [sum(float(t[k][j]) * d[k] for k in range(3)) for j in range(3)]


def rotvec(r: Sequence[Sequence[float]]) -> list:
    """Rotation vector (axis * angle) of a 3x3 rotation matrix."""
    cos = max(-1.0, min(1.0, (r[0][0] + r[1][1] + r[2][2] - 1.0) / 2.0))
    angle = math.acos(cos)
    v = [r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]]
    if angle < 1e-9:
        return [x / 2.0 for x in v]
    if math.pi - angle < 1e-6:
        diag = [max(0.0, (r[i][i] + 1.0) / 2.0) for i in range(3)]
        i = max(range(3), key=lambda k: diag[k])
        axis = [0.0, 0.0, 0.0]
        axis[i] = math.sqrt(diag[i])
        for j in range(3):
            if j != i:
                axis[j] = (r[i][j] + r[j][i]) / (4.0 * axis[i])
        n = math.sqrt(sum(a * a for a in axis))
        return [angle * a / n for a in axis]
    s = angle / (2.0 * math.sin(angle))
    return [s * x for x in v]


def angle_between(r1: Sequence[Sequence[float]], r2: Sequence[Sequence[float]]) -> float:
    tr = sum(float(r1[k][i]) * float(r2[k][i]) for i in range(3) for k in range(3))
    return math.acos(max(-1.0, min(1.0, (tr - 1.0) / 2.0)))


def euler_zyx(yaw: float, pitch: float, roll: float) -> list:
    cy, sy, cp, sp, cr, sr = (math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch),
                              math.cos(roll), math.sin(roll))
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]]


def quat_xyzw_matrix(q: Sequence[float]) -> list:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v))


# ----------------------------------------------------------------------------- trace
TRACE_KEYS = ("tick", "time_s", "phase", "obj_pose", "obj_lin_vel", "obj_ang_vel", "ee_pose",
              "arm_q", "arm_q_target", "gripper_m", "gripper_cmd_m", "contacts")


def decode_trace(meta: Mapping[str, Any], arrays: Mapping[str, Array]) -> tuple:
    """Decode and integrity-check a trace. Returns (data, problems); data is None on failure."""
    problems: list = []
    missing = [k for k in TRACE_KEYS if k not in arrays]
    if missing:
        return None, [f"trace lacks {missing}"]
    if meta.get("format") != TRACE_FORMAT:
        return None, [f"trace format {meta.get('format')!r} != {TRACE_FORMAT}"]
    objects, columns = list(meta.get("objects") or []), list(meta.get("contact_columns") or [])
    m, cols, joints = len(objects), len(columns), len(meta.get("arm_joints") or [])
    n = arrays["tick"].shape[0] if arrays["tick"].shape else 0
    want = {"tick": (n,), "time_s": (n,), "phase": (n,), "obj_pose": (n, m, 4, 4),
            "obj_lin_vel": (n, m, 3), "obj_ang_vel": (n, m, 3), "ee_pose": (n, 4, 4),
            "arm_q": (n, joints), "arm_q_target": (n, joints), "gripper_m": (n,),
            "gripper_cmd_m": (n,), "contacts": (n, m, cols)}
    for key, shape in want.items():
        if arrays[key].shape != shape:
            problems.append(f"{key} shape {arrays[key].shape} != {shape}")
    if arrays["contacts"].descr != "|b1":
        problems.append("contacts is not a boolean array")
    if arrays["phase"].descr[1:2] != "U":
        problems.append("phase is not a text array")
    if n < 2:
        problems.append("trace has fewer than two ticks")
    if problems:
        return None, problems
    data = {k: arrays[k].tolist() for k in TRACE_KEYS}
    hz = float(meta["physics_hz"])
    if data["tick"] != list(range(n)):
        problems.append("tick indices are not 0..n-1")
    if abs(float(data["time_s"][0])) > TICK_TOL_S:
        problems.append("trace does not start at time 0")
    for i in range(1, n):
        dt = float(data["time_s"][i]) - float(data["time_s"][i - 1])
        if not (math.isfinite(dt) and abs(dt - 1.0 / hz) <= TICK_TOL_S):
            problems.append(f"tick {i}: step {dt!r} is not 1/{hz:g} s (gap, repeat or reorder)")
            if len(problems) > MAX_PROBLEMS:
                break
    for key in ("time_s", "obj_lin_vel", "obj_ang_vel", "arm_q", "arm_q_target", "gripper_m",
                "gripper_cmd_m"):
        if not all(math.isfinite(v) for v in arrays[key].flat()):
            problems.append(f"{key} has non-finite values")
    for i in range(n):
        if not is_rigid(data["ee_pose"][i]):
            problems.append(f"tick {i}: ee_pose is not a proper rigid transform")
        for j in range(m):
            if not is_rigid(data["obj_pose"][i][j]):
                problems.append(f"tick {i}: {objects[j]['name']} pose is not a proper rigid transform")
        if not data["phase"][i]:
            problems.append(f"tick {i}: empty phase label")
        if len(problems) > MAX_PROBLEMS:
            break
    return (None if problems else data), problems


def _com_world(t: Sequence[Sequence[float]], com: Sequence[float]) -> list:
    return [float(t[i][3]) + sum(float(t[i][k]) * float(com[k]) for k in range(3)) for i in range(3)]


def motion_report(meta: Mapping[str, Any], data: Mapping[str, Any], integrity: Mapping[str, Any]) -> dict:
    """Pose changes the object's own velocity does not explain (teleports), and settle drift."""
    dt = 1.0 / float(meta["physics_hz"])
    out: dict = {"objects": {}, "reasons": []}
    tol_m = float(integrity["max_teleport_residual_m"])
    tol_rad = float(integrity["max_teleport_residual_rad"])
    drift_tol = float(integrity["max_settle_drift_m"])
    settle = [i for i, ph in enumerate(data["phase"]) if ph == "settle"]
    for j, obj in enumerate(meta["objects"]):
        poses = [row[j] for row in data["obj_pose"]]
        name = obj["name"]
        if not obj.get("dynamic", True):
            moved = max(math.dist(pos(p), pos(poses[0])) for p in poses)
            out["objects"][name] = {"static_max_displacement_m": moved}
            if moved > 1e-9:
                out["reasons"].append(f"static object {name} moved {moved:.3g} m")
            continue
        com = obj.get("com_local_m") or [0.0, 0.0, 0.0]
        lin = rad = 0.0
        for k in range(1, len(poses)):
            p0, p1 = _com_world(poses[k - 1], com), _com_world(poses[k], com)
            v = data["obj_lin_vel"][k][j]
            lin = max(lin, norm([p1[i] - p0[i] - float(v[i]) * dt for i in range(3)]))
            w = data["obj_ang_vel"][k][j]
            dr = rotvec(matmul3(rot(poses[k]), transpose3(rot(poses[k - 1]))))
            rad = max(rad, norm([dr[i] - float(w[i]) * dt for i in range(3)]))
        drift = max((math.dist(pos(poses[i]), pos(poses[0])) for i in settle), default=0.0)
        out["objects"][name] = {"max_residual_m": lin, "max_residual_rad": rad, "settle_drift_m": drift}
        if lin > tol_m:
            out["reasons"].append(f"{name}: position change unexplained by velocity {lin:.3g} m > {tol_m}")
        if rad > tol_rad:
            out["reasons"].append(f"{name}: rotation change unexplained by angular velocity "
                                  f"{rad:.3g} rad > {tol_rad}")
        if drift > drift_tol:
            out["reasons"].append(f"{name}: drifted {drift:.3g} m while settling (> {drift_tol})")
    columns = list(meta["contact_columns"])
    start = int(meta.get("contacts_valid_from_tick", 1))
    if "robot" in columns:
        r = columns.index("robot")
        for i in settle:
            if i >= start and any(data["contacts"][i][j][r] for j in range(len(meta["objects"]))):
                out["reasons"].append(f"tick {i}: the robot touches an object before the controller starts")
                break
    return out


def _condition(c: Mapping[str, Any], k: int, data: Mapping[str, Any], names: list, columns: list,
               valid_from: int) -> bool:
    kind = c["type"]
    if kind == "phase_in":
        return data["phase"][k] in c["phases"]
    if kind == "gripper_open_min":
        return float(data["gripper_m"][k]) >= float(c["aperture_m"])
    j = names.index(c["object"])
    t = data["obj_pose"][k][j]
    if kind == "position_in_box":
        p = pos(t)
        ref = c.get("relative_to", "world")
        if ref != "world":
            p = apply_inverse(data["obj_pose"][k][names.index(ref)], p)
        return all(abs(p[i] - float(c["center_m"][i])) <= float(c["half_extents_m"][i]) for i in range(3))
    if kind == "axis_tilt_max":
        a = "xyz".index(c["axis"])
        return math.degrees(math.acos(max(-1.0, min(1.0, float(t[2][a]))))) <= float(c["max_deg"])
    if kind in ("contact", "no_contact"):
        if k < valid_from:
            return False
        flag = bool(data["contacts"][k][j][columns.index(c["with"])])
        return flag if kind == "contact" else not flag
    if kind == "speed_max":
        return (norm(data["obj_lin_vel"][k][j]) <= float(c["linear_m_s"])
                and norm(data["obj_ang_vel"][k][j]) <= float(c["angular_rad_s"]))
    if kind == "displacement_max":
        p0, p = pos(data["obj_pose"][0][j]), pos(t)
        dims = 2 if c.get("planar", True) else 3
        return math.dist(p[:dims], p0[:dims]) <= float(c["max_m"])
    raise ValueError(f"unknown condition type {kind!r}")


def success_report(success: Mapping[str, Any], meta: Mapping[str, Any], data: Mapping[str, Any]) -> dict:
    """All conditions must hold on every tick of a continuous run that ends at the last tick."""
    names = [o["name"] for o in meta["objects"]]
    columns = list(meta["contact_columns"])
    valid_from = int(meta.get("contacts_valid_from_tick", 1))
    conditions = list(success["conditions"])
    dt = 1.0 / float(meta["physics_hz"])
    n = len(data["tick"])
    run = 0
    states = []
    for k in range(n):
        ok = [_condition(c, k, data, names, columns, valid_from) for c in conditions]
        states.append(ok)
        run = run + 1 if all(ok) else 0
    final_run_s = (run - 1) * dt if run > 0 else 0.0
    hold_s = float(success["hold_s"])
    window = max(1, int(round(hold_s / dt)) + 1)
    failing = {}
    for idx, c in enumerate(conditions):
        count = sum(1 for k in range(max(0, n - window), n) if not states[k][idx])
        if count:
            failing[f"{idx}:{c['type']}"] = count
    reasons = []
    if final_run_s + 1e-9 < hold_s:
        reasons.append(f"conditions held for the final {final_run_s:.3f} s < {hold_s} s")
    seen = set(data["phase"])
    for phase in success.get("require_phases", []):
        if phase not in seen:
            reasons.append(f"required phase {phase!r} never occurred")
    return {"ok": not reasons, "reasons": reasons, "final_run_s": final_run_s,
            "failing_ticks_in_final_window": failing}


def audit_trace(task: Mapping[str, Any], meta: Mapping[str, Any], arrays: Mapping[str, Array]) -> dict:
    """Independent acceptance of one persisted trace against the reviewed task contract."""
    contract = {"success": task["success"], "integrity": task["integrity"]}
    data, problems = decode_trace(meta, arrays)
    if data is None:
        return jsonable({"ok": False, "reasons": ["invalid trace: " + "; ".join(problems[:5])],
                         "integrity_problems": problems[:MAX_PROBLEMS],
                         "contract_sha256": canonical_sha256(contract)})
    motion = motion_report(meta, data, task["integrity"])
    success = success_report(task["success"], meta, data)
    reasons = motion["reasons"] + success["reasons"]
    return jsonable({"ok": not reasons, "reasons": reasons, "motion": motion["objects"],
                     "success": success, "ticks": len(data["tick"]),
                     "contract_sha256": canonical_sha256(contract)})


def initial_pose_problems(meta: Mapping[str, Any], arrays: Mapping[str, Array],
                          layout: Mapping[str, Any], tol: float = 1e-6) -> list:
    """Tick 0 must be the layout the episode was planned with (float32 solver storage)."""
    first = arrays["obj_pose"].tolist()[0]
    problems = []
    for j, obj in enumerate(meta["objects"]):
        want = layout["objects"].get(obj["name"])
        if want is None:
            continue
        if math.dist(pos(first[j]), pos(want)) > tol or angle_between(rot(first[j]), rot(want)) > tol:
            problems.append(f"{obj['name']}: tick-0 pose differs from the layout")
    return problems


# ----------------------------------------------------------------------------- raw episodes
def _format(manifest: Mapping[str, Any], episode: Path) -> dict:
    if manifest.get("format") == RAW_FORMAT:
        cams = [{"name": c["name"], "every": int(c.get("every_n_rows", 1))} for c in manifest["cameras"]]
        return {"id": RAW_FORMAT, "cameras": cams, "grip_measured": "gripper/encoder/measured",
                "grip_command": "gripper/encoder/command"}
    sync = manifest.get("training_sync") or {}
    if sync.get("schema") == "two_rgb_training_sync_v1":
        cams = [{"name": n, "every": 1} for n in ("pikaSideCamera", "pikaGripperDepthCamera")]
        return {"id": PIKA_FORMAT, "cameras": cams, "grip_measured": "gripper/encoder/pikaGripper",
                "grip_command": "gripper/encoder/pikaSensor"}
    raise ValueError(f"{episode}: unknown raw episode format")


def png_size(path: Path) -> tuple:
    with open(path, "rb") as handle:
        head = handle.read(26)
    if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        raise ValueError(f"{path.name}: not a PNG")
    if head[24] != 8 or head[25] not in (2, 6):
        raise ValueError(f"{path.name}: not 8-bit RGB/RGBA")
    return struct.unpack(">II", head[16:24])


class _Category:
    def __init__(self) -> None:
        self.problems: list = []

    def add(self, message: str) -> None:
        if len(self.problems) < MAX_PROBLEMS:
            self.problems.append(message)

    def as_dict(self) -> dict:
        return {"ok": not self.problems, "problems": self.problems}


def _row_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _pose_from_row(row: Mapping[str, Any]) -> tuple:
    p = [float(row["x"]), float(row["y"]), float(row["z"])]
    if all(k in row for k in ("qx", "qy", "qz", "qw")):
        r = quat_xyzw_matrix([row["qx"], row["qy"], row["qz"], row["qw"]])
    else:
        r = euler_zyx(float(row["yaw"]), float(row["pitch"]), float(row["roll"]))
    return p, r


def raw_episode_report(episode: str | Path, *, label_checks: bool = True) -> dict:
    """Schema, numeric, causal, label and recording checks of one finalized raw episode."""
    episode = Path(episode)
    cats = {k: _Category() for k in ("schema", "numeric", "causal", "labels", "recording")}
    out: dict = {"episode": episode.name}

    def finish() -> dict:
        out["categories"] = {k: c.as_dict() for k, c in cats.items()}
        out["raw_schema_valid"] = cats["schema"].as_dict()["ok"] and cats["numeric"].as_dict()["ok"]
        out["conversion_checks_ok"] = all(c.as_dict()["ok"] for c in cats.values())
        return out

    try:
        manifest = json.loads((episode / "manifest.json").read_text())
        fmt = _format(manifest, episode)
    except (OSError, ValueError) as error:
        cats["schema"].add(f"manifest unreadable: {error}")
        return finish()
    out["format"] = fmt["id"]
    s = cats["schema"]
    for name in ("arrays.npz", "camera_geometry.npz", "joint_states.npz", "training_sync.json",
                 "instructions.json", "COMPLETE"):
        if not (episode / name).is_file():
            s.add(f"missing {name}")
    if (episode / "COMPLETE").is_file() and (episode / "COMPLETE").read_text() != "sim episode\n":
        s.add("COMPLETE is not the simulator writer marker")
    if manifest.get("domain") != "sim":
        s.add(f"domain {manifest.get('domain')!r} != 'sim'")
    hashes = manifest.get("input_sha256") or {}
    if set(hashes) != {"arrays.npz", "camera_geometry.npz", "joint_states.npz"}:
        s.add("input_sha256 does not list arrays/camera_geometry/joint_states")
    for name, digest in hashes.items():
        if not (episode / name).is_file() or sha256_file(episode / name) != digest:
            s.add(f"{name} sha256 differs from the manifest")
    if s.problems:
        return finish()
    try:
        arrays = load_npz(episode / "arrays.npz")
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        s.add(f"arrays.npz unreadable: {error}")
        return finish()
    n, k = int(manifest.get("rows", -1)), len(fmt["cameras"])
    want = {"states": ((n, 8), "<f8"), "actions": ((n, 7), "<f4"), "valid": ((n,), "|b1"),
            "times": ((4 + k, n), "<f8"), "intrinsics": ((k, 6), "<f8")}
    for key, (shape, descr) in want.items():
        if key not in arrays:
            s.add(f"arrays.npz lacks {key}")
        elif arrays[key].shape != shape or arrays[key].descr != descr:
            s.add(f"arrays.{key} is {arrays[key].descr}{arrays[key].shape}, expected {descr}{shape}")
    if n < 2:
        s.add("fewer than two rows")
    if s.problems:
        return finish()
    states, actions = arrays["states"].tolist(), arrays["actions"].tolist()
    valid, times = arrays["valid"].tolist(), arrays["times"].tolist()
    intr = arrays["intrinsics"].tolist()
    sync = manifest.get("training_sync") or {}
    grid = [float(t) for t in sync.get("grid_times_s") or []]
    if len(grid) != n:
        s.add("training_sync.grid_times_s length != rows")
        return finish()
    stroke = float(manifest["gripper_stroke_m"])
    out.update(rows=n, valid_count=sum(1 for v in valid if v))

    num = cats["numeric"]
    for key in ("states", "actions", "times", "intrinsics"):
        if not all(math.isfinite(v) for v in arrays[key].flat()):
            num.add(f"{key} has non-finite values")
    for i, row in enumerate(states):
        if abs(norm(row[3:7]) - 1.0) > FLOAT64_TOL:
            num.add(f"row {i}: state quaternion is not unit length")
        if not 0.0 <= row[7] <= 1.0:
            num.add(f"row {i}: state gripper fraction {row[7]} outside [0, 1]")
    for i, row in enumerate(actions):
        if not 0.0 <= row[6] <= 1.0 + FLOAT32_TOL:
            num.add(f"row {i}: action gripper fraction {row[6]} outside [0, 1]")
    for r in range(4):
        if any(abs(times[r][i] - grid[i]) > 0.0 for i in range(n)):
            num.add(f"times row {r} differs from the row grid")
    for i in range(1, n):
        if abs(grid[i] - grid[i - 1] - 1.0 / CONTROL_HZ) > GRID_TOL_S:
            num.add(f"row {i}: not on the {CONTROL_HZ} Hz grid")
    if valid[-1]:
        num.add("last row is marked as a valid transition")
    if not any(valid):
        num.add("no valid transition")
    if list(manifest.get("valid_transitions") or []) != valid:
        num.add("manifest valid_transitions differ from arrays.valid")
    for slot, cam in enumerate(fmt["cameras"]):
        try:
            conf = json.loads((episode / f"camera/color/{cam['name']}/config.json").read_text())
            kf = [float(v) for v in conf["K"]]
            want_row = [kf[0], kf[4], kf[2], kf[5], float(conf["width"]), float(conf["height"])]
            if [float(v) for v in intr[slot]] != want_row:
                num.add(f"{cam['name']}: arrays.intrinsics differ from config.json")
            cam["size"] = (int(conf["width"]), int(conf["height"]))
        except (OSError, KeyError, ValueError, IndexError) as error:
            s.add(f"{cam['name']}/config.json unreadable: {error!r}")
            cam["size"] = None

    cau = cats["causal"]
    for slot, cam in enumerate(fmt["cameras"]):
        stamps = times[4 + slot]
        for i in range(n):
            if stamps[i] > grid[i] + TICK_TOL_S:
                cau.add(f"{cam['name']} row {i}: image time is after the row time")
    for i in range(n):
        if not valid[i]:
            continue
        stamps = [times[4 + slot][i] for slot in range(k)]
        if any(grid[i] - t > MAX_IMAGE_AGE_S + TICK_TOL_S for t in stamps):
            cau.add(f"row {i}: valid row uses an image older than {MAX_IMAGE_AGE_S} s")
        if max(stamps) - min(stamps) > MAX_RGB_SKEW_S + TICK_TOL_S:
            cau.add(f"row {i}: valid row camera skew above {MAX_RGB_SKEW_S} s")

    rec = cats["recording"]
    paths = manifest.get("image_paths") or []
    if len(paths) != k or any(len(p) != n for p in paths):
        rec.add("image_paths do not list one image per camera per row")
    else:
        for slot, cam in enumerate(fmt["cameras"]):
            name, every, last, seen = cam["name"], cam["every"], None, set()
            lines = []
            sync_path = episode / f"camera/color/{name}/sync.txt"
            if sync_path.is_file():
                lines = sync_path.read_text().splitlines()
            captures = []
            for i in range(n):
                rel, stamp = paths[slot][i], times[4 + slot][i]
                if i % every == 0:
                    expected = f"camera/color/{name}/{grid[i]:.6f}.png"
                    if rel != expected or stamp != grid[i]:
                        rec.add(f"{name} row {i}: no fresh capture at the row time")
                    if rel in seen:
                        rec.add(f"{name} row {i}: repeated image file")
                    seen.add(rel)
                    captures.append(f"{grid[i]:.6f} {Path(rel).name}")
                    last = (rel, stamp)
                    file = episode / rel
                    if not file.is_file():
                        rec.add(f"{name} row {i}: missing {rel}")
                    elif cam["size"] is not None:
                        try:
                            if png_size(file) != cam["size"]:
                                rec.add(f"{name} row {i}: image size differs from config.json")
                        except ValueError as error:
                            rec.add(str(error))
                elif last is None or (rel, stamp) != last:
                    rec.add(f"{name} row {i}: held frame is not the previous capture")
            if lines != captures:
                rec.add(f"{name}: sync.txt does not list exactly the captures")

    if label_checks:
        lab = cats["labels"]
        base = {"puppet": "arm/endPose/puppet", "master": "arm/endPose/master",
                "measured": fmt["grip_measured"], "command": fmt["grip_command"]}
        try:
            rows = {key: [_row_json(episode / f"{d}/{t:.6f}.json") for t in grid] for key, d in base.items()}
        except (OSError, ValueError) as error:
            lab.add(f"per-row records unreadable: {error}")
            rows = None
        if rows is not None:
            for i in range(n):
                p, r = _pose_from_row(rows["puppet"][i])
                st = states[i]
                if max(abs(st[a] - p[a] * 100.0) for a in range(3)) > FLOAT64_TOL:
                    lab.add(f"row {i}: state position differs from the measured pose record")
                if angle_between(quat_xyzw_matrix(st[3:7]), r) > FLOAT64_TOL * 10:
                    lab.add(f"row {i}: state rotation differs from the measured pose record")
                if abs(st[7] - float(rows["measured"][i]["distance"]) / stroke) > FLOAT64_TOL:
                    lab.add(f"row {i}: state gripper differs from the measured gripper record")
                if i == n - 1:
                    continue
                p0, r0 = _pose_from_row(rows["master"][i])
                p1, r1 = _pose_from_row(rows["master"][i + 1])
                act = actions[i]
                if max(abs(act[a] - (p1[a] - p0[a]) * 100.0) for a in range(3)) > FLOAT32_TOL:
                    lab.add(f"row {i}: action translation is not next-minus-current command")
                dr = rotvec(matmul3(r1, transpose3(r0)))
                if max(abs(act[3 + a] - dr[a]) for a in range(3)) > FLOAT32_TOL:
                    lab.add(f"row {i}: action rotation is not next-minus-current command")
                if abs(act[6] - float(rows["command"][i + 1]["distance"]) / stroke) > FLOAT32_TOL:
                    lab.add(f"row {i}: action gripper is not the next commanded aperture")
                if len(lab.problems) >= MAX_PROBLEMS:
                    break
    return finish()
