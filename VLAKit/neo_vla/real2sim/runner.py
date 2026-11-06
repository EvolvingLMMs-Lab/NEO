#!/usr/bin/env python3
"""Rollout runner for the configurable SAPIEN route.

A build stages this file next to checks.py (and the reviewed controller plugin, if the task uses
one). The pipeline runs it in your simulator environment as a separate process:

    python -I runner.py probe --out DIR --renderer cpu
    python -I runner.py rollout --spec SPEC.json --out DIR

Isolated mode (-I) puts neither the script's directory nor PYTHONPATH on sys.path, so checks.py is
loaded from this file's own directory by explicit path, never by an import that something else
could satisfy; `runner.py --help` names the file and its sha256.

A rollout builds the scene from the reviewed specification, runs the controller at 30 Hz with the
physics at the declared rate, records a per-tick trace and the raw episode, and audits the
persisted trace with checks.audit_trace. The episode directory is finalized only when that audit
and the raw-episode checks pass; otherwise the partial recording is kept under failures/. The
pipeline's verify stage repeats every check on the persisted files.

Needs numpy, sapien (checked with 3.0.0b1), Pillow and six, which SAPIEN's URDF parser imports when
the first URDF loader is created; OpenCV only for cameras with lens distortion.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import time
import traceback
import types
import uuid
from collections import deque
from pathlib import Path

try:
    import numpy as np
except ImportError:  # reported by `probe`; every other command needs numpy
    np = None


def load_sibling(name: str) -> tuple:
    """(module, sha256) of <name>.py in this file's own directory, loaded by explicit path.

    Works the same when this file runs as a script under `python -I` (the staged runtime), as
    part of the package, or when a test loads it from a path. The bytes that are hashed are the
    bytes that are executed, and the module is not registered in sys.modules under an importable
    name, so no ambient module of the same name can stand in for the reviewed file.
    """
    path = Path(__file__).resolve().parent / f"{name}.py"
    source = path.read_bytes()
    module = types.ModuleType(f"real2sim_staged_{name}")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec", dont_inherit=True), module.__dict__)
    return module, hashlib.sha256(source).hexdigest()


checks, CHECKS_SHA256 = load_sibling("checks")

RUNNER_SCHEMA = "neo-vla-real2sim-runner/1"
SPEC_SCHEMA = "neo-vla-real2sim-rollout/1"
PLUGIN_API = 1
CONTROL_HZ = checks.CONTROL_HZ
# Optical camera C (right, down, forward) to SAPIEN camera entity S (forward, left, up).
T_C_S = ((0.0, -1.0, 0.0, 0.0), (0.0, 0.0, -1.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


class RolloutError(RuntimeError):
    """A rollout failure; the message and info are kept as failure evidence."""

    def __init__(self, message: str, info: dict | None = None) -> None:
        super().__init__(message)
        self.info = info or {}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as handle:
        handle.write(json.dumps(value, indent=1, allow_nan=False) + "\n")


def safe_json(value):
    """JSON-safe copy for failure reports: non-finite floats become null, unknown objects text."""
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if np is not None and isinstance(value, (np.generic, np.ndarray)):
        return safe_json(value.tolist())
    return str(value)


def look_at(position, target):
    """Optical camera pose (right, down, forward) at position looking at target, world +z up."""
    p, t = np.asarray(position, float), np.asarray(target, float)
    forward = (t - p) / np.linalg.norm(t - p)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    out = np.eye(4)
    out[:3, :3] = np.stack([right, np.cross(forward, right), forward], axis=1)
    out[:3, 3] = p
    return out


# ----------------------------------------------------------------------------- transforms
def inverse(t):
    t = np.asarray(t, float)
    out = np.eye(4)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -t[:3, :3].T @ t[:3, 3]
    return out


def translation(x: float, y: float, z: float):
    out = np.eye(4)
    out[:3, 3] = [x, y, z]
    return out


def quat_xyzw(r) -> list:
    r = np.asarray(r, float)
    tr = r[0, 0] + r[1, 1] + r[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        q = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        q = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        q = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
    n = math.sqrt(sum(v * v for v in q))
    q = [v / n for v in q]
    return [-v for v in q] if q[3] < 0 else q


def quat_matrix(q_xyzw):
    return np.asarray(checks.quat_xyzw_matrix(q_xyzw), float)


def rotvec_matrix(v):
    v = np.asarray(v, float)
    angle = float(np.linalg.norm(v))
    if angle < 1e-15:
        return np.eye(3)
    k = v / angle
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(angle) * kx + (1.0 - math.cos(angle)) * kx @ kx


def interpolate(a, b, fraction: float):
    """Pose between a and b: linear translation, rotation along the geodesic."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.eye(4)
    delta = np.asarray(checks.rotvec((b[:3, :3] @ a[:3, :3].T).tolist()), float)
    out[:3, :3] = rotvec_matrix(fraction * delta) @ a[:3, :3]
    out[:3, 3] = (1.0 - fraction) * a[:3, 3] + fraction * b[:3, 3]
    return out


def euler_zyx(r):
    r = np.asarray(r, float)
    yaw = math.atan2(r[1, 0], r[0, 0])
    pitch = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    roll = math.atan2(r[2, 1], r[2, 2])
    return yaw, pitch, roll


def camera_axes(t) -> list:
    t = np.asarray(t, float)
    return [*t[:3, 2].tolist(), *(-t[:3, 1]).tolist()]


def smoothstep(u: float) -> float:
    return u * u * (3.0 - 2.0 * u)


def rigid(value, name: str):
    t = np.asarray(value, float)
    if not checks.is_rigid(t.tolist()):
        raise RolloutError(f"{name}: not a finite proper rigid transform")
    return t


# ----------------------------------------------------------------------------- reused helpers
class Follower:
    """First-order follower with delay; the command (master) is never the filtered state."""

    def __init__(self, initial, alpha: float, delay_steps: int) -> None:
        self.value = np.array(initial, dtype=float)
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.delay = int(delay_steps)
        self.queue = deque([self.value.copy() for _ in range(self.delay + 1)], maxlen=self.delay + 1)

    def step(self, commanded):
        self.queue.append(np.asarray(commanded, float).copy())
        self.value += self.alpha * (self.queue[0] - self.value)
        return self.value.copy()


class LensWarp:
    """Ideal pinhole render -> Brown-distorted RGB, as the reference recordings store raw pixels."""

    def __init__(self, k, d, width: int, height: int) -> None:
        import cv2

        cv2.setNumThreads(1)
        k, d = np.asarray(k, float), np.asarray(d, float)
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        pixels = np.stack([u, v], axis=-1).astype(np.float32)
        ideal = cv2.undistortPoints(pixels.reshape(-1, 1, 2), k, d, P=k).reshape(height, width, 2)
        self.mapx = ideal[:, :, 0].astype(np.float32)
        self.mapy = ideal[:, :, 1].astype(np.float32)

    def __call__(self, image):
        import cv2

        return cv2.remap(image, self.mapx, self.mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def import_sapien(renderer: str, preload: list):
    """Import SAPIEN for the requested renderer; the CPU path insists on a lavapipe device."""
    if renderer == "cpu":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RolloutError("CPU rendering requires CUDA_VISIBLE_DEVICES set to an empty string")
        # SAPIEN 3.0's import helper shells out to nvidia-smi; skip it in this process only.
        sys.modules.setdefault("sapien._vulkan_tricks", types.ModuleType("sapien._vulkan_tricks"))
        icd = Path(os.environ.get("VK_ICD_FILENAMES", ""))
        try:
            library = json.loads(icd.read_text())["ICD"]["library_path"]
        except (OSError, KeyError, ValueError) as error:
            raise RolloutError(f"CPU rendering needs VK_ICD_FILENAMES pointing to a lavapipe ICD: {error}") from error
        if "lvp" not in str(library):
            raise RolloutError(f"VK_ICD_FILENAMES {icd} is not a lavapipe ICD")
        os.environ["VK_DRIVER_FILES"] = str(icd)
    if preload:
        import ctypes

        for library in preload:
            ctypes.CDLL(str(library))
    import sapien

    if renderer == "cpu":
        def forbidden(*args, **kwargs):
            raise RolloutError("GPU PhysX is not used on the CPU rendering path")

        sapien.physx.PhysxGpuSystem = forbidden
    return sapien


# ----------------------------------------------------------------------------- simulator
class Sim:
    """SAPIEN scene, robot and cameras built from a resolved, reviewed scene specification."""

    def __init__(self, scene: dict, renderer: str, preload: list, contact_eps: float) -> None:
        self.s = sapien = import_sapien(renderer, preload)
        self.spec = scene
        physics = scene["physics"]
        self.hz = int(physics["physics_hz"])
        if self.hz % CONTROL_HZ:
            raise RolloutError(f"physics_hz {self.hz} is not a multiple of {CONTROL_HZ}")
        self.substeps = self.hz // CONTROL_HZ
        self.contact_eps = float(contact_eps)
        sapien.physx.set_scene_config(gravity=[float(g) for g in physics["gravity_m_s2"]], cpu_workers=1)
        systems = [sapien.physx.PhysxCpuSystem()]
        self.render = renderer in ("cpu", "gpu")
        self.device = None
        self.materials = None
        if self.render:
            sapien.render.set_camera_shader_dir("default")
            system = sapien.render.RenderSystem() if renderer == "cpu" else sapien.render.RenderSystem("cuda:0")
            device = system.device
            self.device = str(device.name)
            if renderer == "cpu" and (device.cuda_id >= 0 or not any(
                    x in self.device.lower() for x in ("llvmpipe", "lavapipe"))):
                raise RolloutError(f"CPU rendering refused device {self.device}")
            systems.append(system)
            self.materials = sapien.SapienRenderer()
        self.scene = sapien.Scene(systems)
        self.scene.set_timestep(1.0 / self.hz)
        self.tick = 0
        self.trace = {k: [] for k in checks.TRACE_KEYS}
        self._lighting()
        self._table()
        self._robot()
        self._objects()
        self._cameras()

    # -- construction
    def pose(self, t):
        t = np.asarray(t, float)
        q = quat_xyzw(t[:3, :3])
        return self.s.Pose(t[:3, 3].tolist(), [q[3], q[0], q[1], q[2]])

    @staticmethod
    def matrix(pose):
        w, x, y, z = (float(v) for v in pose.q)
        out = np.eye(4)
        out[:3, :3] = quat_matrix([x, y, z, w])
        out[:3, 3] = np.asarray(pose.p, float)
        return out

    def material(self, color, texture=None):
        if self.materials is None:
            return None
        color = [float(c) for c in color]
        if len(color) == 3:
            color.append(1.0)
        m = self.materials.create_material()
        m.set_base_color(color)
        m.set_roughness(0.65)
        m.metallic = 0.0
        m.specular = 0.0
        if texture:
            m.base_color_texture = self.s.render.RenderTexture2D(str(texture))
        return m

    def _lighting(self) -> None:
        if not self.render:
            return
        light = self.spec.get("lighting") or {}
        self.scene.set_ambient_light([float(v) for v in light.get("ambient_rgb", [0.5, 0.5, 0.5])])
        for d in light.get("directional", []):
            self.scene.add_directional_light([float(v) for v in d["direction"]], [float(v) for v in d["rgb"]],
                                             shadow=bool(d.get("shadow", False)))

    def _table(self) -> None:
        t = self.spec["table"]
        length, width, thick = (float(v) for v in t["size_m"])
        mu = float(t["friction"])
        b = self.scene.create_actor_builder()
        half = [length / 2.0, width / 2.0, thick / 2.0]
        b.add_box_collision(half_size=half, material=self.scene.create_physical_material(mu, mu, 0.0))
        if self.render:
            b.add_box_visual(half_size=half, material=self.material(t.get("color_rgba", [0.4, 0.4, 0.4, 1.0])))
        self.table = b.build_static(name="table")
        yaw = float(t["yaw_rad"])
        pose = np.eye(4)
        pose[:3, :3] = rotvec_matrix([0.0, 0.0, yaw])
        pose[:3, 3] = [float(t["center_xy_m"][0]), float(t["center_xy_m"][1]), float(t["top_z_m"]) - thick / 2.0]
        self.table.set_pose(self.pose(pose))

    def _robot(self) -> None:
        r = self.spec["robot"]
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot = loader.load(str(r["urdf"]))
        if self.robot is None:
            raise RolloutError(f"URDF loading failed: {r['urdf']}")
        self.T_W_B = rigid(r["base_pose_W"], "robot.base_pose_W")
        self.T_B_W = inverse(self.T_W_B)
        self.robot.set_root_pose(self.pose(self.T_W_B))
        self.joints = self.robot.get_active_joints()
        names = [j.get_name() for j in self.joints]
        self.arm_names = list(r["arm_joints"])
        grip = r["gripper"]
        try:
            self.arm_idx = [names.index(n) for n in self.arm_names]
            self.grip_idx = [names.index(g["name"]) for g in grip["joints"]]
        except ValueError as error:
            raise RolloutError(f"declared joint not among the URDF's active joints {names}: {error}") from error
        extra = sorted(set(names) - set(self.arm_names) - {g["name"] for g in grip["joints"]})
        if extra:
            raise RolloutError(f"active URDF joints not declared as arm or gripper joints: {extra}")
        self.grip_scales = [float(g["scale"]) for g in grip["joints"]]
        self.grip_offset = float(grip.get("aperture_offset_m", 0.0))
        self.stroke = float(grip["stroke_m"])
        self.limits = {k: (float(v[0]), float(v[1])) for k, v in r["joint_limits"].items()}
        init_q = np.asarray(r["init_arm_q"], float)
        self.init_aperture = float(grip["init_aperture_m"])
        q = np.zeros(len(names))
        q[self.arm_idx] = init_q
        q[self.grip_idx] = self.grip_targets(self.init_aperture)
        self.robot.set_qpos(q)
        drive = r["drive"]
        for i, joint in enumerate(self.joints):
            d = drive["gripper"] if i in self.grip_idx else drive["arm"]
            joint.set_drive_property(float(d["stiffness"]), float(d["damping"]), force_limit=float(d["force_limit"]))
            joint.set_drive_target(float(q[i]))
        self.gravity_compensation = bool(drive.get("gravity_compensation", True))
        self.max_speed = float(drive["max_joint_speed_rad_s"])
        follower = r.get("follower") or {"alpha": 1.0, "delay_steps": 0}
        self.follower = Follower(init_q, float(follower["alpha"]), int(follower["delay_steps"]))
        self.links = self.robot.get_links()
        self.link_names = [link.get_name() for link in self.links]
        if r["ee_link"] not in self.link_names:
            raise RolloutError(f"ee_link {r['ee_link']} is not a URDF link")
        self.ee_index = self.link_names.index(r["ee_link"])
        self.ee_link = self.links[self.ee_index]
        self.T_E_TCP = rigid(r["tcp_in_ee"], "robot.tcp_in_ee")
        self.T_TCP_E = inverse(self.T_E_TCP)
        self.pm = self.robot.create_pinocchio_model()
        self.command_q = init_q.copy()
        self.command_aperture = self.init_aperture
        self.drive_q = init_q.copy()

    def _objects(self) -> None:
        self.objects = []
        cylinder_x_to_z = self.s.Pose([0.0, 0.0, 0.0], [math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0])
        for o in self.spec["objects"]:
            name, shape = o["name"], o["shape"]
            mass, mu, rest = float(o["mass_kg"]), float(o["friction"]), float(o["restitution"])
            b = self.scene.create_actor_builder()
            pm = self.scene.create_physical_material(mu, mu, rest)
            vis = self.material(o.get("color_rgba", [0.7, 0.7, 0.7, 1.0]), o.get("texture"))
            kind, com = shape["type"], [0.0, 0.0, 0.0]
            if kind == "box":
                size = [float(v) for v in shape["size_m"]]
                half = [v / 2.0 for v in size]
                b.add_box_collision(half_size=half, material=pm, density=mass / (size[0] * size[1] * size[2]))
                if self.render:
                    b.add_box_visual(half_size=half, material=vis)
            elif kind == "sphere":
                rad = float(shape["radius_m"])
                b.add_sphere_collision(radius=rad, material=pm, density=mass / (4.0 / 3.0 * math.pi * rad ** 3))
                if self.render:
                    b.add_sphere_visual(radius=rad, material=vis)
            elif kind == "cylinder":
                rad, length = float(shape["radius_m"]), float(shape["length_m"])
                b.add_cylinder_collision(pose=cylinder_x_to_z, radius=rad, half_length=length / 2.0, material=pm,
                                         density=mass / (math.pi * rad * rad * length))
                if self.render:
                    b.add_cylinder_visual(pose=cylinder_x_to_z, radius=rad, half_length=length / 2.0, material=vis)
            elif kind == "mesh":
                scale = [float(v) for v in shape.get("scale", [1.0, 1.0, 1.0])]
                if shape.get("convex") == "multiple":
                    b.add_multiple_convex_collisions_from_file(filename=str(shape["collision"]), scale=scale,
                                                               material=pm)
                else:
                    b.add_convex_collision_from_file(filename=str(shape["collision"]), scale=scale, material=pm)
                if self.render:
                    visual = shape.get("visual")
                    b.add_visual_from_file(filename=str(visual or shape["collision"]), scale=scale,
                                           material=None if visual else vis)
                com = [float(v) for v in shape["com_m"]]
                b.set_mass_and_inertia(mass, self.s.Pose(com), [float(v) for v in shape["inertia_diag_kg_m2"]])
            else:
                raise RolloutError(f"{name}: unsupported shape {kind!r}")
            dynamic = bool(o.get("dynamic", True))
            entity = b.build(name=name) if dynamic else b.build_static(name=name)
            body = entity.find_component_by_type(self.s.physx.PhysxRigidDynamicComponent) if dynamic else None
            self.objects.append({"name": name, "entity": entity, "body": body, "dynamic": dynamic, "com": com})

    def _cameras(self) -> None:
        self.cameras = []
        for c in self.spec["cameras"]:
            w, h = int(c["width"]), int(c["height"])
            k = [[float(v) for v in row] for row in c["K"]]
            near, far = float(c["near_m"]), float(c["far_m"])
            component = None
            if self.render:
                component = self.scene.add_camera(c["name"], w, h, 1.0, near, far)
                component.set_perspective_parameters(near, far, k[0][0], k[1][1], k[0][2], k[1][2], 0.0)
            mount, link = c["mount"], None
            if mount != "world":
                link_name = mount.split(":", 1)[1]
                if link_name not in self.link_names:
                    raise RolloutError(f"camera {c['name']}: mount link {link_name} is not a URDF link")
                link = self.links[self.link_names.index(link_name)]
            d = [float(v) for v in c.get("D") or []]
            warp = LensWarp(k, d, w, h) if self.render and any(d) else None
            self.cameras.append({"name": c["name"], "component": component, "pose": rigid(c["pose"], c["name"]),
                                 "link": link, "every": int(c.get("every_n_rows", 1)), "warp": warp,
                                 "K": k, "size": (w, h)})

    # -- state
    def grip_targets(self, aperture: float) -> list:
        n = len(self.grip_scales)
        return [(float(aperture) - self.grip_offset) / (n * s) for s in self.grip_scales]

    def qpos(self):
        return np.asarray(self.robot.get_qpos(), float)

    def aperture(self) -> float:
        q = self.qpos()
        return self.grip_offset + sum(s * q[i] for s, i in zip(self.grip_scales, self.grip_idx))

    def ee_world(self):
        return self.matrix(self.ee_link.get_pose())

    def tcp_base(self):
        return self.T_B_W @ self.ee_world() @ self.T_E_TCP

    def fk_ee(self, arm_q):
        full = self.qpos()
        full[self.arm_idx] = np.asarray(arm_q, float)
        self.pm.compute_forward_kinematics(full)
        return self.matrix(self.pm.get_link_pose(self.ee_index))

    def fk(self, arm_q) -> list:
        return (self.fk_ee(arm_q) @ self.T_E_TCP).tolist()

    def ik(self, tcp_base, seed, pos_tol: float, rot_tol: float):
        target = np.asarray(tcp_base, float) @ self.T_TCP_E
        full = self.qpos()
        full[self.arm_idx] = np.asarray(seed, float)
        mask = np.zeros(len(full), dtype=np.int32)
        mask[self.arm_idx] = 1
        result, solver_ok, _ = self.pm.compute_inverse_kinematics(
            self.ee_index, self.pose(target), initial_qpos=full, active_qmask=mask, eps=1e-5, max_iterations=1000)
        q = np.asarray(result, float)[self.arm_idx]
        got = self.fk_ee(q)
        info = {"solver_ok": bool(solver_ok),
                "position_error_m": float(np.linalg.norm(got[:3, 3] - target[:3, 3])),
                "rotation_error_rad": checks.angle_between(got[:3, :3].tolist(), target[:3, :3].tolist()),
                "within_limits": all(self.limits[n][0] - 1e-9 <= float(v) <= self.limits[n][1] + 1e-9
                                     for n, v in zip(self.arm_names, q))}
        ok = (info["position_error_m"] <= pos_tol and info["rotation_error_rad"] <= rot_tol
              and info["within_limits"] and all(math.isfinite(v) for v in q))
        return (q.tolist() if ok else None), info

    @staticmethod
    def velocity(body) -> tuple:
        if body is None:
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        lin = body.get_linear_velocity() if hasattr(body, "get_linear_velocity") else body.linear_velocity
        ang = body.get_angular_velocity() if hasattr(body, "get_angular_velocity") else body.angular_velocity
        return [float(v) for v in lin], [float(v) for v in ang]

    def object_state(self) -> dict:
        out = {}
        for o in self.objects:
            lin, ang = self.velocity(o["body"])
            out[o["name"]] = {"pose_W": self.matrix(o["entity"].get_pose()).tolist(), "lin_vel": lin, "ang_vel": ang}
        return out

    def contact_flags(self, valid: bool) -> list:
        names = [o["name"] for o in self.objects]
        m = len(names)
        flags = [[False] * (m + 2) for _ in range(m)]
        if not valid:
            return flags
        for contact in self.scene.get_contacts():
            if not any(float(np.linalg.norm(p.impulse)) > self.contact_eps for p in contact.points):
                continue
            a, b = (body.entity.name for body in contact.bodies)
            for x, y in ((a, b), (b, a)):
                if x not in names:
                    continue
                if y in names:
                    col = names.index(y)
                elif y == "table":
                    col = m
                elif y in self.link_names:
                    col = m + 1
                else:
                    continue
                flags[names.index(x)][col] = True
        return flags

    def snapshot(self, phase: str) -> None:
        state = self.object_state()
        q = self.qpos()
        t = self.trace
        t["tick"].append(self.tick)
        t["time_s"].append(self.tick / self.hz)
        t["phase"].append(phase)
        t["obj_pose"].append([state[o["name"]]["pose_W"] for o in self.objects])
        t["obj_lin_vel"].append([state[o["name"]]["lin_vel"] for o in self.objects])
        t["obj_ang_vel"].append([state[o["name"]]["ang_vel"] for o in self.objects])
        t["ee_pose"].append(self.ee_world().tolist())
        t["arm_q"].append(q[self.arm_idx].tolist())
        t["arm_q_target"].append(self.drive_q.tolist())
        t["gripper_m"].append(self.aperture())
        t["gripper_cmd_m"].append(self.command_aperture)
        t["contacts"].append(self.contact_flags(self.tick >= 1))

    def control_step(self, arm_q, aperture: float, phase: str) -> None:
        self.command_q = np.asarray(arm_q, float).copy()
        self.command_aperture = float(aperture)
        self.drive_q = self.follower.step(self.command_q)
        for index, value in zip(self.arm_idx, self.drive_q):
            self.joints[index].set_drive_target(float(value))
        for index, value in zip(self.grip_idx, self.grip_targets(aperture)):
            self.joints[index].set_drive_target(float(value))
        for _ in range(self.substeps):
            if self.gravity_compensation:
                self.robot.set_qf(self.robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
            self.scene.step()
            self.tick += 1
            self.snapshot(phase)

    def place(self, layout: dict) -> None:
        for o in self.objects:
            o["entity"].set_pose(self.pose(rigid(layout["objects"][o["name"]], o["name"])))

    def camera_world(self, cam):
        return cam["pose"] if cam["link"] is None else self.matrix(cam["link"].get_pose()) @ cam["pose"]

    def capture(self, names: set) -> tuple:
        poses = {}
        for cam in self.cameras:
            poses[cam["name"]] = self.camera_world(cam)
            cam["component"].entity.set_pose(self.pose(poses[cam["name"]] @ np.asarray(T_C_S)))
        self.scene.update_render()
        frames = {}
        for cam in self.cameras:
            if cam["name"] not in names:
                continue
            cam["component"].take_picture()
            rgb = np.clip(cam["component"].get_picture("Color")[:, :, :3] * 255, 0, 255).astype(np.uint8)
            frames[cam["name"]] = cam["warp"](rgb) if cam["warp"] else rgb
        return frames, poses

    def camera_readback(self) -> dict:
        out = {}
        for cam in self.cameras:
            c = cam["component"]
            out[cam["name"]] = {"intrinsic_matrix": np.asarray(c.get_intrinsic_matrix(), float).tolist(),
                                "fx_fy_cx_cy": [float(c.fx), float(c.fy), float(c.cx), float(c.cy)],
                                "width_height": [int(c.width), int(c.height)],
                                "model_matrix": np.asarray(c.get_model_matrix(), float).tolist()}
        return out

    def trace_arrays(self) -> dict:
        t = self.trace
        return {"tick": np.asarray(t["tick"], np.int64), "time_s": np.asarray(t["time_s"], np.float64),
                "phase": np.asarray(t["phase"], dtype=str), "obj_pose": np.asarray(t["obj_pose"], np.float64),
                "obj_lin_vel": np.asarray(t["obj_lin_vel"], np.float64),
                "obj_ang_vel": np.asarray(t["obj_ang_vel"], np.float64),
                "ee_pose": np.asarray(t["ee_pose"], np.float64), "arm_q": np.asarray(t["arm_q"], np.float64),
                "arm_q_target": np.asarray(t["arm_q_target"], np.float64),
                "gripper_m": np.asarray(t["gripper_m"], np.float64),
                "gripper_cmd_m": np.asarray(t["gripper_cmd_m"], np.float64),
                "contacts": np.asarray(t["contacts"], dtype=bool).reshape(len(t["tick"]), len(self.objects),
                                                                         len(self.objects) + 2)}


# ----------------------------------------------------------------------------- controllers
class ControllerError(RolloutError):
    pass


class JointWaypoints:
    """Smoothstep moves between reviewed joint-space waypoints (for example a measured real demo)."""

    def __init__(self, ctx: dict) -> None:
        p = ctx["params"]
        q, g = list(ctx["init_arm_q"]), float(ctx["init_gripper_m"])
        self.queue: list = []
        for wp in p["waypoints"]:
            n = max(1, math.ceil(float(wp["duration_s"]) * CONTROL_HZ - 1e-9))
            target, gt = [float(v) for v in wp["arm_q"]], float(wp.get("gripper_m", g))
            for k in range(1, n + 1):
                s = smoothstep(k / n)
                self.queue.append(([a + s * (b - a) for a, b in zip(q, target)], g + s * (gt - g),
                                   str(wp.get("phase", "waypoint"))))
            q, g = target, gt
        for _ in range(math.ceil(float(p.get("final_hold_s", 0.0)) * CONTROL_HZ - 1e-9)):
            self.queue.append((q, g, "hold"))
        self.last = (q, g)

    def act(self, obs: dict) -> dict:
        if not self.queue:
            return {"arm_q": self.last[0], "gripper_m": self.last[1], "phase": "hold", "done": True}
        q, g, phase = self.queue.pop(0)
        return {"arm_q": q, "gripper_m": g, "phase": phase, "done": not self.queue}


class PickPlace:
    """Scripted grasp-and-place expert with IK, straight-line approach/lift/lower segments and
    placement planned from the live object-in-hand pose."""

    STAGES = ("open", "approach", "descend", "close", "lift", "carry", "lower", "release", "retreat", "hold")

    def __init__(self, ctx: dict) -> None:
        self.p = p = ctx["params"]
        self.ik, self.fk = ctx["ik"], ctx["fk"]
        self.T_W_B = np.asarray(ctx["base_pose_W"], float)
        self.T_B_W = inverse(self.T_W_B)
        self.q = np.asarray(ctx["init_arm_q"], float)
        self.g = float(ctx["init_gripper_m"])
        self.vmax = min(float(p["max_joint_speed_rad_s"]), float(ctx["robot_max_joint_speed_rad_s"]))
        self.max_change = float(p["ik"]["max_joint_change_rad"])
        self.queue: list = []
        self.stage = 0
        self.log: list = []

    def _obj(self, obs: dict, name: str):
        return self.T_B_W @ np.asarray(obs["objects"][name]["pose_W"], float)

    def _raise(self, t_b, height: float):
        t_w = self.T_W_B @ np.asarray(t_b, float)
        t_w[2, 3] += float(height)
        return self.T_B_W @ t_w

    def _solve(self, target, seed, label: str):
        q, info = self.ik(np.asarray(target).tolist(), np.asarray(seed).tolist())
        self.log.append({"segment": label, **info})
        if q is None:
            raise ControllerError(f"{label}: no inverse-kinematics solution within tolerance", info)
        q = np.asarray(q, float)
        if float(np.max(np.abs(q - np.asarray(seed, float)))) > self.max_change:
            raise ControllerError(f"{label}: IK solution jumps more than {self.max_change} rad (branch change)")
        return q

    def _timed(self, path: list, phase: str) -> None:
        path = [np.asarray(q, float) for q in path]
        seg = [float(np.max(np.abs(b - a))) for a, b in zip(path[:-1], path[1:])]
        total = sum(seg)
        duration = max(float(self.p["segment_min_s"]), 1.5 * total / self.vmax)
        n = max(1, math.ceil(duration * CONTROL_HZ - 1e-9))
        cum = [0.0]
        for d in seg:
            cum.append(cum[-1] + d)
        for k in range(1, n + 1):
            s = smoothstep(k / n) * total
            j = 0
            while j < len(seg) - 1 and s > cum[j + 1]:
                j += 1
            frac = 0.0 if seg[j] == 0.0 else min(1.0, max(0.0, (s - cum[j]) / seg[j]))
            q = path[j] + frac * (path[j + 1] - path[j])
            self.queue.append((q.tolist(), self.g, phase))
        self.q = path[-1]

    def _cartesian(self, a, b, phase: str) -> None:
        pieces = int(self.p["cartesian_pieces"])
        path = [self.q]
        for i in range(1, pieces + 1):
            path.append(self._solve(interpolate(a, b, i / pieces), path[-1], f"{phase} {i}/{pieces}"))
        self._timed(path, phase)

    def _gripper(self, target: float, wait_s: float, phase: str) -> None:
        self.g = float(target)
        for _ in range(max(1, math.ceil(float(wait_s) * CONTROL_HZ - 1e-9))):
            self.queue.append((self.q.tolist(), self.g, phase))

    def _plan(self, stage: str, obs: dict) -> None:
        p = self.p
        if stage == "open":
            self._gripper(p["gripper_open_m"], p["open_wait_s"], "open")
        elif stage == "approach":
            self.grasp = self._obj(obs, p["object"]) @ np.asarray(p["grasp_in_object"], float)
            self.pre = self.grasp @ translation(0.0, 0.0, -float(p["approach_distance_m"]))
            self._timed([self.q, self._solve(self.pre, self.q, "approach")], "approach")
        elif stage == "descend":
            self._cartesian(self.pre, self.grasp, "descend")
        elif stage == "close":
            self._gripper(p["gripper_close_m"], p["close_wait_s"], "close")
        elif stage == "lift":
            self.lifted = self._raise(self.grasp, p["lift_height_m"])
            self._cartesian(self.grasp, self.lifted, "lift")
        elif stage == "carry":
            in_hand = inverse(np.asarray(obs["tcp_pose_B"], float)) @ self._obj(obs, p["object"])
            ref = p["place"].get("relative_to", "world")
            frame = self.T_B_W if ref == "world" else self._obj(obs, ref)
            target = frame @ np.asarray(p["place"]["object_in_reference"], float)
            self.place = target @ inverse(in_hand)
            self.preplace = self._raise(self.place, p["lift_height_m"])
            self._timed([self.q, self._solve(self.preplace, self.q, "carry")], "carry")
        elif stage == "lower":
            self._cartesian(self.preplace, self.place, "lower")
        elif stage == "release":
            self._gripper(p["gripper_open_m"], p["open_wait_s"], "release")
        elif stage == "retreat":
            self._cartesian(self.place, self._raise(self.place, p["retreat_height_m"]), "retreat")
        else:
            for _ in range(max(1, math.ceil(float(p["final_hold_s"]) * CONTROL_HZ - 1e-9))):
                self.queue.append((self.q.tolist(), self.g, "hold"))

    def act(self, obs: dict) -> dict:
        while not self.queue and self.stage < len(self.STAGES):
            self._plan(self.STAGES[self.stage], obs)
            self.stage += 1
        if not self.queue:
            return {"arm_q": self.q.tolist(), "gripper_m": self.g, "phase": "hold", "done": True}
        q, g, phase = self.queue.pop(0)
        return {"arm_q": q, "gripper_m": g, "phase": phase,
                "done": not self.queue and self.stage >= len(self.STAGES)}


def load_plugin(path: Path, sha256: str, class_name: str):
    """The reviewed controller class. The plugin is loaded by explicit path, and the bytes checked
    against the reviewed sha256 are the bytes executed (the file is not read a second time)."""
    data = Path(path).read_bytes()
    if sha256_bytes(data) != sha256:
        raise RolloutError("controller plugin bytes differ from the reviewed sha256")
    module = types.ModuleType("real2sim_controller_plugin")
    module.__file__ = str(path)
    exec(compile(data, str(path), "exec", dont_inherit=True), module.__dict__)
    if getattr(module, "API_VERSION", None) != PLUGIN_API:
        raise RolloutError(f"plugin API_VERSION must be {PLUGIN_API}")
    controller = getattr(module, class_name, None)
    if controller is None:
        raise RolloutError(f"plugin has no class {class_name}")
    return controller


def check_command(cmd, sim: Sim, previous) -> tuple:
    if not isinstance(cmd, dict):
        raise RolloutError("controller returned a non-dict command")
    arm, gap = cmd.get("arm_q"), cmd.get("gripper_m")
    phase, done = cmd.get("phase"), cmd.get("done")
    if not isinstance(arm, (list, tuple)) or len(arm) != len(sim.arm_names) or not checks.finite(arm):
        raise RolloutError("command arm_q must list one finite value per arm joint")
    for name, value in zip(sim.arm_names, arm):
        lo, hi = sim.limits[name]
        if not lo - 1e-9 <= float(value) <= hi + 1e-9:
            raise RolloutError(f"command {name}={value} outside the URDF limits [{lo}, {hi}]")
    step = max(abs(float(a) - float(b)) for a, b in zip(arm, previous))
    if step > sim.max_speed / CONTROL_HZ + 1e-9:
        raise RolloutError(f"command moves a joint {step:.4g} rad in one step (limit "
                           f"{sim.max_speed / CONTROL_HZ:.4g} rad = max_joint_speed / {CONTROL_HZ} Hz)")
    if not checks.finite(gap) or not 0.0 <= float(gap) <= sim.stroke:
        raise RolloutError(f"command gripper_m {gap!r} outside [0, {sim.stroke}]")
    if not isinstance(phase, str) or not phase or len(phase) > 64:
        raise RolloutError("command phase must be a short non-empty string")
    if not isinstance(done, bool):
        raise RolloutError("command done must be a boolean")
    return [float(v) for v in arm], float(gap), phase, done


# ----------------------------------------------------------------------------- raw episode writer
class EpisodeWriter:
    """Raw episode writer (the reference recorder layout, generalized to n joints and k cameras).

    State: measured TCP in the robot base frame [xyz cm, quaternion xyzw, measured aperture /
    stroke]. Action: next-minus-current commanded TCP [dxyz cm, spatial rotvec rad, next commanded
    aperture / stroke]. Images are causal: a row uses the latest capture at or before its time.
    """

    def __init__(self, path: Path, *, cameras: list, stroke: float, joint_names: list, instruction: str,
                 calibration: dict, metadata: dict) -> None:
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(self.path)
        self.staging = self.path.parent / f".{self.path.name}.partial-{uuid.uuid4().hex[:8]}"
        self.staging.mkdir(parents=True)
        self.cameras, self.stroke = cameras, float(stroke)
        self.joint_names, self.instruction = list(joint_names), instruction
        self.metadata = dict(metadata)
        self.rows: list = []
        self.latest: dict = {}
        self.closed = False
        self.capture_index = {c["name"]: [] for c in cameras}
        for name, value in calibration.items():
            write_json(self.staging / "calibration" / name, value)
        for c in cameras:
            d = list(c.get("D") or [])
            write_json(self.staging / f"camera/color/{c['name']}/config.json", {
                "K": [float(c["K"][i][j]) for i in range(3) for j in range(3)], "D": d,
                "width": int(c["width"]), "height": int(c["height"]),
                "distortion_model": "brown_conrady" if any(d) else "none",
                "source": "reviewed scene specification", "domain": "sim",
                "pixels": "SAPIEN render, Brown-distorted" if any(d) else "SAPIEN render, ideal pinhole"})

    def append(self, *, time_s: float, measured, command, measured_gap: float, command_gap: float,
               measured_joints, command_joints, frames: dict, camera_poses: dict, valid: bool = True) -> None:
        from PIL import Image

        if self.closed:
            raise RuntimeError("episode already finished")
        t = float(time_s)
        if not math.isfinite(t):
            raise ValueError("non-finite row time")
        if self.rows and abs(t - self.rows[-1]["t"] - 1.0 / CONTROL_HZ) > checks.GRID_TOL_S:
            raise ValueError(f"rows must be on a monotonic {CONTROL_HZ} Hz grid")
        measured, command = rigid(measured, "measured TCP"), rigid(command, "commanded TCP")
        qm, qc = np.asarray(measured_joints, float), np.asarray(command_joints, float)
        if qm.shape != (len(self.joint_names),) or qc.shape != qm.shape or not np.isfinite(np.r_[qm, qc]).all():
            raise ValueError("finite measured and commanded joints required for every arm joint")
        if not 0.0 <= measured_gap <= self.stroke or not 0.0 <= command_gap <= self.stroke:
            raise ValueError("gripper apertures must be in metres within the stroke")
        sizes = {c["name"]: (int(c["height"]), int(c["width"])) for c in self.cameras}
        for name, rgb in frames.items():
            rgb = np.asarray(rgb)
            if name not in sizes or rgb.shape != (*sizes[name], 3) or rgb.dtype != np.uint8:
                raise ValueError(f"{name}: expected RGB uint8 {sizes.get(name)}x3")
            rel = f"camera/color/{name}/{t:.6f}.png"
            Image.fromarray(rgb).save(self.staging / rel, compress_level=1)
            self.latest[name] = {"path": rel, "time": t, "axes": camera_axes(camera_poses[name])}
            self.capture_index[name].append([t, rel])
        if any(c["name"] not in self.latest for c in self.cameras):
            raise ValueError("every camera needs a frame on the first row")
        stamps = [self.latest[c["name"]]["time"] for c in self.cameras]
        camera_valid = (all(t - s <= checks.MAX_IMAGE_AGE_S + 1e-9 for s in stamps)
                        and max(stamps) - min(stamps) <= checks.MAX_RGB_SKEW_S + 1e-9)
        self.rows.append({"t": t, "measured": measured, "command": command, "gap": float(measured_gap),
                          "command_gap": float(command_gap), "qm": qm, "qc": qc,
                          "images": [dict(self.latest[c["name"]]) for c in self.cameras],
                          "valid": bool(valid and camera_valid)})
        for role, tf in (("puppet", measured), ("master", command)):
            yaw, pitch, roll = euler_zyx(tf[:3, :3])
            qx, qy, qz, qw = quat_xyzw(tf[:3, :3])
            write_json(self.staging / f"arm/endPose/{role}/{t:.6f}.json", {
                "x": float(tf[0, 3]), "y": float(tf[1, 3]), "z": float(tf[2, 3]), "yaw": yaw, "pitch": pitch,
                "roll": roll, "qx": qx, "qy": qy, "qz": qz, "qw": qw})
        for name, gap in (("measured", measured_gap), ("command", command_gap)):
            write_json(self.staging / f"gripper/encoder/{name}/{t:.6f}.json", {"distance": float(gap)})

    def finish(self, *, success: bool) -> Path:
        if self.closed:
            raise RuntimeError("episode already finished")
        rows, n, k = self.rows, len(self.rows), len(self.cameras)
        if n < 2:
            raise ValueError("at least two rows required")
        grid = np.array([r["t"] for r in rows])
        valid = np.array([r["valid"] for r in rows], bool)
        valid[:-1] &= valid[1:]
        valid[-1] = False
        states = np.array([[*(r["measured"][:3, 3] * 100.0), *quat_xyzw(r["measured"][:3, :3]),
                            r["gap"] / self.stroke] for r in rows], dtype=np.float64)
        actions = np.zeros((n, 7), np.float32)
        for i, (a, b) in enumerate(zip(rows[:-1], rows[1:])):
            actions[i, :3] = (b["command"][:3, 3] - a["command"][:3, 3]) * 100.0
            actions[i, 3:6] = checks.rotvec((b["command"][:3, :3] @ a["command"][:3, :3].T).tolist())
            actions[i, 6] = b["command_gap"] / self.stroke
        times = np.tile(grid, (4 + k, 1))
        extrinsics = np.full((n, k, 6), np.nan, np.float64)
        for slot in range(k):
            times[4 + slot] = [r["images"][slot]["time"] for r in rows]
            extrinsics[:, slot] = [r["images"][slot]["axes"] for r in rows]
        intrinsics = np.array([[c["K"][0][0], c["K"][1][1], c["K"][0][2], c["K"][1][2], c["width"], c["height"]]
                               for c in self.cameras], dtype=np.float64)
        np.savez_compressed(self.staging / "arrays.npz", states=states, actions=actions, valid=valid,
                            times=times, intrinsics=intrinsics)
        np.savez_compressed(self.staging / "camera_geometry.npz", camera_extrinsics=extrinsics,
                            rgb_times_s=times[4:])
        np.savez_compressed(self.staging / "joint_states.npz", time_s=grid,
                            measured_joints_rad=np.array([r["qm"] for r in rows]),
                            commanded_joints_rad=np.array([r["qc"] for r in rows]), joint_valid=valid)
        verified = bool(success)
        sync = {"schema": "neo_vla_raw_sync_v1", "hz": CONTROL_HZ, "grid_times_s": grid.tolist(), "rows": n,
                "valid_transitions": valid.tolist(), "valid_count": int(valid.sum()),
                "required_rgb": [c["name"] for c in self.cameras],
                "image_selection": "latest captured frame at or before the row time",
                "max_image_age_s": checks.MAX_IMAGE_AGE_S, "max_rgb_skew_s": checks.MAX_RGB_SKEW_S,
                "command_source": "commanded joint targets before the follower filter (forward kinematics)",
                "measured_state_source": "articulation feedback at the row instant",
                "task_success_verified": verified, "physical_exposure_sync_verified": False,
                "source_unique_counts": {name: len(v) for name, v in self.capture_index.items()}}
        manifest = {
            "format": checks.RAW_FORMAT, "episode": self.path.name, "rows": n, "domain": "sim",
            "task": self.instruction, "control_hz": CONTROL_HZ,
            "states": "measured TCP in the robot base frame: xyz cm, quaternion xyzw, aperture / stroke",
            "labels": "next-minus-current commanded TCP: dxyz cm, spatial rotvec rad, next commanded aperture / stroke",
            "gripper_stroke_m": self.stroke, "joint_names": self.joint_names,
            "cameras": [{"name": c["name"], "slot": i, "width": int(c["width"]), "height": int(c["height"]),
                         "K": c["K"], "D": list(c.get("D") or []), "every_n_rows": int(c.get("every_n_rows", 1)),
                         "mount": c["mount"]} for i, c in enumerate(self.cameras)],
            "camera_geometry": {"frame": "robot_base", "channels": "forward(+z), up(-y) of the optical frame"},
            "image_paths": [[r["images"][slot]["path"] for r in rows] for slot in range(k)],
            "valid_transitions": valid.tolist(), "training_sync": sync,
            "training_admission": {"task_success_verified": verified, "accepted": False,
                                   "note": "admission is decided downstream from verification and export checks"},
            "simulation": self.metadata,
            "input_sha256": {name: hashlib.sha256((self.staging / name).read_bytes()).hexdigest()
                             for name in ("arrays.npz", "camera_geometry.npz", "joint_states.npz")}}
        write_json(self.staging / "manifest.json", manifest)
        write_json(self.staging / "training_sync.json", sync)
        write_json(self.staging / "instructions.json", {"instructions": [self.instruction]})
        for name, entries in self.capture_index.items():
            with open(self.staging / f"camera/color/{name}/sync.txt", "x") as handle:
                handle.write("".join(f"{stamp:.6f} {Path(rel).name}\n" for stamp, rel in entries))
        with open(self.staging / "COMPLETE", "x") as handle:
            handle.write("sim episode\n")
        if self.path.exists():
            raise FileExistsError(self.path)
        self.staging.rename(self.path)
        self.closed = True
        return self.path

    def abort(self) -> Path:
        self.closed = True
        return self.staging


# ----------------------------------------------------------------------------- commands
def module_versions() -> dict:
    """Versions of the modules in use; sapien is only read once import_sapien has imported it."""
    out = {"python": sys.version.split()[0]}
    for name in ("numpy", "PIL", "cv2"):
        try:
            out[name] = str(getattr(importlib.import_module(name), "__version__", "unknown"))
        except ImportError:
            out[name] = None
    sapien = sys.modules.get("sapien")
    out["sapien"] = str(getattr(sapien, "__version__", "unknown")) if sapien is not None else None
    return out


def probe(out: Path, renderer: str, preload: list) -> int:
    """Import SAPIEN as a rollout would, create a URDF loader and, with a renderer, draw one frame
    of a lit table."""
    out.mkdir(parents=True, exist_ok=False)
    report = {"schema": "neo-vla-real2sim-probe/1", "renderer": renderer,
              "note": "environment probe only; it does not validate any task's physics or rendering"}
    ok = True
    try:
        sapien = import_sapien(renderer, preload)
        # Forward kinematics of commanded joints and IK use the articulation's Pinocchio model.
        report["pinocchio_model"] = hasattr(sapien.physx.PhysxArticulation, "create_pinocchio_model")
        ok = report["pinocchio_model"]
        sapien.physx.set_scene_config(gravity=[0.0, 0.0, -9.81], cpu_workers=1)
        scene = None
        if renderer in ("cpu", "gpu"):
            sim = Sim.__new__(Sim)
            sim.s, sim.render = sapien, True
            sim.spec = {"lighting": {"ambient_rgb": [0.6, 0.6, 0.6]},
                        "table": {"size_m": [1.0, 1.0, 0.02], "friction": 0.5, "yaw_rad": 0.0,
                                  "center_xy_m": [0.0, 0.0], "top_z_m": 0.0, "color_rgba": [0.9, 0.2, 0.2, 1.0]}}
            system = sapien.render.RenderSystem() if renderer == "cpu" else sapien.render.RenderSystem("cuda:0")
            report["device"] = str(system.device.name)
            sim.materials = sapien.SapienRenderer()
            sim.scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), system])
            sim.scene.set_timestep(1.0 / 240.0)
            sim._lighting()
            sim._table()
            cam = sim.scene.add_camera("probe", 64, 48, 1.0, 0.01, 10.0)
            cam.set_perspective_parameters(0.01, 10.0, 60.0, 60.0, 32.0, 24.0, 0.0)
            cam.entity.set_pose(sim.pose(look_at([0.0, -0.8, 0.6], [0.0, 0.0, 0.0]) @ np.asarray(T_C_S)))
            sim.scene.update_render()
            cam.take_picture()
            rgb = np.clip(cam.get_picture("Color")[:, :, :3] * 255, 0, 255).astype(np.uint8)
            report["render_nonblank"] = bool(rgb.max() != rgb.min())
            ok = ok and report["render_nonblank"]
            scene = sim.scene
        # SAPIEN imports its URDF parser (and that parser's dependencies, such as six) only when a
        # URDF loader is first created; create one as a rollout does, so a missing dependency fails
        # here and not in the first rollout.
        if scene is None:
            scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
        loader = scene.create_urdf_loader()
        report["urdf_loader"] = f"{type(loader).__module__}.{type(loader).__name__}"
    except Exception as error:  # noqa: BLE001 - reported, never hidden
        report.update(error=repr(error), traceback=traceback.format_exc())
        ok = False
    report["versions"] = module_versions()
    ok = ok and all(report["versions"].get(m) for m in ("numpy", "sapien", "PIL"))
    report["ok"] = bool(ok)
    write_json(out / "PROBE.json", safe_json(report))
    return 0 if ok else 3


def rollout(spec_path: Path, out: Path) -> int:
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    raw = spec_path.read_bytes()
    spec = json.loads(raw)
    result: dict = {"schema": RUNNER_SCHEMA, "status": "FAILED", "reasons": [],
                    "runner_sha256": sha256_bytes(Path(__file__).read_bytes()),
                    "checks_sha256": CHECKS_SHA256,   # the bytes this process loaded and runs
                    "spec_sha256": sha256_bytes(raw)}
    sim = writer = None
    try:
        if spec.get("schema") != SPEC_SCHEMA:
            raise RolloutError(f"spec schema {spec.get('schema')!r} != {SPEC_SCHEMA}")
        if result["runner_sha256"] != spec["runtime"]["runner_sha256"] or \
                result["checks_sha256"] != spec["runtime"]["checks_sha256"]:
            raise RolloutError("staged runner/checks bytes differ from the build receipt")
        scene, task, layout = spec["scene"], spec["task"], spec["episode"]["layout"]
        if spec.get("renderer") not in ("cpu", "gpu"):
            raise RolloutError("episodes are recorded with a renderer (cpu or gpu)")
        sim = Sim(scene, spec["renderer"], spec.get("preload_libraries") or [],
                  task["integrity"]["contact_impulse_eps"])
        result["device"] = sim.device
        result["versions"] = module_versions()
        controller_spec = task["controller"]
        robot = scene["robot"]
        ctx = {"api_version": PLUGIN_API, "params": controller_spec.get("params", controller_spec),
               "task": task, "scene": scene, "layout": layout, "control_hz": CONTROL_HZ,
               "arm_joints": list(robot["arm_joints"]), "joint_limits": robot["joint_limits"],
               "init_arm_q": list(robot["init_arm_q"]), "init_gripper_m": float(robot["gripper"]["init_aperture_m"]),
               "gripper_stroke_m": sim.stroke, "base_pose_W": sim.T_W_B.tolist(),
               "robot_max_joint_speed_rad_s": sim.max_speed, "fk": sim.fk}
        ik_cfg = (controller_spec.get("ik") or {"position_tol_m": 1e-3, "rotation_tol_rad": 1e-2})
        ctx["ik"] = lambda target, seed: sim.ik(target, seed, float(ik_cfg["position_tol_m"]),
                                                float(ik_cfg["rotation_tol_rad"]))
        kind = controller_spec["type"]
        if kind == "joint_waypoints":
            factory = JointWaypoints
        elif kind == "pick_place":
            factory = PickPlace
        elif kind == "plugin":
            factory = load_plugin(Path(spec["runtime"]["plugin_path"]), spec["runtime"]["plugin_sha256"],
                                  controller_spec.get("class", "Controller"))
        else:
            raise RolloutError(f"unknown controller type {kind!r}")
        names = [c["name"] for c in sim.cameras]
        metadata = {"spec_sha256": result["spec_sha256"], "scene_sha256": spec["scene_sha256"],
                    "task_sha256": spec["task_sha256"], "plugin_sha256": spec["runtime"].get("plugin_sha256"),
                    "layout": layout, "layout_sha256": spec["episode"]["layout_sha256"],
                    "episode_index": spec["episode"]["index"], "attempt": spec["episode"]["attempt"],
                    "runner_sha256": result["runner_sha256"], "renderer": spec["renderer"], "device": sim.device,
                    "versions": result["versions"], "control_hz": CONTROL_HZ, "physics_hz": sim.hz}
        if spec.get("synthetic_fixture") is True:   # the project is a synthetic test fixture
            metadata["synthetic_fixture"] = True
        calibration = {"cameras.json": scene["cameras"], "robot.json": {
            "base_pose_W": sim.T_W_B.tolist(), "tcp_in_ee": sim.T_E_TCP.tolist(), "ee_link": robot["ee_link"],
            "arm_joints": robot["arm_joints"]}, "scene.json": spec.get("scene_reviewed", scene),
            "task.json": spec.get("task_reviewed", task)}
        writer = EpisodeWriter(out / "episode", cameras=scene["cameras"], stroke=sim.stroke,
                               joint_names=robot["arm_joints"],
                               instruction=spec["instruction"], calibration=calibration, metadata=metadata)
        epoch = float(spec["epoch_s"])
        state = {"row": 0}

        def record() -> None:
            row = state["row"]
            due = {c["name"] for c in sim.cameras if row % c["every"] == 0}
            frames, poses = sim.capture(due)
            writer.append(time_s=epoch + row / CONTROL_HZ, measured=sim.tcp_base(),
                          command=np.asarray(sim.fk(sim.command_q), float),
                          measured_gap=float(np.clip(sim.aperture(), 0.0, sim.stroke)),
                          command_gap=sim.command_aperture, measured_joints=sim.qpos()[sim.arm_idx],
                          command_joints=sim.command_q, frames=frames,
                          camera_poses={n: sim.T_B_W @ poses[n] for n in names})
            if row == 0:
                result["camera_readback"] = sim.camera_readback()

        sim.place(layout)
        sim.snapshot("settle")
        record()
        init_q = np.asarray(robot["init_arm_q"], float)
        for _ in range(math.ceil(float(task["episode"]["settle_s"]) * CONTROL_HZ - 1e-9)):
            sim.control_step(init_q, sim.init_aperture, "settle")
            state["row"] += 1
            record()
        measured = sim.T_W_B @ sim.fk_ee(sim.qpos()[sim.arm_idx])
        actual = sim.ee_world()
        result["fk_check"] = {"position_error_m": float(np.linalg.norm(measured[:3, 3] - actual[:3, 3])),
                              "rotation_error_rad": checks.angle_between(measured[:3, :3].tolist(),
                                                                         actual[:3, :3].tolist())}
        if result["fk_check"]["position_error_m"] > 1e-4 or result["fk_check"]["rotation_error_rad"] > 1e-3:
            raise RolloutError("Pinocchio forward kinematics does not match the articulation (URDF root/base frame?)",
                               result["fk_check"])
        controller = factory(ctx)
        previous = list(sim.command_q)
        budget = math.ceil(float(task["episode"]["max_duration_s"]) * CONTROL_HZ - 1e-9)
        done = False
        for step in range(budget):
            obs = {"step": step, "t_s": step / CONTROL_HZ, "arm_q": sim.qpos()[sim.arm_idx].tolist(),
                   "gripper_m": sim.aperture(), "tcp_pose_B": sim.tcp_base().tolist(),
                   "base_pose_W": sim.T_W_B.tolist(), "objects": sim.object_state(),
                   "command_arm_q": sim.command_q.tolist(), "command_gripper_m": sim.command_aperture}
            arm, gap, phase, done = check_command(controller.act(obs), sim, previous)
            previous = arm
            sim.control_step(arm, gap, phase)
            state["row"] += 1
            record()
            if done:
                break
        result["controller_log"] = list(getattr(controller, "log", []))[-200:]
        if not done:
            raise RolloutError(f"episode budget of {budget} control steps ended before the controller finished")
        result["camera_readback_end"] = sim.camera_readback()
        result.update(rows=state["row"] + 1, ticks=sim.tick + 1)
        np.savez_compressed(out / "trace.npz", **sim.trace_arrays())
        meta = trace_meta(sim, result["spec_sha256"])
        write_json(out / "trace.json", meta)
        result["trace_sha256"] = sha256_bytes((out / "trace.npz").read_bytes())
        loaded = checks.load_npz(out / "trace.npz")   # audit the persisted file, not the memory copy
        result["audit"] = checks.audit_trace(task, meta, loaded)
        result["initial_pose_problems"] = checks.initial_pose_problems(meta, loaded, layout)
        if not result["audit"]["ok"] or result["initial_pose_problems"]:
            raise RolloutError("trace audit rejected the rollout",
                               {"reasons": result["audit"]["reasons"] + result["initial_pose_problems"]})
        writer.metadata.update(trace_sha256=result["trace_sha256"], audit=result["audit"])
        episode = writer.finish(success=True)
        result["raw"] = checks.raw_episode_report(episode)
        if not result["raw"]["conversion_checks_ok"]:
            rejected = out / "failures" / "finalized_rejected"
            rejected.parent.mkdir(parents=True, exist_ok=True)
            episode.rename(rejected)
            raise RolloutError("finalized episode failed the raw-episode checks", result["raw"]["categories"])
        result["status"] = "ACCEPTED_BY_RUNNER"
    except Exception as error:  # noqa: BLE001 - every failure is recorded with its evidence
        result["reasons"].append(str(error))
        result.update(error=repr(error), traceback=traceback.format_exc(), info=getattr(error, "info", None))
        if sim is not None and sim.trace["tick"] and not (out / "trace.npz").exists():
            try:
                np.savez_compressed(out / "trace.npz", **sim.trace_arrays())
                write_json(out / "trace.json", trace_meta(sim, result["spec_sha256"]))
            except Exception as trace_error:  # noqa: BLE001
                result["trace_error"] = repr(trace_error)
        if writer is not None and not writer.closed:
            partial = writer.abort()
            if partial.exists():
                target = out / "failures" / "raw_partial"
                target.parent.mkdir(parents=True, exist_ok=True)
                partial.rename(target)
    finally:
        result["wall_s"] = round(time.monotonic() - started, 3)
        write_json(out / "RUNNER_RESULT.json", safe_json(result))
    return 0 if result["status"] == "ACCEPTED_BY_RUNNER" else 4


def trace_meta(sim: Sim, spec_sha256: str) -> dict:
    return {"format": checks.TRACE_FORMAT, "physics_hz": sim.hz, "control_hz": CONTROL_HZ,
            "objects": [{"name": o["name"], "dynamic": o["dynamic"], "com_local_m": o["com"]} for o in sim.objects],
            "contact_columns": [o["name"] for o in sim.objects] + ["table", "robot"],
            "contacts_valid_from_tick": 1, "arm_joints": sim.arm_names,
            "contact_impulse_eps": sim.contact_eps, "spec_sha256": spec_sha256,
            "no_teleport": "controllers receive observations and return joint/gripper commands only; the runner "
                           "sets object poses once, before tick 0"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=f"checks: {checks.__file__} (sha256 {CHECKS_SHA256})")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--out", required=True)
    p.add_argument("--renderer", choices=("cpu", "gpu", "none"), required=True)
    p.add_argument("--preload", action="append", default=[])
    r = sub.add_parser("rollout")
    r.add_argument("--spec", required=True)
    r.add_argument("--out", required=True)
    a = parser.parse_args(argv)
    if np is None:
        print(json.dumps({"error": "numpy is not importable in this interpreter"}))
        return 3
    if a.cmd == "probe":
        return probe(Path(a.out), a.renderer, a.preload)
    return rollout(Path(a.spec), Path(a.out))


if __name__ == "__main__":
    sys.exit(main())
