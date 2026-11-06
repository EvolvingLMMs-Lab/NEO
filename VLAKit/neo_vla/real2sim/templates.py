"""Templates and specification guides (kept as Python objects so no package data is needed)."""
from __future__ import annotations

import copy
from typing import Any

IDENTITY = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def q(value: Any, unit: str, provenance: str, source: str) -> dict:
    return {"value": value, "unit": unit, "provenance": provenance, "source": source}


def project(route: str = "sapien") -> dict:
    agent = {"enabled": True, "executable": "claude", "model": None, "effort": "high", "timeout_s": 1800,
             "max_attempts": 1}
    collection = {"seed": 0, "episodes": 20, "qualify_episodes": 2, "max_attempts_per_episode": 1,
                  "attempt_timeout_s": 3600, "workers": 1}
    if route == "can_in_tape_reference":
        return {
            "schema": "neo-vla-real2sim-project/1", "name": "can-in-tape-reference",
            "route": "can_in_tape_reference", "instruction": "put the can in the tape roll",
            "inputs": {"images": [], "measurements": [], "calibration": [], "assets": []},
            "reference": {
                "workspace": "/path/to/can_in_tape_runtime",
                "launcher": ["bash", "/path/to/sim_twin/scripts/cpu.sh"],
                "renderer": "cpu", "seed": 20261028, "cpu_lists": [],
                "inputs": {"actuator_profile": "/path/to/actuator_profile.json", "assets": "/path/to/assets_dir",
                           "real_plan": "/path/to/episode1_execution_plan.json"}},
            "collection": {**collection, "episodes": 400, "attempt_timeout_s": 10800},
            "output": {"split_policy": {"groups": {"in_domain": {"splits": ["in_domain"]},
                                                   "ood": {"splits": ["ood"]}}}},
            "agent": {**agent, "enabled": False}}
    return {
        "schema": "neo-vla-real2sim-project/1", "name": "my-task", "route": "sapien",
        "instruction": "describe the task in one sentence",
        "inputs": {
            "images": [{"path": "images/overview.jpg", "shows": "the whole work cell from the side camera",
                        "use": "scene_context"}],
            "measurements": [{"id": "table_length", "quantity": "table length", "value": None, "unit": "m",
                              "provenance": "measured", "method": "tape measure", "by": "your name",
                              "date": "YYYY-MM-DD"}],
            "calibration": [{"id": "side_camera", "path": "calibration/side_camera.json", "kind": "intrinsics+extrinsics",
                             "method": "checkerboard calibration (OpenCV)"}],
            "robot": {"urdf": "robot/robot.urdf", "source": "where the URDF and meshes come from",
                      "license": "the terms you use them under"},
            "assets": []},
        "specs": {"scene": "specs/scene.json", "task": "specs/task.json", "plugin": None},
        "simulator": {"python": "/path/to/sim-env/bin/python", "renderer": "cpu",
                      "env": {"VK_ICD_FILENAMES": "/path/to/lvp_icd.x86_64.json"}, "preload_libraries": []},
        "collection": collection,
        "output": {"split_policy": {"groups": {"train": {"indices": "0-15"}, "heldout": {"indices": "16-19"}}}},
        "agent": agent}


def scene_example(proj: dict | None = None) -> dict:
    """Scene skeleton: every physical value is null until it is measured or estimated and cited."""
    urdf = ((proj.data if hasattr(proj, "data") else proj or {}).get("inputs") or {}).get("robot", {}).get(
        "urdf", "robot/robot.urdf")
    measured = "measured"
    return {
        "schema": "neo-vla-real2sim-scene/1",
        "physics": {"physics_hz": 240, "control_hz": 30, "gravity_m_s2": [0.0, 0.0, -9.81]},
        "table": {"size_m": q(None, "m", measured, "measurement:<id> (length, width, thickness)"),
                  "top_z_m": q(None, "m", measured, "measurement:<id>"),
                  "center_xy_m": q(None, "m", measured, "measurement:<id>"),
                  "yaw_rad": q(None, "rad", measured, "measurement:<id>"),
                  "friction": q(None, "1", "estimate", "basis of the estimate"),
                  "color_rgba": [0.35, 0.35, 0.35, 1.0]},
        "robot": {"urdf": urdf, "base_pose_W": q(None, "4x4", measured, "calibration:<id>#<path>"),
                  "arm_joints": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
                  "init_arm_q": q(None, "rad", "chosen", "a collision-free start configuration"),
                  "ee_link": "link6", "tcp_in_ee": q(None, "4x4", "derived", "gripper CAD or URDF"),
                  "gripper": {"joints": [{"name": "finger_left", "scale": 1.0}, {"name": "finger_right", "scale": 1.0}],
                              "aperture_offset_m": q(None, "m", "derived", "finger geometry"),
                              "stroke_m": q(None, "m", "derived", "gripper datasheet or URDF limits"),
                              "init_aperture_m": q(None, "m", "chosen", "open at start")},
                  "drive": {"arm": {"stiffness": q(None, "N m/rad", "estimate", "basis"),
                                    "damping": q(None, "N m s/rad", "estimate", "basis"),
                                    "force_limit": q(None, "N m", "estimate", "basis")},
                            "gripper": {"stiffness": q(None, "N/m", "estimate", "basis"),
                                        "damping": q(None, "N s/m", "estimate", "basis"),
                                        "force_limit": q(None, "N", "estimate", "basis")},
                            "gravity_compensation": True,
                            "max_joint_speed_rad_s": q(None, "rad/s", "chosen", "slower than the real robot limit")}},
        "objects": [{"name": "cube", "shape": {"type": "box", "size_m": q(None, "m", measured, "measurement:<id>")},
                     "dynamic": True, "mass_kg": q(None, "kg", measured, "measurement:<id> (kitchen scale)"),
                     "friction": q(None, "1", "estimate", "basis"), "restitution": q(None, "1", "estimate", "basis"),
                     "pose_W": q(None, "4x4", "chosen", "nominal placement"), "color_rgba": [0.8, 0.1, 0.1, 1.0]}],
        "cameras": [{"name": "side", "mount": "world", "pose": q(None, "4x4", measured, "calibration:<id>#<path>"),
                     "K": q(None, "px", measured, "calibration:<id>#<path>"), "width": 1280, "height": 720,
                     "near_m": 0.02, "far_m": 5.0, "every_n_rows": 1}],
        "lighting": {"ambient_rgb": [0.5, 0.5, 0.5], "directional": [{"direction": [0.3, 0.3, -1.0],
                                                                     "rgb": [1.0, 1.0, 1.0], "shadow": False}]},
        "randomization": {"objects": {"cube": {"dx_m": [-0.03, 0.03], "dy_m": [-0.03, 0.03],
                                               "dyaw_rad": [-0.5, 0.5]}}, "min_center_distance_m": 0.1}}


def task_example(proj: Any = None) -> dict:
    data = proj.data if hasattr(proj, "data") else proj or {}
    down = [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    return {
        "schema": "neo-vla-real2sim-task/1", "instruction": data.get("instruction", "describe the task"),
        "notes": "illustrative values; choose and review your own",
        "episode": {"max_duration_s": 40.0, "settle_s": 0.5},
        "controller": {"type": "pick_place", "object": "cube", "grasp_in_object": down,
                       "approach_distance_m": 0.08, "lift_height_m": 0.12,
                       "place": {"relative_to": "world", "object_in_reference": copy.deepcopy(IDENTITY)},
                       "retreat_height_m": 0.08, "gripper_open_m": 0.08, "gripper_close_m": 0.0,
                       "open_wait_s": 0.8, "close_wait_s": 1.0, "final_hold_s": 2.5,
                       "max_joint_speed_rad_s": 1.0, "segment_min_s": 1.0, "cartesian_pieces": 10,
                       "ik": {"position_tol_m": 0.001, "rotation_tol_rad": 0.01, "max_joint_change_rad": 1.0}},
        "success": {"hold_s": 2.0, "require_phases": ["close", "release"], "conditions": [
            {"type": "position_in_box", "object": "cube", "relative_to": "world", "center_m": [0.0, 0.0, 0.025],
             "half_extents_m": [0.03, 0.03, 0.01]},
            {"type": "no_contact", "object": "cube", "with": "robot"},
            {"type": "contact", "object": "cube", "with": "table"},
            {"type": "speed_max", "object": "cube", "linear_m_s": 0.02, "angular_rad_s": 0.2},
            {"type": "phase_in", "phases": ["retreat", "hold"]}]},
        "integrity": {"max_teleport_residual_m": 0.005, "max_teleport_residual_rad": 0.05,
                      "max_settle_drift_m": 0.005, "contact_impulse_eps": 1e-6}}


PLUGIN_SKELETON = '''"""Controller plugin for the real2sim runner (plugin API version 1)."""
import math

API_VERSION = 1


class Controller:
    """Called once per 30 Hz control step after the scene has settled.

    context: params (task.controller.params), task, scene, layout, arm_joints, joint_limits,
    init_arm_q, init_gripper_m, gripper_stroke_m, base_pose_W, control_hz,
    robot_max_joint_speed_rad_s, ik(target_tcp_in_base_4x4, seed_arm_q) -> (arm_q or None, info),
    fk(arm_q) -> tcp_in_base_4x4.
    """

    def __init__(self, context):
        self.ctx = context
        self.q = list(context["init_arm_q"])
        self.gripper = float(context["init_gripper_m"])

    def act(self, obs):
        """obs: step, t_s, arm_q, gripper_m, tcp_pose_B, base_pose_W,
        objects {name: {pose_W, lin_vel, ang_vel}}, command_arm_q, command_gripper_m.

        Return the next command. The runner rejects non-finite values, joint targets outside the
        URDF limits, steps faster than max_joint_speed / 30 and apertures outside [0, stroke].
        """
        return {"arm_q": self.q, "gripper_m": self.gripper, "phase": "hold", "done": True}
'''

SCENE_GUIDE = """# Scene specification (neo-vla-real2sim-scene/1)

Units are metres, kilograms, seconds and radians. Poses are 4x4 row-major homogeneous transforms;
T_A_B maps coordinates in frame B into frame A. The world frame W is right-handed with +z up and a
horizontal table top. The robot base frame B is the URDF root link.

Every physical value is a quantity {"value", "unit", "provenance", "source"}:
- measured: measured directly or calibrated; cite "measurement:<id>" (project.json measurements) or
  "calibration:<id>#<json.path>" (a calibration file), which are checked where machine-readable;
- derived: computed from measurements, CAD or the URDF; say how in source;
- estimate: an assumed physical value; give the basis;
- chosen: a design decision (start configuration, nominal placement, speed limit);
- unverified: of unknown origin;
- inferred_from_image: read off a photograph. It is reviewed and never counts as measured geometry,
  and dynamic properties (mass, friction, restitution, drive gains) are never taken from images.
Use null for a value you cannot support.

Measured or derived is required for: table size_m (length, width, thickness), top_z_m, center_xy_m,
yaw_rad; robot base_pose_W and tcp_in_ee; gripper aperture_offset_m and stroke_m; object shape sizes
(and mesh scale and footprint_radius_m); camera pose and K (and D when given).

physics: physics_hz (a multiple of 30), control_hz (30), gravity_m_s2.
table: size_m, top_z_m, center_xy_m, yaw_rad, friction, color_rgba.
robot: urdf (inputs.robot.urdf; mesh paths in it must be relative to the URDF file), base_pose_W,
  arm_joints, init_arm_q, ee_link, tcp_in_ee (TCP in the ee link frame; TCP +z points along the
  approach direction), gripper {joints [{name, scale}], aperture_offset_m, stroke_m, init_aperture_m}
  with aperture = aperture_offset_m + sum(scale * joint position), drive {arm, gripper: {stiffness,
  damping, force_limit}, gravity_compensation, max_joint_speed_rad_s}, optional follower {alpha,
  delay_steps} (first-order lag fitted from real joint logs). Every movable URDF joint must be an arm
  or a gripper joint.
objects: name, shape {type box: size_m | cylinder: radius_m, length_m (axis = object z) | sphere:
  radius_m | mesh: collision, visual, scale, footprint_radius_m, com_m, inertia_diag_kg_m2, convex
  single|multiple}, dynamic, mass_kg, friction, restitution, pose_W (nominal), color_rgba, texture.
cameras: name, mount ("world" or "link:<URDF link>"), pose (optical frame: x right, y down,
  z forward; T_W_C for world mounts, T_L_C for link mounts), K [[fx,0,cx],[0,fy,cy],[0,0,1]] for
  width x height pixels, optional D (OpenCV Brown-Conrady k1, k2, p1, p2[, k3]; frames are then stored
  distorted like the real camera), near_m, far_m, every_n_rows (1 to 3).
lighting: ambient_rgb, directional [{direction, rgb, shadow}].
randomization: objects {name: {dx_m, dy_m, dyaw_rad}} ranges about pose_W, min_center_distance_m.
"""

TASK_GUIDE = """# Task specification (neo-vla-real2sim-task/1)

instruction: identical to project.json.
episode: max_duration_s (the controller must finish within it), settle_s (robot holds still first).
controller, one of:
- joint_waypoints: waypoints [{arm_q, duration_s, gripper_m, phase}], final_hold_s,
  max_joint_speed_rad_s. Smoothstep moves; useful to replay a measured real demonstration.
- pick_place: object, grasp_in_object (TCP pose in the object frame), approach_distance_m,
  lift_height_m, place {relative_to: object or "world", object_in_reference: target object pose},
  retreat_height_m, gripper_open_m, gripper_close_m, open_wait_s, close_wait_s, final_hold_s,
  max_joint_speed_rad_s, segment_min_s, cartesian_pieces, ik {position_tol_m, rotation_tol_rad,
  max_joint_change_rad}. Phases: open, approach, descend, close, lift, carry, lower, release,
  retreat, hold. The place target uses the object pose measured in hand after the lift.
- plugin: path (= specs.plugin), class, params. See the plugin guide.
success: hold_s, require_phases, conditions. Every condition must hold on every physics tick of a
  continuous run that ends at the last tick and lasts at least hold_s. Conditions:
  position_in_box {object, relative_to, center_m, half_extents_m}; axis_tilt_max {object, axis,
  max_deg}; contact and no_contact {object, with: object, "table" or "robot"}; speed_max {object,
  linear_m_s, angular_rad_s}; gripper_open_min {aperture_m}; displacement_max {object, max_m,
  planar}; phase_in {phases}.
integrity: max_teleport_residual_m and max_teleport_residual_rad (pose change not explained by the
  object's own velocity per tick), max_settle_drift_m, contact_impulse_eps.
All thresholds are chosen values that a person reviews. Prefer conditions that can only hold when
the task truly succeeded.
"""

PLUGIN_GUIDE = """# Controller plugin API (version 1)

A plugin is one Python file that sets API_VERSION = 1 and defines a controller class (default name
Controller). The runner constructs it once with a context and calls act(obs) every 30 Hz control
step after the settle phase; act returns {"arm_q", "gripper_m", "phase", "done"}. The controller
sees observations only (no simulator handle) and must not read or write files, start processes or
import modules beyond math-style standard modules and numpy. See controller_skeleton.py for the
context and observation fields. Commands are rejected (and the episode fails) when they are
non-finite, outside the URDF joint limits, faster than max_joint_speed / 30 per step, or when the
aperture leaves [0, stroke]. Success is decided only by the independent checks of the task contract.
"""
