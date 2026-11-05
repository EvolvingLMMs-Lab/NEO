"""RoboTwin 2.0 on a dual-arm Aloha-AgileX, behind RoboTwin's own harness.

RoboTwin's script/eval_policy.py owns the episode loop, the seeds, the
instruction phrasing and the success check; it calls a plug-in for actions
(robotwin_policy/deploy_policy.py). The plug-in holds an Agent, which turns one
RoboTwin observation into a Step, asks the server for one chunk, and executes
the first `horizon` rows as end-effector targets.

The left arm is model slot 0 and the right arm slot 1; the head camera is
camera slot 0 and the left and right wrist cameras slots 1 and 2. Each arm's
actions come back in its own wrist camera's frame (camera:1, camera:2) and are
turned back with that camera's live pose before they are composed onto the
arm's end-effector pose.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy

from ..serve import Client
from .client import BenchPolicy, Step, arm_state, extrinsics_6, is_rotation, noise_seed
from .client import quaternion_matrix
from .spec import EvalSpec

SIDES = ("left", "right")
# Where get_obs writes each arm's readings.
ENDPOSE_KEY = "endpose"
POSE_KEYS = {"left": "left_endpose", "right": "right_endpose"}
GRIPPER_KEYS = {"left": "left_gripper", "right": "right_gripper"}
# Each arm's six drive targets are sent as its joints; the arm has no seventh
# joint, so that entry stays unknown.
JOINT_KEY = "joint_action"
JOINT_KEYS = {"left": "left_arm", "right": "right_arm"}
CAMERA_KEY = "observation"
FRAME_KEY = "rgb"
# SAPIEN states every camera twice: T_cam<-world in OpenCV axes (x right,
# y down, z forward) and T_world<-cam in GL axes (y up, z backward). The first
# transposed is R_base<-camera, since RoboTwin's world frame is the frame its
# end-effector poses are written in; the second, turned into OpenCV axes, must
# agree with it.
EXTRINSIC_CV_KEY = "extrinsic_cv"
MODEL_MATRIX_KEY = "cam2world_gl"
GL_TO_CV = numpy.diag([1.0, -1.0, -1.0])
READINGS_AGREE = 1e-4
CM_PER_METRE = 100.0
DEFAULT_QUATERNION_ORDER = "wxyz"


def camera_poses(observation: dict, cameras) -> dict[int, list[float]]:
    """Every camera's live pose, as the six numbers the model reads, by slot."""
    found = {}
    for camera in cameras:
        block = observation[CAMERA_KEY][camera.name]
        cv = numpy.asarray(block[EXTRINSIC_CV_KEY], dtype=numpy.float64)
        gl = numpy.asarray(block[MODEL_MATRIX_KEY], dtype=numpy.float64)
        rotation = numpy.ascontiguousarray(cv[:3, :3].T)
        apart = float(numpy.abs(gl[:3, :3] @ GL_TO_CV - rotation).max())
        if apart > READINGS_AGREE or not is_rotation(rotation):
            raise ValueError(f"camera {camera.name!r} states two poses {apart:.3e} apart")
        found[camera.slot] = extrinsics_6(rotation)
    return found


def split_endpose(seven, order: str) -> tuple[numpy.ndarray, numpy.ndarray]:
    """A 7-value endpose -> (position in m, xyzw quaternion)."""
    values = numpy.asarray(seven, dtype=numpy.float64).reshape(7)
    turn = values[3:7]
    if order == "wxyz":
        turn = turn[[1, 2, 3, 0]]
    return values[:3], turn


def join_endpose(position_m, quaternion_xyzw, order: str) -> numpy.ndarray:
    turn = numpy.asarray(quaternion_xyzw, dtype=numpy.float64).reshape(4)
    if order == "wxyz":
        turn = turn[[3, 0, 1, 2]]
    return numpy.concatenate([numpy.asarray(position_m, dtype=numpy.float64), turn])


def rotvec_quaternion(rotvec) -> numpy.ndarray:
    """A rotation vector in rad -> its xyzw quaternion."""
    values = numpy.asarray(rotvec, dtype=numpy.float64).reshape(3)
    angle = float(numpy.linalg.norm(values))
    if angle < 1e-8:
        return numpy.array([0.0, 0.0, 0.0, 1.0])
    return numpy.concatenate([values / angle * numpy.sin(angle / 2.0), [numpy.cos(angle / 2.0)]])


def quaternion_multiply(left, right) -> numpy.ndarray:
    """xyzw product left * right: right first, then left."""
    x1, y1, z1, w1 = numpy.asarray(left, dtype=numpy.float64)
    x2, y2, z2, w2 = numpy.asarray(right, dtype=numpy.float64)
    return numpy.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def apply_delta(position_cm, quaternion_xyzw, dpos_cm, drot_rad):
    """One spatial delta onto a pose: the translation adds, the rotation multiplies on the left."""
    rotation = numpy.asarray(quaternion_xyzw, dtype=numpy.float64)
    rotation = rotation / numpy.linalg.norm(rotation)
    if rotation[3] < 0:
        rotation = -rotation
    turned = quaternion_multiply(rotvec_quaternion(drot_rad), rotation)
    return (numpy.asarray(position_cm, dtype=numpy.float64) + numpy.asarray(dpos_cm),
            turned / numpy.linalg.norm(turned))


@dataclass
class Episode:
    index: int
    instruction: str = ""
    success: bool = False
    steps: int = 0
    calls: int = 0
    rows_discarded: int = 0
    seeds: list = field(default_factory=list)


class Agent:
    """What the plug-in holds: the server, the step's eval.json and this task's record."""

    def __init__(self, spec: EvalSpec, client: Client, *, task: str, horizon: int,
                 noise_base: int | None, record: str | Path,
                 quaternion_order: str = DEFAULT_QUATERNION_ORDER):
        if len(spec.arms) != len(SIDES):
            raise ValueError(f"RoboTwin drives two arms; eval.json states {len(spec.arms)}")
        self.spec = spec
        self.policy = BenchPolicy(client, spec, units="robotwin")
        self.task = str(task)
        self.horizon = int(horizon)
        self.noise_base = noise_base
        self.record = Path(record)
        self.order = quaternion_order
        self.episodes: list[Episode] = []
        self.instruction = ""

    def reset(self) -> None:
        """Called by the harness before every episode; episodes are numbered from 1."""
        self.episodes.append(Episode(index=len(self.episodes) + 1))
        self.policy.reset()
        self.instruction = ""
        self.write()

    def write(self) -> None:
        self.record.parent.mkdir(parents=True, exist_ok=True)
        partial = self.record.with_suffix(".partial")
        partial.write_text(json.dumps([asdict(one) for one in self.episodes], indent=1) + "\n")
        partial.replace(self.record)

    def step_of(self, observation: dict) -> Step:
        images = {
            camera.slot: numpy.asarray(observation[CAMERA_KEY][camera.name][FRAME_KEY],
                                       dtype=numpy.uint8)
            for camera in self.spec.cameras
        }
        poses, joints = observation[ENDPOSE_KEY], observation[JOINT_KEY]
        states = []
        for arm, side in zip(self.spec.arms, SIDES):
            position, turn = split_endpose(poses[POSE_KEYS[side]], self.order)
            states.append(arm_state(
                position, quaternion_matrix(turn, "xyzw"),
                gripper_raw=float(numpy.asarray(poses[GRIPPER_KEYS[side]]).reshape(-1)[0]),
                raw_open=arm.raw_open, raw_closed=arm.raw_closed,
                joints=([float(x) for x in numpy.asarray(joints[JOINT_KEYS[side]]).reshape(-1)]
                        if arm.joints else None),
            ))
        return Step(images=images, states=states,
                    poses=camera_poses(observation, self.spec.cameras))

    def rows(self, observation: dict) -> numpy.ndarray:
        """One observation -> the endpose targets take_action executes, [rows, 16].

        Each row's delta is composed onto the previous row's target, starting
        from the pose the simulator reports: the planner moves to each target
        in turn, so a delta read against the starting pose would ask for the
        same small motion at every row.
        """
        episode = self.episodes[-1]
        seed = None
        if self.noise_base is not None:
            seed = noise_seed(self.noise_base, self.task, 0, episode.index, episode.calls)
        step = self.step_of(observation)
        chunk = self.policy.chunk(step, self.instruction, seed=seed)
        executed = min(self.horizon, chunk.shape[1])
        episode.calls += 1
        episode.seeds.append(seed)
        episode.rows_discarded += chunk.shape[1] - executed
        decoded = self.policy.decode(chunk[:, :executed], step)
        self.policy.executed_rows(executed)
        poses = observation[ENDPOSE_KEY]
        per_arm = []
        for arm, side in zip(self.spec.arms, SIDES):
            position, turn = split_endpose(poses[POSE_KEYS[side]], self.order)
            position = position * CM_PER_METRE
            targets = []
            for row in decoded:
                position, turn = apply_delta(position, turn, row[list(arm.columns["dpos"])],
                                             row[list(arm.columns["drot"])])
                canonical = float(row[arm.columns["gripper"][0]])
                gripper = arm.raw_open + canonical * (arm.raw_closed - arm.raw_open)
                targets.append(numpy.concatenate([
                    join_endpose(position / CM_PER_METRE, turn, self.order), [gripper]]))
            per_arm.append(numpy.stack(targets))
        return numpy.concatenate(per_arm, axis=1)

    def act(self, environment, observation: dict) -> None:
        """One call of the harness's eval(): ask once, execute the horizon."""
        episode = self.episodes[-1]
        if not self.instruction:
            self.instruction = str(environment.get_instruction() or "")
            episode.instruction = self.instruction
        for row in self.rows(observation):
            environment.take_action(list(row), action_type="ee")
            episode.steps = int(environment.take_action_cnt)
            if environment.eval_success:
                episode.success = True
                break
        self.write()


def wait_for(client: Client, seconds: float = 900.0, every: float = 2.0) -> None:
    """Block until the server answers /health; a checkpoint takes minutes to load."""
    deadline = time.monotonic() + seconds
    while True:
        try:
            client.health()
            return
        except RuntimeError:
            if time.monotonic() > deadline:
                raise
            time.sleep(every)


def agent_from_settings(path: str | Path) -> Agent:
    """The Agent a task's settings file describes (written by robotwin_run)."""
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    spec = EvalSpec.load(settings["eval_json"])
    client = Client(settings["server"], timeout=float(settings.get("timeout", 300)))
    wait_for(client)
    return Agent(spec, client, task=settings["task"], horizon=int(settings["horizon"]),
                 noise_base=settings.get("noise_seed"), record=settings["record"],
                 quaternion_order=settings.get("quaternion_order", DEFAULT_QUATERNION_ORDER))
