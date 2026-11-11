"""Ask a running neo-vla-serve for chunks and turn them into simulator actions.

A Step is what one simulator step reports, in model terms: an RGB frame per
camera slot, a 29-value state record per arm, and the live pose of every camera
whose pose the simulator states. BenchPolicy sends a Step (with the step it
asked at before as history), then decodes the chunk: actions written in a
camera's frame are turned back into the base frame with that camera's live
pose, divided by the benchmark's unit factors and laid out in the simulator's
own action columns. The gripper column keeps the canonical value, 0 open and
1 closed, which each benchmark maps to its own command.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy

from ..serve import Client, Image, Observation, StateField
from .spec import DPOS, DROT, GRIPPER, EvalSpec, arm_field, camera_field, intrinsics_field
from .spec import pose_field

ROTATION_TOLERANCE = 1e-4


def is_rotation(matrix) -> bool:
    values = numpy.asarray(matrix, dtype=numpy.float64)
    if values.shape != (3, 3) or not numpy.isfinite(values).all():
        return False
    if numpy.abs(values.T @ values - numpy.eye(3)).max() > ROTATION_TOLERANCE:
        return False
    return abs(numpy.linalg.det(values) - 1.0) <= ROTATION_TOLERANCE


def extrinsics_6(rotation_base_from_camera) -> list[float]:
    """R_base<-camera (OpenCV axes) -> forward then up, as neo_vla.data.conditioning writes it."""
    values = numpy.asarray(rotation_base_from_camera, dtype=numpy.float64)
    if not is_rotation(values):
        raise ValueError(f"not a rotation: {values.tolist()}")
    six = numpy.concatenate([values[:, 2], -values[:, 1]]).astype(numpy.float32)
    return [float(x) for x in six]


def rotation_from_extrinsics_6(six) -> numpy.ndarray:
    values = numpy.asarray(six, dtype=numpy.float64).reshape(6)
    forward, up = values[:3], values[3:]
    return numpy.column_stack([numpy.cross(-up, forward), -up, forward])


def quaternion_matrix(quaternion, order: str = "xyzw") -> numpy.ndarray:
    values = numpy.asarray(quaternion, dtype=numpy.float64).reshape(4)
    x, y, z, w = values if order == "xyzw" else values[[1, 2, 3, 0]]
    norm = numpy.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return numpy.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def arm_state(position_m, rotation, *, gripper_raw, raw_open: float, raw_closed: float,
              joints=None) -> list:
    """One arm's readings -> its 29-value record; None where not measured.

    position_m is in the robot base frame, rotation is R_base<-ee, and the
    aperture is the gripper reading mapped so that raw_open is 0 and raw_closed
    is 1. The hand is not measured on any simulated arm here.
    """
    record: list = [None] * 29
    for index, value in enumerate([] if joints is None else list(joints)[:7]):
        record[index] = None if value is None else float(value)
    if position_m is not None:
        matrix = numpy.asarray(rotation, dtype=numpy.float64).reshape(3, 3)
        record[7:10] = [float(x) for x in numpy.asarray(position_m, dtype=numpy.float64)]
        record[10:16] = [float(x) for x in numpy.concatenate([matrix[:, 0], matrix[:, 1]])]
    if gripper_raw is not None:
        travelled = (float(gripper_raw) - raw_open) / (raw_closed - raw_open)
        record[16] = min(max(travelled, 0.0), 1.0)
    return [None if value is None else float(numpy.float32(value)) for value in record]


def noise_seed(base: int, suite: str, task: int, episode: int, call: int) -> int:
    """The decode seed of one request: a hash of where it falls in the run, not of when."""
    payload = "|".join(str(x) for x in (int(base), str(suite), int(task), int(episode), int(call)))
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") >> 1


@dataclass
class Step:
    images: dict[int, numpy.ndarray]
    states: list
    poses: dict[int, list[float]] = field(default_factory=dict)


class BenchPolicy:
    """One episode's view of the server: it keeps the step it last asked at as history."""

    def __init__(self, client: Client, spec: EvalSpec, *, units: str):
        self.client = client
        self.spec = spec
        self.translation, self.rotation = spec.unit_factors(units)
        self.reset()

    def reset(self) -> None:
        self.previous: Observation | None = None
        self.executed = 0
        self.calls = 0

    def observation(self, step: Step, instruction: str, *, seed: int | None) -> Observation:
        images, state = {}, {}
        for camera in self.spec.cameras:
            frame = step.images.get(camera.slot)
            if frame is None:
                continue
            frame = numpy.ascontiguousarray(frame, dtype=numpy.uint8)
            images[camera_field(camera.slot)] = Image(int(frame.shape[1]), int(frame.shape[0]),
                                                      frame.tobytes())
            state[intrinsics_field(camera.slot)] = StateField(camera.intrinsics, "px")
            if camera.slot in step.poses:
                state[pose_field(camera.slot)] = StateField(tuple(step.poses[camera.slot]), "unit")
        for index, record in enumerate(step.states):
            state[arm_field(index)] = StateField(tuple(record), "model_record")
        return Observation(
            request_id=f"call-{self.calls}",
            instruction=instruction,
            # The server turns the time between two requests into executed rows.
            timestamp_s=self.executed / self.spec.control_hz,
            images=images,
            state=state,
            history=() if self.previous is None else (self.previous,),
            seed=seed,
        )

    def chunk(self, step: Step, instruction: str, *, seed: int | None = None) -> numpy.ndarray:
        """[arms, rows, 26] in physical units, NaN where a column was not emitted."""
        observation = self.observation(step, instruction, seed=seed)
        reply = self.client.predict(observation)
        values = numpy.array(
            [[numpy.nan if x is None else x for x in row] for row in reply.actions],
            dtype=numpy.float64)
        arms = len(self.spec.arms)
        chunk = values[:, : arms * 26].reshape(len(values), arms, 26).transpose(1, 0, 2)
        self.previous = Observation(observation.request_id, observation.instruction,
                                    observation.timestamp_s, observation.images,
                                    observation.state)
        self.calls += 1
        return chunk

    def decode(self, chunk: numpy.ndarray, step: Step) -> numpy.ndarray:
        """[arms, rows, 26] -> [rows, action_dim] in the simulator's own units."""
        rows = chunk.shape[1]
        actions = numpy.zeros((rows, self.spec.action_dim), dtype=numpy.float64)
        for index, arm in enumerate(self.spec.arms):
            values = chunk[index].copy()
            if arm.camera is not None:
                reference = rotation_from_extrinsics_6(step.poses[arm.camera])
                # Training turned base-frame increments into the camera frame as
                # rows @ R; this undoes it.
                values[:, DPOS] = values[:, DPOS] @ reference.T
                values[:, DROT] = values[:, DROT] @ reference.T
            parts = {
                "dpos": values[:, DPOS] / numpy.asarray(self.translation),
                "drot": values[:, DROT] / self.rotation,
                "gripper": values[:, GRIPPER:GRIPPER + 1],
            }
            for name, block in parts.items():
                if name not in arm.columns:
                    continue
                if not numpy.isfinite(block).all():
                    raise ValueError(f"the server emitted no {name} for arm slot {arm.slot}")
                actions[:, list(arm.columns[name])] = block
        return actions

    def executed_rows(self, count: int) -> None:
        self.executed += int(count)
