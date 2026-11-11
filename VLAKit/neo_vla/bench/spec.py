"""A step's eval.json, and the neo-vla-serve config a benchmark runs against.

eval.json sits beside state.pt and states what a client needs and cannot
infer: which camera fills which slot and at what calibration,
the frame each arm's actions come back in, which state readings and action
columns the step uses, the gripper's raw open and closed readings, how many
centimetres and radians one unit of each benchmark's own action is, and each
benchmark's protocol (suites, budgets, horizon, seeds).

    python -m neo_vla.bench.spec --checkpoint STEP_DIR --device cuda > server.json
    neo-vla-serve --config server.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ARM_FIELDS = (
    ("joint_targets", 7, "rad", "absolute"),
    ("position_delta", 3, "cm", "delta"),
    ("rotation_delta", 3, "rad", "delta"),
    ("gripper", 1, "ratio", "absolute"),
    ("hand", 12, "native", "absolute"),
)
ARM_WIDTH = sum(width for _, width, _, _ in ARM_FIELDS)
# Column ranges of the model's 26-wide arm row.
DPOS = slice(7, 10)
DROT = slice(10, 13)
GRIPPER = 13


@dataclass(frozen=True)
class Camera:
    slot: int
    name: str
    shape: tuple[int, int]
    intrinsics: tuple[float, ...]


@dataclass(frozen=True)
class Arm:
    slot: int
    frame_ref: str
    raw_open: float
    raw_closed: float
    # Simulator action columns each part of the model's row fills.
    columns: dict[str, tuple[int, ...]]
    # Model action columns asked for, or None for every column the step decodes.
    model_columns: tuple[int, ...] | None = None
    # Whether the arm's joint readings are sent with its state.
    joints: bool = True

    @property
    def camera(self) -> int | None:
        """The camera slot this arm's actions are written in, or None for the base frame."""
        if self.frame_ref == "robot":
            return None
        return int(self.frame_ref.split(":", 1)[1]) if ":" in self.frame_ref else 0


@dataclass(frozen=True)
class EvalSpec:
    embodiment: str
    control_hz: float
    weights: str
    cameras: tuple[Camera, ...]
    arms: tuple[Arm, ...]
    action_dim: int
    units: dict[str, dict]
    benchmarks: dict[str, dict]

    @classmethod
    def load(cls, path: str | Path) -> EvalSpec:
        path = Path(path)
        if path.is_dir():
            path = path / "eval.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            embodiment=str(document["embodiment"]),
            control_hz=float(document["control_hz"]),
            weights=str(document.get("weights", "auto")),
            cameras=tuple(
                Camera(int(one["slot"]), str(one["name"]), tuple(one["shape"]),
                       tuple(float(x) for x in one["intrinsics"]))
                for one in document["cameras"]
            ),
            arms=tuple(
                Arm(int(one["slot"]), str(one["frame_ref"]), float(one["gripper"]["raw_open"]),
                    float(one["gripper"]["raw_closed"]),
                    {k: tuple(int(c) for c in v) for k, v in one["columns"].items()},
                    None if one.get("model_columns") is None
                    else tuple(int(c) for c in one["model_columns"]),
                    bool(one.get("joints", True)))
                for one in document["arms"]
            ),
            action_dim=int(document["action_dim"]),
            units=dict(document["units"]),
            benchmarks=dict(document["benchmarks"]),
        )

    def benchmark(self, name: str) -> dict:
        if name not in self.benchmarks:
            raise ValueError(f"this step's eval.json has no {name!r} protocol; it has "
                             f"{sorted(self.benchmarks)}.")
        return self.benchmarks[name]

    def unit_factors(self, name: str) -> tuple[tuple[float, float, float], float]:
        """(cm per unit of translation, rad per unit of rotation) of one benchmark's action."""
        found = self.units[name]
        return (tuple(float(x) for x in found["translation_cm_per_unit"]),
                float(found["rotation_rad_per_unit"]))


def camera_field(slot: int) -> str:
    return f"camera{slot}"


def intrinsics_field(slot: int) -> str:
    return f"intrinsics{slot}"


def pose_field(slot: int) -> str:
    return f"pose{slot}"


def arm_field(index: int) -> str:
    return f"arm{index}"


def server_config(spec: EvalSpec, checkpoint: str, *, n_views: int = 3,
                  device: str = "cuda") -> dict:
    """The neo-vla-serve backend config the bench clients talk to."""
    by_slot = {camera.slot: camera for camera in spec.cameras}
    views = range(n_views)
    frames = [arm.frame_ref for arm in spec.arms]
    output = []
    for index, arm in enumerate(spec.arms):
        for name, width, unit, mode in ARM_FIELDS:
            field = {"name": f"arm{index}_{name}", "width": width, "unit": unit, "mode": mode}
            if mode == "delta":
                field["frame"] = arm.frame_ref
            output.append(field)
    config = {
        "checkpoint": str(checkpoint),
        "device": device,
        "weights": spec.weights,
        "model_id": "neo-vla",
        "embodiment": spec.embodiment,
        "arm_slots": [arm.slot for arm in spec.arms],
        "frame_ref": frames[0] if len(set(frames)) == 1 else frames,
        "action_hz": spec.control_hz,
        "camera_names": [camera_field(s) if s in by_slot else None for s in views],
        "intrinsics_fields": [intrinsics_field(s) if s in by_slot else None for s in views],
        "extrinsics_fields": [pose_field(s) if s in by_slot else None for s in views],
        "state_fields": [arm_field(i) for i in range(len(spec.arms))],
        "output_fields": output,
    }
    if any(arm.model_columns is not None for arm in spec.arms):
        config["action_columns"] = [
            None if arm.model_columns is None else list(arm.model_columns) for arm in spec.arms]
    return config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Write the neo-vla-serve config for a step")
    parser.add_argument("--checkpoint", required=True, help="step directory holding eval.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    spec = EvalSpec.load(args.checkpoint)
    json.dump(server_config(spec, args.checkpoint, device=args.device), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
