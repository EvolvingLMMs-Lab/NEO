"""The NEO-VLA backend for the HTTP service.

It maps named observation fields onto the native request the Predictor reads
and returns the decoded chunk with the action fields the config describes.
See docs/inference.md for the config keys.
"""
from __future__ import annotations

import math

from .backend import ModelInfo
from .protocol import ActionChunk, ActionField, text


def name_list(config, key: str) -> list:
    names = config[key]
    if not isinstance(names, list):
        raise ValueError(f"{key} must be a list")
    for name in names:
        if name is not None:
            text(name)
    return names


class NeoBackend:
    def __init__(self, config):
        import numpy as np
        import torch

        from .loader import load_policy
        from .predictor import Predictor
        from .prompt import render_prompt

        self.np = np
        self.torch = torch
        self.model_id = text(config.get("model_id", "neo-vla"))
        self.embodiment = text(config["embodiment"])
        self.directory = config.get("directory")
        self.frame_ref = config["frame_ref"]
        self.arm_slots = config["arm_slots"]
        if (not isinstance(self.arm_slots, list) or not self.arm_slots
                or any(type(x) is not int or x < 0 for x in self.arm_slots)
                or len(set(self.arm_slots)) != len(self.arm_slots)):
            raise ValueError("arm_slots must be a list of distinct nonnegative integers")
        self.camera_names = name_list(config, "camera_names")
        self.state_fields = name_list(config, "state_fields")
        self.intrinsics_fields = name_list(config, "intrinsics_fields")
        self.extrinsics_fields = name_list(config, "extrinsics_fields")
        self.body_field = config.get("body_field")
        if len(self.state_fields) != len(self.arm_slots):
            raise ValueError("state_fields must have one entry per arm slot")
        if not len(self.intrinsics_fields) == len(self.extrinsics_fields) == len(
            self.camera_names
        ):
            raise ValueError("intrinsics_fields and extrinsics_fields must match camera_names")
        self.fields = tuple(ActionField.from_dict(x) for x in config["output_fields"])

        device = config.get("device", "cpu")
        self.policy, report = load_policy(
            config["checkpoint"],
            config_path=config.get("model_config"),
            device=device,
            weights=config.get("weights", "auto"),
        )
        self.predictor = Predictor(
            self.policy,
            arm_slots=self.arm_slots,
            min_gap=int(report.meta["history"]["min_gap"]),
            device=device,
        )
        self.action_hz = self.control_hz(config.get("action_hz"))

        template = config.get("prompt_template")
        if template is not None:
            if "{instruction}" not in text(template):
                raise ValueError("prompt_template must contain {instruction}")
            self.prompt = lambda task: template.replace("{instruction}", task)
        else:
            frames = [self.frame_ref] if isinstance(self.frame_ref, str) else self.frame_ref
            kinds = {"robot" if str(one).strip() == "robot" else "camera" for one in frames}
            if len(kinds) != 1:
                raise ValueError("arms in different frame kinds need an explicit prompt_template")
            source = config.get("prompt_source")
            self.prompt = lambda task: render_prompt(
                self.embodiment, frames[0], self.action_hz, task, source=source)

    def control_hz(self, stated) -> float:
        """The directory's control rate, or the configured one when there is no record."""
        record = None
        if self.directory is not None:
            record = self.predictor.embodiments.datasets.get(self.directory)
            if record is None:
                raise ValueError(f"the checkpoint holds no datasets record for {self.directory!r}")
        trained = None if record is None else record.control_hz
        if stated is None and trained is None:
            raise ValueError("action_hz is required when the checkpoint holds no control rate")
        if stated is not None and trained is not None and float(stated) != float(trained):
            raise ValueError(f"action_hz {stated} differs from the directory's {trained}")
        hz = float(trained if stated is None else stated)
        if not math.isfinite(hz) or hz <= 0:
            raise ValueError("action_hz must be positive")
        return hz

    def info(self):
        return ModelInfo(self.model_id, "NEO-VLA")

    @staticmethod
    def _state(observation, name):
        if name is None:
            return None
        found = observation.state.get(name)
        if found is None:
            raise ValueError(f"the observation has no state field {name!r}")
        return list(found.values)

    def _frame(self, observation):
        images = []
        for name in self.camera_names:
            if name is None:
                images.append(None)
                continue
            image = observation.images.get(name)
            if image is None:
                raise ValueError(f"the observation has no image {name!r}")
            images.append(self.np.frombuffer(image.data, dtype=self.np.uint8)
                          .reshape(image.height, image.width, 3).copy())
        return dict(
            images=images,
            state=[self._state(observation, name) for name in self.state_fields],
            body=self._state(observation, self.body_field),
            camera_extrinsics=[self._state(observation, name) for name in self.extrinsics_fields],
        )

    def predict(self, observation):
        current = self._frame(observation)
        history = None
        if observation.history:
            # The model reads one earlier step: the latest one sent.
            previous = observation.history[-1]
            history = dict(
                self._frame(previous),
                gap=round((observation.timestamp_s - previous.timestamp_s) * self.action_hz),
            )
        payload = dict(
            current,
            history=history,
            intrinsics=[self._state(observation, name) for name in self.intrinsics_fields],
            arm_slots=self.arm_slots,
            embodiment=self.embodiment,
            directory=self.directory,
            prompt=self.prompt(observation.instruction),
            frame_ref=self.frame_ref,
            dim_valid=None,
            body_dim_valid=None,
        )
        with self.torch.inference_mode():
            prepared = self.predictor.observation(payload)
            decoded = self.predictor.decode(prepared, self.predictor([prepared])[0])
        return self.chunk(observation, decoded)

    def chunk(self, observation, decoded) -> ActionChunk:
        """Arms in arm_slots order, then the body; invalid components become null."""
        np = self.np
        arm = np.asarray(decoded["actions"], dtype=float)
        valid = np.asarray(decoded["dim_valid"], dtype=bool)
        if arm.ndim != 3 or arm.shape[0] != len(self.arm_slots) or valid.shape != (
            arm.shape[0], arm.shape[2]
        ):
            raise ValueError("unexpected action shape from the model")
        values = arm.transpose(1, 0, 2).reshape(arm.shape[1], -1)
        mask = np.broadcast_to(valid[None], (arm.shape[1], *valid.shape)).reshape(
            arm.shape[1], -1)
        body = decoded.get("body_actions")
        if body is not None:
            body = np.asarray(body, dtype=float)
            body_mask = np.asarray(decoded["body_dim_valid"], dtype=bool)
            if body.ndim != 2 or body.shape[0] != arm.shape[1] or body_mask.shape != (
                body.shape[1],
            ):
                raise ValueError("unexpected body action shape from the model")
            values = np.concatenate((values, body), axis=1)
            mask = np.concatenate((mask, np.broadcast_to(body_mask, body.shape)), axis=1)
        if not np.isfinite(values[mask]).all():
            raise ValueError("the model returned a non-finite valid action")
        rows = tuple(
            tuple(float(value) if emitted else None for value, emitted in zip(row, flags))
            for row, flags in zip(values, mask)
        )
        return ActionChunk(observation.request_id, self.model_id, observation.timestamp_s,
                           self.action_hz, self.fields, rows)

    def close(self):
        self.predictor = None
        self.policy = None


def create_backend(config):
    return NeoBackend(config)
