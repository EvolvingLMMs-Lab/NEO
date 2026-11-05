"""The wire format: observations in, action chunks out, as versioned JSON.

Nothing here imports torch or numpy, so a robot-side client can use it alone.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "1.0"
MAX_IMAGE_BYTES = 12 * 1024 * 1024


def strict_json(data):
    """json.loads that refuses NaN and Infinity."""

    def refuse(value):
        raise ValueError("Nonfinite JSON numbers are not supported")

    return json.loads(data, parse_constant=refuse)


def number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("A finite number is required")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("A finite number is required")
    return result


def text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A nonempty string is required")
    return value


@dataclass(frozen=True)
class Image:
    """Contiguous uint8 RGB bytes, row-major, without padding."""

    width: int
    height: int
    data: bytes

    def __post_init__(self):
        if (
            type(self.width) is not int
            or type(self.height) is not int
            or min(self.width, self.height) < 1
        ):
            raise ValueError("Image dimensions must be positive integers")
        expected = self.width * self.height * 3
        if (
            expected > MAX_IMAGE_BYTES
            or not isinstance(self.data, bytes)
            or len(self.data) != expected
        ):
            raise ValueError(
                "Image byte count differs from RGB dimensions or exceeds the image limit"
            )

    def to_dict(self):
        return dict(
            width=self.width,
            height=self.height,
            encoding="rgb8",
            data=base64.b64encode(self.data).decode("ascii"),
        )

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.get("encoding") != "rgb8":
            raise ValueError("Image encoding must be rgb8")
        encoded = value.get("data")
        if not isinstance(encoded, str) or len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
            raise ValueError("Invalid encoded image")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Invalid base64 image") from exc
        return cls(value.get("width"), value.get("height"), data)


@dataclass(frozen=True)
class StateField:
    """A named physical state vector; None denotes an unknown component."""

    values: tuple[float | None, ...]
    unit: str
    frame: str | None = None
    labels: tuple[str, ...] = ()

    def __post_init__(self):
        text(self.unit)
        if not self.values:
            raise ValueError("A state field must contain at least one component")
        for value in self.values:
            if value is not None:
                number(value)
        if self.frame is not None:
            text(self.frame)
        if self.labels and len(self.labels) != len(self.values):
            raise ValueError("State labels must match the component count")
        for label in self.labels:
            text(label)

    def to_dict(self):
        return dict(
            values=list(self.values), unit=self.unit, frame=self.frame, labels=list(self.labels)
        )

    @classmethod
    def from_dict(cls, value):
        return cls(
            tuple(value["values"]),
            value["unit"],
            value.get("frame"),
            tuple(value.get("labels", [])),
        )


@dataclass(frozen=True)
class Observation:
    request_id: str
    instruction: str
    timestamp_s: float
    images: dict[str, Image] = field(default_factory=dict)
    state: dict[str, StateField] = field(default_factory=dict)
    history: tuple[Observation, ...] = ()
    # Seeds the policy's sampling noise; None leaves the draw to the model.
    seed: int | None = None

    def __post_init__(self):
        if self.seed is not None and (type(self.seed) is not int or not
                                      0 <= self.seed < 2**63):
            raise ValueError("Seed must be a nonnegative 63-bit integer")
        if not isinstance(self.request_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]{1,128}", self.request_id
        ):
            raise ValueError("Invalid request identifier")
        text(self.instruction)
        number(self.timestamp_s)
        for name, image in self.images.items():
            text(name)
            if not isinstance(image, Image):
                raise ValueError("Invalid image value")
        for name, state in self.state.items():
            text(name)
            if not isinstance(state, StateField):
                raise ValueError("Invalid state value")
        previous = -math.inf
        for observation in self.history:
            if not isinstance(observation, Observation) or observation.history:
                raise ValueError("History entries must be observations without nested history")
            if not previous < observation.timestamp_s < self.timestamp_s:
                raise ValueError(
                    "History timestamps must increase and precede the current observation"
                )
            previous = observation.timestamp_s

    def to_dict(self):
        found = dict(
            schema_version=SCHEMA_VERSION,
            request_id=self.request_id,
            instruction=self.instruction,
            timestamp_s=self.timestamp_s,
            images={k: v.to_dict() for k, v in self.images.items()},
            state={k: v.to_dict() for k, v in self.state.items()},
            history=[x.to_dict() for x in self.history],
        )
        if self.seed is not None:
            found["seed"] = self.seed
        return found

    @classmethod
    def from_dict(cls, value, *, allow_history=True):
        if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported observation schema")
        history = value.get("history", [])
        if not isinstance(history, list) or (history and not allow_history):
            raise ValueError("Invalid history structure")
        return cls(
            value["request_id"],
            value["instruction"],
            value["timestamp_s"],
            {k: Image.from_dict(v) for k, v in value.get("images", {}).items()},
            {k: StateField.from_dict(v) for k, v in value.get("state", {}).items()},
            tuple(cls.from_dict(x, allow_history=False) for x in history),
            value.get("seed"),
        )


@dataclass(frozen=True)
class ActionField:
    """A contiguous action channel with an explicit physical interpretation."""

    name: str
    width: int
    unit: str
    mode: str
    frame: str | None = None
    labels: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self):
        text(self.name)
        text(self.unit)
        if type(self.width) is not int or self.width < 1:
            raise ValueError("Action width must be a positive integer")
        if self.mode not in ("absolute", "delta"):
            raise ValueError("Action mode must be absolute or delta")
        if self.frame is not None:
            text(self.frame)
        if self.labels and len(self.labels) != self.width:
            raise ValueError("Action labels must match the component count")
        for label in self.labels:
            text(label)
        if not isinstance(self.description, str):
            raise ValueError("Action description must be a string")

    def to_dict(self):
        return dict(
            name=self.name,
            width=self.width,
            unit=self.unit,
            mode=self.mode,
            frame=self.frame,
            labels=list(self.labels),
            description=self.description,
        )

    @classmethod
    def from_dict(cls, value):
        return cls(
            value["name"],
            value["width"],
            value["unit"],
            value["mode"],
            value.get("frame"),
            tuple(value.get("labels", [])),
            value.get("description", ""),
        )


@dataclass(frozen=True)
class ActionChunk:
    request_id: str
    model_id: str
    observation_timestamp_s: float
    action_hz: float
    fields: tuple[ActionField, ...]
    actions: tuple[tuple[float | None, ...], ...]

    def __post_init__(self):
        if not isinstance(self.request_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]{1,128}", self.request_id
        ):
            raise ValueError("Invalid request identifier")
        text(self.model_id)
        number(self.observation_timestamp_s)
        if number(self.action_hz) <= 0:
            raise ValueError("Action frequency must be positive")
        if (
            not self.fields
            or not self.actions
            or not all(isinstance(x, ActionField) for x in self.fields)
        ):
            raise ValueError("Action fields and rows must be present")
        if len({x.name for x in self.fields}) != len(self.fields):
            raise ValueError("Action field names must be unique")
        width = sum(x.width for x in self.fields)
        for row in self.actions:
            if len(row) != width:
                raise ValueError("Action width differs from the declared fields")
            for value in row:
                if value is not None:
                    number(value)

    def to_dict(self):
        return dict(
            schema_version=SCHEMA_VERSION,
            request_id=self.request_id,
            model_id=self.model_id,
            observation_timestamp_s=self.observation_timestamp_s,
            action_hz=self.action_hz,
            fields=[x.to_dict() for x in self.fields],
            actions=[list(x) for x in self.actions],
            valid=[[x is not None for x in row] for row in self.actions],
        )

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported action schema")
        result = cls(
            value["request_id"],
            value["model_id"],
            value["observation_timestamp_s"],
            value["action_hz"],
            tuple(ActionField.from_dict(x) for x in value["fields"]),
            tuple(tuple(x) for x in value["actions"]),
        )
        expected = [[x is not None for x in row] for row in result.actions]
        if value.get("valid") != expected:
            raise ValueError("Action validity must match the null components")
        if any(type(x) is not bool for row in value["valid"] for x in row):
            raise ValueError("Action validity entries must be booleans")
        return result
