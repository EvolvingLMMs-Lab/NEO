"""What a model has to provide to be served: info, predict and close."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Mapping, Protocol

from .protocol import ActionChunk, Observation, text


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    description: str = ""

    def __post_init__(self):
        text(self.model_id)
        if not isinstance(self.description, str):
            raise ValueError("Model description must be a string")

    def to_dict(self):
        return dict(model_id=self.model_id, description=self.description)


class Backend(Protocol):
    """A backend prepares model inputs and returns decoded physical actions."""

    def info(self) -> ModelInfo: ...
    def predict(self, observation: Observation) -> ActionChunk: ...
    def close(self) -> None: ...


def load_backend(factory: str, config: Mapping) -> Backend:
    """Load a selected module:factory extension."""
    module_name, separator, attribute = factory.partition(":")
    if not separator or not module_name or not attribute.isidentifier():
        raise ValueError("Backend must use module:factory syntax")
    create = getattr(importlib.import_module(module_name), attribute)
    backend = create(dict(config))
    if not all(callable(getattr(backend, name, None)) for name in ("info", "predict", "close")):
        raise TypeError("Backend does not implement the inference interface")
    if not isinstance(backend.info(), ModelInfo):
        raise TypeError("Backend metadata must be ModelInfo")
    return backend
