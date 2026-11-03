"""The data stack: a manifest in, the model's batch out.

The dataset discovers nothing. Everything it reads is named in one manifest
that a tool wrote (tools/prepare_data.py), so the corpus a run trained on is a
file that can be read back.

The names below resolve on first use, so importing one light submodule, such
as neo_vla.data.embodiment inside a simulator environment with no torch, does
not import the dataset and everything it needs.
"""
from __future__ import annotations

import importlib

# Each public name and the submodule that defines it.
_EXPORTS = {
    "CHAT_KEY": ".understanding",
    "SCHEMA_VERSION": ".dataset",
    "DatasetEntry": ".dataset",
    "Draw": ".mixture",
    "Manifest": ".dataset",
    "MixturePlan": ".mixture",
    "RobotDataset": ".dataset",
    "ShareGPT4VCoco": ".understanding",
    "load_manifest": ".dataset",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """Import the submodule that defines a public name, the first time it is asked for."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
