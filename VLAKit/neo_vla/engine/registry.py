"""One name to factory registry.

A run names its model in the config and the model package registers a factory
under that name, so the engine can build a policy it knows nothing about.
Registering a name twice raises, and an unknown name reports what is
registered instead of returning nothing.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator


class Registry:
    """A name to factory map."""

    def __init__(self, name: str):
        self.name = name
        self.factories: dict[str, Callable] = {}

    def register(self, key: str) -> Callable[[Callable], Callable]:
        def decorate(factory: Callable) -> Callable:
            if key in self.factories:
                raise KeyError(
                    f"{self.name}: {key!r} is already registered.\n"
                    "Expected one factory per name.\n"
                    "Rename one of them or delete the duplicate registration."
                )
            self.factories[key] = factory
            return factory

        return decorate

    def get(self, key: str) -> Callable:
        if key not in self.factories:
            raise KeyError(
                f"{self.name}: no factory named {key!r}.\n"
                f"Expected one of {sorted(self.factories)}.\n"
                "Fix the name in the config, or import the module that registers it."
            )
        return self.factories[key]

    def __contains__(self, key: str) -> bool:
        return key in self.factories

    def __iter__(self) -> Iterator[str]:
        return iter(self.factories)

    def __repr__(self) -> str:
        return f"Registry({self.name}, {sorted(self.factories)})"


MODELS = Registry("models")
