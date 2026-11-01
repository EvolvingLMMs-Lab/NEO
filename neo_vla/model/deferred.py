"""Checks on device values, read once per training micro-batch instead of where they are made.

A refusal that reads a value computed on the device -- the perturbed state is
finite, the tower's pixels lie in [-1, 1] -- must wait for that value, and
every such wait in a forward stops the host from queueing work ahead of the
device. Inside flow_step a deferred() scope collects each check as a device
flag and its refusal; when the forward has been queued, one read of all the
flags decides, and a failing flag raises its own refusal, the same text the
check raised in place. A micro-batch with a chat pack opens one scope around
both halves (NeoVlaPolicy.forward), and flow_step's joins it. Outside a scope
(the server, a test, a tool) require reads its flag at once, as before.
"""
from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Callable, Iterator
from typing import NoReturn

import torch

_PENDING: contextvars.ContextVar[list | None] = contextvars.ContextVar("deferred_checks",
                                                                       default=None)


def require(ok: torch.Tensor, refuse: Callable[[], NoReturn]) -> None:
    """Raise refuse() unless ok, a one-element bool tensor; deferred inside a scope."""
    pending = _PENDING.get()
    if pending is None:
        if not bool(ok):
            refuse()
        return
    pending.append((ok.reshape(()), refuse))


@contextlib.contextmanager
def deferred() -> Iterator[None]:
    """Collect require's checks in the body; read them all, once, when it returns.

    Inside another scope it opens none of its own: its checks join the outer
    scope's and are read once, when that one returns (speed C-8+: a packed
    micro-batch's chat half and flow_step's scope, one read for both).
    """
    if _PENDING.get() is not None:
        yield
        return
    pending: list = []
    token = _PENDING.set(pending)
    try:
        yield
    finally:
        _PENDING.reset(token)
    if pending and not bool(torch.stack([ok for ok, _ in pending]).all()):
        for ok, refuse in pending:
            if not bool(ok):
                refuse()
