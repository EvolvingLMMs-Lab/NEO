"""The host copy each device tensor of a micro-batch was made from (speed C2).

The loop copies every micro-batch from pinned host memory to the device
without waiting for it. A check or a host computation that then reads a batch
field off the device makes the host wait for everything queued before it -- a
synchronisation -- and the device idles while the host works and launches the
next kernels. to_device records, for each device tensor it makes, the host
tensor it was copied from; host_of hands that tensor back, so the same check
reads the same values without the wait. A tensor the loop did not move (the
server's batch, a tensor computed on the device, a test's) is copied to the
host as before, so every caller reads the same numbers.

to_host_device is the way back: a host tensor computed per micro-batch goes to
the device from pinned memory without waiting, and is remembered too.
"""
from __future__ import annotations

import weakref
from typing import Any

import torch

# id(device tensor) -> (a weak reference to it, its host copy). The weak
# reference proves the id still names the tensor that was recorded, and its
# callback drops the record when that tensor goes, so a process that never
# calls forget (the server) holds no copy past its tensor.
_HOST: dict[int, tuple[weakref.ref, torch.Tensor]] = {}


def remember(device_tensor: torch.Tensor, host_tensor: torch.Tensor) -> None:
    """Record that device_tensor holds exactly host_tensor's values, while it lives."""
    key = id(device_tensor)

    def drop(reference: weakref.ref) -> None:
        held = _HOST.get(key)
        if held is not None and held[0] is reference:
            del _HOST[key]

    _HOST[key] = (weakref.ref(device_tensor, drop), host_tensor)


def forget() -> None:
    """Drop every record, releasing the host copies (the loop calls this per micro-batch)."""
    _HOST.clear()


def host_of(tensor: torch.Tensor) -> torch.Tensor:
    """The tensor's values on the host: its recorded copy, or a copy made now."""
    if tensor.device.type == "cpu":
        return tensor
    held = _HOST.get(id(tensor))
    if held is not None and held[0]() is tensor:
        return held[1]
    return tensor.detach().cpu()


def to_host_device(host_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """A host tensor on device, copied from pinned memory without waiting, and remembered.

    The caching host allocator records the copy on the pinned block and holds
    the block until the copy has run, so the source cannot be reused early.
    """
    if torch.device(device).type != "cuda":
        return host_tensor.to(device)
    pinned = host_tensor.pin_memory()
    moved = pinned.to(device, non_blocking=True)
    remember(moved, host_tensor)
    return moved


def to_device(batch: Any, device: torch.device) -> Any:
    """loop.to_device, recording each moved tensor's host copy for host_of."""
    if torch.is_tensor(batch):
        moved = batch.to(device, non_blocking=True)
        if moved is not batch:
            remember(moved, batch)
        return moved
    if isinstance(batch, dict):
        return {key: to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(value, device) for value in batch)
    return batch
