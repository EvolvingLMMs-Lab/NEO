"""Process group setup and rank helpers.

torchrun sets RANK, WORLD_SIZE, LOCAL_RANK and the rendezvous address; this
module reads those and nothing else. Before the group opens, rank and world
size come from those variables, so logging carries the right rank through a
rendezvous that can take minutes. A single process launched without them is
the ordinary case, not an error: every helper then answers as world size one
and the collectives become identity functions.
"""
from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

TIMEOUT_MINUTES_VAR = "NEO_VLA_DIST_TIMEOUT_MINUTES"
BACKEND_VAR = "NEO_VLA_DIST_BACKEND"
DEFAULT_TIMEOUT_MINUTES = 30


def is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def world_size() -> int:
    """The size of the group, or the size the launcher asked for."""
    if is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def rank() -> int:
    """This process's rank in the group, or the one the launcher gave it."""
    if is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_rank_zero() -> bool:
    return rank() == 0


def resolve_device() -> torch.device:
    """The device this process trains on: its own gpu, or the cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank())
    return torch.device("cpu")


def init_process_group() -> None:
    """Join the group torchrun launched. Idempotent, and a no-op alone."""
    if is_initialized():
        return
    if world_size() <= 1:
        return
    # The backend follows the hardware, but a machine can override it: a CUDA
    # box running a CPU job needs gloo, and nccl on a node with a bad fabric
    # hangs where gloo would run.
    default_backend = "nccl" if torch.cuda.is_available() else "gloo"
    backend = os.environ.get(BACKEND_VAR, default_backend)
    minutes = int(os.environ.get(TIMEOUT_MINUTES_VAR, str(DEFAULT_TIMEOUT_MINUTES)))
    # Ranks build their share of a large mixture before they get here and can
    # arrive minutes apart, so the rendezvous timeout is a machine setting.
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=minutes))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank())
    logger.info("joined %s group as rank %d of %d", backend, rank(), world_size())


def destroy_process_group() -> None:
    if is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if is_initialized():
        dist.barrier()


def all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    """Mean of `value` across ranks, on a copy."""
    if not is_initialized():
        return value.detach().clone()
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced / world_size()


def all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
    """Sum of `value` across ranks, on a copy; integer counts stay exact."""
    if not is_initialized():
        return value.detach().clone()
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced


def all_ranks_ok(local_ok: bool) -> bool:
    """True when every rank passed, so all of them take the same branch."""
    if not is_initialized():
        return local_ok
    flag = torch.tensor([1.0 if local_ok else 0.0])
    if torch.cuda.is_available():
        flag = flag.cuda()
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item() > 0.5)


def gather_objects(obj: Any) -> list:
    """One entry per rank, in rank order. A single entry when alone."""
    if not is_initialized():
        return [obj]
    gathered: list = [None] * world_size()
    dist.all_gather_object(gathered, obj)
    return gathered
