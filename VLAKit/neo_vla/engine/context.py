"""The per-step object handed to the model.

The loop builds one of these per micro-batch and calls the model as
model(batch, ctx). It carries the device, where the run stands in its step
budget, and the precision policy: bf16 autocast on a gpu, plain fp32 on a cpu,
which is what makes the tests exact. A model must not keep a context past the
step it was given in; it is a view of one step, not state.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch

from . import distributed


@dataclass(frozen=True)
class StepContext:
    device: torch.device
    step: int
    max_steps: int
    is_train: bool = True

    @property
    def rank(self) -> int:
        return distributed.rank()

    @property
    def world_size(self) -> int:
        return distributed.world_size()

    @property
    def is_rank_zero(self) -> bool:
        return distributed.is_rank_zero()

    def autocast(self):
        if self.device.type != "cuda":
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
