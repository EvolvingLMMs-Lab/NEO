"""Replicated and sharded data parallel behind one interface.

A strategy wraps the model, says how to skip gradient synchronisation while
gradients accumulate, clips gradients, and produces and restores the model and
optimizer state the checkpoint writes. The loop and the checkpoint call only
these methods, so neither of them asks which backend is in use. Both
strategies save a full, unsharded state, so the world size may change between
a save and the resume that reads it.
"""
from __future__ import annotations

import contextlib
import functools
import importlib
import logging
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel

from . import distributed

logger = logging.getLogger(__name__)


@runtime_checkable
class Strategy(Protocol):
    """What the loop and the checkpoint need from a parallel backend."""

    def wrap(self, model: nn.Module, device: torch.device) -> nn.Module: ...

    def unwrap(self, model: nn.Module) -> nn.Module: ...

    def no_sync(self, model: nn.Module): ...

    def clip_grad_norm(self, model: nn.Module, max_norm: float) -> torch.Tensor: ...

    def model_state_dict(self, model: nn.Module) -> dict: ...

    def load_model_state_dict(self, model: nn.Module, state: Mapping) -> None: ...

    def optim_state_dict(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> dict: ...

    def load_optim_state_dict(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, state: Mapping
    ) -> None: ...


class DDPStrategy:
    """One replica per rank; gradients are averaged at the sync step."""

    def __init__(
        self,
        find_unused_parameters: bool = False,
        gradient_as_bucket_view: bool = False,
        bucket_cap_mb: float | None = None,
    ):
        self.find_unused_parameters = bool(find_unused_parameters)
        # DDP's allreduce buckets are a SECOND full copy of the gradients unless
        # the gradients ARE the buckets. For the skip28 full finetune that copy
        # is 11.71 GiB a rank -- the room a rank-zero fp32 EMA needs and does
        # not have. Off by default: every config that passes no strategy_options
        # gets exactly the DDP it got before.
        self.gradient_as_bucket_view = bool(gradient_as_bucket_view)
        # The allreduce bucket size in MiB (speed C-3). None passes nothing, so
        # DDP keeps its own default: 25 MiB buckets after a FIRST bucket of
        # dist._DEFAULT_FIRST_BUCKET_BYTES (1 MiB). An explicit 25 is not that:
        # torch 2.6 sizes the first bucket at the stated cap too
        # (bucket_bytes_cap_default, torch/nn/parallel/distributed.py:796-802
        # and :1211-1215). So
        # every config that states no bucket_cap_mb gets the DDP it got before.
        if bucket_cap_mb is not None and (
                isinstance(bucket_cap_mb, bool) or not isinstance(bucket_cap_mb, (int, float))
                or not bucket_cap_mb > 0):
            raise ValueError(
                f"bucket_cap_mb is {bucket_cap_mb!r}.\n"
                "Expected a positive number of MiB, or null for DDP's own 25 MiB default.\n"
                "Fix trainer.strategy_options.bucket_cap_mb."
            )
        self.bucket_cap_mb = None if bucket_cap_mb is None else float(bucket_cap_mb)

    def wrap(self, model: nn.Module, device: torch.device) -> nn.Module:
        model = model.to(device)
        if not distributed.is_initialized():
            return model
        if self.gradient_as_bucket_view and not self.find_unused_parameters:
            # DDP points a gradient into its bucket only when the gradient
            # exists as DDP is built (reducer.cpp initialize_bucket_views) or
            # at the first sync. Without one, a process's first accumulation
            # window collects every gradient in a second fp32 copy until its
            # sync micro-batch: 11.72 GiB more at stage-2 size, on top of the
            # AdamW moments a resume has already loaded, which put a resumed
            # 8-rank packed run out of memory (74.5 GiB allocated where the
            # uninterrupted run peaked at 69.96).
            # Zeros are what keeps_gradient_buckets leaves after every step,
            # and with every parameter used each step (find_unused_parameters
            # off) 0 + g is g: the sums are today's.
            for parameter in model.parameters():
                if parameter.requires_grad and parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
        on_cuda = device.type == "cuda"
        stated = {} if self.bucket_cap_mb is None else {"bucket_cap_mb": self.bucket_cap_mb}
        return DistributedDataParallel(
            model,
            device_ids=[device.index] if on_cuda else None,
            output_device=device.index if on_cuda else None,
            find_unused_parameters=self.find_unused_parameters,
            gradient_as_bucket_view=self.gradient_as_bucket_view,
            **stated,
        )

    def unwrap(self, model: nn.Module) -> nn.Module:
        if isinstance(model, DistributedDataParallel):
            return model.module
        return model

    def no_sync(self, model: nn.Module):
        if isinstance(model, DistributedDataParallel):
            return model.no_sync()
        return contextlib.nullcontext()

    def clip_grad_norm(self, model: nn.Module, max_norm: float) -> torch.Tensor:
        return nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    def model_state_dict(self, model: nn.Module) -> dict:
        return self.unwrap(model).state_dict()

    def load_model_state_dict(self, model: nn.Module, state: Mapping) -> None:
        self.unwrap(model).load_state_dict(state)

    def optim_state_dict(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> dict:
        return optimizer.state_dict()

    def load_optim_state_dict(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, state: Mapping
    ) -> None:
        optimizer.load_state_dict(state)


def resolve_layer_classes(paths: Sequence[str]) -> set:
    """Import 'package.module.ClassName' strings into the classes they name."""
    classes = set()
    for path in paths:
        module_name, _, class_name = str(path).rpartition(".")
        if not module_name:
            raise ValueError(
                f"transformer_layer_cls entry is {path!r}, which names no module.\n"
                "Expected a dotted path such as package.module.LayerClass.\n"
                "Write the full import path to the layer class."
            )
        classes.add(getattr(importlib.import_module(module_name), class_name))
    return classes


class FSDPStrategy:
    """Parameters, gradients and optimizer state sharded across the ranks."""

    def __init__(
        self,
        sharding_strategy: str = "FULL_SHARD",
        auto_wrap_policy: str = "none",
        transformer_layer_cls: Sequence[str] = (),
        cpu_offload: bool = False,
        forward_prefetch: bool = False,
        backward_prefetch: bool = True,
    ):
        if sharding_strategy not in ShardingStrategy.__members__:
            raise ValueError(
                f"sharding_strategy is {sharding_strategy!r}.\n"
                f"Expected one of {sorted(ShardingStrategy.__members__)}.\n"
                "Fix trainer.strategy_options.sharding_strategy."
            )
        if auto_wrap_policy not in ("none", "transformer"):
            raise ValueError(
                f"auto_wrap_policy is {auto_wrap_policy!r}.\n"
                "Expected 'none' for a single unit or 'transformer' for one unit per layer.\n"
                "Fix trainer.strategy_options.auto_wrap_policy."
            )
        if auto_wrap_policy == "transformer" and not transformer_layer_cls:
            raise ValueError(
                "auto_wrap_policy is 'transformer' with an empty transformer_layer_cls.\n"
                "Expected at least one dotted path to the layer class to shard on.\n"
                "Set trainer.strategy_options.transformer_layer_cls."
            )
        self.sharding_strategy = sharding_strategy
        self.auto_wrap_policy = auto_wrap_policy
        self.transformer_layer_cls = list(transformer_layer_cls)
        self.cpu_offload = bool(cpu_offload)
        self.forward_prefetch = bool(forward_prefetch)
        self.backward_prefetch = bool(backward_prefetch)
        self.offload_state = False

    def build_policy(self):
        if self.auto_wrap_policy == "none":
            return None
        return functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=resolve_layer_classes(self.transformer_layer_cls),
        )

    def wrap(self, model: nn.Module, device: torch.device) -> nn.Module:
        if not distributed.is_initialized():
            raise RuntimeError(
                "the fsdp strategy was selected with one process.\n"
                "Expected a process group, which torchrun creates.\n"
                "Launch with torchrun, or select trainer.strategy=ddp."
            )
        on_cuda = device.type == "cuda"
        # A full state dict is gathered into host memory only when the shards
        # live on a gpu. Asking for the same offload with parameters already on
        # the cpu buys nothing and crashes the gloo gather.
        self.offload_state = on_cuda
        precision = None
        if on_cuda:
            # Gradients reduce in fp32: a long pretrain makes many small
            # updates, and bf16 rounds them away.
            precision = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,
            )
        return FullyShardedDataParallel(
            model.to(device),
            sharding_strategy=ShardingStrategy[self.sharding_strategy],
            auto_wrap_policy=self.build_policy(),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE if self.backward_prefetch else None,
            forward_prefetch=self.forward_prefetch,
            mixed_precision=precision,
            use_orig_params=True,
            # The broadcast of rank zero's weights needs them on a gpu, and on
            # one host every rank builds the same weights anyway.
            sync_module_states=on_cuda,
            cpu_offload=CPUOffload(offload_params=self.cpu_offload),
            device_id=device,
        )

    def unwrap(self, model: nn.Module) -> nn.Module:
        if isinstance(model, FullyShardedDataParallel):
            return model.module
        return model

    def no_sync(self, model: nn.Module):
        if isinstance(model, FullyShardedDataParallel):
            return model.no_sync()
        return contextlib.nullcontext()

    def clip_grad_norm(self, model: nn.Module, max_norm: float) -> torch.Tensor:
        if isinstance(model, FullyShardedDataParallel):
            return model.clip_grad_norm_(max_norm)
        return nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    def full_state(self, model: nn.Module, rank0_only: bool):
        """Read or write the unsharded state. Every rank enters together."""
        return FullyShardedDataParallel.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=self.offload_state, rank0_only=rank0_only),
            FullOptimStateDictConfig(offload_to_cpu=self.offload_state, rank0_only=rank0_only),
        )

    def model_state_dict(self, model: nn.Module) -> dict:
        with self.full_state(model, rank0_only=True):
            return model.state_dict()

    def load_model_state_dict(self, model: nn.Module, state: Mapping) -> None:
        with self.full_state(model, rank0_only=False):
            model.load_state_dict(state)

    def optim_state_dict(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> dict:
        with self.full_state(model, rank0_only=True):
            return FullyShardedDataParallel.optim_state_dict(model, optimizer)

    def load_optim_state_dict(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, state: Mapping
    ) -> None:
        with self.full_state(model, rank0_only=False):
            optimizer.load_state_dict(
                FullyShardedDataParallel.optim_state_dict_to_load(model, optimizer, state)
            )


def build_strategy(name: str, options: Mapping[str, Any] | None = None) -> Strategy:
    """Config name to strategy. An unknown option raises here, not at step one."""
    settings = dict(options or {})
    if name == "ddp":
        return DDPStrategy(**settings)
    if name == "fsdp":
        return FSDPStrategy(**settings)
    raise ValueError(
        f"trainer.strategy is {name!r}.\n"
        "Expected 'ddp' or 'fsdp'.\n"
        "Fix trainer.strategy in the config."
    )
