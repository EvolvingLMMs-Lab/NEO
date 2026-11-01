"""One step-bounded training loop.

The loop owns the optimizer. It moves a batch to the device, calls the model
as model(batch, ctx), reads loss out of the dict that comes back, divides by
the accumulation count, backpropagates, clips, steps, and updates the moving
average. It stops when the step budget is spent or the sampler runs dry, and
it has no notion of an epoch. What trains is the model's business; the loop
only checks that the parameters the model declares trainable are exactly the
ones carrying a gradient, and that the optimizer holds those and nothing else.
The model must provide forward(batch, ctx), trainable_parameter_names() and
configure_optimizers(cfg).
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, nullcontext
from itertools import islice
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from . import checkpoint, distributed, hostcopy
from .context import StepContext
from .logging import ProgressLogger
from .sampler import StepItemSampler

logger = logging.getLogger(__name__)

# Diagnostics appended to the progress line when the step produced them, in
# this order. The list is an allowlist and not "every extra key" on purpose: a
# run without the auxiliary loss produces neither of these, so its log line is
# the one it always wrote, and a run with it can be watched without a grading
# slot. The objective's other diagnostics stay out of the line for that same
# reason. state_clipped is the v6 policy's count of state positions the band
# clipped in a micro-batch, averaged over the window: a rising count is a key
# whose q01/q99 no longer covers what the robot does. Each is averaged over
# the micro-batches of the window that reported it: a packing run that names
# its chat micro-batches (shape B, 2 of 8) reports cotrain_loss on those only,
# and a mean over the whole window would read a quarter of the chat loss. A
# term every micro-batch reports reads exactly as before.
STEP_DIAGNOSTICS = (
    "flow_mse_loss", "dct_loss", "cotrain_loss", "cotrain_token_ce", "state_clipped",
)

# Written last, by a run that packs its chat half: each pack counter summed over the log
# window and over EVERY rank,
# then the fill, real tokens over packs x pack_tokens. Unlike the terms above
# they are sums of the whole job, so the all_reduce is taken by every rank on
# every logging step, right after progress.due and before anything rank-local
#: ProgressLogger.update returns early off rank 0, and
# the loss's finiteness is this rank's own, so a collective in either would
# hang the job. The model returns a pack's counters under PACK_COUNTER_OUTPUT, a
# vector in chat_pack.COUNTERS order. The names restate chat_pack.COUNTER_PREFIX
# and COUNTERS, so the engine imports no model module; a test holds them equal.
PACK_COUNTER_OUTPUT = "chat_pack_counters"
PACK_DIAGNOSTICS = (
    "chat_pack_rows", "chat_pack_tokens", "chat_pack_images", "chat_pack_dropped_overlong",
    "chat_pack_spilled", "chat_pack_fill",
)
PACK_TOKENS_READING = "chat_pack_tokens"
PACK_FILL = "chat_pack_fill"

# Written after those: the v7 per-group losses, the reading the 2:1:1:1 group
# weights can be checked against. The objective
# returns each as a micro-batch's pooled mean over its loss triples (slot, step,
# dim) and, under its name plus GROUP_TRIPLES_SUFFIX, how many triples that mean
# is over: 0.0 over 0 triples is a group the micro-batch does not hold. Pooled
# over the window by that count, so a group that some micro-batches lack (the
# hand, the body) reads at its own scale; a group no micro-batch of the window
# held is written None (n/a on the line), never 0. The decoder imports these
# names (FlowDecoder.loss). Like the terms above it is this rank's window:
# reading every rank's would cost a collective per line.
GROUP_DIAGNOSTICS = ("loss_ee", "loss_gripper", "loss_joint", "loss_hand", "loss_body_base",
                     "loss_body_posture")
GROUP_TRIPLES_SUFFIX = "_triples"

# The intra-op threads a host-held average's update runs at, inside
# update only; the process keeps its own count (train.sh: OMP_NUM_THREADS 4)
# everywhere else. mul_ and add_ are elementwise, so the thread count moves only
# where each thread's range starts, never a result: 1, 4, 16 and 32 threads gave
# bitwise the same shadow on AVX512 (0 of 337M elements differed), and
# tests/engine/test_ema_staging.py holds 4 against 16.
EMA_THREADS = 16
# Each tracked tensor's view into the pinned staging buffer starts on a multiple
# of this many float32 elements (64 bytes), so no view straddles a cache line
# it shares with its neighbour. The values copied are the same either way.
EMA_STAGING_ALIGN = 16


@contextmanager
def intra_op_threads(count: int) -> Iterator[None]:
    """torch's intra-op thread count set to count in the body, the old count restored after.

    The count is the process's while the body runs: another thread that starts
    its first OpenMP region then sizes its pool from it (the only such thread
    seen is the speed harness's digest thread [inferred]). The training loop
    runs no compiled call inside the body.
    """
    before = torch.get_num_threads()
    torch.set_num_threads(int(count))
    try:
        yield
    finally:
        torch.set_num_threads(before)


def pack_reading(window: torch.Tensor | None, pack_tokens: int,
                 device: torch.device) -> dict[str, float | None]:
    """A rank's summed pack counters and pack count -> PACK_DIAGNOSTICS over every rank.

    window is the counters (chat_pack.COUNTERS order) then the packs this
    rank ran in the window, or None for none. One all_reduce, which every rank
    must make on the same steps. The fill is None when no rank ran a pack.
    """
    counted = PACK_DIAGNOSTICS[:-1]
    if window is None:
        window = torch.zeros(len(counted) + 1, dtype=torch.long, device=device)
    total = distributed.all_reduce_sum(window).tolist()
    reading: dict[str, float | None] = {
        name: float(value) for name, value in zip(counted, total)}
    packs = total[-1]
    reading[PACK_FILL] = (
        reading[PACK_TOKENS_READING] / (packs * int(pack_tokens)) if packs else None)
    return reading


def group_reading(output: dict, name: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    """(pooled mean x triples, triples) of one group diagnostic, or None when not reported."""
    reading = output.get(name)
    if reading is None:
        return None
    count = output.get(name + GROUP_TRIPLES_SUFFIX)
    if count is None:
        raise ValueError(
            f"the model reports {name} without {name + GROUP_TRIPLES_SUFFIX}.\n"
            "Expected each group loss beside the count of loss triples it is a mean over: "
            "a mean over no triple reads 0.0, which the log must not show as a loss.\n"
            "Return the count from the objective, as FlowDecoder.loss does."
        )
    count = count.detach().float()
    return reading.detach().float() * count, count


@dataclass(frozen=True)
class LoopSpec:
    """The numbers the loop runs on, all resolved before it starts."""

    max_steps: int
    accumulation: int
    gradient_clip: float = 0.0
    log_every: int = 20
    save_every: int = 0
    keep_checkpoints: int = 2
    keep_skipped: int = 0
    samples_per_step: int = 0


@dataclass
class LoopResult:
    step: int
    losses: list[float] = field(default_factory=list)
    skipped: int = 0


class Ema:
    """Moving average of the trainable parameters, kept on the parameter device.

    Rank zero keeps the only copy: the replicas are identical, and in the full
    finetune arm this shadow is what sets peak memory. The average is held in
    fp32 whatever the parameters are: at bf16 a decay close to one rounds each
    update back to the value it started from and the average never moves.
    """

    def __init__(self, module: nn.Module, decay: float, exclude_prefix: str | None = None, device: str | None = None):
        if not 0.0 < decay < 1.0:
            raise ValueError(
                f"ema decay is {decay}.\n"
                "Expected a number strictly between 0 and 1.\n"
                "Fix trainer.ema_decay, or set it to null to train without an average."
            )
        self.decay = float(decay)
        # exclude_prefix: parameters whose name starts with it are NOT averaged.
        # None reproduces the original comprehension exactly. It exists because a
        # full-parameter run's shadow is 3.14B fp32 params on rank zero (~12.6 GB)
        # where a frozen-backbone run's is 0.82B, and runner.initialise_from
        # OVERLAYS the average on the raw weights -- so skipping the backbone
        # gives a downstream arm the same thing a frozen-backbone pretrain does:
        # averaged expert, raw tower. Setting ema_decay null would free the same
        # memory but drop the 'ema' entry entirely, and init_use_ema RAISES on a
        # checkpoint that has none.
        self.exclude_prefix = str(exclude_prefix) if exclude_prefix else None
        every = {
            name: param
            for name, param in module.named_parameters()
            if param.requires_grad
        }
        if self.exclude_prefix is None:
            self.tracked = every
        else:
            self.tracked = {
                name: param
                for name, param in every.items()
                if not name.startswith(self.exclude_prefix)
            }
            # A prefix that matches nothing is a typo that silently averages
            # everything; one that matches everything silently averages nothing.
            # Both look like success in the log, so both refuse here.
            if len(self.tracked) == len(every):
                raise ValueError(
                    f"ema exclude_prefix is {self.exclude_prefix!r} and NO trainable "
                    f"parameter's name begins with it.\n"
                    f"Expected a prefix present in this model; {len(every)} parameters "
                    f"are trainable and none matched.\n"
                    "Fix trainer.ema_exclude_prefix, or set it null to average everything."
                )
            if not self.tracked:
                raise ValueError(
                    f"ema exclude_prefix is {self.exclude_prefix!r} and EVERY trainable "
                    f"parameter's name begins with it, so nothing would be averaged.\n"
                    "Expected a prefix covering part of the model.\n"
                    "Fix trainer.ema_exclude_prefix, or set trainer.ema_decay null."
                )
        # device: where the average LIVES. None => the parameter device, as before.
        # "cpu" moves ~3.3 GB of fp32 shadow out of HBM; it is touched once per
        # optimizer step and by nothing else, so the cost is one device->host copy
        # per step against a step already measured in seconds. At world 64 the save
        # peaked at 80,774 of 81,559 MiB with the shadow resident, so this is the
        # difference between keeping an average and not having one.
        self.device = torch.device(device) if device else None
        self.shadow = {
            name: (
                param.detach().to(torch.float32).to(self.device).clone()
                if self.device is not None
                else param.detach().to(torch.float32).clone()
            )
            for name, param in self.tracked.items()
        }
        # The pinned host copy of the tracked parameters (speed C-1a), built at
        # the first update that needs it: one flat float32 buffer and a view per
        # tensor. None until then, and for good when the average is not held on
        # the host of a cuda model.
        self.staging: torch.Tensor | None = None
        self.staged: dict[str, torch.Tensor] = {}

    def stages_on_host(self) -> bool:
        """Whether update copies the parameters through the pinned buffer.

        The average lives on the cpu (trainer.ema_device cpu) and the parameters
        on a cuda device. Anywhere else update reads them as it always did.
        """
        if self.device is None or self.device.type != "cpu" or not self.tracked:
            return False
        return next(iter(self.tracked.values())).is_cuda

    def stage(self) -> dict[str, torch.Tensor]:
        """The tracked parameters in float32 on the host: one copy each, one wait for all.

        Each parameter is copied into its view of the pinned buffer without
        waiting, all on the current stream, which holds them behind the
        optimizer step that made them; one event then waits for the last copy.
        The values are the bytes the pageable copy of the old update read, so
        the average comes out bitwise the same, without allocating and
        faulting in 3.3 GB of fresh host memory each step (stage 2: 0.82 B
        tracked parameters). The buffer is held for the run, and torch's
        caching host allocator rounds a pinned block up to a power of two
        (torch 2.6 CachingHostAllocator.h:132), so stage 2's 3.28 GB holds
        4 GiB of pinned host memory on the rank that keeps the average
        [derived].
        """
        if self.staging is None:
            offsets, total = {}, 0
            for name, param in self.tracked.items():
                offsets[name] = total
                total += -(-param.numel() // EMA_STAGING_ALIGN) * EMA_STAGING_ALIGN
            self.staging = torch.empty(total, dtype=torch.float32, pin_memory=True)
            self.staged = {
                name: self.staging[offsets[name]:offsets[name] + param.numel()].view(param.shape)
                for name, param in self.tracked.items()
            }
        device = next(iter(self.tracked.values())).device
        for name, param in self.tracked.items():
            self.staged[name].copy_(param.detach().to(torch.float32), non_blocking=True)
        copied = torch.cuda.Event()
        copied.record(torch.cuda.current_stream(device))
        copied.synchronize()
        return self.staged

    @torch.no_grad()
    def update(self) -> None:
        if self.stages_on_host():
            sources = self.stage()
            with intra_op_threads(EMA_THREADS):
                for name, source in sources.items():
                    self.shadow[name].mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            return
        for name, param in self.tracked.items():
            source = param.detach().to(torch.float32)
            if self.device is not None:
                # non_blocking is deliberately NOT used here: the add_ below reads
                # this tensor immediately. A cuda model with the average on the
                # host goes through stage() instead, which waits once for all.
                source = source.to(self.device)
            self.shadow[name].mul_(self.decay).add_(
                source, alpha=1.0 - self.decay
            )

    def state_dict(self) -> dict:
        return {name: value.detach().cpu() for name, value in self.shadow.items()}

    def load_state_dict(self, state: dict) -> None:
        missing = sorted(set(self.shadow) - set(state))
        if missing:
            raise ValueError(
                f"the average on disk is missing {len(missing)} parameters, first {missing[:3]}.\n"
                "Expected one entry per trainable parameter.\n"
                "Resume a checkpoint written by this model, or train from scratch."
            )
        for name, value in self.shadow.items():
            value.copy_(state[name].to(device=value.device, dtype=value.dtype))


def to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(value, device) for value in batch)
    return batch


# What a sharding wrap splices into every parameter path it owns. The mode
# declares names against the unwrapped tree, so the two are compared with this
# taken back out; leaving it in reports every wrapped parameter as both missing
# and extra, which is what an fsdp run met before.
FSDP_SEGMENT = "_fsdp_wrapped_module."


def unwrapped_name(name: str) -> str:
    """A parameter's name as the model declared it, before any sharding wrap."""
    return name.replace(FSDP_SEGMENT, "")


def keeps_gradient_buckets(model: nn.Module) -> bool:
    """Whether the gradients are DDP's buckets, so a step zeroes them in place.

    With gradient_as_bucket_view the buckets are the gradients. Set to None
    after a step, the next accumulation window allocates a second fp32 copy
    that the sync micro-batch then copies into the buckets: 11.72 GiB a rank at
    stage-2 size, the difference between 4 chat rows of 4096 fitting an 80 GB
    card or not. Zeroed in place they stay the buckets, and the sums are the
    same in the same order. Anywhere else a gradient set to None keeps its
    meaning: AdamW skips a parameter that has none.
    """
    return isinstance(model, DistributedDataParallel) and bool(model.gradient_as_bucket_view)


def step_windows(
    loader: Iterable, accumulation: int, prepare: Callable[[list], None],
    done: Callable[[], bool],
) -> Iterator:
    """The loader's micro-batches, a step's worth at a time, each step prepared first.

    A model that sets prepares_steps is shown a step's micro-batches before the
    first of them runs, so that a quantity of the whole step costs one
    collective at its start and none inside a forward. A step the data cannot
    fill is dropped unrun, as the loop drops its gradients otherwise.
    """
    batches = iter(loader)
    while not done():
        window = list(islice(batches, accumulation))
        if len(window) < accumulation:
            if window:
                logger.warning(
                    "the data ran out %d micro-batches into a step; that step was not run",
                    len(window),
                )
            return
        prepare(window)
        yield from window


def micro_batches(loader: Iterable, accumulation: int) -> Iterable:
    """The loader's micro-batches in order: the loader itself, or its step items flattened.

    A packing run's loader yields one optimizer step per item, a list of its
    accumulation micro-batches (runner.build_loader); the loop sees the same stream of
    micro-batches either way, and
    step_windows still drops a short tail as before. Any other loader is
    returned as it is.
    """
    if not (hasattr(loader, "sampler") and isinstance(loader.sampler, StepItemSampler)):
        return loader
    return _flattened(loader, int(accumulation))


def _flattened(loader: Iterable, accumulation: int) -> Iterator:
    for item in loader:
        if not isinstance(item, list) or len(item) != accumulation:
            raise ValueError(
                f"a step item holds {len(item) if isinstance(item, list) else type(item).__name__} "
                f"micro-batches.\n"
                f"Expected a list of accumulation = {accumulation}: one optimizer step.\n"
                "Build the loader with runner.build_loader, which splits each step item."
            )
        yield from item


def check_trainable(module: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """What the mode declares trainable must be what requires a gradient."""
    declared = set(module.trainable_parameter_names())
    live = {
        unwrapped_name(name)
        for name, param in module.named_parameters()
        if param.requires_grad
    }
    if declared != live:
        frozen = sorted(declared - live)[:3]
        extra = sorted(live - declared)[:3]
        raise RuntimeError(
            f"{len(declared)} parameters are declared trainable, {len(live)} require a gradient.\n"
            f"Declared but frozen: {frozen}; requires a gradient but not declared: {extra}.\n"
            "Set requires_grad in the model to match what the mode declares."
        )
    in_optimizer = {
        id(param) for group in optimizer.param_groups for param in group["params"]
    }
    trainable = {id(param) for param in module.parameters() if param.requires_grad}
    if in_optimizer != trainable:
        raise RuntimeError(
            f"the optimizer holds {len(in_optimizer)} parameters, "
            f"{len(trainable)} require a gradient.\n"
            "Expected the optimizer to hold exactly the trainable parameters.\n"
            "Build the parameter groups from those that require a gradient."
        )


def train(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    strategy,
    spec: LoopSpec,
    device: torch.device,
    checkpoint_root: str | Path,
    start_step: int = 0,
    ema: Ema | None = None,
) -> LoopResult:
    """Train until step `spec.max_steps`, then write a final checkpoint."""
    module = strategy.unwrap(model)
    check_trainable(module, optimizer)
    skipped_root = Path(checkpoint_root).parent / "skipped_states"
    progress = ProgressLogger(
        spec.max_steps, spec.log_every, spec.samples_per_step, start_step
    )
    model.train()
    set_to_none = not keeps_gradient_buckets(model)
    optimizer.zero_grad(set_to_none=set_to_none)

    result = LoopResult(step=int(start_step))
    saved_step = int(start_step)
    micro = 0
    window_loss: torch.Tensor | None = None
    window_count = 0
    # Summed on the device like window_loss, and read once when the line is
    # written, so watching the auxiliary term costs no extra synchronisation.
    window_extra: dict[str, torch.Tensor] = {}
    # How many micro-batches of the window reported each of them.
    window_extra_count: dict[str, int] = {}
    # Per group, the window's sum of mean x triples and of triples, on the device.
    window_groups: dict[str, list[torch.Tensor]] = {}
    # A packing run's counters and pack count, summed on the device and read on
    # every rank at each log step (pack_reading). The run packs or not as a
    # whole, by its chat corpus, which the policy's dataset check records.
    chat_pack = module.chat_pack if hasattr(module, "chat_pack") else None
    window_pack: torch.Tensor | None = None
    # One flag a micro-batch, kept on the device so that recording it costs no
    # synchronisation. They are read only when a step is skipped, which is the
    # only time anyone wants to know which micro-batch was the bad one.
    micro_finite: list[torch.Tensor] = []
    stream = micro_batches(loader, spec.accumulation)
    if hasattr(module, "prepares_steps") and module.prepares_steps:
        # The window is prepared once the step before it has run, so result.step
        # is the global step the window trains (the step item's own number).
        stream = step_windows(
            stream, spec.accumulation,
            lambda window: module.prepare_step(window, step=result.step),
            lambda: result.step >= spec.max_steps,
        )

    for batch in stream:
        if result.step >= spec.max_steps:
            break
        # The host copies of the last micro-batch go; this one's are recorded
        # as it moves, for the model's checks to read (engine.hostcopy).
        hostcopy.forget()
        batch = hostcopy.to_device(batch, device)
        context = StepContext(device=device, step=result.step, max_steps=spec.max_steps)
        is_sync = (micro + 1) % spec.accumulation == 0
        with nullcontext() if is_sync else strategy.no_sync(model):
            with context.autocast():
                output = model(batch, context)
            (output["loss"] / spec.accumulation).backward()
        detached = output["loss"].detach()
        micro_finite.append(torch.isfinite(detached))
        window_loss = detached if window_loss is None else window_loss + detached
        for name in STEP_DIAGNOSTICS:
            reading = output.get(name)
            if reading is not None:
                held = window_extra.get(name)
                reading = reading.detach()
                window_extra[name] = reading if held is None else held + reading
                window_extra_count[name] = window_extra_count.get(name, 0) + 1
        for name in GROUP_DIAGNOSTICS:
            pair = group_reading(output, name)
            if pair is None:
                continue
            held = window_groups.get(name)
            window_groups[name] = list(pair) if held is None else [
                held[0] + pair[0], held[1] + pair[1]]
        counters = output.get(PACK_COUNTER_OUTPUT)
        if counters is not None:
            reading = torch.cat([counters.detach().long(), counters.new_ones(1, dtype=torch.long)])
            window_pack = reading if window_pack is None else window_pack + reading
        window_count += 1
        micro += 1
        if not is_sync:
            continue

        grad_norm = None
        if spec.gradient_clip > 0:
            grad_norm = strategy.clip_grad_norm(model, spec.gradient_clip)

        # A gradient that is not finite must not reach the weights. Reading the
        # norm costs one device synchronisation a step, which the previous
        # version avoided by reading it only on the steps that log -- and the
        # loss check below runs there too, AFTER optimizer.step(). So a single
        # non-finite micro-batch was applied to every parameter, and up to
        # log_every further steps then trained on the wreckage before anything
        # said so. Skipping the update is not a tolerance: an update of nan
        # destroys the model, and not taking it leaves the weights exactly
        # where the last good step put them.
        #
        # DDP has already all-reduced the gradients, so every rank reads the
        # same norm and decides alike; no extra collective is needed and none
        # is taken, because a rank that decided differently would desynchronise
        # the group rather than report anything.
        skipped = grad_norm is not None and not bool(torch.isfinite(grad_norm))
        if skipped:
            result.skipped += 1
            # Name the micro-batches, so this is a lead and not just a survival.
            # The sampler is a pure function of (seed, rank, start_batch), so a
            # step and a position inside it name the rows exactly;
            # Reading the flags synchronises, which is why it happens
            # here and not on every step.
            bad = [at for at, ok in enumerate(micro_finite) if not bool(ok)]
            # The update was not applied, so these weights are the ones that
            # produced the gradient. Written here and nowhere else: the nearest
            # periodic checkpoint is hundreds of steps behind, and a replay
            # from it cannot reproduce a failure that depends on the state.
            if spec.keep_skipped > 0:
                checkpoint.save(
                    skipped_root, result.step + 1, strategy, model, optimizer,
                    scheduler, ema, spec.keep_skipped,
                )
            logger.warning(
                "step %d: the gradient is %s, so this update is skipped "
                "(%d skipped so far); the weights are unchanged and written "
                "to %s. "
                "rank %d micro-batches %s carried a loss that was not finite; "
                "draw identity: step %d, rank %d, micro-batches %s",
                result.step + 1, float(grad_norm), result.skipped,
                skipped_root, distributed.rank(), bad or "none",
                result.step + 1, distributed.rank(),
                ",".join(str(at) for at in bad) or "none",
            )
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=set_to_none)
        micro_finite = []
        result.step += 1
        if ema is not None and not skipped:
            ema.update()

        if progress.due(result.step):
            # Every rank, every logging step, before any rank-local branch.
            packed = {} if chat_pack is None else pack_reading(
                window_pack, chat_pack.tokens, device)
            window_pack = None
            value = float(window_loss / window_count)
            extra: dict[str, float | None] = {
                name: float(held / window_extra_count[name])
                for name, held in window_extra.items()
            }
            for name in GROUP_DIAGNOSTICS:
                if name in window_groups:
                    total, count = (float(value) for value in window_groups[name])
                    extra[name] = total / count if count > 0 else None
            extra.update(packed)
            window_loss, window_count = None, 0
            window_extra, window_extra_count, window_groups = {}, {}, {}
            if math.isfinite(value):
                result.losses.append(value)
                progress.update(
                    result.step, value,
                    None if grad_norm is None else float(grad_norm),
                    extra or None,
                )
            else:
                # The update behind it was skipped, so the weights are exactly
                # where the last good step left them and the run goes on. What
                # this reports is that the window held a micro-batch whose loss
                # was not finite -- not that the run is broken. It must not
                # return early: the checkpoint below is what a later resume
                # needs, and skipping a save because one micro-batch in the
                # window was not finite would lose the very state that came
                # through it intact.
                logger.warning(
                    "step %d: the mean loss over the window is %s; %d update(s) "
                    "skipped so far in this run",
                    result.step, value, result.skipped,
                )

        if spec.save_every > 0 and result.step % spec.save_every == 0:
            checkpoint.save(
                checkpoint_root, result.step, strategy, model, optimizer,
                scheduler, ema, spec.keep_checkpoints,
            )
            saved_step = result.step

    if micro % spec.accumulation:
        logger.warning(
            "the data ran out %d micro-batches into a step; those gradients were dropped",
            micro % spec.accumulation,
        )
    if result.step != saved_step:
        checkpoint.save(
            checkpoint_root, result.step, strategy, model, optimizer,
            scheduler, ema, spec.keep_checkpoints,
        )
    distributed.barrier()
    return result
