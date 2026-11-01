"""Build the data, build the model, call the loop.

This is the composition root: the only place that reads the whole config,
opens the process group, resolves the three batch numbers, and decides where
checkpoints go. The batch numbers have one authority here. global_batch,
per_device_batch and gradient_accumulation_steps must agree with the world
size; leave one of them null and it is derived, state all three and a
disagreement stops the run before it trains. The old launcher derived the
per-device batch without knowing about accumulation, so every run with
accumulation above one trained at that multiple of its intended batch.
"""
from __future__ import annotations

import functools
import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .. import constants
from . import checkpoint, distributed
from .loop import Ema, LoopSpec, train
from .registry import MODELS
from .sampler import ResumableSampler, StepItemSampler
from .strategy import build_strategy

logger = logging.getLogger(__name__)

WORKERS_VAR = "NEO_VLA_DATALOADER_WORKERS"
PREFETCH_VAR = "NEO_VLA_DATALOADER_PREFETCH"
# torch's own prefetch_factor, the read-ahead per worker when PREFETCH_VAR is
# unset, in micro-batches as the variable counts.
DEFAULT_PREFETCH = 2

# The windows a key's fit is measured on at start-up (R14), drawn from the run's
# own manifest under the frame policies it trains.
FIT_WINDOWS = 4000

# What OmegaConf.select returns for a key the config does not state, told apart
# from a key stated as null.
NOT_STATED = object()

# The history context every stage trains with: dropout
# p = 0.1 is required, and the training-only state noise is 0.02 in model-input
# units, 0.02 rad, on 80% of samples. The data groups state the same numbers;
# these are what a run's resolved group is held to, so an override such as
# data.train.history_dropout=0 cannot train a whole stage without a word.
HISTORY_DROPOUT = 0.1
STATE_NOISE = {"sigma": 0.02, "rot_sigma_rad": 0.02, "prob": 0.8}


@dataclass(frozen=True)
class BatchPlan:
    global_batch: int
    per_device_batch: int
    accumulation: int
    world_size: int


def resolve_batch_plan(
    global_batch: int | None,
    per_device_batch: int | None,
    accumulation: int | None,
    world_size: int,
) -> BatchPlan:
    """Derive the missing number, or check that all three agree."""
    given = {
        "global_batch": global_batch,
        "per_device_batch": per_device_batch,
        "gradient_accumulation_steps": accumulation,
    }
    missing = sorted(name for name, value in given.items() if value is None)
    if len(missing) > 1:
        raise ValueError(
            f"the schedule leaves {missing} unset with world_size {world_size}.\n"
            "Expected at most one of global_batch, per_device_batch and "
            "gradient_accumulation_steps to be null.\n"
            "Give two of the three in the schedule config."
        )
    if global_batch is None:
        global_batch = per_device_batch * accumulation * world_size
    elif per_device_batch is None:
        per_device_batch = global_batch // (accumulation * world_size)
    elif accumulation is None:
        accumulation = global_batch // (per_device_batch * world_size)
    if min(global_batch, per_device_batch, accumulation) < 1:
        raise ValueError(
            f"global_batch {global_batch}, per_device_batch {per_device_batch}, "
            f"gradient_accumulation_steps {accumulation}, world_size {world_size}.\n"
            "Expected every one of them to be at least 1.\n"
            "Raise global_batch, or run on fewer processes."
        )
    product = per_device_batch * accumulation * world_size
    if product != global_batch:
        raise ValueError(
            f"global_batch {global_batch}, but per_device_batch {per_device_batch} x "
            f"gradient_accumulation_steps {accumulation} x world_size {world_size} = {product}.\n"
            "Expected global_batch to equal that product.\n"
            "Fix one of the three in the schedule config, or set it to null and it is derived."
        )
    return BatchPlan(global_batch, per_device_batch, accumulation, world_size)


def check_stream_span(sampler: ResumableSampler, total_batches: int) -> None:
    """Refuse a schedule whose draw numbers pass the data's stream span.

    Pass p of the sampler yields p * dataset_size + position, and RobotDataset
    keys every per-draw choice by that number through torch's 32-bit seed
    (neo_vla.data.mixture.STREAM_SPAN), refusing a number past it at that draw.
    total_batches is the whole schedule's micro-batches per rank, so a run that
    would reach past the span stops here, before its first step, rather than
    hours in; a resume is held to the same whole schedule.
    """
    from ..data.mixture import STREAM_SPAN

    if total_batches < 1:
        return
    last_pass = (int(total_batches) - 1) // sampler.batches_per_pass
    reach = last_pass * sampler.dataset_size + sampler.usable
    if reach > STREAM_SPAN:
        raise ValueError(
            f"the schedule's {int(total_batches)} micro-batches per rank reach draw number "
            f"{reach - 1} (pass {last_pass} over {sampler.dataset_size} draws).\n"
            f"Expected every draw number below {STREAM_SPAN}: the dataset seeds each "
            "draw's choices from it with torch's 32-bit seed.\n"
            "Lower schedule.max_steps, or data.train.epoch_samples."
        )


def build_loader(cfg, plan: BatchPlan, start_step: int, device: torch.device) -> DataLoader:
    """The dataset, sharded by rank and cut to the batches the run still owes.

    A run whose chat corpus packs reads a whole
    optimizer step per loader item: the same sampler's indices, grouped by
    StepItemSampler, read by the dataset as a StepItem and split by its
    collate_step into the step's micro-batches, each beside its chat pack.
    Without a pack every setting below is the one it always was.
    """
    dataset = instantiate(cfg.data.train)
    pack = dataset.chat_pack if hasattr(dataset, "chat_pack") else None
    remaining = max(int(cfg.schedule.max_steps) - start_step, 0)
    sampler = ResumableSampler(
        dataset_size=len(dataset),
        batch_size=plan.per_device_batch,
        num_batches=remaining * plan.accumulation,
        world_size=plan.world_size,
        rank=distributed.rank(),
        seed=int(cfg.seed),
        start_batch=start_step * plan.accumulation,
    )
    check_stream_span(sampler, int(cfg.schedule.max_steps) * plan.accumulation)
    # Loader workers are a property of the node, not of the run: the sampler
    # fixes the order, so the count changes throughput and nothing else.
    workers = int(os.environ.get(WORKERS_VAR, "8"))
    settings = {
        "batch_size": plan.per_device_batch,
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if workers > 0:
        settings["persistent_workers"] = True
    # The loader's own generator: every iterator draws its workers' base seed
    # from it rather than from the global CPU generator, so a resumed run's CPU
    # generator is where an uninterrupted run's is (without it the resumed
    # iterator's draw, taken after restore_rng, left the generator one draw
    # ahead). Keyed on (seed, rank); the
    # sampler fixes the order and every draw is pure in (index, pass, seed).
    generator = torch.Generator()
    generator.manual_seed(int(cfg.seed) * 65_537 + distributed.rank())
    settings["generator"] = generator
    # A dataset that assembles its own batches says so; the engine knows no
    # field of a batch. Without one, torch's default collate, as before.
    if hasattr(dataset, "collate"):
        settings["collate_fn"] = dataset.collate
    # Batches reach the rank in sampler order, so one worker held up by a slow
    # read stops its rank once the batches read ahead of it are used up, and
    # every other rank then waits for it at the next gradient sync. The
    # read-ahead is workers x prefetch batches: torch's default prefetch of 2
    # gives 24 batches at 12 workers, a step and a half at 16 accumulations.
    # Like the worker count, it changes throughput and nothing else.
    prefetch = os.environ.get(PREFETCH_VAR)
    if prefetch:
        if workers == 0 or int(prefetch) < 1:
            raise ValueError(
                f"{PREFETCH_VAR}={prefetch} with {WORKERS_VAR}={workers}.\n"
                "Expected a prefetch of at least 1 and at least one loader worker: "
                "without workers nothing is read ahead.\n"
                f"Unset {PREFETCH_VAR}, or give the loader workers and a positive prefetch."
            )
        settings["prefetch_factor"] = int(prefetch)
    if pack is not None:
        from ..data.chat_pack import pack_log_line, step_item_prefetch

        # The chat micro-batches against this run's accumulation, before a
        # worker starts: a list without the step's last micro-batch (the one
        # DDP synchronises) is refused here on every rank (shape B).
        pack.chat_micro_batches(plan.accumulation)

        settings.update(
            batch_size=None,
            sampler=StepItemSampler(sampler, plan.accumulation),
            # rank and world key a chain that several steps share (shape B,
            # chat_pack.group_key); a step's own chain is keyed by its draws.
            collate_fn=functools.partial(dataset.collate_step,
                                         per_device=plan.per_device_batch,
                                         accumulation=plan.accumulation,
                                         rank=distributed.rank(), world=plan.world_size),
        )
        # The read-ahead keeps counting micro-batches:
        # an item is a whole step, so 16 micro-batches at accumulation 8 is 2
        # items a worker, today's depth and not 8 times it.
        if workers > 0:
            settings["prefetch_factor"] = step_item_prefetch(
                int(prefetch) if prefetch else DEFAULT_PREFETCH, plan.accumulation)
        logger.info(pack_log_line(pack.tokens, pack.micro_batches, plan.accumulation,
                                  pack.every))
        if pack.force:
            logger.info("chat pack: steps 0..%d carry the forced fixtures %s",
                        len(pack.force) - 1, list(pack.force))
    return DataLoader(dataset, **settings)


def samples_per_item(loader: DataLoader, plan: BatchPlan) -> int:
    """The robot samples one item of a build_loader loader holds.

    A micro-batch, per_device_batch; a whole step, per_device_batch x
    accumulation, when the run packs chat and the loader reads step items.
    What a tool that counts samples by items (bench_loader) multiplies by.
    """
    if isinstance(loader.sampler, StepItemSampler):
        return plan.per_device_batch * plan.accumulation
    return plan.per_device_batch


def build_ema(cfg, module, strategy_name: str) -> Ema | None:
    """The moving average, on rank zero, or None when the run wants none."""
    decay = cfg.trainer.ema_decay
    if decay is None:
        return None
    if strategy_name != "ddp":
        raise ValueError(
            f"trainer.ema_decay is {decay} with trainer.strategy {strategy_name!r}.\n"
            "Expected ddp: rank zero keeps the only average, and under fsdp it "
            "would hold one shard of the weights.\n"
            "Set trainer.ema_decay null, or train this arm with ddp."
        )
    if not distributed.is_rank_zero():
        return None
    # None => average every trainable parameter, exactly as before this key existed.
    exclude = getattr(cfg.trainer, "ema_exclude_prefix", None)
    where = getattr(cfg.trainer, "ema_device", None)
    ema = Ema(module, float(decay), exclude_prefix=exclude, device=where)
    logger.info(
        "ema: averaging %d of %d trainable tensors%s",
        len(ema.tracked),
        sum(1 for _, p in module.named_parameters() if p.requires_grad),
        f", excluding {exclude!r}" if exclude else "",
    )
    if where:
        logger.info("ema: shadow held on %s", where)
    return ema


def initialise_from(
    path: Path, strategy, model, use_ema: bool = False, skip: Sequence[str] = (), *,
    new_keys: bool, drop_chat_head: bool = False,
) -> str | None:
    """Start this run from another run's weights, and from nothing else of it.

    Returns the digest of the init step's embodiments.json, the parent the
    run's own file records, or None for a model that carries no such file.
    new_keys is the data group's data.new_keys: whether the run's embodiments
    file may hold keys the init step's does not (check_inherited).

    A resume continues one run: it restores the optimiser, the schedule, the
    step and the random state, because leaving any of them behind would change
    the run it is continuing. A second stage is the opposite -- it wants the
    weights and a clean everything else, and taking the first stage's Adam
    moments and a schedule written for a different number of steps would be a
    silent way to train something other than what was asked for.

    Only rank zero reads the file; every rank then loads the same object, the
    way a resume does. The step stays zero, which is what the caller relies on.

    A second stage may be a LARGER model than the stage it starts from, and that
    is not an error. A co-train stage builds a co-train head that the stage it
    starts from does not have, and that head is meant to start fresh, so
    demanding an exact key set would refuse the path from stage 1 to stage 2.

    What must never happen is the other direction. A tensor the checkpoint holds
    and the model has no slot for means the two are not the same model, and
    loading the rest of it would train something other than what was asked for
    under a name that says otherwise -- so an unexpected key still refuses. The
    count of tensors actually taken is logged and asserted against the
    checkpoint, because "loaded from X" in a log is worth nothing if it does not
    say how many of X's tensors landed.

    `skip` names key PREFIXES the checkpoint is not allowed to supply, so those
    modules keep their build init while everything else comes from the pretrain.
    It exists to ask which PART of a pretrain carries an effect: ten gd5-initialised
    LIBERO arms sit at 78-83 on the two-object libero_10 tasks against 90-92 for
    no-pretrain arms, and the deficit survives every finetune knob AND survives
    scrambling the expert's entire layer wiring -- which points at the few small
    tensors that load identically regardless of arch (the action head and the
    tokenizers) rather than at the expert. Dropping a prefix is only meaningful
    for a module the mode actually trains; a frozen one would sit at random init
    for the whole run instead of being relearned.

    A prefix that matches nothing REFUSES rather than being quietly inert, because
    an ablation that silently did not happen looks exactly like one that did and
    had no effect -- the most expensive way to be wrong. Default empty, so every
    run that does not set it loads byte-identically to before.
    """
    state = path / checkpoint.STATE_FILE
    if not state.is_file():
        raise FileNotFoundError(
            f"trainer.init_from names {path}, which holds no state.pt.\n"
            "Expected a checkpoint directory written by this trainer.\n"
            "Name a step directory under some run's checkpoints/."
        )
    # Before the state is read: the format (R11), the embodiments file (R1,
    # R2), the stamps (pixel convention, prefix layout, state and history)
    # and every inherited entry (R8). A second stage under another convention
    # or another normaliser would start from a model whose every input moved.
    where = f"trainer.init_from {path}"
    checkpoint.check_step(path, strategy.unwrap(model), where)
    parent = check_inherited(strategy.unwrap(model), path, where, new_keys=new_keys)
    payload = torch.load(state, map_location="cpu", weights_only=False)
    if "model" not in payload:
        raise ValueError(
            f"{state} holds no model entry.\n"
            "Expected a checkpoint this trainer wrote.\n"
            "Name a step directory under some run's checkpoints/."
        )
    weights = init_weights(payload, strategy.unwrap(model), state, drop_chat_head, use_ema=use_ema)
    if use_ema:
        # geoA's finetune sets init_use_ema: true, and its record was scored from an EMA.
        # Our pretrains keep one (trainer.ema_decay) but initialise_from read payload["model"]
        # unconditionally, so the average sat in state.pt unused. The EMA covers only the
        # tensors that took gradient -- 446 of 972 for a frozen-backbone pretrain -- so it is
        # OVERLAID on the raw weights rather than replacing them: EMA where it exists, raw
        # elsewhere. A stray ema key would mean the two are not the same model, so it refuses.
        ema_w = payload.get("ema")
        if not ema_w:
            raise ValueError(
                f"trainer.init_use_ema is set but {state} holds no ema entry.\n"
                "Expected a checkpoint whose run kept an EMA (trainer.ema_decay).\n"
                "Name a checkpoint that has one, or leave init_use_ema false."
            )
        stray = sorted(set(ema_w) - set(weights))
        if stray:
            raise ValueError(
                f"{state}'s ema holds {len(stray)} tensors its model does not, "
                f"beginning {stray[:3]}.\n"
                "Expected the ema to shadow a subset of the model."
            )
        weights.update(ema_w)
        logger.info(
            "init_use_ema: %d of %d tensors taken from the checkpoint's EMA, the rest raw",
            len(ema_w), len(weights),
        )
    else:
        logger.info("init_weights: raw model, %d tensors; no EMA overlay", len(weights))
    present = set(strategy.unwrap(model).state_dict().keys())
    # Checked against the FULL checkpoint, before any skip is applied: a skip says
    # "do not take this tensor", never "this checkpoint may be a different model".
    unexpected = sorted(set(weights) - present)
    if unexpected:
        raise ValueError(
            f"{state} holds {len(unexpected)} tensors this model has no slot for, "
            f"beginning {unexpected[:3]}.\n"
            "Expected a checkpoint of the same model, or of a smaller stage of it.\n"
            "Check arch and mode match the run that wrote the checkpoint."
        )
    prefixes = [str(p) for p in (skip or ())]
    if prefixes:
        supplied_before = len(weights)
        inert = []
        for prefix in prefixes:
            dropped = sorted(k for k in weights if k.startswith(prefix))
            if not dropped:
                inert.append(prefix)
                continue
            for key in dropped:
                del weights[key]
            shown = dropped if len(dropped) <= 40 else dropped[:40]
            logger.info(
                "init_skip %r: dropped %d tensors from the checkpoint -- %s%s",
                prefix, len(dropped), shown,
                "" if len(dropped) <= 40 else f" (+{len(dropped) - 40} more)",
            )
        if inert:
            raise ValueError(
                f"trainer.init_skip lists {len(inert)} prefix(es) matching NO key in "
                f"{state}: {inert}.\n"
                "Expected every prefix to name tensors the checkpoint actually holds.\n"
                "A prefix that matches nothing would leave the run identical to one "
                "without init_skip, so the ablation would read as 'ran, no effect' when "
                "it never ran at all. Check the spelling against the checkpoint's keys."
            )
        logger.info(
            "init_skip: %d prefixes dropped %d of the checkpoint's %d tensors; "
            "%d will be supplied to this model's %d parameters",
            len(prefixes), supplied_before - len(weights), supplied_before,
            len(weights), len(present),
        )
    missing = sorted(present - set(weights))
    strategy.unwrap(model).load_state_dict(weights, strict=False)
    if missing:
        logger.info(
            "init_from supplies %d of this model's %d tensors; the other %d keep what the "
            "build gave them, beginning %s",
            len(weights), len(present), len(missing), missing[:3],
        )
    logger.info(
        "initialised %d tensors from %s; optimiser, schedule and step start fresh",
        len(weights), path,
    )
    return parent


def read_step_meta(path: Path) -> dict:
    """The meta.json of a step directory; R11 when there is none.

    A trainer step is the only thing init_from reads: the graft's graft.json and
    its truncated-backbone state are retired with the old lineage.
    """
    return checkpoint.read_meta(path)


def read_step_embodiments(path: Path) -> tuple[dict, str]:
    """A step's embodiments.json -> (the parsed document, its file digest).

    Read after checkpoint.check_embodiments_file has held the bytes to the
    digest meta.json names, so the stored entry digests are the ones the tree
    that wrote the step computed.
    """
    data = (Path(path) / constants.EMBODIMENTS_FILE).read_bytes()
    return json.loads(data.decode("utf-8")), checkpoint.file_digest(data)


def step_entry_digests(path: Path) -> tuple[dict[str, str], str | None, str]:
    """A step's embodiments.json -> (entry digest by key, its parent, its file digest)."""
    document, digest = read_step_embodiments(path)
    entries = {
        str(key): str(entry.get("digest")) for key, entry in document["embodiments"].items()
    }
    return entries, document.get("parent"), digest


def check_copied_action(entry, stored: dict, where: str) -> None:
    """R8 unless a key new to the lineage holds the action block it names as its source.

    tools/build_stats.py --inherit-action KEY=SRC copies SRC's rows of low,
    high, constant and absent for KEY's slots on the columns of the groups
    action.from_groups names (every column when it names none), gripper_valid
    with the ee group, and SRC's mask, and writes action.from = SRC (V6 4.7,
    LIBERO's libero_franka_sim from franka_panda: the executed ee block, "same
    numbers", its joints its own, U2 option (c)). The entry digest covers the
    numbers and not the source, so a copied column edited after the copy, or
    copied from another file, is caught here and nowhere else.
    """
    from ..data.embodiment import inherited_columns

    source = entry.action_from
    columns = list(inherited_columns(entry.action_from_groups))
    held = stored.get(source)
    if not isinstance(held, dict):
        raise ValueError(
            f"{constants.refusal_prefix(8)} {entry.key!r} copies its action block from "
            f"{source!r}, which {where} does not hold; it holds {sorted(stored)}.\n"
            "Expected action.from to name a key of the init step's embodiments file "
            "(V6 4.7b): the copied numbers are that entry's.\n"
            "Build the key with tools/build_stats.py --inherit-action against this step's file."
        )
    action = held.get("action") or {}
    slots = [int(slot) for slot in held.get("slots", [])]
    missing = [slot for slot in entry.slots if slot not in slots]
    if missing:
        raise ValueError(
            f"{constants.refusal_prefix(8)} {entry.key!r} copies {source!r}'s action block "
            f"and fills slots {missing} that {source!r} does not cover ({slots}).\n"
            "Expected every slot of the key to be a slot of its source.\n"
            "Inherit from a key that covers them, or measure the key."
        )
    rows = [slots.index(slot) for slot in entry.slots]
    own = {"low": entry.low, "high": entry.high, "constant": entry.constant,
           "absent": entry.absent}
    if constants.GRIPPER_COLUMN in columns:
        own["gripper_valid"] = entry.gripper_valid
    differ = []
    for name, values in own.items():
        stated = action.get(name)
        if not isinstance(stated, list) or any(row >= len(stated) for row in rows):
            differ.append(name)
            continue
        copied = numpy.asarray([stated[row] for row in rows], dtype=numpy.float64)
        mine = numpy.asarray(values, dtype=numpy.float64)
        if name != "gripper_valid":
            if copied.ndim != 2 or mine.ndim != 2 or copied.shape[1] != constants.ACTION_DIM:
                differ.append(name)
                continue
            copied, mine = copied[:, columns], mine[:, columns]
        if copied.shape != mine.shape or not numpy.array_equal(copied, mine, equal_nan=True):
            differ.append(name)
    if tuple(action.get("mask") or ()) != tuple(entry.mask):
        differ.append("mask")
    if differ:
        raise ValueError(
            f"{constants.refusal_prefix(8)} {entry.key!r} names {source!r} as the source of "
            f"its action block and differs from it in {differ} on slots {list(entry.slots)} "
            f"(copied columns {columns}).\n"
            "Expected the source's numbers on every copied column, row for row (V6 4.7: same "
            "numbers).\n"
            "Rebuild the key with tools/build_stats.py --inherit-action against this step's file."
        )


def check_inherited(module, path: Path, where: str, *, new_keys: bool) -> str | None:
    """V6 4.7 against the step a run starts or resumes from; returns the step file's digest.

    R8 unless the run's file keeps the lineage:
    - the normaliser header is the step's (transform, quantiles, out-of-band rule
      and band alpha, which sets the band, the state clip and the decode clamp of
      every key, a new one included; 4.7b copies it from the init file);
    - every key both files hold has the step's entry digest, numbers and clause
      alike (4.7a);
    - a key new to the lineage is allowed only under new_keys (the data group's
      data.new_keys at init_from; never at a resume): stage 2 adds no key to
      stage 1's (V6 4.10(3)), a post-train or debug group does;
    - a key new to the lineage whose action block names a source (action.from)
      holds that source's rows (check_copied_action).
    R9 when a key of the run has the clause of a different key of the step
    (4.7b: a clause distinct from every existing key's). A key the step has and
    the run lacks is allowed. The digest returned becomes the run's parent. A
    model without an embodiments file (the engine's test models) is skipped.
    """
    embodiments = vars(module).get("embodiments")
    if embodiments is None:
        return None
    document, digest = read_step_embodiments(path)
    stored = document["embodiments"]
    header = document.get("normaliser")
    if header != embodiments.header:
        raise ValueError(
            f"{constants.refusal_prefix(8)} {where} states normaliser {header!r}; this "
            f"run's file states {embodiments.header!r}.\n"
            "Expected the lineage's header (V6 4.7b): one transform, quantile pair, "
            "out-of-band rule and band alpha for every key the weights were trained on.\n"
            "Build the run's embodiments file with tools/build_stats.py --from this step's file."
        )
    compared = []
    for key, entry in sorted(embodiments.entries.items()):
        held = stored.get(key)
        if held is None:
            continue
        compared.append(key)
        if str(held.get("digest")) != entry.digest:
            raise ValueError(
                f"{constants.refusal_prefix(8)} {where} normalised {key!r} through entry "
                f"{held.get('digest')}; this run's file gives it {entry.digest}.\n"
                "Expected an inherited key to keep its entry, numbers and clause alike "
                "(V6 4.7a): the weights were trained on targets scaled by it.\n"
                "Copy the init step's entry for this key into the run's embodiments file."
            )
    new = [key for key in sorted(embodiments.entries) if key not in stored]
    if new and not new_keys:
        raise ValueError(
            f"{constants.refusal_prefix(8)} this run's embodiments file holds {len(new)} "
            f"keys {where} does not, first {new[:3]}.\n"
            "Expected no key new to the lineage: the data group states data.new_keys false "
            "(stage 2 adds no key to stage 1's, V6 4.10(3)), and a resume continues one run.\n"
            "Build the run's file from the init step's keys, or set data.new_keys true in a "
            "post-train or debug group."
        )
    for key in new:
        entry = embodiments.entries[key]
        if entry.action_from is not None:
            check_copied_action(entry, stored, where)
        for other, held in sorted(stored.items()):
            if held.get("clause") == entry.clause:
                raise ValueError(
                    f"{constants.refusal_prefix(9)} {key!r} has the clause of {other!r} in "
                    f"{where}: {entry.clause!r}.\n"
                    "Expected a key new to the lineage to have a clause distinct from every "
                    "existing key's (V6 4.7b): the clause names the normaliser.\n"
                    "Give the new key a clause of its own in config/embodiments.yaml."
                )
    logger.info(
        "%s: R8 compared %d of this run's %d keys with the step's %d (%s); %d new to the "
        "lineage (%s); parent %s",
        where, len(compared), len(embodiments.entries), len(stored), compared[:8],
        len(new), new[:8], digest,
    )
    return digest


def load_embodiments(cfg):
    """The run's embodiments file from data.train.normaliser; refused when unset."""
    path = OmegaConf.select(cfg, "data.train.normaliser", default=None)
    if path is None:
        raise ValueError(
            "data.train.normaliser is not set for a model that normalises through an "
            "embodiments file.\n"
            "Expected the corpus's prep_<corpus>/embodiments.json in every training data "
            "group.\nSet data.train.normaliser; there is no default normaliser."
        )
    from ..data.embodiment import Embodiments

    loaded = Embodiments.load(str(path))
    logger.info(
        "embodiments: %d keys from %s (%s)", len(loaded.entries), path, loaded.file_digest
    )
    return loaded


def lineage_heads(document: dict | None) -> dict[tuple[str, str], str]:
    """The prompt heads an init step's run trained, as (head, tail) -> key.

    Read from the trained section its runner wrote (trained.keys[K].prompt_heads);
    an init file without one (a corpus copy) contributes none.
    """
    trained = (document or {}).get("trained") or {}
    heads: dict[tuple[str, str], str] = {}
    for key, facts in sorted((trained.get("keys") or {}).items()):
        for head, tail in facts.get("prompt_heads", []):
            heads.setdefault((str(head), str(tail)), str(key))
    return heads


def check_shared_clauses(embodiments, summary: dict, lineage: dict | None = None) -> None:
    """R9 when two keys share a clause, or two keys share a prompt head.

    The clause and the head are how the prompt tells one normaliser from
    another; two keys that read the same would be two scales the model cannot
    tell apart. lineage is the init step's embodiments document, whose trained
    heads the run's heads are held against too (V6 4.7b): a key new to the
    lineage must not read like one the weights already know. The clauses of the
    two files are held against each other by check_inherited.
    """
    seen: dict[str, str] = {}
    for key, entry in sorted(embodiments.entries.items()):
        other = seen.setdefault(entry.clause, key)
        if other != key:
            raise ValueError(
                f"{constants.refusal_prefix(9)} keys {other!r} and {key!r} share the clause "
                f"{entry.clause!r}.\n"
                "Expected a distinct clause per key: it is what names the normaliser "
                "in the prompt.\nGive the new key a clause of its own."
            )
    inherited = lineage_heads(lineage)
    heads = dict(inherited)
    for key, facts in sorted(summary.get("keys", {}).items()):
        for head, tail in facts.get("prompt_heads", []):
            other = heads.setdefault((head, tail), key)
            if other != key:
                origin = " (trained by the init step)" if inherited.get((head, tail)) else ""
                raise ValueError(
                    f"{constants.refusal_prefix(9)} keys {other!r}{origin} and {key!r} share "
                    f"the prompt head {head!r}.\n"
                    "Expected every key's rendered head to be its own.\n"
                    "Give the keys distinct clauses and re-render the prompt cache."
                )


def check_served_directories(embodiments, dataset) -> None:
    """Refuse a manifest directory the embodiments file has no datasets record for.

    The server resolves a client's directory to its record digest (R4), so a
    directory without one could be trained on and never served. The
    directories are the dataset's manifest's (RobotDataset.manifest.entries);
    a dataset that states no manifest, or one with no directory, is refused
    rather than passed on nothing.
    """
    manifest = dataset.manifest if hasattr(dataset, "manifest") else None
    entries = manifest.entries if hasattr(manifest, "entries") else None
    if entries is None:
        raise ValueError(
            f"the training dataset ({type(dataset).__name__}) states no manifest.entries.\n"
            "Expected a RobotDataset, whose manifest lists every directory the run draws.\n"
            "Train through neo_vla.data.dataset.RobotDataset."
        )
    entries = tuple(entries)
    if not entries:
        raise ValueError(
            "the training dataset's manifest lists 0 directories.\n"
            "Expected at least one: a record check over none passes on no evidence.\n"
            "Point data.train.manifest at the corpus this run trains."
        )
    missing = sorted(entry.name for entry in entries if entry.name not in embodiments.datasets)
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(entries)} manifest directories have no datasets "
            f"record in the embodiments file, first {missing[:3]}.\n"
            "Expected a record per directory: it is what a client's handshake is held "
            "against.\nRebuild the embodiments file from this run's manifest."
        )
    logger.info("embodiments: %d of %d manifest directories have a datasets record",
                len(entries), len(entries))


def required_switch(cfg, key: str) -> bool:
    """A boolean the config must state: absent, null or not a bool refuses, naming it."""
    value = OmegaConf.select(cfg, key, default=NOT_STATED)
    if not isinstance(value, bool):
        stated = "absent" if value is NOT_STATED else repr(value)
        raise ValueError(
            f"{key} is {stated}.\n"
            "Expected true or false, stated by the config that composes this run: the "
            "switch has no default.\n"
            f"Set {key} in the group."
        )
    return value


def check_fit(cfg, embodiments, dataset, *, step: Path | None = None) -> None:
    """R14 at start-up, when the data group asks for it (data.fit_check, required).

    Rank zero measures (measure_fit) and every rank takes its verdict: the
    windows are drawn from the run's own manifest and seed, so every rank would
    draw the same ones, and reading them once is what keeps a debug run's
    start-up from reading every key's windows on all of its ranks. A refusal on
    rank zero is raised on every rank, so none waits on a collective the others
    left.
    """
    if not required_switch(cfg, "data.fit_check"):
        return
    failure, local = None, None
    if distributed.is_rank_zero():
        try:
            measure_fit(cfg, embodiments, dataset, step=step)
        except Exception as error:  # every rank must hear it, whatever it is
            failure, local = str(error), error
    failure = distributed.gather_objects(failure)[0]
    if local is not None:
        raise local
    if failure is not None:
        raise ValueError(failure)


def measure_fit(cfg, embodiments, dataset, *, step: Path | None = None) -> None:
    """R14 on one process: every key the run trains, every slot of its entry, every field.

    The targets of FIT_WINDOWS windows drawn from this run's own manifest (so
    under the frame policies this run draws) go through embodiment.fit_check
    one field at a time (joints, the EE range, hand: FIT_FIELDS; the gripper
    is exempt) and, for a key with a body, through fit_check_body one group at
    a time (base, posture), and require_fit refuses a failing report. A row
    counts for a field where its step exists and every column of the field it
    checks was read there (Targets.read, body_read): an unread joint is a 0 in
    the chunk, never a target. The columns checked are the ones both the
    window and the entry decode; windows whose directories decode different
    columns (a masked drot, a dead joint) are checked as separate groups, so a
    column is gated only where it trains. The base is judged on its moving
    rows only (base_moving), as its scale was fitted. A slot the run draws no
    window at is not checked; a slot it draws windows at and measures no range
    column of (a gripper-only window, T7) is refused. The per-planned-policy
    check is build_stats --check-fit.
    """
    from ..data.embodiment import (
        BODY_FIT_FIELDS,
        FIT_FIELDS,
        base_moving,
        fit_check,
        fit_check_body,
        require_fit,
    )

    alpha = float(embodiments.band_alpha)
    # A resume supplies its step explicitly; otherwise use the configured init.
    if step is None:
        step = OmegaConf.select(cfg, "trainer.init_from", default=None)
    lineage = read_step_embodiments(Path(step))[0] if step is not None else {}
    summary = dataset.trained_summary()["keys"]
    for key in sorted(summary):
        entry = embodiments.entry(key)
        stored = lineage.get("embodiments", {}).get(key)
        inherited = stored is not None and stored.get("digest") == entry.digest
        frames = ((lineage.get("trained") or {}).get("keys", {}).get(key, {}).get("frames")
                  if inherited else None)
        run_frames = {frame for frame, share in summary[key].get("frames", {}).items()
                      if share > 0}
        lineage_frames = ({frame for frame, share in frames.items() if share > 0}
                          if frames is not None else None)

        def require(report, *, where):
            if inherited:
                require_fit(report, where=where, run_frames=run_frames,
                            lineage_frames=lineage_frames)
            else:
                require_fit(report, where=where)
        targets = [dataset.targets(draw) for draw in
                   dataset.key_windows(key, FIT_WINDOWS, seed=int(cfg.seed))]
        for slot in entry.slots:
            decoded = torch.as_tensor(entry.decodable(slot), dtype=torch.bool)
            drawn, groups = 0, {}
            for target in targets:
                for arm, held in enumerate(target.slots):
                    if int(held) != int(slot):
                        continue
                    drawn += 1
                    dims = torch.as_tensor(target.dims_valid[arm], dtype=torch.bool) & decoded
                    read = torch.as_tensor(target.read[arm], dtype=torch.bool)
                    steps = torch.as_tensor(target.steps_valid[arm], dtype=torch.bool)
                    for field, columns in FIT_FIELDS.items():
                        checked = torch.zeros_like(dims)
                        checked[list(columns)] = dims[list(columns)]
                        if not bool(checked.any()):
                            continue
                        rows = steps & read[:, checked].all(dim=-1)
                        groups.setdefault((field, tuple(checked.tolist())), []).append(
                            (target.chunk[arm], rows))
            measured = _judge(groups, lambda chunks, dims, steps: fit_check(
                chunks, entry, int(slot), alpha=alpha, dims_valid=dims, steps_valid=steps),
                require, where=f"{key} slot {slot}")
            if drawn and not measured:
                raise ValueError(
                    f"{constants.refusal_prefix(14)} {key} slot {slot} has no window with a "
                    f"read, decoded range column among {drawn} arm-windows drawn.\n"
                    "Expected the fit gate to measure something; a gate over nothing "
                    "passes on no evidence (the gripper alone is exempt).\n"
                    "Check the key's directories and frame policies."
                )
        body = getattr(entry, "body", None)
        if body is None or not body.present:
            continue
        decoded = torch.as_tensor(body.decodable(), dtype=torch.bool)
        command = body.base == "command"
        groups = {}
        for target in targets:
            dims = torch.as_tensor(target.body_dims, dtype=torch.bool) & decoded
            read = torch.as_tensor(target.body_read, dtype=torch.bool)
            values = torch.as_tensor(target.body, dtype=torch.float64)
            base = values[:, constants.BODY_BASE].clone()
            base[~read[:, constants.BODY_BASE]] = float("nan")
            moving = torch.as_tensor(base_moving(base.numpy(), command=command))
            for field, columns in BODY_FIT_FIELDS.items():
                checked = torch.zeros_like(dims)
                checked[list(columns)] = dims[list(columns)]
                if not bool(checked.any()):
                    continue
                rows = read[:, checked].all(dim=-1)
                if field == "base":
                    rows = rows & moving
                groups.setdefault((field, tuple(checked.tolist())), []).append((values, rows))
        _judge(groups, lambda chunks, dims, steps: fit_check_body(
            chunks, entry, alpha=alpha, dims_valid=dims, steps_valid=steps,
            moving=torch.ones_like(steps)), require, where=f"{key} body")


def _judge(groups: dict, check, require, *, where: str) -> int:
    """Each (field, columns) group with a counted row through check and require; the count."""
    measured = 0
    for (field, dims), rows in sorted(groups.items()):
        steps = torch.stack([torch.as_tensor(valid, dtype=torch.bool) for _, valid in rows])
        if not bool(steps.any()):
            continue
        chunks = torch.stack([torch.as_tensor(chunk, dtype=torch.float64) for chunk, _ in rows])
        report = check(chunks, torch.tensor(dims), steps)
        require(report, where=f"{where} {field}, {len(rows)} windows")
        measured += 1
    return measured


def check_history_knobs(cfg) -> None:
    """Refuse a run whose data group trains another history dropout or state noise.

    The history dropout (HISTORY_DROPOUT) and the state noise (STATE_NOISE)
    are fixed for every stage, so the context is trained alike in stage 1,
    stage 2 and post-training. There is no ablation switch.
    """
    dropout = OmegaConf.select(cfg, "data.train.history_dropout", default=NOT_STATED)
    noise = OmegaConf.select(cfg, "data.train.state_noise", default=NOT_STATED)
    if OmegaConf.is_config(noise):
        noise = OmegaConf.to_container(noise, resolve=True)
    number = not isinstance(dropout, bool) and isinstance(dropout, (int, float))
    if not number or float(dropout) != HISTORY_DROPOUT or noise != STATE_NOISE:
        stated = lambda value: "absent" if value is NOT_STATED else repr(value)  # noqa: E731
        raise ValueError(
            f"data.train.history_dropout is {stated(dropout)} and data.train.state_noise "
            f"is {stated(noise)}.\n"
            f"Expected {HISTORY_DROPOUT} and {STATE_NOISE} in every stage: the "
            "history context and the state noise are trained alike from stage 1 to "
            "post-training.\n"
            "Drop the override, or restore the data group's values."
        )


def trained_stamp(cfg, summary: dict) -> dict:
    """The trained section of the run's embodiments file, from the resolved config.

    The dataset's summary (frames, prompt heads and rates per key, blocked
    episodes) plus the history and the state noise this run's data group
    states, held to the fixed values first. Taken from the config as
    resolved, never from a default, so the stamp and the config cannot differ.
    """
    check_history_knobs(cfg)
    train_group = cfg.data.train
    return {
        "run": OmegaConf.select(cfg, "run_name", default=None),
        "manifest_sha256": summary["manifest_sha256"],
        "keys": summary["keys"],
        "history": {
            "H": constants.HISTORY_SIZE,
            "min_gap": constants.MIN_GAP,
            "sampling": constants.SAMPLING,
            "no_history": constants.NO_HISTORY_FORM,
            "dropout": float(train_group.history_dropout),
            "blocked_episodes": int(summary["blocked_episodes"]),
        },
        "state_noise": OmegaConf.to_container(train_group.state_noise, resolve=True),
    }


def shared_summary(dataset) -> dict:
    """The dataset's trained summary, drawn on rank zero and handed to every rank.

    trained_summary draws FRAME_DRAWS windows per key and reads each camera-frame
    directory's presence and state rows; every rank would draw the same ones
    (the run's manifest and seed), so rank zero draws them once and every rank
    stamps the same file from its summary. A failure on rank zero is raised on
    every rank, so none waits on a collective the others left.
    """
    failure, local, summary = None, None, None
    if distributed.is_rank_zero():
        try:
            summary = dataset.trained_summary()
        except Exception as error:  # every rank must hear it, whatever it is
            failure, local = str(error), error
    failure, summary = distributed.gather_objects((failure, summary))[0]
    if local is not None:
        raise local
    if failure is not None:
        raise ValueError(failure)
    return summary


def stamp_embodiments(cfg, module, dataset, run_dir: Path, parent: str | None,
                      lineage: dict | None = None, *, fit_step: Path | None = None) -> None:
    """The run's embodiments file: checked, stamped with what it trains, written.

    R9 on shared clauses and heads (the heads also against lineage, the init
    step's embodiments document, when the run starts from one), every directory
    served, R14 when the group asks, then policy.embodiments becomes the file
    with its trained section and parent, which every checkpoint.save writes
    beside the step, and rank zero writes <run>/embodiments.json.
    """
    embodiments = module.embodiments
    summary = shared_summary(dataset)
    check_shared_clauses(embodiments, summary, lineage)
    check_served_directories(embodiments, dataset)
    check_fit(cfg, embodiments, dataset, step=fit_step)
    module.embodiments = embodiments.with_trained(trained_stamp(cfg, summary), parent=parent)
    if distributed.is_rank_zero():
        target = Path(run_dir) / constants.EMBODIMENTS_FILE
        target.write_bytes(module.embodiments.dumps())
        logger.info(
            "embodiments: wrote %s (%s), parent %s", target,
            checkpoint.file_digest(module.embodiments.dumps()), parent,
        )


def check_resumed_rates(configured: list[float], scheduler, where: Path) -> None:
    """Raise unless the resumed schedule runs at the rates this config states.

    A resume restores the optimizer's groups and the scheduler's base rates from
    the checkpoint, so a rate changed in the config since would be dropped
    without a word. A changed group count already raises inside torch.
    """
    saved = [float(rate) for rate in scheduler.base_lrs]
    if saved != configured:
        raise ValueError(
            f"{where.name} resumes base rates {saved}; this config states {configured}.\n"
            "Expected the rates the run was started with: a resume keeps the saved ones.\n"
            "Restore the schedule's rates, or start a new run from it with trainer.init_from."
        )


def compile_robot_path(cfg, policy, strategy_name: str) -> None:
    """trainer.compile_robot: the robot path's layers through torch.compile, before the wrap.

    Stage 2's lever: the backbone's layers as the robot
    prefix calls them and the expert's layers run compiled, each backbone layer
    holding its recomputation inside the compiled graph; the chat pack keeps
    its eager layers and its own compiled flex kernel (compiled_layers.
    compile_robot). Accepted with ddp only, with or without recomputation and
    co-training; refused beside trainer.compile_layers, which compiles the
    same layers another way.
    """
    if not bool(OmegaConf.select(cfg, "trainer.compile_robot", default=False)):
        return
    if strategy_name != "ddp":
        raise ValueError(
            f"trainer.compile_robot is true with trainer.strategy {strategy_name!r}.\n"
            "Expected ddp, the setting the compiled robot path was measured in.\n"
            "Set trainer.compile_robot false, or train with ddp."
        )
    if bool(OmegaConf.select(cfg, "trainer.compile_layers", default=False)):
        raise ValueError(
            "trainer.compile_robot and trainer.compile_layers are both true.\n"
            "Expected one of them: both compile the backbone's and the expert's layers.\n"
            "Set trainer.compile_layers false for a co-train run."
        )
    if not hasattr(policy, "compile_robot"):
        raise ValueError(
            f"trainer.compile_robot is true and model {cfg.model.name} has no compile_robot.\n"
            "Expected a model that names its robot path.\n"
            "Set trainer.compile_robot false for this model."
        )
    eager_numerics = OmegaConf.select(cfg, "trainer.robot_eager_numerics", default=False)
    if not isinstance(eager_numerics, bool):
        raise ValueError(
            f"trainer.robot_eager_numerics is {eager_numerics!r}.\n"
            "Expected false (the default: inductor's defaults) or true (eager's bf16 "
            "rounding of pointwise ops, the attention kept for the backward; +1.07 GiB "
            "reserved a rank).\n"
            "Fix trainer.robot_eager_numerics."
        )
    if eager_numerics:
        policy.compile_robot(eager_numerics=True)
    else:
        policy.compile_robot()
    logger.info(
        "compile_robot: the robot prefix's backbone layers (recomputation inside the graph) and "
        "the expert's layers run through torch.compile (inductor, static shapes, optimize_ddp "
        "off)%s; the chat pack's layers stay eager; the first step compiles them",
        ", eager numerics (emulate_precision_casts, attention saved in the recomputed "
        "layers)" if eager_numerics else ", inductor's default numerics",
    )


def robot_recompute(cfg, policy) -> None:
    """trainer.robot_recompute_layers: how many backbone layers a robot-only micro-batch recomputes.

    null (the default) recomputes every layer under gradient checkpointing,
    as before; k recomputes the first k of them and keeps the others'
    activations, on a micro-batch without a chat half only (a chat micro-batch
    always recomputes every layer). It moves memory, never a number.
    """
    value = OmegaConf.select(cfg, "trainer.robot_recompute_layers", default=None)
    if value is None:
        return
    depth = len(policy.encoder.backbone.layers) if hasattr(policy, "encoder") else None
    if isinstance(value, bool) or not isinstance(value, int) or depth is None \
            or not 0 <= value <= depth:
        raise ValueError(
            f"trainer.robot_recompute_layers is {value!r}.\n"
            f"Expected null (every layer) or a whole number from 0 to the backbone's "
            f"{depth} layers.\n"
            "Fix trainer.robot_recompute_layers."
        )
    if not bool(cfg.trainer.gradient_checkpointing):
        raise ValueError(
            f"trainer.robot_recompute_layers is {value} and trainer.gradient_checkpointing "
            "is false.\n"
            "Expected recomputation on: the key says how many layers a robot-only micro-batch "
            "recomputes, and with it off none is.\n"
            "Turn trainer.gradient_checkpointing on, or set trainer.robot_recompute_layers null."
        )
    policy.robot_recompute_layers = int(value)
    logger.info(
        "robot_recompute_layers: a robot-only micro-batch recomputes %d of the backbone's %d "
        "layers; a micro-batch with a chat half recomputes all of them", value, depth,
    )


def compile_stacks(cfg, policy, strategy_name: str) -> None:
    """trainer.compile_layers: the policy's layer stacks through torch.compile, before the wrap.

    Only the setting the speed runs measured is accepted, ddp without
    recomputation on a robot-only run: a recompute or a sharded unit around a
    compiled layer, or a co-train run's variable-length chat rows through it,
    was never run, so it is refused here rather than tried on a 64-GPU launch.
    """
    if not bool(OmegaConf.select(cfg, "trainer.compile_layers", default=False)):
        return
    recompute = bool(cfg.trainer.gradient_checkpointing)
    if strategy_name != "ddp" or recompute:
        raise ValueError(
            f"trainer.compile_layers is true with trainer.strategy {strategy_name!r} and "
            f"trainer.gradient_checkpointing {recompute}.\n"
            "Expected ddp without recomputation, the setting the compiled layers were "
            "measured in.\n"
            "Set trainer.compile_layers false, or train with ddp and no recomputation."
        )
    if OmegaConf.select(cfg, "data.train.understanding", default=None) is not None:
        raise ValueError(
            "trainer.compile_layers is true and data.train.understanding is set.\n"
            "Expected a robot-only run, the setting the compiled layers were measured in: "
            "chat rows are padded to their own lengths, and a static-shape layer compiles "
            "again for every new one.\n"
            "Set trainer.compile_layers false for a co-train run."
        )
    if not hasattr(policy, "compile_layers"):
        raise ValueError(
            f"trainer.compile_layers is true and model {cfg.model.name} has no compile_layers.\n"
            "Expected a model that names the layers it compiles.\n"
            "Set trainer.compile_layers false for this model."
        )
    counts = policy.compile_layers()
    logger.info(
        "compile_layers: %s run through torch.compile (inductor, static shapes); "
        "the first step compiles them",
        ", ".join(f"{count} {kind}" for kind, count in counts.items()),
    )


def trim_prefix(cfg, policy) -> None:
    """trainer.trim_prefix: the policy runs each micro-batch's prefix only as wide as it needs."""
    bucket = int(OmegaConf.select(cfg, "trainer.trim_prefix", default=0) or 0)
    if bucket == 0:
        return
    if bucket < 0:
        raise ValueError(
            f"trainer.trim_prefix is {bucket}.\n"
            "Expected 0 (every Lp column) or a positive bucket of columns.\n"
            "Set trainer.trim_prefix to 0 or to a bucket such as 64."
        )
    if not hasattr(policy, "trim_prefix"):
        raise ValueError(
            f"trainer.trim_prefix is {bucket} and model {cfg.model.name} has no trim_prefix.\n"
            "Expected a model whose prefix carries trailing pad it can cut.\n"
            "Set trainer.trim_prefix 0 for this model."
        )
    policy.trim_prefix(bucket)
    logger.info(
        "trim_prefix: the prefix runs at its longest row rounded up to %d columns, at most Lp",
        bucket,
    )


def choose_resume(root: Path) -> Path | None:
    """Rank zero picks the step directory; every rank reads that one, or none.

    start_step decides how many batches a rank asks its sampler for, so two
    ranks that disagree run different numbers of collectives and the job hangs
    rather than stopping.
    """
    chosen = checkpoint.latest(root) if distributed.is_rank_zero() else None
    name = distributed.gather_objects(chosen.name if chosen is not None else None)[0]
    if name is None:
        return None
    resume_from = root / name
    if not distributed.all_ranks_ok(resume_from in checkpoint.step_directories(root)):
        raise RuntimeError(
            f"rank zero resumes {name}, and another rank cannot read it under {root}.\n"
            "Expected every rank to see the same published step directory.\n"
            "Put the run directory on a filesystem every rank shares."
        )
    return resume_from


def run(cfg, run_dir: str | Path) -> int:
    """Train one run. Returns the step it stopped at."""
    distributed.init_process_group()
    device = distributed.resolve_device()
    torch.manual_seed(int(cfg.seed) + distributed.rank())

    plan = resolve_batch_plan(
        cfg.schedule.global_batch,
        cfg.schedule.per_device_batch,
        cfg.schedule.gradient_accumulation_steps,
        distributed.world_size(),
    )
    logger.info(
        "batch plan: global %d = per device %d x accumulation %d x world size %d",
        plan.global_batch, plan.per_device_batch, plan.accumulation, plan.world_size,
    )
    # Resolved now, before the model is built and a checkpoint is read, so a data
    # group naming an unset environment variable stops the launch in seconds
    # rather than minutes into it on every rank.
    OmegaConf.to_container(cfg.data, resolve=True)

    strategy_name = str(cfg.trainer.strategy)
    options = OmegaConf.to_container(cfg.trainer.strategy_options, resolve=True)
    strategy = build_strategy(strategy_name, options)
    policy = MODELS.get(str(cfg.model.name))(cfg)
    # A model that normalises through an embodiments file gets the data
    # config's before anything reads a checkpoint: init_from and resume hold
    # their inherited entries against it (R8).
    if "embodiments" in vars(policy):
        policy.embodiments = load_embodiments(cfg)
        check_history_knobs(cfg)
    compile_stacks(cfg, policy, strategy_name)
    compile_robot_path(cfg, policy, strategy_name)
    robot_recompute(cfg, policy)
    trim_prefix(cfg, policy)
    if bool(cfg.trainer.gradient_checkpointing):
        # Before the wrap, because a sharded unit reads the flag as it builds.
        policy.encoder.backbone.gradient_checkpointing = True
        logger.info("recomputing the backbone's layer activations")
    model = strategy.wrap(policy, device)
    module = strategy.unwrap(model)
    logger.info("model %s on %s under %s", cfg.model.name, device, strategy_name)
    # After the wrap: an optimizer built before it binds the unsharded weights.
    optimizer, scheduler = module.configure_optimizers(cfg)
    # EMA is built AFTER initialise_from / before checkpoint.load, inside each branch
    # below. Ema.__init__ clones every trainable param at construction (loop.py), so an
    # average built here -- before init_from lands -- starts from the BUILD weights and
    # only decays toward the real ones (36.8% stale at step 1000 with decay 0.999).

    checkpoint_root = Path(run_dir) / "checkpoints"
    if distributed.is_rank_zero():
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    distributed.barrier()
    start_step = 0
    parent = None
    lineage = None
    resume_from = choose_resume(checkpoint_root)
    if resume_from is None:
        logger.info("no checkpoint under %s; training from step 0", checkpoint_root)
        if cfg.trainer.init_from is not None:
            init_from = Path(cfg.trainer.init_from)
            # Stated, never defaulted: stage 2 starts from stage 1's average
            # (DESIGN 1.1), and which weights a run took belongs in its log;
            # whether the run may add keys to the lineage is the group's.
            use_ema = required_switch(cfg, "trainer.init_use_ema")
            new_keys = required_switch(cfg, "data.new_keys")
            # The frozen-ft twin's switch (init_weights); absent everywhere else, so every
            # other run reads its init exactly as before.
            drop_chat_head = bool(
                OmegaConf.select(cfg, "trainer.init_drop_chat_head", default=False)
            )
            parent = initialise_from(
                init_from, strategy, model, use_ema,
                list(OmegaConf.select(cfg, "trainer.init_skip", default=None) or ()),
                new_keys=new_keys, drop_chat_head=drop_chat_head,
            )
            logger.info("init_from %s: %s weights", init_from,
                        "EMA over raw" if use_ema else "raw (the model entry)")
            if vars(module).get("embodiments") is not None:
                lineage = read_step_embodiments(init_from)[0]
        ema = build_ema(cfg, module, strategy_name)
    else:
        ema = build_ema(cfg, module, strategy_name)
        if vars(module).get("embodiments") is not None:
            # The resumed step's file: its entries must be the run's, and its
            # parent is the run's parent.
            checkpoint.check_step(resume_from, module, f"resume checkpoint {resume_from}")
            check_inherited(module, resume_from, f"resume checkpoint {resume_from}",
                            new_keys=False)
            parent = step_entry_digests(resume_from)[1]
        configured = [float(group["initial_lr"]) for group in optimizer.param_groups]
        start_step = checkpoint.load(
            resume_from, strategy, model, optimizer, scheduler, ema
        )
        check_resumed_rates(configured, scheduler, resume_from)
        steps = set(distributed.gather_objects(start_step))
        if len(steps) > 1:
            raise RuntimeError(
                f"the ranks resumed {resume_from.name} at different steps {sorted(steps)}.\n"
                "Expected one step, since every rank read the same directory.\n"
                "Check that the checkpoint root is one filesystem, not a copy per node."
            )

    loader = build_loader(cfg, plan, start_step, device)
    if hasattr(module, "check_dataset"):
        module.check_dataset(loader.dataset)
    if vars(module).get("embodiments") is not None:
        stamp_embodiments(cfg, module, loader.dataset, Path(run_dir), parent, lineage,
                          fit_step=resume_from)
    spec = LoopSpec(
        max_steps=int(cfg.schedule.max_steps),
        accumulation=plan.accumulation,
        gradient_clip=float(cfg.trainer.gradient_clip),
        log_every=int(cfg.trainer.log_every),
        save_every=int(cfg.trainer.save_every),
        keep_checkpoints=int(cfg.trainer.keep_checkpoints),
        keep_skipped=int(cfg.trainer.keep_skipped),
        samples_per_step=plan.global_batch,
    )
    result = train(
        model, loader, optimizer, scheduler, strategy, spec, device,
        checkpoint_root, start_step=start_step, ema=ema,
    )
    logger.info("run finished at step %d", result.step)
    release_device_cache()
    distributed.destroy_process_group()
    return result.step


def release_device_cache() -> None:
    """Hand the allocator's cached blocks back to the device once training is over.

    The process group's teardown on rank 0 took about 1.9 GiB of device memory above the
    run's own peak, on top of what the caching allocator still held (measured on 8-rank
    stage-2 runs with nvidia-smi after "run finished"). Every tensor of the run is
    still alive here, so only blocks nothing uses are returned; no number changes.
    """
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# The subtree a stage-2 step holds and a build with no chat half does not
# (model.cotrain null): the co-train head.
CHAT_HEAD_PREFIX = "cotrain_head."


def init_weights(payload, model, state: Path, drop_chat_head: bool, *, use_ema: bool = True) -> dict:
    """The model entry initialise_from loads: the checkpoint's, less its chat head on request.

    drop_chat_head is trainer.init_drop_chat_head, for a build with no chat head started
    from a co-trained step. Such a build has no chat head, and a stage-2 step
    holds cotrain_head.head.weight in its model entry and in its EMA, which initialise_from
    refuses as a tensor with no slot. With the switch, those tensors leave both entries
    before the EMA overlay, so the counts logged after are this build's. It refuses a build
    that has a head and a checkpoint that holds none: it never drops anything by accident.
    """
    weights = checkpoint.model_entry(payload)
    if not drop_chat_head:
        return weights
    if model.cotrain_head is not None:
        raise ValueError(
            "trainer.init_drop_chat_head is set, but this build has a co-train head.\n"
            "Expected a build with no chat half (model.cotrain null, the frozen-ft twin).\n"
            "Unset trainer.init_drop_chat_head: a co-train build takes the head too."
        )
    ema = (payload.get("ema") or {}) if use_ema else {}
    dropped = sorted(key for key in weights if key.startswith(CHAT_HEAD_PREFIX))
    from_ema = sorted(key for key in ema if key.startswith(CHAT_HEAD_PREFIX))
    if not dropped:
        raise ValueError(
            f"trainer.init_drop_chat_head is set, but {state} holds no "
            f"{CHAT_HEAD_PREFIX} tensor.\n"
            "Expected a step of a co-train run (stage 2), whose head this build drops.\n"
            "Unset trainer.init_drop_chat_head for a step without a chat head (stage 1)."
        )
    for key in dropped:
        del weights[key]
    for key in from_ema:
        del ema[key]
    logger.info(
        "init_drop_chat_head: dropped %d tensors from the checkpoint's model entry and %d "
        "from its EMA, beginning %s; this build has no co-train head",
        len(dropped), len(from_ema), dropped[:3],
    )
    return weights
