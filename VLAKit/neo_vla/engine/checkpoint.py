"""One checkpoint format: a directory per step, published by rename.

save writes state.pt, meta.json and the files the model names (the policy's
embodiments.json) into a temporary directory, flushes them to the disk, and
renames the directory into place, so a step directory that exists is complete:
a full filesystem leaves a partial directory that no loader will look at.
meta.json holds the step, the world size, a fingerprint of the weights and the
model's stamps; load recomputes the fingerprint from the model it has just
filled and raises if the two disagree, which is how a resume proves it
happened. Every check a reader makes on meta.json and on the files beside it
runs before torch.load. The state is unsharded and includes the parameters the
mode froze, so the weights need no second file to be whole.
"""
from __future__ import annotations

import errno
import json
import logging
import os
import random
import shutil
import time
from collections.abc import Mapping
from pathlib import Path

import numpy
import torch
import torch.nn as nn

from .. import constants
from . import distributed

logger = logging.getLogger(__name__)

# v4 is the v7 layout: the 26-wide arm row, the 29-wide state record, the body
# block and its two <state> columns (Lp 908), with the state, action and body
# stamps in meta.json and embodiments.json v2 beside every step. A new name, so
# a tree from before it refuses these steps where it reads meta.json, and this
# tree reads no older one: an older step is served and continued by the tree
# that wrote it (OLD_TREES).
FORMAT = constants.CHECKPOINT_FORMAT
READABLE_FORMATS = (FORMAT,)
OLD_CHECKPOINT_FORMATS = constants.OLD_CHECKPOINT_FORMATS
OLD_TREES = {"neo_vla_step_v1": "dcd90a5", "neo_vla_step_v2": "dcd90a5",
             "neo_vla_step_v3": "0ca67f7"}
# The format name stops an older tree only where it reads meta.json: a resume and
# a serving load. Its trainer.init_from reads state.pt alone. So the model entry
# of every v4 state also holds this key, which no module has a slot for. Every
# tree that has init_from refuses a key its model has no slot for -- a strict
# load_state_dict until b82c1f9, an explicit refusal since -- and an older
# model_entry drops only its own format's key (v2's, v3's), so that tree stops
# here, naming this key, before a weight is taken. It is an integer, so fingerprint() skips it, and
# every reader in this tree takes the entry through model_entry(), which drops it.
FORMAT_GUARD = constants.FORMAT_GUARD
EMBODIMENTS_FILE = constants.EMBODIMENTS_FILE
STEP_PREFIX = "step_"
PARTIAL_SUFFIX = ".partial"

# A save that runs out of room on a shared volume is worth retrying: the space is usually another
# tenant's and comes back. The budget is bounded by the collective the OTHER ranks are blocked in
# -- distributed.DEFAULT_TIMEOUT_MINUTES is 30 -- so 60+180+300+420 = 960 s of sleeping leaves
# fourteen minutes of margin. Raising these past that trades a lost run for a hung one.
SPACE_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT})
SPACE_BACKOFF_SECONDS = (60, 180, 300, 420)
STATE_FILE = "state.pt"
META_FILE = "meta.json"
FINGERPRINT_TOLERANCE = 1e-9


def step_directories(root: str | Path) -> list[Path]:
    """Published step directories under `root`, oldest first."""
    root = Path(root)
    if not root.is_dir():
        return []
    found = []
    for path in sorted(root.glob(f"{STEP_PREFIX}*")):
        # A directory still being written is named for its step too, and a
        # crash can leave it holding both files; only the rename publishes it.
        if not path.is_dir() or not path.name[len(STEP_PREFIX):].isdigit():
            continue
        if (path / META_FILE).exists() and (path / STATE_FILE).exists():
            found.append(path)
    return found


def latest(root: str | Path) -> Path | None:
    """The newest published step directory, or None on a fresh run."""
    directories = step_directories(root)
    return directories[-1] if directories else None


def stated_keys(stated: dict) -> list[str] | None:
    """The embodiment keys meta.json states, or None when it states none.

    A step this tree writes lists them; an older step maps each key to a
    checksum string, which is not read.
    """
    keys = stated.get("keys") if isinstance(stated, dict) else None
    if isinstance(keys, dict):
        keys = list(keys)
    if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
        return None
    return sorted(keys)


def file_keys(data: bytes) -> list[str] | None:
    """The embodiment keys an embodiments.json holds, or None when it is not one."""
    try:
        document = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    entries = document.get("embodiments") if isinstance(document, dict) else None
    return sorted(entries) if isinstance(entries, dict) else None


def read_meta(path: str | Path) -> dict:
    """A step directory's meta.json, or R11 when it has none."""
    found = Path(path) / META_FILE
    if not found.is_file():
        raise ValueError(
            f"{constants.refusal_prefix(11)} {path} holds no {META_FILE}.\n"
            f"Expected a step directory of format {FORMAT}, which states itself there.\n"
            "Name a step directory written by this tree."
        )
    return json.loads(found.read_text(encoding="utf-8"))


def check_format(meta: dict, where: str) -> None:
    """R11 unless meta.json states this tree's checkpoint format."""
    stated = meta.get("format") if isinstance(meta, dict) else None
    if stated not in READABLE_FORMATS:
        older = " an older format" if stated in OLD_CHECKPOINT_FORMATS else ""
        raise ValueError(
            f"{constants.refusal_prefix(11)} {where} is format {stated!r},{older}.\n"
            f"Expected {FORMAT}: this tree reads no other, and "
            f"{list(OLD_CHECKPOINT_FORMATS)} are served and continued by the tree that "
            f"wrote them ({OLD_TREES.get(stated, 'the tree that wrote it')}).\n"
            "Name a step written by this tree, or train from scratch."
        )


def check_embodiments_file(path: str | Path, meta: dict, where: str) -> None:
    """R1 and R2 on the embodiments.json a step's meta.json names, before torch.load.

    A step whose meta names an embodiments file must hold it (R1), and the file
    must hold exactly the keys meta.json lists (R2): the normaliser the weights
    were trained under is part of the checkpoint. A meta that states none is
    left to the model's own check, which refuses it for a policy.
    """
    stated = meta.get("embodiments") if isinstance(meta, dict) else None
    if stated is None:
        return
    found = Path(path) / EMBODIMENTS_FILE
    if not found.is_file():
        raise ValueError(
            f"{constants.refusal_prefix(1)} {where} holds no {EMBODIMENTS_FILE}.\n"
            "Expected the embodiments file its meta.json names beside state.pt.\n"
            "Copy the step directory whole; the normaliser is part of the checkpoint."
        )
    held, keys = file_keys(found.read_bytes()), stated_keys(stated)
    if held is None or held != keys:
        raise ValueError(
            f"{constants.refusal_prefix(2)} {where}'s {EMBODIMENTS_FILE} holds keys "
            f"{held}, its meta.json names {keys}.\n"
            "Expected the file written beside these weights.\n"
            "Restore the step's own embodiments.json; another normaliser is a "
            "different model."
        )


def check_step(path: str | Path, module: nn.Module, where: str) -> dict:
    """Every check a reader makes before torch.load; returns meta.json.

    R11 on the format, R1 and R2 on the embodiments file, then the model's own
    checks on the stamps (pixel convention, prefix layout, state and history).
    """
    meta = read_meta(path)
    check_format(meta, where)
    check_embodiments_file(path, meta, where)
    check_model_meta(module, meta, where)
    return meta


def fingerprint(state: Mapping[str, torch.Tensor]) -> float:
    """Sum of absolute values over a state dict, in a fixed key order."""
    total = 0.0
    for name in sorted(state):
        tensor = state[name]
        if torch.is_tensor(tensor) and tensor.is_floating_point() and tensor.numel():
            total += float(tensor.detach().to(torch.float64).abs().sum())
    return total


def save(
    root: str | Path,
    step: int,
    strategy,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    ema=None,
    keep: int = 2,
) -> Path:
    """Write step `step` under `root`. Every rank calls this; rank zero writes."""
    # On every rank, before any collective: a raise here stops all of them.
    stated = model_meta(strategy.unwrap(model))
    files = model_files(strategy.unwrap(model), stated)
    model_state = strategy.model_state_dict(model)
    optim_state = strategy.optim_state_dict(model, optimizer)
    rng_by_rank = distributed.gather_objects(rng_state())
    target = Path(root) / f"{STEP_PREFIX}{int(step):09d}"
    failure = None
    if distributed.is_rank_zero():
        payload = {
            "model": model_state,
            "optimizer": optim_state,
            "scheduler": scheduler.state_dict(),
            "ema": ema.state_dict() if ema is not None else None,
            "rng_by_rank": rng_by_rank,
        }
        meta = {
            "format": FORMAT,
            "step": int(step),
            "world_size": distributed.world_size(),
            "fingerprint": fingerprint(model_state),
            **stated,
        }
        for attempt in range(len(SPACE_BACKOFF_SECONDS) + 1):
            try:
                publish(target, payload, meta, files)
                prune(root, keep)
            except OSError as error:
                failure = f"{type(error).__name__}: {error}"
                out_of_room = error.errno in SPACE_ERRNOS
                last = attempt >= len(SPACE_BACKOFF_SECONDS)
                if not out_of_room or last:
                    break
                # The .partial directory is holding the bytes that just failed to fit. Removing it
                # is what gives the retry room to succeed; leaving it would guarantee the same
                # failure with less space each time.
                partial = target.with_name(target.name + PARTIAL_SUFFIX)
                if partial.exists():
                    shutil.rmtree(partial, ignore_errors=True)
                delay = SPACE_BACKOFF_SECONDS[attempt]
                logger.warning(
                    "no room for %s (%s); dropped the partial write and retrying in %d s "
                    "(attempt %d of %d)",
                    target.name, failure, delay, attempt + 1,
                    len(SPACE_BACKOFF_SECONDS) + 1,
                )
                time.sleep(delay)
            else:
                failure = None
                logger.info(
                    "saved %s (fingerprint %.9e)%s",
                    target.name, meta["fingerprint"],
                    f" after {attempt + 1} attempts" if attempt else "",
                )
                break
    # This gather is where the other ranks wait, and it carries rank zero's
    # verdict, so a disk that refuses the write stops the job instead of
    # leaving them in a barrier rank zero never reaches.
    reason = distributed.gather_objects(failure)[0]
    if reason is not None:
        raise RuntimeError(
            f"rank zero could not write {target.name}: {reason}.\n"
            "Expected a writable checkpoint root with room for the state.\n"
            f"A space failure was retried up to {len(SPACE_BACKOFF_SECONDS)} times over "
            f"{sum(SPACE_BACKOFF_SECONDS)} s before this.\n"
            "Free space under the checkpoint root, or point the run at another disk."
        )
    return target


def model_meta(module: nn.Module) -> dict:
    """What the model asks meta.json to state beside the step, or nothing.

    The policy states its pixel convention, its prefix layout, its
    embodiments file's keys and its state and history stamps, which is what
    lets a resume, an init_from and a serving load refuse a checkpoint written
    under another. A model without the method writes the four fields always
    written.
    """
    stated = module.checkpoint_meta() if hasattr(module, "checkpoint_meta") else {}
    clash = sorted(set(stated) & {"format", "step", "world_size", "fingerprint"})
    if clash:
        raise ValueError(
            f"the model's checkpoint_meta names {clash}, which meta.json already holds.\n"
            "Expected fields of its own.\nRename them in checkpoint_meta."
        )
    return dict(stated)


def model_files(module: nn.Module, stated: dict) -> dict[str, bytes]:
    """The files the model asks to be written beside state.pt, checked against its meta.

    The policy names its embodiments.json, and meta.json names that file's
    keys; the two are held together here, before anything is written, so no
    writer can publish a step whose meta describes another file.
    """
    files = dict(module.checkpoint_files()) if hasattr(module, "checkpoint_files") else {}
    clash = sorted(set(files) & {STATE_FILE, META_FILE})
    if clash:
        raise ValueError(
            f"the model's checkpoint_files names {clash}, which the step already holds.\n"
            "Expected files of its own.\nRename them in checkpoint_files."
        )
    stated_file = stated.get("embodiments")
    written = files.get(EMBODIMENTS_FILE)
    held = None if written is None else file_keys(written)
    if (stated_file is None) != (written is None) or (
        written is not None and (held is None or held != stated_keys(stated_file))
    ):
        raise ValueError(
            f"meta.json names embodiments {stated_file!r} and the step would hold "
            f"{'no file' if written is None else f'a file with keys {held}'}.\n"
            f"Expected {EMBODIMENTS_FILE} beside state.pt exactly when meta names it, "
            "with the keys meta states.\n"
            "Take both from the same policy.embodiments."
        )
    return files


def check_model_meta(module: nn.Module, meta: dict, where: str) -> None:
    """Raise if the model says the checkpoint behind `meta` does not suit it."""
    if hasattr(module, "check_checkpoint_meta"):
        module.check_checkpoint_meta(meta, where)


def guarded(weights: Mapping[str, torch.Tensor]) -> dict:
    """A model entry as a v4 state.pt holds it: the weights and FORMAT_GUARD."""
    return {**weights, FORMAT_GUARD: torch.tensor(3, dtype=torch.int64)}


def model_entry(payload: Mapping) -> dict:
    """The weights of a state.pt, without FORMAT_GUARD."""
    weights = dict(payload["model"])
    weights.pop(FORMAT_GUARD, None)
    return weights


def publish(
    target: Path, payload: dict, meta: dict, files: Mapping[str, bytes] | None = None
) -> None:
    """Write every file, flush them, then rename the directory into place.

    A v3 step's model entry is written guarded, so no writer of this format
    can leave the guard out. files are written beside state.pt and meta.json
    inside the partial directory, so the rename publishes them together.
    """
    if meta.get("format") == FORMAT and payload.get("model") is not None:
        payload = {**payload, "model": guarded(payload["model"])}
    partial = target.with_name(target.name + PARTIAL_SUFFIX)
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    with open(partial / STATE_FILE, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    with open(partial / META_FILE, "wb") as handle:
        handle.write(json.dumps(meta, indent=2).encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    for name, data in sorted((files or {}).items()):
        with open(partial / name, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    sync_directory(partial)
    if target.exists():
        shutil.rmtree(target)
    partial.rename(target)
    sync_directory(target.parent)


def sync_directory(path: Path) -> None:
    """Flush a directory entry, so the rename survives a power cut."""
    handle = os.open(path, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def prune(root: str | Path, keep: int) -> None:
    """Drop all but the newest `keep` published step directories."""
    directories = step_directories(root)
    for old in directories[: max(len(directories) - max(int(keep), 1), 0)]:
        shutil.rmtree(old)


def load(
    path: str | Path,
    strategy,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    ema=None,
) -> int:
    """Fill the model, optimizer, scheduler and average; return the step."""
    path = Path(path)
    # Before the state is read: a resume of another format, without its
    # embodiments file or under other stamps is refused for the price of the
    # json already in hand.
    meta = check_step(path, strategy.unwrap(model), f"resume checkpoint {path}")
    payload = torch.load(path / STATE_FILE, map_location="cpu", weights_only=False)
    strategy.load_model_state_dict(model, model_entry(payload))
    strategy.load_optim_state_dict(model, optimizer, payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    local_reason = None
    if distributed.is_rank_zero():
        local_reason = restore_ema(path.name, ema, payload["ema"])
    # Only rank zero holds an average, so only rank zero can judge it; its
    # verdict travels to the others, which would otherwise walk into the
    # all-reduce inside verify that rank zero had already left.
    reason = distributed.gather_objects(local_reason)[0]
    if reason is not None:
        raise ValueError(reason)
    restore_rng(payload["rng_by_rank"], int(meta["world_size"]))
    verify(path, strategy, model, float(meta["fingerprint"]))
    step = int(meta["step"])
    logger.info("resumed from %s at step %d", path.name, step)
    return step


def restore_ema(name: str, ema, saved) -> str | None:
    """Fill the moving average from the checkpoint, or say why it cannot be.

    The caller raises the reason on every rank. Dropping an average the
    checkpoint holds would lose it for good: retention deletes the last
    directory that carries one, and no average can be rebuilt from weights.
    """
    if ema is None:
        if saved is None:
            return None
        return (
            f"{name} holds a moving average and this run keeps none.\n"
            "Expected trainer.ema_decay to hold the value the checkpoint was written with.\n"
            "Set trainer.ema_decay, or resume a checkpoint written without an average."
        )
    if saved is None:
        return (
            f"the run keeps a moving average, {name} holds none.\n"
            "Expected an ema entry in the checkpoint.\n"
            "Resume a checkpoint written with the average on, or set trainer.ema_decay null."
        )
    try:
        ema.load_state_dict(saved)
    except ValueError as error:
        return str(error)
    return None


def verify(path: Path, strategy, model: nn.Module, saved: float) -> None:
    """Compare the weights in memory with the fingerprint on the disk."""
    loaded = fingerprint(strategy.model_state_dict(model))
    agrees = True
    if distributed.is_rank_zero():
        agrees = abs(loaded - saved) <= FINGERPRINT_TOLERANCE * max(abs(saved), 1.0)
        if agrees:
            logger.info("resume fingerprint verified (%.9e)", loaded)
    if distributed.all_ranks_ok(agrees):
        return
    if distributed.is_rank_zero():
        raise RuntimeError(
            f"{path.name} was written at fingerprint {saved:.9e}, the model reads {loaded:.9e}.\n"
            "Expected them to match; the weights on disk are not the weights in memory.\n"
            "Resume an earlier step directory, or start the run again."
        )
    raise RuntimeError(
        f"resume of {path.name} failed verification on rank zero.\n"
        "Expected the loaded weights to match the fingerprint.\n"
        "Read rank zero's log for the two numbers."
    )


def rng_state() -> dict:
    """This rank's random state, for every generator the run draws from."""
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "numpy": numpy.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng(rng_by_rank: list, saved_world_size: int) -> None:
    """Put this rank's own state back. Sharing one state across ranks would
    make every rank draw the same augmentation."""
    if not rng_by_rank:
        return
    if saved_world_size != distributed.world_size():
        logger.warning(
            "checkpoint holds %d random states for %d ranks; reusing them in order",
            saved_world_size,
            distributed.world_size(),
        )
    state = rng_by_rank[distributed.rank() % len(rng_by_rank)]
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])
    numpy.random.set_state(state["numpy"])
    random.setstate(state["python"])
