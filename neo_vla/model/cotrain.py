"""The co-training head: a language head over the backbone's final hidden state.

Co-training mixes vision-language batches into the run so the backbone keeps
the language ability the robot data would otherwise erode. The loss is
next-token cross entropy under NEO's 'square' reduction (square_weighted_loss
below), and the total is flow_loss + cotrain.weight * cotrain_loss.

This module holds a language head and nothing else. It does not hold the
backbone, take one in its constructor, or hide one in a list to dodge submodule
registration: the policy runs encoder.encode_chat_rows (or encode_chat_pack)
and passes the hidden state in. So there is one owner of the backbone and one
set of parameter names. A chat pack's rows are scored per segment, each as a
row of its own: NEO's sum(w * CE) / sum(w) over a pack (row_losses).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from torch.utils.checkpoint import checkpoint as recompute

from ..engine import distributed
from ..engine.hostcopy import host_of, to_host_device
from .backbone.neo_llm import LANGUAGE_HEAD_PREFIX

# A label position that carries no supervision, which is torch's own default.
IGNORE_INDEX = -100

# The checkpoint tensor this head is filled from. load_backbone_weights skips
# it by the same prefix, so exactly one module loads it.
HEAD_WEIGHT = f"{LANGUAGE_HEAD_PREFIX}.weight"

# Supervised positions scored per recomputed chunk. The full-vocabulary logits
# of one chunk are what the head holds at a time: 1024 x 151936 in float32 is
# 0.6 GB, where a 4096-token row scored at once would hold 2.5 GB, and the
# chunks are recomputed in the backward rather than kept.
HEAD_CHUNK = 1024


def square_weighted_loss(
    ce_sums: torch.Tensor, counts: torch.Tensor, shared: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row summed cross entropy and target counts -> (loss, local mean).

    NEO's 'square' reduction (len2weight in its build_dataloader.py, the
    weighted loss in ce_loss.py, loss_reduction_all_gather true): every target
    of a row with n targets weighs n ** -0.5, and the weighted sum is divided
    by the weight sum averaged over the data-parallel ranks. shared is that
    average when the caller has it already, over every micro-batch of the step
    (step_weight_mean): gradients averaged over ranks and micro-batches then
    give sum(w * ce) / sum(w) over the whole step, as NEO's one micro-batch a
    step does. Without it, this micro-batch's own, at one all_reduce. The
    second value is this rank's own weighted mean, detached, for the log.
    The check reads counts' host copy (engine.hostcopy): a pack's counts were
    made on the host, so it costs no wait there (speed C-8+); rows' are copied
    back as before.
    """
    host_counts = host_of(counts)
    if bool((host_counts < 1).any()):
        raise ValueError(
            f"a chat row carries {host_counts.tolist()} targets.\n"
            "Expected at least one per row; the renderer drops a row with none.\n"
            "Fix the renderer rather than weighting an empty row."
        )
    roots = counts.to(torch.float32).sqrt()
    numerator = (ce_sums.to(torch.float32) / roots).sum()
    denominator = roots.sum()
    if shared is None:
        shared = distributed.all_reduce_mean(denominator)
    return numerator / shared, (numerator / denominator).detach()


def row_targets(labels: torch.Tensor) -> torch.Tensor:
    """Chat labels [B, T] -> each row's targets [B]: its labels that are not -100."""
    return (labels != IGNORE_INDEX).sum(1)


def row_sums(losses: torch.Tensor, owners: torch.Tensor, rows: int) -> torch.Tensor:
    """Losses [N] and the row owning each [N] -> each row's sum [rows], the same bits every call.

    index_add on a CUDA tensor adds through atomics, in the order the threads
    arrive: 200 calls on one H100 at a pack's size (15,000 losses over 59
    rows) gave 200 different results, and a resumed 8-rank run's first
    step, a forward of the same weights on the same batch, matched the
    uninterrupted run's chat loss on 1 to 5 of 8 ranks while every robot loss
    matched on all 8. index_put with accumulate sorts the owners and adds each
    row's losses in one order (the call index_add makes itself under
    torch.use_deterministic_algorithms): 1 result in 200, within float32
    rounding of the same sum. Its gradient is the same gather.
    """
    return losses.new_zeros(int(rows)).index_put((owners,), losses, accumulate=True)


def step_weight_mean(
    target_counts: list[torch.Tensor], micro_batches: int, device: torch.device
) -> torch.Tensor:
    """The chat targets of a step -> its square weight sum per micro-batch and rank.

    target_counts holds one vector per micro-batch with a chat half: each
    row's targets (row_targets of its labels), or each packed row's (the
    pack's segment_targets). Every row weighs sqrt(n) in the sum, packed or
    not. Averaged over every rank and all micro_batches of the step, so that
    each micro-batch dividing by it, and the loop averaging over micro-batches
    and ranks, gives the step's global weighted mean. One all_reduce, which
    every rank makes once a step, before the step's first forward.
    """
    total = torch.zeros((), dtype=torch.float32)
    for counts in target_counts:
        total += counts.to(torch.float32).sqrt().sum()
    return distributed.all_reduce_mean((total / int(micro_batches)).to(device))


class CoTrainHead(nn.Module):
    """The checkpoint's language head, and the cross entropy over it."""

    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        checkpoint_path: str | Path | None,
        dtype: torch.dtype,
    ):
        """Builds a bias-free linear and fills it from the checkpoint.

        checkpoint_path None leaves the head at its random initialisation,
        which is what the tests on a small backbone use. A checkpoint whose
        head is not [vocab_size, hidden_size] raises: the embedding table is
        never resized in this tree, so a mismatch means the two files disagree
        about the vocabulary.
        """
        super().__init__()
        self.head = nn.Linear(int(hidden_size), int(vocab_size), bias=False)
        if checkpoint_path is not None:
            self.load_head(Path(checkpoint_path))
        self.to(dtype)

    def load_head(self, path: Path) -> None:
        """Copy the checkpoint's language head into this one."""
        tensors = path / "model.safetensors"
        with safe_open(str(tensors), framework="pt", device="cpu") as reader:
            if HEAD_WEIGHT not in reader.keys():
                raise RuntimeError(
                    f"{tensors} carries no {HEAD_WEIGHT}.\n"
                    "Expected the language head co-training scores with.\n"
                    "Load a checkpoint that has one, or train without co-training."
                )
            weight = reader.get_tensor(HEAD_WEIGHT)
        if tuple(weight.shape) != tuple(self.head.weight.shape):
            raise RuntimeError(
                f"{HEAD_WEIGHT} is {tuple(weight.shape)} in {tensors}.\n"
                f"Expected {tuple(self.head.weight.shape)}, the built head's shape.\n"
                "The two files disagree about the vocabulary; this tree never resizes it."
            )
        with torch.no_grad():
            self.head.weight.copy_(weight.to(self.head.weight.dtype))

    def forward(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Refused: the token-mean loss of the legacy chat half is retired.

        A chat row is scored by row_losses and weighted by square_weighted_loss,
        NEO's own reduction; nothing calls the head as a module.
        """
        raise RuntimeError(
            "CoTrainHead was called for the token-mean chat loss, which is retired with "
            "the legacy chat format.\n"
            "Expected row_losses and square_weighted_loss, NEO's square reduction.\n"
            "Run the chat half through NeoVlaPolicy.cotrain_step."
        )

    def chunk_losses(self, hidden: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Selected hidden states [N, hidden] and their targets [N] -> CE [N], float32."""
        logits = self.head(hidden.to(self.head.weight.dtype))
        return F.cross_entropy(logits.float(), target, reduction="none")

    def row_losses(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        *,
        chunk: int = HEAD_CHUNK,
        segments: torch.Tensor | None = None,
        rows: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """hidden [B, T, hidden] and labels [B, T] -> (summed CE [B], targets [B]).

        The next-token cross entropy of NEO's chat rows, scored only at the
        supervised positions and in recomputed chunks, so the full
        [B, T, vocab] logits never exist (forward, the retired token mean,
        refuses). targets counts the labels that are
        not -100 before the shift, which is the n NEO weights a row by.

        For a pack, segments [1, T] numbers its rows (PAD_SEGMENT on the
        padding) and rows is how many there are; the sums and targets are then
        per packed row, [rows]. The shift runs over the whole pack, as NEO's
        collator shifts it (dataset_interleaved_iterable.py:1862-1875), which
        scores each row exactly as alone because every row opens with a
        masked label: a target and the position predicting it are in one row.
        That is checked here once a pack. A pack's supervised positions, that
        check and its per-row counts are computed on the labels' and segments'
        host copies (engine.hostcopy) and the indices moved to the device
        without waiting, so a pack costs the host no wait here (speed C-8+);
        the same index values reach the same gathers, so the sums are
        bitwise the ones the device-side indices gave. Rows keep the device path.
        """
        width = int(self.head.in_features)
        if hidden.dim() != 3 or int(hidden.shape[-1]) != width:
            raise ValueError(
                f"hidden is {tuple(hidden.shape)}, expected [batch, tokens, {width}].\n"
                "Pass encoder.encode_chat_rows's return value unchanged."
            )
        if tuple(labels.shape) != tuple(hidden.shape[:2]) or int(hidden.shape[1]) < 2:
            raise ValueError(
                f"labels is {tuple(labels.shape)} for hidden {tuple(hidden.shape)}.\n"
                "Expected one label per rendered token and rows of at least two tokens."
            )
        target = labels[:, 1:]
        if segments is None:
            rows_of, columns = (target != IGNORE_INDEX).nonzero(as_tuple=True)
            owners = rows_of
        else:
            if tuple(segments.shape) != tuple(labels.shape) or int(labels.shape[0]) != 1 \
                    or rows is None:
                raise ValueError(
                    f"segments is {tuple(segments.shape)} for labels {tuple(labels.shape)}, "
                    f"rows {rows!r}.\n"
                    "Expected a pack's [1, T] segments, one per label, and its row count.\n"
                    "Pass the pack's segments and len(segment_targets)."
                )
            host_labels, host_segments = host_of(labels), host_of(segments)
            host_rows, host_columns = (host_labels[:, 1:] != IGNORE_INDEX).nonzero(as_tuple=True)
            host_owners = host_segments[:, 1:][host_rows, host_columns]
            if not torch.equal(host_owners, host_segments[:, :-1][host_rows, host_columns]):
                raise ValueError(
                    "a pack's target sits in another row than the position predicting it.\n"
                    "Expected every row to open with a masked label, so the whole-pack shift "
                    "never scores one row's last token against the next row's first.\n"
                    "Keep every row's masked header (collate_chat_pack checks it)."
                )
            rows_of, columns, owners = (to_host_device(index, hidden.device)
                                        for index in (host_rows, host_columns, host_owners))
        picked = hidden[:, :-1][rows_of, columns]
        goals = target[rows_of, columns]
        pieces = []
        for start in range(0, int(goals.shape[0]), int(chunk)):
            end = start + int(chunk)
            if torch.is_grad_enabled():
                pieces.append(
                    recompute(self.chunk_losses, picked[start:end], goals[start:end],
                              use_reentrant=False)
                )
            else:
                pieces.append(self.chunk_losses(picked[start:end], goals[start:end]))
        losses = torch.cat(pieces) if pieces else picked.new_zeros(0, dtype=torch.float32)
        if segments is None:
            sums = row_sums(losses, owners, int(labels.shape[0]))
            return sums, row_targets(labels)
        sums = row_sums(losses, owners, int(rows))
        supervised = host_segments[host_labels != IGNORE_INDEX]
        return sums, to_host_device(torch.bincount(supervised, minlength=int(rows)), hidden.device)
