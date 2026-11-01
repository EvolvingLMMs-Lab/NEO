"""Resumable sampler: a fixed number of equal batches on every rank.

Training is bounded by steps, so the sampler emits exactly the micro-batches
the run still owes: num_batches of them per rank, starting at start_batch. A
pass over the data is a permutation of the seed and the pass index, truncated
to a whole number of global batches, so no rank runs out of batches before
another and hangs the collectives. Rank r takes slice r of each global batch,
and a resume starts at its step's batch rather than at the top of the data.

An index is a draw number: pass p yields p * dataset_size + position. A
dataset reads its window from index % len(dataset) and keys its per-draw
randomness by the whole index, so a position replayed in a later pass is a new
draw (history step, dropout, noise and reference camera depend on index, pass
and seed). Every index is a pure function of (seed,
world size, rank, batch), so a resume reproduces them.
"""
from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

import torch
from torch.utils.data import Sampler

# Keeps pass p of one seed from repeating pass 0 of the next seed.
SEED_STRIDE = 1_000_003


class ResumableSampler(Sampler[int]):
    """Index stream for one rank, resumable at a micro-batch boundary.

    One pass is one int64 permutation of dataset_size, held by the rank that
    iterates it. Measured at 307,200,000 = 75,000 x 4,096 draws: 2.29 GiB per
    rank and 16.5 s on one thread. Stage 1's 928,972,800 = 113,400 x 8,192 is
    3.02 x that, 6.92 GiB per rank and about 50 s [derived], so the permutation
    stays a plain randperm.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        num_batches: int,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 0,
        start_batch: int = 0,
    ):
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.seed = int(seed)
        self.start_batch = int(start_batch)
        self.global_batch = self.batch_size * self.world_size
        self.batches_per_pass = self.dataset_size // self.global_batch
        if self.batches_per_pass < 1:
            raise ValueError(
                f"dataset holds {self.dataset_size} samples, one global batch is "
                f"{self.batch_size} x {self.world_size} = {self.global_batch}.\n"
                "Expected at least one whole global batch of data.\n"
                "Lower per_device_batch, or point the run at more data."
            )
        self.usable = self.batches_per_pass * self.global_batch
        self.cached_pass = -1
        self.cached_order = torch.empty(0, dtype=torch.long)

    def order(self, pass_index: int) -> torch.Tensor:
        """The positions of one pass, a permutation truncated to whole global batches."""
        if pass_index != self.cached_pass:
            generator = torch.Generator()
            generator.manual_seed(self.seed * SEED_STRIDE + pass_index)
            permutation = torch.randperm(self.dataset_size, generator=generator)
            # A tensor, because the same indices as python integers cost five
            # times the memory and a large corpus keeps a pass of them per rank.
            self.cached_order = permutation[: self.usable]
            self.cached_pass = pass_index
        return self.cached_order

    def __iter__(self) -> Iterator[int]:
        for batch in range(self.start_batch, self.start_batch + self.num_batches):
            pass_index, offset = divmod(batch, self.batches_per_pass)
            order = self.order(pass_index)
            first = offset * self.global_batch + self.rank * self.batch_size
            positions = order[first : first + self.batch_size]
            yield from (positions + pass_index * self.dataset_size).tolist()

    def __len__(self) -> int:
        return self.num_batches * self.batch_size


class StepItemSampler(Sampler):
    """A rank's index stream, a whole optimizer step at a time.

    Wraps the rank's ResumableSampler and yields StepItem(step, indices):
    per_device x accumulation consecutive indices of its stream, the draws
    the micro-batches of that step read, in sampler order, starting at the
    sampler's start step. The stream is the ResumableSampler's own, so a
    packing run draws exactly the indices a run without packing draws, and a
    resume at step s yields the items of the uninterrupted run from s on.
    The sampler's batches are whole steps (build_loader: start_step x
    accumulation and remaining x accumulation), so no item is ragged.
    """

    def __init__(self, sampler: ResumableSampler, accumulation: int):
        # Imported here: the step item is the data stack's, and the engine
        # package imports this module before any of it.
        from ..data.chat_pack import StepItem

        self.sampler = sampler
        self.accumulation = int(accumulation)
        if self.accumulation < 1 or sampler.start_batch % self.accumulation \
                or sampler.num_batches % self.accumulation:
            raise ValueError(
                f"the sampler starts at micro-batch {sampler.start_batch} and yields "
                f"{sampler.num_batches} at accumulation {self.accumulation}.\n"
                "Expected whole steps: both a multiple of the accumulation.\n"
                "Build the sampler at start_step x accumulation for remaining x accumulation."
            )
        self.item = StepItem
        self.start_step = sampler.start_batch // self.accumulation
        self.size = sampler.batch_size * self.accumulation

    def __iter__(self) -> Iterator:
        indices = iter(self.sampler)
        for step in range(self.start_step, self.start_step + len(self)):
            yield self.item(step, tuple(islice(indices, self.size)))

    def __len__(self) -> int:
        return self.sampler.num_batches // self.accumulation
