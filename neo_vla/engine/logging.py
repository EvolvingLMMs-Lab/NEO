"""Module logger setup and the rank-zero progress line.

setup_logging installs one handler with the rank in the format and quiets every
rank but zero, so an eight-process node does not repeat each line eight times.
ProgressLogger writes the training line. Its rate comes from elapsed wall time
over the interval, not from an average of per-step times: step times are
bimodal once data loading stalls, and the per-step average reads high.
"""
from __future__ import annotations

import logging
import time
from datetime import timedelta

from . import distributed

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Install one stream handler. Ranks above zero report warnings only."""
    process_rank = distributed.rank()
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            f"%(asctime)s %(levelname)s rank{process_rank} %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level if process_rank == 0 else logging.WARNING)


class ProgressLogger:
    """The rank-zero training line, written every `log_every` steps."""

    def __init__(self, max_steps: int, log_every: int, samples_per_step: int, start_step: int = 0):
        self.max_steps = int(max_steps)
        self.log_every = max(int(log_every), 1)
        self.samples_per_step = int(samples_per_step)
        self.start_step = int(start_step)
        self.start_time = time.monotonic()
        self.window_step = int(start_step)
        self.window_time = self.start_time

    def due(self, step: int) -> bool:
        return step % self.log_every == 0 or step >= self.max_steps

    def update(
        self,
        step: int,
        loss: float,
        grad_norm: float | None = None,
        extra: dict[str, float | None] | None = None,
    ) -> None:
        """One progress line. extra appends named terms in the order given.

        An empty or absent extra writes the line this wrote before extra
        existed, character for character, which is what lets an optional loss
        term be watched without changing the log of a run that has none. A term
        given as None had no reading in the window and is written n/a.
        """
        now = time.monotonic()
        steps = step - self.window_step
        elapsed = max(now - self.window_time, 1e-9)
        self.window_step = step
        self.window_time = now
        if not distributed.is_rank_zero():
            return
        rate = steps / elapsed
        done = max(step - self.start_step, 1)
        eta = timedelta(seconds=int((self.max_steps - step) * (now - self.start_time) / done))
        line = (
            f"step {step}/{self.max_steps} loss {loss:.4f} "
            f"{rate:.2f} step/s {rate * self.samples_per_step:.1f} sample/s eta {eta}"
        )
        if grad_norm is not None:
            line += f" grad_norm {grad_norm:.3f}"
        for name, value in (extra or {}).items():
            line += f" {name} n/a" if value is None else f" {name} {value:.5f}"
        logger.info("%s", line)
