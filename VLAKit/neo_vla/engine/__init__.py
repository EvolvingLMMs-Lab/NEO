"""The training engine: a step loop, one checkpoint format, and the ranks.

Nothing here knows what a policy is. The loop calls model(batch, ctx) and
reads loss out of the dict it gets back. The composition root lives in
neo_vla.engine.runner, which is imported on its own so that loading this
package does not pull in hydra.
"""
from . import checkpoint, distributed
from .context import StepContext
from .logging import ProgressLogger, setup_logging
from .loop import Ema, LoopResult, LoopSpec, check_trainable, train
from .registry import MODELS, Registry
from .sampler import ResumableSampler
from .strategy import DDPStrategy, FSDPStrategy, Strategy, build_strategy

__all__ = [
    "DDPStrategy",
    "Ema",
    "FSDPStrategy",
    "LoopResult",
    "LoopSpec",
    "MODELS",
    "ProgressLogger",
    "Registry",
    "ResumableSampler",
    "StepContext",
    "Strategy",
    "build_strategy",
    "check_trainable",
    "checkpoint",
    "distributed",
    "setup_logging",
    "train",
]
