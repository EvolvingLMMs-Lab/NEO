"""Load one step directory for inference.

A step directory holds state.pt, meta.json and embodiments.json. The policy is
built from the run config, checked against meta.json before torch.load (the
same checks a resume and an init_from make), filled strictly, and handed the
step's embodiments file: the normaliser the weights were trained under, and
the only one their actions can be read back through.

A step copied out of its run directory is built from config/model/neo2b.yaml
with config/arch/skip28.yaml, the build the stage launchers train.
model.backbone_path still comes from NEO_VLA_BACKBONE.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ..constants import EMBODIMENTS_FILE
from ..data.embodiment import Embodiments
from ..engine import checkpoint
from ..engine.registry import MODELS
from ..model import NeoVlaPolicy

logger = logging.getLogger(__name__)

# Hydra writes the composed config here; the runner keeps steps in checkpoints/.
RUN_CONFIG = Path(".hydra") / "config.yaml"
CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
DEFAULT_ARCH = "skip28"
# A co-trained step carries the chat head, which serving does not build.
CHAT_HEAD_PREFIX = "cotrain_head."
WEIGHT_CHOICES = ("auto", "model", "ema")


@dataclass(frozen=True)
class LoadReport:
    step_dir: str
    step: int | None
    weights: str
    tensors: int
    dropped: tuple[str, ...]
    meta: dict = field(default_factory=dict)

    def line(self) -> str:
        step = "unknown step" if self.step is None else f"step {self.step}"
        dropped = f", dropped {len(self.dropped)} chat-head tensors" if self.dropped else ""
        return (f"loaded {self.step_dir} ({step}, {self.weights} weights): "
                f"{self.tensors} tensors{dropped}")


def default_config(arch: str = DEFAULT_ARCH):
    """model/neo2b.yaml with arch/<arch>.yaml, as Hydra composes them for training."""
    model = OmegaConf.merge(OmegaConf.load(CONFIG_DIR / "model" / "neo2b.yaml"),
                            OmegaConf.load(CONFIG_DIR / "arch" / f"{arch}.yaml"))
    return OmegaConf.create({"model": model})


def run_config(step_dir: str | Path, config_path: str | Path | None = None):
    """The config to build the step with.

    config_path when given; else run/.hydra/config.yaml, two levels above
    run/checkpoints/step_N; else the default build. A wrong build does not
    load: the stamps in meta.json and the strict load refuse it.
    """
    if config_path is not None:
        return OmegaConf.load(Path(config_path))
    found = Path(step_dir).resolve().parent.parent / RUN_CONFIG
    if found.is_file():
        return OmegaConf.load(found)
    logger.info("no run config beside %s; building model/neo2b.yaml with arch/%s.yaml",
                step_dir, DEFAULT_ARCH)
    return default_config()


def serving_config(cfg):
    """The run config with the chat half off; serving builds the robot path only."""
    served = copy.deepcopy(cfg)
    served.model.cotrain = None
    return served


def pick_weights(asked: str, average: dict | None, name: str) -> str:
    """'auto' takes the moving average when the step holds one."""
    if asked not in WEIGHT_CHOICES:
        raise ValueError(f"weights is {asked!r}.\nExpected one of {list(WEIGHT_CHOICES)}.")
    if asked == "auto":
        return "ema" if average is not None else "model"
    if asked == "ema" and average is None:
        raise ValueError(
            f"{name} holds no moving average and weights is 'ema'.\n"
            "Expected an ema entry in state.pt.\n"
            "Pass weights='model' or 'auto'."
        )
    return asked


def load_policy(
    step_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    weights: str = "auto",
) -> tuple[NeoVlaPolicy, LoadReport]:
    """Build, check and fill the policy of one step; return it in eval mode."""
    step_dir = Path(step_dir)
    cfg = serving_config(run_config(step_dir, config_path))
    policy = MODELS.get(str(cfg.model.name))(cfg, for_training=False)
    meta = checkpoint.check_step(step_dir, policy, f"serving step {step_dir}")
    payload = torch.load(step_dir / checkpoint.STATE_FILE, map_location="cpu", weights_only=False)
    state = checkpoint.model_entry(payload)
    # A step stripped down to its weights may no longer state the step number
    # or the fingerprint; a step straight from a run has both.
    if "fingerprint" in meta:
        saved, found = float(meta["fingerprint"]), checkpoint.fingerprint(state)
        if abs(found - saved) > checkpoint.FINGERPRINT_TOLERANCE * max(abs(saved), 1.0):
            raise ValueError(
                f"{step_dir.name} was written at fingerprint {saved:.9e}, its state.pt "
                f"reads {found:.9e}.\n"
                "Expected them to match; state.pt is not the file meta.json was written with.\n"
                "Copy the step directory whole."
            )
    average = payload.get("ema")
    chosen = pick_weights(weights, average, step_dir.name)
    if chosen == "ema":
        unknown = sorted(set(average) - set(state))
        if unknown:
            raise ValueError(
                f"{step_dir.name} averages {len(unknown)} tensors its model entry does not "
                f"hold, first {unknown[:3]}."
            )
        state.update(average)
    dropped = tuple(sorted(name for name in state if name.startswith(CHAT_HEAD_PREFIX)))
    for name in dropped:
        del state[name]
    policy.load_state_dict(state, strict=True)
    policy.embodiments = Embodiments.load(step_dir / EMBODIMENTS_FILE)
    policy.eval()
    policy.requires_grad_(False)
    policy.to(device)
    report = LoadReport(
        step_dir=str(step_dir),
        step=None if meta.get("step") is None else int(meta["step"]),
        weights=chosen,
        tensors=len(state),
        dropped=dropped,
        meta=dict(meta),
    )
    logger.info("%s", report.line())
    return policy, report
