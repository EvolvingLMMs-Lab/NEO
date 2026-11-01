"""Hydra entry point. One entry for every run.

    torchrun --nproc_per_node=8 train.py run_name=skip28_pretrain \
        arch=skip28 mode=full_ft data=oxe3src schedule=pretrain

Choices come from the config groups under config/. run_name fixes the run
directory, which holds the resolved config and the checkpoints; relaunching
with the same run_name is what resumes a run.
"""
import importlib
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig

from neo_vla.engine import runner, setup_logging


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg) -> int:
    # Importing the model package is what puts its factories in the registry.
    importlib.import_module("neo_vla.model")
    setup_logging()
    return runner.run(cfg, Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()
