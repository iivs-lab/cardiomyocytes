from __future__ import annotations

import os
from typing import TYPE_CHECKING

import hydra
from dotenv import load_dotenv

from scripts._compute import ComputeConfig, WorkerLogFolder, run_all
from scripts._hydra import apply_schema, is_multirun, output_directory
from scripts.data._process import SourceConfig, TargetConfig, build_phase_stages

if TYPE_CHECKING:
    from omegaconf import DictConfig


load_dotenv()

CONFIG_PATH = os.environ["CONFIGS_ROOT"]
CONFIG_NAME = "data/preprocess/config"

STAGE = "preprocess"


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    compute_config = apply_schema(ComputeConfig, cfg.compute)
    source_config = apply_schema(SourceConfig, cfg.source)
    target_config = apply_schema(TargetConfig, cfg.target)
    filter_config: DictConfig | None = cfg.filter

    if target_config.save_frames and is_multirun():
        msg = "cannot write frames in a sweep: run the winning config alone instead"
        raise ValueError(msg)

    log_folder = WorkerLogFolder(output_directory())
    log_folder.clear()

    stages = build_phase_stages(
        source_config,
        target_config,
        filter_config,
        name=STAGE,
        output_root=log_folder.root,
    )

    run_all(stages, compute_config, unit="seq", log_folder=log_folder)


if __name__ == "__main__":
    main()
