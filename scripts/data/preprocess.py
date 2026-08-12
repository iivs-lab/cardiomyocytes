from __future__ import annotations

import os
from typing import TYPE_CHECKING, Final

import hydra
from dotenv import load_dotenv
from kaparoo.filesystem import ensure_dir_exists

from scripts._common.compute import ComputeConfig, WorkerLogFolder, run_all
from scripts._common.dataset import SequenceSelectConfig, SourceConfig
from scripts._common.hydra import apply_schema, is_multirun, output_directory
from scripts.data._process import TargetConfig, build_preprocess_stages

if TYPE_CHECKING:
    from omegaconf import DictConfig


load_dotenv()

CONFIG_PATH: Final = os.environ["CONFIGS_ROOT"]
CONFIG_NAME: Final = "data/preprocess/config"

STAGE: Final = "preprocess"


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(config: DictConfig) -> None:
    compute_config = apply_schema(ComputeConfig, config.compute)
    source_config = apply_schema(SourceConfig, config.source)
    target_config = apply_schema(TargetConfig, config.target)
    sequence_config = apply_schema(SequenceSelectConfig, config.select)
    filter_config: DictConfig | None = config.get("filter")

    if target_config.frames.save and is_multirun():
        msg = "cannot write frames in a sweep: run the winning config alone instead"
        raise ValueError(msg)

    output_root = ensure_dir_exists(output_directory())

    stages = build_preprocess_stages(
        source_config,
        sequence_config,
        target_config,
        filter_config,
        output_root=output_root,
        name=STAGE,
    )

    log_folder = WorkerLogFolder(output_root, stages.name)
    log_folder.clear()

    run_all(stages, compute_config, unit="seq", log_folder=log_folder)


if __name__ == "__main__":
    main()
