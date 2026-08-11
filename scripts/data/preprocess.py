from __future__ import annotations

import os
from typing import TYPE_CHECKING, Final

import hydra
from dotenv import load_dotenv

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
def main(cfg: DictConfig) -> None:
    """Run one job: read the settings, build the run, and carry it out.

    Writing frames is refused in a sweep, since every job of a sweep would
    write the same tree and the last one would be all that was left.

    Raises:
        ValueError: If the settings ask for frames in a sweep, or describe
            a run that cannot be built.
        IncompleteRunError: If any sequence failed.
    """
    compute_config = apply_schema(ComputeConfig, cfg.compute)
    source_config = apply_schema(SourceConfig, cfg.source)
    select_config = apply_schema(SequenceSelectConfig, cfg.select)
    target_config = apply_schema(TargetConfig, cfg.target)
    filter_config: DictConfig | None = cfg.get("filter")

    if target_config.frames.save and is_multirun():
        msg = "cannot write frames in a sweep: run the winning config alone instead"
        raise ValueError(msg)

    log_folder = WorkerLogFolder(output_directory(), STAGE)

    stages = build_preprocess_stages(
        source_config,
        select_config,
        target_config,
        filter_config,
        output_root=log_folder.root,
        name=STAGE,
    )

    log_folder.clear()
    run_all(stages, compute_config, unit="seq", log_folder=log_folder)


if __name__ == "__main__":
    main()
