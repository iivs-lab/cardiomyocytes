from __future__ import annotations

import os
from typing import TYPE_CHECKING, Final

import hydra
from dotenv import load_dotenv
from kaparoo.filesystem import ensure_dir_exists

from scripts._common.compute import WorkerLogFolder, run_all
from scripts._common.hydra import ensure_sweep_runs, is_multirun, output_directory
from scripts.optical_flow._process import FlowConfig, build_flow_stages

if TYPE_CHECKING:
    from omegaconf import DictConfig


load_dotenv()

CONFIG_PATH: Final = os.environ["CONFIGS_ROOT"]
CONFIG_NAME: Final = "optical_flow/estimate/config"

STAGE: Final = "optical_flow"


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(config: DictConfig) -> None:
    ensure_sweep_runs()

    config: FlowConfig = FlowConfig.read(config)

    if config.target.flows.save and is_multirun():
        msg = "cannot write flows in a sweep: run the winning config alone instead"
        raise ValueError(msg)

    output_root = ensure_dir_exists(output_directory())

    stages = build_flow_stages(
        config.source,
        config.select,
        config.estimator,
        config.normalize,
        config.kernel,
        config.target,
        device=config.compute.device,
        output_root=output_root,
        name=STAGE,
    )

    log_folder = WorkerLogFolder(output_root, stages.name)
    log_folder.clear()

    run_all(stages, config.compute, unit="seq", log_folder=log_folder)


if __name__ == "__main__":
    main()
