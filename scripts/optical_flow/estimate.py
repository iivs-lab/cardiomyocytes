from __future__ import annotations

import os
from typing import TYPE_CHECKING, Final

import hydra
from dotenv import load_dotenv
from kaparoo.filesystem import ensure_dir_exists

from scripts._common.compute import WorkerLogFolder, run_all
from scripts._common.hydra import ensure_sweep_runs, is_multirun, output_directory
from scripts.optical_flow._process import FlowInputs, build_flow_stages

if TYPE_CHECKING:
    from omegaconf import DictConfig


load_dotenv()

CONFIG_PATH: Final = os.environ["CONFIGS_ROOT"]
CONFIG_NAME: Final = "optical_flow/estimate/config"

STAGE: Final = "optical_flow"


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(config: DictConfig) -> None:
    ensure_sweep_runs()

    inputs = FlowInputs.read(config)

    if inputs.target.flows.save and is_multirun():
        msg = "cannot write flows in a sweep: run the winning config alone instead"
        raise ValueError(msg)

    output_root = ensure_dir_exists(output_directory())

    stages = build_flow_stages(
        inputs.source,
        inputs.select,
        inputs.estimator,
        inputs.normalize,
        inputs.kernel,
        inputs.target,
        output_root=output_root,
        name=STAGE,
    )

    log_folder = WorkerLogFolder(output_root, stages.name)
    log_folder.clear()

    run_all(stages, inputs.compute, unit="seq", log_folder=log_folder)


if __name__ == "__main__":
    main()
