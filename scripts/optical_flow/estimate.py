from __future__ import annotations

import os
from typing import TYPE_CHECKING, Final

import hydra
from dotenv import load_dotenv
from kaparoo.filesystem import ensure_dir_exists

from scripts._common.compute import ComputeConfig, WorkerLogFolder, run_all
from scripts._common.dataset import SequenceSelectConfig
from scripts._common.hydra import (
    apply_schema,
    ensure_sweep_runs,
    is_multirun,
    output_directory,
)
from scripts.data._filtering import parse_filter_config
from scripts.optical_flow._estimating import parse_estimator_config
from scripts.optical_flow._normalizing import NormalizeConfig
from scripts.optical_flow._process import (
    FlowSourceConfig,
    FlowTargetConfig,
    build_flow_stages,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


load_dotenv()

CONFIG_PATH: Final = os.environ["CONFIGS_ROOT"]
CONFIG_NAME: Final = "optical_flow/estimate/config"

STAGE: Final = "optical_flow"


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(config: DictConfig) -> None:
    ensure_sweep_runs()

    source_config = apply_schema(FlowSourceConfig, config.source)
    sequence_config = apply_schema(SequenceSelectConfig, config.select)
    kernel_config = parse_filter_config(config.get("filter"))
    normalize_config = apply_schema(NormalizeConfig, config.normalize)
    estimator_config = parse_estimator_config(config.get("estimator"))
    target_config = apply_schema(FlowTargetConfig, config.target)
    compute_config = apply_schema(ComputeConfig, config.compute)

    if target_config.flows.save and is_multirun():
        msg = "cannot write flows in a sweep: run the winning config alone instead"
        raise ValueError(msg)

    output_root = ensure_dir_exists(output_directory())

    stages = build_flow_stages(
        source_config,
        sequence_config,
        estimator_config,
        normalize_config,
        kernel_config,
        target_config,
        output_root=output_root,
        name=STAGE,
    )

    log_folder = WorkerLogFolder(output_root, stages.name)
    log_folder.clear()

    run_all(stages, compute_config, unit="seq", log_folder=log_folder)


if __name__ == "__main__":
    main()
