from __future__ import annotations

import os
from typing import TYPE_CHECKING

import hydra
from dotenv import load_dotenv

if TYPE_CHECKING:
    from omegaconf import DictConfig

load_dotenv()

CONFIG_PATH = os.environ["CONFIGS_ROOT"]
CONFIG_NAME = "optical_flow/estimators/config"


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None: ...


if __name__ == "__main__":
    main()
