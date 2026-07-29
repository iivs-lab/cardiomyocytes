from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import hydra
from dotenv import load_dotenv
from iivs.dhm.data.phase import search_phase_bin_folders
from omegaconf import MISSING

from scripts._config import apply_schema

if TYPE_CHECKING:
    from iivs.dhm.data.phase import PhaseBinFolder
    from omegaconf import DictConfig

    from iivs_cardio.data.transforms.filtering import FilterKernel, KernelParams

load_dotenv()

CONFIG_PATH = os.environ["CONFIGS_ROOT"]
CONFIG_NAME = "data/phase_range/config"

# A `_target_` is a dotted path the config chooses and `instantiate` imports and
# calls, so hydra 1.4 wants the callsite to say what it means to build.
KERNEL_TARGETS = ("iivs_cardio.data.transforms.filtering.kernel.*",)


@dataclass
class SourceConfig:
    """Which sequences to read, and how to read each one."""

    root: str = MISSING
    unit: str | None = None
    include: list[str] | str | None = None
    exclude: list[str] | str | None = None
    frame_step: int = 1


@dataclass
class TargetConfig:
    """Where the outputs land, and which of them to write."""

    root: str = MISSING
    save_ranges: bool = True
    save_frames: bool = False


@dataclass
class ComputeConfig:
    """Where the work runs, and how many workers share it.

    `workers` and `gpu_ids` come from the `compute` config group, which supplies
    only the one its device uses; the other keeps the default here.
    """

    device: str = "cpu"
    workers: int | None = 0
    gpu_ids: list[int] | None = field(default_factory=lambda: [0])


def search_sources(config: SourceConfig) -> list[PhaseBinFolder]:
    return search_phase_bin_folders(config.root)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    compute_config = apply_schema(ComputeConfig, cfg.compute)
    source_config = apply_schema(SourceConfig, cfg.source)
    target_config = apply_schema(TargetConfig, cfg.target)

    kernel: FilterKernel | None = None
    if node := cfg.get("filter"):
        params: KernelParams = hydra.utils.instantiate(
            node, _target_whitelist_=KERNEL_TARGETS
        )
        kernel = params.build()

    sources = search_sources(source_config)
    if not sources:
        msg = f"no sources found in {source_config.root}"
        raise ValueError(msg)


if __name__ == "__main__":
    main()
