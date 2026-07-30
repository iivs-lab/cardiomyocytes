from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import hydra
from dotenv import load_dotenv
from iivs.dhm.data.koala import PHASE_FLOAT_BIN
from iivs.dhm.data.phase import resolve_phase_unit, search_phase_bin_folders
from kaparoo.filesystem.search import select
from kaparoo.filesystem.utils import stringify_path
from kaparoo.utils.optional import unwrap_or_default
from omegaconf import MISSING

from scripts._config import apply_schema
from scripts.data._filtering import build_filter_kernel

if TYPE_CHECKING:
    from iivs.dhm.data.phase import PhaseFileFolder
    from omegaconf import DictConfig

load_dotenv()

CONFIG_PATH = os.environ["CONFIGS_ROOT"]
CONFIG_NAME = "data/phase_range/config"


@dataclass
class ComputeConfig:
    device: str = "cpu"
    workers: int | None = 0
    gpu_ids: list[int] | None = field(default_factory=lambda: [0])


@dataclass
class SourceConfig:
    root: str = MISSING
    unit: str | None = None
    subpath: str | None = None
    include: list[str] | str | None = None
    exclude: list[str] | str | None = None
    frame_step: int = 1


@dataclass
class TargetConfig:
    root: str = MISSING
    save_ranges: bool = True
    save_frames: bool = False


def search_sources(config: SourceConfig) -> list[PhaseFileFolder]:
    subpath = unwrap_or_default(config.subpath, PHASE_FLOAT_BIN)

    def folder_subpath(folder: PhaseFileFolder) -> str:
        return stringify_path(folder.root, after=config.root, before=subpath)

    sources: list[PhaseFileFolder] = select(
        search_phase_bin_folders(config.root, subpath=subpath),
        key=folder_subpath,
        include=config.include,
        exclude=config.exclude,
    )

    if not sources:
        msg = f"no time-lapse holds a {subpath!r} folder: {config.root}"
        raise ValueError(msg)

    if config.unit is not None:
        unit = resolve_phase_unit(config.unit)
        sources = [source.with_unit(unit) for source in sources]

    return sources


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    compute_config = apply_schema(ComputeConfig, cfg.compute)
    source_config = apply_schema(SourceConfig, cfg.source)
    target_config = apply_schema(TargetConfig, cfg.target)

    sources = search_sources(source_config)
    kernel = build_filter_kernel(cfg.filter)


if __name__ == "__main__":
    main()
