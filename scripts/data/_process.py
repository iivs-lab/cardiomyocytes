from __future__ import annotations

__all__ = (
    "SourceConfig",
    "TargetConfig",
    "build_phase_stages",
    "build_sequences",
    "search_sources",
)

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from iivs.dhm.data.koala import PHASE_FLOAT_BIN
from iivs.dhm.data.phase import PhaseFileFolder, PhaseUnit, search_phase_bin_folders
from kaparoo.filesystem import stringify_path
from kaparoo.filesystem.search import select
from kaparoo.utils.optional import unwrap_or_default, unwrap_or_factory
from omegaconf import MISSING

from iivs_cardio.data.pipeline import (
    FrameTree,
    PhaseFilteredSequence,
    PhaseStageFactory,
    RangeDocument,
)
from scripts._hydra import output_directory
from scripts.data._filtering import describe_filter_kernel, parse_filter_config

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from logging import Logger

    from kaparoo.filesystem.types import StrPath
    from omegaconf import DictConfig
    from torch import Tensor

    from iivs_cardio.common.pipeline import SideBranch
    from iivs_cardio.data.transforms.filtering.kernel import KernelConfig


@dataclass
class SourceConfig:
    root: str = MISSING
    subpath: str | None = None
    include: list[str] | str | None = None
    exclude: list[str] | str | None = None
    frame_step: int = 1


@dataclass
class TargetConfig:
    root: str = MISSING
    overwrite: bool = False
    save_frames: bool = False
    save_ranges: bool = True
    range_file: str = "value_range"


def _subpath(source_config: SourceConfig) -> str:
    """Which folder of a time-lapse a run reads, defaulted once."""
    return unwrap_or_default(source_config.subpath, PHASE_FLOAT_BIN)


def search_sources(config: SourceConfig) -> list[PhaseFileFolder]:
    subpath = _subpath(config)

    folders = search_phase_bin_folders(config.root, subpath=subpath)
    if (num_folders := len(folders)) == 0:
        msg = f"no time-lapse holds a {subpath!r} folder: {config.root}"
        raise ValueError(msg)

    def folder_subpath(folder: PhaseFileFolder) -> str:
        return stringify_path(folder.root, after=config.root, before=subpath)

    sources: list[PhaseFileFolder] = select(
        folders,
        key=folder_subpath,
        include=config.include,
        exclude=config.exclude,
    )

    if not sources:
        msg = f"include/exclude left none of the {num_folders} sequences: {config.root}"
        raise ValueError(msg)

    return [source.with_unit(PhaseUnit.RADIANS) for source in sources]


def build_sequences(
    source_config: SourceConfig, kernel_config: KernelConfig
) -> list[PhaseFilteredSequence]:
    sources = search_sources(source_config)

    kernel = kernel_config.build()
    subpath = _subpath(source_config)

    def build_sequence(source: PhaseFileFolder) -> PhaseFilteredSequence:
        return PhaseFilteredSequence(
            source,
            kernel,
            root=source_config.root,
            subpath=subpath,
            step=source_config.frame_step,
        )

    return [build_sequence(source) for source in sources]


def _log_source_config(source_config: SourceConfig, logger: Logger) -> None:
    logger.info("source configuration:")
    logger.info("  root: %s", source_config.root)
    logger.info("  subpath: %s", _subpath(source_config))
    logger.info("  frame step: %d", source_config.frame_step)

    def log_listed(name: str, value: list[str] | str | None) -> None:
        if value is None:
            return

        if isinstance(value, str):
            logger.info("  %s: %s", name, value)

        logger.info("  %s:", name)
        for item in value:
            logger.info("    %s", item)

    log_listed("include", source_config.include)
    log_listed("exclude", source_config.exclude)


def _log_filter_config(filtering_info: Mapping[str, Any], logger: Logger) -> None:
    logger.info("filter: %s", filtering_info["kind"])

    settings = ", ".join(
        f"{key}={value}" for key, value in filtering_info.items() if key != "kind"
    )
    logger.info(
        "source: filtered with the %s kernel%s",
        filtering_info["kind"],
        f" ({settings})" if settings else "",
    )


def _log_target_config(target_config: TargetConfig, logger: Logger) -> None:
    logger.info("target configuration:")
    logger.info("  root: %s", target_config.root)
    logger.info("  overwrite: %s", target_config.overwrite)
    logger.info("  save frames: %s", target_config.save_frames)
    logger.info("  save ranges: %s", target_config.save_ranges)
    logger.info("  range file: %s", target_config.range_file)


def _log_configs(
    source_config: SourceConfig,
    target_config: TargetConfig | None,
    kernel_config: KernelConfig,
    *,
    name: str,
) -> None:
    logger = logging.getLogger(name)

    _log_source_config(source_config, logger)
    _log_filter_config(describe_filter_kernel(kernel_config), logger)

    if target_config is not None:
        _log_target_config(target_config, logger)


def build_branches(
    source_config: SourceConfig,
    target_config: TargetConfig,
    kernel_config: KernelConfig,
    sequences: Sequence[PhaseFilteredSequence],
    *,
    output_root: StrPath | None = None,
) -> list[SideBranch[PhaseFilteredSequence, Tensor, Path]]:
    if not (target_config.save_ranges or target_config.save_frames):
        msg = "nothing to do: set `target.save_ranges` or `target.save_frames`"
        raise ValueError(msg)

    branches = []

    root = unwrap_or_factory(output_root, output_directory)
    overwrite = target_config.overwrite

    if target_config.save_frames:
        branches.append(FrameTree(root, _subpath(source_config), overwrite=overwrite))

    if target_config.save_ranges:
        path = Path(root, target_config.range_file)

        settings = {
            "source": {"frame_step": source_config.frame_step},
            "filter": describe_filter_kernel(kernel_config),
        }

        branches.append(
            RangeDocument(
                path,
                settings,
                expected=[sequence.name for sequence in sequences],
                source=source_config.root,
                overwrite=overwrite,
            )
        )

    return branches


def build_phase_stages(
    source_config: SourceConfig,
    target_config: TargetConfig | None = None,
    filter_config: DictConfig | None = None,
    *,
    name: str,
    output_root: StrPath | None = None,
) -> PhaseStageFactory:
    kernel_config = parse_filter_config(filter_config)

    _log_configs(source_config, target_config, kernel_config, name=name)

    sequences = build_sequences(source_config, kernel_config)
    branches = []

    if target_config is not None:
        branches = build_branches(
            source_config,
            target_config,
            kernel_config,
            sequences,
            output_root=output_root,
        )

    return PhaseStageFactory(sequences, *branches, name=name)
