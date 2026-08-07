from __future__ import annotations

__all__ = (
    "SourceConfig",
    "TargetConfig",
    "build_branches",
    "build_phase_stages",
    "build_sequences",
    "log_configs",
    "log_filter_config",
    "log_source_config",
    "log_target_config",
    "search_sources",
)

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from iivs.dhm.data.koala import PHASE_FLOAT_BIN
from iivs.dhm.data.phase import PhaseFileFolder, PhaseUnit, search_phase_bin_folders
from kaparoo.filesystem import ensure_file_extension, stringify_path
from kaparoo.filesystem.search import select
from kaparoo.utils.optional import unwrap_or_default
from omegaconf import MISSING

from iivs_cardio.common.logging import log_indented
from iivs_cardio.data.pipeline import (
    DOCUMENT_EXT,
    FrameTree,
    PhaseFilteredSequence,
    PhaseStageFactory,
    RangeDocument,
)
from scripts.data._filtering import describe_filter_kernel, parse_filter_config

if TYPE_CHECKING:
    from collections.abc import Sequence
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


SELECTION_LIMIT = 5
SELECTION_SPECS = (".json", ".txt")


def _subpath(source_config: SourceConfig) -> str:
    return unwrap_or_default(source_config.subpath, PHASE_FLOAT_BIN)


def _log_selection(logger: Logger, verb: str, value: list[str] | str) -> None:
    if isinstance(value, str):
        if value.endswith(SELECTION_SPECS):
            log_indented(logger, "%s as listed in %s", verb, value)
        else:
            log_indented(logger, "%s %s", verb, value)
        return

    if (count := len(value)) > SELECTION_LIMIT:
        record = ".hydra/{config,overrides}.yaml"
        log_indented(logger, "%s %d, listed in %s", verb, count, record)
        return

    log_indented(logger, "%s:", verb)
    for item in value:
        log_indented(logger, "%s", item, depth=2)


def log_source_config(source_config: SourceConfig, logger: Logger) -> None:
    log_indented(logger, "source: %s", source_config.root, depth=0)

    log_indented(logger, "reading <sequence>/%s", _subpath(source_config))

    if (step := source_config.frame_step) > 1:
        kept = ", ".join(str(index * step) for index in range(3))
        log_indented(logger, "reading frames %s, ...", kept)

    if source_config.include:
        _log_selection(logger, "including", source_config.include)

    if source_config.exclude:
        _log_selection(logger, "excluding", source_config.exclude)


def log_filter_config(kernel_config: KernelConfig, logger: Logger) -> None:
    described = describe_filter_kernel(kernel_config)
    kind = described.pop("kind")
    settings = ", ".join(f"{key}={value}" for key, value in described.items())
    settings = f" ({settings})" if settings else ""
    log_indented(logger, "filter: %s kernel%s", kind, settings, depth=0)


def log_target_config(
    target_config: TargetConfig, logger: Logger, *, subpath: str | None = None
) -> None:
    log_indented(logger, "target: %s", target_config.root, depth=0)

    if target_config.save_frames:
        layout = f"<sequence>/{subpath}" if subpath else "<sequence>/*"
        log_indented(logger, "writing the filtered frames to %s", layout)

    if target_config.save_ranges:
        name = ensure_file_extension(target_config.range_file, DOCUMENT_EXT, add=True)
        log_indented(logger, "writing the value ranges to %s", name)

    if not (target_config.save_frames or target_config.save_ranges):
        log_indented(logger, "writing nothing")

    if target_config.overwrite:
        log_indented(logger, "replacing what is already there")


def log_configs(
    source_config: SourceConfig,
    target_config: TargetConfig | None,
    kernel_config: KernelConfig,
    *,
    name: str,
) -> None:
    logger = logging.getLogger(name)

    log_source_config(source_config, logger)
    log_filter_config(kernel_config, logger)

    if target_config is not None:
        log_target_config(target_config, logger, subpath=_subpath(source_config))


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
    subpath = _subpath(source_config)

    kernel = kernel_config.build()

    def build_sequence(source: PhaseFileFolder) -> PhaseFilteredSequence:
        return PhaseFilteredSequence(
            source,
            kernel,
            root=source_config.root,
            subpath=subpath,
            step=source_config.frame_step,
        )

    return [build_sequence(source) for source in sources]


def build_branches(
    source_config: SourceConfig,
    target_config: TargetConfig,
    kernel_config: KernelConfig,
    output_root: StrPath,
    expected: Sequence[str],
) -> list[SideBranch[PhaseFilteredSequence, Tensor, Path]]:
    if not (target_config.save_ranges or target_config.save_frames):
        msg = "nothing to do: set `target.save_ranges` or `target.save_frames`"
        raise ValueError(msg)

    branches = []

    subpath = _subpath(source_config)
    overwrite = target_config.overwrite

    if target_config.save_frames:
        branches.append(FrameTree(output_root, subpath, overwrite=overwrite))

    if target_config.save_ranges:
        path = Path(output_root, target_config.range_file)
        source = source_config.root
        settings = {
            "source": {"subpath": subpath, "frame_step": source_config.frame_step},
            "filter": describe_filter_kernel(kernel_config),
        }

        branches.append(
            RangeDocument(path, source, expected, settings, overwrite=overwrite)
        )

    return branches


def build_phase_stages(
    source_config: SourceConfig,
    target_config: TargetConfig | None = None,
    filter_config: DictConfig | None = None,
    *,
    output_root: StrPath,
    name: str,
) -> PhaseStageFactory:
    kernel_config = parse_filter_config(filter_config)

    log_configs(source_config, target_config, kernel_config, name=name)

    sequences = build_sequences(source_config, kernel_config)
    branches = []

    if target_config is not None:
        branches = build_branches(
            source_config,
            target_config,
            kernel_config,
            output_root,
            [sequence.name for sequence in sequences],
        )

    return PhaseStageFactory(sequences, *branches, name=name)
