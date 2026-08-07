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
from typing import TYPE_CHECKING, Final

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
    """Which sequences a run reads, and how much of each.

    Attributes:
        root: the dataset folder to search for sequences.
        subpath: where a sequence's frames sit inside its time lapse, or `None`
            for the usual one.
        include: the sequences to take, as names or as a path to a file listing
            them; `None` takes all of them.
        exclude: the same, for sequences to leave out.
        frame_step: take every `frame_step`th frame of each sequence.
    """

    root: str = MISSING
    subpath: str | None = None
    include: list[str] | str | None = None
    exclude: list[str] | str | None = None
    frame_step: int = 1


@dataclass
class TargetConfig:
    """What a run writes, and where.

    Attributes:
        root: where the run's outputs land.
        overwrite: whether what is already there may be replaced.
        save_frames: whether to write the filtered frames, laid out like the
            source.
        save_ranges: whether to write the value ranges as one document.
        range_file: what that document is called, given `.json` if it has no
            extension.
    """

    root: str = MISSING
    overwrite: bool = False
    save_frames: bool = False
    save_ranges: bool = True
    range_file: str = "value_range"


SELECTION_LIMIT: Final = 5
SELECTION_SPECS: Final = (".json", ".txt")


def _subpath(source_config: SourceConfig) -> str:
    """Which folder of a time lapse a run reads, defaulted in one place."""
    return unwrap_or_default(source_config.subpath, PHASE_FLOAT_BIN)


def _validate_source(source: PhaseFileFolder, name: str) -> None:
    """Raise unless `source` is the unbroken run of frames it is read as.

    A gap otherwise opens as an ordinary shorter sequence: the filter joins the
    frames on either side as though they were neighbours, and what is written
    back out is numbered from zero without one, so nothing downstream can tell
    there was a gap at all.

    Args:
        source: the folder to check.
        name: what the sequence is called, so a run over a dataset says which
            of its sequences is the broken one.

    Raises:
        ValueError: If the frame files are not numbered from zero without a gap.
    """
    try:
        source.validate_if_supported(level="names")
    except ValueError as error:
        msg = f"{name}: {error}"
        raise ValueError(msg) from error


def _validate_target_config(target_config: TargetConfig) -> None:
    """Raise if `target_config` names no side branch to write through."""
    if not (target_config.save_ranges or target_config.save_frames):
        msg = "nothing to do: set `target.save_ranges` or `target.save_frames`"
        raise ValueError(msg)


def _log_selection(logger: Logger, verb: str, value: list[str] | str) -> None:
    """Log a selection, listing it only while a list is short enough to read."""
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
    """Log what a run reads, naming only the settings that were moved."""
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
    """Log the filter a run applies, with the settings that shape it."""
    described = describe_filter_kernel(kernel_config)
    kind = described.pop("kind")
    settings = ", ".join(f"{key}={value}" for key, value in described.items())
    settings = f" ({settings})" if settings else ""
    log_indented(logger, "filter: %s kernel%s", kind, settings, depth=0)


def log_target_config(
    target_config: TargetConfig, logger: Logger, *, subpath: str | None = None
) -> None:
    """Log what a run writes and where, naming each output it will produce.

    Args:
        target_config: what the run was told to write.
        logger: where the lines go.
        subpath: how a written sequence is laid out, when frames are written.
    """
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
    """Log the whole configuration of a run, as one block per part.

    A run that writes nothing has no target to describe, which is what an
    absent `target_config` means.
    """
    logger = logging.getLogger(name)

    log_source_config(source_config, logger)
    log_filter_config(kernel_config, logger)

    if target_config is not None:
        log_target_config(target_config, logger, subpath=_subpath(source_config))


def search_sources(config: SourceConfig) -> list[PhaseFileFolder]:
    """Find the sequences a run reads, narrowed by what it was told to take.

    Every sequence taken is checked for a missing frame before any of them is
    run, since a gap is a fault in the dataset rather than in one item of work.

    Returns:
        One folder per sequence, each set to give its frames in radians.

    Raises:
        ValueError: If the root holds no sequence at all, if the selection
            leaves none of the ones it holds, or if a sequence taken is missing
            a frame. The first two are told apart, since they are fixed
            differently.
    """
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

    taken = []
    for source in sources:
        _validate_source(source, folder_subpath(source))
        taken.append(source.with_unit(PhaseUnit.RADIANS))

    return taken


def build_sequences(
    source_config: SourceConfig, kernel_config: KernelConfig
) -> list[PhaseFilteredSequence]:
    """Build one filtered view per sequence, all sharing a single kernel.

    A kernel holds only the shape it reads, never frames, so one serves every
    sequence of the run.

    Returns:
        The sequences, in the order the search found them.
    """
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
    sequence_names: Sequence[str],
) -> list[SideBranch[PhaseFilteredSequence, Tensor, Path]]:
    """Build the branches a target describes, in the order they will watch.

    Args:
        source_config: what the run reads, recorded in what the branches write.
        target_config: what the run writes.
        kernel_config: the filter, recorded for a later run to compare against.
        output_root: where the branches write.
        sequence_names: every sequence the run set out to cover.

    Returns:
        The branches, empty of neither output when the target asks for both.

    Raises:
        ValueError: If the target writes nothing, which is a mistake rather
            than a way to ask for a run that only reads.
    """
    _validate_target_config(target_config)

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
            RangeDocument(path, source, sequence_names, settings, overwrite=overwrite)
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
    """Assemble everything a run needs from the configuration it was given.

    The configuration is logged before the sources are searched, so a run says
    what it was asked to do even when it cannot do it. A target that writes
    nothing is refused at the same point, before the search costs anything.

    Args:
        source_config: which sequences to read, and how much of each.
        target_config: what to write, or `None` for a run that only reads.
        filter_config: the filter to apply, or `None` to leave frames as they
            are.
        output_root: where the branches write.
        name: what the run is called.

    Returns:
        The factory a driver runs the sequences through.

    Raises:
        ValueError: If the target writes nothing, or the source search finds
            nothing to run.
    """
    kernel_config = parse_filter_config(filter_config)

    log_configs(source_config, target_config, kernel_config, name=name)

    if target_config is not None:
        _validate_target_config(target_config)

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
