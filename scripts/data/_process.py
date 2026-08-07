from __future__ import annotations

__all__ = (
    "SourceConfig",
    "TargetConfig",
    "TreeConfig",
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
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, ClassVar, Final

from iivs.dhm.data.koala import PHASE_FLOAT_BIN
from iivs.dhm.data.phase import PhaseFileFolder, PhaseUnit, search_phase_bin_folders
from kaparoo.filesystem import ensure_file_extension, stringify_path
from kaparoo.filesystem.exceptions import UnsupportedExtensionError
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


SELECTION_LIMIT: Final = 5
SELECTION_SPECS: Final = (".json", ".txt")


@dataclass
class TreeConfig:
    """A tree of sequences, and where inside one of them the frames sit.

    What a run reads and what it writes are the same shape, so the pair of
    settings that says where a tree is lives here once. A subclass sets
    `DEFAULT_SUBPATH` to the layout its own end of a stage uses, which is what
    an unset `subpath` comes to when the caller offers nothing to follow.

    Attributes:
        root: the folder the sequences sit under.
        subpath: where a sequence's frames sit inside its own folder, or `None`
            to follow what the caller offers, and this end's own layout when it
            offers nothing.
    """

    DEFAULT_SUBPATH: ClassVar[str]

    root: str = MISSING
    subpath: str | None = None

    def resolve_subpath(self, follow: str | None = None) -> str:
        """Return where the frames sit, settling an unset `subpath`.

        The answer is always a path a sequence's own folder contains, which is
        what lets two of them be compared as they stand: one that could reach
        outside would leave whatever compares them looking at the wrong pair.

        Args:
            follow: what an unset `subpath` takes, such as where the other end
                of the stage keeps its own. `None` leaves it to
                `DEFAULT_SUBPATH`, which is this end's own layout.

        Raises:
            ValueError: If the answer would reach outside a sequence's folder.
        """
        default = unwrap_or_default(follow, self.DEFAULT_SUBPATH)
        subpath = unwrap_or_default(self.subpath, default)

        path = PurePath(subpath)
        if path.anchor or ".." in path.parts:
            msg = f"invalid subpath {subpath!r}: expected a relative path, no '..'"
            raise ValueError(msg)

        return subpath


@dataclass
class SourceConfig(TreeConfig):
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

    DEFAULT_SUBPATH: ClassVar[str] = PHASE_FLOAT_BIN

    include: list[str] | str | None = None
    exclude: list[str] | str | None = None
    frame_step: int = 1


@dataclass
class TargetConfig(TreeConfig):
    """What a run writes, and where.

    Attributes:
        root: where the run's outputs land.
        subpath: where a written sequence keeps its frames inside its own
            folder, or `None` to put them where the source keeps its own.
            Naming one is what lets a run write beside the frames it read
            rather than over them.
        overwrite: whether what is already there may be replaced.
        save_frames: whether to write the filtered frames, laid out like the
            source.
        save_ranges: whether to write the value ranges as one document.
        range_file: what that document is called, given `.json` if it has no
            extension.
    """

    DEFAULT_SUBPATH: ClassVar[str] = PHASE_FLOAT_BIN

    overwrite: bool = False
    save_frames: bool = False
    save_ranges: bool = True
    range_file: str = "value_range"


def _validate_output(
    source_config: SourceConfig, target_config: TargetConfig, output_root: StrPath
) -> None:
    """Raise unless the target names an output this run can safely write.

    A sequence is written by replacing its folder whole, so a destination that
    is a source folder, holds one, or sits inside one is refused, as is one the
    source search would find again. A tree written beside the source under a
    name of its own collides with neither, and is left open.

    Raises:
        ValueError: If the target writes nothing, or the frames it writes would
            land on the source they are read from.
    """
    if not target_config.save_frames:
        if not target_config.save_ranges:
            msg = "nothing to do: set `target.save_ranges` or `target.save_frames`"
            raise ValueError(msg)
        return

    source_root = Path(source_config.root).resolve()
    output_root = Path(output_root).resolve()

    if not output_root.is_relative_to(source_root):
        return

    subpath = PurePath(source_config.resolve_subpath())
    target_subpath = PurePath(target_config.resolve_subpath(subpath.as_posix()))

    if subpath.is_relative_to(target_subpath) or target_subpath.is_relative_to(subpath):
        where = f"{output_root.as_posix()}/*/{target_subpath.as_posix()}"
        fix = "`target.subpath` beside it, or `target.root` outside the source"
        msg = f"frames would land on the source at {where}: set {fix}"
        raise ValueError(msg)


def _range_file(target_config: TargetConfig) -> Path:
    """Return what the range document is called, given `.json` if it has none.

    Raises:
        ValueError: If the name carries some other extension. The library's own
            refusal names the extensions but not the setting that holds one,
            which is the only part a reader has to go and change.
    """
    try:
        return ensure_file_extension(target_config.range_file, DOCUMENT_EXT, add=True)
    except UnsupportedExtensionError as error:
        msg = f"invalid `target.range_file` {target_config.range_file!r}: {error}"
        raise ValueError(msg) from error


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

    log_indented(logger, "reading <sequence>/%s", source_config.resolve_subpath())

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
    target_config: TargetConfig,
    logger: Logger,
    output_root: StrPath,
    *,
    subpath: str | None = None,
) -> None:
    """Log what a run writes and where, naming each output it will produce.

    Args:
        target_config: what the run was told to write.
        logger: where the lines go.
        output_root: where the branches actually write, which is not
            `target.root`: that setting places the job's directory, and a sweep
            gives each of its jobs one of its own beneath it.
        subpath: how a written sequence is laid out, when frames are written.
    """
    log_indented(logger, "target: %s", output_root, depth=0)

    if target_config.save_frames:
        layout = f"<sequence>/{subpath}" if subpath else "<sequence>/*"
        log_indented(logger, "writing the filtered frames to %s", layout)

    if target_config.save_ranges:
        name = _range_file(target_config)
        log_indented(logger, "writing the value ranges to %s", name)

    if not (target_config.save_frames or target_config.save_ranges):
        log_indented(logger, "writing nothing")

    if target_config.overwrite:
        log_indented(logger, "replacing what is already there")


def log_configs(
    source_config: SourceConfig,
    target_config: TargetConfig | None,
    kernel_config: KernelConfig,
    output_root: StrPath,
    *,
    name: str,
) -> None:
    """Log the whole configuration of a run, as one block per part.

    A run that writes nothing has no target to describe, which is what an
    absent `target_config` means.

    One pairing gets a warning of its own: a stride makes the two outputs name
    the same frame differently, since the ranges are filed under the source and
    a cache numbers its own frames from zero. Both are right on their own, so
    the run says which of them a reader should join by.
    """
    logger = logging.getLogger(name)

    log_source_config(source_config, logger)
    log_filter_config(kernel_config, logger)

    if target_config is not None:
        subpath = target_config.resolve_subpath(source_config.resolve_subpath())
        log_target_config(target_config, logger, output_root, subpath=subpath)

        if target_config.save_frames and source_config.frame_step > 1:
            fix = "join the value ranges to it by position rather than by name"
            logger.warning("the cache renumbers the frames it keeps: %s", fix)


def search_sources(config: SourceConfig) -> tuple[list[PhaseFileFolder], int]:
    """Find the sequences a run reads, narrowed by what it was told to take.

    Every sequence taken is checked for a missing frame before any of them is
    run, since a gap is a fault in the dataset rather than in one item of work.
    A gap otherwise opens as an ordinary shorter sequence, and what is written
    back out is numbered without one, so nothing downstream can tell.

    Returns:
        One folder per sequence, each set to give its frames in radians, and
        how many the root held before the selection narrowed them. The second
        is what lets a document say it describes part of a dataset rather than
        the whole of a smaller one.

    Raises:
        ValueError: If the root holds no sequence at all, if the selection
            leaves none of the ones it holds, or if a sequence taken is missing
            a frame. The first two are told apart, since they are fixed
            differently.
    """
    subpath = config.resolve_subpath()

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
        try:
            source.validate_if_supported(level="names")
            taken.append(source.with_unit(PhaseUnit.RADIANS))
        except ValueError as error:
            msg = f"{folder_subpath(source)}: {error}"
            raise ValueError(msg) from error

    return taken, num_folders


def build_sequences(
    source_config: SourceConfig, kernel_config: KernelConfig
) -> tuple[list[PhaseFilteredSequence], int]:
    """Build one filtered view per sequence, all sharing a single kernel.

    A kernel holds only the shape it reads, never frames, so one serves every
    sequence of the run.

    Returns:
        The sequences, in the order the search found them, and how many the
        root held before the selection narrowed them.
    """
    sources, found = search_sources(source_config)
    subpath = source_config.resolve_subpath()

    kernel = kernel_config.build()

    def build_sequence(source: PhaseFileFolder) -> PhaseFilteredSequence:
        return PhaseFilteredSequence(
            source,
            kernel,
            root=source_config.root,
            subpath=subpath,
            step=source_config.frame_step,
        )

    return [build_sequence(source) for source in sources], found


def build_branches(
    source_config: SourceConfig,
    target_config: TargetConfig,
    kernel_config: KernelConfig,
    output_root: StrPath,
    sequence_names: Sequence[str],
    found: int | None = None,
) -> list[SideBranch[PhaseFilteredSequence, Tensor, Path]]:
    """Build the branches a target describes, in the order they will watch.

    Args:
        source_config: what the run reads, recorded in what the branches write.
        target_config: what the run writes.
        kernel_config: the filter, recorded for a later run to compare against.
        output_root: where the branches write.
        sequence_names: every sequence the run set out to cover.
        found: how many the source held before the selection narrowed it, or
            `None` when nothing narrowed it.

    Returns:
        The branches, empty of neither output when the target asks for both.

    Raises:
        ValueError: If the target writes nothing, which is a mistake rather
            than a way to ask for a run that only reads, or if the frames it
            writes would land on the source they are read from.
    """
    _validate_output(source_config, target_config, output_root)

    branches = []

    subpath = source_config.resolve_subpath()
    overwrite = target_config.overwrite

    if target_config.save_frames:
        written_at = target_config.resolve_subpath(subpath)
        branches.append(FrameTree(output_root, written_at, overwrite=overwrite))

    if target_config.save_ranges:
        path = Path(output_root, target_config.range_file)
        source = source_config.root
        settings = {
            "source": {"subpath": subpath, "frame_step": source_config.frame_step},
            "filter": describe_filter_kernel(kernel_config),
        }

        branches.append(
            RangeDocument(
                path,
                source,
                sequence_names,
                settings,
                found=found,
                overwrite=overwrite,
            )
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
    nothing, or that would write over the source, is refused at the same point,
    before the search costs anything.

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
        ValueError: If the target writes nothing, if it would write over the
            source, or if the source search finds nothing to run.
    """
    kernel_config = parse_filter_config(filter_config)

    log_configs(source_config, target_config, kernel_config, output_root, name=name)

    if target_config is not None:
        _validate_output(source_config, target_config, output_root)

    sequences, found = build_sequences(source_config, kernel_config)
    branches = []

    if target_config is not None:
        branches = build_branches(
            source_config,
            target_config,
            kernel_config,
            output_root,
            [sequence.name for sequence in sequences],
            found,
        )

    return PhaseStageFactory(sequences, *branches, name=name)
