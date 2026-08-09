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
from kaparoo.filesystem import (
    UnsupportedExtensionError,
    dir_exists,
    ensure_file_extension,
    select,
    stringify_path,
)
from kaparoo.utils import unwrap_or_default
from omegaconf import MISSING

from iivs_cardio.common.logging import log_indented
from iivs_cardio.common.pipeline import (
    EXISTING_OUTPUT_POLICIES,
    UNSOURCED_OUTPUT_POLICIES,
    read_policy,
)
from iivs_cardio.data.phase import PhaseFilteredSequence
from iivs_cardio.data.pipeline import (
    DOCUMENT_EXT,
    FRAME_POLICIES,
    FrameTree,
    PhaseStageFactory,
    RangeDocument,
)
from scripts.data._filtering import describe_filter_kernel, parse_filter_config

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from logging import Logger

    from kaparoo.filesystem.types import StrPath
    from omegaconf import DictConfig
    from torch import Tensor

    from iivs_cardio.common.pipeline import SideBranch
    from iivs_cardio.data.transforms.filtering.kernel import KernelConfig


# ========================== #
#          Settings          #
# ========================== #


@dataclass
class TreeConfig:
    """A tree of sequences, and where inside one of them the frames sit.

    What a run reads and what it writes are the same shape, so the pair of
    settings that says where a tree is lives here once. A subclass sets
    `DEFAULT_SUBPATH` to the layout its own end of a stage uses, which is what
    an unset `subpath` comes to when the caller offers nothing to follow.

    Attributes:
        root: The folder the sequences sit under.
        subpath: The path to a sequence's frames inside its own folder. Defaults
            to `None`, which follows what the caller offers and falls back to
            this end's own layout when it offers nothing.
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
            follow: The layout an unset `subpath` takes, such as the one the
                other end of the stage keeps its frames in. Defaults to `None`,
                which leaves it to `DEFAULT_SUBPATH`.

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
        root: The dataset folder to search for sequences.
        subpath: The path to a sequence's frames inside its time lapse. Defaults
            to `None`, which takes the usual one.
        include: The sequences to take, as names or as a path to a file listing
            them. Defaults to `None`, which takes all of them.
        exclude: The same, for sequences to leave out. Defaults to `None`.
        frame_step: The stride to read each sequence at, so that every
            `frame_step`th frame is taken. Defaults to 1.
    """

    DEFAULT_SUBPATH: ClassVar[str] = PHASE_FLOAT_BIN

    include: list[str] | str | None = None
    exclude: list[str] | str | None = None
    frame_step: int = 1


@dataclass
class TargetConfig(TreeConfig):
    """What a run writes, and where.

    Attributes:
        root: The folder the run's outputs land under.
        subpath: The path a written sequence keeps its frames at inside its own
            folder. Naming one is what lets a run write beside the frames it
            read rather than over them. Defaults to `None`, which puts them
            where the source keeps its own.
        save_frames: Whether to write the filtered frames, laid out like the
            source. Defaults to `False`.
        save_ranges: Whether to write the value ranges as one document. Defaults
            to `True`.
        range_file: The name that document is given, given `.json` if it has no
            extension. Defaults to `"value_range"`.
        if_frames_exist: The policy for a sequence that already has a frame
            folder. `"reuse"` keeps the ones an earlier run left whose record
            still describes this one, and writes the rest. Defaults to
            `"error"`.
        if_ranges_exist: The policy for a sequence that already has a range
            part. `"reuse"` keeps the ones an earlier run left that still
            describe this one, and measures the rest. Defaults to `"error"`.
        if_sources_gone: The policy for an output whose sequence the source no
            longer holds. Defaults to `"keep"`: the same absence is what a half
            mounted share looks like, and what is kept is always said out loud.
    """

    DEFAULT_SUBPATH: ClassVar[str] = PHASE_FLOAT_BIN

    save_frames: bool = False
    save_ranges: bool = True
    range_file: str = "value_range"
    if_frames_exist: str = "error"
    if_ranges_exist: str = "error"
    if_sources_gone: str = "keep"


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


# ========================== #
#          Logging           #
# ========================== #


SELECTION_LIMIT: Final = 5
SELECTION_SPECS: Final = (".json", ".txt")


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
        target_config: The settings saying what the run was told to write.
        logger: The logger the lines go to.
        output_root: The folder the branches actually write under, which is not
            `target.root`: that setting places the job's directory, and a sweep
            gives each of its jobs one of its own beneath it.
        subpath: The layout a written sequence is given, when frames are
            written. Defaults to `None`, which leaves the layout unnamed.
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

    if target_config.save_frames:
        _log_policy(logger, "frames", target_config.if_frames_exist)

    if target_config.save_ranges:
        _log_policy(logger, "ranges", target_config.if_ranges_exist)

    if target_config.if_sources_gone != "keep":
        log_indented(logger, "dropping outputs whose sequence the source has lost")


def _log_policy(logger: Logger, what: str, policy: str) -> None:
    """Say what a run does where that output is already there, unless it refuses."""
    said = {
        "overwrite": f"replacing the {what} already there",
        "reuse": f"keeping the {what} already there that still describe this run",
    }
    if (line := said.get(policy)) is not None:
        log_indented(logger, "%s", line)


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


# ========================== #
#          Building          #
# ========================== #


def search_sources(
    config: SourceConfig,
) -> tuple[list[PhaseFileFolder], dict[str, tuple[str, ...]]]:
    """Find the sequences a run reads, narrowed by what it was told to take.

    Every sequence taken is checked for a missing frame before any of them is
    run, since a gap is a fault in the dataset rather than in one item of work.
    A gap otherwise opens as an ordinary shorter sequence, and what is written
    back out is numbered without one, so nothing downstream can tell.

    Nothing inside a time-lapse is descended into. Opening one lists its frames
    already, and the walk has no reason to list them a second time looking for
    a time-lapse that cannot be nested there.

    Returns:
        One folder per sequence taken, each set to give its frames in radians,
        and a contents of every sequence the root holds against the frames the
        run would measure it over. The contents covers what the selection left
        out too, which is what lets a document say it describes part of a
        dataset rather than the whole of a smaller one, and what an output with
        no sequence behind it is measured against.

    Raises:
        ValueError: If the root holds no sequence at all, if the selection
            leaves none of the ones it holds, or if a sequence taken is missing
            a frame. The first two are told apart, since they are fixed
            differently.
    """
    subpath = config.resolve_subpath()

    folders = search_phase_bin_folders(
        config.root,
        subpath=subpath,
        exclude=lambda folder: dir_exists(folder.parent / subpath),
    )

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

    step = config.frame_step
    contents = {
        folder_subpath(folder): tuple(file.name for file in folder.files[::step])
        for folder in folders
    }

    return taken, contents


def build_sequences(
    source_config: SourceConfig, kernel_config: KernelConfig
) -> tuple[list[PhaseFilteredSequence], dict[str, tuple[str, ...]]]:
    """Build one filtered view per sequence, all sharing a single kernel.

    A kernel holds only the shape it reads, never frames, so one serves every
    sequence of the run.

    Returns:
        The sequences, in the order the search found them, and the contents of
        the whole dataset they were selected from.
    """
    sources, contents = search_sources(source_config)
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

    return [build_sequence(source) for source in sources], contents


def build_branches(
    source_config: SourceConfig,
    target_config: TargetConfig,
    kernel_config: KernelConfig,
    output_root: StrPath,
    contents: Mapping[str, Sequence[str]],
    selected: Sequence[str] | None = None,
) -> list[SideBranch[PhaseFilteredSequence, Tensor, Path]]:
    """Build the branches a target describes, in the order they will watch.

    Args:
        source_config: The settings the run reads by, recorded in what the
            branches write.
        target_config: The settings saying what the run writes.
        kernel_config: The filter, recorded for a later run to compare against.
        output_root: The folder the branches write under.
        contents: Every sequence the source holds, against the frames each would
            be measured over.
        selected: The sequences of those this run was given. Defaults to `None`,
            which takes all of them.

    Returns:
        The branches, empty of neither output when the target asks for both.

    Raises:
        ValueError: If the target writes nothing, which is a mistake rather
            than a way to ask for a run that only reads, if the frames it
            writes would land on the source they are read from, or if a policy
            names something no branch offers.
    """
    _validate_output(source_config, target_config, output_root)

    branches = []

    subpath = source_config.resolve_subpath()
    source_policy = read_policy(
        target_config.if_sources_gone, UNSOURCED_OUTPUT_POLICIES, "if_sources_gone"
    )

    settings = {
        "source": {"subpath": subpath, "frame_step": source_config.frame_step},
        "filter": describe_filter_kernel(kernel_config),
    }

    if target_config.save_frames:
        target_subpath = target_config.resolve_subpath(subpath)
        frame_policy = read_policy(
            target_config.if_frames_exist, FRAME_POLICIES, "if_frames_exist"
        )
        branches.append(
            FrameTree(
                output_root,
                target_subpath,
                contents,
                settings,
                selected=selected,
                if_frames_exist=frame_policy,
                if_sources_gone=source_policy,
            )
        )

    if target_config.save_ranges:
        path = Path(output_root, target_config.range_file)
        source = source_config.root
        range_policy = read_policy(
            target_config.if_ranges_exist, EXISTING_OUTPUT_POLICIES, "if_ranges_exist"
        )

        branches.append(
            RangeDocument(
                path,
                source,
                contents,
                settings,
                selected=selected,
                if_ranges_exist=range_policy,
                if_sources_gone=source_policy,
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
        source_config: The settings saying which sequences to read, and how
            much of each.
        target_config: The settings saying what to write. Defaults to `None`,
            for a run that only reads.
        filter_config: The filter to apply. Defaults to `None`, which leaves the
            frames as they are.
        output_root: The folder the branches write under.
        name: The name the run is called by.

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

    sequences, contents = build_sequences(source_config, kernel_config)
    branches = []

    if target_config is not None:
        branches = build_branches(
            source_config,
            target_config,
            kernel_config,
            output_root,
            contents,
            [sequence.name for sequence in sequences],
        )

    return PhaseStageFactory(sequences, *branches, name=name)
