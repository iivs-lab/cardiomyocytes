from __future__ import annotations

__all__ = (
    "TargetConfig",
    "build_branches",
    "build_preprocess_stages",
    "log_configs",
    "log_target_config",
    "search_sources",
)

import logging
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import TYPE_CHECKING

from kaparoo.filesystem import UnsupportedExtensionError, ensure_file_extension
from kaparoo.utils import quantify
from omegaconf import MISSING

from iivs_cardio.common.logging import log_indented
from iivs_cardio.common.pipeline import (
    EXISTING_OUTPUT_POLICIES,
    UNSOURCED_OUTPUT_POLICIES,
    ensure_policy,
)
from iivs_cardio.data.pipeline import (
    DOCUMENT_EXT,
    FrameTree,
    PhaseStageFactory,
    RangeDocument,
)
from scripts._common.dataset import SourceConfig, log_source_config, resolve_subpath
from scripts._common.phase import DEFAULT_SUBPATH, build_sequences, search_sources
from scripts.data._filtering import (
    describe_filter_kernel,
    log_filter_config,
    parse_filter_config,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from logging import Logger

    from kaparoo.filesystem.types import StrPath
    from omegaconf import DictConfig
    from torch import Tensor

    from iivs_cardio.common.pipeline import SideBranch
    from iivs_cardio.common.pipeline.branch import (
        ExistingOutputPolicy,
        UnsourcedOutputPolicy,
    )
    from iivs_cardio.data.phase import PhaseFilteredSequence
    from iivs_cardio.data.transforms.filtering.kernel import KernelConfig
    from scripts._common.dataset import FrameSelectConfig, SequenceSelectConfig


# ========================== #
#          Settings          #
# ========================== #


@dataclass
class BranchConfig:
    """What one side branch writes, and what it does where it finds an output.

    Every branch answers the same three questions, so a stage adding one adds a
    block of this shape rather than another three keys beside the others.

    Attributes:
        save: Whether to write this output at all. Defaults to `False`.
        if_exists: The policy for a sequence this output already covers.
            `"reuse"` keeps what an earlier run left that still describes this
            one, and writes the rest. Defaults to `"error"`.
        if_unsourced: The policy for part of this output whose sequence the
            source no longer holds. Defaults to `"keep"`: the same absence is
            what a half mounted share looks like, and what is kept is always
            said out loud.
    """

    save: bool = False
    if_exists: str = "error"
    if_unsourced: str = "keep"


@dataclass
class FrameBranchConfig(BranchConfig):
    """The branch that writes each sequence back out as a tree of frames.

    Attributes:
        save: Whether to write the filtered frames. Defaults to `False`.
        subpath: The path a written sequence keeps its frames at inside its own
            folder. Naming one is what lets a run write beside the frames it
            read rather than over them. Defaults to `None`, which puts them
            where the source keeps its own.
        record_file: The name of the file each written folder keeps its own
            account in, given `.json` if it has no extension. A later run reads
            it to decide whether what is there still describes this run.
            Defaults to `"source"`.
        if_exists: As `BranchConfig`, judged by the settings and the source
            frames' names rather than by what those frames hold. A source
            re-exported under the same names is kept rather than written again,
            so a run that follows one takes `"overwrite"`.
        if_unsourced: As `BranchConfig`.
    """

    subpath: str | None = None
    record_file: str = "source"


@dataclass
class RangeBranchConfig(BranchConfig):
    """The branch that gathers every sequence's value range into one document.

    Attributes:
        save: Whether to write the document. Defaults to `True`.
        file: The name the document is given, given `.json` if it has no
            extension. Defaults to `"value_range"`.
        if_exists: As `BranchConfig`, judged by the settings and the source
            frames' names rather than by what those frames hold.
        if_unsourced: As `BranchConfig`.
    """

    save: bool = True
    file: str = "value_range"


@dataclass
class TargetConfig:
    """Where a run's outputs land, and which of them it writes.

    Not a tree of its own: the frames go out as one and the ranges as a single
    file, so what the two share is the root they sit under and nothing else.

    Attributes:
        root: The folder the run's outputs land under.
        frames: The branch writing the filtered frames.
        ranges: The branch writing the value ranges.
    """

    root: str = MISSING
    frames: FrameBranchConfig = field(default_factory=FrameBranchConfig)
    ranges: RangeBranchConfig = field(default_factory=RangeBranchConfig)


def _existing(branch: BranchConfig, output: str) -> ExistingOutputPolicy:
    """Read a branch's `if_exists`, naming the key a reader has to go and change."""
    return ensure_policy(
        branch.if_exists, EXISTING_OUTPUT_POLICIES, f"target.{output}.if_exists"
    )


def _unsourced(branch: BranchConfig, output: str) -> UnsourcedOutputPolicy:
    """Read a branch's `if_unsourced`, naming the key a reader has to change."""
    return ensure_policy(
        branch.if_unsourced, UNSOURCED_OUTPUT_POLICIES, f"target.{output}.if_unsourced"
    )


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
    if not target_config.frames.save:
        if not target_config.ranges.save:
            msg = "nothing to do: set `target.ranges.save` or `target.frames.save`"
            raise ValueError(msg)
        return

    source_root = Path(source_config.root).resolve()
    output_root = Path(output_root).resolve()

    if not output_root.is_relative_to(source_root):
        return

    subpath = PurePath(resolve_subpath(source_config.subpath, default=DEFAULT_SUBPATH))
    written = PurePath(
        resolve_subpath(
            target_config.frames.subpath, subpath.as_posix(), default=DEFAULT_SUBPATH
        )
    )

    if subpath.is_relative_to(written) or written.is_relative_to(subpath):
        where = f"{output_root.as_posix()}/*/{written.as_posix()}"
        fix = "`target.frames.subpath` beside it, or `target.root` outside the source"
        msg = f"frames would land on the source at {where}: set {fix}"
        raise ValueError(msg)


# ========================== #
#          Logging           #
# ========================== #


def _range_file(target_config: TargetConfig) -> Path:
    """Return what the range document is called, given `.json` if it has none.

    Raises:
        ValueError: If the name carries some other extension. The library's own
            refusal names the extensions but not the setting that holds one,
            which is the only part a reader has to go and change.
    """
    try:
        return ensure_file_extension(target_config.ranges.file, DOCUMENT_EXT, add=True)
    except UnsupportedExtensionError as error:
        named = target_config.ranges.file
        msg = f"invalid `target.ranges.file` {named!r}: {error}"
        raise ValueError(msg) from error


def _log_branch(output: str, branch: BranchConfig, logger: Logger) -> None:
    """Say what a branch does with what it finds, unless it refuses to run.

    Set in under the line naming the output it belongs to, since the target
    writes two and a policy at the same depth as both would read as either.
    """
    said = {
        "overwrite": f"overwriting the {output} it finds",
        "reuse": f"reusing the {output} that match this run",
    }
    if (line := said.get(branch.if_exists)) is not None:
        log_indented(logger, "%s", line, depth=2)

    if branch.if_unsourced != "keep":
        log_indented(logger, "dropping the %s a source no longer has", output, depth=2)


def log_target_config(
    target_config: TargetConfig,
    output_root: StrPath,
    logger: Logger,
    *,
    subpath: str | None = None,
) -> None:
    """Log what a run writes and where, naming each output it will produce.

    Args:
        target_config: The settings saying what the run was told to write.
        output_root: The folder the branches actually write under, which is not
            `target.root`: that setting places the job's directory, and a sweep
            gives each of its jobs one of its own beneath it.
        logger: The logger the lines go to.
        subpath: The layout a written sequence is given, when frames are
            written. Defaults to `None`, which leaves the layout unnamed.
    """
    log_indented(logger, "target: %s", output_root, depth=0)

    if not (target_config.frames.save or target_config.ranges.save):
        log_indented(logger, "writing nothing")
        return

    if target_config.frames.save:
        layout = f"<sequence>/{subpath}" if subpath else "<sequence>/*"
        log_indented(logger, "writing the filtered frames to %s", layout)
        _log_branch("frames", target_config.frames, logger)

    if target_config.ranges.save:
        name = _range_file(target_config)
        log_indented(logger, "writing the value ranges to %s", name)
        _log_branch("ranges", target_config.ranges, logger)


def log_configs(
    source_config: SourceConfig,
    sequence_config: SequenceSelectConfig,
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
    read = resolve_subpath(source_config.subpath, default=DEFAULT_SUBPATH)

    log_source_config(source_config, sequence_config, logger, subpath=read)
    log_filter_config(kernel_config, logger)

    if target_config is not None:
        subpath = resolve_subpath(
            target_config.frames.subpath, read, default=DEFAULT_SUBPATH
        )
        log_target_config(target_config, output_root, logger, subpath=subpath)

        renumbered = (source_config.frames.start, source_config.frames.step) != (0, 1)
        if target_config.frames.save and renumbered:
            fix = "join the value ranges to it by position rather than by name"
            logger.warning("the cache renumbers the frames it keeps: %s", fix)


def _log_short_sequences(
    frame_config: FrameSelectConfig,
    taken: Sequence[PhaseFilteredSequence],
    contents: Mapping[str, Sequence[str]],
    *,
    name: str,
) -> None:
    """Name the sequences that could not supply the count that was asked for.

    Said after the search rather than with the rest of the configuration,
    since it is what the dataset turned out to hold and not what the run was
    told to do. `"error"` never reaches here: the search refuses there.
    """
    count = frame_config.count
    if count is None:
        return

    short = []
    for sequence in taken:
        held = len(contents[sequence.name])
        if held < count:
            short.append(f"{sequence.name} ({held})")

    if not short:
        return

    logger = logging.getLogger(name)
    listed = ", ".join(short)

    sequences = quantify(len(short), "sequence")
    logger.warning("%s gave fewer than %d: %s", sequences, count, listed)


# ========================== #
#          Building          #
# ========================== #


def build_branches(
    source_config: SourceConfig,
    target_config: TargetConfig,
    kernel_config: KernelConfig,
    output_root: StrPath,
    contents: Mapping[str, Sequence[str]],
    selected: Sequence[str] | None = None,
) -> list[SideBranch[PhaseFilteredSequence, Tensor, Path]]:
    """Build the branches a target describes, in the order they will watch.

    Which sequences a run took is not recorded, since it changes what the run
    covers rather than what any sequence's numbers mean, and `coverage` reports
    it already. Recording it would refuse reuse to a run that narrowed itself,
    and the signature is what keeps it out: no selection reaches here.

    Args:
        source_config: The tree the run reads, recorded in what the branches
            write.
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

    subpath = resolve_subpath(source_config.subpath, default=DEFAULT_SUBPATH)
    frames = source_config.frames

    settings = {
        "source": {
            "subpath": subpath,
            "frames": {
                "start": frames.start,
                "step": frames.step,
                "count": frames.count,
            },
        },
        "filter": describe_filter_kernel(kernel_config),
    }

    if (branch := target_config.frames).save:
        branches.append(
            FrameTree(
                output_root,
                resolve_subpath(branch.subpath, subpath, default=DEFAULT_SUBPATH),
                contents,
                settings,
                selected=selected,
                if_frames_exist=_existing(branch, "frames"),
                if_sources_gone=_unsourced(branch, "frames"),
            )
        )

    if (branch := target_config.ranges).save:
        branches.append(
            RangeDocument(
                Path(output_root, branch.file),
                source_config.root,
                contents,
                settings,
                selected=selected,
                if_ranges_exist=_existing(branch, "ranges"),
                if_sources_gone=_unsourced(branch, "ranges"),
            )
        )

    return branches


def build_preprocess_stages(
    source_config: SourceConfig,
    sequence_config: SequenceSelectConfig,
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
        source_config: The tree the sequences are read from.
        sequence_config: Which of its sequences to read.
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

    log_configs(
        source_config,
        sequence_config,
        target_config,
        kernel_config,
        output_root,
        name=name,
    )

    if target_config is not None:
        _validate_output(source_config, target_config, output_root)

    sequences, contents = build_sequences(source_config, sequence_config, kernel_config)
    _log_short_sequences(source_config.frames, sequences, contents, name=name)

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
