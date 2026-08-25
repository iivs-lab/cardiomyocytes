from __future__ import annotations

__all__ = (
    "PreprocessSourceConfig",
    "PreprocessTargetConfig",
    "build_branches",
    "build_preprocess_stages",
    "log_configs",
    "log_target_config",
)

import logging
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, ClassVar

from kaparoo.utils import unwrap_or_factory

from iivs_cardio.common.logging import log_indented
from iivs_cardio.common.pipeline.branch import (
    ensure_json_name,
)
from iivs_cardio.data.pipeline import FrameTree, RangeDocument, SequenceStageFactory
from iivs_cardio.data.transforms.filtering.kernel import IdentityConfig
from scripts._common.dataset import (
    BranchConfig,
    TreeBranchConfig,
    ensure_output_clear,
    log_branch_policies,
    log_source_config,
)
from scripts._common.phase import (
    PhaseSourceConfig,
    build_sequences,
    log_short_sequences,
)
from scripts.data._filtering import describe_filter_kernel, log_filter_config

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from logging import Logger

    from kaparoo.filesystem.types import StrPath
    from torch import Tensor

    from iivs_cardio.common.pipeline import SideBranch
    from iivs_cardio.data.phase import PhaseFilteredSequence
    from iivs_cardio.data.transforms.filtering.kernel import KernelConfig
    from scripts._common.dataset import SequenceSelectConfig


# ========================== #
#          Settings          #
# ========================== #


@dataclass
class PreprocessSourceConfig(PhaseSourceConfig):
    """The tree this stage reads, which is phase as it comes off the microscope.

    Attributes:
        DEFAULT_SUBPATH: As `PhaseSourceConfig`.
        subpath: As `PhaseSourceConfig`.
        root: As `PhaseSourceConfig`.
        frames: As `PhaseSourceConfig`.
    """


@dataclass
class FrameBranchConfig(TreeBranchConfig):
    """The branch that writes each sequence back out as a tree of frames.

    Attributes:
        DEFAULT_SUBPATH: Where the frames go for a branch that names no layout
            and is given nothing to follow.
        save: As `TreeBranchConfig`.
        subpath: As `TreeBranchConfig`.
        record_file: As `TreeBranchConfig`.
        if_present: As `TreeBranchConfig`.
        if_unsourced: As `TreeBranchConfig`.
    """

    DEFAULT_SUBPATH: ClassVar[str] = "frames"


@dataclass
class RangeBranchConfig(BranchConfig):
    """The branch that gathers every sequence's value range into one document.

    Attributes:
        save: Whether to write the document. Defaults to `True`.
        file: The name the document is given, given `.json` if it has no
            extension. Defaults to `"value_range"`.
        if_present: As `BranchConfig`, judged by the settings and the source
            frames' names rather than by what those frames hold.
        if_unsourced: As `BranchConfig`.
    """

    save: bool = True
    file: str = "value_range"


@dataclass
class PreprocessTargetConfig:
    """What a run writes, one block per branch.

    Where they land is not here: `run_root` places the job's directory, and the
    folder a branch actually writes under is the one hydra made for the job.

    Attributes:
        frames: The branch writing the filtered frames.
        ranges: The branch writing the value ranges.
    """

    frames: FrameBranchConfig = field(default_factory=FrameBranchConfig)
    ranges: RangeBranchConfig = field(default_factory=RangeBranchConfig)


def _validate_output(
    source_config: PreprocessSourceConfig,
    target_config: PreprocessTargetConfig,
    output_root: StrPath,
) -> None:
    """Raise unless the target names an output this run can safely write.

    Raises:
        ValueError: If the target writes nothing, or the frames it writes would
            land on the source they are read from.
    """
    if not target_config.frames.save:
        if not target_config.ranges.save:
            msg = "nothing to do: set `target.ranges.save` or `target.frames.save`"
            raise ValueError(msg)
        return

    read = source_config.resolve_subpath()

    ensure_output_clear(
        source_config.root,
        output_root,
        what="frames",
        read=read,
        written=target_config.frames.resolve_subpath(read),
        fix="`target.frames.subpath` beside it, or `run_root` outside the source",
    )


# ========================== #
#          Logging           #
# ========================== #


def _range_file(target_config: PreprocessTargetConfig) -> str:
    """Return what the range document is called, given `.json` if it has none.

    Raises:
        ValueError: If the name is a path or carries some other extension. The
            library's own refusal says what is wrong with the name but not
            which setting holds it, which is where a reader has to go.
    """
    try:
        return ensure_json_name(target_config.ranges.file)
    except ValueError as error:
        msg = f"`target.ranges.file`: {error}"
        raise ValueError(msg) from error


def log_target_config(
    target_config: PreprocessTargetConfig,
    logger: Logger,
    *,
    output_root: StrPath,
    follow: str | None = None,
) -> None:
    """Log what a run writes and where, naming each output it will produce.

    Args:
        target_config: The settings saying what the run was told to write.
        logger: The logger the lines go to.
        output_root: The folder the branches write under, which hydra made for
            this job. A sweep gives each of its jobs one of its own beneath
            `run_root`, so the two are the same path only in a lone run.
        follow: The layout the source reads at, which the frames branch takes
            where it names none of its own. Defaults to `None`, which leaves
            the branch to its own.
    """
    log_indented(logger, "target: %s", PurePath(output_root).as_posix(), depth=0)

    if not (target_config.frames.save or target_config.ranges.save):
        log_indented(logger, "writing nothing")
        return

    if target_config.frames.save:
        written = target_config.frames.resolve_subpath(follow)
        layout = f"<sequence>/{written}" if written else "<sequence>/*"
        log_indented(logger, "writing the filtered frames to %s", layout)
        log_branch_policies("frames", target_config.frames, logger)

    if target_config.ranges.save:
        name = _range_file(target_config)
        log_indented(logger, "writing the value ranges to %s", name)
        log_branch_policies("ranges", target_config.ranges, logger)


def log_configs(
    source_config: PreprocessSourceConfig,
    sequence_config: SequenceSelectConfig,
    kernel_config: KernelConfig,
    target_config: PreprocessTargetConfig | None,
    *,
    output_root: StrPath,
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

    log_source_config(source_config, sequence_config, logger)
    log_filter_config(kernel_config, logger)

    if target_config is not None:
        read = source_config.resolve_subpath()
        log_target_config(target_config, logger, output_root=output_root, follow=read)

        renumbered = (source_config.frames.start, source_config.frames.step) != (0, 1)
        if target_config.frames.save and renumbered:
            fix = "join the value ranges to it by position rather than by name"
            logger.warning("the cache renumbers the frames it keeps: %s", fix)


# ========================== #
#          Building          #
# ========================== #


def build_branches(
    source_config: PreprocessSourceConfig,
    kernel_config: KernelConfig,
    target_config: PreprocessTargetConfig,
    *,
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
        contents: Every sequence the source holds, against the frames each
            would be measured over.
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
                branch.resolve_subpath(subpath),
                contents,
                settings,
                selected=selected,
                record_file=branch.record_file,
                if_present=branch.if_present,
                if_unsourced=branch.if_unsourced,
            )
        )

    if (branch := target_config.ranges).save:
        branches.append(
            RangeDocument(
                Path(output_root, _range_file(target_config)),
                source_config.root,
                contents,
                settings,
                selected=selected,
                if_present=branch.if_present,
                if_unsourced=branch.if_unsourced,
            )
        )

    return branches


def build_preprocess_stages(
    source_config: PreprocessSourceConfig,
    sequence_config: SequenceSelectConfig,
    kernel_config: KernelConfig | None = None,
    target_config: PreprocessTargetConfig | None = None,
    *,
    output_root: StrPath,
    name: str,
) -> SequenceStageFactory:
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
        kernel_config: The filter to apply. Defaults to `None`, which leaves the
            frames as they are.
        output_root: The folder the branches write under.
        name: The name the run is called by.

    Returns:
        The factory a driver runs the sequences through.

    Raises:
        ValueError: If the target writes nothing, if it would write over the
            source, or if the source search finds nothing to run.
    """
    kernel_config = unwrap_or_factory(kernel_config, IdentityConfig)

    log_configs(
        source_config,
        sequence_config,
        kernel_config,
        target_config,
        output_root=output_root,
        name=name,
    )

    if target_config is not None:
        _validate_output(source_config, target_config, output_root)

    sequences, contents = build_sequences(source_config, sequence_config, kernel_config)

    log_short_sequences(
        source_config.frames,
        sequences=sequences,
        contents=contents,
        name=name,
    )

    branches = []

    if target_config is not None:
        selected = [sequence.name for sequence in sequences]

        branches = build_branches(
            source_config,
            kernel_config,
            target_config,
            output_root=output_root,
            contents=contents,
            selected=selected,
        )

    return SequenceStageFactory(sequences, *branches, name=name)
