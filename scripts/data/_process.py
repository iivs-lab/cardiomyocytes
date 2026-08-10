from __future__ import annotations

__all__ = (
    "TargetConfig",
    "build_branches",
    "build_phase_stages",
    "log_configs",
    "log_filter_config",
    "log_target_config",
    "search_sources",
)

import logging
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import TYPE_CHECKING

from kaparoo.filesystem import (
    UnsupportedExtensionError,
    ensure_file_extension,
)
from kaparoo.utils import quantify

from iivs_cardio.common.logging import log_indented
from iivs_cardio.common.pipeline import (
    EXISTING_OUTPUT_POLICIES,
    UNSOURCED_OUTPUT_POLICIES,
    ensure_policy,
)
from iivs_cardio.data.pipeline import (
    DOCUMENT_EXT,
    FRAME_POLICIES,
    FrameTree,
    PhaseStageFactory,
    RangeDocument,
)
from scripts._phase import build_sequences, search_sources
from scripts._trees import TreeConfig, log_source_config
from scripts.data._filtering import describe_filter_kernel, parse_filter_config

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from logging import Logger

    from kaparoo.filesystem.types import StrPath
    from omegaconf import DictConfig
    from torch import Tensor

    from iivs_cardio.common.pipeline import SideBranch
    from iivs_cardio.data.phase import PhaseFilteredSequence
    from iivs_cardio.data.transforms.filtering.kernel import KernelConfig
    from scripts._trees import SelectConfig


# ========================== #
#          Settings          #
# ========================== #


@dataclass
class TargetConfig(TreeConfig):
    """What a run writes, and where.

    Both `"reuse"` policies judge by the settings and by the source frames'
    names, never by what those frames hold. A source re-exported under the same
    names is kept rather than written again, so a run that follows one takes
    `"overwrite"` instead.

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

    save_frames: bool = False
    save_ranges: bool = True
    range_file: str = "value_range"
    if_frames_exist: str = "error"
    if_ranges_exist: str = "error"
    if_sources_gone: str = "keep"


def _validate_output(
    source_config: TreeConfig, target_config: TargetConfig, output_root: StrPath
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
    source_config: TreeConfig,
    select_config: SelectConfig,
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

    log_source_config(source_config, select_config, logger)
    log_filter_config(kernel_config, logger)

    if target_config is not None:
        subpath = target_config.resolve_subpath(source_config.resolve_subpath())
        log_target_config(target_config, logger, output_root, subpath=subpath)

        renumbered = (select_config.frame_start, select_config.frame_step) != (0, 1)
        if target_config.save_frames and renumbered:
            fix = "join the value ranges to it by position rather than by name"
            logger.warning("the cache renumbers the frames it keeps: %s", fix)


# ========================== #
#          Building          #
# ========================== #


def _log_short(
    select_config: SelectConfig,
    sequences: Sequence[PhaseFilteredSequence],
    contents: Mapping[str, Sequence[str]],
    *,
    name: str,
) -> None:
    """Name the sequences that could not supply the count that was asked for.

    Said after the search rather than with the rest of the configuration,
    since it is what the dataset turned out to hold and not what the run was
    told to do. `"error"` never reaches here: the search refuses there.
    """
    count = select_config.frame_count
    if count is None:
        return

    short = []
    for sequence in sequences:
        held = len(contents[sequence.name])
        if held < count:
            short.append(f"{sequence.name} ({held})")

    if not short:
        return

    logger = logging.getLogger(name)
    listed = ", ".join(short)

    label = quantify(len(short), "sequence")
    logger.warning("%s gave fewer than %d: %s", label, count, listed)


def build_branches(
    source_config: TreeConfig,
    select_config: SelectConfig,
    target_config: TargetConfig,
    kernel_config: KernelConfig,
    output_root: StrPath,
    contents: Mapping[str, Sequence[str]],
    selected: Sequence[str] | None = None,
) -> list[SideBranch[PhaseFilteredSequence, Tensor, Path]]:
    """Build the branches a target describes, in the order they will watch.

    Args:
        source_config: The tree the run reads, recorded in what the branches
            write.
        select_config: The sequences and frames it takes from that tree,
            recorded alongside it.
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
    source_policy = ensure_policy(
        target_config.if_sources_gone, UNSOURCED_OUTPUT_POLICIES, "if_sources_gone"
    )

    settings = {
        "source": {
            "subpath": subpath,
            "frame_start": select_config.frame_start,
            "frame_step": select_config.frame_step,
            "frame_count": select_config.frame_count,
        },
        "filter": describe_filter_kernel(kernel_config),
    }

    if target_config.save_frames:
        target_subpath = target_config.resolve_subpath(subpath)
        frame_policy = ensure_policy(
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
        range_policy = ensure_policy(
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
    source_config: TreeConfig,
    select_config: SelectConfig,
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
        select_config: Which of them to read, and how much of each.
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
        select_config,
        target_config,
        kernel_config,
        output_root,
        name=name,
    )

    if target_config is not None:
        _validate_output(source_config, target_config, output_root)

    sequences, contents = build_sequences(source_config, select_config, kernel_config)
    _log_short(select_config, sequences, contents, name=name)

    branches = []

    if target_config is not None:
        branches = build_branches(
            source_config,
            select_config,
            target_config,
            kernel_config,
            output_root,
            contents,
            [sequence.name for sequence in sequences],
        )

    return PhaseStageFactory(sequences, *branches, name=name)
