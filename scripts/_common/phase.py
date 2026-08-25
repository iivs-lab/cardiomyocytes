from __future__ import annotations

import logging

__all__ = (
    "LAST_SEARCH",
    "PhaseSourceConfig",
    "SearchResult",
    "build_sequences",
    "log_short_sequences",
    "search_sources",
)

from dataclasses import astuple, dataclass, fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from iivs.dhm.data.koala import PHASE_FLOAT_BIN
from iivs.dhm.data.phase import PhaseFileFolder, PhaseUnit, search_phase_bin_folders
from kaparoo.filesystem import contains, is_spec_file, select, stringify_path
from kaparoo.utils import quantify

from iivs_cardio.common.pipeline import Named, ensure_policy
from iivs_cardio.data.phase import PhaseFilteredSequence
from scripts._common.dataset import (
    LISTING_LIMIT,
    SHORT_SEQUENCE_POLICIES,
    SourceConfig,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from iivs_cardio.data.transforms.filtering.kernel import KernelConfig
    from scripts._common.dataset import FrameSelectConfig, SequenceSelectConfig


# One folder per sequence taken, against the contents of the dataset they came from.
type SearchResult = tuple[list[PhaseFileFolder], dict[str, tuple[str, ...]]]

# The newest search, held for the next job of a sweep to take.
LAST_SEARCH: dict[tuple[object, ...], SearchResult] = {}


@dataclass
class PhaseSourceConfig(SourceConfig):
    """A tree of phase sequences, laid out the way an acquisition arrives.

    Every stage that reads phase reads it the same way, whether it filters the
    frames or estimates flows across them.

    Attributes:
        DEFAULT_SUBPATH: Koala's own layout, which is where a phase sequence
            comes off the microscope. A tree holding another modality names its
            own `subpath`, and one that names the wrong layout is found empty.
        subpath: As `SourceConfig`.
        root: As `SourceConfig`.
        frames: As `SourceConfig`.
    """

    DEFAULT_SUBPATH: ClassVar[str] = PHASE_FLOAT_BIN


def _source_key(
    source_config: SourceConfig,
    select_config: SequenceSelectConfig,
) -> tuple[object, ...]:
    """Return what makes two searches the same search.

    Every field is read rather than a chosen few, so a setting added later
    cannot quietly reuse an answer it would have changed. The working directory
    joins them because a relative `root` or `include` names a different folder
    once `hydra.job.chdir` has moved it.
    """

    def frozen(value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        if is_dataclass(value) and not isinstance(value, type):
            return astuple(value)

        return value

    values: list[object] = [str(Path.cwd())]

    for config in (source_config, select_config):
        values.extend(getattr(config, field.name) for field in fields(config))

    return tuple(frozen(value) for value in values)


def _ensure_selection(value: list[str] | str | None, key: str) -> None:
    """Raise unless a selection names something to select by.

    An empty one reads as no selection at all, so a list that came out empty
    takes the whole dataset rather than none of it, and says nothing about
    either. A file is opened only once the walk is done, so a mistyped path
    spent the whole search before failing, and failed in the library's words
    rather than naming the setting that carried it.

    Args:
        value: What the setting holds: names, a path to a file of them, or
            `None`.
        key: The setting's own name, so a refusal says where to go and fix it.

    Raises:
        ValueError: If `value` is empty, or a path to a file that is not there.
    """
    if value is None:
        return

    if not value:
        msg = f"empty {key}: leave it null to take every sequence"
        raise ValueError(msg)

    if isinstance(value, str) and is_spec_file(value) and not Path(value).is_file():
        msg = f"no such {key} listing: {value}"
        raise ValueError(msg)


def _search_sources(
    source_config: SourceConfig,
    select_config: SequenceSelectConfig,
) -> SearchResult:
    """Walk the root, open every sequence it holds, and keep the ones taken."""
    _ensure_selection(select_config.include, "select.include")
    _ensure_selection(select_config.exclude, "select.exclude")

    root = source_config.root
    subpath = source_config.resolve_subpath()
    holds_frames = contains(subpath, kind="dir")

    def descend(folder: Path) -> bool:
        return not holds_frames(folder)

    folders = search_phase_bin_folders(root, subpath=subpath, descend=descend)

    if (num_folders := len(folders)) == 0:
        msg = f"no time-lapse holds a {subpath!r} folder: {root}"
        raise ValueError(msg)

    def folder_subpath(folder: PhaseFileFolder) -> str:
        return stringify_path(folder.root, after=root, before=subpath)

    sources: list[PhaseFileFolder] = select(
        folders,
        key=folder_subpath,
        include=select_config.include,
        exclude=select_config.exclude,
    )

    if not sources:
        msg = f"include/exclude left none of the {num_folders} sequences: {root}"
        raise ValueError(msg)

    taken = []
    for source in sources:
        try:
            source.validate_if_supported(level="names")
            taken.append(source.with_unit(PhaseUnit.RADIANS))
        except ValueError as error:
            msg = f"{folder_subpath(source)}: {error}"
            raise ValueError(msg) from error

    frames = source_config.frames

    contents: dict[str, tuple[str, ...]] = {}
    for folder in folders:
        indices = frames.indices(len(folder.files))
        names = tuple(folder.files[index].name for index in indices)
        contents[folder_subpath(folder)] = names

    count = frames.count
    policy = ensure_policy(frames.if_short, SHORT_SEQUENCE_POLICIES, "frames.if_short")

    for source in sources:
        name = folder_subpath(source)
        held = len(contents[name])

        if not held:
            whole = quantify(len(source.files), "frame")
            fix = "lower `source.frames.start` or `source.frames.step`"
            msg = f"{name}: the frame selection takes none of its {whole}: {fix}"
            raise ValueError(msg)

        if count is not None and policy == "error" and held < count:
            asked = quantify(count, "frame")
            short_of = f"short of the {asked} asked for"
            msg = f"{name}: {held} frames after the stride, {short_of}"
            raise ValueError(msg)

    return taken, contents


def search_sources(
    source_config: SourceConfig,
    select_config: SequenceSelectConfig,
) -> SearchResult:
    """Find the sequences a run reads, narrowed by what it was told to take.

    Every sequence taken is checked for a missing frame before any of them is
    run, since a gap is a fault in the dataset rather than in one item of work.
    A gap otherwise opens as an ordinary shorter sequence, and what is written
    back out is numbered without one, so nothing downstream can tell.

    A selection that lands on none of a sequence's frames is refused there too,
    and is not the policy's to decide: a run reading nothing at all is a
    setting that was written wrong rather than a dataset that came up short.

    Nothing inside a time-lapse is descended into. Opening one lists its frames
    already, and the walk has no reason to list them a second time looking for
    a time-lapse that cannot be nested there.

    The answer is held for the next call asking the same thing, since a sweep
    runs every job in one process and only the filter differs between them.
    Only the newest is held, so a call asking for something else pays what it
    would have paid anyway. A sweep cannot write frames at all, which is what
    leaves the answer standing for as long as one runs.

    Returns:
        One folder per sequence taken, each set to give its frames in radians,
        and a contents of every sequence the root holds against the frames the
        run would measure it over. The contents covers what the selection left
        out too, which is what lets a document say it describes part of a
        dataset rather than the whole of a smaller one, and what an output with
        no sequence behind it is measured against. Both are the caller's own to
        reorder or add to; the folders inside them are shared and read-only.

    Raises:
        ValueError: If the root holds no sequence at all, if the selection
            leaves none of the ones it holds, or if a sequence taken is missing
            a frame. The first two are told apart, since they are fixed
            differently.
    """
    key = _source_key(source_config, select_config)

    if (found := LAST_SEARCH.get(key)) is None:
        found = _search_sources(source_config, select_config)
        LAST_SEARCH.clear()
        LAST_SEARCH[key] = found

    sources, contents = found

    return list(sources), dict(contents)


def build_sequences(
    source_config: SourceConfig,
    select_config: SequenceSelectConfig,
    kernel_config: KernelConfig,
) -> tuple[list[PhaseFilteredSequence], dict[str, tuple[str, ...]]]:
    """Build one filtered view per sequence, all sharing a single kernel.

    A kernel holds only the shape it reads, never frames, so one serves every
    sequence of the run.

    Returns:
        The sequences, in the order the search found them, and the contents of
        the whole dataset they were selected from.
    """
    sources, contents = search_sources(source_config, select_config)
    subpath = source_config.resolve_subpath()
    frames = source_config.frames

    kernel = kernel_config.build()

    def build_sequence(source: PhaseFileFolder) -> PhaseFilteredSequence:
        return PhaseFilteredSequence(
            source,
            kernel,
            root=source_config.root,
            subpath=subpath,
            start=frames.start,
            step=frames.step,
            count=frames.count,
        )

    return [build_sequence(source) for source in sources], contents


def log_short_sequences(
    frame_config: FrameSelectConfig,
    *,
    sequences: Sequence[Named],
    contents: Mapping[str, Sequence[str]],
    name: str,
) -> None:
    """Name the sequences that could not supply the count that was asked for.

    Said after the search rather than with the rest of the configuration, since
    it is what the dataset turned out to hold and not what the run was told to
    do. `"error"` never reaches here: the search refuses there.

    Args:
        frame_config: The frame selection, for the count to fall short of.
        sequences: The sequences the run took.
        contents: Every sequence the source holds, against the frames each
            would be read over.
        name: The run's name, which the warning is filed under.
    """
    count = frame_config.count
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

    listed = ", ".join(short[:LISTING_LIMIT])
    if (rest := len(short) - LISTING_LIMIT) > 0:
        listed = f"{listed}, and {rest} more"

    counted = quantify(len(short), "sequence")

    logger.warning("%s gave fewer than %d: %s", counted, count, listed)
