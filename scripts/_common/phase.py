from __future__ import annotations

__all__ = (
    "DEFAULT_SUBPATH",
    "LAST_SEARCH",
    "SearchResult",
    "build_sequences",
    "search_sources",
)

from dataclasses import astuple, fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from iivs.dhm.data.koala import PHASE_FLOAT_BIN
from iivs.dhm.data.phase import PhaseFileFolder, PhaseUnit, search_phase_bin_folders
from kaparoo.filesystem import contains, select, stringify_path
from kaparoo.utils import quantify

from iivs_cardio.common.pipeline import SHORT_INPUT_POLICIES, ensure_policy
from iivs_cardio.data.phase import PhaseFilteredSequence
from scripts._common.dataset import resolve_subpath

if TYPE_CHECKING:
    from iivs_cardio.data.transforms.filtering.kernel import KernelConfig
    from scripts._common.dataset import SequenceSelectConfig, SourceConfig


# Where a phase sequence keeps its frames, for a tree that names no layout of
# its own. The reader's to know, since a hologram search takes no subpath.
DEFAULT_SUBPATH: Final = PHASE_FLOAT_BIN

# One folder per sequence taken, against the contents of the dataset they came from.
type SearchResult = tuple[list[PhaseFileFolder], dict[str, tuple[str, ...]]]

# The newest search, held for the next job of a sweep to take.
LAST_SEARCH: dict[tuple[object, ...], SearchResult] = {}


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


def _search_sources(
    source_config: SourceConfig,
    select_config: SequenceSelectConfig,
) -> SearchResult:
    """Walk the root, open every sequence it holds, and keep the ones taken."""
    root = source_config.root
    subpath = resolve_subpath(source_config.subpath, default=DEFAULT_SUBPATH)
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
    if count is not None:
        policy = ensure_policy(frames.if_fewer, SHORT_INPUT_POLICIES, "frames.if_fewer")

        for source in sources:
            name = folder_subpath(source)
            held = len(contents[name])
            if policy == "error" and held < count:
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
    subpath = resolve_subpath(source_config.subpath, default=DEFAULT_SUBPATH)
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
