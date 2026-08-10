from __future__ import annotations

__all__ = (
    "SELECTION_LIMIT",
    "SelectConfig",
    "TreeConfig",
    "log_source_config",
)

from dataclasses import dataclass
from pathlib import PurePath
from typing import TYPE_CHECKING, ClassVar, Final

from iivs.dhm.data.koala import PHASE_FLOAT_BIN
from kaparoo.filesystem import is_spec_file
from kaparoo.utils import quantify, unwrap_or_default
from omegaconf import MISSING

from iivs_cardio.common.logging import log_indented
from iivs_cardio.data.transforms.filtering import frame_indices

if TYPE_CHECKING:
    from logging import Logger


# ========================== #
#           Trees            #
# ========================== #


@dataclass
class TreeConfig:
    """A tree of sequences, and where inside one of them the frames sit.

    What a run reads and what it writes are the same shape, so the pair of
    settings that says where a tree is lives here once. A subclass may set
    `DEFAULT_SUBPATH` to the layout its own end of a stage uses, which is what
    an unset `subpath` comes to when the caller offers nothing to follow.

    `DEFAULT_SUBPATH` is the one thing here that assumes a modality. It sits on
    the base while phase is the only one read, and moves to whatever names the
    reader when a second arrives: a hologram search takes no `subpath` at all,
    so the layout is the reader's to know rather than the tree's.

    Attributes:
        root: The folder the sequences sit under.
        subpath: The path to a sequence's frames inside its own folder. Defaults
            to `None`, which follows what the caller offers and falls back to
            this end's own layout when it offers nothing.
    """

    DEFAULT_SUBPATH: ClassVar[str] = PHASE_FLOAT_BIN

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
class SelectConfig:
    """Which sequences a run takes, and how much of each.

    Held apart from the tree they are taken from, since a run reading two trees
    takes the same sequences and the same frames from both. Two copies of this
    could disagree, and a run that paired frames which do not correspond would
    compute on them quietly.

    Attributes:
        include: The sequences to take, as names or as a path to a file listing
            them. Defaults to `None`, which takes all of them.
        exclude: The same, for sequences to leave out. Defaults to `None`.
        frame_start: The first source frame to take. Defaults to 0.
        frame_step: The stride to read each sequence at, so that every
            `frame_step`th frame from `frame_start` is taken. Defaults to 1.
        frame_count: How many frames to take once the stride has been applied.
            Defaults to `None`, which takes them all.
        if_frames_short: The policy for a sequence that cannot supply
            `frame_count`, which says nothing when there is no count to fall
            short of. `"take"` takes what there is and names the sequence in
            the log. Defaults to `"take"`.
    """

    include: list[str] | str | None = None
    exclude: list[str] | str | None = None
    frame_start: int = 0
    frame_step: int = 1
    frame_count: int | None = None
    if_frames_short: str = "take"

    def frame_indices(self, total: int) -> range:
        """Return which of `total` source frames this run takes, in order."""
        return frame_indices(
            total,
            start=self.frame_start,
            step=self.frame_step,
            count=self.frame_count,
        )


# ========================== #
#          Logging           #
# ========================== #


SELECTION_LIMIT: Final = 5


def _log_selection(logger: Logger, verb: str, value: list[str] | str) -> None:
    """Log a selection, listing it only while a list is short enough to read."""
    if isinstance(value, str):
        if is_spec_file(value):
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


def log_source_config(
    source_config: TreeConfig, select_config: SelectConfig, logger: Logger
) -> None:
    """Log what a run reads, naming only the settings that were moved."""
    log_indented(logger, "source: %s", source_config.root, depth=0)

    log_indented(logger, "reading <sequence>/%s", source_config.resolve_subpath())

    _log_frames(select_config, logger)

    if select_config.include:
        _log_selection(logger, "including", select_config.include)

    if select_config.exclude:
        _log_selection(logger, "excluding", select_config.exclude)


def _log_frames(select_config: SelectConfig, logger: Logger) -> None:
    """Log which source frames a run takes, unless it takes every one.

    Shown as the first few positions rather than as the three settings, since
    what a reader checks is whether the frames are the ones they meant and the
    settings are what they already wrote.
    """
    start = select_config.frame_start
    step = select_config.frame_step
    count = select_config.frame_count
    if (start, step, count) == (0, 1, None):
        return

    shown = [start + index * step for index in range(3)][: count or 3]
    listed = ", ".join(str(index) for index in shown)
    tail = "" if count is not None and count <= len(shown) else ", ..."
    held = "" if count is None else f" (at most {quantify(count, 'frame')})"
    log_indented(logger, "reading frames %s%s%s", listed, tail, held)

    if count is not None and select_config.if_frames_short == "error":
        log_indented(logger, "refusing a sequence that cannot supply them")
