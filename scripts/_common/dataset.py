from __future__ import annotations

__all__ = ("SelectConfig", "TreeConfig", "log_source_config")

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
#          Configs           #
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


# How many of the positions a run takes are shown before the line trails off.
_PREVIEW_LIMIT: Final = 3

# How many sequences are named before the line points at the config instead.
_LISTING_LIMIT: Final = 5


def _log_frame_selection(select_config: SelectConfig, logger: Logger) -> None:
    start = select_config.frame_start
    step = select_config.frame_step
    count = select_config.frame_count
    if (start, step, count) == (0, 1, None):
        return

    counted = count is not None
    preview = min(count, _PREVIEW_LIMIT) if counted else _PREVIEW_LIMIT

    listed = ", ".join(str(start + index * step) for index in range(preview))
    if not counted or count > preview:
        listed = f"{listed}, ..."
    if counted:
        listed = f"{listed} (at most {quantify(count, 'frame')})"

    log_indented(logger, "reading frames %s", listed)

    if counted and select_config.if_frames_short == "error":
        log_indented(logger, "a sequence with fewer stops the run", depth=2)


def _log_sequence_selection(label: str, value: list[str] | str, logger: Logger) -> None:
    if isinstance(value, str):
        if is_spec_file(value):
            log_indented(logger, "%s the sequences listed in %s", label, value)
        else:
            log_indented(logger, "%s %s", label, value)
        return

    count = len(value)
    sequences = quantify(count, "sequence")

    if count > _LISTING_LIMIT:
        record = ".hydra/{config,overrides}.yaml"
        log_indented(logger, "%s %s, listed in %s", label, sequences, record)
        return

    log_indented(logger, "%s %s:", label, sequences)
    for item in value:
        log_indented(logger, "%s", item, depth=2)


def log_source_config(
    source_config: TreeConfig,
    select_config: SelectConfig,
    logger: Logger,
) -> None:
    """Log what a run reads, naming only the settings that were moved."""
    log_indented(logger, "source: %s", source_config.root, depth=0)

    log_indented(logger, "reading <sequence>/%s", source_config.resolve_subpath())

    _log_frame_selection(select_config, logger)

    if select_config.include:
        _log_sequence_selection("including", select_config.include, logger)

    if select_config.exclude:
        _log_sequence_selection("excluding", select_config.exclude, logger)
