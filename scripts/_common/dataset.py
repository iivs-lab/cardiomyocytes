from __future__ import annotations

__all__ = (
    "LISTING_LIMIT",
    "SHORT_SEQUENCE_POLICIES",
    "FrameSelectConfig",
    "SequenceLayout",
    "SequenceSelectConfig",
    "ShortSequencePolicy",
    "SourceConfig",
    "log_source_config",
)

from dataclasses import dataclass, field
from pathlib import PurePath
from typing import TYPE_CHECKING, ClassVar, Final, Literal

from kaparoo.filesystem import is_spec_file
from kaparoo.utils import literal_values, quantify, unwrap_or_default
from omegaconf import MISSING

from iivs_cardio.common.logging import log_indented
from iivs_cardio.data.transforms.filtering import frame_indices

if TYPE_CHECKING:
    from logging import Logger


# ========================== #
#          Configs           #
# ========================== #


type ShortSequencePolicy = Literal["take", "error"]

SHORT_SEQUENCE_POLICIES: Final[tuple[ShortSequencePolicy, ...]] = literal_values(
    ShortSequencePolicy
)


@dataclass
class SequenceLayout:
    """Where one end of a stage keeps a sequence's frames, inside its own folder.

    A subclass supplies `DEFAULT_SUBPATH`, since a phase tree keeps its frames
    somewhere a flow tree does not and nothing here knows which end this is.
    One that does not is refused the moment it has to settle an unset `subpath`.

    Attributes:
        DEFAULT_SUBPATH: The layout this end keeps its frames in, for a config
            that names none and follows nothing.
        subpath: The layout that was asked for. Defaults to `None`, which takes
            whichever the class or the other end settles on.
    """

    DEFAULT_SUBPATH: ClassVar[str]

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

        Returns:
            The layout, in posix form, and empty for one naming the sequence's
            own folder. Both spellings of that come back the same, so whatever
            reads the answer has one of them to handle rather than two.

        Raises:
            ValueError: If the answer would reach outside a sequence's folder.
        """
        default = unwrap_or_default(follow, self.DEFAULT_SUBPATH)
        subpath = unwrap_or_default(self.subpath, default)

        path = PurePath(subpath)
        if path.anchor or ".." in path.parts:
            msg = f"invalid subpath {subpath!r}: expected a relative path, no '..'"
            raise ValueError(msg)

        return "/".join(path.parts)


@dataclass
class FrameSelectConfig:
    """Which frames of a sequence a run takes, and how much of it they are.

    Held per tree rather than per run, since two trees a run reads may be at
    different rates: a source at 20 Hz asked for 10 Hz takes every second
    frame, where a flow cache already written at 10 Hz takes every one. The
    numbers differ because the trees differ, and what they arrive at is the
    same rate.

    Attributes:
        start: The first source frame to take. Defaults to 0.
        step: The stride to read the tree at, so that every `step`th frame from
            `start` is taken. Defaults to 1.
        count: How many frames to take once the stride has been applied.
            Defaults to `None`, which takes them all.
        if_short: The policy for a sequence that cannot supply `count`, which
            says nothing when there is no count to fall short of. `"take"`
            takes what there is and names the sequence in the log. Defaults to
            `"take"`.
    """

    start: int = 0
    step: int = 1
    count: int | None = None
    if_short: ShortSequencePolicy = "take"

    def indices(self, total: int) -> range:
        """Return which of `total` source frames this run takes, in order."""
        return frame_indices(total, start=self.start, step=self.step, count=self.count)


@dataclass
class SourceConfig(SequenceLayout):
    """A tree a run reads frames from, and which of them it takes.

    A stage names the subclass it reads, so the layout a bare `subpath` falls
    back to is the one that stage's own trees keep their frames in.

    Attributes:
        DEFAULT_SUBPATH: As `SequenceLayout`, supplied by the stage's subclass.
        subpath: The path to a sequence's frames inside its own folder. Defaults
            to `None`, which takes `DEFAULT_SUBPATH`.
        root: The folder the sequences sit under.
        frames: Which frames of each sequence to take. Defaults to all of them.
    """

    root: str = MISSING
    frames: FrameSelectConfig = field(default_factory=FrameSelectConfig)


@dataclass
class SequenceSelectConfig:
    """Which sequences of a tree a run takes.

    One per run rather than one per tree, since a sequence keeps its name
    wherever it is written: a cache holds `plate_A/TL_01` under that name too,
    so the same two settings pick the same sequences from every tree. Frame
    numbers do not survive that way, which is why they are the tree's.

    Attributes:
        include: The sequences to take, as names or as a path to a file listing
            them. Defaults to `None`, which takes all of them.
        exclude: The same, for sequences to leave out. Defaults to `None`.
    """

    include: list[str] | str | None = None
    exclude: list[str] | str | None = None


# ========================== #
#          Logging           #
# ========================== #


# How many of the positions a run takes are shown before the line trails off.
_PREVIEW_LIMIT: Final = 3

# How many sequences a line names before it gives their number instead.
LISTING_LIMIT: Final = 5


def _log_frame_selection(frame_config: FrameSelectConfig, logger: Logger) -> None:
    start = frame_config.start
    step = frame_config.step
    count = frame_config.count
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

    if counted and frame_config.if_short == "error":
        log_indented(logger, "a short sequence stops the run", depth=2)


def _log_sequence_selection(label: str, value: list[str] | str, logger: Logger) -> None:
    if isinstance(value, str):
        if is_spec_file(value):
            log_indented(logger, "%s the sequences listed in %s", label, value)
        else:
            log_indented(logger, "%s %s", label, value)
        return

    count = len(value)
    sequences = quantify(count, "sequence")

    if count > LISTING_LIMIT:
        record = ".hydra/{config,overrides}.yaml"
        log_indented(logger, "%s %s, listed in %s", label, sequences, record)
        return

    log_indented(logger, "%s %s:", label, sequences)
    for item in value:
        log_indented(logger, "%s", item, depth=2)


def log_source_config(
    source_config: SourceConfig,
    sequence_config: SequenceSelectConfig,
    logger: Logger,
) -> None:
    """Log what a run reads, naming only the settings that were moved.

    Args:
        source_config: The tree the run reads.
        sequence_config: Which of its sequences it takes.
        logger: The logger the lines go to.
    """
    log_indented(logger, "source: %s", source_config.root, depth=0)

    log_indented(logger, "reading <sequence>/%s", source_config.resolve_subpath())

    _log_frame_selection(source_config.frames, logger)

    if sequence_config.include:
        _log_sequence_selection("including", sequence_config.include, logger)

    if sequence_config.exclude:
        _log_sequence_selection("excluding", sequence_config.exclude, logger)
