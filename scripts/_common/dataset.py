from __future__ import annotations

__all__ = (
    "LISTING_LIMIT",
    "SHORT_SEQUENCE_POLICIES",
    "BranchConfig",
    "FrameSelectConfig",
    "SequenceLayout",
    "SequenceSelectConfig",
    "ShortSequencePolicy",
    "SourceConfig",
    "TreeBranchConfig",
    "ensure_output_clear",
    "log_branch_policies",
    "log_source_config",
)

from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, ClassVar, Final, Literal

from kaparoo.filesystem import is_spec_file
from kaparoo.utils import literal_values, quantify, unwrap_or_default
from omegaconf import MISSING

from iivs_cardio.common.logging import log_indented
from iivs_cardio.common.pipeline import PresentPolicy, UnsourcedPolicy
from iivs_cardio.data.transforms.filtering import frame_indices

if TYPE_CHECKING:
    from logging import Logger

    from kaparoo.filesystem.types import StrPath


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


@dataclass
class BranchConfig:
    """What one side branch writes, and what it does where it finds an output.

    Every branch answers the same three questions, so a stage adding one adds a
    block of this shape rather than another three keys beside the others.

    Attributes:
        save: Whether to write this output at all. Defaults to `False`.
        if_present: The policy for a sequence this output already covers.
            `"reuse"` keeps what an earlier run left that still describes this
            one, and writes the rest. Defaults to `"error"`.
        if_unsourced: The policy for part of this output whose sequence the
            source no longer holds. Defaults to `"keep"`: the same absence is
            what a half mounted share looks like, and what is kept is always
            said out loud.
    """

    save: bool = False
    if_present: PresentPolicy = "error"
    if_unsourced: UnsourcedPolicy = "keep"


@dataclass
class TreeBranchConfig(BranchConfig, SequenceLayout):
    """The branch that writes each sequence back out as a tree of its own.

    A stage names the subclass it writes, so the layout a bare `subpath` falls
    back to is the one that stage's own trees keep their frames in.

    Attributes:
        DEFAULT_SUBPATH: As `SequenceLayout`, supplied by the stage's subclass.
            Never reached while the run has a source to follow, and there to
            keep a caller without one from writing into the sequence folder
            itself.
        save: As `BranchConfig`.
        subpath: The path a written sequence keeps its frames at inside its own
            folder. Naming one is what lets a run write beside the frames it
            read rather than over them. Defaults to `None`, which puts them
            where the source keeps its own.
        record_file: The name of the file each written folder keeps its own
            account in, given `.json` if it has no extension. A later run reads
            it to decide whether what is there still describes this run.
            Defaults to `"source"`.
        if_present: As `BranchConfig`, judged by the settings and the source
            frames' names rather than by what those frames hold. A source
            re-exported under the same names is kept rather than written again,
            so a run that follows one takes `"overwrite"`.
        if_unsourced: As `BranchConfig`.
    """

    record_file: str = "source"


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


def log_branch_policies(output: str, branch: BranchConfig, logger: Logger) -> None:
    """Say what a branch does with what it finds, unless it refuses to run.

    Set in under the line naming the output it belongs to, since a target
    writes more than one and a policy at the same depth as both would read as
    either.

    Args:
        output: What the branch writes, named as a plural the lines read with.
        branch: The branch whose policies are being said.
        logger: The logger the lines go to.
    """
    lines = {
        "overwrite": f"overwriting the {output} it finds",
        "reuse": f"reusing the {output} that match this run",
    }
    if (line := lines.get(branch.if_present)) is not None:
        log_indented(logger, "%s", line, depth=2)

    if branch.if_unsourced != "keep":
        log_indented(logger, "dropping the %s a source no longer has", output, depth=2)


# ========================== #
#          Outputs           #
# ========================== #


def ensure_output_clear(
    source_root: StrPath,
    output_root: StrPath,
    *,
    what: str,
    read: str,
    written: str,
    fix: str,
) -> None:
    """Raise where the tree a run writes would land on the one it reads.

    A sequence is written by replacing its folder whole, so an output under the
    source root is refused wherever its layout would land on the frames being
    read: the same folder, or either one holding the other.

    An output beside the source, or above it, is left open. It writes a tree of
    its own and collides with nothing here, and whether a later run pointed at
    a parent of both would then find two of every sequence is that run's own
    `source.root` to get right.

    Args:
        source_root: The folder the sequences are read from.
        output_root: The folder this run writes under.
        what: What is being written, named as a plural the refusal reads with.
        read: The layout the frames are read at, inside a sequence's folder.
        written: The layout this run would write at, in the same terms.
        fix: The settings to change, named as the refusal should say them.

    Raises:
        ValueError: If the two layouts would land on one another.
    """
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()

    if not output_root.is_relative_to(source_root):
        return

    reading = PurePath(read)
    writing = PurePath(written)

    if reading.is_relative_to(writing) or writing.is_relative_to(reading):
        where = f"{output_root.as_posix()}/*/{writing.as_posix()}"
        msg = f"{what} would land on the source at {where}: set {fix}"
        raise ValueError(msg)
