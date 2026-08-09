from __future__ import annotations

__all__ = ("FRAME_POLICIES", "FrameTree")

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self

from kaparoo.filesystem import dir_exists, search_dirs
from kaparoo.utils.optional import unwrap_or_default

from iivs_cardio.common.pipeline.branch import (
    EXISTING_OUTPUT_POLICIES,
    STAGING,
    as_read_back,
    counted,
    find_unsourced,
    prune_above,
    read_policy,
)
from iivs_cardio.data.phase import phase_frame_writer
from iivs_cardio.data.writer import RECORD_FILE

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import TracebackType

    from kaparoo.filesystem.types import StrPath
    from torch import Tensor

    from iivs_cardio.common.pipeline.branch import (
        ExistingOutputPolicy,
        UnsourcedOutputPolicy,
    )
    from iivs_cardio.data.phase import PhaseFilteredSequence
    from iivs_cardio.data.writer import KoalaFrameWriter

FRAME_POLICIES: Final[tuple[ExistingOutputPolicy, ...]] = EXISTING_OUTPUT_POLICIES


@dataclass(frozen=True, slots=True)
class FrameTree:
    """The side branch that writes each sequence back out under a new root.

    A written sequence keeps the name and the layout it had in the source, so
    the result can be read by whatever reads the source. Each writer takes the
    pixel size, height scale and unit from the sequence it was made for.

    The tree has a lifetime as well as its writers, because two things outlive
    any one of them. A writer clears up after itself only while it is alive, so
    a worker killed part way leaves a staged folder and the empty folders above
    it; and a sequence the dataset has dropped leaves a folder no writer will
    ever be made for. Both are found by looking at the tree, which is what
    closing it does.

    Attributes:
        root: The directory the tree is written under.
        subpath: The path to a sequence's frames inside its own folder.
        contents: Every sequence the source holds, which is what tells a folder
            with no sequence behind it from one this run simply did not take.
            Given rather than defaulted, since an empty one leaves every folder
            here unsourced and `if_sources_gone` would then take the whole tree.
        settings: The settings that shaped the frames, filed inside each
            sequence's folder beside the source names its writer collected. A
            phase header carries no time and no source name, so without this a
            written sequence cannot be traced back to the acquisition it came
            from. Defaults to `None`, which files nothing.
        selected: The sequences of the contents this run was given to write.
            Repeats count once. Defaults to `None`, which takes all of them.
        if_frames_exist: The policy for a sequence that already has a folder
            here. `"reuse"` keeps one whose record still describes this run and
            writes the rest. Defaults to `"error"`.
        if_sources_gone: The policy for a folder whose sequence the source has
            lost. Defaults to `"keep"`.

    Raises:
        ValueError: If `if_frames_exist` is not a policy a tree offers, or if
            `selected` names something the contents does not hold.
    """

    root: StrPath
    subpath: str
    contents: Mapping[str, Sequence[str]]
    settings: Mapping[str, object] | None = None
    selected: Sequence[str] | None = field(default=None, kw_only=True)
    if_frames_exist: ExistingOutputPolicy = field(default="error", kw_only=True)
    if_sources_gone: UnsourcedOutputPolicy = field(default="keep", kw_only=True)
    _taken: frozenset[str] = field(init=False, repr=False)
    _recorded: object = field(init=False, repr=False)
    _reused: set[str] = field(default_factory=set, init=False, repr=False)
    _dropped: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        read_policy(self.if_frames_exist, FRAME_POLICIES, "if_frames_exist")

        names = unwrap_or_default(self.selected, tuple(self.contents))
        if unknown := [name for name in names if name not in self.contents]:
            msg = f"selected {unknown[0]!r}, which the source does not hold"
            raise ValueError(msg)

        object.__setattr__(self, "_taken", frozenset(names))
        object.__setattr__(self, "_recorded", as_read_back(self.settings))

    @property
    def _replacing(self) -> bool:
        """Whether a folder already there may be written over.

        `"reuse"` replaces as readily as `"overwrite"` does: what it keeps it
        keeps by never making a writer for it, so a writer that was made has
        already been told the folder does not describe this run.
        """
        return self.if_frames_exist != "error"

    def get_hook(
        self, source: PhaseFilteredSequence
    ) -> KoalaFrameWriter[Tensor] | None:
        """Return the writer for `source`, or `None` to keep what is there.

        Whether a folder still stands for this run was settled when the tree
        opened, where the whole dataset was in view; this only looks the answer
        up. A sequence nothing has to write costs no frames at all, which is
        what reuse is for.

        The record the writer files names the sequence as the dataset does, so
        a folder read on its own says which acquisition it came from. The root
        it sat under is left out: an absolute path does not survive the move
        from this machine to the server, and a wrong one is worse than none.

        Returns:
            The writer, placed where the source sits, or `None` when a folder
            already there was found to still describe this run.
        """
        if source.name in self._reused:
            return None

        origin = source.origin
        header = origin.header

        record = None
        if self.settings is not None:
            record = {"settings": dict(self.settings), "source": source.name}

        return phase_frame_writer(
            Path(self.root, source.name, self.subpath),
            pixel_size=header.pixel_size,
            height_scale=header.height_scale,
            unit=unwrap_or_default(origin.target_unit, header.unit),
            overwrite=self._replacing,
            record=record,
        )

    def list_sequences(self) -> list[str]:
        """Return every sequence this tree already holds frames for, sorted.

        A sequence is named by where its own folder sits under the root, and is
        recognised by holding `subpath` rather than by the walk reaching it, so
        nothing below one is ever listed. A folder inside a sequence is
        therefore never taken for one itself.
        """
        root = Path(self.root)
        if not root.is_dir():
            return []

        subpath = self.subpath
        found = search_dirs(
            root,
            predicate=lambda folder: dir_exists(folder / subpath),
            exclude=lambda folder: dir_exists(folder.parent / subpath),
            ordered=False,
        )

        return sorted(folder.relative_to(root).as_posix() for folder in found)

    def _still_describes(self, name: str) -> bool:
        """Whether the folder already written for `name` stands for this run.

        Three things can have moved since it was written, and none of them
        shows in the folder's name: the settings that shaped the frames, which
        frames the source holds, and whether the folder still holds all of what
        its record says it does. The third has no counterpart in a range part,
        which is one file and so is either there or not; a folder can be half
        removed, and reusing that would leave a short sequence reading as a
        whole one.
        """
        folder = Path(self.root, name, self.subpath)

        try:
            read = (folder / RECORD_FILE).read_text(encoding="utf-8")
            record = json.loads(read)
        except (OSError, ValueError):
            return False

        if not isinstance(record, dict):
            return False

        if record.get("settings") != self._recorded:
            return False

        frames = record.get("frames")
        if not isinstance(frames, list) or tuple(frames) != tuple(self.contents[name]):
            return False

        return self._count_frames(folder) == len(frames)

    @staticmethod
    def _count_frames(folder: Path) -> int:
        """Count what the folder holds beside the record it carries."""
        return sum(1 for path in folder.iterdir() if path.name != RECORD_FILE)

    def list_unsourced(self) -> list[str]:
        """Return the sequences this tree holds that the source has lost, sorted.

        A source that looks smaller than it is reads the same from here, so
        acting on the list is `if_sources_gone`'s to decide and naming it is not.
        """
        return find_unsourced(self.list_sequences(), self.contents)

    def drop_unsourced(self) -> list[str]:
        """Remove the folders of sequences the source has lost, and name them.

        The folders a removal empties go with it, so a sequence dropped from a
        nested dataset does not leave the path down to it standing.

        Returns:
            The sequences whose folders were removed, in the order they were
            found. They are kept for `report`, so a removal is said to have
            happened and not only to have been due.
        """
        root = Path(self.root)
        dropped = []

        for name in self.list_unsourced():
            folder = root / name
            shutil.rmtree(folder)
            prune_above(folder.parent, root)
            dropped.append(name)

        self._dropped.extend(dropped)

        return dropped

    def clear_staging(self) -> None:
        """Drop what a writer that did not live to clear up after itself left.

        A staged folder sits under a hidden name ending in `.tmp`, next to the
        destination it is to become, and the parents made to hold it are left
        empty when the move never happens. The writer undoes both while it is
        alive; a worker killed outright leaves them here. Only that shape is
        taken, and only the folders it leaves empty, since a run writes into
        the directory that also keeps its configuration and its logs.
        """
        root = Path(self.root)
        if not root.is_dir():
            return

        for folder in search_dirs(root, name_filter=STAGING):
            shutil.rmtree(folder, ignore_errors=True)
            prune_above(folder.parent, root)

    def report(self) -> str | None:
        """Return one line naming what was kept and removed, or `None` if neither.

        What was written is not counted here: the run's own summary already
        says how many sequences it computed, and the document that indexes the
        frames counts them again. What is left is the two a tree alone knows,
        both of them about folders it did not write this time.
        """
        said = []
        if self._reused:
            said.append(
                f"kept {counted(len(self._reused), 'sequence')} already written"
            )
        if self._dropped:
            said.append(
                f"removed {counted(len(self._dropped), 'folder')} with no source"
            )

        return ", ".join(said) or None

    def __enter__(self) -> Self:
        """Settle what is already here before a single frame is read.

        Judging happens here, with the whole dataset in view and in one
        process. A worker holds a copy of this branch and nothing it learns
        comes home, so a folder judged there could not be counted.

        `"error"` refuses here too, rather than leaving it to the writer that
        meets the folder: the writer meets them one at a time, so a run over
        500 sequences whose 300th is already written pays for 299 of them
        first. What the writer refuses is the same thing, a moment too late.

        Raises:
            FileExistsError: If `if_frames_exist` is `"error"` and a sequence
                this run would write already has a folder here.
        """
        if self.if_frames_exist == "reuse":
            self._reused.update(
                name for name in self._written() if self._still_describes(name)
            )
        elif self.if_frames_exist == "error" and (written := self._written()):
            counts = counted(len(written), "sequence")
            fix = "set `if_frames_exist` to 'overwrite' or 'reuse'"
            msg = f"{counts} already written, from {written[0]!r}: {fix}"
            raise FileExistsError(msg)

        return self

    def _written(self) -> list[str]:
        """Return the sequences this run would write that already have a folder."""
        return [name for name in self.list_sequences() if name in self._taken]

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Clear up after the writers, whether or not the run reached the end.

        Debris always goes, since it is this code's own unfinished business and
        nothing else will collect it. Whole folders go only where the policy
        says so.
        """
        self.clear_staging()

        if self.if_sources_gone == "delete":
            self.drop_unsourced()
