from __future__ import annotations

__all__ = ("FrameBranch", "FrameTree")

import json
import shutil
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Self, override

from kaparoo.filesystem import contains, prune_upward, search_dirs, search_files
from kaparoo.filesystem.types import StrPath
from kaparoo.utils import quantify
from kaparoo.utils.optional import unwrap_or_default

from iivs_cardio.common.pipeline.base import Named
from iivs_cardio.common.pipeline.branch import (
    PRESENT_POLICIES,
    STAGING,
    PresentPolicy,
    UnsourcedPolicy,
    as_json_value,
    ensure_json_name,
    ensure_policy,
)
from iivs_cardio.data.phase import phase_frame_writer
from iivs_cardio.data.writer import RECORD_FILE

if TYPE_CHECKING:
    from types import TracebackType

    from torch import Tensor

    from iivs_cardio.data.phase import PhaseFilteredSequence
    from iivs_cardio.data.writer import KoalaFrameWriter


@dataclass(frozen=True, slots=True)
class FrameBranch[S: Named, T](ABC):
    """The side branch that writes each sequence back out under a new root.

    A written sequence keeps the name and the layout it had in the source, so
    the result can be read by whatever reads the source.

    A subclass says two things and inherits the rest: how to make the writer
    for one sequence, and how many frames this stage owes that sequence. The
    second is not always as many as the source holds, and a stage that gives
    back fewer would otherwise never reuse anything it wrote.

    Type Parameters:
        S: The thing a writer is made for, which is one sequence of a dataset,
            named the way the tree files it.
        T: The type of one frame, as the writer is handed it.

    The tree has a lifetime as well as its writers, because two things outlive
    any one of them. A writer clears up after itself only while it is alive, so
    a worker killed part way leaves a staged folder and the empty folders above
    it; and a sequence the dataset has dropped leaves a folder no writer will
    ever be made for. Both are found by looking at the tree, which is what
    closing it does.

    Attributes:
        root: The directory the tree is written under.
        subpath: The path to a sequence's frames inside its own folder. Empty
            puts them in the folder itself, which only a dataset whose names
            are one level deep can be read back from: a sequence is recognised
            by holding this, so with nothing to hold the walk stops at the
            first level and calls the folders there sequences.
        contents: Every sequence the source holds, which is what tells a folder
            with no sequence behind it from one this run simply did not take.
            Given rather than defaulted, since an empty one leaves every folder
            here unsourced and `if_unsourced` would then take the whole tree.
        settings: The settings that shaped the frames, filed inside each
            sequence's folder beside the source names its writer collected. A
            phase header carries no time and no source name, so without this a
            written sequence cannot be traced back to the acquisition it came
            from. Defaults to `None`, which files nothing.
        selected: The sequences of the contents this run was given to write.
            Repeats count once. Defaults to `None`, which takes all of them.
        record_file: The name each written folder keeps its own account under,
            given `.json` if it has no extension. A folder is read back by this
            name too, so a tree given one name cannot reuse what a tree given
            another wrote. Defaults to `RECORD_FILE`.
        if_present: The policy for a sequence that already has a folder here.
            `"reuse"` keeps one whose record still describes this run and writes
            the rest. Defaults to `"error"`.
        if_unsourced: The policy for a folder whose sequence the source has lost.
            Defaults to `"keep"`.

    Raises:
        ValueError: If `subpath` is empty for a dataset whose names are
            nested, if `if_present` is not a policy a tree offers, if
            `record_file` carries a directory part, or if `selected` names
            something the contents does not hold.
    """

    root: StrPath
    subpath: str
    contents: Mapping[str, Sequence[str]]
    settings: Mapping[str, object] | None = None
    selected: Sequence[str] | None = field(default=None, kw_only=True)
    record_file: str = field(default=RECORD_FILE, kw_only=True)
    if_present: PresentPolicy = field(default="error", kw_only=True)
    if_unsourced: UnsourcedPolicy = field(default="keep", kw_only=True)
    _taken: frozenset[str] = field(init=False, repr=False)
    _recorded: object = field(init=False, repr=False)
    _reused: set[str] = field(default_factory=set, init=False, repr=False)
    _dropped: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        ensure_policy(self.if_present, PRESENT_POLICIES, "if_present")
        object.__setattr__(self, "record_file", ensure_json_name(self.record_file))

        if not self.subpath and any("/" in name for name in self.contents):
            fix = "give the branch a `subpath`"
            msg = f"a nested dataset cannot be found again without a layout: {fix}"
            raise ValueError(msg)

        names = unwrap_or_default(self.selected, tuple(self.contents))
        if unknown := [name for name in names if name not in self.contents]:
            msg = f"selected {unknown[0]!r}, which the source does not hold"
            raise ValueError(msg)

        object.__setattr__(self, "_taken", frozenset(names))
        object.__setattr__(self, "_recorded", as_json_value(self.settings))

    @property
    def _replacing(self) -> bool:
        """Whether a folder already there may be written over.

        `"reuse"` replaces as readily as `"overwrite"` does: what it keeps it
        keeps by never making a writer for it, so a writer that was made has
        already been told the folder does not describe this run.
        """
        return self.if_present != "error"

    def get_hook(self, source: S) -> KoalaFrameWriter[T] | None:
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

        record = None
        if self.settings is not None:
            record = {"settings": dict(self.settings), "source": source.name}

        dest = Path(self.root, source.name, self.subpath)

        return self._make_writer(dest, source, overwrite=self._replacing, record=record)

    @abstractmethod
    def _make_writer(
        self,
        dest: Path,
        source: S,
        *,
        overwrite: bool,
        record: Mapping[str, object] | None,
    ) -> KoalaFrameWriter[T]:
        """Return the writer that puts `source`'s frames under `dest`.

        Args:
            dest: The folder the frames go to, which is where the source sits
                under this tree's own root.
            source: The sequence the frames come from, for whatever the format
                takes from it that a frame alone does not carry.
            overwrite: Whether a folder already there may be replaced.
            record: The block to file beside the frames, or `None` for none.
        """

    @abstractmethod
    def _expected(self, names: Sequence[str]) -> Sequence[str]:
        """Return the sources of the frames this stage owes for `names`.

        A stage that answers one frame per source returns what it was given. One
        that reads a pair to answer once returns fewer, and saying so is what
        lets a written folder be recognised: the record holds one name per frame
        written, so comparing it against the source's own would refuse every
        folder a stage like that ever wrote.

        Args:
            names: The frames the source holds, in order.
        """

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

        holds_frames = contains(self.subpath, kind="dir")
        found = search_dirs(
            root,
            predicate=holds_frames,
            descend=lambda folder: not holds_frames(folder),
            ordered=False,
        )

        return sorted(folder.relative_to(root).as_posix() for folder in found)

    def _still_describes(self, name: str) -> bool:
        """Whether the folder already written for `name` stands for this run.

        Three things can have moved since it was written, and none of them
        shows in the folder's name: the settings that shaped the frames, which
        frames the source holds by name, and whether the folder still holds all
        of what its record says it does. The third has no counterpart in a part,
        which is one file and so is either there or not; a folder can be half
        removed, and reusing that would leave a short sequence reading as a
        whole one.
        """
        folder = Path(self.root, name, self.subpath)

        try:
            read = (folder / self.record_file).read_text(encoding="utf-8")
            record = json.loads(read)
        except (OSError, ValueError):
            return False

        if not isinstance(record, dict):
            return False

        if record.get("settings") != self._recorded:
            return False

        owed = tuple(self._expected(self.contents[name]))
        frames = record.get("frames")
        if not isinstance(frames, list) or tuple(frames) != owed:
            return False

        return self._count_frames(folder) == len(frames)

    def _count_frames(self, folder: Path) -> int:
        """Count the files the folder holds beside the record it carries.

        Files only, and only the folder's own. Anything else in there is not a
        frame, and counting one would let it stand in for a frame that has
        gone: a folder holding one frame and one directory counts as two, which
        is exactly the number a two-frame record expects.

        Only this tree's own record is set aside. One left under another name
        counts as a frame, which is what stops a folder written by a differently
        named run from reading as a whole one here.
        """
        return len(search_files(folder, max_depth=1, exclude=self.record_file))

    def list_unsourced(self) -> list[str]:
        """Return the sequences this tree holds that the source has lost, sorted.

        A source that looks smaller than it is reads the same from here, so
        acting on the list is `if_unsourced`'s to decide and naming it is not.
        """
        return [name for name in self.list_sequences() if name not in self.contents]

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
            prune_upward(folder.parent, root)
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
            prune_upward(folder.parent, root)

    def report(self) -> str | None:
        """Return one line naming what was kept and removed, or `None` if neither.

        What was written is not counted here: the run's own summary already
        says how many sequences it computed, and the document that indexes the
        frames counts them again. What is left is the two a tree alone knows,
        both of them about folders it did not write this time.
        """
        said = []
        if self._reused:
            kept = quantify(len(self._reused), "sequence")
            said.append(f"kept {kept} already written")
        if self._dropped:
            gone = quantify(len(self._dropped), "folder")
            said.append(f"removed {gone} with no source")

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

        Staging is cleared at both ends, since only the end that runs collects
        it: a run killed outright never reaches the other one, and what it
        staged would otherwise sit here for as long as the tree does. Two runs
        opening one root would take each other's, which nothing here or
        anywhere else in this tree is written to survive.

        Raises:
            FileExistsError: If `if_present` is `"error"` and a sequence this
                run would write already has a folder here.
        """
        self.clear_staging()

        if self.if_present == "reuse":
            here = self._written()
            self._reused.update(name for name in here if self._still_describes(name))
        elif self.if_present == "error" and (written := self._written()):
            sequences = quantify(len(written), "sequence")
            fix = "set `if_present` to 'overwrite' or 'reuse'"
            msg = f"{sequences} already written, from {written[0]!r}: {fix}"
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

        if self.if_unsourced == "delete":
            self.drop_unsourced()


class FrameTree(FrameBranch["PhaseFilteredSequence", "Tensor"]):
    """The frame tree of a phase stage, which answers one frame per source.

    Each writer takes the pixel size, height scale and unit from the sequence it
    was made for, since a phase file carries them and a frame alone does not.

    Attributes:
        root: As `FrameBranch`.
        subpath: As `FrameBranch`.
        contents: As `FrameBranch`.
        settings: As `FrameBranch`.
        selected: As `FrameBranch`.
        record_file: As `FrameBranch`.
        if_present: As `FrameBranch`.
        if_unsourced: As `FrameBranch`.
    """

    __slots__ = ()

    @override
    def _make_writer(
        self,
        dest: Path,
        source: PhaseFilteredSequence,
        *,
        overwrite: bool,
        record: Mapping[str, object] | None,
    ) -> KoalaFrameWriter[Tensor]:
        origin = source.origin
        header = origin.header

        return phase_frame_writer(
            dest,
            pixel_size=header.pixel_size,
            height_scale=header.height_scale,
            unit=unwrap_or_default(origin.target_unit, header.unit),
            overwrite=overwrite,
            record=record,
            record_file=self.record_file,
        )

    @override
    def _expected(self, names: Sequence[str]) -> Sequence[str]:
        """Every source frame, filtering being one frame in and one frame out."""
        return names
