from __future__ import annotations

__all__ = ("RECORD_FILE", "FrameBranch", "FrameWriter")

import json
import shutil
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self

from kaparoo.filesystem import (
    StagedDirectory,
    contains,
    ensure_dir_exists,
    prune_upward,
    search_dirs,
    search_files,
)
from kaparoo.filesystem.types import StrPath
from kaparoo.utils import quantify
from kaparoo.utils.optional import unwrap_or_default

from iivs_cardio.common.pipeline.base import Named, Step
from iivs_cardio.common.pipeline.branch import (
    PRESENT_POLICIES,
    STAGING,
    PresentPolicy,
    UnsourcedPolicy,
    as_json_value,
    ensure_json_name,
    ensure_policy,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

# What a written folder says about itself, filed inside it, unless the caller
# asks for another name. The readers select frames by extension, so a name of
# another kind sits there unnoticed.
RECORD_FILE: Final = "source.json"


class FrameWriter[T, E = Path]:
    """A hook that writes the frames it is given, one file each.

    Frames are numbered from the first that arrives rather than from the
    source, and staged until a clean close. What the renumbering loses,
    `record` keeps.

    Type Parameters:
        T: The type of one frame, as `save_fn` expects it.
        E: The type of what a step says about where its frame came from,
            which `source_fn` reads. Defaults to `Path`.

    Args:
        dest: The folder the finished frames go into.
        save_fn: A function writing one frame into a folder, under the
            number it is given. Naming it is the caller's, since the name a
            format takes and the format itself are one choice.
        source_fn: A function naming where one frame came from, from what its
            step carries. Read only where a `record` is filed.
        overwrite: Whether an existing folder may be replaced. Defaults to
            False.
        record: The block the folder should carry about itself, beside the
            source names the writer collects. Defaults to `None`, which files
            nothing and asks nothing of the steps.
        record_file: The name that block is filed under, given `.json` if it
            has no extension. Defaults to `RECORD_FILE`.

    Raises:
        FileExistsError: If the destination is there and `overwrite` is not set.
        ValueError: If `record_file` carries a directory part.
    """

    def __init__(
        self,
        dest: StrPath,
        save_fn: Callable[[Path, int, T], object],
        source_fn: Callable[[E], str],
        *,
        overwrite: bool = False,
        record: Mapping[str, object] | None = None,
        record_file: str = RECORD_FILE,
    ) -> None:
        record_file = ensure_json_name(record_file)  # before anything is made

        # read before the staging makes them
        self._untouched = next(p for p in Path(dest).parents if p.is_dir())
        self._staged = StagedDirectory(dest, overwrite=overwrite, make_parents=True)

        self._save_fn = save_fn
        self._source_fn = source_fn

        self._record = record
        self._record_file = record_file

        self._sources: list[str] = []
        self._written = 0
        self._last_index: int | None = None

        self._entered = False
        self._committed = False

    def __call__(self, step: Step[T, E]) -> None:
        self.write(step)

    def write(self, step: Step[T, E]) -> None:
        if step.value is None:
            return

        last = self._last_index
        if last is not None and step.index != last + 1:
            msg = f"frame {step.index} does not follow {last}: expected {last + 1}"
            raise ValueError(msg)

        if self._record is not None:
            self._sources.append(self._source_fn(step.require_extra()))

        self._save_fn(self._staged.workdir, self._written, step.value)

        self._written += 1
        self._last_index = step.index

    def report(self) -> str | None:
        if not self._committed:
            return None

        return f"wrote {quantify(self._written, 'frame')}"

    def _save_record(self) -> None:
        if self._record is None:
            return

        document = {**self._record, "sources": self._sources}
        written = json.dumps(document, allow_nan=False)
        (self._staged.workdir / self._record_file).write_text(written, encoding="utf-8")

    def _abort(self) -> None:
        self._staged.abort()

        prune_upward(self._staged.path.parent, self._untouched)

    def __enter__(self) -> Self:
        if self._entered:
            msg = f"{self._staged.path} was opened already: one writer per walk"
            raise RuntimeError(msg)

        self._entered = True

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            self._abort()
            return

        if not self._written:
            self._abort()
            msg = f"no frame was written: nothing to commit at {self._staged.path}"
            raise ValueError(msg)

        try:
            self._save_record()
            self._staged.commit()
        except BaseException:
            self._abort()
            raise

        self._committed = True


@dataclass(frozen=True, slots=True)
class FrameBranch[N: Named, T](ABC):
    """The side branch that writes each sequence back out under a new root.

    A written sequence keeps the name and the layout it had in the source, so
    the result can be read by whatever reads the source.

    A subclass says two things and inherits the rest: how to make the writer
    for one sequence, and how many frames this stage owes that sequence. The
    second is not always as many as the source holds, and a stage that gives
    back fewer would otherwise never reuse anything it wrote.

    Type Parameters:
        N: The sequence a writer is made for, named the way the tree files it.
        T: The type of one frame, as the writer is handed it.

    Attributes:
        root: The directory the tree is written under.
        subpath: The path to a sequence's frames inside its own folder. Empty
            reads back only for a dataset whose names are one level deep, a
            sequence being recognised by holding this.
        contents: Every sequence the source holds, which is what tells a folder
            with no sequence behind it from one this run did not take. An empty
            one would leave every folder here unsourced.
        settings: The settings that shaped the frames, filed inside each
            sequence's folder beside the source names its writer collected.
            Defaults to `None`, which files nothing.
        selected: The sequences of the contents this run was given to write.
            Repeats count once. Defaults to `None`, which takes all of them.
        record_file: The name each written folder keeps its own account under,
            given `.json` if it has no extension. A folder is read back by this
            name too. Defaults to `RECORD_FILE`.
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

    def get_hook(self, source: N) -> FrameWriter[T] | None:
        """Return the writer for `source`, or `None` to keep what is there.

        Whether a folder still stands for this run was settled when the tree
        opened, where the whole dataset was in view; this only looks the answer
        up. A sequence nothing has to write costs no frames at all, which is
        what reuse is for.

        The record names the sequence as the dataset does and leaves out the
        root it sat under, an absolute path not surviving a move.

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
        source: N,
        *,
        overwrite: bool,
        record: Mapping[str, object] | None,
    ) -> FrameWriter[T]:
        """Return the writer that puts `source`'s frames under `dest`.

        Args:
            dest: Where the source sits under this tree's own root.
            source: The sequence the frames come from, for whatever the format
                takes from it that a frame alone does not carry.
            overwrite: Whether a folder already there may be replaced.
            record: The block to file beside the frames, or `None` for none.
        """

    @abstractmethod
    def _expected(self, names: Sequence[str]) -> Sequence[str]:
        """Return the sources of the frames this stage owes for `names`.

        A stage answering one frame per source returns what it was given; one
        reading a pair to answer once returns fewer. The record holds one name
        per frame written, so a stage like that would otherwise never recognise
        a folder it wrote.

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

        Three things can have moved since it was written and none shows in the
        folder's name: the settings that shaped the frames, which frames the
        source holds by name, and whether the folder still holds all its record
        says. A folder can be half removed where a single file cannot.
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
        sources = record.get("sources")
        if not isinstance(sources, list) or tuple(sources) != owed:
            return False

        return self._count_frames(folder) == len(sources)

    def _count_frames(self, folder: Path) -> int:
        """Count the files the folder holds beside the record it carries.

        Files only, and only the folder's own, so nothing else stands in for a
        frame that has gone. Only this tree's own record is set aside: one left
        under another name counts as a frame, which stops a folder a
        differently named run wrote from reading as whole here.
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
            found.
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

        A staged folder sits under a hidden name ending in `.tmp` beside the
        destination it is to become, and the writer undoes it while it is
        alive. Only that shape is taken, and only the folders it leaves empty,
        since a run writes into the directory that keeps its logs too.
        """
        root = Path(self.root)
        if not root.is_dir():
            return

        for folder in search_dirs(root, name_filter=STAGING):
            shutil.rmtree(folder, ignore_errors=True)
            prune_upward(folder.parent, root)

    def report(self) -> str | None:
        """Return one line naming what was kept and removed, or `None` if neither.

        Not what was written, which the run's own summary already counts, but
        the two a tree alone knows, both about folders it did not write.
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

        `"error"` refuses here rather than at the writer that meets the
        folder, which meets them one at a time and so refuses only once the
        sequences before it have been paid for.

        The root is made here rather than left to the first writer, which
        would otherwise count it among the folders it made and take it away
        again when it gave up.

        Raises:
            FileExistsError: If `if_present` is `"error"` and a sequence this
                run would write already has a folder here.
        """
        ensure_dir_exists(self.root, make=True)

        self.clear_staging()

        if self.if_present == "reuse":
            here = self._already_written()
            self._reused.update(name for name in here if self._still_describes(name))
        elif self.if_present == "error" and (written := self._already_written()):
            sequences = quantify(len(written), "sequence")
            fix = "set `if_present` to 'overwrite' or 'reuse'"
            msg = f"{sequences} already written, from {written[0]!r}: {fix}"
            raise FileExistsError(msg)

        return self

    def _already_written(self) -> list[str]:
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
