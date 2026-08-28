from __future__ import annotations

__all__ = ("RECORD_FILE", "FrameBranch", "FrameWriter")

import json
import shutil
from abc import ABC, abstractmethod
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
from kaparoo.utils import quantify
from kaparoo.utils.optional import unwrap_or_default

from iivs_cardio.common.pipeline.base import Named, Step
from iivs_cardio.common.pipeline.branch import (
    PRESENT_POLICIES,
    STAGING,
    UNSOURCED_POLICIES,
    PresentPolicy,
    UnsourcedPolicy,
    as_json_value,
    ensure_json_name,
    ensure_policy,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from types import TracebackType

    from kaparoo.filesystem.types import StrPath


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
        """Write `step`, so the writer can be registered as a hook directly."""
        self.write(step)

    def write(self, step: Step[T, E]) -> None:
        """Write the frame in `step`, numbered after the last one written.

        A step carrying no frame is passed over, which is what lets a sequence
        start late or end early.

        Raises:
            ValueError: If a frame does not follow the one before it at the
                source, since the numbering here would close the gap.
        """
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
        """Return one line naming how many frames landed, or `None` before any did.

        Nothing is reported until the folder reaches its destination, since a
        sequence that gave up has none to point at however many it staged.
        """
        if not self._committed:
            return None

        return f"wrote {quantify(self._written, 'frame')}"

    def _save_record(self) -> None:
        """Write what the folder says about itself, into the staged folder.

        Into the staged one, so the move that makes the frames visible makes
        this visible with them.
        """
        if self._record is None:
            return

        document = {**self._record, "sources": self._sources}
        written = json.dumps(document, allow_nan=False)
        (self._staged.workdir / self._record_file).write_text(written, encoding="utf-8")

    def _abort(self) -> None:
        """Drop the staged folder, and the empty ones opening it made.

        Left behind, they put an empty sequence in the output tree. The climb
        stops below what was already standing, and at the first folder
        something else landed in meanwhile.
        """
        self._staged.abort()

        prune_upward(self._staged.path.parent, self._untouched)

    def __enter__(self) -> Self:
        """Take the writer, refusing one that has been through a walk already.

        Raises:
            RuntimeError: If it has been opened before. Closing takes the
                staged folder away, so a second walk writes where nothing is.
        """
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
        """Move the folder into place, unless nothing was written or it failed.

        A move that fails takes the staged folder with it, the only other
        reference to it dying with the process.

        Raises:
            ValueError: If the sequence ended without a single frame, an empty
                folder reading as a finished one.
        """
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
        selected: The sequences of the contents this run was given to write,
            repeats counted once. Taking all of them when `None` is given.
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
            nested, if `if_present` or `if_unsourced` is not a policy a tree
            offers, if `record_file` carries a directory part, or if `selected`
            names something the contents does not hold.
    """

    def __init__(
        self,
        root: StrPath,
        subpath: str,
        contents: Mapping[str, Sequence[str]],
        settings: Mapping[str, object] | None = None,
        *,
        selected: Sequence[str] | None = None,
        record_file: str = RECORD_FILE,
        if_present: PresentPolicy = "error",
        if_unsourced: UnsourcedPolicy = "keep",
    ) -> None:
        self.if_present = ensure_policy(if_present, PRESENT_POLICIES, "if_present")
        self.if_unsourced = ensure_policy(
            if_unsourced, UNSOURCED_POLICIES, "if_unsourced"
        )

        if not subpath and any("/" in name for name in contents):
            fix = "give the branch a `subpath`"
            msg = f"a nested dataset cannot be found again without a layout: {fix}"
            raise ValueError(msg)

        self.root = Path(root)
        self.subpath = subpath
        self.contents = {name: tuple(frames) for name, frames in contents.items()}
        self.settings = settings

        self.record_file = ensure_json_name(record_file)

        names = unwrap_or_default(selected, tuple(self.contents))
        if unknown := [name for name in names if name not in self.contents]:
            msg = f"selected {unknown[0]!r}, which the source does not hold"
            raise ValueError(msg)
        self.selected = tuple(dict.fromkeys(names))

        # settled here, read from every sequence
        self._wanted = frozenset(self.selected)
        self._recorded = as_json_value(settings)

        # what the run leaves behind, which `report` counts
        self._reused: set[str] = set()
        self._dropped: list[str] = []

    @property
    def _replacing(self) -> bool:
        """Whether a folder already there may be written over.

        `"reuse"` replaces as readily as `"overwrite"`: what it keeps it keeps
        by never making a writer for it.
        """
        return self.if_present != "error"

    def get_hook(self, source: N) -> FrameWriter[T] | None:
        """Return the writer for `source`, or `None` to keep what is there.

        Whether a folder still stands for this run was settled when the tree
        opened; this only looks the answer up. The record names the sequence as
        the dataset does and leaves out the root, an absolute path not
        surviving a move.
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
            source: The sequence the frames come from, for whatever the format
                takes from it that a frame alone does not carry.
        """

    @abstractmethod
    def _expected(self, names: Sequence[str]) -> Sequence[str]:
        """Return the sources of the frames this stage owes for `names`.

        A stage answering one frame per source returns what it was given; one
        reading a pair to answer once returns fewer, and saying so is what lets
        a folder it wrote be recognised again.
        """

    def list_sequences(self) -> list[str]:
        """Return every sequence this tree already holds frames for, sorted.

        A sequence is recognised by holding `subpath` rather than by the walk
        reaching it, so nothing below one is ever listed.
        """
        if not self.root.is_dir():
            return []

        holds_frames = contains(self.subpath, kind="dir")
        found = search_dirs(
            self.root,
            predicate=holds_frames,
            descend=lambda folder: not holds_frames(folder),
            ordered=False,
        )

        return sorted(folder.relative_to(self.root).as_posix() for folder in found)

    def _still_describes(self, name: str) -> bool:
        """Whether the folder already written for `name` stands for this run.

        Three things can have moved and none shows in the folder's name: the
        settings, which frames the source holds by name, and whether the folder
        still holds all its record says.
        """
        folder = Path(self.root, name, self.subpath)

        try:
            text = (folder / self.record_file).read_text(encoding="utf-8")
            record = json.loads(text)
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

        Only this tree's own record is set aside: one left under another name
        counts as a frame, which stops a folder another run wrote from reading
        as whole here.
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
        """
        dropped = []

        for name in self.list_unsourced():
            folder = self.root / name
            shutil.rmtree(folder)
            prune_upward(folder.parent, self.root)
            dropped.append(name)

        self._dropped.extend(dropped)

        return dropped

    def clear_staging(self) -> None:
        """Drop what a writer that did not live to clear up after itself left.

        Only that shape is taken, a hidden name ending in `.tmp`, and only the
        folders it leaves empty, since a run writes into the directory that
        keeps its logs too.
        """
        if not self.root.is_dir():
            return

        for folder in search_dirs(self.root, name_filter=STAGING):
            shutil.rmtree(folder, ignore_errors=True)
            prune_upward(folder.parent, self.root)

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
        process: a worker holds a copy of this branch and nothing it learns
        comes home. The root is made here too, which a writer would otherwise
        count among the folders it made and take away when it gave up.

        Raises:
            FileExistsError: If `if_present` is `"error"` and a sequence this
                run would write already has a folder here. Refused here rather
                than at the writer, which meets them one at a time.
        """
        ensure_dir_exists(self.root, make=True)

        self.clear_staging()

        if self.if_present == "reuse":
            written = self._already_written()
            self._reused.update(n for n in written if self._still_describes(n))
        elif self.if_present == "error" and (written := self._already_written()):
            sequences = quantify(len(written), "sequence")
            fix = "set `if_present` to 'overwrite' or 'reuse'"
            msg = f"{sequences} already written, from {written[0]!r}: {fix}"
            raise FileExistsError(msg)

        return self

    def _already_written(self) -> list[str]:
        """Return the sequences this run would write that already have a folder."""
        return [name for name in self.list_sequences() if name in self._wanted]

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
