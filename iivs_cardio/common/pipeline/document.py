from __future__ import annotations

__all__ = (
    "Coverage",
    "DatasetResult",
    "DocumentBranch",
    "ResultWriter",
    "SequenceResult",
    "Sourced",
    "save_document",
)

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Self

from kaparoo.filesystem import (
    StagedFile,
    ensure_dir_exists,
    ensure_file_extension,
    prune_upward,
    reserve_path,
    search_files,
    stringify_path,
)
from kaparoo.filters import EndsWith
from kaparoo.utils import quantify
from kaparoo.utils.optional import unwrap_or_default

from iivs_cardio.common.pipeline.base import Named
from iivs_cardio.common.pipeline.branch import (
    JSON_EXT,
    PRESENT_POLICIES,
    STAGING,
    UNSOURCED_POLICIES,
    PresentPolicy,
    UnsourcedPolicy,
    as_json_value,
    ensure_policy,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence
    from pathlib import Path
    from types import TracebackType

    from kaparoo.filesystem.types import StrPath


class Sourced(Protocol):
    """Whatever a document needs of a measurement: what it was taken from."""

    @property
    def source(self) -> str: ...


class SequenceResult(Sourced, Protocol):
    """Whatever a document needs of one sequence's result.

    Its own source, so a result filed under one name and holding another can be
    caught, and the frames it covers, so a result written before the source
    changed can be told from one that still describes it.
    """

    @property
    def frames(self) -> Sequence[Sourced]: ...

    def to_dict(self) -> dict[str, Any]: ...


class DatasetResult(Protocol):
    """Whatever a document needs of a dataset: the sequences that went into it."""

    @property
    def sequences(self) -> Sequence[Sourced]: ...

    def to_dict(self) -> dict[str, Any]: ...


# ========================== #
#          Coverage          #
# ========================== #


@dataclass(frozen=True, slots=True)
class Coverage:
    """How much of the dataset a document accounts for, and how it got there.

    Measured against the whole dataset rather than against what one run was
    given, since a document may combine results an earlier run left: counting
    against the selection would say `31 of 31` while the numbers came from 121.
    Every sequence the source holds is in exactly one of `covered`, `skipped`
    and `unselected`.

    The two names it misses are kept apart because they call for different
    things. One was given to the run and left no result, which is what a retry is
    built from; the other was never given, which is what `include` and
    `exclude` do and needs nothing.

    Attributes:
        found: How many sequences the source holds.
        selected: How many of those the run was given to cover.
        covered: How many the document has a result for.
        reused: How many of those came from a result the run did not measure.
            Defaults to none of them.
        skipped: The selected names the document has no result for. Defaults to
            no names.
        unselected: The names the run was not given and has no result for.
            Defaults to no names.

    Raises:
        ValueError: If the three groups do not add up to what the source holds,
            if more was selected than found, or if more was reused than
            covered. None is a coverage a run can have had.
    """

    found: int
    selected: int
    covered: int
    reused: int = 0
    skipped: tuple[str, ...] = ()
    unselected: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Refuse a coverage that contradicts itself."""
        accounted = self.covered + len(self.skipped) + len(self.unselected)
        if accounted != self.found:
            counted = f"{self.covered} + {len(self.skipped)} + {len(self.unselected)}"
            msg = f"coverage does not add up: {counted} is not {self.found}"
            raise ValueError(msg)

        if self.selected > self.found:
            msg = f"selected {self.selected} of the {self.found} found"
            raise ValueError(msg)

        if self.reused > self.covered:
            msg = f"reused {self.reused} of the {self.covered} covered"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Return the coverage as plain data, ready to be written as JSON."""
        return {
            "found": self.found,
            "selected": self.selected,
            "covered": self.covered,
            "reused": self.reused,
            "skipped": list(self.skipped),
            "unselected": list(self.unselected),
        }


def save_document(
    path: StrPath,
    dataset: DatasetResult | None,
    *,
    settings: Mapping[str, object] | None = None,
    coverage: Coverage | None = None,
    overwrite: bool = False,
) -> Path:
    """Write one document, with what was combined and how it was made.

    `coverage` comes before `dataset` so that whoever opens the file to read the
    numbers meets the statement of what they cover first.

    Args:
        path: The file to write, given `.json` if it has no extension.
        dataset: The combine the document is written to carry, or `None` when
            nothing was combined. Combining nothing has no numbers to invent, so
            the document then carries what it covers and no more.
        settings: The block a later run would compare to decide whether this
            document still describes it, such as the filter and the frame step.
            Defaults to `None`, which records nothing.
        coverage: The statement of how much of the dataset the combine accounts
            for. Defaults to `None`, which leaves the document silent on it.
        overwrite: Whether an existing document may be replaced. Defaults to
            `False`.

    Returns:
        The path actually written, extension included.

    Raises:
        FileExistsError: If the document is already there and `overwrite` is
            not set.
    """
    path = ensure_file_extension(path, JSON_EXT, add=True)

    document: dict[str, object] = {}
    if settings is not None:
        document["settings"] = dict(settings)
    if coverage is not None:
        document["coverage"] = coverage.to_dict()
    if dataset is not None:
        document["dataset"] = dataset.to_dict()

    with StagedFile(
        path,
        overwrite=overwrite,
        make_parents=True,
        encoding="utf-8",
    ) as file:
        file.write(json.dumps(document, allow_nan=False))

    return path


# ========================== #
#           Meter            #
# ========================== #


class ResultWriter[S: SequenceResult](ABC):
    """Measure one sequence as its frames go by, then write down the result.

    This is the hook a document hands to a sequence. Writing is how the result
    gets home: a sequence may be measured in a worker process of its own, and
    nothing it keeps in memory comes back.

    A close that follows an error writes nothing, so a result on disk always
    stands for a sequence that finished. Another hook of the same sequence
    failing to commit is that same thing seen a moment later, and `revert` is
    how the result goes with it.

    A subclass says one thing and inherits the rest: what the frames it watched
    combine into.

    Type Parameters:
        S: What this sequence's result holds.

    Args:
        root: The folder the result is written into, created if it is not there.
        source: The name the sequence has, used both in the record and as the
            name of the file it is written to.
        settings: The settings that shaped the numbers, written into the result so
            it can be told from one an earlier run left under different ones.
            The document carries the same block, and a result outliving the
            document is the case that needs its own copy. Defaults to `None`,
            which records nothing and so can never be reused.
        overwrite: Whether a result already filed under `source` may be replaced.
            Its own run clears the folder on the way in, so one that is there
            belongs to something else: two sequences whose names came out the
            same, most likely, which is a mistake rather than a second attempt.
            Defaults to `False`.
    """

    def __init__(
        self,
        root: StrPath,
        source: str,
        settings: Mapping[str, object] | None = None,
        *,
        overwrite: bool = False,
    ) -> None:
        root = ensure_dir_exists(root, make=True)

        self._path = root / f"{source}{JSON_EXT}"
        self._source = source
        self._settings = settings
        self._overwrite = overwrite
        self._saved = False

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._source!r})"

    @abstractmethod
    def _result(self) -> S:
        """Combine what has been measured so far into this sequence's result.

        Raises:
            ValueError: If nothing has been measured, since a result standing for
                a sequence that said nothing would count as covered.
        """

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Write the sequence's result, unless the sequence ended in an error.

        Raises:
            FileExistsError: If a result is already filed under this name and
                this writer was not told it may replace it.
        """
        if exc_type is not None:
            return

        document: dict[str, object] = {}
        if self._settings is not None:
            document["settings"] = dict(self._settings)
        document |= self._result().to_dict()

        with StagedFile(
            self._path,
            overwrite=self._overwrite,
            make_parents=True,
            encoding="utf-8",
        ) as file:
            file.write(json.dumps(document, allow_nan=False))

        self._saved = True

    def revert(self) -> None:
        """Take back the result, if this writer got as far as writing one.

        A result on disk stands for a sequence that finished, and one whose other
        outputs could not be committed did not. Taking it back is what puts the
        sequence back among the skipped rather than leaving the document
        counting it as covered while its frames are nowhere.
        """
        if not self._saved:
            return

        self._path.unlink(missing_ok=True)
        self._saved = False


# ========================== #
#           Branch           #
# ========================== #


class DocumentBranch[N: Named, S: SequenceResult, D: DatasetResult, W](ABC):
    """The side branch that gathers a dataset's results into one document.

    It hands each sequence a writer, and each writer leaves its own result in a
    folder beside the document. Closing it combines the results that belong to the
    dataset and writes the document, together with a statement of how much of
    that dataset those results account for.

    Since the results are read off disk rather than passed back, it does not
    matter which process measured them.

    A subclass says four things and inherits the rest: what writer to hand a
    sequence, how to read one result back, how to combine them, and how many values
    this stage owes a sequence. The last is not always as many as the source
    holds, and a stage that gives back fewer would otherwise never reuse a result
    it wrote.

    Type Parameters:
        N: The thing a writer is made for, which is one sequence of a dataset,
            named the way the document files it.
        S: What one sequence's result holds, once read back.
        D: What the results combine into.
        W: The writer itself, as the branch hands it out.

    Args:
        path: The file to write the document to, given `.json` if it has none.
        source: The dataset root the run read, recorded so two documents can be
            told apart before anyone merges them.
        contents: Every sequence the source holds, each mapped to the frames it
            would be measured over. The whole dataset rather than the run's own
            selection, since a document may combine results an earlier run left and
            coverage counted against the selection would call that complete.
        settings: The block a later run would compare against this one. Defaults
            to `None`, which records nothing and so can never be reused.
        selected: The sequences of the contents this run was given to cover.
            Repeats count once. Defaults to `None`, which takes all of them.
        if_present: The policy for a sequence that already has a result here.
            Defaults to `"error"`.
        if_unsourced: The policy for a result whose sequence the source has lost.
            Defaults to `"keep"`.

    Attributes:
        RESULTS_SUFFIX: What the folder of results beside the document is called.
        path: The document itself, extension included.
        results_root: The folder the results are written into.
        source: As given.
        contents: As given, each sequence's frames kept as a tuple.
        settings: As given.
        selected: As given, with repeats dropped.
        if_present: As given.
        if_unsourced: As given.

    Raises:
        ValueError: If `if_present` or `if_unsourced` is not a policy a document
            offers, if `contents` is empty, since coverage would then have nothing
            to be measured against, or if `selected` names something the contents
            does not hold.
    """

    RESULTS_SUFFIX = ".results"

    def __init__(
        self,
        path: StrPath,
        source: str,
        contents: Mapping[str, Sequence[str]],
        settings: Mapping[str, object] | None = None,
        *,
        selected: Sequence[str] | None = None,
        if_present: PresentPolicy = "error",
        if_unsourced: UnsourcedPolicy = "keep",
    ) -> None:
        self.if_present = ensure_policy(if_present, PRESENT_POLICIES, "if_present")
        self.if_unsourced = ensure_policy(
            if_unsourced, UNSOURCED_POLICIES, "if_unsourced"
        )

        if not contents:
            msg = "no sequence to cover: `contents` must hold at least one"
            raise ValueError(msg)

        self.path = ensure_file_extension(path, JSON_EXT, add=True)
        self.results_root = self.path.with_suffix(self.RESULTS_SUFFIX)

        self.source = source
        self.contents = {name: tuple(frames) for name, frames in contents.items()}
        self.settings = settings

        names = unwrap_or_default(selected, tuple(self.contents))
        self.selected = tuple(dict.fromkeys(names))

        if unknown := [n for n in self.selected if n not in self.contents]:
            msg = f"selected {unknown[0]!r}, which the source does not hold"
            raise ValueError(msg)

        self._recorded = as_json_value(settings)
        self._entered = False
        self._reused: frozenset[str] = frozenset()
        self._saved: D | None = None
        self._written: Path | None = None

    @abstractmethod
    def _make_writer(self, source: N) -> W:
        """Return the writer that will measure `source` and leave its own result.

        Called only for a sequence this run has to measure, so nothing here has
        to ask again whether it does.

        Args:
            source: The sequence the writer is to be made for.
        """

    @abstractmethod
    def _parse(self, document: Mapping[str, Any]) -> S:
        """Return what one result holds, read back off disk.

        Raises:
            ValueError: If the document is not one this branch wrote.
        """

    @abstractmethod
    def _combine(self, results: tuple[S, ...]) -> D:
        """Combine the results of this dataset into the value the document carries.

        Args:
            results: What each sequence left, ordered by the sequence it belongs
                to and never empty: combining nothing is answered before it
                gets here.
        """

    @abstractmethod
    def _expected(self, names: Sequence[str]) -> Sequence[str]:
        """Return the sources this stage owes for a sequence holding `names`.

        The same question the frame branch of a stage answers, and for the same
        reason: a stage reading a pair to answer once owes one fewer than it was
        given, and its results would otherwise never be found to still describe
        the run that wrote them.

        Args:
            names: The frames the source holds, in order.
        """

    @property
    def found(self) -> int:
        """How many sequences the source holds."""
        return len(self.contents)

    @property
    def _replacing(self) -> bool:
        """Whether a document or result already there may be written over."""
        return self.if_present != "error"

    def get_hook(self, source: N) -> W | None:
        """Return the writer that will measure `source`, or `None` to reuse.

        Whether a result still stands for this run was settled when the document
        opened, where the whole dataset was in view; this only looks the answer
        up. A sequence nothing has to measure costs no frames at all, which is
        what reuse is for.

        Returns:
            The writer, filed under the sequence's name, or `None` when a result
            already there was found to still describe this run.
        """
        if source.name in self._reused:
            return None

        return self._make_writer(source)

    def list_results(self) -> list[Path]:
        """Return every result on disk, ordered by the sequence it belongs to."""
        results = search_files(
            self.results_root,
            name_filter=EndsWith(JSON_EXT),
            ordered=False,
        )

        return sorted(results, key=self._source_of)

    def _list_staging(self) -> list[Path]:
        """Return the staging files an interrupted run left among the results.

        A result is written beside its destination under a hidden name ending in
        `.tmp` and moved into place on a clean close, so anything of that shape
        still here belongs to a run that never got to close. Nothing else
        collects them: they are hidden from `list_results`, and the only other
        hand on them dies with the process that staged them.
        """
        return search_files(self.results_root, name_filter=STAGING)

    def _drop(self, results: Iterable[Path]) -> None:
        """Remove `results`, and the folders their removal leaves empty."""
        emptied = set()

        for result in results:
            emptied.add(result.parent)
            result.unlink()

        for folder in emptied:
            prune_upward(folder, self.results_root)

    def _still_describes(self, document: Mapping[str, Any], result: S) -> bool:
        """Whether a result on disk stands for what this run would measure.

        Two things can have moved since it was written, and neither shows in
        the result's own name: the settings that shaped its numbers, and which
        frames the source holds by name. A result failing either is stale rather
        than broken, so it is passed over rather than refused.

        Args:
            document: The result as it was read, for the settings it records.
            result: What that result holds. Its source is one the contents holds,
                which `_read_valid` has established by the time it asks.
        """
        if document.get("settings") != self._recorded:
            return False

        listed = self._expected(self.contents[result.source])

        return tuple(frame.source for frame in result.frames) == tuple(listed)

    def _read_result(self, result: Path) -> dict[str, Any]:
        """Read one result off disk, refusing anything that is not a mapping."""
        with result.open(encoding="utf-8") as file:
            document = json.load(file)

        if not isinstance(document, dict):
            msg = f"malformed document: {type(document).__name__} at the top"
            raise ValueError(msg)  # noqa: TRY004

        return document

    def _source_of(self, result: Path) -> str:
        """Return the sequence a result belongs to, read off where it sits."""
        return stringify_path(result.with_suffix(""), after=self.results_root)

    def _read_valid(self, *, strict: bool) -> Iterator[tuple[Path, S]]:
        """Yield each result that still stands for this run, with what it holds.

        The one place a result is judged, so opening and closing cannot come to
        different answers. A result reused on the way in that the combine then
        passed over would leave a sequence nothing measured and nothing counted.

        Args:
            strict: Whether a result that cannot be read, or that is filed under
                a sequence other than the one it holds, stops the run. Judging
                is not reading, so opening passes over such a result and only the
                combine refuses it.

        Yields:
            Each result and what it holds, in the order `list_results` gives.

        Raises:
            ValueError: Under `strict`, if a result cannot be read or is filed
                under the wrong sequence. The result is named, since the folder
                holds one file per sequence and only the name says which to go
                and look at.
        """
        for result in self.list_results():
            name = self._source_of(result)
            if name not in self.contents:
                continue

            try:
                document = self._read_result(result)
                held = self._parse(document)
            except (OSError, ValueError) as error:
                if not strict:
                    continue
                msg = f"unreadable result {name!r}: {error}. Remove it, or run it again"
                raise ValueError(msg) from error

            if name != held.source:
                if not strict:
                    continue
                msg = f"result {name!r} holds {held.source!r}: run {name!r} again"
                raise ValueError(msg)

            if self._still_describes(document, held):
                yield result, held

    def to_dataset(self, *, strict: bool = True) -> D | None:
        """Combine every result on disk into the one value the document carries.

        Only the results of sequences the source still holds, and only those a run
        with these settings could have written. One left by a run whose dataset
        was larger describes a sequence that is not there, and one left under a
        different filter describes numbers this run would not produce; combining
        either would move the answer by something nothing here accounts for.
        Both stay on disk, since a source that looks smaller than it is makes
        exactly the same absence as one that shrank.

        A result is filed under the sequence it belongs to and says so again
        inside, and the two must agree. Nothing else compares them, so a result
        that disagrees would be sorted under one name and counted under
        another, which no number in the finished document would show.

        Args:
            strict: Whether a result that cannot be read stops the combine. `False`
                passes over it, which leaves its sequence out of the dataset and
                so among the coverage's `skipped`, where a retry will find it.
                Defaults to `True`.

        Returns:
            The combine, or `None` when no result is there. A run whose sequences
            all failed has nothing to combine, and that is what `coverage` is for:
            the document says it covers none of them rather than not being
            written at all.

        Raises:
            ValueError: Under `strict`, if one of the results cannot be read, or
                one is filed under a sequence other than the one it holds. A
                result that cannot be read is named, since the folder holds one
                file per sequence and only the name says which to go and look
                at.
        """
        results = tuple(held for _, held in self._read_valid(strict=strict))
        if not results:
            return None

        return self._combine(results)

    def get_coverage(self, dataset: D | None) -> Coverage:
        """Measure `dataset` against the whole dataset this document describes.

        What is missing is worked out from the lists rather than reported by
        the run, so a sequence counts as missing whether it failed, went down
        with its worker, or was never given to this run at all. Which of those
        it was is what the two lists keep apart.

        Every number is read off the contents, so the three groups always add up
        to it. Counting one from the contents and another from disk let the two
        disagree.
        """
        combined = set() if dataset is None else {s.source for s in dataset.sequences}
        given = set(self.selected)

        missing = [name for name in self.contents if name not in combined]

        selected = len(self.selected)
        covered = sum(name in combined for name in self.contents)
        reused = sum(name in combined for name in self._reused)
        skipped = tuple(name for name in self.selected if name not in combined)
        unselected = tuple(name for name in missing if name not in given)

        return Coverage(self.found, selected, covered, reused, skipped, unselected)

    def save(self, *, strict: bool = True) -> Path:
        """Combine the results and write the document, coverage included.

        What was combined is remembered once the file is on disk and not before,
        so a write that failed leaves this branch with nothing to report rather
        than a line about a document that is not there.

        Args:
            strict: Whether a result that cannot be read stops the combine, as for
                `to_dataset`. Defaults to `True`.

        Returns:
            The path actually written, extension included.

        Raises:
            ValueError: Under `strict`, if one of the results cannot be read.
            FileExistsError: If the document is already there and this one was
                not told it may replace it.
        """
        dataset = self.to_dataset(strict=strict)

        written = save_document(
            self.path,
            dataset,
            settings=self.settings,
            coverage=self.get_coverage(dataset),
            overwrite=self._replacing,
        )
        self._saved = dataset
        self._written = written

        return written

    def report(self) -> str | None:
        """Return one line naming what was written, or `None` before it was.

        The line counts against the dataset rather than against what this run
        was given, so a document combined over result of one cannot be mistaken for
        one combined over all of it. What is missing is split the way `coverage`
        splits it, since a sequence that failed and one nobody asked for call
        for different things. A document that covered none has nothing combined
        to name and says only what it covers.
        """
        if self._written is None:
            return None

        dataset = self._saved
        coverage = self.get_coverage(dataset)

        sequences = quantify(coverage.found, "sequence")
        if coverage.covered != coverage.found:
            sequences = f"{coverage.covered} of {sequences}"

        if coverage.reused:
            sequences = f"{sequences}, {coverage.reused} reused"

        if coverage.skipped:
            sequences = f"{sequences}, {len(coverage.skipped)} skipped"

        if coverage.unselected:
            sequences = f"{sequences}, {len(coverage.unselected)} not taken"

        if dataset is None:
            return f"wrote {self.path.name} from {sequences}"

        return f"wrote {self.path.name} from {sequences}: {dataset}"

    def __enter__(self) -> Self:
        """Take the document's place, then settle what an earlier run left.

        In that order, since what follows is not undoable: the document is
        written once every sequence has run, so a run that may not replace it
        would otherwise pay for the whole dataset and be refused at the end,
        having already dropped the results an earlier run left behind.

        What an earlier run staged and never committed always goes, since
        nothing else is in a position to collect it. What it committed depends
        on the policy: `"reuse"` keeps every result still describing this run and
        leaves the rest where they are, `"overwrite"` clears the folder so that
        everything combined at the end is this run's own, and `"error"` refuses.

        `"error"` refuses here rather than leaving it to the writer that meets
        the result, the way the frame tree does with a folder: a writer meets them
        one at a time, so a run whose hundredth sequence is already measured
        pays for ninety-nine of them first. Refusing is also what a run killed
        outright leaves behind, since its results are committed and its document
        is not, and clearing them would spend its whole measurement again
        without saying so.

        Judging happens here, with the whole dataset in view and in one process.
        A worker holds a copy of this branch and nothing it learns comes home,
        so a result judged there could not be counted.

        Raises:
            FileExistsError: If the document is already there, or a sequence
                already has a result here, and this one was not told it may
                replace them.
            RuntimeError: If this document has been opened before.
        """
        if self._entered:
            msg = f"{self.path.name} was opened already: open it once per run"
            raise RuntimeError(msg)

        self._entered = True
        reserve_path(self.path, exist_ok=self._replacing, make_parents=True)

        ensure_dir_exists(self.results_root, make=True)

        self._drop(self._list_staging())

        if self.if_present == "reuse":
            kept = (result for result, _ in self._read_valid(strict=False))
            self._reused = frozenset(map(self._source_of, kept))
        elif present := self.list_results():
            if self.if_present == "error":
                left = quantify(len(present), "result")
                first = self._source_of(present[0])
                fix = "set `if_present` to 'reuse' or 'overwrite'"
                msg = f"{left} already here, from {first!r}: {fix}"
                raise FileExistsError(msg)

            self._drop(present)

        return self

    def list_unsourced(self) -> list[str]:
        """Return the sequences a result is filed under that the source has lost.

        Named rather than acted on: the same absence is what a half mounted
        share and a misspelt subpath produce, so what to do with them is the
        caller's policy and saying they are there is not.
        """
        filed = map(self._source_of, self.list_results())

        return [name for name in filed if name not in self.contents]

    def drop_unsourced(self) -> list[str]:
        """Remove the results of sequences the source has lost, and name them.

        The folders a removal empties go with it, the same way opening prunes
        the ones it emptied: a sequence dropped from a nested dataset would
        otherwise leave the path down to it standing.

        Returns:
            The sequences whose results were removed, in the order they were
            filed under.
        """
        dropped = []

        for name in self.list_unsourced():
            result = self.results_root / f"{name}{JSON_EXT}"
            result.unlink()
            prune_upward(result.parent, self.results_root)
            dropped.append(name)

        return dropped

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Write the document, whether or not the run reached the end.

        A run that gave up part way still measured what it measured, and the
        results it left are of sequences that finished. Writing them is what
        `coverage` is for: the document says which of the dataset it accounts
        for and names the rest, where refusing to write leaves the healthy
        results on disk with nothing to read them by.

        A result that cannot be read is the one thing that could take the whole
        document with it, since the combine refuses such a result rather than
        passing it over. It is written from what does read instead, which
        leaves that sequence out of the dataset and so among the coverage's
        `skipped`, where a retry will find it, and the refusal is raised once
        the document is on disk rather than instead of it.

        Parts of sequences the source has lost go afterwards where the policy
        says so. The combine passes over them either way, so removing them is
        tidying rather than part of the answer, and one that cannot be removed
        must not cost the document.

        The failure itself is not this branch's to report. It reaches the
        driver, which is what decides the run's verdict.

        Raises:
            ValueError: If one of the results cannot be read, after the document
                combined from the rest has been written.
        """
        try:
            self.save()
        except ValueError:
            self.save(strict=False)
            raise

        if self.if_unsourced == "delete":
            self.drop_unsourced()
