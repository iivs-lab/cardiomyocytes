from __future__ import annotations

__all__ = (
    "DOCUMENT_EXT",
    "CompositeRange",
    "Coverage",
    "DatasetRange",
    "FrameRange",
    "Named",
    "RangeDocument",
    "SequenceRange",
    "SequenceRangeMeter",
    "ValueRange",
    "save_range_document",
)

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import TYPE_CHECKING, Any, Final, Protocol, Self, override

from kaparoo.filesystem import (
    StagedFile,
    ensure_dir_exists,
    ensure_file_extension,
    prune_upward,
    reserve_path,
    search_files,
    stringify_path,
)
from kaparoo.filters import And, EndsWith, StartsWith
from kaparoo.utils import quantify
from kaparoo.utils.optional import unwrap_or_default

from iivs_cardio.common.pipeline.branch import (
    as_read_back,
    find_unsourced,
)
from iivs_cardio.common.range import finite_range

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path
    from types import TracebackType

    from kaparoo.filesystem.types import StrPath
    from torch import Tensor

    from iivs_cardio.common.pipeline import Step
    from iivs_cardio.common.pipeline.branch import (
        ExistingOutputPolicy,
        UnsourcedOutputPolicy,
    )


DOCUMENT_EXT: Final = ".json"


def _entry[T](
    document: Mapping[str, Any],
    key: str,
    kind: type[T] | tuple[type[T], ...],
) -> T:
    """Read `key` from a document, refusing it by name when it cannot be read.

    Absent and wrong type are one rejection: a document read back off disk is
    just data either way, and neither makes it a range document.
    """
    value = document.get(key)
    if not isinstance(value, kind):
        msg = f"malformed range document: {key!r} is {value!r}"
        raise ValueError(msg)  # noqa: TRY004

    return value


def _number(document: Mapping[str, Any], key: str) -> float:
    """Read `key` from a document as a bound, refusing what only looks like one.

    `bool` is an `int` to `isinstance`, so `true` would otherwise read as 1.0
    and a pair of them as a range running backwards. A non-finite bound is
    refused here rather than folded: `min` and `max` carry a NaN through or
    drop it depending on where it sits, so one that got in would fold to
    whatever the order of the parts happened to be.

    Raises:
        ValueError: If the value is absent, not a number, or not finite.
    """
    value = _entry(document, key, (int, float))
    if isinstance(value, bool) or not isfinite(value):
        msg = f"malformed range document: {key!r} is {value!r}"
        raise ValueError(msg)

    return float(value)


# ========================== #
#           Ranges           #
# ========================== #


@dataclass(frozen=True, slots=True)
class ValueRange(ABC):
    """The lowest and highest value found in something, and what that was.

    Attributes:
        source: The thing the range was measured over, named the way a reader
            of the document would look it up.
        min_value: The lowest value found.
        max_value: The highest value found.
    """

    source: str
    min_value: float
    max_value: float

    def __post_init__(self) -> None:
        """Refuse a range whose two ends are the wrong way round.

        A folded range takes each end from the part that holds it, so it cannot
        invert once its parts are this way up.

        Raises:
            ValueError: If the lowest value is above the highest.
        """
        if self.min_value > self.max_value:
            msg = f"inverted range in {self.source!r}: {self} runs backwards"
            raise ValueError(msg)

    def __str__(self) -> str:
        """The two bounds, shortened for reading rather than for reloading."""
        return f"[{self.min_value:.4g}, {self.max_value:.4g}]"

    def to_dict(self) -> dict[str, Any]:
        """Return the range as plain data, ready to be written as JSON."""
        return asdict(self)

    @classmethod
    @abstractmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Self:
        """Rebuild a range from what `to_dict` produced.

        Raises:
            ValueError: If a key the range needs is absent or unreadable.
        """
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class FrameRange(ValueRange):
    """The range of one frame.

    Attributes:
        source: The file the frame was read from, which is the name it has at
            the source and not necessarily the one a cache of the same run
            gives it.
        min_value: The lowest value in the frame.
        max_value: The highest value in the frame.
    """

    @override
    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Self:
        """Rebuild a frame range from `source`, `min_value` and `max_value`.

        Raises:
            ValueError: If any of the three is absent or unreadable, or if the
                two bounds run backwards.
        """
        return cls(
            _entry(document, "source", str),
            _number(document, "min_value"),
            _number(document, "max_value"),
        )


@dataclass(frozen=True, slots=True)
class CompositeRange(ValueRange, ABC):
    """A range folded from smaller ones, keeping where each end came from.

    The bounds are not given but taken from the parts, and each is remembered
    with the part it came from, so a wide dataset range can be traced back to
    the one sequence or frame that widened it.

    Attributes:
        source: The thing the folded range was measured over.
        min_value: The lowest value across every part.
        max_value: The highest value across every part.
        min_index: The position of the part holding the lowest value.
        max_index: The position of the part holding the highest value.

    Raises:
        ValueError: If there are no parts, since a range over nothing has no
            meaning to fall back on.
    """

    min_value: float = field(init=False)
    max_value: float = field(init=False)
    min_index: int = field(init=False)
    max_index: int = field(init=False)

    @property
    @abstractmethod
    def parts(self) -> Sequence[ValueRange]:
        """The ranges this one is folded from, in the order they were taken."""
        raise NotImplementedError

    def __len__(self) -> int:
        """The number of ranges folded here."""
        return len(self.parts)

    def __post_init__(self) -> None:
        """Take the bounds from the parts, and note which part gave each."""
        parts = self.parts
        if not parts:
            msg = f"value range is undefined: {type(self).__name__} holds nothing"
            raise ValueError(msg)

        indices = range(len(parts))
        min_index = min(indices, key=lambda i: parts[i].min_value)
        max_index = max(indices, key=lambda i: parts[i].max_value)

        object.__setattr__(self, "min_value", parts[min_index].min_value)
        object.__setattr__(self, "max_value", parts[max_index].max_value)
        object.__setattr__(self, "min_index", min_index)
        object.__setattr__(self, "max_index", max_index)


@dataclass(frozen=True, slots=True)
class SequenceRange(CompositeRange):
    """The range of one sequence, folded from the frames it was measured over.

    Position is the key, not the name. Each frame is filed under the source it
    was read from, while a cache the same run writes numbers its frames from
    zero without a gap, so the two disagree wherever the run read the source
    with a stride or the source itself was sparse. The nth entry here is the
    nth frame either way.

    Attributes:
        source: The name the sequence has in its dataset.
        min_value: The lowest value across every frame.
        max_value: The highest value across every frame.
        min_index: The position of the frame holding the lowest value.
        max_index: The position of the frame holding the highest value.
        frames: The range of each frame, in the order they were read, which is
            the order a cache of the same run writes them in.
    """

    frames: tuple[FrameRange, ...]

    @property
    @override
    def parts(self) -> tuple[FrameRange, ...]:
        """The frame ranges this sequence is folded from."""
        return self.frames

    @classmethod
    @override
    def from_dict(cls, document: Mapping[str, Any]) -> SequenceRange:
        """Rebuild a sequence range from its `source` and its `frames`.

        Raises:
            ValueError: If either key is absent, or a frame cannot be read.
        """
        source = _entry(document, "source", str)
        frames = _entry(document, "frames", (list, tuple))
        frames = tuple(FrameRange.from_dict(frame) for frame in frames)
        return cls(source, frames)


@dataclass(frozen=True, slots=True)
class DatasetRange(CompositeRange):
    """The range of a whole dataset, folded from the sequences it covers.

    Attributes:
        source: The dataset root the run read, which is what tells two
            documents apart when someone comes to merge them.
        min_value: The lowest value across every sequence.
        max_value: The highest value across every sequence.
        min_index: The position of the sequence holding the lowest value.
        max_index: The position of the sequence holding the highest value.
        sequences: The range of each sequence, in the order they were folded.
    """

    sequences: tuple[SequenceRange, ...]

    @property
    @override
    def parts(self) -> tuple[SequenceRange, ...]:
        """The sequence ranges this dataset is folded from."""
        return self.sequences

    @classmethod
    @override
    def from_dict(cls, document: Mapping[str, Any]) -> DatasetRange:
        """Rebuild a dataset range from its `source` and its `sequences`.

        Raises:
            ValueError: If either key is absent, or a sequence cannot be read.
        """
        source = _entry(document, "source", str)
        sequences = _entry(document, "sequences", (list, tuple))
        sequences = tuple(SequenceRange.from_dict(sequence) for sequence in sequences)
        return cls(source, sequences)


# ========================== #
#         Measuring          #
# ========================== #


class SequenceRangeMeter:
    """Measure the range of every frame of one sequence, then write the result.

    This is the hook a range document hands to a sequence. It records a range
    per frame as the frames go by, and on a clean close writes them beside the
    document as that sequence's part. Writing is how the result gets home: a
    sequence may be measured in a worker process of its own, and nothing it
    keeps in memory comes back.

    A close that follows an error writes nothing, so a part on disk always
    stands for a sequence that finished. Another hook of the same sequence
    failing to commit is that same thing seen a moment later, and `revert` is
    how the part goes with it.

    Args:
        root: The folder the part is written into, created if it is not there.
        source: The name the sequence has, used both in the record and as the
            name of the file it is written to.
        settings: The settings that shaped the numbers, written into the part so
            it can be told from one an earlier run left under different ones.
            The document carries the same block, and a part outliving the
            document is the case that needs its own copy. Defaults to `None`,
            which records nothing and so can never be reused.
        overwrite: Whether a part already filed under `source` may be replaced.
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
        file = f"{source}{DOCUMENT_EXT}"

        self._path = root / file
        self._source = source
        self._settings = settings
        self._overwrite = overwrite
        self._frames: list[FrameRange] = []
        self._cached: SequenceRange | None = None
        self._saved = False

    def __call__(self, step: Step[Tensor, Path]) -> None:
        """Measure `step`, so the meter can be registered as a hook directly."""
        self.measure(step)

    def measure(self, step: Step[Tensor, Path]) -> None:
        """Record the range of the frame in `step`, named after its own file.

        Raises:
            ValueError: If the step carries no frame or no path, or if the
                frame holds no finite value to take a range from.
        """
        frame = step.require()
        path = step.require_extra()

        found = finite_range(frame)
        if found is None:
            msg = f"no finite value in {path.name} (sequence: {self._source})"
            raise ValueError(msg)

        self._frames.append(FrameRange(path.name, *found))

    def to_range(self) -> SequenceRange:
        """Fold what has been measured so far into one range for the sequence.

        Raises:
            ValueError: If no frame has been measured yet.
        """
        if self._cached is None or len(self._cached) != len(self._frames):
            self._cached = SequenceRange(self._source, tuple(self._frames))
        return self._cached

    def report(self) -> str | None:
        """Return one line naming the range measured, or `None` if none was."""
        if not self._frames:
            return None

        frames = quantify(len(self._frames), "frame")
        return f"measured {self.to_range()} across {frames}"

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Write the sequence's part, unless the sequence ended in an error.

        Raises:
            FileExistsError: If a part is already filed under this name and
                this meter was not told it may replace it.
        """
        if exc_type is not None:
            return

        cache = self.to_range()

        document: dict[str, object] = {}
        if self._settings is not None:
            document["settings"] = dict(self._settings)
        document |= cache.to_dict()

        with StagedFile(
            self._path,
            overwrite=self._overwrite,
            make_parents=True,
            encoding="utf-8",
        ) as file:
            file.write(json.dumps(document, allow_nan=False))

        self._cached = cache
        self._saved = True

    def revert(self) -> None:
        """Take back the part, if this meter got as far as writing one.

        A part on disk stands for a sequence that finished, and one whose other
        outputs could not be committed did not. Taking it back is what puts the
        sequence back among the skipped rather than leaving the document
        counting it as covered while its frames are nowhere.
        """
        if not self._saved:
            return

        self._path.unlink(missing_ok=True)
        self._saved = False


# ========================== #
#          Document          #
# ========================== #


@dataclass(frozen=True, slots=True)
class Coverage:
    """How much of the dataset a document accounts for, and how it got there.

    Measured against the whole dataset rather than against what one run was
    given, since a document may fold parts an earlier run left: counting
    against the selection would say `31 of 31` while the bounds came from 121.
    Every sequence the source holds is in exactly one of `covered`, `skipped`
    and `unselected`.

    The two names it misses are kept apart because they call for different
    things. One was given to the run and left no range, which is what a retry
    is built from; the other was never given, which is what `include` and
    `exclude` do and needs nothing.

    Attributes:
        found: How many sequences the source holds.
        selected: How many of those the run was given to cover.
        covered: How many the document has a range for.
        reused: How many of those came from a part the run did not measure.
            Defaults to none of them.
        skipped: The selected names the document has no range for. Defaults to
            no names.
        unselected: The names the run was not given and has no range for.
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


def save_range_document(
    path: StrPath,
    dataset: DatasetRange | None,
    *,
    settings: Mapping[str, object] | None = None,
    coverage: Coverage | None = None,
    overwrite: bool = False,
) -> Path:
    """Write one range document, with what was measured and how it was made.

    `coverage` comes before `dataset` so that whoever opens the file to read the
    bounds meets the statement of what they cover first.

    Args:
        path: The file to write, given `.json` if it has no extension.
        dataset: The folded range the document is written to carry, or `None`
            when nothing was folded. A range over nothing has no bounds to
            invent, so the document then carries what it covers and no more.
        settings: The block a later run would compare to decide whether this
            document still describes it, such as the filter and the frame step.
            Defaults to `None`, which records nothing.
        coverage: The statement of how much of the dataset the range accounts
            for. Defaults to `None`, which leaves the document silent on it.
        overwrite: Whether an existing document may be replaced. Defaults to
            `False`.

    Returns:
        The path actually written, extension included.

    Raises:
        FileExistsError: If the document is already there and `overwrite` is
            not set.
    """
    path = ensure_file_extension(path, DOCUMENT_EXT, add=True)

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
#           Branch           #
# ========================== #


class Named(Protocol):
    """Whatever a range document needs of a sequence: what to file it under."""

    @property
    def name(self) -> str: ...


class RangeDocument:
    """The side branch that gathers a dataset's ranges into one document.

    It hands each sequence a meter, and each meter leaves its own part in a
    folder beside the document. Closing it folds the parts that belong to the
    dataset and writes the document, together with a statement of how much of
    that dataset those parts account for.

    Since the parts are read off disk rather than passed back, it does not
    matter which process measured them.

    Args:
        path: The file to write the document to, given `.json` if it has none.
        source: The dataset root the run read, recorded so two documents can be
            told apart before anyone merges them.
        contents: Every sequence the source holds, each mapped to the frames it
            would be measured over. The whole dataset rather than the run's own
            selection, since a document may fold parts an earlier run left and
            coverage counted against the selection would call that complete.
        settings: The block a later run would compare against this one. Defaults
            to `None`, which records nothing and so can never be reused.
        selected: The sequences of the contents this run was given to cover.
            Repeats count once. Defaults to `None`, which takes all of them.
        if_ranges_exist: The policy for a sequence that already has a part here.
            Defaults to `"error"`.
        if_sources_gone: The policy for a part whose sequence the source has
            lost. Defaults to `"keep"`.

    Raises:
        ValueError: If `contents` is empty, since coverage would then have nothing
            to be measured against, or if `selected` names something the contents
            does not hold.
    """

    PARTS_SUFFIX = ".parts"

    def __init__(
        self,
        path: StrPath,
        source: str,
        contents: Mapping[str, Sequence[str]],
        settings: Mapping[str, object] | None = None,
        *,
        selected: Sequence[str] | None = None,
        if_ranges_exist: ExistingOutputPolicy = "error",
        if_sources_gone: UnsourcedOutputPolicy = "keep",
    ) -> None:
        if not contents:
            msg = "no sequence to cover: `contents` must hold at least one"
            raise ValueError(msg)

        self.path = ensure_file_extension(path, DOCUMENT_EXT, add=True)
        self.parts_root = self.path.with_suffix(self.PARTS_SUFFIX)

        self.source = source
        self.contents = {name: tuple(frames) for name, frames in contents.items()}
        self.settings = settings
        self.if_ranges_exist = if_ranges_exist
        self.if_sources_gone = if_sources_gone

        names = unwrap_or_default(selected, tuple(self.contents))
        self.selected = tuple(dict.fromkeys(names))

        if unknown := [n for n in self.selected if n not in self.contents]:
            msg = f"selected {unknown[0]!r}, which the source does not hold"
            raise ValueError(msg)

        self._recorded = as_read_back(settings)
        self._entered = False
        self._reused: frozenset[str] = frozenset()
        self._saved: DatasetRange | None = None
        self._written: Path | None = None

    @property
    def found(self) -> int:
        """How many sequences the source holds."""
        return len(self.contents)

    @property
    def _replacing(self) -> bool:
        """Whether a document or part already there may be written over."""
        return self.if_ranges_exist != "error"

    def get_hook(self, source: Named) -> SequenceRangeMeter | None:
        """Return the meter that will measure `source`, or `None` to reuse.

        Whether a part still stands for this run was settled when the document
        opened, where the whole dataset was in view; this only looks the answer
        up. A sequence nothing has to measure costs no frames at all, which is
        what reuse is for.

        Returns:
            The meter, filed under the sequence's name, or `None` when a part
            already there was found to still describe this run.
        """
        if source.name in self._reused:
            return None

        return SequenceRangeMeter(
            self.parts_root, source.name, self.settings, overwrite=self._replacing
        )

    def list_parts(self) -> list[Path]:
        """Return every part on disk, ordered by the sequence it belongs to."""
        parts = search_files(
            self.parts_root,
            name_filter=EndsWith(DOCUMENT_EXT),
            ordered=False,
        )

        return sorted(parts, key=self._source_of)

    def _list_staging(self) -> list[Path]:
        """Return the staging files an interrupted run left among the parts.

        A part is written beside its destination under a hidden name ending in
        `.tmp` and moved into place on a clean close, so anything of that shape
        still here belongs to a run that never got to close. Nothing else
        collects them: they are hidden from `list_parts`, and the only other
        hand on them dies with the process that staged them.
        """
        temp_filter = And((StartsWith("."), EndsWith(".tmp")))
        return search_files(self.parts_root, name_filter=temp_filter)

    def _still_describes(
        self, document: Mapping[str, Any], sequence: SequenceRange
    ) -> bool:
        """Whether a part on disk stands for what this run would measure.

        Two things can have moved since it was written, and neither shows in
        the part's own name: the settings that shaped its numbers, and which
        frames the source holds by name. A part failing either is stale rather
        than broken, so it is passed over rather than refused.

        Args:
            document: The part as it was read, for the settings it records.
            sequence: The range that part holds. Its source is one the contents
                holds, which `_read_valid` has established by the time it asks.
        """
        if document.get("settings") != self._recorded:
            return False

        listed = self.contents[sequence.source]

        return tuple(frame.source for frame in sequence.frames) == listed

    def _read_part(self, part: Path) -> dict[str, Any]:
        """Read one part off disk, refusing anything that is not a mapping."""
        with part.open(encoding="utf-8") as file:
            document = json.load(file)

        if not isinstance(document, dict):
            msg = f"malformed range document: {type(document).__name__} at the top"
            raise ValueError(msg)  # noqa: TRY004

        return document

    def _source_of(self, part: Path) -> str:
        """Return the sequence a part belongs to, read off where it sits."""
        return stringify_path(part.with_suffix(""), after=self.parts_root)

    def _read_valid(self, *, strict: bool) -> Iterator[tuple[Path, SequenceRange]]:
        """Yield each part that still stands for this run, with what it holds.

        The one place a part is judged, so opening and closing cannot come to
        different answers. A part reused on the way in that the fold then
        passed over would leave a sequence nothing measured and nothing counted.

        Args:
            strict: Whether a part that cannot be read, or that is filed under
                a sequence other than the one it holds, stops the run. Judging
                is not reading, so opening passes over such a part and only the
                fold refuses it.

        Yields:
            Each part and the range it holds, in the order `list_parts` gives.

        Raises:
            ValueError: Under `strict`, if a part cannot be read or is filed
                under the wrong sequence. The part is named, since the folder
                holds one file per sequence and only the name says which to go
                and look at.
        """
        for part in self.list_parts():
            name = self._source_of(part)
            if name not in self.contents:
                continue

            try:
                document = self._read_part(part)
                sequence = SequenceRange.from_dict(document)
            except (OSError, ValueError) as error:
                if not strict:
                    continue
                msg = f"unreadable part {name!r}: {error}. Remove it, or run it again"
                raise ValueError(msg) from error

            if name != sequence.source:
                if not strict:
                    continue
                msg = f"part {name!r} holds {sequence.source!r}: run {name!r} again"
                raise ValueError(msg)

            if self._still_describes(document, sequence):
                yield part, sequence

    def to_range(self) -> DatasetRange | None:
        """Fold every part on disk into one range for the dataset.

        Only the parts of sequences the source still holds, and only those a run
        with these settings could have written. One left by a run whose dataset
        was larger describes a sequence that is not there, and one left under a
        different filter describes numbers this run would not produce; folding
        either would widen the bounds with a range nothing here accounts for.
        Both stay on disk, since a source that looks smaller than it is makes
        exactly the same absence as one that shrank.

        A part is filed under the sequence it belongs to and says so again
        inside, and the two must agree. Nothing else compares them, so a part
        that disagrees would be sorted under one name and counted under
        another, which no number in the finished document would show.

        Returns:
            The folded range, or `None` when no part is there. A run whose
            sequences all failed has nothing to take bounds from, and that is
            what `coverage` is for: the document says it covers none of them
            rather than not being written at all.

        Raises:
            ValueError: If one of the parts cannot be read, or one is filed
                under a sequence other than the one it holds. A part that
                cannot be read is named, since the folder holds one file per
                sequence and only the name says which to go and look at.
        """
        folded = tuple(sequence for _, sequence in self._read_valid(strict=True))
        if not folded:
            return None

        return DatasetRange(self.source, folded)

    def get_coverage(self, dataset: DatasetRange | None) -> Coverage:
        """Measure `dataset` against the whole dataset this document describes.

        What is missing is worked out from the lists rather than reported by
        the run, so a sequence counts as missing whether it failed, went down
        with its worker, or was never given to this run at all. Which of those
        it was is what the two lists keep apart.

        Every number is read off the contents, so the three groups always add up
        to it. Counting one from the contents and another from disk let the two
        disagree.
        """
        folded = set() if dataset is None else {s.source for s in dataset.sequences}
        given = set(self.selected)

        missing = [name for name in self.contents if name not in folded]

        selected = len(self.selected)
        covered = sum(name in folded for name in self.contents)
        reused = sum(name in folded for name in self._reused)
        skipped = tuple(name for name in self.selected if name not in folded)
        unselected = tuple(name for name in missing if name not in given)

        return Coverage(self.found, selected, covered, reused, skipped, unselected)

    def save(self) -> Path:
        """Fold the parts and write the document, coverage included.

        What was folded is remembered once the file is on disk and not before,
        so a write that failed leaves this branch with nothing to report rather
        than a line about a document that is not there.

        Returns:
            The path actually written, extension included.

        Raises:
            ValueError: If one of the parts cannot be read.
            FileExistsError: If the document is already there and this one was
                not told it may replace it.
        """
        dataset = self.to_range()

        written = save_range_document(
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
        was given, so a document folded over part of one cannot be mistaken for
        one folded over all of it. What is missing is split the way `coverage`
        splits it, since a sequence that failed and one nobody asked for call
        for different things. A document that covered none has no bounds to
        name and says only what it covers.
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
        having already dropped the parts an earlier run left behind.

        What an earlier run staged and never committed always goes, since
        nothing else is in a position to collect it. What it committed depends
        on the policy: `"reuse"` keeps every part still describing this run and
        leaves the rest where they are, and the other two clear the folder, so
        that everything folded at the end is this run's own.

        Judging happens here, with the whole dataset in view and in one process.
        A worker holds a copy of this branch and nothing it learns comes home,
        so a part judged there could not be counted.

        Raises:
            FileExistsError: If the document is already there and this one was
                not told it may replace it.
            RuntimeError: If this document has been opened before.
        """
        if self._entered:
            msg = f"{self.path.name} was opened already: open it once per run"
            raise RuntimeError(msg)

        self._entered = True
        reserve_path(self.path, exist_ok=self._replacing, make_parents=True)

        ensure_dir_exists(self.parts_root, make=True)

        emptied = set()
        for stale in self._list_staging():
            emptied.add(stale.parent)
            stale.unlink()

        if self.if_ranges_exist == "reuse":
            kept = (part for part, _ in self._read_valid(strict=False))
            self._reused = frozenset(map(self._source_of, kept))
        else:
            for stale in self.list_parts():
                emptied.add(stale.parent)
                stale.unlink()

        for folder in emptied:
            prune_upward(folder, self.parts_root)

        return self

    def list_unsourced(self) -> list[str]:
        """Return the sequences a part is filed under that the source has lost.

        Named rather than acted on: the same absence is what a half mounted
        share and a misspelt subpath produce, so what to do with them is the
        caller's policy and saying they are there is not.
        """
        return find_unsourced(map(self._source_of, self.list_parts()), self.contents)

    def drop_unsourced(self) -> list[str]:
        """Remove the parts of sequences the source has lost, and name them.

        The folders a removal empties go with it, the same way opening prunes
        the ones it emptied: a sequence dropped from a nested dataset would
        otherwise leave the path down to it standing.

        Returns:
            The sequences whose parts were removed, in the order they were
            filed under.
        """
        dropped = []

        for name in self.list_unsourced():
            part = self.parts_root / f"{name}{DOCUMENT_EXT}"
            part.unlink()
            prune_upward(part.parent, self.parts_root)
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
        parts it left are of sequences that finished. Writing them is what
        `coverage` is for: the document says which of the dataset it accounts
        for and names the rest, where refusing to write leaves the healthy
        results on disk with nothing to read them by.

        Parts of sequences the source has lost go afterwards where the policy
        says so. The fold passes over them either way, so removing them is
        tidying rather than part of the answer, and one that cannot be removed
        must not cost the document.

        The failure itself is not this branch's to report. It reaches the
        driver, which is what decides the run's verdict.
        """
        self.save()

        if self.if_sources_gone == "delete":
            self.drop_unsourced()
