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
from typing import TYPE_CHECKING, Any, Final, Protocol, Self, override

from kaparoo.filesystem import (
    StagedFile,
    dir_empty,
    ensure_dir_exists,
    ensure_file_extension,
    search_dirs,
    search_files,
    stringify_path,
)
from kaparoo.filters import EndsWith

from iivs_cardio.common.range import finite_range

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path
    from types import TracebackType

    from kaparoo.filesystem.types import StrPath
    from torch import Tensor

    from iivs_cardio.common.pipeline import Step


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


def _counted(count: int, noun: str) -> str:
    """`count` of `noun`, pluralised for every count but one."""
    return f"{count} {noun}{'s' if count != 1 else ''}"


@dataclass(frozen=True, slots=True)
class ValueRange(ABC):
    """The lowest and highest value found in something, and what that was.

    Attributes:
        source: what the range was measured over, named the way a reader of the
            document would look it up.
        min_value: the lowest value found.
        max_value: the highest value found.
    """

    source: str
    min_value: float
    max_value: float

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
        source: the file the frame was read from.
        min_value: the lowest value in the frame.
        max_value: the highest value in the frame.
    """

    @override
    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Self:
        """Rebuild a frame range from `source`, `min_value` and `max_value`.

        Raises:
            ValueError: If any of the three is absent or unreadable.
        """
        return cls(
            _entry(document, "source", str),
            float(_entry(document, "min_value", (int, float))),
            float(_entry(document, "max_value", (int, float))),
        )


@dataclass(frozen=True, slots=True)
class CompositeRange(ValueRange, ABC):
    """A range folded from smaller ones, keeping where each end came from.

    The bounds are not given but taken from the parts, and each is remembered
    with the part it came from, so a wide dataset range can be traced back to
    the one sequence or frame that widened it.

    Attributes:
        source: what the folded range was measured over.
        min_value: the lowest value across every part.
        max_value: the highest value across every part.
        min_index: which part holds the lowest value.
        max_index: which part holds the highest value.

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

    Attributes:
        source: what the sequence is called in its dataset.
        min_value: the lowest value across every frame.
        max_value: the highest value across every frame.
        min_index: which frame holds the lowest value.
        max_index: which frame holds the highest value.
        frames: the range of each frame, in the order they were read.
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
        source: the dataset root the run read, which is what tells two
            documents apart when someone comes to merge them.
        min_value: the lowest value across every sequence.
        max_value: the highest value across every sequence.
        min_index: which sequence holds the lowest value.
        max_index: which sequence holds the highest value.
        sequences: the range of each sequence, in the order they were folded.
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


class SequenceRangeMeter:
    """Measure the range of every frame of one sequence, then write the result.

    This is the hook a range document hands to a sequence. It records a range
    per frame as the frames go by, and on a clean close writes them beside the
    document as that sequence's part. Writing is how the result gets home: a
    sequence may be measured in a worker process of its own, and nothing it
    keeps in memory comes back.

    A close that follows an error writes nothing, so a part on disk always
    stands for a sequence that finished.

    Args:
        root: the folder the part is written into, created if it is not there.
        source: what the sequence is called, used both in the record and as the
            name of the file it is written to.
    """

    def __init__(self, root: StrPath, source: str) -> None:
        root = ensure_dir_exists(root, make=True)
        file = f"{source}{DOCUMENT_EXT}"

        self._path = root / file
        self._source = source
        self._frames: list[FrameRange] = []
        self._cached: SequenceRange | None = None

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

        counted = _counted(len(self._frames), "frame")
        return f"measured {self.to_range()} across {counted}"

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Write the sequence's part, unless the sequence ended in an error."""
        if exc_type is not None:
            return

        cache = self.to_range()

        with StagedFile(
            self._path,
            overwrite=True,
            make_parents=True,
            encoding="utf-8",
        ) as file:
            file.write(json.dumps(cache.to_dict(), indent=2))

        self._cached = cache


@dataclass(frozen=True, slots=True)
class Coverage:
    """How much of what a run set out to cover it actually did.

    Attributes:
        total: how many sequences the run was given to cover.
        covered: how many of them the document has a range for.
        skipped: the names it does not, in the order they were given.
    """

    total: int
    covered: int
    skipped: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the coverage as plain data, ready to be written as JSON."""
        skipped = list(self.skipped)
        return {"total": self.total, "covered": self.covered, "skipped": skipped}


def save_range_document(
    path: StrPath,
    dataset: DatasetRange,
    *,
    settings: Mapping[str, object] | None = None,
    coverage: Coverage | None = None,
    overwrite: bool = False,
) -> Path:
    """Write one range document, with what was measured and how it was made.

    `coverage` comes before `dataset` so that whoever opens the file to read the
    bounds meets the statement of what they cover first.

    Args:
        path: where to write, given `.json` if it has no extension.
        dataset: the folded range the document is written to carry.
        settings: what a later run would compare to decide whether this document
            still describes it, such as the filter and the frame step.
        coverage: how much of the run's own list the dataset accounts for.
        overwrite: whether an existing document may be replaced.

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
    document["dataset"] = dataset.to_dict()

    with StagedFile(
        path,
        overwrite=overwrite,
        make_parents=True,
        encoding="utf-8",
    ) as file:
        file.write(json.dumps(document, indent=2))

    return path


class Named(Protocol):
    """Whatever a range document needs of a sequence: what to file it under."""

    @property
    def name(self) -> str: ...


class RangeDocument:
    """The side branch that gathers a dataset's ranges into one document.

    It hands each sequence a meter, and each meter leaves its own part in a
    folder beside the document. Opening the branch clears whatever an earlier
    run left there; closing it folds the parts that are present and writes the
    document, together with a statement of how much of the run's own list those
    parts account for.

    Since the parts are read off disk rather than passed back, it does not
    matter which process measured them.

    Args:
        path: where to write the document, given `.json` if it has none.
        source: the dataset root the run read, recorded so two documents can be
            told apart before anyone merges them.
        sequence_names: every sequence the run set out to cover. Repeats count
            once, and the order is the one `coverage` reports in.
        settings: what a later run would compare against this one.
        overwrite: whether an existing document may be replaced.

    Raises:
        ValueError: If `sequence_names` is empty, since coverage would then have
            nothing to be measured against.
    """

    PARTS_SUFFIX = ".parts"

    def __init__(
        self,
        path: Path,
        source: str,
        sequence_names: Sequence[str],
        settings: Mapping[str, object] | None = None,
        *,
        overwrite: bool = False,
    ) -> None:
        if not sequence_names:
            msg = "no sequence to cover: `sequence_names` must hold at least one"
            raise ValueError(msg)

        self.path = ensure_file_extension(path, DOCUMENT_EXT, add=True)
        self.parts_root = self.path.with_suffix(self.PARTS_SUFFIX)

        self.source = source
        self.sequence_names = tuple(dict.fromkeys(sequence_names))
        self.settings = settings
        self.overwrite = overwrite

        self._saved: DatasetRange | None = None

    def get_hook(self, source: Named) -> SequenceRangeMeter:
        """Return the meter that will measure `source`, filed under its name."""
        return SequenceRangeMeter(self.parts_root, source.name)

    def list_parts(self) -> list[Path]:
        """Return every part on disk, ordered by the sequence it belongs to."""
        parts = search_files(
            self.parts_root,
            name_filter=EndsWith(DOCUMENT_EXT),
            ordered=False,
        )

        return sorted(parts, key=self._source_of)

    def _source_of(self, part: Path) -> str:
        """The sequence a part belongs to, read back from where it sits."""
        return stringify_path(part.with_suffix(""), after=self.parts_root)

    def to_range(self) -> DatasetRange:
        """Fold every part on disk into one range for the dataset.

        Raises:
            ValueError: If no part is there, or one of them cannot be read.
        """
        sequences = []

        for part in self.list_parts():
            with part.open(encoding="utf-8") as file:
                document = json.load(file)
                sequences.append(SequenceRange.from_dict(document))

        return DatasetRange(self.source, tuple(sequences))

    def get_coverage(self, dataset: DatasetRange) -> Coverage:
        """Measure `dataset` against the sequences this document was given.

        What is missing is worked out from the two lists rather than reported
        by the run, so a sequence counts as skipped whether it failed, went
        down with its worker, or never started.
        """
        covered = {sequence.source for sequence in dataset.sequences}
        skipped = tuple(name for name in self.sequence_names if name not in covered)

        return Coverage(len(self.sequence_names), len(covered), skipped)

    def save(self) -> Path:
        """Fold the parts and write the document, coverage included.

        Returns:
            The path actually written, extension included.

        Raises:
            ValueError: If no part is there, or one of them cannot be read.
            FileExistsError: If the document is already there and this one was
                not told it may replace it.
        """
        dataset = self.to_range()
        self._saved = dataset

        return save_range_document(
            self.path,
            dataset,
            settings=self.settings,
            coverage=self.get_coverage(dataset),
            overwrite=self.overwrite,
        )

    def report(self) -> str | None:
        """Return one line naming what was written, or `None` before it was.

        The line says how many sequences were covered, and out of how many when
        some are missing, so a document folded over part of a dataset cannot be
        mistaken for one folded over all of it.
        """
        if self._saved is None:
            return None

        dataset = self._saved
        coverage = self.get_coverage(dataset)

        counted = _counted(coverage.covered, "sequence")
        if coverage.skipped:
            whole = _counted(coverage.total, "sequence")
            counted = f"{coverage.covered} of {whole}, {len(coverage.skipped)} skipped"

        return f"wrote {self.path.name} from {counted}: {dataset}"

    def __enter__(self) -> Self:
        """Clear what an earlier run left, so only this run's parts are folded.

        Folders left empty by that clearing go too, since a sequence dropped
        from the dataset would otherwise go on looking present in the tree.
        """
        ensure_dir_exists(self.parts_root, make=True)

        for stale in self.list_parts():
            stale.unlink()

        for folder in reversed(search_dirs(self.parts_root)):
            if dir_empty(folder):
                folder.rmdir()

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Write the document, unless the run itself ended in an error."""
        if exc_type is not None:
            return

        self.save()
