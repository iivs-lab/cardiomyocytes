from __future__ import annotations

__all__ = (
    "CompositeRange",
    "DatasetRange",
    "FrameRange",
    "RangeDocument",
    "SequenceRange",
    "SequenceRangeMeter",
    "ValueRange",
)

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import TYPE_CHECKING, Any, Self, override

from kaparoo.filesystem import (
    StagedFile,
    ensure_dir_exists,
)
from kaparoo.utils import quantify

from iivs_cardio.common.pipeline.branch import JSON_EXT
from iivs_cardio.common.pipeline.document import DocumentBranch
from iivs_cardio.common.range import finite_range

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path
    from types import TracebackType

    from kaparoo.filesystem.types import StrPath
    from torch import Tensor

    from iivs_cardio.common.pipeline import Step
    from iivs_cardio.common.pipeline.branch import (
        Named,
    )


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
        file = f"{source}{JSON_EXT}"

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


class RangeDocument(
    DocumentBranch["Named", SequenceRange, DatasetRange, SequenceRangeMeter]
):
    """The document a phase stage writes, gathering every sequence's range.

    Attributes:
        PARTS_SUFFIX: As `DocumentBranch`.
    """

    @override
    def _make_meter(self, source: Named) -> SequenceRangeMeter:
        return SequenceRangeMeter(
            self.parts_root, source.name, self.settings, overwrite=self._replacing
        )

    @override
    def _parse(self, document: Mapping[str, Any]) -> SequenceRange:
        return SequenceRange.from_dict(document)

    @override
    def _fold(self, parts: tuple[SequenceRange, ...]) -> DatasetRange:
        return DatasetRange(self.source, parts)

    @override
    def _expected(self, names: Sequence[str]) -> Sequence[str]:
        """Every frame the source holds, a range being one frame in, one out."""
        return names
