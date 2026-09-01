from __future__ import annotations

__all__ = (
    "Bounds",
    "DatasetRange",
    "FrameRange",
    "RangeDocument",
    "RangeWriter",
    "SequenceRange",
)

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Self, override

from kaparoo.utils import quantify

from iivs_cardio.common.pipeline.document import (
    DatasetResult,
    DocumentBranch,
    ResultWriter,
    SequenceResult,
    StepResult,
    read_entry,
    read_number,
)
from iivs_cardio.common.range import finite_range

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from kaparoo.filesystem.types import StrPath
    from torch import Tensor

    from iivs_cardio.common.pipeline import Step
    from iivs_cardio.common.pipeline.base import Named


# ========================== #
#           Ranges           #
# ========================== #


@dataclass(frozen=True, slots=True)
class Bounds:
    """The lowest and highest value found in something.

    One measurement rather than two, since the pair is what a later run scales by and
    neither end means anything without the other.

    Attributes:
        min_value: The lowest value found.
        max_value: The highest value found.

    Raises:
        ValueError: If the lowest value is above the highest.
    """

    min_value: float
    max_value: float

    def __post_init__(self) -> None:
        """Refuse a pair whose two ends are the wrong way round."""
        if self.min_value > self.max_value:
            msg = f"inverted range {self}: the lowest value is above the highest"
            raise ValueError(msg)

    def __str__(self) -> str:
        """The two bounds, shortened for reading rather than for reloading."""
        return f"[{self.min_value:.4g}, {self.max_value:.4g}]"

    def to_dict(self) -> dict[str, Any]:
        """Return the pair as plain data, ready to be written as JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Self:
        """Rebuild a pair from its `min_value` and `max_value`.

        Raises:
            ValueError: If either is absent or unreadable, or if the two run backwards.
        """
        return cls(
            read_number(document, "min_value"),
            read_number(document, "max_value"),
        )


@dataclass(frozen=True, slots=True)
class FrameRange(StepResult):
    """The range of one frame.

    Attributes:
        source: The file the frame was read from, which is the name it has at the source
            and not necessarily the one a cache of the same run gives it.
        bounds: The lowest and highest value in the frame.
    """

    bounds: Bounds

    @override
    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Self:
        """Rebuild a frame range from its `source` and its `bounds`.

        Raises:
            ValueError: If either key is absent, or the bounds cannot be read.
        """
        return cls(
            read_entry(document, "source", str),
            Bounds.from_dict(read_entry(document, "bounds", dict)),
        )


@dataclass(frozen=True, slots=True)
class SequenceRange(SequenceResult[FrameRange]):
    """The range of one sequence, summarised from the frames it was measured over.

    Position is what the ends are named by, not the frame's own name. Each frame is
    filed under the source it was read from, while a cache the same run writes numbers
    its frames from zero without a gap, so the two disagree wherever the run read the
    source with a stride or the source itself was sparse. The nth entry here is the nth
    frame either way.

    Attributes:
        source: The name the sequence has in its dataset.
        steps: The range of each frame, in the order they were read, which is the order
            a cache of the same run writes them in.
        bounds: The lowest and highest value across every frame.
        min_index: The position of the frame holding the lowest value.
        max_index: The position of the frame holding the highest value.

    Raises:
        ValueError: If there are no frames, since a range over nothing has no meaning to
            fall back on.
    """

    bounds: Bounds = field(init=False)
    min_index: int = field(init=False)
    max_index: int = field(init=False)

    def __str__(self) -> str:
        """The two bounds, shortened for reading rather than for reloading."""
        return str(self.bounds)

    @override
    def _aggregate(self) -> None:
        """Take the widest range the frames reach, and note which gave each end."""
        indices = range(len(self.steps))
        low = min(indices, key=lambda i: self.steps[i].bounds.min_value)
        high = max(indices, key=lambda i: self.steps[i].bounds.max_value)

        object.__setattr__(
            self,
            "bounds",
            Bounds(self.steps[low].bounds.min_value, self.steps[high].bounds.max_value),
        )
        object.__setattr__(self, "min_index", low)
        object.__setattr__(self, "max_index", high)

    @override
    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Self:
        """Rebuild a sequence range from its `source` and its `steps`.

        Raises:
            ValueError: If either key is absent, or a frame cannot be read.
        """
        steps = read_entry(document, "steps", (list, tuple))

        return cls(
            read_entry(document, "source", str),
            tuple(FrameRange.from_dict(step) for step in steps),
        )


@dataclass(frozen=True, slots=True)
class DatasetRange(DatasetResult[SequenceRange]):
    """The range of a whole dataset, summarised from the sequences it covers.

    The ends are named rather than numbered, a sequence keeping the name it has at the
    source wherever a run writes it.

    Attributes:
        source: The dataset root the run read, which is what tells two documents apart
            when someone comes to merge them.
        sequences: The range of each sequence, in the order they were summarised.
        bounds: The lowest and highest value across every sequence.
        min_source: The sequence holding the lowest value.
        max_source: The sequence holding the highest value.

    Raises:
        ValueError: If there are no sequences, or if two are filed under one name.
    """

    bounds: Bounds = field(init=False)
    min_source: str = field(init=False)
    max_source: str = field(init=False)

    def __str__(self) -> str:
        """The two bounds, shortened for reading rather than for reloading."""
        return str(self.bounds)

    @override
    def _aggregate(self) -> None:
        """Take the widest range the sequences reach, and name which gave each end."""
        low = min(self.sequences, key=lambda one: one.bounds.min_value)
        high = max(self.sequences, key=lambda one: one.bounds.max_value)

        object.__setattr__(
            self, "bounds", Bounds(low.bounds.min_value, high.bounds.max_value)
        )
        object.__setattr__(self, "min_source", low.source)
        object.__setattr__(self, "max_source", high.source)

    @override
    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Self:
        """Rebuild a dataset range from its `source` and its `sequences`.

        Raises:
            ValueError: If either key is absent, or a sequence cannot be read.
        """
        sequences = read_entry(document, "sequences", (list, tuple))

        return cls(
            read_entry(document, "source", str),
            tuple(SequenceRange.from_dict(one) for one in sequences),
        )


# ========================== #
#         Measuring          #
# ========================== #


class RangeWriter(ResultWriter[SequenceRange]):
    """Measure the range of every frame of one sequence, then write the result.

    This is the hook a range document hands to a sequence. It records a range per frame
    as the frames go by, and on a clean close writes them beside the document as that
    sequence's result.

    Args:
        root: The folder the result is written into, created if it is not there.
        source: The name the sequence has, used both in the record and as the name of
            the file it is written to.
        settings: The settings that shaped the ranges, written into the result so it can
            be told from one an earlier run left under different ones. The document
            carries the same block, and a result outliving the document is the case that
            needs its own copy. Defaults to `None`, which records nothing and so can
            never be reused.
        overwrite: Whether a result already filed under `source` may be replaced. Its
            own run clears the folder on the way in, so one that is there belongs to
            something else: two sequences whose names came out the same, most likely,
            which is a mistake rather than a second attempt. Defaults to `False`.
    """

    def __init__(
        self,
        root: StrPath,
        source: str,
        settings: Mapping[str, object] | None = None,
        *,
        overwrite: bool = False,
    ) -> None:
        super().__init__(root, source, settings, overwrite=overwrite)

        self._frames: list[FrameRange] = []
        self._cached: SequenceRange | None = None

    def __call__(self, step: Step[Tensor, Path]) -> None:
        """Measure `step`, so the writer can be registered as a hook directly."""
        self.measure(step)

    def measure(self, step: Step[Tensor, Path]) -> None:
        """Record the range of the frame in `step`, named after its own file.

        Raises:
            ValueError: If the step carries no frame or no path, or if the frame holds
                no finite value to take a range from.
        """
        frame = step.require()
        path = step.require_extra()

        found = finite_range(frame)
        if found is None:
            msg = f"no finite value in {path.name} (sequence: {self._source})"
            raise ValueError(msg)

        self._frames.append(FrameRange(path.name, Bounds(*found)))

    def to_range(self) -> SequenceRange:
        """Combine what has been measured so far into one range for the sequence.

        Raises:
            ValueError: If no frame has been measured yet.
        """
        if self._cached is None or len(self._cached) != len(self._frames):
            self._cached = SequenceRange(self._source, tuple(self._frames))
        return self._cached

    @override
    def _result(self) -> SequenceRange:
        return self.to_range()

    def report(self) -> str | None:
        """Return one line naming the range measured, or `None` if none was."""
        if not self._frames:
            return None

        frames = quantify(len(self._frames), "frame")
        return f"measured {self.to_range().bounds} across {frames}"


# ========================== #
#          Document          #
# ========================== #


class RangeDocument(DocumentBranch["Named", SequenceRange, DatasetRange]):
    """The document a phase stage writes, gathering every sequence's range.

    Attributes:
        RESULTS_SUFFIX: What the folder of results beside the document is called.
        path: The document itself, extension included.
        results_root: The folder each sequence's range is written into.
        source: The dataset root the run read, recorded so two documents can be told
            apart before anyone merges them.
        contents: Every sequence the source holds, each mapped to the frames its range
            is measured over. A range is one frame in and one out, so a sequence is owed
            a frame range for every name here.
        settings: The block a later run would compare against this one, `None` where
            nothing was recorded and so nothing can be reused.
        selected: The sequences of the contents this run was given to cover, repeats
            counted once.
        if_present: The policy for a sequence that already has a range here. `"reuse"`
            keeps one still describing this run and measures the rest.
        if_unsourced: The policy for a range whose sequence the source has lost.
    """

    @override
    def _make_writer(
        self,
        root: Path,
        source: Named,
        settings: Mapping[str, object] | None,
        *,
        overwrite: bool,
    ) -> RangeWriter:
        return RangeWriter(root, source.name, settings, overwrite=overwrite)

    @override
    def _parse(self, document: Mapping[str, Any]) -> SequenceRange:
        return SequenceRange.from_dict(document)

    @override
    def _combine(self, results: tuple[SequenceRange, ...]) -> DatasetRange:
        return DatasetRange(self.source, results)

    @override
    def _expected(self, names: Sequence[str]) -> Sequence[str]:
        """Every frame the source holds, a range being one frame in, one out."""
        return names
