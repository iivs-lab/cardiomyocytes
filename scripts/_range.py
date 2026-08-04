from __future__ import annotations

__all__ = (
    "CompositeRange",
    "DatasetRange",
    "DatasetRangeCollector",
    "FrameRange",
    "RangeCollector",
    "SequenceRange",
    "ValueRange",
    "as_dict",
    "save_range_document",
)

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, Self

from kaparoo.filesystem import StagedFile, ensure_file_extension

from iivs_cardio.common.range import finite_range

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path
    from types import TracebackType

    from kaparoo.filesystem.types import StrPath
    from torch import Tensor

    from iivs_cardio.common.pipeline import Slot


class ValueRange(Protocol):
    @property
    def min_value(self) -> float: ...

    @property
    def max_value(self) -> float: ...


@dataclass(frozen=True, slots=True)
class CompositeRange(ValueRange, ABC):
    min_value: float = field(init=False)
    max_value: float = field(init=False)
    min_index: int = field(init=False)
    max_index: int = field(init=False)

    @property
    @abstractmethod
    def parts(self) -> Sequence[ValueRange]: ...

    def __post_init__(self) -> None:
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
class FrameRange:
    source: str  # frame name relative to the sequence folder
    min_value: float
    max_value: float


@dataclass(frozen=True, slots=True)
class SequenceRange(CompositeRange):
    source: str  # folder path relative to the dataset root
    frames: tuple[FrameRange, ...]

    @property
    def parts(self) -> tuple[FrameRange, ...]:
        return self.frames


@dataclass(frozen=True, slots=True)
class DatasetRange(CompositeRange):
    sequences: tuple[SequenceRange, ...]

    @property
    def parts(self) -> tuple[SequenceRange, ...]:
        return self.sequences


def as_dict(dataset: DatasetRange) -> dict[str, Any]:
    """The `dataset` as nested dicts, each level leading with what it names.

    Field order is otherwise the dataclass's own, which puts the folded bounds
    ahead of the parts they were folded from. That reads well but buries each
    `source` under four numbers, and the source is what a reader scans for.
    """

    def source_first(items: list[tuple[str, Any]]) -> dict[str, Any]:
        return dict(sorted(items, key=lambda item: item[0] != "source"))

    return asdict(dataset, dict_factory=source_first)


class RangeCollector:
    """Folds the steps of one scanned sequence into a `SequenceRange`.

    A hook, so ranging costs the traversal nothing extra: the field is already
    in hand where reading it again would re-run the filter kernel, which is the
    expensive half of a filtered read.

    Attach it to the node it should watch rather than its `observe`, so that
    `Node.run` can tell it when the traversal ended. A range is only a range once
    every step has been seen, and nothing else in the chain knows the difference
    between a fold that finished and one that stopped part-way.

    Args:
        source: The sequence's path relative to the dataset root, which every
            `FrameRange` is reported under and which names a frame in the error.
    """

    def __init__(self, source: str) -> None:
        self._source = source
        self._frames: list[FrameRange] = []
        self._finished = False

    def observe(self, slot: Slot[tuple[Tensor, Path]]) -> None:
        """Range this step's field, recording it under the file it came from.

        Raises:
            ValueError: If the field holds no finite value, naming the file.
        """
        field, path = slot.require()

        found = finite_range(field)
        if found is None:
            msg = f"no finite value in {self._source}/{path.name}"
            raise ValueError(msg)

        self._frames.append(FrameRange(path.name, *found))

    def __call__(self, slot: Slot[tuple[Tensor, Path]]) -> None:
        """`observe`, so the collector attaches to a node as the hook it is."""
        self.observe(slot)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # Only a clean exit means every step was seen. A traversal that died
        # part-way leaves a fold over a prefix, which is not this sequence's
        # range and must not be reported as one.
        self._finished = exc_type is None

    def collected(self) -> SequenceRange:
        """The fold over every step of the sequence.

        Raises:
            ValueError: If the traversal has not finished cleanly, since a fold
                over a prefix is not a range, or if nothing was observed, since
                a range over no frame is undefined rather than empty.
        """
        if not self._finished:
            msg = f"the traversal of {self._source} did not finish"
            raise ValueError(msg)

        return SequenceRange(self._source, tuple(self._frames))


def save_range_document(
    dataset: DatasetRange,
    path: StrPath,
    *,
    provenance: Mapping[str, object] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write `dataset` as one JSON document, under what the run can say about it.

    The dataset level is the parent's to write: a worker sees one sequence, so
    folding across them and naming the result happens where they come back. Any
    driver that ranges what it reads wants this, which is why it does not live
    beside the one that ranges phase.

    `provenance` is the run's own -- the unit it read in, the filter it applied,
    whatever else a reader of the numbers would have to ask. It is separate
    because only the driver knows it, and it is optional because `.hydra/` beside
    the document already holds the composed config; what belongs here is the
    little a report still means something without.

    Args:
        dataset: The fold across every sequence the run scanned.
        path: Where to write. A `.json` suffix is added if absent.
        provenance: Merged in beside the dataset, ahead of it.
        overwrite: Whether to replace an existing document.

    Returns:
        The path written, with its suffix settled.
    """
    document = {**(provenance or {}), "dataset": as_dict(dataset)}
    path = ensure_file_extension(path, ".json", add=True)

    with StagedFile(
        path, overwrite=overwrite, make_parents=True, encoding="utf-8"
    ) as file:
        file.write(json.dumps(document, indent=2))

    return path


class DatasetRangeCollector:
    """Gathers every sequence's range for one run, and writes the document at the end.

    Lives as long as the run, where a `RangeCollector` lives as long as one
    sequence. That split is forced: a worker sees one sequence, so the folding
    across them can only happen where their results come back. Owning both the
    fold and the write here is what keeps a driver from restating either.

    A copy of it crosses to each worker, which is safe because `collector_for`
    only builds: the collector it returns lives and fills in that process, and
    comes back filled. Mutating the copy would be lost, so `absorb` is the only
    way in -- one finished collector at a time, since that is how they finish.

    Nothing outside holds a `SequenceRange`. A driver asks for a collector,
    hands the filled one back, and asks for the document; what a sequence's
    range is, and when it is one, stays here.

    This is also the only object that sees how many arrived against how many were
    dispatched, which is what a partial run will have to report against once a
    failed sequence stops taking the pool down with it.
    """

    def __init__(self) -> None:
        self._sequences: list[SequenceRange] = []

    def collector_for(self, source: str) -> RangeCollector:
        """The collector for one sequence, to attach where that sequence is read.

        Here rather than at the call site so that what a run collects is decided
        once. Nothing configures a range today, but a switch to per-frame
        histograms would be this object's to hold, not every driver's.
        """
        return RangeCollector(source)

    def absorb(self, collector: RangeCollector) -> None:
        """Take one finished collector, in the order they finish.

        Raises:
            ValueError: If its traversal did not finish, since a fold over a
                prefix is not the sequence's range.
        """
        self._sequences.append(collector.collected())

    def collected(self) -> DatasetRange:
        """The fold across everything absorbed.

        Raises:
            ValueError: If nothing was absorbed, since a range over no sequence
                is undefined rather than empty.
        """
        return DatasetRange(tuple(self._sequences))

    def save(
        self,
        path: StrPath,
        *,
        provenance: Mapping[str, object] | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Write what was absorbed as one document. See `save_range_document`."""
        return save_range_document(
            self.collected(), path, provenance=provenance, overwrite=overwrite
        )
