from __future__ import annotations

__all__ = (
    "CompositeRange",
    "DatasetRange",
    "FrameRange",
    "RangeCollector",
    "SequenceRange",
    "ValueRange",
    "as_dict",
)

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from iivs_cardio.common.range import finite_range

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

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

    Args:
        source: The sequence's path relative to the dataset root, which every
            `FrameRange` is reported under and which names a frame in the error.
    """

    def __init__(self, source: str) -> None:
        self._source = source
        self._frames: list[FrameRange] = []

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

    def collected(self) -> SequenceRange:
        """Everything observed so far, folded.

        Raises:
            ValueError: If nothing was observed, since a range over no frame is
                undefined rather than empty.
        """
        return SequenceRange(self._source, tuple(self._frames))
