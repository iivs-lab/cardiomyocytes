from __future__ import annotations

__all__ = (
    "CompositeRange",
    "DatasetRange",
    "DatasetRangeCollector",
    "FrameRange",
    "SequenceRange",
    "SequenceRangeCollector",
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

    from iivs_cardio.common.pipeline import Step


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
    def source_first(items: list[tuple[str, Any]]) -> dict[str, Any]:
        return dict(sorted(items, key=lambda item: item[0] != "source"))

    return asdict(dataset, dict_factory=source_first)


class SequenceRangeCollector:
    def __init__(self, source: str, into: DatasetRangeCollector) -> None:
        self._source = source
        self._into = into
        self._frames: list[FrameRange] = []
        self._finished = False

    def observe(self, step: Step[Tensor, Path]) -> None:
        """Range this step's frame, recording it under the file it came from.

        Raises:
            ValueError: If the frame holds no finite value, naming the file.
        """
        frame = step.require()
        path = step.require_extra()

        found = finite_range(frame)
        if found is None:
            msg = f"no finite value in {self._source}/{path.name}"
            raise ValueError(msg)

        self._frames.append(FrameRange(path.name, *found))

    def __call__(self, step: Step[Tensor, Path]) -> None:
        """`observe`, so the collector attaches to a stage as the hook it is."""
        self.observe(step)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            return

        self._finished = True
        self._into.take(self)

    def collected(self) -> SequenceRange:
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
    document = {**(provenance or {}), "dataset": as_dict(dataset)}
    path = ensure_file_extension(path, ".json", add=True)

    with StagedFile(
        path, overwrite=overwrite, make_parents=True, encoding="utf-8"
    ) as file:
        file.write(json.dumps(document, indent=2))

    return path


class DatasetRangeCollector:
    def __init__(
        self,
        path: Path,
        provenance: Mapping[str, object] | None = None,
        *,
        overwrite: bool = False,
    ) -> None:
        self._sequences: list[SequenceRange] = []
        self._path = path
        self._provenance = provenance
        self._overwrite = overwrite

    def collector_for(self, source: str) -> SequenceRangeCollector:
        return SequenceRangeCollector(source, self)

    def take(self, collector: SequenceRangeCollector) -> None:
        self._sequences.append(collector.collected())

    def fresh(self) -> DatasetRangeCollector:
        return DatasetRangeCollector(
            path=self._path,
            provenance=self._provenance,
            overwrite=self._overwrite,
        )

    def merge(self, other: DatasetRangeCollector) -> None:
        self._sequences.extend(other._sequences)

    def collected(self) -> DatasetRange:
        return DatasetRange(tuple(self._sequences))

    def save(self) -> Path:
        return save_range_document(
            self.collected(),
            self._path,
            provenance=self._provenance,
            overwrite=self._overwrite,
        )
