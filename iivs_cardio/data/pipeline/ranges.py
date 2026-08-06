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
from typing import TYPE_CHECKING, Any, Protocol, Self, override

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


DOCUMENT_EXT = ".json"


def _entry[T](
    document: Mapping[str, Any],
    key: str,
    kind: type[T] | tuple[type[T], ...],
) -> T:
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
    source: str
    min_value: float
    max_value: float

    def __str__(self) -> str:
        return f"[{self.min_value:.4g}, {self.max_value:.4g}]"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    @abstractmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Self:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class FrameRange(ValueRange):
    @override
    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Self:
        return cls(
            _entry(document, "source", str),
            float(_entry(document, "min_value", (int, float))),
            float(_entry(document, "max_value", (int, float))),
        )


@dataclass(frozen=True, slots=True)
class CompositeRange(ValueRange, ABC):
    min_value: float = field(init=False)
    max_value: float = field(init=False)
    min_index: int = field(init=False)
    max_index: int = field(init=False)

    @property
    @abstractmethod
    def parts(self) -> Sequence[ValueRange]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.parts)

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
class SequenceRange(CompositeRange):
    frames: tuple[FrameRange, ...]

    @property
    def parts(self) -> tuple[FrameRange, ...]:
        return self.frames

    @classmethod
    @override
    def from_dict(cls, document: Mapping[str, Any]) -> SequenceRange:
        source = _entry(document, "source", str)
        frames = _entry(document, "frames", (list, tuple))
        frames = tuple(FrameRange.from_dict(frame) for frame in frames)
        return cls(source, frames)


@dataclass(frozen=True, slots=True)
class DatasetRange(CompositeRange):
    sequences: tuple[SequenceRange, ...]

    @property
    def parts(self) -> tuple[SequenceRange, ...]:
        return self.sequences

    @classmethod
    @override
    def from_dict(cls, document: Mapping[str, Any]) -> DatasetRange:
        source = _entry(document, "source", str)
        sequences = _entry(document, "sequences", (list, tuple))
        sequences = tuple(SequenceRange.from_dict(sequence) for sequence in sequences)
        return cls(source, sequences)


class SequenceRangeMeter:
    def __init__(self, root: StrPath, source: str) -> None:
        root = ensure_dir_exists(root, make=True)
        file = f"{source}{DOCUMENT_EXT}"

        self._path = root / file
        self._source = source
        self._frames: list[FrameRange] = []
        self._cached: SequenceRange | None = None

    def __call__(self, step: Step[Tensor, Path]) -> None:
        self.measure(step)

    def measure(self, step: Step[Tensor, Path]) -> None:
        frame = step.require()
        path = step.require_extra()

        found = finite_range(frame)
        if found is None:
            msg = f"no finite value in {path.name} (sequence: {self._source})"
            raise ValueError(msg)

        self._frames.append(FrameRange(path.name, *found))

    def to_range(self) -> SequenceRange:
        if self._cached is None or len(self._cached) != len(self._frames):
            self._cached = SequenceRange(self._source, tuple(self._frames))
        return self._cached

    def report(self) -> str | None:
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
    """How much of what a run set out to cover it actually did."""

    total: int
    covered: int
    skipped: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
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
    PARTS_SUFFIX = ".parts"

    def __init__(
        self,
        path: Path,
        source: str,
        expected: Sequence[str],
        settings: Mapping[str, object] | None = None,
        *,
        overwrite: bool = False,
    ) -> None:
        if not expected:
            msg = "no sequence to cover: `expected` must name at least one"
            raise ValueError(msg)

        self.path = ensure_file_extension(path, DOCUMENT_EXT, add=True)
        self.parts_root = self.path.with_suffix(self.PARTS_SUFFIX)

        self.source = source
        self.expected = tuple(dict.fromkeys(expected))
        self.settings = settings
        self.overwrite = overwrite

        self._saved: DatasetRange | None = None

    def get_hook(self, source: Named) -> SequenceRangeMeter:
        return SequenceRangeMeter(self.parts_root, source.name)

    def list_parts(self) -> list[Path]:
        parts = search_files(
            self.parts_root,
            name_filter=EndsWith(DOCUMENT_EXT),
            ordered=False,
        )

        return sorted(parts, key=self._source_of)

    def _source_of(self, part: Path) -> str:
        return stringify_path(part.with_suffix(""), after=self.parts_root)

    def to_range(self) -> DatasetRange:
        sequences = []

        for part in self.list_parts():
            with part.open(encoding="utf-8") as file:
                document = json.load(file)
                sequences.append(SequenceRange.from_dict(document))

        return DatasetRange(self.source, tuple(sequences))

    def get_coverage(self, dataset: DatasetRange) -> Coverage:
        covered = {sequence.source for sequence in dataset.sequences}
        skipped = tuple(name for name in self.expected if name not in covered)

        return Coverage(len(self.expected), len(covered), skipped)

    def save(self) -> Path:
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
        if exc_type is not None:
            return

        self.save()
