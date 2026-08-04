from __future__ import annotations

__all__ = (
    "CompositeRange",
    "DatasetRange",
    "FrameRange",
    "RangeDocument",
    "SequenceRange",
    "SequenceRangeMeter",
    "ValueRange",
    "as_dict",
    "save_range_document",
)

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, Self
from urllib.parse import quote

from kaparoo.filesystem import StagedFile, ensure_file_extension

from iivs_cardio.common.range import finite_range

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path
    from types import TracebackType

    from iivs.dhm.data.phase import PhaseFileFolder
    from kaparoo.filesystem.types import StrPath
    from torch import Tensor

    from iivs_cardio.common.pipeline import Step

PARTS_SUFFIX = ".parts"


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


def _source_first(items: list[tuple[str, Any]]) -> dict[str, Any]:
    return dict(sorted(items, key=lambda item: item[0] != "source"))


def as_dict(ranged: CompositeRange) -> dict[str, Any]:
    return asdict(ranged, dict_factory=_source_first)


def _sequence_from_dict(document: Mapping[str, Any]) -> SequenceRange:
    frames = tuple(
        FrameRange(frame["source"], frame["min_value"], frame["max_value"])
        for frame in document["frames"]
    )

    return SequenceRange(document["source"], frames)


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


class SequenceRangeMeter:
    """Range every frame of one sequence, and leave the result where it can be found.

    A hook, so it reads what the stage was going to yield anyway rather than
    forcing a second pass over the folder.

    On a clean exit it writes its `SequenceRange` into `parts` as its own file.
    That is what carries the answer out of a worker: the document that folds
    them runs in the parent, and a `shared_objects` copy travels one way only.
    A traversal that died writes nothing -- the frames it did see are a prefix,
    and a prefix folded into the dataset's bounds is a hole nobody would see.

    Args:
        source: What names this sequence in the dataset.
        parts: The folder every sequence of this run leaves its result in.
    """

    def __init__(self, source: str, parts: Path) -> None:
        self._source = source
        self._parts = parts
        self._frames: list[FrameRange] = []

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
        """`observe`, so the meter registers on a stage as the hook it is."""
        self.observe(step)

    def collected(self) -> SequenceRange:
        """What this sequence ranged to.

        Raises:
            ValueError: If no frame was seen, which no fold can answer for.
        """
        return SequenceRange(self._source, tuple(self._frames))

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

        # Percent-encoding, because a name is a path and would otherwise nest.
        path = self._parts / f"{quote(self._source, safe='')}.json"
        with StagedFile(
            path, overwrite=True, make_parents=True, encoding="utf-8"
        ) as file:
            file.write(json.dumps(as_dict(self.collected()), indent=2))


@dataclass(frozen=True, slots=True)
class RangeDocument:
    """Where a run reports the value range of everything it read.

    A side branch: it hands each sequence a meter, and folds what the meters
    left behind once every sequence has run. The fold is the parent's alone --
    a worker's copy of this only ever mints meters, since only the parent is
    opened around the run.

    The parts survive the run. A run that died part-way leaves the sequences it
    did finish, and entering clears them, so a re-run into the same output
    directory cannot fold a previous run's answers in with its own.

    Args:
        path: Where the folded document goes. `.json` is added if absent.
        provenance: What a reader of the numbers cannot get from the numbers.
        overwrite: Whether to replace a document already at `path`.
    """

    path: Path
    provenance: Mapping[str, object] | None = None
    overwrite: bool = False

    @property
    def parts(self) -> Path:
        """The folder the sequences of this run leave their results in."""
        return self.path.with_suffix(PARTS_SUFFIX)

    def hook_for(self, name: str, origin: PhaseFileFolder) -> SequenceRangeMeter:  # noqa: ARG002
        """The meter for the sequence `name`, which `origin` was read out of."""
        return SequenceRangeMeter(name, self.parts)

    def collected(self) -> DatasetRange:
        """Fold what the sequences of this run left behind, in name order.

        Ordered by name rather than by arrival, so `min_index` and `max_index`
        mean the same thing whichever worker finished first.

        Raises:
            ValueError: If no sequence left a result.
        """
        parts = sorted(self.parts.glob("*.json"))
        sequences = tuple(
            _sequence_from_dict(json.loads(part.read_text(encoding="utf-8")))
            for part in parts
        )

        return DatasetRange(tuple(sorted(sequences, key=lambda s: s.source)))

    def save(self) -> Path:
        """Write the folded document."""
        return save_range_document(
            self.collected(),
            self.path,
            provenance=self.provenance,
            overwrite=self.overwrite,
        )

    def __enter__(self) -> Self:
        for stale in self.parts.glob("*.json"):
            stale.unlink()

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
