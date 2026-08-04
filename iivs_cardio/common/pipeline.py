from __future__ import annotations

__all__ = ("Hook", "SequenceStage", "SideBranch", "Stage", "Step")

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Self

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from kaparoo.data import DataSequence


@dataclass(frozen=True, slots=True)
class Step[T, E = None]:
    index: int
    value: T | None
    extra: E | None = None

    def require(self) -> T:
        if self.value is None:
            msg = f"step {self.index} holds no value"
            raise ValueError(msg)

        return self.value

    def require_extra(self) -> E:
        if self.extra is None:
            msg = f"step {self.index} holds nothing beside its value"
            raise ValueError(msg)

        return self.extra


type Hook[T, E = None] = Callable[[Step[T, E]], None]


class SideBranch[S, T, E = None](Protocol):
    def hook_for(self, name: str, source: S, /) -> Hook[T, E]: ...


class Stage[T, E = None](ABC):
    def __init__(self, *sources: Stage[Any, Any], window: int = 1) -> None:
        if window < 1:
            msg = f"invalid window {window}: expected 1 or more"
            raise ValueError(msg)

        self._sources = sources
        self._window = window
        self._cache: dict[int, Step[T, E]] = {}
        self._reach = -1

        self._hooks: list[Hook[T, E]] = []
        self._notified: set[int] = set()

    @property
    def sources(self) -> tuple[Stage[Any, Any], ...]:
        return self._sources

    def register_hooks(self, *hooks: Hook[T, E]) -> Self:
        self._hooks.extend(hooks)

        return self

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def _compute(self, index: int) -> T | None: ...

    def _describe(self, index: int) -> E | None:  # noqa: ARG002
        return None

    def __getitem__(self, index: int) -> Step[T, E]:
        if not 0 <= index < len(self):
            msg = f"step {index} is outside 0..{len(self) - 1}"
            raise IndexError(msg)

        if (cached := self._cache.get(index)) is not None:
            return cached

        step: Step[T, E] = Step(index, self._compute(index), self._describe(index))
        self._cache[index] = step
        self._forget(index)
        self._notify(step)

        return step

    def __iter__(self) -> Iterator[Step[T, E]]:
        for index in range(len(self)):
            yield self[index]

    def _forget(self, index: int) -> None:
        self._reach = max(self._reach, index)
        oldest = self._reach - self._window + 1
        self._cache = {i: s for i, s in self._cache.items() if i >= oldest}

    def _notify(self, step: Step[T, E]) -> None:
        if step.index in self._notified:
            return

        self._notified.add(step.index)
        for hook in self._hooks:
            hook(step)

    def _all_hooks(self, seen: set[int] | None = None) -> Iterator[Hook[Any, Any]]:
        seen = set() if seen is None else seen
        if id(self) in seen:
            return

        seen.add(id(self))
        yield from self._hooks

        for source in self._sources:
            yield from source._all_hooks(seen)  # noqa: SLF001

    def run(self) -> None:
        with ExitStack() as stack:
            for hook in self._all_hooks():
                if isinstance(hook, AbstractContextManager):
                    stack.enter_context(hook)

            for _ in self:
                pass


class SequenceStage[T, M](Stage[T, M]):
    def __init__(self, sequence: DataSequence[T, M], *, window: int = 1) -> None:
        super().__init__(window=window)
        self._sequence = sequence

    def __len__(self) -> int:
        return len(self._sequence)

    def _compute(self, index: int) -> T:
        return self._sequence.get_item(index)

    def _describe(self, index: int) -> M:
        return self._sequence.get_meta(index)
