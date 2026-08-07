from __future__ import annotations

__all__ = (
    "Hook",
    "Reporting",
    "SequenceStage",
    "SideBranch",
    "Stage",
    "StageFactory",
    "Step",
)

import logging
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Self, override, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from kaparoo.data import DataSequence

    from iivs_cardio.common.device import Device


_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Step[T, E = None]:
    """One index of a stage, together with what was computed there.

    Both `value` and `extra` are optional because a stage may have nothing to
    give at an index: a quantity taken from a pair of frames has no answer at
    the first one, and a stage that carries no side information leaves `extra`
    unset. A hook that needs either should ask for it by name.

    Type Parameters:
        T: what the stage computes.
        E: the side information it carries, if any.

    Attributes:
        index: which index of the stage this is.
        value: what was computed there, or `None` when there was nothing.
        extra: side information about the value, such as where it came from.
    """

    index: int
    value: T | None
    extra: E | None = None

    def require(self) -> T:
        """Return the value, refusing a step that has none.

        Raises:
            ValueError: If the step holds no value.
        """
        if self.value is None:
            msg = f"step {self.index} holds no value"
            raise ValueError(msg)

        return self.value

    def require_extra(self) -> E:
        """Return the side information, refusing a step that has none.

        Raises:
            ValueError: If the step holds nothing beside its value.
        """
        if self.extra is None:
            msg = f"step {self.index} holds nothing beside its value"
            raise ValueError(msg)

        return self.extra


type Hook[T, E = None] = Callable[[Step[T, E]], None]


class SideBranch[S, T, E = None](Protocol):
    """A source of hooks, one for each thing a run will pass through.

    A branch that has to gather across a whole run, rather than finish with one
    item, may also be a context manager; a driver opens it around the run. That
    is an option rather than a requirement, so a branch whose work ends with the
    item it watched stays a plain object.

    Type Parameters:
        S: what a hook is made for, such as one sequence of a dataset.
        T: what the hooks receive.
        E: the side information those steps carry.
    """

    def get_hook(self, source: S, /) -> Hook[T, E]: ...


@runtime_checkable
class Reporting(Protocol):
    """Something that can say in one line what it did."""

    def report(self) -> str | None: ...


class StageFactory(Protocol):
    """The whole of what a driver needs to run a job's items.

    A driver asks how many items there are, runs each on a device it chose, and
    names its log lines after the factory. Anything that has to be opened once
    for the whole run rather than per item is opened by `running`.
    """

    @property
    def name(self) -> str: ...

    def __len__(self) -> int: ...

    def get_name(self, index: int, /) -> str: ...

    def run_stage(self, index: int, device: Device, /) -> None: ...

    def running(self) -> AbstractContextManager[Any]: ...


def _close_hooks(
    hooks: list[AbstractContextManager[Any]], error: BaseException | None
) -> None:
    """Close every hook in `hooks`, in reverse, each told only about `error`.

    Args:
        hooks: the hooks to close, in the order they were opened.
        error: what the walk they bracket ended with, or `None` if it finished.

    Raises:
        BaseException: What closing raised, once every hook has been closed
            rather than at the one that raised it. Only the first is carried,
            since a second means the destination itself has gone.
    """
    failure: BaseException | None = None

    for hook in reversed(hooks):
        try:
            if error is None:
                hook.__exit__(None, None, None)
            else:
                hook.__exit__(type(error), error, error.__traceback__)
        except BaseException as closing:  # noqa: BLE001
            failure = failure or closing

    if failure is not None:
        raise failure


class Stage[T, E = None](ABC):
    """A source addressed by index, which computes each index only once.

    Asking for an index computes it on a miss and keeps it for a short while, so
    two consumers reading the same index share one computation. Every hook is
    called exactly once per index, on the computation rather than on the access,
    which is what lets a value be read again without being written twice.

    A stage may be built over other stages. `run` then opens the hooks of the
    whole chain together, each one once even when two paths reach it.

    Type Parameters:
        T: what this stage computes.
        E: the side information it carries about each value.

    Args:
        sources: stages this one is built over, opened along with it.
        window: how many recent indices to keep. One is enough to walk the
            indices in order; a consumer that looks back needs at least as many
            as it looks back by, or the value is computed a second time.

    Raises:
        ValueError: If `window` is less than one.
    """

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
        self._walked = False

    @property
    def sources(self) -> tuple[Stage[Any, Any], ...]:
        """The stages this one is built over."""
        return self._sources

    @property
    def hooks(self) -> tuple[Hook[T, E], ...]:
        """The hooks registered on this stage, in the order they were added."""
        return tuple(self._hooks)

    def register_hooks(self, *hooks: Hook[T, E]) -> Self:
        """Add hooks to be called once for each index this stage computes.

        Returns:
            This stage, so registering can be chained onto construction.
        """
        self._hooks.extend(hooks)

        return self

    @abstractmethod
    def __len__(self) -> int:
        """The number of indices this stage answers for."""
        ...

    @abstractmethod
    def _compute(self, index: int) -> T | None:
        """Produce the value at `index`, or `None` when there is none."""
        ...

    def _describe(self, index: int) -> E | None:  # noqa: ARG002
        """Side information about `index`. Nothing unless a subclass says so."""
        return None

    def __getitem__(self, index: int) -> Step[T, E]:
        """Return the step at `index`, computing it if it is not still held.

        A freshly computed step goes to every hook before it is returned. One
        that was still held is returned as it stands, so the hooks see it once.

        Raises:
            IndexError: If `index` is outside the range this stage answers for.
        """
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
        """Yield every step in order, from the first index to the last."""
        for index in range(len(self)):
            yield self[index]

    def _forget(self, index: int) -> None:
        """Drop everything held from further back than the window reaches."""
        self._reach = max(self._reach, index)
        oldest = self._reach - self._window + 1
        self._cache = {i: s for i, s in self._cache.items() if i >= oldest}

    def _notify(self, step: Step[T, E]) -> None:
        """Give `step` to every hook, unless this index has already gone out."""
        if step.index in self._notified:
            _logger.debug("step %d refilled (window %d)", step.index, self._window)
            return

        self._notified.add(step.index)
        for hook in self._hooks:
            hook(step)

    def _all_hooks(self, seen: set[int] | None = None) -> Iterator[Hook[Any, Any]]:
        """Yield the hooks of this stage and its sources, each stage once."""
        seen = set() if seen is None else seen
        if id(self) in seen:
            return

        seen.add(id(self))
        yield from self._hooks

        for source in self._sources:
            yield from source._all_hooks(seen)  # noqa: SLF001

    def run(self) -> None:
        """Walk every index once, with the whole chain's hooks open around it.

        Hooks that are context managers are opened before the walk and closed
        after it, so a hook that gathers across indices commits at the end and
        leaves nothing half finished if the walk stops early.

        Every hook is closed against the walk's own outcome and never against
        another hook's. They are siblings writing separate outputs, so one that
        cannot commit must not tell the rest their work failed. A hook cannot
        suppress the walk's failure either: the driver's verdict for the whole
        run is that exception reaching it.

        One walk per stage. Every index a hook has seen is remembered so it sees
        each exactly once, and that memory is what a second walk would run into:
        it would open the hooks, fire none of them, and close them again.

        Raises:
            RuntimeError: If this stage has been walked before.
        """
        if self._walked:
            msg = f"{type(self).__name__} has been run: build a stage per walk"
            raise RuntimeError(msg)

        self._walked = True
        opened: list[AbstractContextManager[Any]] = []

        try:
            for hook in self._all_hooks():
                managed = isinstance(hook, AbstractContextManager)
                if managed and not any(hook is other for other in opened):
                    hook.__enter__()
                    opened.append(hook)

            for _ in self:
                pass
        except BaseException as error:
            _close_hooks(opened, error)
            raise

        _close_hooks(opened, None)


class SequenceStage[T, M](Stage[T, M]):
    """A stage that reads an existing sequence rather than computing anything.

    This is what makes a cached result and a freshly computed one the same to
    whoever reads them: both arrive as a stage, so hooks and chaining work the
    same either way.

    Type Parameters:
        T: what the sequence holds.
        M: the metadata it carries per item.

    Args:
        sequence: the sequence to read.
        window: how many recent indices to keep, as for any stage.
    """

    def __init__(self, sequence: DataSequence[T, M], *, window: int = 1) -> None:
        super().__init__(window=window)
        self._sequence = sequence

    def __len__(self) -> int:
        """The number of items in the sequence."""
        return len(self._sequence)

    @override
    def _compute(self, index: int) -> T:
        return self._sequence.get_item(index)

    @override
    def _describe(self, index: int) -> M:
        return self._sequence.get_meta(index)
