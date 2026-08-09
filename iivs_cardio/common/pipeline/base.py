from __future__ import annotations

__all__ = (
    "Holding",
    "Hook",
    "Reporting",
    "Reverting",
    "SequenceStage",
    "SideBranch",
    "Stage",
    "StageFactory",
    "Step",
    "close_together",
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


# ========================== #
#           Steps            #
# ========================== #


@dataclass(frozen=True, slots=True)
class Step[T, E = None]:
    """One index of a stage, together with what was computed there.

    Both `value` and `extra` are optional because a stage may have nothing to
    give at an index: a quantity taken from a pair of frames has no answer at
    the first one, and a stage that carries no side information leaves `extra`
    unset. A hook that needs either should ask for it by name.

    Type Parameters:
        T: The type of what the stage computes.
        E: The type of the side information it carries, if any.

    Attributes:
        index: The index of the stage this step is.
        value: The result computed there, or `None` where there was none.
        extra: The side information about the value, such as where it came
            from. Defaults to None.
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


# ========================== #
#        Capabilities        #
# ========================== #


class SideBranch[S, T, E = None](Protocol):
    """A source of hooks, one for each thing a run will pass through.

    One that gathers across a whole run rather than finishing with the item it
    watched may also be a context manager, which a driver opens around the run.

    Type Parameters:
        S: The type a hook is made for, such as one sequence of a dataset.
        T: The type the hooks receive.
        E: The type of the side information those steps carry.
    """

    def get_hook(self, source: S, /) -> Hook[T, E] | None:
        """Return the hook that will watch `source`, or `None` to leave it.

        Args:
            source: The thing the hook is to be made for.

        Returns:
            The hook, or `None` where this branch already holds what it would
            have produced for `source`.
        """
        ...


@runtime_checkable
class Holding(Protocol):
    """A branch that keeps an output of its own for each item it watched.

    A dataset grows and shrinks between runs, so an output may outlive the item
    it was made for. Only the branch knows what one of its outputs looks like on
    disk, which is why the question is asked here rather than of the run.
    """

    def list_unsourced(self) -> list[str]:
        """Return the names of the outputs the source no longer holds.

        Returns:
            The names, which are this branch's to remove or keep by its own
            policy. The same absence is what a half mounted share and a
            misspelt subpath produce, so naming them is all this can do.
        """
        ...


@runtime_checkable
class Reporting(Protocol):
    """Something that can say in one line what it did."""

    def report(self) -> str | None:
        """Return one line naming what this committed.

        Returns:
            The line, or `None` where nothing was committed. Read after the
            close, so a line only ever describes an output that is there.
        """
        ...


@runtime_checkable
class Reverting(Protocol):
    """A hook that can take back what closing it cleanly put in place.

    Closing one after another is not one commit, so a hook whose output reached
    disk may find the next could not, leaving it standing for work the item did
    not finish. Only worth implementing where taking it back is possible: one
    that replaced an output already there cannot put that one back.
    """

    def revert(self) -> None:
        """Take back what closing this cleanly put in place.

        Called only where another hook of the same item could not commit, and
        only on hooks that closed without raising. Doing nothing is a valid
        answer for one that committed nothing.
        """
        ...


class StageFactory(Protocol):
    """The whole of what a driver needs to run a job's items.

    A driver asks how many items there are, runs each on a device it chose, and
    names its log lines after the factory. Anything that has to be opened once
    for the whole run rather than per item is opened by `running`.
    """

    @property
    def name(self) -> str:
        """The run's name, which every line of it is filed under."""
        ...

    def __len__(self) -> int:
        """The number of items this run was given."""
        ...

    def get_name(self, index: int, /) -> str:
        """Return what the item at `index` is called.

        Args:
            index: The item to name.

        Returns:
            The name, which is what a log line and a retry list carry rather
            than the index.
        """
        ...

    def run_stage(self, index: int, device: Device, /) -> bool:
        """Carry out the item at `index` on `device`.

        Args:
            index: The item to carry out.
            device: The device to carry it out on.

        Returns:
            Whether the item was computed. One that every branch already holds
            what it needs for is not read at all, and the frames that would
            have cost are the whole point of asking first.
        """
        ...

    def running(self) -> AbstractContextManager[Any]:
        """Return the bracket around the whole run.

        Returns:
            A context manager holding open whatever outlives one item, such as
            a branch that gathers across the dataset.
        """
        ...


# ========================== #
#           Stages           #
# ========================== #


class Stage[T, E = None](ABC):
    """A source addressed by index, whose hooks fire once for each of them.

    An index is computed on a miss and kept for as long as the window reaches
    back, so two consumers of the same index share one computation. A miss
    computes it again, and the hooks do not fire a second time: that is what
    lets a value be read again without being written twice. A stage may be
    built over others, and `run` then opens the whole chain's hooks together.

    Type Parameters:
        T: The type of what this stage computes.
        E: The type of the side information it carries about each value.

    Args:
        sources: The stages this one is built over, opened along with it.
        window: How many recent indices to keep. One is enough to walk the
            indices in order; a consumer that looks back needs at least as many
            as it looks back by, or the value is computed a second time.
            Defaults to 1.

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
        """Produce the value at `index`, or `None` when there is none.

        A pure function of `index`: an index the window has let go of is
        computed again on the next read, and the hooks that fired on the first
        one do not fire on the second. Two answers that differ would leave what
        was written and what the consumer holds describing different things.

        Args:
            index: The index to produce the value for, already in range.

        Returns:
            The value, or `None` where the stage has nothing at that index.
        """
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
        after it against the walk's own outcome, never against another hook's,
        so one that cannot commit neither tells the rest their work failed nor
        hides the failure from the driver. One walk per stage: the memory that
        fires each hook once per index would leave a second walk opening the
        hooks, firing none of them, and closing them again.

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
            close_together(opened, error)
            raise

        close_together(opened, None)


class SequenceStage[T, M](Stage[T, M]):
    """A stage whose values come from a sequence rather than from other stages.

    It has no sources of its own: asking for an index asks the sequence for that
    item, and whether the sequence reads it off disk or works it out on the spot
    is the sequence's own business. That is what makes a cached result and a
    freshly computed one the same to whoever reads them: both arrive as a stage,
    so hooks and chaining work the same either way.

    Type Parameters:
        T: The type of one item, as the sequence yields it.
        M: The type of the metadata the sequence carries per item.

    Args:
        sequence: The sequence to take the values from.
        window: How many recent indices to keep, as for any stage. Defaults to 1.
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


# ========================== #
#          Closing           #
# ========================== #


def close_together(
    opened: list[AbstractContextManager[Any]], error: BaseException | None
) -> None:
    """Close everything in `opened`, in reverse, each told only about `error`.

    They write separate outputs rather than nesting one inside another, so one
    that cannot commit must not tell the rest their work failed, and an
    `ExitStack` does exactly that. Where one does fail, those that can take
    their output back are asked to, and a revert that fails is logged rather
    than raised: the failure being answered is the one worth carrying.

    Args:
        opened: The context managers to close, in the order they were opened.
        error: The exception the block they bracket ended with, or `None`
            if it finished.

    Raises:
        BaseException: What closing raised, once everything has been closed
            rather than at the one that raised it. Only the first is carried,
            since a second means the destination itself has gone.
    """
    failure: BaseException | None = None
    closed: list[AbstractContextManager[Any]] = []

    for hook in reversed(opened):
        try:
            if error is None:
                hook.__exit__(None, None, None)
            else:
                hook.__exit__(type(error), error, error.__traceback__)
        except BaseException as closing:  # noqa: BLE001
            failure = failure or closing
        else:
            closed.append(hook)

    if failure is None:
        return

    for hook in closed:
        if isinstance(hook, Reverting):
            try:
                hook.revert()
            except Exception:
                _logger.exception("could not take back %r", hook)

    raise failure
