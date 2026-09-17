from __future__ import annotations

__all__ = (
    "Hook",
    "Named",
    "SequenceStage",
    "SideBranch",
    "SingleUse",
    "Stage",
    "Step",
    "SupportsReport",
    "SupportsRevert",
    "SupportsUnsourced",
    "close_together",
)

import logging
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    Self,
    override,
    runtime_checkable,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from kaparoo.data import DataSequence


_logger = logging.getLogger(__name__)


# ========================== #
#           Steps            #
# ========================== #


@dataclass(frozen=True, slots=True)
class Step[T, E = None]:
    """One index, together with what was computed there.

    Both `value` and `extra` are optional because an index may have nothing to give: a
    quantity taken from a pair of frames has no answer at the first one, and side
    information is not always carried at all. A hook that needs either should ask for it
    by name.

    Type Parameters:
        T: The type of what was computed.
        E: The type of the side information carried beside it, if any.

    Attributes:
        index: The position this step stands for.
        value: The result computed there, or `None` where there was none.
        extra: The side information about the value, such as where it came from.
            Defaults to None.
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


class Named(Protocol):
    """Something with a name, which is what an output or a log line is filed under."""

    @property
    def name(self) -> str: ...


class SideBranch[S, T, E = None](Protocol):
    """A source of hooks, one for each thing a run will pass through.

    One that gathers across a whole run rather than finishing with the item it watched
    may also be a context manager, which a driver opens around the run.

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
            The hook, or `None` where this branch already holds what it would have
            produced for `source`.
        """
        ...


@runtime_checkable
class SupportsUnsourced(Protocol):
    """Something that can name the outputs no item stands behind any more.

    A dataset grows and shrinks between runs, so an output may outlive the item it was
    made for. Only what wrote an output knows what it looks like on disk, which is why
    it is asked rather than the run.
    """

    def list_unsourced(self) -> list[str]:
        """Return the names of the outputs the source no longer holds.

        Returns:
            The names. Removing or keeping them is a policy, and the same absence is
            what a half mounted share and a misspelt subpath produce, so naming them is
            all this can do.
        """
        ...


@runtime_checkable
class SupportsReport(Protocol):
    """Something that can say in one line what it committed."""

    def report(self) -> str | None:
        """Return one line naming what this committed.

        Returns:
            The line, or `None` where nothing was committed. Read after the close, so a
            line only ever describes an output that is there.
        """
        ...


@runtime_checkable
class SupportsRevert(Protocol):
    """Something that can take back what closing it cleanly put in place.

    Closing one after another is not one commit, so one whose output reached disk may
    find the next could not, leaving it standing for work the item did not finish. Only
    worth implementing where taking it back is possible: one that replaced an output
    already there cannot put that one back.
    """

    def revert(self) -> None:
        """Take back what closing this cleanly put in place.

        Called only where another closing over the same item could not commit, and only
        on those that closed without raising. Doing nothing is a valid answer for one
        that committed nothing.
        """
        ...


class SingleUse:
    """Something that can be opened once, and refuses to be opened again.

    What one opening did, such as a staged folder or the measurements taken, stays in
    it, so a second opening would start from that rather than from nothing. Giving one
    object twice in a list is not a second opening: whoever opens the list opens each
    object once.
    """

    # A class default, so a subclass adds nothing to its `__init__` to take this.
    _entered: bool = False

    def _mark_entered(self, name: object) -> None:
        """Record that this has been opened, refusing one opened before.

        Args:
            name: What this is opened for, such as the output it writes, which the
                refusal names.

        Raises:
            RuntimeError: If this has been opened before.
        """
        if self._entered:
            msg = f"{name} has been opened: build a new {type(self).__name__}"
            raise RuntimeError(msg)

        self._entered = True


# ========================== #
#          Closing           #
# ========================== #


def close_together(
    opened: Sequence[AbstractContextManager[object]], error: BaseException | None
) -> None:
    """Close everything in `opened`, in reverse, in place of a nested `with`.

    Every close is told about `error` alone, never about what another close raised, so a
    failure does not reach work that did not fail. Once all are closed, a failure sends
    `revert` to those that closed cleanly and can take their output back: the set
    commits whole or not at all, except for one that cannot revert and keeps what it
    committed. A revert that fails is logged rather than raised.

    Args:
        opened: The context managers to close, in the order they were opened. Read
            rather than emptied, so a caller keeps whatever it built.
        error: The exception the block they bracket ended with, or `None` if it
            finished.

    Raises:
        BaseExceptionGroup: Every closing that failed, once all of them have been closed
            rather than at the one that raised. Grouped even where only one did, so a
            caller reads them one way: these write to different destinations and can
            fail for unrelated reasons, which is the same reason none is told about
            another's failure, and carrying one would leave the rest unsaid.
    """
    failures: list[BaseException] = []
    closed: list[AbstractContextManager[object]] = []

    for hook in reversed(opened):
        try:
            if error is None:
                hook.__exit__(None, None, None)
            else:
                hook.__exit__(type(error), error, error.__traceback__)
        except BaseException as closing:  # noqa: BLE001
            failures.append(closing)
        else:
            closed.append(hook)

    if not failures:
        return

    for hook in closed:
        if isinstance(hook, SupportsRevert):
            try:
                hook.revert()
            except Exception:
                _logger.exception("could not take back %r", hook)

    msg = "could not close what the walk opened"
    raise BaseExceptionGroup(msg, failures)


# ========================== #
#           Stages           #
# ========================== #


class Stage[T, E = None](ABC):
    """A source addressed by index, whose hooks fire once for each of them.

    An index is computed on a miss and kept for as long as the window reaches back, so
    two consumers of the same index share one computation. A miss computes it again, and
    the hooks do not fire a second time: that is what lets a value be read again without
    being written twice. A stage may be built over others, and `run` then opens the
    whole chain's hooks together.

    Type Parameters:
        T: The type of what this stage computes.
        E: The type of the side information it carries about each value.

    Args:
        sources: The stages this one is built over, opened along with it.
        window: How many recent indices to keep. One is enough to walk the indices in
            order; a consumer that looks back needs at least as many as it looks back
            by, or the value is computed a second time. Defaults to 1.

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

    def all_hooks(self, *, upward: bool = False) -> tuple[Hook[Any, Any], ...]:
        """Return the hooks of this stage and every stage it is built over, each once.

        What `run` opens and what a caller asks to report once it is done, so the two
        cannot come to cover different hooks. Only the order differs. From this stage
        down by default, which is the order `run` opens them in. From the bottom of the
        chain up where `upward` is set, which is the order values are computed and
        stages close in, and so the order to ask what was committed.

        Within a stage the hooks keep the order they were registered in either way. A
        hook registered on two stages is one hook, and appears where the order first
        meets it.
        """
        chain = tuple(self._chain())
        held: list[Hook[Any, Any]] = []
        for stage in reversed(chain) if upward else chain:
            for hook in stage.hooks:
                if not any(hook is other for other in held):
                    held.append(hook)

        return tuple(held)

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
        raise NotImplementedError

    @abstractmethod
    def _compute(self, index: int) -> T | None:
        """Produce the value at `index`, or `None` when there is none.

        A pure function of `index`: an index the window has let go of is computed again
        on the next read, and the hooks that fired on the first one do not fire on the
        second. Two answers that differ would leave what was written and what the
        consumer holds describing different things.

        Args:
            index: The index to produce the value for, already in range.

        Returns:
            The value, or `None` where the stage has nothing at that index.
        """
        raise NotImplementedError

    def _describe(self, index: int) -> E | None:  # noqa: ARG002
        """Side information about `index`. Nothing unless a subclass says so."""
        return None

    def __getitem__(self, index: int) -> Step[T, E]:
        """Return the step at `index`, computing it if it is not still held.

        A freshly computed step goes to every hook before it is returned. One that was
        still held is returned as it stands, so the hooks see it once.

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

    def _chain(self, seen: set[int] | None = None) -> Iterator[Stage[Any, Any]]:
        """Yield this stage and those it is built over, top down, each once."""
        seen = set() if seen is None else seen
        if id(self) in seen:
            return

        seen.add(id(self))
        yield self

        for source in self._sources:
            yield from source._chain(seen)  # noqa: SLF001

    def run(self) -> None:
        """Walk every index once, with the whole chain's hooks open around it.

        Hooks that are context managers are opened before the walk and closed after it
        against the walk's own outcome, never against another hook's, so one that cannot
        commit neither tells the rest their work failed nor hides the failure from the
        driver. One walk per stage: the memory that fires each hook once per index would
        leave a second walk opening the hooks, firing none of them, and closing them
        again.

        Raises:
            RuntimeError: If this stage has been walked before.
            BaseExceptionGroup: What closing the hooks raised, as `close_together`
                groups it. The walk's own failure is the context it carries.
        """
        if self._walked:
            msg = f"{type(self).__name__} has been run: build a stage per walk"
            raise RuntimeError(msg)

        self._walked = True
        opened: list[AbstractContextManager[object]] = []

        try:
            for hook in self.all_hooks():
                if isinstance(hook, AbstractContextManager):
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

    It has no sources of its own: asking for an index asks the sequence for that item,
    and whether the sequence reads it off disk or works it out on the spot is the
    sequence's own business. That is what makes a cached result and a freshly computed
    one the same to whoever reads them: both arrive as a stage, so hooks and chaining
    work the same either way.

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
