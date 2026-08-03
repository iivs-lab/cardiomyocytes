from __future__ import annotations

__all__ = ("Hook", "Node", "Slot", "Steps")

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from kaparoo.data import DataSequence


@dataclass(frozen=True, slots=True)
class Slot[T]:
    """One node's value for one step of a sequence, or the absence of one.

    `index` is what the value describes, not where it arrived. A node buffering a
    temporal window falls behind its input by as many steps as that window is
    deep, so at any moment the nodes of one pipeline sit at different indices --
    a variance over single frames can be reporting step 40 while an acceleration
    three differences downstream is still reporting 38. Alignment between nodes,
    and every record a hook keeps, is therefore by `index` and never by arrival
    order.

    `value` is absent wherever the node cannot produce one. Under the forward
    convention -- `node[i]` reads from step `i` onward -- that is the tail, where
    the future a step would need does not exist. Absence propagates rather than
    being skipped: a node handed an absent input has nothing to compute from and
    passes an absent slot on, so the index still reaches the end of the chain and
    a hook can record which steps went unfilled.

    Type Parameters:
        T: What the node yields when it has something -- a frame, a flow field,
            a mapping of metrics. Absence is `Slot`'s to express, so `T` itself
            should not be optional.
    """

    index: int
    value: T | None

    def require(self) -> T:
        """The value, where absence would be a defect rather than a tail.

        A node reading a source fills every step it yields, so a consumer of one
        has nothing to branch on -- asking here says that and narrows the type in
        the same move, instead of a check no test can reach. Read `value`
        directly wherever absence is expected: below a node holding a window it
        is the ordinary end of the stream.

        Raises:
            ValueError: If this slot holds no value.
        """
        if self.value is None:
            msg = f"step {self.index} holds no value"
            raise ValueError(msg)

        return self.value


type Hook[T] = Callable[[Slot[T]], None]


class Node[T](ABC):
    """One stage of a pipeline: slots in order, with hooks watching them pass.

    Hooks belong to the node rather than to whoever drains the chain, because a
    chain is drained only at its end -- hanging them off the drain would leave
    every stage but the last unobservable. Saving filtered frames watches the
    filter, scoring a flow watches the estimator, and both run in one traversal
    of a chain whose end is something else entirely.

    A hook observes; it never consumes. Its slot is passed downstream either way,
    and its return value is ignored, so whatever accumulates a per-step result --
    and owns whatever lifetime that takes -- is the object that supplied the
    hook, not this one. A writer therefore opens and commits around the
    traversal.

    Subclasses implement `produce`, which is where the stage's own work lives.

    Type Parameters:
        T: What this stage yields at a step it can fill.
    """

    def __init__(self, source: Node[Any] | None = None) -> None:
        self._source = source
        self._hooks: list[Hook[T]] = []

    def _chain(self) -> Iterator[Node[Any]]:
        """This node and everything it draws from, nearest first."""
        node: Node[Any] | None = self
        while node is not None:
            yield node
            node = node._source  # noqa: SLF001

    def attach(self, *hooks: Hook[T]) -> Self:
        """Register `hooks` to be called with every slot this node yields.

        Returns this node, so a stage can be built and watched in one expression.
        Hooks are called in the order attached, before the slot goes downstream.
        """
        self._hooks.extend(hooks)

        return self

    @abstractmethod
    def produce(self) -> Iterator[Slot[T]]:
        """Yield this stage's slots, in index order and one per step.

        A stage holding a temporal window owes a flush once its own input ends --
        the steps it buffered are still its to give. Returning when the input
        does drops the tail of everything downstream, silently and in proportion
        to how deep the chain is.
        """

    def run(self) -> None:
        """Pull every slot, holding open whatever the chain's hooks need held.

        The end of a chain is often wanted for nothing but its hooks -- a scan
        writing a range document reads every frame and keeps no frame. Saying so
        beats a loop over a discarded name, which reads like an oversight.

        A hook that is a context manager is entered here and left on the way
        out, across the whole chain rather than this node alone: writers hang off
        the stages whose output they save, which is rarely the last one. That
        makes their commits all-or-nothing together -- a run that dies part-way
        leaves no folder behind rather than some of them -- and it keeps the
        caller from stacking a `with` per writer as the chain grows.

        Exhausting the chain is also what flushes it: every stage owes the steps
        it buffered once its input ends, and they arrive only while something
        keeps pulling. Call this on the *last* stage; running a middle one leaves
        everything below it unfed.
        """
        with ExitStack() as stack:
            for node in self._chain():
                for hook in node._hooks:  # noqa: SLF001
                    if isinstance(hook, AbstractContextManager):
                        stack.enter_context(hook)

            for _ in self:
                pass

    def __iter__(self) -> Iterator[Slot[T]]:
        for slot in self.produce():
            for hook in self._hooks:
                hook(slot)

            yield slot


class Steps[T, M](Node[tuple[T, M]]):
    """The source node: a `DataSequence` read in order, one slot per step.

    Each slot carries the pair the sequence itself yields -- the item, and
    whatever it records about where the item came from. The second half travels
    because a downstream node cannot reach back into the sequence for it without
    re-coupling to it, and a frame's own name is what a range or a report is
    written against.

    Indices are positions in `sequence`, which is what makes them positions in
    every node downstream. A sequence that is already a strided view of another
    therefore numbers from `0` over what it exposes, not over what it reads.
    """

    def __init__(self, sequence: DataSequence[T, M]) -> None:
        super().__init__()
        self._sequence = sequence

    def produce(self) -> Iterator[Slot[tuple[T, M]]]:
        """Yield every step in order. Never absent -- a read gives an item or raises."""
        for index in range(len(self._sequence)):
            yield Slot(index, self._sequence.get_pair(index))
