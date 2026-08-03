from __future__ import annotations

__all__ = ("Hook", "Slot", "drain", "steps")

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

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


def steps[T, M](sequence: DataSequence[T, M]) -> Iterator[Slot[tuple[T, M]]]:
    """Read `sequence` in order, one slot per step.

    The source node every chain starts from. Each slot carries the pair the
    sequence itself yields -- the item and whatever it records about where the
    item came from -- because a downstream node cannot reach back into the
    sequence for the second half without re-coupling to it, and a frame's own
    name is what a range or a report is written against.

    Indices here are positions in `sequence`, which is what makes them positions
    in every node downstream. A sequence that is already a strided view of
    another therefore numbers from `0` over what it exposes, not over what it
    reads.

    Yields:
        Every step, in order, each holding its `(item, metadata)` pair. Never
        absent: a read either produces an item or raises.
    """
    for index in range(len(sequence)):
        yield Slot(index, sequence.get_pair(index))


def drain[T](items: Iterable[Slot[T]], *hooks: Hook[T]) -> None:
    """Pull `items` to exhaustion, handing each slot to every hook in order.

    Hooks observe; they are not the consumer, and the chain runs whether or not
    any are given. Nothing here reads a hook's return value: a per-step result
    belongs to whatever provided the hook, and accumulating it -- along with
    whatever lifetime that takes -- is that object's job rather than the
    pipeline's. A writer therefore opens and commits itself *around* this call,
    not through it.

    Absent slots are passed on unfiltered. Which steps went unfilled is a fact
    some hooks record and others ignore, and only the hook can say which.

    Exhausting `items` is what drains the pipeline, but only because a node owes
    a flush once its own input ends -- a node holding a window still has the
    steps it buffered to give. One that simply returns when its input does drops
    the tail of everything downstream, silently and in proportion to how deep the
    chain is.

    Args:
        items: The end of the chain. Pulled once, so a generator is spent after.
        hooks: Called with every slot, in the order given.
    """
    for item in items:
        for hook in hooks:
            hook(item)
