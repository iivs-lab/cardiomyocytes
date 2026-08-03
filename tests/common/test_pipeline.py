from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Self

import pytest

from iivs_cardio.common.pipeline import Node, Slot, Steps

if TYPE_CHECKING:
    from collections.abc import Iterator


class _Sequence:
    """The slice of `DataSequence` that `Steps` uses, recording what it was asked."""

    def __init__(self, items: list[str]) -> None:
        self._items = items
        self.reads: list[int] = []

    def __len__(self) -> int:
        return len(self._items)

    def get_pair(self, index: int) -> tuple[str, str]:
        self.reads.append(index)
        return self._items[index], self._items[index].upper()


class _Fixed(Node[str]):
    """A node yielding what it was given, recording when it was pulled."""

    def __init__(self, *slots: Slot[str]) -> None:
        super().__init__()
        self._slots = slots
        self.produced: list[int] = []

    def produce(self) -> Iterator[Slot[str]]:
        for slot in self._slots:
            self.produced.append(slot.index)
            yield slot


def test_slot_keeps_index_and_value() -> None:
    slot = Slot(3, "frame")

    assert slot.index == 3
    assert slot.value == "frame"


def test_slot_holds_an_index_without_a_value() -> None:
    slot: Slot[str] = Slot(7, None)

    assert slot.index == 7
    assert slot.value is None


def test_slot_is_frozen() -> None:
    slot = Slot(0, "frame")

    with pytest.raises(dataclasses.FrozenInstanceError):
        slot.index = 1  # ty: ignore[invalid-assignment]


def test_slot_rejects_an_attribute_it_does_not_declare() -> None:
    slot = Slot(0, "frame")

    with pytest.raises(AttributeError):
        slot.source = "somewhere"  # ty: ignore[unresolved-attribute]


def test_require_returns_a_present_value() -> None:
    assert Slot(4, "frame").require() == "frame"


def test_require_refuses_an_absent_one() -> None:
    slot: Slot[str] = Slot(4, None)

    with pytest.raises(ValueError, match=r"step 4 holds no value"):
        slot.require()


def test_a_node_yields_what_it_produces() -> None:
    node = _Fixed(Slot(0, "a"), Slot(1, "b"))

    assert [slot.value for slot in node] == ["a", "b"]


def test_a_node_without_hooks_still_yields() -> None:
    node = _Fixed(Slot(0, "a"))

    assert [slot.index for slot in node] == [0]


def test_attach_returns_the_node_it_was_called_on() -> None:
    node = _Fixed(Slot(0, "a"))

    assert node.attach(lambda _: None) is node


def test_attached_hooks_see_every_slot() -> None:
    slots = [Slot(0, "a"), Slot(1, "b")]
    first: list[Slot[str]] = []
    second: list[Slot[str]] = []
    node = _Fixed(*slots).attach(first.append, second.append)

    node.run()

    assert first == slots
    assert second == slots


def test_hooks_fire_in_the_order_attached() -> None:
    calls: list[tuple[str, int]] = []

    def record(name: str) -> object:
        return lambda slot: calls.append((name, slot.index))

    node = _Fixed(Slot(0, "a"), Slot(1, "b"))
    node.attach(record("first")).attach(record("second"))

    node.run()

    assert calls == [("first", 0), ("second", 0), ("first", 1), ("second", 1)]


def test_a_hook_sees_each_slot_before_the_consumer_does() -> None:
    # A hook watches the stage it is attached to, so it runs where that stage
    # produced -- not after whatever is downstream has had the slot.
    order: list[str] = []
    node = _Fixed(Slot(0, "a"), Slot(1, "b"))
    node.attach(lambda slot: order.append(f"hook{slot.index}"))

    slots = iter(node)
    order.append(f"consumer{next(slots).index}")
    order.append(f"consumer{next(slots).index}")

    assert order == ["hook0", "consumer0", "hook1", "consumer1"]


def test_hooks_see_absent_slots_too() -> None:
    # Which steps went unfilled is a fact only the hook can decide to record.
    seen: list[Slot[str]] = []
    node = _Fixed(Slot(0, "a"), Slot[str](1, None)).attach(seen.append)

    node.run()

    assert [(slot.index, slot.value) for slot in seen] == [(0, "a"), (1, None)]


def test_steps_reads_a_sequence_in_order() -> None:
    slots = list(Steps(_Sequence(["a", "b", "c"])))

    assert [slot.index for slot in slots] == [0, 1, 2]
    assert [slot.require() for slot in slots] == [("a", "A"), ("b", "B"), ("c", "C")]


def test_steps_reads_each_index_once_and_only_when_pulled() -> None:
    sequence = _Sequence(["a", "b", "c"])
    walked = iter(Steps(sequence))

    assert sequence.reads == []
    next(walked)
    assert sequence.reads == [0]

    for _ in walked:
        pass
    assert sequence.reads == [0, 1, 2]


def test_steps_over_an_empty_sequence_yields_nothing() -> None:
    assert list(Steps(_Sequence([]))) == []


def test_run_pulls_a_node_to_its_flush() -> None:
    # A stage owes its buffered tail once its input ends; `run` gets that only
    # because it keeps pulling rather than stopping with the source.
    seen: list[Slot[str]] = []
    node = _Fixed(
        Slot(0, "a"), Slot(1, "b"), Slot[str](2, None)
    )  # the flush a windowing stage still owes
    node.attach(seen.append)

    node.run()

    assert [(slot.index, slot.value) for slot in seen] == [
        (0, "a"),
        (1, "b"),
        (2, None),
    ]


class _Passthrough(Node[str]):
    """A stage that just forwards, so a chain has something to walk."""

    def __init__(self, source: Node[str]) -> None:
        super().__init__(source)
        self._upstream = source

    def produce(self) -> Iterator[Slot[str]]:
        yield from self._upstream


class _Managed:
    """A hook that also wants opening and closing, the way a writer does."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def __call__(self, slot: Slot[str]) -> None:
        self.events.append(f"see{slot.index}")

    def __enter__(self) -> Self:
        self.events.append("open")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.events.append("abort" if exc_type is not None else "close")


def test_run_opens_and_closes_a_managed_hook() -> None:
    managed = _Managed()
    _Fixed(Slot(0, "a"), Slot(1, "b")).attach(managed).run()

    assert managed.events == ["open", "see0", "see1", "close"]


def test_run_opens_a_managed_hook_attached_upstream() -> None:
    # A writer hangs off the stage whose output it saves, which is rarely the
    # last one; running the end has to reach it anyway.
    managed = _Managed()
    source = _Fixed(Slot(0, "a")).attach(managed)

    _Passthrough(source).run()

    assert managed.events == ["open", "see0", "close"]


def test_a_failure_aborts_every_managed_hook_together() -> None:
    def explode(slot: Slot[str]) -> None:
        msg = "the hook gave up"
        raise RuntimeError(msg)

    first, second = _Managed(), _Managed()
    source = _Fixed(Slot(0, "a")).attach(first)
    node = _Passthrough(source).attach(second, explode)

    with pytest.raises(RuntimeError, match="the hook gave up"):
        node.run()

    assert first.events == ["open", "see0", "abort"]
    assert second.events == ["open", "see0", "abort"]
