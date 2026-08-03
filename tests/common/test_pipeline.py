from __future__ import annotations

import dataclasses

import pytest

from iivs_cardio.common.pipeline import Slot, drain


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


def test_drain_hands_every_slot_to_every_hook() -> None:
    slots = [Slot(index, index * 10) for index in range(3)]
    first: list[Slot[int]] = []
    second: list[Slot[int]] = []

    drain(slots, first.append, second.append)

    assert first == slots
    assert second == slots


def test_drain_calls_hooks_in_the_order_given() -> None:
    calls: list[tuple[str, int]] = []

    def record(name: str) -> object:
        return lambda slot: calls.append((name, slot.index))

    drain([Slot(0, "a"), Slot(1, "b")], record("first"), record("second"))

    assert calls == [("first", 0), ("second", 0), ("first", 1), ("second", 1)]


def test_drain_passes_absent_slots_through() -> None:
    slots: list[Slot[str]] = [Slot(0, "a"), Slot(1, None), Slot(2, "c")]
    seen: list[Slot[str]] = []

    drain(slots, seen.append)

    assert seen == slots
    assert [slot.index for slot in seen if slot.value is None] == [1]


def test_drain_exhausts_the_iterable_without_hooks() -> None:
    pulled: list[int] = []

    def source() -> object:
        for index in range(3):
            pulled.append(index)
            yield Slot(index, index)

    drain(source())

    assert pulled == [0, 1, 2]


def test_drain_pulls_a_generator_to_its_flush() -> None:
    # A node owes its buffered tail once its input ends; `drain` gets that only
    # because it exhausts the iterable rather than stopping with the source.
    def node() -> object:
        yield Slot(0, "a")
        yield Slot(1, "b")
        yield Slot(2, None)  # the flush a windowing node still owes

    seen: list[Slot[str]] = []

    drain(node(), seen.append)

    assert [(slot.index, slot.value) for slot in seen] == [
        (0, "a"),
        (1, "b"),
        (2, None),
    ]
