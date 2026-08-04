from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Self

import pytest

from iivs_cardio.common.pipeline import SequenceStage, Stage, Step

if TYPE_CHECKING:
    from iivs_cardio.common.pipeline import Hook


class _Fixed(Stage[str]):
    """A source stage over fixed values, recording every index it computed."""

    def __init__(self, *values: str | None, window: int = 1) -> None:
        super().__init__(window=window)
        self._values = values
        self.computed: list[int] = []

    def __len__(self) -> int:
        return len(self._values)

    def _compute(self, index: int) -> str | None:
        self.computed.append(index)
        return self._values[index]


class _Passthrough(Stage[str]):
    """A stage that forwards its source, so a graph has something to walk."""

    def __init__(self, source: Stage[str], *, window: int = 1) -> None:
        super().__init__(source, window=window)
        self._source = source

    def __len__(self) -> int:
        return len(self._source)

    def _compute(self, index: int) -> str | None:
        return self._source[index].value


class _Joined(Stage[str]):
    """A stage over two sources, so the graph can be a diamond."""

    def __init__(self, left: Stage[str], right: Stage[str]) -> None:
        super().__init__(left, right)
        self._left = left
        self._right = right

    def __len__(self) -> int:
        return min(len(self._left), len(self._right))

    def _compute(self, index: int) -> str | None:
        return f"{self._left[index].value}{self._right[index].value}"


class _Sequence:
    """The slice of `DataSequence` that `SequenceStage` uses."""

    def __init__(self, items: list[str]) -> None:
        self._items = items
        self.reads: list[int] = []

    def __len__(self) -> int:
        return len(self._items)

    def get_item(self, index: int) -> str:
        self.reads.append(index)
        return self._items[index]

    def get_meta(self, index: int) -> str:
        return self._items[index].upper()


class _Managed:
    """A hook that also wants opening and closing, the way a writer does."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def __call__(self, step: Step[str, None]) -> None:
        self.events.append(f"see{step.index}")

    def __enter__(self) -> Self:
        self.events.append("open")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.events.append("abort" if exc_type is not None else "close")


class TestStep:
    def test_it_keeps_index_and_value(self) -> None:
        step = Step(3, "frame")

        assert step.index == 3
        assert step.value == "frame"
        assert step.extra is None

    def test_it_holds_an_index_without_a_value(self) -> None:
        step: Step[str] = Step(7, None)

        assert step.index == 7
        assert step.value is None

    def test_it_carries_what_the_stream_does_not_consume(self) -> None:
        step = Step(2, "frame", "00002_phase.bin")

        assert step.value == "frame"
        assert step.extra == "00002_phase.bin"

    def test_it_is_frozen(self) -> None:
        step = Step(0, "frame")

        with pytest.raises(dataclasses.FrozenInstanceError):
            step.index = 1  # ty: ignore[invalid-assignment]

    def test_it_rejects_an_attribute_it_does_not_declare(self) -> None:
        step = Step(0, "frame")

        with pytest.raises(AttributeError):
            step.source = "somewhere"  # ty: ignore[unresolved-attribute]

    def test_require_returns_a_present_value(self) -> None:
        assert Step(4, "frame").require() == "frame"

    def test_require_refuses_an_absent_one(self) -> None:
        step: Step[str] = Step(4, None)

        with pytest.raises(ValueError, match=r"step 4 holds no value"):
            step.require()

    def test_require_extra_returns_a_present_one(self) -> None:
        assert Step(4, "frame", "name").require_extra() == "name"

    def test_require_extra_refuses_an_absent_one(self) -> None:
        step = Step(4, "frame")

        with pytest.raises(ValueError, match=r"step 4 holds nothing beside its value"):
            step.require_extra()


class TestStageIndexing:
    def test_a_window_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"invalid window 0: expected 1 or more"):
            _Fixed("a", window=0)

    def test_it_answers_the_index_it_was_asked_for(self) -> None:
        stage = _Fixed("a", "b")

        assert stage[1].index == 1
        assert stage[1].value == "b"

    def test_an_index_outside_the_stage_is_refused(self) -> None:
        stage = _Fixed("a", "b")

        with pytest.raises(IndexError, match=r"step 2 is outside 0\.\.1"):
            stage[2]

    def test_a_negative_index_is_refused(self) -> None:
        # No wraparound: an index is what a value describes, not a position.
        stage = _Fixed("a", "b")

        with pytest.raises(IndexError, match=r"step -1 is outside 0\.\.1"):
            stage[-1]

    def test_asking_twice_computes_once(self) -> None:
        # The whole point of the model: one filtered frame for every consumer.
        stage = _Fixed("a", "b")

        assert stage[0].value == stage[0].value == "a"
        assert stage.computed == [0]

    def test_nothing_is_computed_until_it_is_asked_for(self) -> None:
        stage = _Fixed("a", "b", "c")

        assert stage.computed == []
        stage[1]
        assert stage.computed == [1]

    def test_iterating_walks_every_index_in_order(self) -> None:
        stage = _Fixed("a", "b", "c")

        assert [step.index for step in stage] == [0, 1, 2]
        assert [step.value for step in stage] == ["a", "b", "c"]

    def test_an_empty_stage_yields_nothing(self) -> None:
        assert list(_Fixed()) == []

    def test_sources_are_the_stages_it_draws_from(self) -> None:
        source = _Fixed("a")
        stage = _Passthrough(source)

        assert stage.sources == (source,)
        assert source.sources == ()


class TestStageWindow:
    def test_the_window_keeps_the_most_recent_steps(self) -> None:
        stage = _Fixed("a", "b", "c", window=2)

        for index in range(3):
            stage[index]

        stage[2]
        stage[1]
        assert stage.computed == [0, 1, 2]  # both still held

        stage[0]
        assert stage.computed == [0, 1, 2, 0]  # fell out of the window

    def test_a_window_too_small_recomputes_rather_than_fails(self) -> None:
        stage = _Fixed("a", "b", window=1)

        assert stage[0].value == "a"
        assert stage[1].value == "b"
        assert stage[0].value == "a"  # the answer is still right
        assert stage.computed == [0, 1, 0]

    def test_stepping_back_does_not_evict_the_frontier(self) -> None:
        # The window follows the furthest index yet, so a consumer looking back
        # does not cost the later steps it is about to ask for again.
        stage = _Fixed("a", "b", "c", window=2)

        stage[0]
        stage[1]
        stage[0]  # already held, no recompute
        stage[1]
        assert stage.computed == [0, 1]


class TestStageHooks:
    def test_register_hooks_returns_the_stage(self) -> None:
        stage = _Fixed("a")

        assert stage.register_hooks(lambda _: None) is stage

    def test_every_hook_sees_every_step(self) -> None:
        first: list[Step[str, None]] = []
        second: list[Step[str, None]] = []
        stage = _Fixed("a", "b").register_hooks(first.append, second.append)

        stage.run()

        assert [step.value for step in first] == ["a", "b"]
        assert [step.value for step in second] == ["a", "b"]

    def test_hooks_fire_in_the_order_registered(self) -> None:
        calls: list[tuple[str, int]] = []

        def record(name: str) -> Hook[str, None]:
            return lambda step: calls.append((name, step.index))

        stage = _Fixed("a", "b")
        stage.register_hooks(record("first")).register_hooks(record("second"))

        stage.run()

        assert calls == [("first", 0), ("second", 0), ("first", 1), ("second", 1)]

    def test_a_hook_fires_once_per_index_even_when_the_step_is_recomputed(self) -> None:
        # A writer that saw an index twice would write the same frame twice.
        seen: list[int] = []
        stage = _Fixed("a", "b", window=1)
        stage.register_hooks(lambda step: seen.append(step.index))

        stage[0]
        stage[1]
        stage[0]

        assert stage.computed == [0, 1, 0]
        assert seen == [0, 1]

    def test_hooks_see_absent_steps_too(self) -> None:
        # Which steps went unfilled is a fact only the hook can decide to record.
        seen: list[Step[str, None]] = []
        stage = _Fixed("a", None).register_hooks(seen.append)

        stage.run()

        assert [(step.index, step.value) for step in seen] == [(0, "a"), (1, None)]

    def test_a_hook_sees_a_step_before_the_consumer_does(self) -> None:
        order: list[str] = []
        stage = _Fixed("a", "b")
        stage.register_hooks(lambda step: order.append(f"hook{step.index}"))

        steps = iter(stage)
        order.append(f"consumer{next(steps).index}")
        order.append(f"consumer{next(steps).index}")

        assert order == ["hook0", "consumer0", "hook1", "consumer1"]


class TestStageRun:
    def test_it_walks_every_step(self) -> None:
        stage = _Fixed("a", "b", "c")

        stage.run()

        assert stage.computed == [0, 1, 2]

    def test_it_opens_and_closes_a_managed_hook(self) -> None:
        managed = _Managed()

        _Fixed("a", "b").register_hooks(managed).run()

        assert managed.events == ["open", "see0", "see1", "close"]

    def test_it_opens_a_managed_hook_registered_on_a_source(self) -> None:
        # A writer hangs off the stage whose output it saves, which is rarely
        # the top; running the top has to reach it anyway.
        managed = _Managed()
        source = _Fixed("a").register_hooks(managed)

        _Passthrough(source).run()

        assert managed.events == ["open", "see0", "close"]

    def test_a_shared_stage_is_opened_once_however_many_paths_reach_it(self) -> None:
        # The diamond: an online estimator and a metric both read the filter.
        managed = _Managed()
        shared = _Fixed("a", "b").register_hooks(managed)

        _Joined(shared, _Passthrough(shared)).run()

        assert managed.events == ["open", "see0", "see1", "close"]

    def test_a_failure_aborts_every_managed_hook_together(self) -> None:
        def explode(step: Step[str, None]) -> None:
            msg = "the hook gave up"
            raise RuntimeError(msg)

        first, second = _Managed(), _Managed()
        source = _Fixed("a").register_hooks(first)
        stage = _Passthrough(source).register_hooks(second, explode)

        with pytest.raises(RuntimeError, match="the hook gave up"):
            stage.run()

        assert first.events == ["open", "see0", "abort"]
        assert second.events == ["open", "see0", "abort"]

    def test_running_the_top_fires_a_hook_further_down(self) -> None:
        seen: list[int] = []
        source = _Fixed("a", "b")
        source.register_hooks(lambda step: seen.append(step.index))

        _Passthrough(source).run()

        assert seen == [0, 1]


class TestSequenceStage:
    def test_it_reads_a_sequence_in_order(self) -> None:
        steps = list(SequenceStage(_Sequence(["a", "b", "c"])))

        assert [step.index for step in steps] == [0, 1, 2]
        assert [step.require() for step in steps] == ["a", "b", "c"]

    def test_the_sequence_metadata_rides_beside_the_value(self) -> None:
        stage = SequenceStage(_Sequence(["a", "b"]))

        assert stage[1].value == "b"
        assert stage[1].extra == "B"

    def test_it_reads_each_index_once_and_only_when_asked(self) -> None:
        sequence = _Sequence(["a", "b", "c"])
        stage = SequenceStage(sequence)

        assert sequence.reads == []
        stage[0]
        stage[0]
        assert sequence.reads == [0]

        stage.run()
        assert sequence.reads == [0, 1, 2]

    def test_its_length_is_the_sequences(self) -> None:
        assert len(SequenceStage(_Sequence(["a", "b"]))) == 2

    def test_an_empty_sequence_yields_nothing(self) -> None:
        assert list(SequenceStage(_Sequence([]))) == []
