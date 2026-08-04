from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from iivs_cardio.common.pipeline import SequenceStage, Step
from scripts.data._range import (
    DatasetRange,
    DatasetRangeCollector,
    FrameRange,
    SequenceRange,
    as_dict,
)


def _sequence(source: str, *bounds: tuple[float, float]) -> SequenceRange:
    frames = tuple(
        FrameRange(f"{index:05d}_phase.bin", low, high)
        for index, (low, high) in enumerate(bounds)
    )
    return SequenceRange(source, frames)


# --------------------------------- the fold ------------------------------- #


def test_the_bounds_come_from_whichever_parts_hold_them():
    # The two extremes sit in different frames, and neither is the first, so a
    # fold that ignored the parts could not land on this pair by accident.
    sequence = _sequence("TL_00", (1.0, 2.0), (-4.0, 3.0), (0.0, 9.0), (0.5, 1.5))

    assert (sequence.min_value, sequence.max_value) == (-4.0, 9.0)
    assert (sequence.min_index, sequence.max_index) == (1, 2)


def test_a_bound_is_read_from_its_own_side():
    # The frame holding the lowest minimum is not the one holding the highest
    # maximum, so reading both off a single "widest" part would be wrong.
    sequence = _sequence("TL_00", (-8.0, -7.0), (5.0, 6.0))

    assert (sequence.min_value, sequence.max_value) == (-8.0, 6.0)
    assert (sequence.min_index, sequence.max_index) == (0, 1)


def test_a_tie_is_reported_against_the_earliest_part():
    # Undefined by the maths and settled by `min`/`max`, which keep the first
    # among equals. Pinned because a rewrite could silently answer the last.
    sequence = _sequence("TL_00", (0.0, 5.0), (0.0, 5.0), (0.0, 5.0))

    assert (sequence.min_index, sequence.max_index) == (0, 0)


def test_the_dataset_bound_indexes_a_sequence_rather_than_a_frame():
    # Each level folds only its own parts, so a dataset index names a sequence.
    # The extremes are in the last sequence, whose own extremes are frames 1
    # and 0 -- numbers that must not leak upwards.
    dataset = DatasetRange(
        (
            _sequence("TL_00", (1.0, 2.0), (1.5, 2.5)),
            _sequence("TL_01", (0.5, 3.0)),
            _sequence("TL_02", (4.0, 4.0), (-9.0, 12.0)),
        )
    )

    assert (dataset.min_value, dataset.max_value) == (-9.0, 12.0)
    assert (dataset.min_index, dataset.max_index) == (2, 2)
    assert dataset.sequences[2].min_index == 1


@pytest.mark.parametrize(
    ("build", "named"),
    (
        (lambda: SequenceRange("TL_00", ()), "SequenceRange"),
        (lambda: DatasetRange(()), "DatasetRange"),
    ),
)
def test_a_level_holding_nothing_is_refused_by_name(build, named):
    # There is no range to report and no index to point at, so this cannot answer
    # with a sentinel. The message names the level, since a dataset with an empty
    # sequence in it is fixed differently from an empty dataset.
    with pytest.raises(ValueError, match=rf"{named} holds nothing"):
        build()


# ------------------------------ serialization ----------------------------- #


def test_every_level_leads_with_the_source_it_names():
    # The declared field order puts four folded numbers ahead of the source, and
    # the source is what a reader scans for.
    dataset = DatasetRange((_sequence("TL_00", (0.0, 1.0)),))

    document = as_dict(dataset)
    sequence = document["sequences"][0]

    assert next(iter(sequence)) == "source"
    assert next(iter(sequence["frames"][0])) == "source"
    assert next(iter(document)) == "min_value"  # a dataset names no source


def test_the_nesting_survives_the_conversion():
    dataset = DatasetRange((_sequence("TL_00", (1.0, 2.0), (-4.0, 3.0)),))

    document = as_dict(dataset)

    assert document["min_value"] == -4.0
    assert document["max_index"] == 0
    assert [frame["source"] for frame in document["sequences"][0]["frames"]] == [
        "00000_phase.bin",
        "00001_phase.bin",
    ]
    assert document["sequences"][0]["frames"][1] == {
        "source": "00001_phase.bin",
        "min_value": -4.0,
        "max_value": 3.0,
    }


def test_the_document_survives_the_serializer_it_is_written_with():
    # `json.dumps` is the only consumer, and it refuses a dataclass left whole at
    # any depth. Asserted against the encoder rather than against isinstance,
    # since it is what the writer actually runs.
    dataset = DatasetRange((_sequence("TL_00", (0.0, 1.0)),))

    encoded = json.loads(json.dumps(as_dict(dataset)))

    assert encoded["sequences"][0]["frames"][0]["source"] == "00000_phase.bin"


class _Frames:
    """The slice of `DataSequence` a scan reads, as frames with their filenames."""

    def __init__(self, *frames: torch.Tensor) -> None:
        self._frames = frames

    def __len__(self) -> int:
        return len(self._frames)

    def get_item(self, index: int) -> torch.Tensor:
        return self._frames[index]

    def get_meta(self, index: int) -> Path:
        return Path(f"{index:05d}_phase.bin")


def _collector(tmp_path, provenance=None) -> DatasetRangeCollector:
    return DatasetRangeCollector(tmp_path / "range", provenance)


def _scan(ranges: DatasetRangeCollector, source: str, *bounds: tuple[float, float]):
    """Range one sequence the way a driver would: attach, run, call nothing."""
    frames = _Frames(*(torch.tensor([[low, high]]) for low, high in bounds))
    SequenceStage(frames).register_hooks(ranges.collector_for(source)).run()


def test_a_finished_collector_hands_itself_over(tmp_path):
    ranges = _collector(tmp_path)

    _scan(ranges, "a", (-1.0, 2.0))
    _scan(ranges, "b", (-3.0, 1.0))

    dataset = ranges.collected()
    assert dataset.min_value == -3.0
    assert dataset.max_value == 2.0
    assert dataset.min_index == 1  # the second sequence held the low
    assert dataset.max_index == 0


def test_a_collector_absorbs_across_calls(tmp_path):
    # Results come back per worker, so absorbing is not a single handover.
    ranges = _collector(tmp_path)

    _scan(ranges, "a", (0.0, 1.0))
    _scan(ranges, "b", (0.0, 5.0))

    assert ranges.collected().max_value == 5.0


def test_a_collector_that_absorbed_nothing_refuses_to_fold(tmp_path):
    with pytest.raises(ValueError, match="holds nothing"):
        _collector(tmp_path).collected()


def test_a_collector_writes_the_document_it_gathered(tmp_path):
    ranges = _collector(tmp_path, {"filter": "identity"})
    _scan(ranges, "a", (-1.0, 2.0))

    path = ranges.save()

    assert path == tmp_path / "range.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["filter"] == "identity"
    assert document["dataset"]["max_value"] == 2.0
    assert document["dataset"]["sequences"][0]["source"] == "a"


def test_a_run_tells_the_collector_the_traversal_ended(tmp_path):
    # Registered as itself, so `Stage.run` can signal it -- nothing calls `collected`
    # at the right moment by hand.
    collector = _collector(tmp_path).collector_for("seq")
    frames = _Frames(torch.tensor([[0.0, 2.0]]), torch.tensor([[-1.0, 1.0]]))

    SequenceStage(frames).register_hooks(collector).run()

    ranged = collector.collected()
    assert (ranged.min_value, ranged.max_value) == (-1.0, 2.0)
    assert [frame.source for frame in ranged.frames] == [
        "00000_phase.bin",
        "00001_phase.bin",
    ]


def test_a_collector_refuses_a_fold_over_a_prefix(tmp_path):
    collector = _collector(tmp_path).collector_for("seq")
    collector.observe(Step(0, torch.tensor([[1.0]]), Path("00000_phase.bin")))

    with pytest.raises(ValueError, match="the traversal of seq did not finish"):
        collector.collected()


def test_a_collector_refuses_after_a_traversal_that_died(tmp_path):
    # The steps it saw are a prefix, not this sequence's range, and reporting
    # them would put a hole in the dataset's bounds where nobody would see it.
    def explode(step: Step[torch.Tensor, Path]) -> None:
        msg = "the run gave up"
        raise RuntimeError(msg)

    collector = _collector(tmp_path).collector_for("seq")
    stage = SequenceStage(_Frames(torch.tensor([[1.0]])))
    stage.register_hooks(collector, explode)

    with pytest.raises(RuntimeError, match="the run gave up"):
        stage.run()

    with pytest.raises(ValueError, match="did not finish"):
        collector.collected()


def test_a_traversal_that_died_hands_over_nothing(tmp_path):
    # A prefix cannot reach the dataset's bounds, and no driver had to know.
    def explode(step: Step[torch.Tensor, Path]) -> None:
        msg = "the run gave up"
        raise RuntimeError(msg)

    ranges = _collector(tmp_path)
    stage = SequenceStage(_Frames(torch.tensor([[1.0]])))
    stage.register_hooks(ranges.collector_for("seq"), explode)

    with pytest.raises(RuntimeError, match="the run gave up"):
        stage.run()

    with pytest.raises(ValueError, match="holds nothing"):
        ranges.collected()


def test_merge_brings_a_workers_copy_home(tmp_path):
    parent, worker = _collector(tmp_path), _collector(tmp_path)
    _scan(parent, "a", (0.0, 1.0))
    _scan(worker, "b", (0.0, 5.0))

    parent.merge(worker)

    assert [s.source for s in parent.collected().sequences] == ["a", "b"]
