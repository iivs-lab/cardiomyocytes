from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from iivs_cardio.common.pipeline import SequenceStage, Step
from scripts.data._range import (
    DatasetRange,
    FrameRange,
    RangeDocument,
    SequenceRange,
    SequenceRangeMeter,
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


def _meter(tmp_path, source: str = "seq") -> SequenceRangeMeter:
    return SequenceRangeMeter(source, tmp_path / "range.parts")


def _range(*bounds: tuple[float, float]) -> _Frames:
    return _Frames(*(torch.tensor([[low, high]]) for low, high in bounds))


def _scan(meter: SequenceRangeMeter, *bounds: tuple[float, float]) -> None:
    """Range one sequence the way a stage would: register, run, call nothing."""
    SequenceStage(_range(*bounds)).register_hooks(meter).run()


# ---------------------------------- the meter ----------------------------------


def test_a_meter_ranges_every_frame_under_the_file_it_came_from(tmp_path):
    meter = _meter(tmp_path)

    _scan(meter, (0.0, 2.0), (-1.0, 1.0))

    ranged = meter.collected()
    assert (ranged.min_value, ranged.max_value) == (-1.0, 2.0)
    assert [frame.source for frame in ranged.frames] == [
        "00000_phase.bin",
        "00001_phase.bin",
    ]


def test_a_meter_refuses_a_frame_with_no_finite_value(tmp_path):
    meter = _meter(tmp_path, "TL_00")
    frames = _Frames(torch.tensor([[float("nan")]]))

    with pytest.raises(ValueError, match=r"no finite value in TL_00/00000_phase.bin"):
        SequenceStage(frames).register_hooks(meter).run()


def test_a_finished_meter_leaves_its_part_behind(tmp_path):
    # What carries the answer out of a worker: `shared_objects` travels one way.
    meter = _meter(tmp_path)

    _scan(meter, (-1.0, 2.0))

    part = tmp_path / "range.parts" / "seq.json"
    written = json.loads(part.read_text(encoding="utf-8"))

    assert written == json.loads(json.dumps(as_dict(meter.collected())))


def test_a_name_that_is_a_path_does_not_nest(tmp_path):
    _scan(_meter(tmp_path, "plate/TL_00"), (0.0, 1.0))

    assert [p.name for p in (tmp_path / "range.parts").iterdir()] == [
        "plate%2FTL_00.json"
    ]


def test_a_traversal_that_died_leaves_nothing(tmp_path):
    # The frames it saw are a prefix, and a prefix folded into the dataset's
    # bounds is a hole where nobody would see it.
    def explode(step: Step[torch.Tensor, Path]) -> None:
        msg = "the run gave up"
        raise RuntimeError(msg)

    stage = SequenceStage(_range((0.0, 1.0)))
    stage.register_hooks(_meter(tmp_path), explode)

    with pytest.raises(RuntimeError, match="the run gave up"):
        stage.run()

    assert not (tmp_path / "range.parts").exists()


def test_a_meter_that_saw_nothing_is_refused_by_name(tmp_path):
    with (
        pytest.raises(ValueError, match="SequenceRange holds nothing"),
        _meter(tmp_path),
    ):
        pass


# --------------------------------- the document --------------------------------


def test_the_parts_folder_sits_beside_the_document(tmp_path):
    named = RangeDocument(tmp_path / "phase_range.json")
    bare = RangeDocument(tmp_path / "phase_range")

    assert named.parts == bare.parts == tmp_path / "phase_range.parts"


def test_a_document_asks_a_sequence_only_what_it_is_called(tmp_path):
    # `Named` is the whole of what a range document needs, so it never learns
    # what a phase folder or a filtered sequence is.
    @dataclass(frozen=True, slots=True)
    class _Named:
        name: str

    meter = RangeDocument(tmp_path / "range").hook_for(_Named("plate/TL_00"))
    with meter:
        meter(Step(0, torch.tensor([[0.0, 1.0]]), Path("00000_phase.bin")))

    assert (tmp_path / "range.parts" / "plate%2FTL_00.json").exists()


def test_a_document_folds_the_parts_in_name_order(tmp_path):
    document = RangeDocument(tmp_path / "range")
    with document:
        _scan(_meter(tmp_path, "b"), (0.0, 5.0))
        _scan(_meter(tmp_path, "a"), (-3.0, 1.0))

    dataset = json.loads((tmp_path / "range.json").read_text(encoding="utf-8"))

    assert [s["source"] for s in dataset["dataset"]["sequences"]] == ["a", "b"]
    assert dataset["dataset"]["min_index"] == 0  # `a` held the low
    assert dataset["dataset"]["max_index"] == 1


def test_a_document_writes_the_provenance_it_was_given(tmp_path):
    with RangeDocument(tmp_path / "range", provenance={"filter": "identity"}):
        _scan(_meter(tmp_path, "a"), (-1.0, 2.0))

    document = json.loads((tmp_path / "range.json").read_text(encoding="utf-8"))
    assert document["filter"] == "identity"
    assert document["dataset"]["max_value"] == 2.0


def test_a_document_that_gathered_nothing_is_refused_by_name(tmp_path):
    with (
        pytest.raises(ValueError, match="DatasetRange holds nothing"),
        RangeDocument(tmp_path / "range"),
    ):
        pass


def test_a_run_that_died_writes_no_document(tmp_path):
    # The parts it did finish stay: they are what a re-run does not have to redo.
    def die() -> None:
        with RangeDocument(tmp_path / "range"):
            _scan(_meter(tmp_path, "a"), (0.0, 1.0))
            msg = "the run gave up"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="the run gave up"):
        die()

    assert not (tmp_path / "range.json").exists()
    assert (tmp_path / "range.parts" / "a.json").exists()


def test_entering_drops_what_an_earlier_run_left(tmp_path):
    # `output_directory` is pinned rather than timestamped, so a re-run lands in
    # the same folder -- and a stale part would fold in as if it were this run's.
    with RangeDocument(tmp_path / "range"):
        _scan(_meter(tmp_path, "a"), (0.0, 1.0))
        _scan(_meter(tmp_path, "b"), (0.0, 5.0))

    with RangeDocument(tmp_path / "range", overwrite=True):
        _scan(_meter(tmp_path, "a"), (0.0, 1.0))

    document = json.loads((tmp_path / "range.json").read_text(encoding="utf-8"))
    assert [s["source"] for s in document["dataset"]["sequences"]] == ["a"]
