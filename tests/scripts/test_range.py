from __future__ import annotations

import json

import pytest

from scripts._range import DatasetRange, FrameRange, SequenceRange, as_dict


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
