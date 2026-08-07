from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from iivs_cardio.common.pipeline import SequenceStage, Step
from iivs_cardio.data.pipeline.ranges import (
    Coverage,
    DatasetRange,
    FrameRange,
    RangeDocument,
    SequenceRange,
    SequenceRangeMeter,
    save_range_document,
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
        "plate_A",
        (
            _sequence("TL_00", (1.0, 2.0), (1.5, 2.5)),
            _sequence("TL_01", (0.5, 3.0)),
            _sequence("TL_02", (4.0, 4.0), (-9.0, 12.0)),
        ),
    )

    assert (dataset.min_value, dataset.max_value) == (-9.0, 12.0)
    assert (dataset.min_index, dataset.max_index) == (2, 2)
    assert dataset.sequences[2].min_index == 1


@pytest.mark.parametrize(
    ("build", "named"),
    (
        (lambda: SequenceRange("TL_00", ()), "SequenceRange"),
        (lambda: DatasetRange("plate_A", ()), "DatasetRange"),
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
    # The source is what a reader scans for, and it leads at every level without
    # the document being reordered on the way out -- `CompositeRange` declaring
    # it ahead of the four folded numbers is what buys that.
    dataset = DatasetRange("plate_A", (_sequence("TL_00", (0.0, 1.0)),))

    document = dataset.to_dict()
    sequence = document["sequences"][0]

    assert next(iter(document)) == "source"
    assert next(iter(sequence)) == "source"
    assert next(iter(sequence["frames"][0])) == "source"


def test_the_nesting_survives_the_conversion():
    dataset = DatasetRange("plate_A", (_sequence("TL_00", (1.0, 2.0), (-4.0, 3.0)),))

    document = dataset.to_dict()

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
    dataset = DatasetRange("plate_A", (_sequence("TL_00", (0.0, 1.0)),))

    encoded = json.loads(json.dumps(dataset.to_dict()))

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
    return SequenceRangeMeter(tmp_path / "range.parts", source)


def _range(*bounds: tuple[float, float]) -> _Frames:
    return _Frames(*(torch.tensor([[low, high]]) for low, high in bounds))


def _scan(meter: SequenceRangeMeter, *bounds: tuple[float, float]) -> None:
    """Range one sequence the way a stage would: register, run, call nothing."""
    SequenceStage(_range(*bounds)).register_hooks(meter).run()


# ---------------------------------- the meter ----------------------------------


def test_a_meter_ranges_every_frame_under_the_file_it_came_from(tmp_path):
    meter = _meter(tmp_path)

    _scan(meter, (0.0, 2.0), (-1.0, 1.0))

    ranged = meter.to_range()
    assert (ranged.min_value, ranged.max_value) == (-1.0, 2.0)
    assert [frame.source for frame in ranged.frames] == [
        "00000_phase.bin",
        "00001_phase.bin",
    ]


def test_a_meter_refuses_a_frame_with_no_finite_value(tmp_path):
    meter = _meter(tmp_path, "TL_00")
    frames = _Frames(torch.tensor([[float("nan")]]))

    with pytest.raises(
        ValueError, match=r"no finite value in 00000_phase.bin \(sequence: TL_00\)"
    ):
        SequenceStage(frames).register_hooks(meter).run()


def test_a_finished_meter_leaves_its_part_behind(tmp_path):
    # What carries the answer out of a worker: `shared_objects` travels one way.
    meter = _meter(tmp_path)

    _scan(meter, (-1.0, 2.0))

    part = tmp_path / "range.parts" / "seq.json"
    written = json.loads(part.read_text(encoding="utf-8"))

    assert written == json.loads(json.dumps(meter.to_range().to_dict()))


def test_a_name_that_is_a_path_lands_where_the_path_says(tmp_path):
    # Mirrored rather than flattened, so the parts read like the frame tree
    # written beside them instead of like a pile of encoded keys.
    _scan(_meter(tmp_path, "plate_A/2026-03-11/TL_00"), (0.0, 1.0))

    part = tmp_path / "range.parts" / "plate_A" / "2026-03-11" / "TL_00.json"
    assert part.exists()
    assert json.loads(part.read_text(encoding="utf-8"))["source"] == (
        "plate_A/2026-03-11/TL_00"
    )


def test_a_meter_reports_the_bounds_it_reached(tmp_path):
    meter = _meter(tmp_path)

    assert meter.report() is None  # nothing seen, so nothing to say

    _scan(meter, (0.0, 2.0), (-1.5, 1.0))

    assert meter.report() == "measured [-1.5, 2] across 2 frames"


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

    assert not list((tmp_path / "range.parts").rglob("*.json"))


def test_a_meter_that_saw_nothing_is_refused_by_name(tmp_path):
    with (
        pytest.raises(ValueError, match="SequenceRange holds nothing"),
        _meter(tmp_path),
    ):
        pass


# --------------------------------- the document --------------------------------


def test_the_parts_folder_sits_beside_the_document(tmp_path):
    named = RangeDocument(
        tmp_path / "phase_range.json", sequence_names=["a"], source="plate_A"
    )
    bare = RangeDocument(
        tmp_path / "phase_range", sequence_names=["a"], source="plate_A"
    )

    assert named.parts_root == bare.parts_root == tmp_path / "phase_range.parts"


def test_a_document_asks_a_sequence_only_what_it_is_called(tmp_path):
    # `Named` is the whole of what a range document needs, so it never learns
    # what a phase folder or a filtered sequence is.
    @dataclass(frozen=True, slots=True)
    class _Named:
        name: str

    meter = RangeDocument(
        tmp_path / "range", sequence_names=["plate/TL_00"], source="plate_A"
    ).get_hook(_Named("plate/TL_00"))
    with meter:
        meter(Step(0, torch.tensor([[0.0, 1.0]]), Path("00000_phase.bin")))

    assert (tmp_path / "range.parts" / "plate" / "TL_00.json").exists()


def test_a_document_folds_the_parts_in_name_order(tmp_path):
    document = RangeDocument(
        tmp_path / "range", sequence_names=["a", "b"], source="plate_A"
    )
    with document:
        _scan(_meter(tmp_path, "b"), (0.0, 5.0))
        _scan(_meter(tmp_path, "a"), (-3.0, 1.0))

    dataset = json.loads((tmp_path / "range.json").read_text(encoding="utf-8"))

    assert [s["source"] for s in dataset["dataset"]["sequences"]] == ["a", "b"]
    assert dataset["dataset"]["min_index"] == 0  # `a` held the low
    assert dataset["dataset"]["max_index"] == 1


def test_a_document_writes_the_settings_it_was_given(tmp_path):
    with RangeDocument(
        tmp_path / "range",
        settings={"filter": "identity"},
        sequence_names=["a"],
        source="plate_A",
    ):
        _scan(_meter(tmp_path, "a"), (-1.0, 2.0))

    document = json.loads((tmp_path / "range.json").read_text(encoding="utf-8"))
    assert document["settings"]["filter"] == "identity"
    assert document["dataset"]["max_value"] == 2.0


def test_a_document_that_gathered_nothing_is_refused_by_name(tmp_path):
    with (
        pytest.raises(ValueError, match="DatasetRange holds nothing"),
        RangeDocument(tmp_path / "range", sequence_names=["a"], source="plate_A"),
    ):
        pass


def test_a_run_that_died_writes_no_document(tmp_path):
    # The parts it did finish stay: they are what a re-run does not have to redo.
    def die() -> None:
        with RangeDocument(
            tmp_path / "range", sequence_names=["a", "b"], source="plate_A"
        ):
            _scan(_meter(tmp_path, "a"), (0.0, 1.0))
            msg = "the run gave up"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="the run gave up"):
        die()

    assert not (tmp_path / "range.json").exists()
    assert (tmp_path / "range.parts" / "a.json").exists()


def test_a_document_reports_only_once_it_has_saved(tmp_path):
    document = RangeDocument(
        tmp_path / "range", sequence_names=["a", "b"], source="plate_A"
    )

    assert document.report() is None  # a worker's copy never folds anything

    with document:
        _scan(_meter(tmp_path, "a"), (0.0, 5.0))
        _scan(_meter(tmp_path, "b"), (-2.25, 1.0))

    assert document.report() == "wrote range.json from 2 sequences: [-2.25, 5]"


def test_entering_drops_what_an_earlier_run_left(tmp_path):
    # `output_directory` is pinned rather than timestamped, so a re-run lands in
    # the same folder -- and a stale part would fold in as if it were this run's.
    with RangeDocument(tmp_path / "range", sequence_names=["a", "b"], source="plate_A"):
        _scan(_meter(tmp_path, "a"), (0.0, 1.0))
        _scan(_meter(tmp_path, "b"), (0.0, 5.0))

    with RangeDocument(
        tmp_path / "range", sequence_names=["a"], overwrite=True, source="plate_A"
    ):
        _scan(_meter(tmp_path, "a"), (0.0, 1.0))

    document = json.loads((tmp_path / "range.json").read_text(encoding="utf-8"))
    assert [s["source"] for s in document["dataset"]["sequences"]] == ["a"]


def test_a_document_it_may_not_replace_is_refused_before_anything_is_dropped(tmp_path):
    # Clearing cannot be undone and the document is only written once every
    # sequence has run, so refusing at the end meant paying for the whole
    # dataset -- hours of it -- and dropping the earlier run's parts on the way
    # in. The place is taken first, which costs nothing and settles it at once.
    with RangeDocument(tmp_path / "range", sequence_names=["a", "b"], source="plate_A"):
        _scan(_meter(tmp_path, "a"), (0.0, 1.0))
        _scan(_meter(tmp_path, "b"), (0.0, 5.0))

    parts = sorted(p.name for p in (tmp_path / "range.parts").iterdir())

    with (
        pytest.raises(FileExistsError),
        RangeDocument(tmp_path / "range", sequence_names=["a"], source="plate_A"),
    ):
        pytest.fail("the run started over a document it may not replace")

    assert sorted(p.name for p in (tmp_path / "range.parts").iterdir()) == parts
    assert json.loads((tmp_path / "range.json").read_text(encoding="utf-8"))["coverage"]


def test_entering_collects_what_an_interrupted_run_only_staged(tmp_path):
    # A part is staged beside its destination under a hidden `.tmp` name and
    # moved into place on a clean close, so a worker killed part way leaves one
    # there. `list_parts` cannot see it, the only other hand on it died with
    # that process, and the folder it sits in cannot be cleared while it is
    # there -- so it accumulates, one per interrupted run.
    staged = tmp_path / "range.parts" / "plate_A" / ".TL_00.json.ctgx5mjr.tmp"
    staged.parent.mkdir(parents=True)
    staged.write_text("half a part", encoding="utf-8")

    with RangeDocument(
        tmp_path / "range", sequence_names=["plate_A/TL_00"], source="plate_A"
    ):
        _scan(_meter(tmp_path, "plate_A/TL_00"), (0.0, 1.0))

    assert not staged.exists()


def test_entering_drops_the_folders_that_layout_left_too(tmp_path):
    # A mirrored part nests, so clearing the files alone would keep every folder
    # the source ever had -- and a plate dropped from the dataset would go on
    # looking present to anyone reading the tree.
    with RangeDocument(tmp_path / "range", sequence_names=["a", "b"], source="plate_A"):
        _scan(_meter(tmp_path, "plate_A/TL_00"), (0.0, 1.0))

    with RangeDocument(
        tmp_path / "range", sequence_names=["a"], overwrite=True, source="plate_A"
    ):
        _scan(_meter(tmp_path, "plate_B/TL_00"), (0.0, 1.0))

    assert not (tmp_path / "range.parts" / "plate_A").exists()
    assert (tmp_path / "range.parts" / "plate_B" / "TL_00.json").exists()


def test_entering_leaves_alone_what_it_did_not_write(tmp_path):
    # Clearing prunes the folders its own parts emptied, not the folder itself:
    # `parts` is derived from a configured path, and a wholesale wipe would take
    # whatever a caller pointed it at.
    parts = tmp_path / "range.parts"
    (parts / "plate_A").mkdir(parents=True)
    (parts / "plate_A" / "notes.txt").write_text("mine", encoding="utf-8")

    with RangeDocument(tmp_path / "range", sequence_names=["a", "b"], source="plate_A"):
        _scan(_meter(tmp_path, "plate_B/TL_00"), (0.0, 1.0))

    assert (parts / "plate_A" / "notes.txt").read_text(encoding="utf-8") == "mine"


# --------------------------------- coverage ------------------------------- #


def _saved(tmp_path) -> dict:
    return json.loads((tmp_path / "range.json").read_text(encoding="utf-8"))


def test_a_document_told_its_roster_says_what_it_covered(tmp_path):
    with RangeDocument(tmp_path / "range", sequence_names=["a", "b"], source="plate_A"):
        _scan(_meter(tmp_path, "a"), (0.0, 1.0))
        _scan(_meter(tmp_path, "b"), (0.0, 5.0))

    assert _saved(tmp_path)["coverage"] == {
        "found": 2,
        "selected": 2,
        "covered": 2,
        "skipped": [],
    }


def test_a_document_names_the_sequences_that_left_nothing(tmp_path):
    # Bounds folded over a subset are not the dataset's, and a consumer setting
    # a normalization policy from them would read a hole as data.
    with RangeDocument(
        tmp_path / "range", sequence_names=["a", "b", "c"], source="plate_A"
    ):
        _scan(_meter(tmp_path, "a"), (0.0, 1.0))

    assert _saved(tmp_path)["coverage"] == {
        "found": 3,
        "selected": 3,
        "covered": 1,
        "skipped": ["b", "c"],
    }


def test_a_part_filed_under_the_wrong_sequence_is_refused_by_name(tmp_path):
    # The path says which sequence a part belongs to and the body says it again;
    # nothing compared them, so a part sorted under one name and counted under
    # another produced "covered everything, skipped one" -- a document that
    # contradicts itself in a way no number in it points at.
    with RangeDocument(tmp_path / "range", sequence_names=["a", "b"], source="plate_A"):
        _scan(_meter(tmp_path, "a"), (0.0, 1.0))
        _scan(_meter(tmp_path, "b"), (0.0, 5.0))

    part = tmp_path / "range.parts" / "b.json"
    document = json.loads(part.read_text(encoding="utf-8"))
    document["source"] = "a"
    part.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=r"part 'b' holds 'a'"):
        RangeDocument(
            tmp_path / "range", sequence_names=["a", "b"], source="plate_A"
        ).to_range()


def test_a_document_that_could_not_be_written_reports_nothing(tmp_path):
    # What was folded was remembered before the file reached disk, so a refused
    # write still said "wrote range.json from 1 sequence: [...]" -- a line about
    # a document that is not there, and about bounds nobody can read back.
    document = RangeDocument(
        tmp_path / "range", sequence_names=["a"], source="plate_A", overwrite=False
    )
    _scan(_meter(tmp_path, "a"), (0.0, 1.0))
    (tmp_path / "range.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError):
        document.save()

    assert document.report() is None
    assert (tmp_path / "range.json").read_text(encoding="utf-8") == "{}"


def test_a_document_says_how_much_of_the_source_the_run_took(tmp_path):
    # The one a retry produced: `include` narrows the roster, so a run over one
    # of four covered all it was given and the block read as complete. Nothing
    # else in the file could tell "a dataset of one" from "one of four" --
    # `settings` leaves the selection out on purpose, since recording it there
    # would refuse reuse to a run that narrowed.
    with RangeDocument(
        tmp_path / "range", sequence_names=["b"], source="plate_A", found=4
    ) as document:
        _scan(_meter(tmp_path, "b"), (0.0, 1.0))

    assert _saved(tmp_path)["coverage"] == {
        "found": 4,
        "selected": 1,
        "covered": 1,
        "skipped": [],
    }
    assert document.report() == (
        "wrote range.json from 1 sequence, selected from 4: [0, 1]"
    )


def test_a_run_that_narrowed_nothing_says_the_roster_was_all_there_was(tmp_path):
    # Written whichever way round, so an absent `found` can never be read as
    # "unknown". A caller that did not narrow says so by not saying anything.
    with RangeDocument(
        tmp_path / "range", sequence_names=["a", "b"], source="plate_A"
    ) as document:
        _scan(_meter(tmp_path, "a"), (0.0, 1.0))
        _scan(_meter(tmp_path, "b"), (2.0, 3.0))

    assert _saved(tmp_path)["coverage"]["found"] == 2
    assert "selected from" not in (document.report() or "")


def test_a_roster_larger_than_what_was_found_is_refused(tmp_path):
    # No selection can leave more than there was, so this is the caller having
    # passed the wrong number -- and it has to fail where that is still visible
    # rather than as a coverage nobody can explain.
    with pytest.raises(ValueError, match=r"selected 2 of the 1"):
        RangeDocument(
            tmp_path / "range", sequence_names=["a", "b"], source="plate_A", found=1
        )


def test_a_coverage_that_does_not_add_up_cannot_be_built(tmp_path):
    # `covered` was counted off disk while `total` and `skipped` came from the
    # roster, so the three could disagree. The type refuses the arithmetic now,
    # which is what keeps a future caller from writing one.
    with pytest.raises(ValueError, match=r"coverage does not add up"):
        Coverage(found=2, selected=2, covered=3, skipped=("b",))

    with pytest.raises(ValueError, match=r"selected 2 of the 1 found"):
        Coverage(found=1, selected=2, covered=2, skipped=())


def test_coverage_sits_ahead_of_the_numbers_it_qualifies(tmp_path):
    # A reader who takes the bounds and stops has to have passed this on the way.
    with RangeDocument(
        tmp_path / "range",
        settings={"filter": "identity"},
        sequence_names=["a"],
        source="plate_A",
    ):
        _scan(_meter(tmp_path, "a"), (0.0, 1.0))

    assert list(_saved(tmp_path)) == ["settings", "coverage", "dataset"]


def test_a_document_with_no_sequence_to_cover_is_refused(tmp_path):
    # `search_sources` refuses a root holding nothing and a selection that kept
    # nothing, so a run cannot reach here with an empty roster. Refused at the
    # door anyway: a document that cannot name what it set out to cover would
    # write a `coverage` that reads as "complete" rather than "unknown".
    with pytest.raises(ValueError, match=r"no sequence to cover"):
        RangeDocument(tmp_path / "range", sequence_names=[], source="plate_A")


def test_a_roster_naming_one_sequence_twice_counts_it_once(tmp_path):
    # `covered` reads the parts back as a set, and a name can leave only one
    # part, so a repeated name would hold `covered` below `selected` for a run
    # that measured everything -- and report the sequence as skipped besides.
    with RangeDocument(
        tmp_path / "range", sequence_names=["a", "b", "a"], source="plate_A"
    ) as document:
        assert document.sequence_names == ("a", "b")

        _scan(_meter(tmp_path, "a"), (0.0, 1.0))
        _scan(_meter(tmp_path, "b"), (2.0, 3.0))

    assert _saved(tmp_path)["coverage"] == {
        "found": 2,
        "selected": 2,
        "covered": 2,
        "skipped": [],
    }


def test_the_roster_keeps_the_order_it_was_first_given_in(tmp_path):
    # `skipped` is reported in roster order, so the survivor of a repeat has to
    # be the first sighting rather than the last.
    document = RangeDocument(
        tmp_path / "range", sequence_names=["b", "a", "b", "c"], source="plate_A"
    )

    assert document.sequence_names == ("b", "a", "c")


def test_every_document_carries_its_coverage(tmp_path):
    # Always written, never omitted: an absent block would leave a reader
    # guessing whether it means complete or unknown.
    with RangeDocument(tmp_path / "range", sequence_names=["a"], source="plate_A"):
        _scan(_meter(tmp_path, "a"), (0.0, 1.0))

    assert _saved(tmp_path)["coverage"] == {
        "found": 1,
        "selected": 1,
        "covered": 1,
        "skipped": [],
    }


def test_a_partial_document_reads_differently_from_a_whole_one(tmp_path):
    # A count that usually matches is a number nobody checks, so the partial
    # line is shaped differently rather than carrying the same one.
    whole = RangeDocument(tmp_path / "range", sequence_names=["a"], source="plate_A")
    with whole:
        _scan(_meter(tmp_path, "a"), (0.0, 1.0))

    assert whole.report() == "wrote range.json from 1 sequence: [0, 1]"

    partial = RangeDocument(
        tmp_path / "range",
        sequence_names=["a", "b", "c"],
        overwrite=True,
        source="plate_A",
    )
    with partial:
        _scan(_meter(tmp_path, "a"), (0.0, 1.0))
        _scan(_meter(tmp_path, "c"), (-2.0, 4.0))

    assert partial.report() == (
        "wrote range.json from 2 of 3 sequences, 1 skipped: [-2, 4]"
    )


def test_the_skipped_list_keeps_the_roster_s_own_order(tmp_path):
    # Not the folded order and not sorted: the roster is what a caller reads the
    # list back against, and it is the order the run was going to take.
    document = RangeDocument(
        tmp_path / "range", sequence_names=["c", "a", "b"], source="plate_A"
    )
    with document:
        _scan(_meter(tmp_path, "a"), (0.0, 1.0))

    coverage = document.get_coverage(document.to_range())

    assert coverage is not None
    assert coverage.skipped == ("c", "b")


# --------------------------------- reading back --------------------------- #


def test_a_range_survives_the_round_trip_it_was_written_for():
    # Both directions, and without the file in between: `to_dict` hands back
    # tuples where JSON hands back lists, and the pair has to take either.
    frame = FrameRange("00000_phase.bin", 0.0, 1.0)
    sequence = SequenceRange("TL_00", (frame,))
    dataset = DatasetRange("plate_A", (sequence,))

    for ranged in (frame, sequence, dataset):
        kind = type(ranged)
        assert kind.from_dict(ranged.to_dict()) == ranged
        assert kind.from_dict(json.loads(json.dumps(ranged.to_dict()))) == ranged


@pytest.mark.parametrize(
    ("kind", "document", "named"),
    (
        (FrameRange, {"min_value": 0.0, "max_value": 1.0}, "source"),
        (FrameRange, {"source": "a", "max_value": 1.0}, "min_value"),
        (FrameRange, {"source": "a", "min_value": 0.0}, "max_value"),
        (
            FrameRange,
            {"source": "a", "min_value": "low", "max_value": 1.0},
            "min_value",
        ),
        (SequenceRange, {"source": "a"}, "frames"),
        (SequenceRange, {"source": "a", "frames": "00000.bin"}, "frames"),
        (DatasetRange, {"source": "a"}, "sequences"),
        (DatasetRange, {"source": "a", "frames": []}, "sequences"),
    ),
)
def test_a_malformed_document_is_refused_by_the_entry_it_stumbled_on(
    kind, document, named
):
    # A part is read back by the same code that wrote it, so the realistic cause
    # is a hand edit or a version skew -- either way the message has to say which
    # entry, since the file is what the reader will go and look at.
    with pytest.raises(ValueError, match=rf"malformed range document: '{named}'"):
        kind.from_dict(document)


def test_an_integer_bound_is_taken_as_the_float_it_stands_for():
    # JSON writes a whole number without its point, so a bound that happens to
    # land on one comes back as `int` and must not be refused for it.
    frame = FrameRange.from_dict({"source": "a", "min_value": 0, "max_value": 2})

    assert (frame.min_value, frame.max_value) == (0.0, 2.0)
    assert isinstance(frame.min_value, float)


def test_a_document_written_without_coverage_leaves_the_block_out(tmp_path):
    # The writer keeps `coverage` optional for a caller that has no roster to
    # measure against -- a merge over several documents, whose own coverage is
    # the union of theirs and has to be recomputed rather than carried over.
    dataset = DatasetRange("plate_A", (_sequence("TL_00", (0.0, 1.0)),))

    save_range_document(tmp_path / "merged", dataset, settings={"filter": "identity"})

    document = json.loads((tmp_path / "merged.json").read_text(encoding="utf-8"))
    assert list(document) == ["settings", "dataset"]


@pytest.mark.parametrize(
    "source",
    ("TL_00", "plate/2026.03.11/TL_00", "plate/TL_00.v2", "plate/well_A1.2"),
)
def test_a_sequence_named_with_a_dot_still_files_and_reads_back(tmp_path, source):
    @dataclass(frozen=True, slots=True)
    class _Named:
        name: str

    # A time-lapse folder may carry a dot -- a date, a magnification, a repeat
    # marker -- and the tail after it is part of the name, not an extension the
    # part file has to justify.
    document = RangeDocument(
        tmp_path / "range", sequence_names=[source], source="plate_A"
    )
    with document:
        meter = document.get_hook(_Named(source))
        with meter:
            meter(Step(0, torch.tensor([[0.5, 2.0]]), Path("00000_phase.bin")))

    parts = document.list_parts()
    assert [part.relative_to(document.parts_root).as_posix() for part in parts] == [
        f"{source}.json"
    ]
    assert [sequence.source for sequence in document.to_range().sequences] == [source]
    assert _saved(tmp_path)["coverage"]["skipped"] == []
