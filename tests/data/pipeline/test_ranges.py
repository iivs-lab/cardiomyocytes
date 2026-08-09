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


def _contents(*names: str, frames: int = 1) -> dict[str, tuple[str, ...]]:
    """Every sequence the source holds, against the frames each would measure.

    `_Frames` names its frames the way a phase folder does, so the contents a
    document is given lines up with what a meter records under each part.
    """
    listed = tuple(f"{index:05d}_phase.bin" for index in range(frames))

    return dict.fromkeys(names, listed)


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
    # and 0, numbers that must not leak upwards.
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
    # the document being reordered on the way out: `CompositeRange` declaring
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


def _meter(tmp_path, source: str = "seq", settings=None) -> SequenceRangeMeter:
    return SequenceRangeMeter(tmp_path / "range.parts", source, settings)


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
        tmp_path / "phase_range.json", contents=_contents("a"), source="plate_A"
    )
    bare = RangeDocument(
        tmp_path / "phase_range", contents=_contents("a"), source="plate_A"
    )

    assert named.parts_root == bare.parts_root == tmp_path / "phase_range.parts"


def test_a_document_asks_a_sequence_only_what_it_is_called(tmp_path):
    # `Named` is the whole of what a range document needs, so it never learns
    # what a phase folder or a filtered sequence is.
    @dataclass(frozen=True, slots=True)
    class _Named:
        name: str

    meter = RangeDocument(
        tmp_path / "range", contents=_contents("plate/TL_00"), source="plate_A"
    ).get_hook(_Named("plate/TL_00"))
    with meter:
        meter(Step(0, torch.tensor([[0.0, 1.0]]), Path("00000_phase.bin")))

    assert (tmp_path / "range.parts" / "plate" / "TL_00.json").exists()


def test_a_document_folds_the_parts_in_name_order(tmp_path):
    document = RangeDocument(
        tmp_path / "range", contents=_contents("a", "b"), source="plate_A"
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
        contents=_contents("a"),
        settings={"filter": "identity"},
        source="plate_A",
    ):
        _scan(_meter(tmp_path, "a", {"filter": "identity"}), (-1.0, 2.0))

    document = json.loads((tmp_path / "range.json").read_text(encoding="utf-8"))
    assert document["settings"]["filter"] == "identity"
    assert document["dataset"]["max_value"] == 2.0


def test_a_document_that_gathered_nothing_says_so_rather_than_not_being_written(
    tmp_path,
):
    # `coverage: 0 of N` is wanted most exactly when every sequence failed, and
    # refusing to write is what used to take it away. There are no bounds to
    # invent, so the document carries what it covers and no `dataset` block.
    with RangeDocument(tmp_path / "range", contents=_contents("a"), source="plate_A"):
        pass

    document = _saved(tmp_path)

    assert "dataset" not in document
    assert document["coverage"] == {
        "found": 1,
        "selected": 1,
        "covered": 0,
        "reused": 0,
        "skipped": ["a"],
        "unselected": [],
    }


def test_a_run_that_died_still_folds_the_parts_that_finished(tmp_path):
    # The healthy parts were on disk and nothing read them: a worker dying took
    # the document with it, so a run that measured 90 of 121 left no record of
    # the 90. What it covers is `coverage`'s to say, not a reason to write
    # nothing.
    def die() -> None:
        with RangeDocument(
            tmp_path / "range", contents=_contents("a", "b"), source="plate_A"
        ):
            _scan(_meter(tmp_path, "a"), (0.0, 1.0))
            msg = "the run gave up"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="the run gave up"):
        die()

    document = _saved(tmp_path)

    assert (tmp_path / "range.parts" / "a.json").exists()
    assert document["dataset"]["sequences"][0]["source"] == "a"
    assert document["coverage"]["covered"] == 1
    assert document["coverage"]["skipped"] == ["b"]


def test_a_document_reports_only_once_it_has_saved(tmp_path):
    document = RangeDocument(
        tmp_path / "range", contents=_contents("a", "b"), source="plate_A"
    )

    assert document.report() is None  # a worker's copy never folds anything

    with document:
        _scan(_meter(tmp_path, "a"), (0.0, 5.0))
        _scan(_meter(tmp_path, "b"), (-2.25, 1.0))

    assert document.report() == "wrote range.json from 2 sequences: [-2.25, 5]"


def test_entering_drops_what_an_earlier_run_left(tmp_path):
    # `output_directory` is pinned rather than timestamped, so a re-run lands in
    # the same folder, and a stale part would fold in as if it were this run's.
    with RangeDocument(
        tmp_path / "range", contents=_contents("a", "b"), source="plate_A"
    ):
        _scan(_meter(tmp_path, "a"), (0.0, 1.0))
        _scan(_meter(tmp_path, "b"), (0.0, 5.0))

    with RangeDocument(
        tmp_path / "range",
        contents=_contents("a"),
        if_ranges_exist="overwrite",
        source="plate_A",
    ):
        _scan(_meter(tmp_path, "a"), (0.0, 1.0))

    document = json.loads((tmp_path / "range.json").read_text(encoding="utf-8"))
    assert [s["source"] for s in document["dataset"]["sequences"]] == ["a"]


def test_entering_leaves_an_empty_folder_it_did_not_empty(tmp_path):
    # The tree is cleared so a sequence dropped from the dataset stops looking
    # present, which is about folders this clearing emptied. One that was
    # already empty is somebody else's, a place held for a plate still being
    # copied most likely, and taking it is not this run's to do.
    mine = tmp_path / "range.parts" / "mine"
    mine.mkdir(parents=True)

    with RangeDocument(tmp_path / "range", contents=_contents("a"), source="plate_A"):
        _scan(_meter(tmp_path, "a"), (0.0, 1.0))

    assert mine.is_dir()


def test_a_meter_will_not_replace_a_part_it_did_not_write(tmp_path):
    # Its own run empties the folder on the way in, so a part sitting under this
    # name belongs to something else: two sequences whose names came out the
    # same, most likely. Replacing it silently lost one of the two.
    meter = _meter(tmp_path, "a")
    (tmp_path / "range.parts" / "a.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _scan(meter, (0.0, 1.0))

    assert (tmp_path / "range.parts" / "a.json").read_text(encoding="utf-8") == "{}"


def test_a_document_is_opened_once(tmp_path):
    # Opening is what makes the folder this run's, and it clears the folder to
    # do it, so a second one would drop the parts the first has gathered.
    document = RangeDocument(
        tmp_path / "range", contents=_contents("a"), source="plate_A"
    )

    with document:
        _scan(_meter(tmp_path, "a"), (0.0, 1.0))

        with pytest.raises(RuntimeError, match=r"opened already"):
            document.__enter__()

    assert (tmp_path / "range.parts" / "a.json").exists()


def test_a_document_it_may_not_replace_is_refused_before_anything_is_dropped(tmp_path):
    # Clearing cannot be undone and the document is only written once every
    # sequence has run, so refusing at the end meant paying for the whole
    # dataset, hours of it, and dropping the earlier run's parts on the way in.
    # The place is taken first, which costs nothing and settles it at once.
    with RangeDocument(
        tmp_path / "range", contents=_contents("a", "b"), source="plate_A"
    ):
        _scan(_meter(tmp_path, "a"), (0.0, 1.0))
        _scan(_meter(tmp_path, "b"), (0.0, 5.0))

    parts = sorted(p.name for p in (tmp_path / "range.parts").iterdir())

    with (
        pytest.raises(FileExistsError),
        RangeDocument(tmp_path / "range", contents=_contents("a"), source="plate_A"),
    ):
        pytest.fail("the run started over a document it may not replace")

    assert sorted(p.name for p in (tmp_path / "range.parts").iterdir()) == parts
    assert json.loads((tmp_path / "range.json").read_text(encoding="utf-8"))["coverage"]


def test_entering_collects_what_an_interrupted_run_only_staged(tmp_path):
    # A part is staged beside its destination under a hidden `.tmp` name and
    # moved into place on a clean close, so a worker killed part way leaves one
    # there. `list_parts` cannot see it, the only other hand on it died with
    # that process, and the folder it sits in cannot be cleared while it is
    # there, so it accumulates, one per interrupted run.
    staged = tmp_path / "range.parts" / "plate_A" / ".TL_00.json.ctgx5mjr.tmp"
    staged.parent.mkdir(parents=True)
    staged.write_text("half a part", encoding="utf-8")

    with RangeDocument(
        tmp_path / "range", contents=_contents("plate_A/TL_00"), source="plate_A"
    ):
        _scan(_meter(tmp_path, "plate_A/TL_00"), (0.0, 1.0))

    assert not staged.exists()


def test_entering_drops_the_folders_that_layout_left_too(tmp_path):
    # A mirrored part nests, so clearing the files alone would keep every folder
    # the source ever had, and a plate dropped from the dataset would go on
    # looking present to anyone reading the tree.
    with RangeDocument(
        tmp_path / "range", contents=_contents("a", "b"), source="plate_A"
    ):
        _scan(_meter(tmp_path, "plate_A/TL_00"), (0.0, 1.0))

    with RangeDocument(
        tmp_path / "range",
        contents=_contents("a"),
        if_ranges_exist="overwrite",
        source="plate_A",
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

    with RangeDocument(
        tmp_path / "range", contents=_contents("a", "b"), source="plate_A"
    ):
        _scan(_meter(tmp_path, "plate_B/TL_00"), (0.0, 1.0))

    assert (parts / "plate_A" / "notes.txt").read_text(encoding="utf-8") == "mine"


# --------------------------------- coverage ------------------------------- #


def _saved(tmp_path) -> dict:
    return json.loads((tmp_path / "range.json").read_text(encoding="utf-8"))


def test_a_document_told_its_contents_says_what_it_covered(tmp_path):
    with RangeDocument(
        tmp_path / "range", contents=_contents("a", "b"), source="plate_A"
    ):
        _scan(_meter(tmp_path, "a"), (0.0, 1.0))
        _scan(_meter(tmp_path, "b"), (0.0, 5.0))

    assert _saved(tmp_path)["coverage"] == {
        "found": 2,
        "selected": 2,
        "covered": 2,
        "reused": 0,
        "skipped": [],
        "unselected": [],
    }


def test_a_document_names_the_sequences_that_left_nothing(tmp_path):
    # Bounds folded over a subset are not the dataset's, and a consumer setting
    # a normalization policy from them would read a hole as data.
    with RangeDocument(
        tmp_path / "range", contents=_contents("a", "b", "c"), source="plate_A"
    ):
        _scan(_meter(tmp_path, "a"), (0.0, 1.0))

    assert _saved(tmp_path)["coverage"] == {
        "found": 3,
        "selected": 3,
        "covered": 1,
        "reused": 0,
        "skipped": ["b", "c"],
        "unselected": [],
    }


def test_a_part_filed_under_the_wrong_sequence_is_refused_by_name(tmp_path):
    # The path says which sequence a part belongs to and the body says it again;
    # nothing compared them, so a part sorted under one name and counted under
    # another produced "covered everything, skipped one": a document that
    # contradicts itself in a way no number in it points at.
    with RangeDocument(
        tmp_path / "range", contents=_contents("a", "b"), source="plate_A"
    ):
        _scan(_meter(tmp_path, "a"), (0.0, 1.0))
        _scan(_meter(tmp_path, "b"), (0.0, 5.0))

    part = tmp_path / "range.parts" / "b.json"
    document = json.loads(part.read_text(encoding="utf-8"))
    document["source"] = "a"
    part.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=r"part 'b' holds 'a'"):
        RangeDocument(
            tmp_path / "range", contents=_contents("a", "b"), source="plate_A"
        ).to_range()


def test_a_document_that_could_not_be_written_reports_nothing(tmp_path):
    # What was folded was remembered before the file reached disk, so a refused
    # write still said "wrote range.json from 1 sequence: [...]", a line about
    # a document that is not there, and about bounds nobody can read back.
    document = RangeDocument(
        tmp_path / "range",
        contents=_contents("a"),
        source="plate_A",
        if_ranges_exist="error",
    )
    _scan(_meter(tmp_path, "a"), (0.0, 1.0))
    (tmp_path / "range.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError):
        document.save()

    assert document.report() is None
    assert (tmp_path / "range.json").read_text(encoding="utf-8") == "{}"


def test_a_document_says_how_much_of_the_source_the_run_took(tmp_path):
    # The one a retry produced: `include` narrows the contents, so a run over one
    # of four covered all it was given and the block read as complete. Nothing
    # else in the file could tell "a dataset of one" from "one of four" --
    # `settings` leaves the selection out on purpose, since recording it there
    # would refuse reuse to a run that narrowed.
    with RangeDocument(
        tmp_path / "range",
        contents=_contents("a", "b", "c", "d"),
        source="plate_A",
        selected=["b"],
    ) as document:
        _scan(_meter(tmp_path, "b"), (0.0, 1.0))

    assert _saved(tmp_path)["coverage"] == {
        "found": 4,
        "selected": 1,
        "covered": 1,
        "reused": 0,
        "skipped": [],
        "unselected": ["a", "c", "d"],
    }
    assert document.report() == (
        "wrote range.json from 1 of 4 sequences, 3 not taken: [0, 1]"
    )


def test_a_run_that_narrowed_nothing_says_the_contents_was_all_there_was(tmp_path):
    # Written whichever way round, so an absent `found` can never be read as
    # "unknown". A caller that did not narrow says so by not saying anything.
    with RangeDocument(
        tmp_path / "range", contents=_contents("a", "b"), source="plate_A"
    ) as document:
        _scan(_meter(tmp_path, "a"), (0.0, 1.0))
        _scan(_meter(tmp_path, "b"), (2.0, 3.0))

    assert _saved(tmp_path)["coverage"]["found"] == 2
    assert "selected from" not in (document.report() or "")


def test_a_selection_naming_what_the_source_lacks_is_refused(tmp_path):
    # The contents is the dataset, so a selection outside it is the caller having
    # built one of the two from somewhere else, and it has to fail where that
    # is still visible rather than as a coverage nobody can explain.
    with pytest.raises(ValueError, match=r"selected 'c', which the source does not"):
        RangeDocument(
            tmp_path / "range",
            contents=_contents("a", "b"),
            source="plate_A",
            selected=["a", "c"],
        )


def test_a_coverage_that_does_not_add_up_cannot_be_built(tmp_path):
    # `covered` was counted off disk while `total` and `skipped` came from the
    # contents, so the three could disagree. The type refuses the arithmetic now,
    # which is what keeps a future caller from writing one.
    with pytest.raises(ValueError, match=r"coverage does not add up"):
        Coverage(found=2, selected=2, covered=3, skipped=("b",))

    with pytest.raises(ValueError, match=r"selected 2 of the 1 found"):
        Coverage(found=1, selected=2, covered=1, skipped=())

    with pytest.raises(ValueError, match=r"reused 2 of the 1 covered"):
        Coverage(found=1, selected=1, covered=1, reused=2)


def test_coverage_sits_ahead_of_the_numbers_it_qualifies(tmp_path):
    # A reader who takes the bounds and stops has to have passed this on the way.
    with RangeDocument(
        tmp_path / "range",
        contents=_contents("a"),
        settings={"filter": "identity"},
        source="plate_A",
    ):
        _scan(_meter(tmp_path, "a", {"filter": "identity"}), (0.0, 1.0))

    assert list(_saved(tmp_path)) == ["settings", "coverage", "dataset"]


def test_a_document_with_no_sequence_to_cover_is_refused(tmp_path):
    # `search_sources` refuses a root holding nothing and a selection that kept
    # nothing, so a run cannot reach here with an empty contents. Refused at the
    # door anyway: a document that cannot name what it set out to cover would
    # write a `coverage` that reads as "complete" rather than "unknown".
    with pytest.raises(ValueError, match=r"no sequence to cover"):
        RangeDocument(tmp_path / "range", contents={}, source="plate_A")


def test_a_selection_naming_one_sequence_twice_counts_it_once(tmp_path):
    # `covered` reads the parts back as a set, and a name can leave only one
    # part, so a repeated name would hold `covered` below `selected` for a run
    # that measured everything, and report the sequence as skipped besides.
    with RangeDocument(
        tmp_path / "range",
        contents=_contents("a", "b"),
        source="plate_A",
        selected=["a", "b", "a"],
    ) as document:
        assert document.selected == ("a", "b")

        _scan(_meter(tmp_path, "a"), (0.0, 1.0))
        _scan(_meter(tmp_path, "b"), (2.0, 3.0))

    assert _saved(tmp_path)["coverage"] == {
        "found": 2,
        "selected": 2,
        "covered": 2,
        "reused": 0,
        "skipped": [],
        "unselected": [],
    }


def test_the_selection_keeps_the_order_it_was_first_given_in(tmp_path):
    # `skipped` is reported in the order the run was going to take, so the
    # survivor of a repeat has to be the first sighting rather than the last.
    document = RangeDocument(
        tmp_path / "range",
        contents=_contents("a", "b", "c"),
        source="plate_A",
        selected=["b", "a", "b", "c"],
    )

    assert document.selected == ("b", "a", "c")


def test_every_document_carries_its_coverage(tmp_path):
    # Always written, never omitted: an absent block would leave a reader
    # guessing whether it means complete or unknown.
    with RangeDocument(tmp_path / "range", contents=_contents("a"), source="plate_A"):
        _scan(_meter(tmp_path, "a"), (0.0, 1.0))

    assert _saved(tmp_path)["coverage"] == {
        "found": 1,
        "selected": 1,
        "covered": 1,
        "reused": 0,
        "skipped": [],
        "unselected": [],
    }


def test_a_partial_document_reads_differently_from_a_whole_one(tmp_path):
    # A count that usually matches is a number nobody checks, so the partial
    # line is shaped differently rather than carrying the same one.
    whole = RangeDocument(tmp_path / "range", contents=_contents("a"), source="plate_A")
    with whole:
        _scan(_meter(tmp_path, "a"), (0.0, 1.0))

    assert whole.report() == "wrote range.json from 1 sequence: [0, 1]"

    partial = RangeDocument(
        tmp_path / "range",
        contents=_contents("a", "b", "c"),
        if_ranges_exist="overwrite",
        source="plate_A",
    )
    with partial:
        _scan(_meter(tmp_path, "a"), (0.0, 1.0))
        _scan(_meter(tmp_path, "c"), (-2.0, 4.0))

    assert partial.report() == (
        "wrote range.json from 2 of 3 sequences, 1 skipped: [-2, 4]"
    )


def test_the_line_says_how_much_of_it_cost_nothing(tmp_path):
    # The whole point of the second run, and the number that says the reuse
    # worked: without it a run that measured two and a run that measured none
    # write the same line.
    settings = {"filter": {"kind": "identity"}}
    first = RangeDocument(
        tmp_path / "range",
        contents=_contents("a", "b"),
        source="plate_A",
        settings=settings,
    )
    with first:
        _scan(_meter(tmp_path, "a", settings), (0.0, 1.0))
        _scan(_meter(tmp_path, "b", settings), (2.0, 3.0))

    again = RangeDocument(
        tmp_path / "range",
        contents=_contents("a", "b"),
        source="plate_A",
        settings=settings,
        if_ranges_exist="reuse",
    )
    with again:
        pass

    assert again.report() == "wrote range.json from 2 sequences, 2 reused: [0, 3]"


def test_a_document_that_covered_none_names_no_bounds(tmp_path):
    # A run whose every sequence failed has nothing to take bounds from, and
    # the line has to stop rather than reach for a range that is not there.
    document = RangeDocument(
        tmp_path / "range", contents=_contents("a", "b"), source="plate_A"
    )
    with document:
        pass

    assert "dataset" not in _saved(tmp_path)
    assert document.report() == "wrote range.json from 0 of 2 sequences, 2 skipped"


def test_the_skipped_list_keeps_the_contents_s_own_order(tmp_path):
    # Not the folded order and not sorted: the contents is what a caller reads the
    # list back against, and it is the order the run was going to take.
    document = RangeDocument(
        tmp_path / "range", contents=_contents("c", "a", "b"), source="plate_A"
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
    # is a hand edit or a version skew: either way the message has to say which
    # entry, since the file is what the reader will go and look at.
    with pytest.raises(ValueError, match=rf"malformed range document: '{named}'"):
        kind.from_dict(document)


@pytest.mark.parametrize(
    ("named", "value"), (("min_value", True), ("max_value", False))
)
def test_a_boolean_bound_is_refused_rather_than_read_as_one_or_zero(named, value):
    # `isinstance(True, int)` is true, so `true` would read as 1.0, and the
    # pair {"min_value": true, "max_value": false} as [1.0, 0.0], a range
    # running backwards that nothing downstream would question.
    document = {"source": "a", "min_value": 0.0, "max_value": 1.0, named: value}

    with pytest.raises(ValueError, match=rf"malformed range document: '{named}'"):
        FrameRange.from_dict(document)


@pytest.mark.parametrize(
    ("named", "value"),
    (
        ("min_value", float("nan")),
        ("max_value", float("nan")),
        ("min_value", float("-inf")),
        ("max_value", float("inf")),
    ),
)
def test_a_non_finite_bound_is_refused_where_it_is_read(named, value):
    # `min` and `max` carry a NaN through or drop it depending on which part
    # holds it, so a document with one folds to whatever order its parts were
    # in. Refused at the read, since nothing this project writes holds one.
    document = {"source": "a", "min_value": 0.0, "max_value": 1.0, named: value}

    with pytest.raises(ValueError, match=rf"malformed range document: '{named}'"):
        FrameRange.from_dict(document)


def test_a_range_that_runs_backwards_is_refused():
    # What a bound is for is comparison, and a pair the wrong way round answers
    # every one of them wrongly without ever looking malformed.
    with pytest.raises(ValueError, match=r"inverted range in 'a'"):
        FrameRange("a", 1.0, 0.0)

    with pytest.raises(ValueError, match=r"inverted range in 'a'"):
        FrameRange.from_dict({"source": "a", "min_value": 1.0, "max_value": 0.0})


def test_an_integer_bound_is_taken_as_the_float_it_stands_for():
    # JSON writes a whole number without its point, so a bound that happens to
    # land on one comes back as `int` and must not be refused for it.
    frame = FrameRange.from_dict({"source": "a", "min_value": 0, "max_value": 2})

    assert (frame.min_value, frame.max_value) == (0.0, 2.0)
    assert isinstance(frame.min_value, float)


def test_a_document_written_without_coverage_leaves_the_block_out(tmp_path):
    # The writer keeps `coverage` optional for a caller that has no contents to
    # measure against: a merge over several documents, whose own coverage is
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

    # A time-lapse folder may carry a dot: a date, a magnification, a repeat
    # marker. The tail after it is part of the name, not an extension the part
    # file has to justify.
    document = RangeDocument(
        tmp_path / "range", contents=_contents(source), source="plate_A"
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


def test_a_fold_is_the_same_whichever_order_the_parts_arrive_in():
    # The order dependence a NaN used to introduce, pinned from the other side:
    # the same parts in any order fold to the same bounds.
    bounds = ((2.0, 3.0), (0.5, 9.0), (1.0, 4.0))

    forward = _sequence("TL_00", *bounds)
    reversed_ = _sequence("TL_00", *reversed(bounds))

    assert (forward.min_value, forward.max_value) == (0.5, 9.0)
    assert (reversed_.min_value, reversed_.max_value) == (0.5, 9.0)


def test_a_document_is_written_as_json_a_strict_reader_accepts(tmp_path):
    # `json.dumps` writes `NaN` and `Infinity` by default, which are not JSON
    # and which a strict reader refuses. Nothing here can hold one, so the
    # guard is what says a settings block cannot smuggle one in either.
    path = tmp_path / "value_range.json"
    dataset = DatasetRange("plate_A", (_sequence("TL_00", (0.0, 1.0)),))

    with pytest.raises(ValueError, match=r"Out of range float"):
        save_range_document(path, dataset, settings={"threshold": float("nan")})

    assert not path.exists()


# ------------------------- a part that cannot be read --------------------- #


def test_an_unreadable_part_says_which_one_it_was(tmp_path):
    # The folder holds one file per sequence, so without the name there is
    # nothing to go and look at, and the fold reads them in sorted order,
    # which is not the order they were written in.
    document = RangeDocument(
        tmp_path / "range", contents=_contents("a", "b"), source="p"
    )
    _scan(_meter(tmp_path, "a"), (0.0, 1.0))
    _scan(_meter(tmp_path, "b"), (0.0, 5.0))
    (document.parts_root / "b.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match=r"unreadable part 'b'.*run it again"):
        document.to_range()


def test_a_part_holding_a_backwards_range_is_named_too(tmp_path):
    # Malformed is not only unparseable: a part whose bounds are the wrong way
    # round parses cleanly and is refused a layer later, and the name has to
    # survive that far too.
    document = RangeDocument(tmp_path / "range", contents=_contents("a"), source="p")
    _scan(_meter(tmp_path, "a"), (0.0, 1.0))

    frame = {"source": "f", "min_value": 1.0, "max_value": 0.0}
    broken = json.dumps({"source": "a", "frames": [frame]})
    (document.parts_root / "a.json").write_text(broken, encoding="utf-8")

    with pytest.raises(ValueError, match=r"unreadable part 'a'.*inverted range"):
        document.to_range()


def test_a_meter_takes_its_part_back_when_another_branch_fails(tmp_path):
    # The document counts a sequence as covered when its part is there, so a
    # part outliving the frames it was measured beside would be counted while
    # nothing of that sequence is on disk.
    meter = _meter(tmp_path, "a")
    _scan(meter, (0.0, 1.0))
    part = tmp_path / "range.parts" / "a.json"
    assert part.exists()

    meter.revert()

    assert not part.exists()


def test_a_meter_that_wrote_nothing_reverts_to_nothing(tmp_path):
    # A meter closed after an error writes no part, so reverting has nothing of
    # its own to remove, and must not reach for a name someone else holds.
    meter = _meter(tmp_path, "a")
    (tmp_path / "range.parts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "range.parts" / "a.json").write_text("theirs", encoding="utf-8")

    meter.revert()

    assert (tmp_path / "range.parts" / "a.json").read_text(encoding="utf-8") == "theirs"


# ------------------------------- reusing parts ---------------------------- #

SETTINGS = {"filter": {"kind": "identity"}}


def _measured(document: RangeDocument, tmp_path, name: str, *bounds) -> bool:
    """Range `name` through the document, saying whether it had to be measured."""
    hook = document.get_hook(_Named(name))
    if hook is None:
        return False

    _scan(hook, *bounds)

    return True


class _Named:
    def __init__(self, name: str) -> None:
        self.name = name


def _reusing(tmp_path, *names: str, settings=SETTINGS) -> RangeDocument:
    return RangeDocument(
        tmp_path / "range",
        contents=_contents(*names),
        settings=settings,
        source="plate_A",
        if_ranges_exist="reuse",
    )


def test_a_part_that_still_describes_the_run_is_kept_rather_than_measured(tmp_path):
    # What reuse is for is the frames it does not read, so the meter being absent
    # is the assertion: a document that handed one out would have re-read the
    # sequence whatever the numbers came out as.
    with _reusing(tmp_path, "a", "b") as first:
        assert _measured(first, tmp_path, "a", (0.0, 1.0))
        assert _measured(first, tmp_path, "b", (2.0, 3.0))

    with _reusing(tmp_path, "a", "b") as second:
        assert not _measured(second, tmp_path, "a", (9.0, 9.0))
        assert not _measured(second, tmp_path, "b", (9.0, 9.0))

    assert _saved(tmp_path)["coverage"]["reused"] == 2
    assert _saved(tmp_path)["dataset"]["max_value"] == 3.0  # not the 9.0 above


def test_a_part_left_under_other_settings_is_measured_again(tmp_path):
    # The numbers a part holds are the ones its settings produced, so a run that
    # filters differently cannot stand on them, and the stale part must not
    # reach the fold either, or the document would mix two filters.
    with _reusing(tmp_path, "a") as first:
        _measured(first, tmp_path, "a", (0.0, 1.0))

    changed = {"filter": {"kind": "median", "radius": [1, 1, 1]}}
    with _reusing(tmp_path, "a", settings=changed) as second:
        assert _measured(second, tmp_path, "a", (4.0, 5.0))

    assert _saved(tmp_path)["dataset"]["max_value"] == 5.0
    assert _saved(tmp_path)["settings"] == changed


def test_a_sequence_whose_frames_changed_is_measured_again(tmp_path):
    # Same name, same settings, different frames: the part describes a read the
    # source would no longer produce. Only the contents knows, which is why it
    # carries the frame names rather than the count.
    with _reusing(tmp_path, "a") as first:
        _measured(first, tmp_path, "a", (0.0, 1.0))

    grown = RangeDocument(
        tmp_path / "range",
        contents=_contents("a", frames=2),
        settings=SETTINGS,
        source="plate_A",
        if_ranges_exist="reuse",
    )
    with grown as second:
        assert _measured(second, tmp_path, "a", (0.0, 1.0), (2.0, 3.0))

    assert _saved(tmp_path)["coverage"]["reused"] == 0


def test_a_part_the_source_has_lost_stays_out_of_the_document(tmp_path):
    # Folding it would widen the bounds with a sequence nobody can go and look
    # at. Left on disk, since a half mounted share makes the same absence.
    with _reusing(tmp_path, "a", "gone") as first:
        _measured(first, tmp_path, "a", (0.0, 1.0))
        _measured(first, tmp_path, "gone", (-9.0, 9.0))

    with _reusing(tmp_path, "a") as second:
        assert second.list_unsourced() == ["gone"]

    assert (tmp_path / "range.parts" / "gone.json").exists()
    assert _saved(tmp_path)["dataset"]["max_value"] == 1.0
    assert _saved(tmp_path)["coverage"]["found"] == 1


def test_a_part_the_source_has_lost_goes_when_the_policy_says_so(tmp_path):
    with _reusing(tmp_path, "a", "gone") as first:
        _measured(first, tmp_path, "a", (0.0, 1.0))
        _measured(first, tmp_path, "gone", (-9.0, 9.0))

    dropping = RangeDocument(
        tmp_path / "range",
        contents=_contents("a"),
        settings=SETTINGS,
        source="plate_A",
        if_ranges_exist="reuse",
        if_sources_gone="delete",
    )
    with dropping:
        pass

    assert not (tmp_path / "range.parts" / "gone.json").exists()
    assert (tmp_path / "range.parts" / "a.json").exists()


def test_a_narrowed_run_leaves_the_parts_it_was_not_given(tmp_path):
    # The retry: name the one that failed and keep the rest. Clearing what the
    # contents does not name would take them, which is why the contents is the
    # dataset and the selection is a separate thing.
    with _reusing(tmp_path, "a", "b", "c") as first:
        for name, low in (("a", 0.0), ("b", 2.0), ("c", 4.0)):
            _measured(first, tmp_path, name, (low, low + 1.0))

    retry = RangeDocument(
        tmp_path / "range",
        contents=_contents("a", "b", "c"),
        settings=SETTINGS,
        source="plate_A",
        selected=["b"],
        if_ranges_exist="reuse",
    )
    with retry:
        assert not _measured(retry, tmp_path, "b", (0.0, 0.0))

    coverage = _saved(tmp_path)["coverage"]
    assert coverage == {
        "found": 3,
        "selected": 1,
        "covered": 3,
        "reused": 3,
        "skipped": [],
        "unselected": [],
    }


@pytest.mark.parametrize("policy", ("error", "overwrite"))
def test_the_other_policies_still_clear_the_folder(tmp_path, policy):
    # Reuse is what stopped the clearing, so the two that did not ask for it
    # must still get a folder holding only what this run put there.
    with _reusing(tmp_path, "a", "b") as first:
        _measured(first, tmp_path, "a", (0.0, 1.0))
        _measured(first, tmp_path, "b", (2.0, 3.0))

    fresh = RangeDocument(
        tmp_path / "range",
        contents=_contents("a", "b"),
        settings=SETTINGS,
        source="plate_A",
        if_ranges_exist=policy,
    )
    if policy == "error":
        (tmp_path / "range.json").unlink()

    with fresh:
        assert _measured(fresh, tmp_path, "a", (5.0, 6.0))
        assert not (tmp_path / "range.parts" / "b.json").exists()

    assert _saved(tmp_path)["coverage"]["reused"] == 0
    assert _saved(tmp_path)["coverage"]["skipped"] == ["b"]


def test_a_part_that_is_not_a_mapping_at_all_is_refused(tmp_path):
    # JSON at the top may be a list or a bare number, and `.get` on either is an
    # attribute error several frames later rather than a malformed document.
    document = _reusing(tmp_path, "a")
    document.parts_root.mkdir(parents=True, exist_ok=True)
    (document.parts_root / "a.json").write_text("[1, 2]", encoding="utf-8")

    with pytest.raises(ValueError, match=r"unreadable part 'a'.*list at the top"):
        document.to_range()


def test_a_part_that_cannot_be_read_is_measured_again_rather_than_reused(tmp_path):
    # The judgement is "does this still describe the run", and one that cannot
    # be parsed cannot describe anything, so it is not reusable, and the fold
    # is where it becomes an error.
    document = _reusing(tmp_path, "a")
    document.parts_root.mkdir(parents=True, exist_ok=True)
    (document.parts_root / "a.json").write_text("{not json", encoding="utf-8")

    with document:
        assert _measured(document, tmp_path, "a", (0.0, 1.0))

    assert _saved(tmp_path)["coverage"]["reused"] == 0


def test_a_part_filed_under_another_sequence_is_not_reused(tmp_path):
    # It parses and its settings match, but the name it is filed under is not
    # the one it holds, so nothing about it can be trusted to describe this
    # sequence. Measuring it again is also what repairs it: the fold would
    # otherwise refuse the whole document over one part.
    with _reusing(tmp_path, "a", "b") as first:
        _measured(first, tmp_path, "a", (0.0, 1.0))
        _measured(first, tmp_path, "b", (2.0, 3.0))

    part = tmp_path / "range.parts" / "b.json"
    body = json.loads(part.read_text(encoding="utf-8"))
    body["source"] = "a"
    part.write_text(json.dumps(body), encoding="utf-8")

    with _reusing(tmp_path, "a", "b") as second:
        assert _measured(second, tmp_path, "b", (4.0, 5.0))

    assert _saved(tmp_path)["coverage"]["reused"] == 1  # 'a', not 'b'
    assert _saved(tmp_path)["dataset"]["max_value"] == 5.0


def test_dropping_the_unsourced_names_what_it_removed(tmp_path):
    with _reusing(tmp_path, "a", "gone") as first:
        _measured(first, tmp_path, "a", (0.0, 1.0))
        _measured(first, tmp_path, "gone", (2.0, 3.0))

    assert _reusing(tmp_path, "a").drop_unsourced() == ["gone"]
    assert _reusing(tmp_path, "a").drop_unsourced() == []


def test_a_stale_part_nobody_re_measured_stays_out_of_the_fold(tmp_path):
    # Change the filter and narrow at once: the sequences left out keep parts
    # the new settings did not produce, and nothing in this run will overwrite
    # them. Folding them would put two filters in one document.
    with _reusing(tmp_path, "a", "b") as first:
        _measured(first, tmp_path, "a", (0.0, 1.0))
        _measured(first, tmp_path, "b", (8.0, 9.0))

    changed = {"filter": {"kind": "median", "radius": [1, 1, 1]}}
    narrowed = RangeDocument(
        tmp_path / "range",
        contents=_contents("a", "b"),
        settings=changed,
        source="plate_A",
        selected=["a"],
        if_ranges_exist="reuse",
    )
    with narrowed:
        assert _measured(narrowed, tmp_path, "a", (2.0, 3.0))

    assert (tmp_path / "range.parts" / "b.json").exists()  # left alone
    assert _saved(tmp_path)["dataset"]["max_value"] == 3.0  # not b's 9.0
    assert _saved(tmp_path)["coverage"]["unselected"] == ["b"]


def test_settings_are_compared_as_a_part_on_disk_holds_them(tmp_path):
    # A tuple is written as a list and read back as one, so comparing what is
    # held against what was written finds every part stale: this run's own
    # included, which leaves the document folding nothing at all.
    tupled = {"filter": {"kind": "median", "radius": (1, 1, 1)}}
    document = RangeDocument(
        tmp_path / "range",
        contents=_contents("a"),
        settings=tupled,
        source="plate_A",
        if_ranges_exist="reuse",
    )
    with document:
        assert _measured(document, tmp_path, "a", (0.0, 1.0))

    assert _saved(tmp_path)["dataset"]["sequences"][0]["source"] == "a"
    assert _saved(tmp_path)["coverage"]["covered"] == 1


def test_a_stale_part_is_left_out_of_the_fold_as_well_as_of_the_reuse(tmp_path):
    # Judging and folding ask the same question, so a part whose frames have
    # moved is passed over by both. Folding it would put a range measured over
    # a different read of the sequence into the document, with nothing saying so.
    settings = {"filter": {"kind": "identity"}}
    with RangeDocument(
        tmp_path / "range",
        contents=_contents("a", "b"),
        settings=settings,
        source="plate_A",
        if_ranges_exist="reuse",
    ) as first:
        _measured(first, tmp_path, "a", (0.0, 1.0))
        _measured(first, tmp_path, "b", (8.0, 9.0))

    # `b` grew a frame and is not selected, so nothing this run does can refresh it.
    grown = RangeDocument(
        tmp_path / "range",
        contents={"a": _contents("a")["a"], "b": _contents("b", frames=2)["b"]},
        settings=settings,
        source="plate_A",
        selected=["a"],
        if_ranges_exist="reuse",
    )
    with grown:
        pass

    assert (tmp_path / "range.parts" / "b.json").exists()
    assert _saved(tmp_path)["dataset"]["max_value"] == 1.0  # not b's 9.0
    assert _saved(tmp_path)["coverage"]["unselected"] == ["b"]


def test_a_document_is_written_even_when_the_unsourced_cannot_be_dropped(
    tmp_path, monkeypatch
):
    # Removing them is tidying rather than part of the answer, so a folder that
    # will not come away must not cost the run its document, which is exactly
    # what refusing to write left behind before.
    with _reusing(tmp_path, "a", "gone") as first:
        _measured(first, tmp_path, "a", (0.0, 1.0))
        _measured(first, tmp_path, "gone", (2.0, 3.0))

    dropping = RangeDocument(
        tmp_path / "range",
        contents=_contents("a"),
        settings=SETTINGS,
        source="plate_A",
        if_ranges_exist="reuse",
        if_sources_gone="delete",
    )

    def refuse(self) -> list[str]:
        msg = "it will not come away"
        raise OSError(msg)

    monkeypatch.setattr(type(dropping), "drop_unsourced", refuse)

    with pytest.raises(OSError, match="will not come away"), dropping:
        pass

    assert _saved(tmp_path)["coverage"]["covered"] == 1


def test_dropping_a_nested_unsourced_part_takes_what_it_empties(tmp_path):
    document = RangeDocument(
        tmp_path / "range",
        contents=_contents("plate/2026.03.11/kept"),
        settings=SETTINGS,
        source="plate_A",
        if_ranges_exist="reuse",
        if_sources_gone="delete",
    )
    part = document.parts_root / "plate" / "2026.03.12" / "gone.json"
    part.parent.mkdir(parents=True)
    part.write_text('{"source": "x", "frames": []}', encoding="utf-8")

    document.drop_unsourced()

    assert not (document.parts_root / "plate" / "2026.03.12").exists()
