from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from kaparoo.filesystem import StagedDirectory

from iivs_cardio.common.pipeline import Step
from iivs_cardio.common.pipeline.frames import RECORD_FILE, FrameWriter
from iivs_cardio.optical_flow.data.folder import OpticalFlowFolder, save_flow_npy

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from numpy.typing import NDArray


def _name(index: int) -> str:
    return f"{index:05d}_frame.txt"


def _save_text(folder: Path, index: int, frame: str) -> None:
    (folder / _name(index)).write_text(frame, encoding="utf-8")


def _refuse(folder: Path, index: int, frame: str) -> None:
    msg = "the disk gave up"
    raise RuntimeError(msg)


def test_a_writer_is_opened_once(tmp_path: Path) -> None:
    # Committing takes the staged folder away, so a second walk would write
    # where nothing is and file a record naming the frames of both.
    writer = FrameWriter(tmp_path / "out", _save_text)
    _drive(writer, [Step(0, "a")])

    with (
        pytest.raises(RuntimeError, match=r"opened already: one writer per walk"),
        writer,
    ):
        pass


def _write_all(
    dest: Path,
    steps: Iterable[Step[str]],
    *,
    save: Callable[[Path, int, str], None] = _save_text,
    overwrite: bool = False,
) -> None:
    """Drive a whole folder in one call, so `pytest.raises` wraps one statement."""
    writer = FrameWriter(dest, save, overwrite=overwrite)
    with writer:
        for step in steps:
            writer.write(step)


def _drive(writer: FrameWriter[str], steps: Iterable[Step[str]]) -> None:
    """The same, for a writer the caller keeps a hold of to ask afterwards."""
    with writer:
        for step in steps:
            writer.write(step)


def _refuse_the_third(folder: Path, index: int, frame: str) -> None:
    """A save that works twice and gives up part way, with frames already down."""
    if index == 2:
        msg = "the disk gave up"
        raise RuntimeError(msg)

    _save_text(folder, index, frame)


def _names(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir())


def test_written_folder_reads_back_through_its_own_reader(tmp_path: Path) -> None:
    # The naming is `koala_frame_name`'s and the folder readers discover by it,
    # so a round trip is what would catch the two drifting apart.
    dest = tmp_path / "flow"
    flows: list[NDArray[np.float32]] = [
        np.full((2, 4, 5), index, dtype=np.float32) for index in range(3)
    ]

    def save_fn(folder: Path, index: int, flow: NDArray[np.float32]) -> None:
        save_flow_npy(folder / f"{index:05d}_flow.npy", flow)

    with FrameWriter(dest, save_fn) as writer:
        for index, flow in enumerate(flows):
            writer.write(Step(index, flow))

    folder = OpticalFlowFolder(dest)

    assert len(folder) == 3
    for index, expected in enumerate(flows):
        np.testing.assert_array_equal(folder.get_item(index), expected)


def test_names_each_frame_by_its_step_index(tmp_path: Path) -> None:
    dest = tmp_path / "frames"

    _write_all(dest, [Step(index, f"f{index}") for index in range(3)])

    assert _names(dest) == ["00000_frame.txt", "00001_frame.txt", "00002_frame.txt"]
    assert (dest / "00002_frame.txt").read_text(encoding="utf-8") == "f2"


def test_an_absent_step_writes_nothing(tmp_path: Path) -> None:
    dest = tmp_path / "frames"

    # The tail a stage spanning several steps leaves: the folder ends earlier
    # than the sequence it came from, rather than gaining a placeholder.
    _write_all(dest, [Step(0, "a"), Step(1, "b"), Step[str](2, None)])

    assert _names(dest) == ["00000_frame.txt", "00001_frame.txt"]


def test_a_gap_is_refused_rather_than_closed(tmp_path: Path) -> None:
    dest = tmp_path / "frames"
    steps = [Step(0, "a"), Step[str](1, None), Step[str](2, "c")]

    with pytest.raises(ValueError, match=r"frame 2 does not follow 0: expected 1"):
        _write_all(dest, steps)

    assert not dest.exists()


def test_a_repeated_step_is_refused(tmp_path: Path) -> None:
    dest = tmp_path / "frames"

    with pytest.raises(ValueError, match=r"frame 0 does not follow 0: expected 1"):
        _write_all(dest, [Step(0, "a"), Step(0, "again")])

    assert not dest.exists()


def test_a_stream_that_starts_late_is_numbered_from_its_first_frame(
    tmp_path: Path,
) -> None:
    # A stage needing two frames to make one says nothing at step 0, and that is
    # the ordinary shape rather than a hole: the folder holds one frame per
    # thing produced, numbered from zero, and nothing is missing from it.
    dest = tmp_path / "frames"
    steps = [Step[str](0, None), Step(1, "a"), Step(2, "b")]

    _write_all(dest, steps)

    assert sorted(path.name for path in dest.iterdir()) == [
        "00000_frame.txt",
        "00001_frame.txt",
    ]
    assert (dest / "00000_frame.txt").read_text(encoding="utf-8") == "a"


def test_a_late_start_is_not_a_licence_for_a_gap_after_it(tmp_path: Path) -> None:
    # The two look alike from inside `write`: a step with nothing in it. What
    # tells them apart is whether a frame has arrived yet, so the leading run
    # being allowed must not carry over to a hole further along.
    dest = tmp_path / "frames"
    steps = [Step[str](0, None), Step(1, "a"), Step[str](2, None), Step(3, "c")]

    with pytest.raises(ValueError, match=r"frame 3 does not follow 1: expected 2"):
        _write_all(dest, steps)

    assert not dest.exists()


def test_writing_nothing_commits_nothing(tmp_path: Path) -> None:
    dest = tmp_path / "frames"

    with pytest.raises(ValueError, match=r"no frame was written"):
        _write_all(dest, [Step[str](0, None)])

    assert not dest.exists()


def test_a_failure_part_way_leaves_no_folder(tmp_path: Path) -> None:
    dest = tmp_path / "frames"

    with pytest.raises(RuntimeError, match="the disk gave up"):
        _write_all(dest, [Step(0, "a")], save=_refuse)

    assert not dest.exists()


def test_a_failure_leaves_no_empty_shell_of_a_sequence_behind(tmp_path: Path) -> None:
    # Staging needs a parent to sit in, so the whole `<sequence>/Phase/Float`
    # was made before a single frame was written, and the abort dropped only
    # the staged folder. A run over a dataset then left one empty shell per
    # failed sequence in the output tree, each reading as a sequence that is
    # there.
    out = tmp_path / "out"
    out.mkdir()
    dest = out / "TL_00" / "Phase" / "Float" / "Bin"

    with pytest.raises(RuntimeError, match="the disk gave up"):
        _write_all(dest, [Step(0, "a")], save=_refuse)

    assert list(out.iterdir()) == []


def test_a_failure_stops_climbing_at_what_it_did_not_empty(tmp_path: Path) -> None:
    # The climb walks the folders this writer's own opening made, which reaches
    # up to the sequence. Anything that landed under one of them meanwhile is
    # not this writer's to take, whether a second cache of the same time-lapse
    # or whatever else shares the ancestor, so the climb stops there.
    out = tmp_path / "out"
    out.mkdir()
    sequence = out / "TL_00"
    writer = FrameWriter(sequence / "Phase" / "Float" / "Bin", _refuse)
    (sequence / "notes.txt").write_text("a neighbour", encoding="utf-8")

    with pytest.raises(RuntimeError, match="the disk gave up"):
        _drive(writer, [Step(0, "a")])

    assert not (sequence / "Phase").exists()
    assert _names(sequence) == ["notes.txt"]


def test_a_move_that_fails_leaves_no_staging_behind(
    tmp_path: Path, monkeypatch
) -> None:
    # The staged folder sits beside the destination under a hidden name, so a
    # move that fails left a `.tmp` in the output tree for a sequence with none
    # of its frames. The finalizer is the only other hand on it, and it dies
    # with the process, so a run killed here left it for good. The writer is
    # kept alive here for that reason: let it fall out of scope and the
    # finalizer clears up, which is the very thing a killed run cannot do.
    def refuse(self: StagedDirectory) -> None:
        msg = "rename failed"
        raise OSError(msg)

    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(StagedDirectory, "commit", refuse)
    writer = FrameWriter(out / "Bin", _save_text)

    with pytest.raises(OSError, match="rename failed"):
        _drive(writer, [Step(0, "a")])

    assert list(out.iterdir()) == []
    assert writer.report() is None


def test_a_failure_leaves_an_existing_folder_untouched(tmp_path: Path) -> None:
    dest = tmp_path / "frames"
    dest.mkdir()
    (dest / "00000_frame.txt").write_text("original", encoding="utf-8")

    with pytest.raises(RuntimeError, match="the disk gave up"):
        _write_all(dest, [Step(0, "replacement")], save=_refuse, overwrite=True)

    assert (dest / "00000_frame.txt").read_text(encoding="utf-8") == "original"


def test_overwrite_replaces_the_folder_wholesale(tmp_path: Path) -> None:
    dest = tmp_path / "frames"
    dest.mkdir()
    (dest / "00000_frame.txt").write_text("original", encoding="utf-8")
    (dest / "00001_frame.txt").write_text("stale", encoding="utf-8")

    _write_all(dest, [Step(0, "fresh")], overwrite=True)

    assert _names(dest) == ["00000_frame.txt"]
    assert (dest / "00000_frame.txt").read_text(encoding="utf-8") == "fresh"


def test_a_writer_is_callable_so_a_stage_can_register_it(tmp_path):
    dest = tmp_path / "frames"

    with FrameWriter(dest, _save_text) as writer:
        writer(Step(0, "a"))

    assert (dest / "00000_frame.txt").read_text(encoding="utf-8") == "a"


def test_a_writer_ignores_what_a_step_carries_beside_its_value(tmp_path: Path) -> None:
    # `extra` is for the side branches that want it; naming is the index's job.
    dest = tmp_path / "frames"

    with FrameWriter(dest, _save_text) as writer:
        writer.write(Step(0, "a", "some_other_name.txt"))

    assert _names(dest) == ["00000_frame.txt"]


def test_a_writer_reports_what_it_committed(tmp_path):
    # The count rather than the destination: a count below the sequence's length
    # is what says frames were skipped, and nothing else in a log says it.
    writer = FrameWriter(tmp_path / "out", _save_text)

    assert writer.report() is None  # nothing written, so nothing to say

    with writer:
        writer.write(Step(0, "a"))
        writer.write(Step(1, "b"))

    assert writer.report() == "wrote 2 frames"


def test_a_writer_that_gave_up_reports_nothing(tmp_path):
    # The count was read off what had been staged, which survives the abort, so
    # a sequence that lost its folder still said "wrote 2 frames". A branch with
    # nothing committed reports `None`, which is the contract the block relies
    # on to leave the line out rather than print an empty one.
    dest = tmp_path / "out"
    writer = FrameWriter(dest, _refuse_the_third)
    steps = [Step(0, "a"), Step(1, "b"), Step(2, "c")]

    with pytest.raises(RuntimeError, match="the disk gave up"):
        _drive(writer, steps)

    assert not dest.exists()
    assert writer.report() is None


def test_a_writer_counts_what_it_wrote_not_what_it_was_offered(tmp_path):
    # A step with no value is skipped rather than written, so the two differ.
    writer = FrameWriter(tmp_path / "out", _save_text)
    with writer:
        writer.write(Step(0, "a"))
        writer.write(Step(1, None))

    assert writer.report() == "wrote 1 frame"
    assert _names(tmp_path / "out") == ["00000_frame.txt"]


# ------------------------------ the record -------------------------------- #


def _sourced(*names: str) -> list[Step[str, Path]]:
    """Steps carrying where each frame came from, as a real stage does."""
    return [Step(i, name, Path("a/b") / name) for i, name in enumerate(names)]


def test_a_record_says_which_source_frame_each_written_one_came_from(tmp_path):
    # Renumbering is what makes the folder readable again and what loses the
    # source: at a stride, written frame 1 is the source's frame 2, and a phase
    # header carries neither a time nor a name to recover that from.
    dest = tmp_path / "cache"
    writer = FrameWriter(
        dest,
        _save_text,
        record={"settings": {"filter": {"kind": "identity"}}, "source": "plate/TL_00"},
    )

    _drive(writer, _sourced("00000_phase.bin", "00002_phase.bin", "00004_phase.bin"))

    written = json.loads((dest / RECORD_FILE).read_text(encoding="utf-8"))
    assert written == {
        "settings": {"filter": {"kind": "identity"}},
        "source": "plate/TL_00",
        "frames": ["00000_phase.bin", "00002_phase.bin", "00004_phase.bin"],
    }
    assert RECORD_FILE in _names(dest)


def test_a_record_is_filed_under_the_name_the_writer_was_given(tmp_path):
    # The name travels with the tree that will read it back, so a run may keep
    # its account under one the source folders do not already use.
    dest = tmp_path / "cache"
    writer = FrameWriter(
        dest,
        _save_text,
        record={"source": "plate/TL_00"},
        record_file="origin",
    )

    _drive(writer, _sourced("00000_phase.bin"))

    assert "origin.json" in _names(dest)
    assert RECORD_FILE not in _names(dest)

    written = json.loads((dest / "origin.json").read_text(encoding="utf-8"))
    assert written == {"source": "plate/TL_00", "frames": ["00000_phase.bin"]}


def test_a_record_name_that_could_reach_out_of_the_folder_is_refused(tmp_path):
    # It is joined onto the staged folder, so a directory part would file the
    # account somewhere the commit never moves and a reader never looks.
    with pytest.raises(ValueError, match=r"invalid file name '\.\./origin\.json'"):
        FrameWriter(tmp_path / "cache", _save_text, record_file="../origin.json")


def test_a_writer_told_nothing_files_nothing_and_asks_nothing_of_a_step(tmp_path):
    # Flow and metric writers have no record to file yet, and their steps may
    # carry nothing about a source, so the collecting must not happen at all.
    dest = tmp_path / "cache"

    _write_all(dest, [Step(0, "a"), Step(1, "b")])

    assert _names(dest) == ["00000_frame.txt", "00001_frame.txt"]


def test_a_record_asked_for_over_steps_that_name_no_source_is_refused(tmp_path):
    # Better here than a record whose `frames` is short by however many steps
    # said nothing, which reads as a shorter acquisition.
    dest = tmp_path / "cache"
    writer = FrameWriter(dest, _save_text, record={"source": "plate/TL_00"})

    with pytest.raises(ValueError, match="holds nothing beside its value"), writer:
        writer.write(Step(0, "a"))


def test_the_record_lands_with_the_frames_or_not_at_all(tmp_path):
    # Written into the staged folder, so the move that makes the frames visible
    # makes it visible with them. A run that gave up leaves neither.
    dest = tmp_path / "cache"
    writer = FrameWriter(
        dest,
        _refuse_the_third,
        record={"source": "plate/TL_00"},
    )

    with pytest.raises(RuntimeError, match="the disk gave up"):
        _drive(writer, _sourced("a.bin", "b.bin", "c.bin"))

    assert not dest.exists()
