from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from iivs_cardio.common.pipeline import Step
from iivs_cardio.data.writer import KoalaFrameWriter
from iivs_cardio.optical_flow.data.folder import OpticalFlowFolder, save_flow_npy

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from numpy.typing import NDArray


def _save_text(path: Path, frame: str) -> None:
    path.write_text(frame, encoding="utf-8")


def _refuse(path: Path, frame: str) -> None:
    msg = "the disk gave up"
    raise RuntimeError(msg)


def _write_all(
    dest: Path,
    steps: Iterable[Step[str]],
    *,
    save: Callable[[Path, str], None] = _save_text,
    overwrite: bool = False,
) -> None:
    """Drive a whole folder in one call, so `pytest.raises` wraps one statement."""
    writer = KoalaFrameWriter(dest, save, stem="frame", ext="txt", overwrite=overwrite)
    with writer:
        for step in steps:
            writer.write(step)


def _drive(writer: KoalaFrameWriter[str], steps: Iterable[Step[str]]) -> None:
    """The same, for a writer the caller keeps a hold of to ask afterwards."""
    with writer:
        for step in steps:
            writer.write(step)


def _refuse_the_third(path: Path, frame: str) -> None:
    """A save that works twice and gives up part way, with frames already down."""
    if path.name.startswith("00002"):
        msg = "the disk gave up"
        raise RuntimeError(msg)

    _save_text(path, frame)


def _names(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir())


def test_written_folder_reads_back_through_its_own_reader(tmp_path: Path) -> None:
    # The naming is `koala_frame_name`'s and the folder readers discover by it,
    # so a round trip is what would catch the two drifting apart.
    dest = tmp_path / "flow"
    flows: list[NDArray[np.float32]] = [
        np.full((2, 4, 5), index, dtype=np.float32) for index in range(3)
    ]

    with KoalaFrameWriter(dest, save_flow_npy, stem="flow", ext="npy") as writer:
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

    with pytest.raises(ValueError, match=r"non-contiguous frame 2: expected 1"):
        _write_all(dest, steps)

    assert not dest.exists()


def test_a_repeated_step_is_refused(tmp_path: Path) -> None:
    dest = tmp_path / "frames"

    with pytest.raises(ValueError, match=r"non-contiguous frame 0: expected 1"):
        _write_all(dest, [Step(0, "a"), Step(0, "again")])

    assert not dest.exists()


def test_a_folder_not_starting_at_zero_is_refused(tmp_path: Path) -> None:
    dest = tmp_path / "frames"

    with pytest.raises(ValueError, match=r"non-contiguous frame 1: expected 0"):
        _write_all(dest, [Step(1, "a")])

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

    with KoalaFrameWriter(dest, _save_text, stem="frame", ext="txt") as writer:
        writer(Step(0, "a"))

    assert (dest / "00000_frame.txt").read_text(encoding="utf-8") == "a"


def test_a_writer_ignores_what_a_step_carries_beside_its_value(tmp_path: Path) -> None:
    # `extra` is for the side branches that want it; naming is the index's job.
    dest = tmp_path / "frames"

    with KoalaFrameWriter(dest, _save_text, stem="frame", ext="txt") as writer:
        writer.write(Step(0, "a", "some_other_name.txt"))

    assert _names(dest) == ["00000_frame.txt"]


def test_a_writer_reports_what_it_committed(tmp_path):
    # The count rather than the destination: a count below the sequence's length
    # is what says frames were skipped, and nothing else in a log says it.
    writer = KoalaFrameWriter(tmp_path / "out", _save_text, stem="frame", ext="txt")

    assert writer.report() is None  # nothing written, so nothing to say

    with writer:
        writer.write(Step(0, "a"))
        writer.write(Step(1, "b"))

    assert writer.report() == "wrote 2 frames"


def test_a_writer_that_gave_up_reports_nothing(tmp_path):
    # The count was read off what had been staged, which survives the abort, so
    # a sequence that lost its folder still said "wrote 2 frames". A branch with
    # nothing committed reports `None` -- that is the contract the block relies
    # on to leave the line out rather than print an empty one.
    dest = tmp_path / "out"
    writer = KoalaFrameWriter(dest, _refuse_the_third, stem="frame", ext="txt")
    steps = [Step(0, "a"), Step(1, "b"), Step(2, "c")]

    with pytest.raises(RuntimeError, match="the disk gave up"):
        _drive(writer, steps)

    assert not dest.exists()
    assert writer.report() is None


def test_a_writer_counts_what_it_wrote_not_what_it_was_offered(tmp_path):
    # A step with no value is skipped rather than written, so the two differ.
    writer = KoalaFrameWriter(tmp_path / "out", _save_text, stem="frame", ext="txt")
    with writer:
        writer.write(Step(0, "a"))
        writer.write(Step(1, None))

    assert writer.report() == "wrote 1 frame"
    assert _names(tmp_path / "out") == ["00000_frame.txt"]
