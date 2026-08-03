from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from iivs_cardio.common.pipeline import Slot, drain
from iivs_cardio.common.writer import FieldWriter
from iivs_cardio.optical_flow.data.folder import OpticalFlowFolder, save_flow_npy

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from numpy.typing import NDArray


def _save_text(path: Path, field: str) -> None:
    path.write_text(field, encoding="utf-8")


def _refuse(path: Path, field: str) -> None:
    msg = "the disk gave up"
    raise RuntimeError(msg)


def _write_all(
    dest: Path,
    slots: Iterable[Slot[str]],
    *,
    save: Callable[[Path, str], None] = _save_text,
    overwrite: bool = False,
) -> None:
    """Drive a whole folder in one call, so `pytest.raises` wraps one statement."""
    with FieldWriter(dest, save, stem="field", ext="txt", overwrite=overwrite) as w:
        drain(slots, w.write)


def _names(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir())


def test_written_folder_reads_back_through_its_own_reader(tmp_path: Path) -> None:
    # The naming is `koala_frame_name`'s and the folder readers discover by it,
    # so a round trip is what would catch the two drifting apart.
    dest = tmp_path / "flow"
    flows: list[NDArray[np.float32]] = [
        np.full((2, 4, 5), index, dtype=np.float32) for index in range(3)
    ]

    with FieldWriter(dest, save_flow_npy, stem="flow", ext="npy") as writer:
        drain([Slot(i, flow) for i, flow in enumerate(flows)], writer.write)

    folder = OpticalFlowFolder(dest)

    assert len(folder) == 3
    for index, expected in enumerate(flows):
        np.testing.assert_array_equal(folder.get_item(index), expected)


def test_names_each_field_by_its_slot_index(tmp_path: Path) -> None:
    dest = tmp_path / "fields"

    _write_all(dest, [Slot(index, f"f{index}") for index in range(3)])

    assert _names(dest) == ["00000_field.txt", "00001_field.txt", "00002_field.txt"]
    assert (dest / "00002_field.txt").read_text(encoding="utf-8") == "f2"


def test_an_absent_slot_writes_nothing(tmp_path: Path) -> None:
    dest = tmp_path / "fields"

    # The tail a forward-convention node leaves: the folder ends earlier than
    # the sequence it came from, rather than gaining a placeholder.
    _write_all(dest, [Slot(0, "a"), Slot(1, "b"), Slot(2, None)])

    assert _names(dest) == ["00000_field.txt", "00001_field.txt"]


def test_a_gap_is_refused_rather_than_closed(tmp_path: Path) -> None:
    dest = tmp_path / "fields"
    slots = [Slot(0, "a"), Slot(1, None), Slot[str](2, "c")]

    with pytest.raises(ValueError, match=r"non-contiguous field 2: expected 1"):
        _write_all(dest, slots)

    assert not dest.exists()


def test_a_repeated_step_is_refused(tmp_path: Path) -> None:
    dest = tmp_path / "fields"

    with pytest.raises(ValueError, match=r"non-contiguous field 0: expected 1"):
        _write_all(dest, [Slot(0, "a"), Slot(0, "again")])

    assert not dest.exists()


def test_a_folder_not_starting_at_zero_is_refused(tmp_path: Path) -> None:
    dest = tmp_path / "fields"

    with pytest.raises(ValueError, match=r"non-contiguous field 1: expected 0"):
        _write_all(dest, [Slot(1, "a")])

    assert not dest.exists()


def test_writing_nothing_commits_nothing(tmp_path: Path) -> None:
    dest = tmp_path / "fields"

    with pytest.raises(ValueError, match=r"no field was written"):
        _write_all(dest, [Slot[str](0, None)])

    assert not dest.exists()


def test_a_failure_part_way_leaves_no_folder(tmp_path: Path) -> None:
    dest = tmp_path / "fields"

    with pytest.raises(RuntimeError, match="the disk gave up"):
        _write_all(dest, [Slot(0, "a")], save=_refuse)

    assert not dest.exists()


def test_a_failure_leaves_an_existing_folder_untouched(tmp_path: Path) -> None:
    dest = tmp_path / "fields"
    dest.mkdir()
    (dest / "00000_field.txt").write_text("original", encoding="utf-8")

    with pytest.raises(RuntimeError, match="the disk gave up"):
        _write_all(dest, [Slot(0, "replacement")], save=_refuse, overwrite=True)

    assert (dest / "00000_field.txt").read_text(encoding="utf-8") == "original"


def test_overwrite_replaces_the_folder_wholesale(tmp_path: Path) -> None:
    dest = tmp_path / "fields"
    dest.mkdir()
    (dest / "00000_field.txt").write_text("original", encoding="utf-8")
    (dest / "00001_field.txt").write_text("stale", encoding="utf-8")

    _write_all(dest, [Slot(0, "fresh")], overwrite=True)

    assert _names(dest) == ["00000_field.txt"]
    assert (dest / "00000_field.txt").read_text(encoding="utf-8") == "fresh"
