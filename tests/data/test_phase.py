from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from iivs.dhm.data.phase import PhaseBinFolder, PhaseUnit, read_phase_bin_header

from iivs_cardio.common.pipeline import Slot
from iivs_cardio.data.phase import phase_field_writer, save_phase_bin_folder

PIXEL_SIZE = 1.5e-7
SOURCE = Path("00000_phase.bin")  # a step says where it was read from
HEIGHT_SCALE = 2.0e-7


def _frames(count: int = 4, height: int = 4, width: int = 5) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.random((height, width), dtype=np.float32) for _ in range(count)]


def _save(dest, frames, **kwargs):
    save_phase_bin_folder(
        dest, frames, pixel_size=PIXEL_SIZE, height_scale=HEIGHT_SCALE, **kwargs
    )


def test_written_folder_reads_back_with_the_same_values(tmp_path):
    frames = _frames()
    dest = tmp_path / "Bin"

    _save(dest, frames)

    folder = PhaseBinFolder(dest)
    assert len(folder) == len(frames)
    for index, frame in enumerate(frames):
        assert np.array_equal(folder[index], frame)


def test_frames_are_numbered_from_zero(tmp_path):
    # The source may be strided or start elsewhere; the written tree is dense.
    dest = tmp_path / "Bin"

    _save(dest, _frames(3))

    assert sorted(p.name for p in dest.iterdir()) == [
        "00000_phase.bin",
        "00001_phase.bin",
        "00002_phase.bin",
    ]


def test_the_header_carries_the_scale_it_was_given(tmp_path):
    dest = tmp_path / "Bin"

    _save(dest, _frames(1))

    header = read_phase_bin_header(dest / "00000_phase.bin")
    assert header.pixel_size == pytest.approx(PIXEL_SIZE)
    assert header.height_scale == pytest.approx(HEIGHT_SCALE)
    assert header.unit == PhaseUnit.RADIANS


def test_nanometres_are_normalized_to_metres(tmp_path):
    # The `.bin` header cannot hold NANOMETERS, so the writer converts values and
    # records METERS. A caller therefore states the unit its frames are in and
    # does not pre-convert -- doing both would scale twice.
    dest = tmp_path / "Bin"
    frame = np.array([[1000.0]], dtype=np.float32)  # nanometres

    _save(dest, [frame], unit=PhaseUnit.NANOMETERS)

    assert read_phase_bin_header(dest / "00000_phase.bin").unit == PhaseUnit.METERS
    assert PhaseBinFolder(dest)[0] == pytest.approx(1e-6)  # metres


def test_frames_are_consumed_one_at_a_time(tmp_path):
    # A whole sequence must never be held: the writer has to pull from the
    # generator, not materialise it.
    live = []

    def stream():
        for frame in _frames(4):
            live.append(len(live))
            yield frame

    _save(tmp_path / "Bin", stream())

    assert len(live) == 4


def test_an_existing_folder_is_kept_unless_overwrite(tmp_path):
    dest = tmp_path / "Bin"
    _save(dest, _frames(2))

    with pytest.raises(FileExistsError):
        _save(dest, _frames(3))

    assert len(PhaseBinFolder(dest)) == 2  # the first write is still intact


def test_overwrite_replaces_the_folder(tmp_path):
    dest = tmp_path / "Bin"
    _save(dest, _frames(2))

    _save(dest, _frames(3), overwrite=True)

    assert len(PhaseBinFolder(dest)) == 3


def test_a_failure_part_way_leaves_the_previous_folder_untouched(tmp_path):
    dest = tmp_path / "Bin"
    _save(dest, _frames(2))

    def stream():
        yield from _frames(1)
        msg = "boom"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="boom"):
        _save(dest, stream(), overwrite=True)

    assert len(PhaseBinFolder(dest)) == 2  # staged: no half-written tree


def test_an_empty_sequence_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty phase sequence"):
        _save(tmp_path / "Bin", [])


def test_a_nonfinite_frame_is_written_without_complaint(tmp_path, recwarn):
    # `on_nonfinite` defaults to "ignore" here: a filter dropping an out-of-range
    # neighbour can produce NaN, and the range document reports it.
    frame = _frames(1)[0]
    frame[0, 0] = np.nan

    _save(tmp_path / "Bin", [frame])

    assert not recwarn.list


def test_the_push_writer_matches_what_the_pull_one_writes(tmp_path):
    # `phase_field_writer` is the same folder for a traversal that hands one step
    # at a time; the two shapes must not disagree on the bytes.
    frames = _frames(3)
    pulled = tmp_path / "pulled"
    pushed = tmp_path / "pushed"

    _save(pulled, frames)
    with phase_field_writer(
        pushed, pixel_size=PIXEL_SIZE, height_scale=HEIGHT_SCALE
    ) as writer:
        for index, frame in enumerate(frames):
            writer.write(Slot(index, (torch.from_numpy(frame), SOURCE)))

    assert sorted(p.name for p in pushed.iterdir()) == sorted(
        p.name for p in pulled.iterdir()
    )
    for index in range(len(frames)):
        assert np.array_equal(
            PhaseBinFolder(pushed)[index], PhaseBinFolder(pulled)[index]
        )


def test_the_push_writer_records_the_scale_and_unit_it_was_given(tmp_path):
    dest = tmp_path / "Bin"

    with phase_field_writer(
        dest,
        pixel_size=PIXEL_SIZE,
        height_scale=HEIGHT_SCALE,
        unit=PhaseUnit.NANOMETERS,
    ) as writer:
        writer.write(Slot(0, (torch.from_numpy(_frames(1)[0]), SOURCE)))

    header = read_phase_bin_header(dest / "00000_phase.bin")
    assert header.pixel_size == pytest.approx(PIXEL_SIZE)
    assert header.height_scale == pytest.approx(HEIGHT_SCALE)
    assert header.unit is PhaseUnit.METERS  # nanometres cannot live in a header
