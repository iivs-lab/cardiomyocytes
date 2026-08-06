from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from iivs.dhm.data.phase import PhaseBinFolder, PhaseUnit, read_phase_bin_header

from iivs_cardio.common.pipeline import Step
from iivs_cardio.data.phase import phase_frame_writer

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

PIXEL_SIZE = 1.5e-7
HEIGHT_SCALE = 2.0e-7


def _frames(count: int = 4, height: int = 4, width: int = 5) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.random((height, width), dtype=np.float32) for _ in range(count)]


def _write(dest: Path, frames: Iterable[np.ndarray], **kwargs) -> None:
    writer = phase_frame_writer(
        dest, pixel_size=PIXEL_SIZE, height_scale=HEIGHT_SCALE, **kwargs
    )
    with writer:
        for index, frame in enumerate(frames):
            writer.write(Step(index, torch.from_numpy(frame)))


def test_written_folder_reads_back_with_the_same_values(tmp_path):
    frames = _frames()
    dest = tmp_path / "Bin"

    _write(dest, frames)

    folder = PhaseBinFolder(dest)
    assert len(folder) == len(frames)
    for index, frame in enumerate(frames):
        assert np.array_equal(folder[index], frame)


def test_frames_are_numbered_from_zero(tmp_path):
    # The source may be strided or start elsewhere; the written tree is dense.
    dest = tmp_path / "Bin"

    _write(dest, _frames(3))

    assert sorted(p.name for p in dest.iterdir()) == [
        "00000_phase.bin",
        "00001_phase.bin",
        "00002_phase.bin",
    ]


def test_the_header_carries_the_scale_it_was_given(tmp_path):
    dest = tmp_path / "Bin"

    _write(dest, _frames(1))

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

    _write(dest, [frame], unit=PhaseUnit.NANOMETERS)

    assert read_phase_bin_header(dest / "00000_phase.bin").unit == PhaseUnit.METERS
    assert PhaseBinFolder(dest)[0] == pytest.approx(1e-6)  # metres


def test_an_existing_folder_is_kept_unless_overwrite(tmp_path):
    dest = tmp_path / "Bin"
    _write(dest, _frames(2))

    with pytest.raises(FileExistsError):
        _write(dest, _frames(3))

    assert len(PhaseBinFolder(dest)) == 2  # the first write is still intact


def test_overwrite_replaces_the_folder(tmp_path):
    dest = tmp_path / "Bin"
    _write(dest, _frames(2))

    _write(dest, _frames(3), overwrite=True)

    assert len(PhaseBinFolder(dest)) == 3


def test_a_failure_part_way_leaves_the_previous_folder_untouched(tmp_path):
    dest = tmp_path / "Bin"
    _write(dest, _frames(2))

    def stream():
        yield from _frames(1)
        msg = "boom"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="boom"):
        _write(dest, stream(), overwrite=True)

    assert len(PhaseBinFolder(dest)) == 2  # staged: no half-written tree


def test_an_empty_sequence_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="no frame was written"):
        _write(tmp_path / "Bin", [])


@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf))
def test_a_nonfinite_frame_is_refused_rather_than_cached(tmp_path, bad):
    # What this writes is the cache the next stage reads, and the format itself
    # stores a NaN happily -- so a value let through here is one the run that
    # meets it has no way to trace back. The folder is left uncommitted.
    dest = tmp_path / "Bin"
    frame = _frames(1)[0]
    frame[0, 0] = bad

    with pytest.raises(ValueError, match="finite"):
        _write(dest, [frame])

    assert not dest.exists()


def test_an_absent_step_writes_nothing(tmp_path):
    # The tail of a stage spanning several steps: the folder ends earlier.
    dest = tmp_path / "Bin"
    writer = phase_frame_writer(dest, pixel_size=PIXEL_SIZE, height_scale=HEIGHT_SCALE)

    with writer:
        writer.write(Step(0, torch.from_numpy(_frames(1)[0])))
        writer.write(Step[torch.Tensor, None](1, None))

    assert len(PhaseBinFolder(dest)) == 1
