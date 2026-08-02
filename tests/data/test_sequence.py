from __future__ import annotations

import numpy as np
import pytest
import torch
from iivs.dhm.data.hologram import HologramNpyFolder
from iivs.dhm.data.phase import PhaseBinFolder, PhaseBinList, save_phase_bin
from kaparoo.data.sequences import DataSequence
from numpy.typing import NDArray

from iivs_cardio.common import Device
from iivs_cardio.data.sequence import FrameSequence
from iivs_cardio.data.transforms.filtering import (
    FilteredSequence,
    IdentityKernel,
    IdentityParams,
    MedianKernel,
    MedianParams,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="no CUDA-capable GPU detected",
)

PIXEL_SIZE = 1.5e-7
HEIGHT_SCALE = 2.0e-7
RADIUS = (1, 1, 1)


def _frames(count: int = 5, height: int = 4, width: int = 5) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.random((height, width), dtype=np.float32) for _ in range(count)]


def _bin_folder(root, frames: list[np.ndarray]):
    root.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        save_phase_bin(
            root / f"{index:05d}_phase.bin",
            frame,
            pixel_size=PIXEL_SIZE,
            height_scale=HEIGHT_SCALE,
            on_nonfinite="ignore",  # some cases write NaN on purpose
        )
    return root


def _holo_folder(root, count: int = 5):
    root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        np.save(root / f"{index:05d}_holo.npy", np.full((4, 5), index * 10, np.uint8))
    return root


# ------------------------------ unfiltered path -------------------------------- #


def test_unfiltered_returns_float32_tensors_matching_the_source(tmp_path):
    frames = _frames()
    root = _bin_folder(tmp_path / "Bin", frames)
    sequence = FrameSequence(PhaseBinFolder(root), IdentityKernel())

    assert len(sequence) == len(frames)
    item = sequence[2]
    assert isinstance(item, torch.Tensor)
    assert item.dtype == torch.float32
    assert item.shape == (4, 5)
    assert torch.equal(item, torch.from_numpy(frames[2]))


def test_unfiltered_iterates_in_order(tmp_path):
    frames = _frames(3)
    root = _bin_folder(tmp_path / "Bin", frames)
    sequence = FrameSequence(PhaseBinFolder(root), IdentityKernel())
    for loaded, expected in zip(sequence, frames, strict=True):
        assert torch.equal(loaded, torch.from_numpy(expected))


def test_meta_is_the_source_file(tmp_path):
    root = _bin_folder(tmp_path / "Bin", _frames(3))
    sequence = FrameSequence(PhaseBinFolder(root), IdentityKernel())
    assert sequence.get_meta(1) == root / "00001_phase.bin"


def test_negative_index_reaches_the_last_frame(tmp_path):
    frames = _frames(3)
    root = _bin_folder(tmp_path / "Bin", frames)
    sequence = FrameSequence(PhaseBinFolder(root), IdentityKernel())
    assert torch.equal(sequence[-1], torch.from_numpy(frames[-1]))


# ------------------------------- filtered path --------------------------------- #


def test_filtered_matches_the_kernel_applied_to_the_window(tmp_path):
    # Verified against the kernel run on an independently stacked window, not
    # against whatever the sequence itself produced.
    frames = _frames()
    root = _bin_folder(tmp_path / "Bin", frames)
    kernel = MedianKernel(RADIUS)
    sequence = FrameSequence(PhaseBinFolder(root), kernel)

    target = 2
    radius = kernel.temporal_radius
    window = torch.from_numpy(np.stack(frames[target - radius : target + radius + 1]))
    expected = MedianKernel(RADIUS).apply(window, radius)

    assert torch.equal(sequence[target], expected)


def test_filtered_differs_from_the_raw_frame(tmp_path):
    root = _bin_folder(tmp_path / "Bin", _frames())
    plain = FrameSequence(PhaseBinFolder(root), IdentityKernel())
    filtered = FrameSequence(PhaseBinFolder(root), MedianKernel(RADIUS))
    assert not torch.equal(filtered[2], plain[2])


def test_filtered_keeps_the_source_length_and_meta(tmp_path):
    # Filtering truncates the window at the ends rather than dropping frames, so
    # one output per source frame, and the metadata still points at the file.
    root = _bin_folder(tmp_path / "Bin", _frames(4))
    sequence = FrameSequence(PhaseBinFolder(root), MedianKernel(RADIUS))
    assert len(sequence) == 4
    assert sequence.get_meta(0) == root / "00000_phase.bin"


# ------------------------------ beyond one modality ----------------------------- #


@pytest.mark.parametrize("kernel", (IdentityKernel(), MedianKernel(RADIUS)))
def test_accepts_a_uint8_source_without_a_header(tmp_path, kernel):
    # A hologram folder is uint8 and carries no acquisition header, so it only
    # fits because the source is bounded by what it yields rather than by the
    # float-with-header shape phase happens to have.
    root = _holo_folder(tmp_path / "holo")
    sequence = FrameSequence(HologramNpyFolder(root), kernel)
    item = sequence[2]
    assert item.dtype == torch.float32  # read as float32 whatever the source stored
    assert item.shape == (4, 5)
    assert sequence.get_meta(2) == root / "00002_holo.npy"


@pytest.mark.parametrize("kernel", (IdentityKernel(), MedianKernel(RADIUS)))
def test_accepts_a_file_list_that_is_not_a_folder(tmp_path, kernel):
    # `PhaseBinList` has no numbering or discovery, so it is not a Koala frame
    # folder at all. Bounding on the folder type would have excluded it.
    frames = _frames()
    root = _bin_folder(tmp_path / "Bin", frames)
    files = [root / f"{i:05d}_phase.bin" for i in (4, 0, 2)]  # arbitrary order
    sequence = FrameSequence(PhaseBinList(files), kernel)
    assert len(sequence) == 3
    assert sequence.get_meta(1) == root / "00000_phase.bin"
    if isinstance(kernel, IdentityKernel):
        assert torch.equal(sequence[1], torch.from_numpy(frames[0]))


class _InMemory(DataSequence[NDArray[np.float32], str]):
    """A source backed by no files at all, with metadata that is not a path."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames

    def __len__(self) -> int:
        return len(self._frames)

    def get_item(self, index: int) -> NDArray[np.float32]:
        return self._frames[index]

    def get_meta(self, index: int) -> str:
        return f"frame-{index}"


@pytest.mark.parametrize("kernel", (IdentityKernel(), MedianKernel(RADIUS)))
def test_accepts_a_source_that_is_not_file_backed(tmp_path, kernel):
    # The bound is what the source yields, not where it stores it, so a sequence
    # with no files and `str` metadata fits and that metadata passes through.
    frames = _frames()
    sequence = FrameSequence(_InMemory(frames), kernel)
    assert len(sequence) == len(frames)
    assert sequence.get_meta(2) == "frame-2"
    assert sequence[2].dtype == torch.float32


# ---------------------------------- the view ------------------------------------ #


def test_source_is_the_filtering_view_when_a_kernel_is_given(tmp_path):
    root = _bin_folder(tmp_path / "Bin", _frames())
    sequence = FrameSequence(PhaseBinFolder(root), MedianKernel(RADIUS))
    assert isinstance(sequence.source, FilteredSequence)


def test_the_view_is_a_filtering_one_even_without_filtering(tmp_path):
    # `IdentityKernel` is what removes the second view: every read goes through
    # one path, so no consumer branches on whether a run filters.
    frames = _frames()
    root = _bin_folder(tmp_path / "Bin", frames)
    sequence = FrameSequence(PhaseBinFolder(root), IdentityKernel())

    assert isinstance(sequence.source, FilteredSequence)
    assert torch.equal(sequence[2], torch.from_numpy(frames[2]))


@pytest.mark.parametrize("kernel", (IdentityKernel(), MedianKernel(RADIUS)))
def test_the_view_reads_the_frames_it_was_given(tmp_path, kernel):
    # Wrapped, not reopened: whichever view stands in front, it reads the source
    # instance handed in rather than opening the folder again.
    frames = _frames()
    root = _bin_folder(tmp_path / "Bin", frames)
    sequence = FrameSequence(PhaseBinFolder(root), kernel)
    assert len(sequence.source) == len(frames)
    assert torch.equal(sequence.source.get_item(2), sequence[2])


# ---------------------------------- device -------------------------------------- #


def test_device_defaults_to_cpu(tmp_path):
    root = _bin_folder(tmp_path / "Bin", _frames())
    sequence = FrameSequence(PhaseBinFolder(root), IdentityKernel())
    assert sequence.device == Device("cpu")
    assert sequence[0].device.type == "cpu"


def test_rejects_an_unsupported_device(tmp_path):
    # `meta` is a real torch device, so this reaches the project's own kind check
    # rather than tripping torch's parser first.
    root = _bin_folder(tmp_path / "Bin", _frames())
    with pytest.raises(ValueError, match=r"unsupported device 'meta'"):
        FrameSequence(PhaseBinFolder(root), IdentityKernel(), device="meta")


@requires_cuda
@pytest.mark.parametrize("kernel", (IdentityKernel(), MedianKernel(RADIUS)))
def test_frames_land_on_the_requested_device(tmp_path, kernel):
    root = _bin_folder(tmp_path / "Bin", _frames())
    sequence = FrameSequence(PhaseBinFolder(root), kernel, device="cuda")
    assert sequence[2].device.type == "cuda"


def test_device_is_the_views_rather_than_a_second_copy(tmp_path):
    # One owner, so reassigning it cannot leave the two disagreeing.
    root = _bin_folder(tmp_path / "Bin", _frames())
    sequence = FrameSequence(PhaseBinFolder(root), IdentityKernel())

    assert sequence.device is sequence.source.device


def test_reassigning_the_same_device_keeps_the_buffered_window(tmp_path):
    # Dropping the buffer costs a window of re-reads, so only a real move pays it.
    root = _bin_folder(tmp_path / "Bin", _frames())
    sequence = FrameSequence(PhaseBinFolder(root), MedianKernel(RADIUS))
    sequence[2]
    buffered = dict(sequence.source._buffer)  # noqa: SLF001
    assert buffered

    sequence.device = torch.device("cpu")

    assert sequence.source._buffer == buffered  # noqa: SLF001


@requires_cuda
def test_reassigning_the_device_drops_the_buffered_window(tmp_path):
    # Buffered frames sit on the old device, so keeping them would stack frames
    # from two devices together on the next window.
    root = _bin_folder(tmp_path / "Bin", _frames())
    sequence = FrameSequence(PhaseBinFolder(root), MedianKernel(RADIUS))
    sequence[2]
    assert sequence.source._buffer  # noqa: SLF001

    sequence.device = "cuda"

    assert not sequence.source._buffer  # noqa: SLF001
    assert sequence[2].device.type == "cuda"


# ----------------------------------- step ---------------------------------------- #


@pytest.mark.parametrize(
    ("step", "kept"),
    ((1, [0, 1, 2, 3, 4]), (2, [0, 2, 4]), (3, [0, 3]), (9, [0])),
)
def test_step_keeps_every_nth_frame_from_the_first(tmp_path, step, kept):
    frames = _frames()
    root = _bin_folder(tmp_path / "Bin", frames)
    sequence = FrameSequence(PhaseBinFolder(root), IdentityKernel(), step=step)

    assert len(sequence) == len(kept)
    for position, source_index in enumerate(kept):
        assert torch.equal(sequence[position], torch.from_numpy(frames[source_index]))


def test_step_renumbers_the_metadata_too(tmp_path):
    root = _bin_folder(tmp_path / "Bin", _frames())
    sequence = FrameSequence(PhaseBinFolder(root), IdentityKernel(), step=2)

    assert sequence.get_meta(1) == root / "00002_phase.bin"


def test_step_is_applied_before_filtering(tmp_path):
    # The whole point of the ordering: a dropped frame must not reach a kept
    # one. Filtering first would give frame 0 the median of frames 0 and 1.
    frames = [np.full((2, 2), float(i), dtype=np.float32) for i in range(5)]
    root = _bin_folder(tmp_path / "Bin", frames)
    sequence = FrameSequence(PhaseBinFolder(root), MedianKernel((0, 0, 1)), step=2)

    # Kept frames are 0, 2, 4; a temporal radius of 1 averages the middle two of
    # an even sample count, so the first is (0 + 2) / 2 and the last (2 + 4) / 2.
    assert [float(frame.flatten()[0]) for frame in sequence] == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("step", (0, -1))
def test_step_below_one_is_rejected(tmp_path, step):
    root = _bin_folder(tmp_path / "Bin", _frames())
    with pytest.raises(ValueError, match="invalid frame step"):
        FrameSequence(PhaseBinFolder(root), IdentityKernel(), step=step)


# --------------------------------- from_params ---------------------------------- #


def test_from_params_builds_the_kernel_it_describes(tmp_path):
    frames = _frames()
    root = _bin_folder(tmp_path / "Bin", frames)
    built = FrameSequence.from_params(PhaseBinFolder(root), MedianParams(RADIUS))
    direct = FrameSequence(PhaseBinFolder(root), MedianKernel(RADIUS))
    assert torch.equal(built[2], direct[2])


def test_from_params_reads_the_frames_as_stored_for_identity(tmp_path):
    # The filter is a config group a run may leave out, so None must reach the
    # unfiltered path rather than forcing the caller to branch.
    frames = _frames()
    root = _bin_folder(tmp_path / "Bin", frames)
    sequence = FrameSequence.from_params(PhaseBinFolder(root), IdentityParams())
    assert torch.equal(sequence[2], torch.from_numpy(frames[2]))


def test_from_params_forwards_the_device(tmp_path):
    root = _bin_folder(tmp_path / "Bin", _frames())
    sequence = FrameSequence.from_params(
        PhaseBinFolder(root), MedianParams(RADIUS), device="cpu"
    )
    assert sequence.device == Device("cpu")
    assert sequence[1].device.type == "cpu"


# -------------------------------- value_range ----------------------------------- #


def _ramp_folder(root, count: int = 5):
    # Frame i holds only the value i, so every expected range is exact.
    frames = [np.full((3, 4), float(i), dtype=np.float32) for i in range(count)]
    return _bin_folder(root, frames)


def test_value_range_of_one_frame(tmp_path):
    sequence = FrameSequence(
        PhaseBinFolder(_ramp_folder(tmp_path / "Bin")), IdentityKernel()
    )
    assert sequence.value_range(3) == (3.0, 3.0)


def test_value_range_of_a_negative_index(tmp_path):
    sequence = FrameSequence(
        PhaseBinFolder(_ramp_folder(tmp_path / "Bin")), IdentityKernel()
    )
    assert sequence.value_range(-1) == (4.0, 4.0)


@pytest.mark.parametrize(
    ("selection", "expected"),
    (
        (slice(1, 4), (1.0, 3.0)),
        (slice(None, None, 2), (0.0, 4.0)),
        (slice(-2, None), (3.0, 4.0)),
        (range(1, 3), (1.0, 2.0)),
        ([4, 0], (0.0, 4.0)),
        ((2,), (2.0, 2.0)),
    ),
)
def test_value_range_of_a_selection(tmp_path, selection, expected):
    sequence = FrameSequence(
        PhaseBinFolder(_ramp_folder(tmp_path / "Bin")), IdentityKernel()
    )
    assert sequence.value_range(selection) == expected


def test_value_range_of_the_whole_sequence(tmp_path):
    sequence = FrameSequence(
        PhaseBinFolder(_ramp_folder(tmp_path / "Bin")), IdentityKernel()
    )
    assert sequence.value_range() == (0.0, 4.0)
    assert sequence.value_range() == sequence.value_range(slice(None))


def test_value_range_ignores_non_finite_values(tmp_path):
    # A NaN must be dropped, not propagated: `min` over a tensor holding one
    # returns NaN, which would poison every range that touched the frame.
    frames = [np.full((3, 4), float(i), dtype=np.float32) for i in range(3)]
    frames[1][0, 0] = np.nan
    root = _bin_folder(tmp_path / "Bin", frames)
    sequence = FrameSequence(PhaseBinFolder(root), IdentityKernel())
    assert sequence.value_range(1) == (1.0, 1.0)
    assert sequence.value_range() == (0.0, 2.0)


def test_value_range_rejects_a_frame_with_no_finite_value(tmp_path):
    frames = [np.full((3, 4), np.nan, dtype=np.float32)]
    root = _bin_folder(tmp_path / "Bin", frames)
    sequence = FrameSequence(PhaseBinFolder(root), IdentityKernel())
    with pytest.raises(ValueError, match="the selection holds no finite value"):
        sequence.value_range(0)


def test_value_range_rejects_a_frame_with_no_pixels():
    # A zero-sized frame has no finite value either, and it has to be caught
    # before the fused pass: `aminmax` raises on an empty tensor rather than
    # reporting no bounds.
    sequence = FrameSequence(
        _InMemory([np.zeros((0, 4), np.float32)]), IdentityKernel()
    )

    with pytest.raises(ValueError, match="the selection holds no finite value"):
        sequence.value_range(0)


def test_value_range_rejects_an_empty_selection(tmp_path):
    sequence = FrameSequence(
        PhaseBinFolder(_ramp_folder(tmp_path / "Bin")), IdentityKernel()
    )
    with pytest.raises(ValueError, match="undefined for an empty selection"):
        sequence.value_range([])
    with pytest.raises(ValueError, match="undefined for an empty selection"):
        sequence.value_range(slice(2, 2))


def test_value_range_rejects_an_out_of_range_index(tmp_path):
    sequence = FrameSequence(
        PhaseBinFolder(_ramp_folder(tmp_path / "Bin")), IdentityKernel()
    )
    with pytest.raises(IndexError):
        sequence.value_range(99)


def test_value_range_caches_only_the_whole_sequence(monkeypatch, tmp_path):
    # The global range reads every frame, so it is computed once; a subset is not
    # cached, which a read counter shows and equal return values cannot.
    sequence = FrameSequence(
        PhaseBinFolder(_ramp_folder(tmp_path / "Bin")), IdentityKernel()
    )
    reads = 0
    real_get_item = type(sequence).get_item

    def counting(self, index):
        nonlocal reads
        reads += 1
        return real_get_item(self, index)

    monkeypatch.setattr(type(sequence), "get_item", counting)

    sequence.value_range()
    after_first = reads
    sequence.value_range()
    assert reads == after_first == 5  # cached: no further reads

    sequence.value_range(slice(0, 2))
    sequence.value_range(slice(0, 2))
    assert reads == after_first + 4  # recomputed both times


def test_value_range_reports_the_filtered_values(tmp_path):
    # Filtering changes the values, so the range must come from what this sequence
    # yields rather than from the source it wraps. The spike is a single pixel:
    # a whole frame of them would outvote its own neighbourhood and survive.
    frames = [np.full((3, 4), float(i), dtype=np.float32) for i in range(5)]
    frames[2][1, 1] = 100.0
    root = _bin_folder(tmp_path / "Bin", frames)

    plain = FrameSequence(PhaseBinFolder(root), IdentityKernel())
    filtered = FrameSequence(PhaseBinFolder(root), MedianKernel(RADIUS))
    assert plain.value_range() == (0.0, 100.0)
    assert filtered.value_range() == (0.0, 4.0)  # the median deletes the spike
