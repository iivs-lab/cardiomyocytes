from __future__ import annotations

from typing import override

import numpy as np
import pytest
import torch
from kaparoo.data.sequences import DataSequence
from numpy.typing import NDArray

from iivs_cardio.data.preprocessing.filtering import (
    FilteredSequence,
    MedianKernel,
    MedianParams,
)


class _Frames(DataSequence[NDArray[np.float32], int]):
    """An in-memory phase sequence that counts how often a frame is read."""

    def __init__(self, frames: NDArray[np.float32]) -> None:
        self._frames = frames
        self.reads = 0

    @override
    def __len__(self) -> int:
        return len(self._frames)

    @override
    def get_item(self, index: int) -> NDArray[np.float32]:
        self.reads += 1
        return self._frames[index]

    @override
    def get_meta(self, index: int) -> int:
        return index * 10  # a stand-in for a timestamp or source path


def _frames(count: int, height: int = 4, width: int = 5) -> NDArray[np.float32]:
    rng = np.random.default_rng(0)
    return rng.random((count, height, width), dtype=np.float32)


# --------------------------- the filtered sequence ------------------------ #


def test_the_filtered_view_is_as_long_as_its_source():
    # Truncated ends, not dropped ones: every source frame yields an output.
    source = _Frames(_frames(7))
    filtered = FilteredSequence(source, MedianKernel((1, 1, 2)))

    assert len(filtered) == len(source) == 7
    assert filtered.source is source  # reachable, for provenance in the cache


def test_indexed_access_matches_a_window_over_the_whole_sequence():
    frames = _frames(9)
    kernel = MedianKernel((1, 1, 2))
    filtered = FilteredSequence(_Frames(frames), kernel)
    whole = torch.from_numpy(frames)

    for index in range(len(frames)):
        assert torch.equal(filtered[index], kernel.apply(whole, index))


def test_out_of_order_access_returns_what_a_forward_pass_returned():
    # The property a delay line cannot offer, and the reason this owns its
    # source: frame `i` never depends on which window asked for it.
    frames = _frames(9)
    kernel = MedianKernel((1, 1, 2))

    forward = list(FilteredSequence(_Frames(frames), kernel))
    shuffled = FilteredSequence(_Frames(frames), kernel)

    for index in (5, 0, 8, 3, 8, 1):
        assert torch.equal(shuffled[index], forward[index])


def test_a_forward_pass_reads_each_source_frame_once():
    # What the buffer is for. Without it every item re-reads its whole window,
    # costing 2*rz+1 reads per frame instead of one.
    source = _Frames(_frames(9))
    for _ in FilteredSequence(source, MedianKernel((1, 1, 2))):
        pass

    assert source.reads == 9


def test_negative_indices_count_from_the_end():
    # Wrapping and bounds come from `DataSequence._normalize_index`; what this
    # pins is that `get_item` and `get_meta` go through it. Skip the call and a
    # negative index reaches the window arithmetic, where it silently yields an
    # empty range rather than the frame counted from the end.
    filtered = FilteredSequence(_Frames(_frames(6)), MedianKernel((1, 1, 1)))

    assert torch.equal(filtered[-1], filtered[5])
    assert filtered.get_meta(-1) == filtered.get_meta(5)

    for outside in (6, -7):
        with pytest.raises(IndexError, match="out of range"):
            filtered[outside]


def test_metadata_passes_through_untouched():
    # Filtering changes pixels, not which acquisition a frame came from.
    filtered = FilteredSequence(_Frames(_frames(4)), MedianKernel((1, 1, 1)))

    assert filtered.get_meta(2) == 20
    assert filtered.get_pair(2)[1] == 20


def test_frames_come_back_as_float32_tensors():
    filtered = FilteredSequence(_Frames(_frames(4)), MedianKernel((1, 1, 1)))
    frame = filtered[0]

    assert isinstance(frame, torch.Tensor)
    assert frame.dtype == torch.float32
    assert frame.shape == (4, 5)


# ------------------------------ from_params ------------------------------- #


def test_from_params_builds_the_kernel_it_describes():
    frames = _frames(6)
    params = MedianParams((1, 1, 1), shape="cuboid")

    built = FilteredSequence.from_params(_Frames(frames), params)
    direct = FilteredSequence(_Frames(frames), MedianKernel((1, 1, 1), shape="cuboid"))

    assert isinstance(built.kernel, MedianKernel)
    assert built.kernel.shape == "cuboid"  # not the default, so it came from params
    for index in range(len(frames)):
        assert torch.equal(built[index], direct[index])


def test_from_params_passes_a_short_radius_through_to_the_kernel():
    # The record holds the radius verbatim, so this is the only step that
    # expands it -- and the one a config-driven caller depends on.
    built = FilteredSequence.from_params(_Frames(_frames(4)), MedianParams((1, 0)))

    assert built.kernel.radius == (1, 1, 0)
    assert built.kernel.temporal_radius == 0
