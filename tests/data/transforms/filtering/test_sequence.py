from __future__ import annotations

from typing import override

import numpy as np
import pytest
import torch
from kaparoo.data.sequences import DataSequence
from numpy.typing import NDArray

from iivs_cardio.common import Device
from iivs_cardio.data.transforms.filtering import (
    FilteredSequence,
    IdentityConfig,
    IdentityKernel,
    MedianConfig,
    MedianKernel,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="no CUDA-capable GPU detected",
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
    assert filtered.origin is source  # reachable, for provenance in the cache


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


def test_a_non_float32_source_is_read_as_float32_and_filtered_the_same():
    # A uint8 phase-image source is as valid as a float32 one; the kernel reduces
    # float32 regardless, and an even-count median lands on a half no integer
    # dtype could hold -- so casting must happen before the reduction, not after.
    frames = (_frames(4) * 255).astype(np.uint8)
    filtered = FilteredSequence(_Frames(frames), MedianKernel((1, 1, 1)))
    reference = MedianKernel((1, 1, 1)).apply(torch.from_numpy(frames).float(), 0)

    assert filtered[0].dtype == torch.float32
    assert torch.equal(filtered[0], reference)


class _Uint8Frames(DataSequence[NDArray[np.uint8], None]):
    """A source whose declared dtype is uint8, to fix the type parameter."""

    @override
    def __len__(self) -> int:
        return 4

    @override
    def get_item(self, index: int) -> NDArray[np.uint8]:
        return np.full((3, 3), index, dtype=np.uint8)

    @override
    def get_meta(self, index: int) -> None:
        return None


def test_the_source_dtype_is_carried_on_the_type_parameter():
    # `origin` reflects the declared dtype rather than a hard-coded float32, so a
    # uint8 source is describable without a lie in the signature.
    filtered: FilteredSequence[_Uint8Frames, None, np.uint8] = FilteredSequence(
        _Uint8Frames(), MedianKernel(0)
    )

    assert filtered.origin.get_item(2).dtype == np.uint8
    assert filtered[2].dtype == torch.float32


# ------------------------------ from_config ------------------------------- #


def test_from_params_builds_the_kernel_it_describes():
    frames = _frames(6)
    config = MedianConfig((1, 1, 1), shape="cuboid")

    built = FilteredSequence.from_config(_Frames(frames), config)
    direct = FilteredSequence(_Frames(frames), MedianKernel((1, 1, 1), shape="cuboid"))

    assert isinstance(built.kernel, MedianKernel)
    assert built.kernel.shape == "cuboid"  # not the default, so it came from config
    for index in range(len(frames)):
        assert torch.equal(built[index], direct[index])


def test_from_params_passes_a_short_radius_through_to_the_kernel():
    # The record holds the radius verbatim, so this is the only step that
    # expands it -- and the one a config-driven caller depends on.
    built = FilteredSequence.from_config(_Frames(_frames(4)), MedianConfig((1, 0)))

    assert built.kernel.radius == (1, 1, 0)
    assert built.kernel.temporal_radius == 0


# ------------------------------- the stride ------------------------------- #


@pytest.mark.parametrize(("step", "kept"), ((1, 6), (2, 3), (3, 2), (7, 1)))
def test_step_keeps_every_nth_frame_from_the_first(step, kept):
    source = _Frames(_frames(6))

    sequence = FilteredSequence(source, IdentityKernel(), step=step)

    assert len(sequence) == kept
    for index in range(kept):
        assert torch.equal(sequence[index], torch.from_numpy(source[index * step]))


def test_step_leaves_the_metadata_naming_the_source_frame():
    # The opposite of what this used to be called: an index counts kept frames,
    # but the metadata keeps naming the frame the source knows, so a message
    # about one points at the file to go and look at. That is also why a value
    # range filed under it does not line up with a cache numbered from zero --
    # both are right, and position is what joins them.
    sequence = FilteredSequence(_Frames(_frames(6)), IdentityKernel(), step=2)

    # View 0, 1, 2 are source 0, 2, 4 -- whose metadata is 0, 20, 40.
    assert [sequence.get_meta(i) for i in range(len(sequence))] == [0, 20, 40]
    assert [sequence.get_meta(i) for i in range(len(sequence))] != [0, 10, 20]


def test_step_is_applied_before_filtering():
    # Filtering first would fold the dropped frames into the kept ones, so a
    # strided read would not measure the frame rate it claims to.
    source = _frames(6)
    kernel = MedianKernel((0, 0, 1))  # temporal only, so the order is visible

    strided = FilteredSequence(_Frames(source), kernel, step=2)
    presliced = FilteredSequence(_Frames(source[::2]), kernel)

    assert torch.equal(strided[1], presliced[1])


@pytest.mark.parametrize("step", (0, -1))
def test_step_below_one_is_rejected(step):
    with pytest.raises(ValueError, match=r"invalid frame step"):
        FilteredSequence(_Frames(_frames(3)), IdentityKernel(), step=step)


def test_from_params_forwards_the_step():
    sequence = FilteredSequence.from_config(
        _Frames(_frames(6)), IdentityConfig(), step=3
    )

    assert len(sequence) == 2


# ------------------------------- the origin ------------------------------- #


def test_origin_is_the_sequence_it_was_opened_over():
    source = _Frames(_frames(6))

    sequence = FilteredSequence(source, IdentityKernel(), step=2)

    assert sequence.origin is source


# -------------------------------- the device ------------------------------ #


def test_device_defaults_to_cpu():
    sequence = FilteredSequence(_Frames(_frames(3)), IdentityKernel())

    assert sequence.device == Device("cpu")
    assert sequence[0].device.type == "cpu"


def test_an_unsupported_device_is_rejected():
    # `meta` is a real torch device, so this reaches the project's own kind check
    # rather than tripping the spec parser first.
    with pytest.raises(ValueError, match=r"unsupported device 'meta'"):
        FilteredSequence(_Frames(_frames(3)), IdentityKernel(), device="meta")


@requires_cuda
@pytest.mark.parametrize("kernel", (IdentityKernel(), MedianKernel((1, 1, 1))))
def test_frames_land_on_the_requested_device(kernel):
    sequence = FilteredSequence(_Frames(_frames(4)), kernel, device="cuda")

    assert sequence[2].device.type == "cuda"


def test_reassigning_the_same_device_keeps_the_buffered_window():
    source = _Frames(_frames(4))
    sequence = FilteredSequence(source, MedianKernel((0, 0, 1)))
    _ = sequence[1]
    reads = source.reads

    sequence.device = "cpu"
    _ = sequence[1]

    assert source.reads == reads


class _Strict(_Frames):
    """A source that refuses the empty read a fully buffered window would make."""

    @override
    def get_items(self, indices):
        if not list(indices):
            msg = "asked for no frames at all"
            raise AssertionError(msg)

        return super().get_items(indices)


def test_a_window_already_held_asks_the_source_for_nothing():
    # The last window of a walk is entirely buffered, so the read that follows
    # it has nothing to fetch -- and the source was still called, with an empty
    # list. Harmless for a folder, and not something a source should have to
    # expect.
    source = _Strict(_frames(4))
    sequence = FilteredSequence(source, MedianKernel((0, 0, 1)))

    for index in range(len(sequence)):
        _ = sequence[index]

    assert source.reads == len(sequence)


def test_releasing_drops_the_buffered_window():
    # The window is whatever the last read needed, and nothing else lets go of
    # it -- a driver holding every sequence of a dataset for the whole run holds
    # one window per sequence with it, on the device, and again in each worker.
    source = _Frames(_frames(4))
    sequence = FilteredSequence(source, MedianKernel((0, 0, 1)))
    _ = sequence[1]
    reads = source.reads

    sequence.release()
    _ = sequence[1]

    assert source.reads > reads


def test_a_released_view_still_answers():
    # Letting go is not closing: the frames come back off the source.
    source = _Frames(_frames(4))
    sequence = FilteredSequence(source, MedianKernel((0, 0, 1)))
    before = sequence[1]

    sequence.release()

    assert torch.equal(sequence[1], before)


def test_changing_the_device_drops_the_buffered_window():
    # Buffered frames sit on the old device, so they cannot be reused. Routed back
    # to the cpu so the re-read runs here; `Device.resolve` never touches a driver.
    source = _Frames(_frames(4))
    sequence = FilteredSequence(source, MedianKernel((0, 0, 1)))
    _ = sequence[1]
    reads = source.reads

    sequence.device = "cuda:0"
    sequence.device = "cpu"
    _ = sequence[1]

    assert source.reads > reads


# ------------------------------ non-finite input -------------------------- #


@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf))
def test_a_non_finite_source_frame_is_refused_on_the_way_in(bad):
    # Nothing downstream has an answer for one: it survives every arithmetic
    # step to come, and the formats this project reads store one happily.
    frames = _frames(5)
    frames[2, 1, 3] = bad

    filtered = FilteredSequence(_Frames(frames), IdentityKernel())

    with pytest.raises(ValueError, match=r"non-finite value in 20"):
        filtered.get_item(2)


def test_the_refusal_names_the_frame_by_the_source_s_own_metadata():
    # An index alone does not locate the file to repair, and the stride between
    # the view and its source means the view's index is not the source's either.
    frames = _frames(6)
    frames[4] = np.nan

    filtered = FilteredSequence(_Frames(frames), IdentityKernel(), step=2)

    # view 2 is source 4, whose metadata is 40 -- neither of the other numbers.
    with pytest.raises(ValueError, match=r"non-finite value in 40"):
        filtered.get_item(2)


def test_a_neighbour_is_refused_even_when_the_target_itself_is_clean():
    # A temporal kernel reads past its target, so a frame nobody asked for still
    # reaches the arithmetic -- and one NaN there spreads across the output.
    frames = _frames(5)
    frames[3] = np.nan

    filtered = FilteredSequence(_Frames(frames), MedianKernel((0, 0, 1)))

    with pytest.raises(ValueError, match=r"non-finite value in 30"):
        filtered.get_item(2)


def test_a_clean_sequence_is_untouched_by_the_check():
    frames = _frames(4)
    filtered = FilteredSequence(_Frames(frames), IdentityKernel())

    assert torch.equal(filtered.get_item(1), torch.from_numpy(frames[1]))


def test_a_value_only_a_wider_source_could_hold_is_refused():
    # The check has to follow the cast, not precede it: 1e39 is an ordinary
    # float64 and infinite as a float32, so a check on the source array passes
    # and the tensor the kernel reduces is infinite anyway.
    frames = _frames(3).astype(np.float64)
    frames[1, 0, 0] = 1e39

    filtered = FilteredSequence(_Frames(frames), IdentityKernel())

    assert np.isfinite(frames).all()  # the source itself is clean
    with pytest.raises(ValueError, match=r"non-finite value in 10"):
        filtered.get_item(1)


# ----------------------------- what memory is whose ----------------------- #


def test_the_buffer_does_not_view_the_source_s_own_storage():
    # `_Frames` hands back a slice of the array it keeps, as a memmap-backed
    # source does -- and a float32 one needs no cast, so without the copy the
    # buffer, the filter, and the caller all read the source's own memory.
    frames = _frames(4)
    sequence = FilteredSequence(_Frames(frames), MedianKernel((0, 0, 1)))
    before = sequence[1]

    frames[1] = 0.0  # the source changes under a window already read

    assert torch.equal(sequence[1], before)


def test_the_frame_handed_back_is_not_a_view_of_the_buffer():
    # The identity kernel returns its target frame, which is the buffered one;
    # a caller normalizing in place would rewrite the window behind it, and the
    # next read of a neighbour would see the normalized values.
    frames = _frames(4)
    sequence = FilteredSequence(_Frames(frames), IdentityKernel())

    sequence[1].fill_(0.0)

    assert torch.equal(sequence[1], torch.from_numpy(frames[1]))
