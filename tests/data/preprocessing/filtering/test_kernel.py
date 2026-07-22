from __future__ import annotations

from dataclasses import FrozenInstanceError
from statistics import median
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from iivs_cardio.data.preprocessing.filtering import MedianKernel, MedianParams

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _frames(count: int, height: int = 4, width: int = 5) -> NDArray[np.float32]:
    rng = np.random.default_rng(0)
    return rng.random((count, height, width), dtype=np.float32)


def _in_bounds(shape: tuple[int, int, int], z: int, y: int, x: int) -> bool:
    depth, height, width = shape
    return 0 <= z < depth and 0 <= y < height and 0 <= x < width


def _brute_median(
    frames: NDArray[np.float32], kernel: MedianKernel, index: int
) -> torch.Tensor:
    """Median per pixel from an explicit neighbour list, in plain Python.

    Independent of the implementation: `statistics.median` averages the middle
    two of an even count by definition, and the bounds check spells out
    truncation one neighbour at a time.
    """
    _, height, width = frames.shape
    out = torch.zeros(height, width)
    for y in range(height):
        for x in range(width):
            samples = [
                float(frames[index + dz, y + dy, x + dx])
                for dx, dy, dz in kernel.offsets
                if _in_bounds(frames.shape, index + dz, y + dy, x + dx)
            ]
            out[y, x] = median(samples)
    return out


# ------------------------------ radius forms ------------------------------ #


def test_a_scalar_radius_applies_to_every_axis():
    assert MedianKernel(2).radius == (2, 2, 2)


def test_a_pair_sets_both_spatial_axes_and_the_temporal_one_apart():
    # The values differ so a swapped or copied element cannot pass: `(2, 5)`
    # means rx = ry = 2 and rz = 5, never (2, 5, 5) or (2, 2, 2).
    kernel = MedianKernel((2, 5))

    assert kernel.radius == (2, 2, 5)
    assert kernel.spatial_radius == (2, 2)
    assert kernel.temporal_radius == 5


def test_an_explicit_triple_is_kept_axis_by_axis():
    kernel = MedianKernel((3, 2, 1))

    assert kernel.radius == (3, 2, 1)
    assert kernel.spatial_radius == (3, 2)
    assert kernel.temporal_radius == 1


def test_the_three_forms_agree_where_they_describe_one_kernel():
    # Not just equal radii: the offsets they sample must match too, so a form
    # that normalized correctly but was read elsewhere would still fail.
    window = torch.from_numpy(_frames(5))

    scalar = MedianKernel(1)
    pair = MedianKernel((1, 1))
    triple = MedianKernel((1, 1, 1))

    assert scalar.offsets == pair.offsets == triple.offsets
    assert torch.equal(scalar.apply(window, 2), triple.apply(window, 2))


def test_a_radius_of_no_recognised_form_is_rejected():
    with pytest.raises(ValueError, match=r"expected int r"):
        MedianKernel((1, 1, 1, 1))  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "radius",
    (
        pytest.param(2.0, id="scalar-float"),
        pytest.param((2.0, 5), id="float-in-a-pair"),
        pytest.param((2, 2, 5.0), id="float-in-a-triple"),
        pytest.param(("2", "5"), id="digits-as-strings"),
        pytest.param((None, 1), id="none"),
    ),
)
def test_a_radius_holding_a_non_int_is_rejected(radius):
    # A config file reaches here having been through YAML or JSON, where `2.0`
    # and a quoted `"2"` are easy to write. Unchecked, a float gets as far as
    # `range()` in `_build_offsets` and raises there, naming neither the radius
    # nor the caller's mistake.
    with pytest.raises(ValueError, match="invalid radius"):
        MedianKernel(radius)  # ty: ignore[invalid-argument-type]


def test_a_sequence_is_accepted_however_a_config_parser_spelled_it():
    # YAML and JSON have no tuple, so a radius arrives as a list.
    assert MedianKernel([2, 5]).radius == (2, 2, 5)
    assert MedianKernel([3, 2, 1]).radius == (3, 2, 1)


def test_a_negative_scalar_is_caught_after_expansion():
    # The check runs on the normalized triple, so no form can slip past it.
    with pytest.raises(ValueError, match="negative radius"):
        MedianKernel(-1)


# ------------------------------- shared base ------------------------------ #


def test_temporal_radius_reports_the_frames_a_window_needs_either_side():
    # The z radius alone, not the largest of the three: the spatial radii are
    # `apply`'s business, and a window is sized only in frames.
    assert MedianKernel((2, 1, 3)).temporal_radius == 3
    assert MedianKernel((5, 4, 1)).temporal_radius == 1


def test_apply_rejects_a_target_outside_the_window():
    with pytest.raises(ValueError, match="not an index into"):
        MedianKernel((1, 1, 1)).apply(torch.zeros(3, 4, 4), 3)


def test_a_target_need_not_be_the_middle_of_its_window():
    # What the old name `center` claimed and a truncated window disproves: at a
    # sequence end the frame being filtered sits at the edge of its own window.
    window = torch.zeros(3, 4, 4)
    window[0] = 5.0

    first = MedianKernel((0, 0, 1)).apply(window, 0)
    middle = MedianKernel((0, 0, 1)).apply(window, 1)

    # Frame 0 sees only itself and frame 1 -> median of (5, 0) averages to 2.5.
    assert torch.equal(first, torch.full((4, 4), 2.5))
    # Frame 1 sees all three -> median of (5, 0, 0) is 0.
    assert torch.equal(middle, torch.zeros(4, 4))


def test_apply_rejects_a_window_that_is_not_float32():
    with pytest.raises(Exception, match=r"\[3,4,4\]"):
        MedianKernel((1, 1, 0)).apply(torch.zeros(3, 4, 4, dtype=torch.float64), 0)


# --------------------------------- median --------------------------------- #


def test_ellipsoid_samples_fewer_offsets_than_the_cuboid_box():
    # The whole point of the ellipsoid: 33 samples per pixel instead of 125.
    assert len(MedianKernel((2, 2, 2)).offsets) == 33
    assert len(MedianKernel((2, 2, 2), shape="cuboid").offsets) == 125


def test_cuboid_takes_the_whole_box_in_scan_order():
    # `cuboid` applies no predicate at all. Spelled out with explicit loops
    # rather than the tool the source uses, so the two cannot agree by sharing
    # a mistake.
    expected = [
        (dx, dy, dz)
        for dx in range(-2, 3)
        for dy in range(-1, 2)
        for dz in range(-3, 4)
    ]

    assert len(expected) == 5 * 3 * 7
    assert MedianKernel((2, 1, 3), shape="cuboid").offsets == tuple(expected)


def test_the_ellipsoid_is_a_strict_subset_of_its_box_in_the_same_order():
    # Selecting offsets must not reorder them: `offsets` documents scan order.
    box = MedianKernel((3, 2, 4), shape="cuboid").offsets
    ellipsoid = MedianKernel((3, 2, 4)).offsets
    kept = set(ellipsoid)

    assert kept < set(box)
    assert list(ellipsoid) == [offset for offset in box if offset in kept]


def test_a_zero_radius_disables_that_axis():
    # Legacy could not express this -- it divides by each radius unguarded.
    spatial = MedianKernel((1, 1, 0))

    assert {dz for _, _, dz in spatial.offsets} == {0}
    assert len(spatial.offsets) == 5  # the cross: centre plus one step each way
    assert spatial.temporal_radius == 0


def test_a_zero_radius_everywhere_is_the_identity():
    kernel = MedianKernel((0, 0, 0))
    window = torch.rand(1, 4, 4)

    assert kernel.offsets == ((0, 0, 0),)
    assert torch.equal(kernel.apply(window, 0), window[0])


def test_median_rejects_a_negative_radius():
    with pytest.raises(ValueError, match="negative radius"):
        MedianKernel((1, -1, 1))


def test_border_pixels_drop_offsets_and_average_the_middle_two():
    # A column of 1, 2, 10 with a vertical radius of 1. The middle pixel sees
    # all three -> 2. Each end loses one offset to the border, leaving an even
    # two to average: (1 + 2) / 2 and (2 + 10) / 2. `torch.median` would return
    # the lower of each pair instead, giving [1, 2, 2] -- which is why it cannot
    # be used directly, and what this test exists to catch.
    window = torch.tensor([[[1.0], [2.0], [10.0]]])  # (T=1, H=3, W=1)

    out = MedianKernel((0, 1, 0)).apply(window, 0)

    assert out.squeeze().tolist() == pytest.approx([1.5, 2.0, 6.0])


def test_a_spike_is_replaced_by_its_neighbourhood():
    # What the median is for: one hot pixel is outvoted by the four around it.
    window = torch.zeros(1, 5, 5)
    window[0, 2, 2] = 100.0

    assert torch.equal(MedianKernel((1, 1, 0)).apply(window, 0), torch.zeros(5, 5))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize(
    ("radius", "shape"),
    (
        pytest.param((1, 1, 1), "ellipsoid", id="7-samples-below-the-range"),
        pytest.param((2, 2, 2), "ellipsoid", id="33-samples-inside-the-range"),
        pytest.param((2, 2, 2), "cuboid", id="125-samples-above-the-range"),
    ),
)
def test_cuda_returns_what_the_cpu_returns(radius, shape):
    # The device picks how much of the order to materialize, purely for speed,
    # so the answer must not move with it. The radii straddle the sample count
    # where that choice flips.
    window = torch.rand(5, 32, 32, generator=torch.Generator().manual_seed(0))
    kernel = MedianKernel(radius, shape=shape)

    on_cpu = kernel.apply(window, 2)
    on_cuda = kernel.apply(window.cuda(), 2)

    assert torch.equal(on_cpu, on_cuda.cpu())


def test_an_odd_count_returns_the_sample_itself_bit_for_bit():
    # An odd count averages each sample with itself, so this pins that the
    # averaging is exact and not merely close -- `torch.equal`, not `allclose`,
    # over values that have no exact halves.
    awkward = torch.rand(1, 64, 64, generator=torch.Generator().manual_seed(0))

    # One sample per pixel: the centre, alone.
    assert torch.equal(MedianKernel(0).apply(awkward, 0), awkward[0])

    # Three per pixel, spanning eight orders of magnitude so a lost bit shows.
    # The centre row sees all three, and their median by value is the first
    # row's `0.1` -- sorted they run 1e-8, 0.1, 12345.679.
    column = torch.tensor([[[0.1], [12345.679], [1e-8]]])  # (T=1, H=3, W=1)
    filtered = MedianKernel((0, 1, 0)).apply(column, 0)

    assert torch.equal(filtered[1], column[0, 0])


def test_median_matches_a_brute_force_pass_over_explicit_neighbours():
    # Everything at once -- offset selection, truncation at all six borders, and
    # even-count averaging -- against a computation sharing no code with it.
    frames = _frames(5)
    kernel = MedianKernel((2, 1, 1))
    window = torch.from_numpy(frames)

    for index in range(len(frames)):
        assert torch.allclose(
            kernel.apply(window, index), _brute_median(frames, kernel, index)
        )


# --------------------------------- params --------------------------------- #


def test_params_hold_what_they_were_given():
    # A plain record: it neither expands the radius nor checks it, so a config
    # round-trips through it unchanged and the kernel stays the one place that
    # interprets a radius. Whatever builds a kernel from these does the rest.
    assert MedianParams(2).radius == 2
    assert MedianParams((2, 5)).radius == (2, 5)
    assert MedianParams((1, -1, 1)).radius == (1, -1, 1)  # invalid, and still held

    assert MedianParams((1, 1, 1)).shape == "ellipsoid"  # the only default


def test_params_are_frozen_records():
    # They are what the cache sidecar records; a mutated one would describe a
    # cache that had been built with something else.
    params = MedianParams((1, 1, 1))

    assert params.shape == "ellipsoid"
    with pytest.raises(FrozenInstanceError):
        params.radius = (2, 2, 2)  # ty: ignore[invalid-assignment]
