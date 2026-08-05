from __future__ import annotations

from dataclasses import FrozenInstanceError
from math import exp
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from iivs_cardio.data.transforms.filtering import (
    GaussianConfig,
    GaussianKernel,
    KernelConfig,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _frames(count: int, height: int = 4, width: int = 5) -> NDArray[np.float32]:
    rng = np.random.default_rng(0)
    return rng.random((count, height, width), dtype=np.float32)


def _in_bounds(shape: tuple[int, int, int], z: int, y: int, x: int) -> bool:
    depth, height, width = shape
    return 0 <= z < depth and 0 <= y < height and 0 <= x < width


def _brute_gaussian(
    frames: NDArray[np.float32], kernel: GaussianKernel, index: int
) -> torch.Tensor:
    """Weighted mean per pixel over the surviving neighbours, in plain Python.

    Builds the 3D weight from the separable definition directly, not from the
    kernel's own 1D tensors, and divides by the weight that landed in bounds --
    the definition the separable implementation must match exactly.
    """
    _, height, width = frames.shape
    rx, ry, rz = kernel.radius
    sx, sy, sz = kernel.sigma

    def weight(delta: int, sigma: float) -> float:
        return 1.0 if sigma == 0 else exp(-(delta**2) / (2.0 * sigma**2))

    out = torch.zeros(height, width)
    for y in range(height):
        for x in range(width):
            total = 0.0
            mass = 0.0
            for dz in range(-rz, rz + 1):
                for dy in range(-ry, ry + 1):
                    for dx in range(-rx, rx + 1):
                        if not _in_bounds(frames.shape, index + dz, y + dy, x + dx):
                            continue
                        w = weight(dx, sx) * weight(dy, sy) * weight(dz, sz)
                        total += w * float(frames[index + dz, y + dy, x + dx])
                        mass += w
            out[y, x] = total / mass
    return out


# ------------------------------- sigma forms ------------------------------ #


def test_sigma_and_truncate_set_the_radius():
    # scipy's rule, `int(truncate * sigma + 0.5)`, so a radius cached from the
    # legacy tooling reproduces here.
    assert GaussianKernel((1.0, 2.0, 0.5), truncate=4.0).radius == (4, 8, 2)
    assert GaussianKernel((1.0, 1.0, 1.0), truncate=1.0).radius == (1, 1, 1)


def test_a_scalar_sigma_applies_to_every_axis():
    assert GaussianKernel(1.0, truncate=2.0).sigma == (1.0, 1.0, 1.0)


def test_a_pair_sets_both_spatial_axes_and_the_temporal_one_apart():
    kernel = GaussianKernel((2.0, 0.5), truncate=2.0)

    assert kernel.sigma == (2.0, 2.0, 0.5)
    assert kernel.spatial_radius == (4, 4)  # int(2*2 + 0.5)
    assert kernel.temporal_radius == 1  # int(2*0.5 + 0.5)


def test_an_int_sigma_is_coerced_to_float():
    # A config or a bare literal writes `1`, not `1.0`; a radius must be int, but
    # a sigma is a width and takes either.
    kernel = GaussianKernel((1, 1, 2))

    assert kernel.sigma == (1.0, 1.0, 2.0)
    assert all(isinstance(s, float) for s in kernel.sigma)


def test_a_sequence_is_accepted_however_a_config_parser_spelled_it():
    assert GaussianKernel([1.0, 2.0], truncate=2.0).sigma == (1.0, 1.0, 2.0)


def test_a_sigma_of_no_recognised_form_is_rejected():
    with pytest.raises(ValueError, match="invalid sigma"):
        GaussianKernel((1.0, 1.0, 1.0, 1.0))  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "sigma",
    (
        pytest.param(("1", "1"), id="digits-as-strings"),
        pytest.param((None, 1.0), id="none"),
    ),
)
def test_a_sigma_holding_a_non_number_is_rejected(sigma):
    with pytest.raises(ValueError, match="invalid sigma"):
        GaussianKernel(sigma)  # ty: ignore[invalid-argument-type]


def test_gaussian_rejects_a_negative_sigma():
    with pytest.raises(ValueError, match="negative sigma"):
        GaussianKernel((1.0, -1.0, 1.0))


def test_gaussian_rejects_a_non_positive_truncate():
    with pytest.raises(ValueError, match="truncate must be positive"):
        GaussianKernel((1.0, 1.0, 1.0), truncate=0.0)


# -------------------------------- weights --------------------------------- #


def test_weights_are_symmetric_and_sum_to_one():
    weights = GaussianKernel((1.0, 1.0, 1.0), truncate=2.0).weights(0)

    assert weights.sum().item() == pytest.approx(1.0)
    assert weights.tolist() == pytest.approx(weights.flip(0).tolist())
    assert weights.argmax().item() == len(weights) // 2  # heaviest at the centre


def test_a_zero_sigma_axis_is_a_pass_through_weight():
    kernel = GaussianKernel((1.0, 1.0, 0.0))

    assert kernel.temporal_radius == 0
    assert kernel.weights(2).tolist() == [1.0]


# --------------------------------- apply ---------------------------------- #


def test_apply_rejects_a_target_outside_the_window():
    with pytest.raises(ValueError, match="not an index into"):
        GaussianKernel((1.0, 1.0, 1.0)).apply(torch.zeros(3, 4, 4), 3)


def test_apply_rejects_a_window_that_is_not_float32():
    with pytest.raises(Exception, match=r"\[3,4,4\]"):
        GaussianKernel((1.0, 1.0, 0.0)).apply(
            torch.zeros(3, 4, 4, dtype=torch.float64), 0
        )


def test_a_zero_sigma_everywhere_is_the_identity():
    kernel = GaussianKernel((0.0, 0.0, 0.0))
    window = torch.rand(1, 4, 4)

    assert torch.allclose(kernel.apply(window, 0), window[0])


def test_a_flat_field_survives_unchanged_across_the_border():
    # The renormalization that matters: without dividing by the weight that
    # actually landed, every border pixel would darken toward zero.
    window = torch.full((3, 6, 7), 5.0)

    out = GaussianKernel((1.0, 1.0, 1.0), truncate=2.0).apply(window, 0)

    assert torch.allclose(out, torch.full((6, 7), 5.0), atol=1e-5)


def test_a_spike_is_spread_over_its_neighbourhood_not_deleted():
    # The Gaussian's defining contrast with the median: a lone spike bleeds into
    # its neighbours instead of being outvoted away.
    window = torch.zeros(1, 5, 5)
    window[0, 2, 2] = 100.0

    blurred = GaussianKernel((1.0, 1.0, 0.0), truncate=1.0).apply(window, 0)

    assert 0.0 < blurred[2, 2].item() < 100.0  # centre reduced but not erased
    assert blurred[2, 1].item() > 0.0  # the spike bled into a neighbour


def test_gaussian_matches_a_brute_force_weighted_mean():
    # The separable pass, with time collapsed first and one final division, must
    # equal the full 3D normalized weighting -- not a per-axis approximation.
    frames = _frames(5)
    kernel = GaussianKernel((1.0, 1.0, 1.0), truncate=1.5)
    window = torch.from_numpy(frames)

    for index in range(len(frames)):
        assert torch.allclose(
            kernel.apply(window, index),
            _brute_gaussian(frames, kernel, index),
            atol=1e-6,
        )


# --------------------------------- config --------------------------------- #


def test_params_hold_what_they_were_given():
    assert GaussianConfig(1.0).sigma == 1.0
    assert GaussianConfig((1.0, 2.0)).sigma == (1.0, 2.0)
    assert GaussianConfig((1.0, 1.0, 1.0)).truncate == 4.0  # the only default


def test_params_are_frozen_records():
    config = GaussianConfig((1.0, 1.0, 1.0))

    with pytest.raises(FrozenInstanceError):
        config.truncate = 2.0  # ty: ignore[invalid-assignment]


def test_build_expands_the_held_sigma_into_a_kernel():
    config = GaussianConfig((1.0, 0.5), truncate=2.0)
    kernel = config.build()

    assert isinstance(config, KernelConfig)
    assert isinstance(kernel, GaussianKernel)
    assert kernel.sigma == (1.0, 1.0, 0.5)
    assert kernel.truncate == 2.0
