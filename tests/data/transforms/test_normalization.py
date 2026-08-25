from __future__ import annotations

import pytest
import torch

from iivs_cardio.data.transforms.normalization import (
    FrameNormalizer,
    NormalizerConfig,
)


def _ramp(low: float, high: float, size: int = 8) -> torch.Tensor:
    """Return a `(size, size)` frame whose values run from `low` to `high`."""
    return torch.linspace(low, high, size * size).reshape(size, size)


# ========================== #
#          Scaling           #
# ========================== #


def test_the_source_ends_land_on_the_target_ends():
    scaled = FrameNormalizer((0.0, 2.0), (0.0, 1.0), torch.float32).apply(
        _ramp(0.0, 2.0)
    )

    assert scaled.min().item() == pytest.approx(0.0)
    assert scaled.max().item() == pytest.approx(1.0)


def test_two_frames_of_one_sequence_scale_by_the_same_constants():
    # The whole reason the range is fixed rather than measured per call: a
    # dimmer frame comes out dimmer. Scaling each frame by its own range would
    # send both to 1.0 and destroy the brightness constancy a dense estimator
    # reads motion from.
    normalizer = FrameNormalizer((0.0, 2.0), (0.0, 1.0), torch.float32)

    dim = normalizer.apply(_ramp(0.0, 1.0))
    bright = normalizer.apply(_ramp(0.0, 2.0))

    assert dim.max().item() == pytest.approx(0.5)
    assert bright.max().item() == pytest.approx(1.0)


def test_a_value_outside_the_source_range_is_clamped():
    # A range measured across a dataset is not a bound on any one frame of it.
    scaled = FrameNormalizer((0.0, 1.0), (0.0, 1.0), torch.float32).apply(
        _ramp(-1.0, 2.0)
    )

    assert scaled.min().item() == pytest.approx(0.0)
    assert scaled.max().item() == pytest.approx(1.0)


def test_scaling_is_affine_between_the_ends():
    # Verified against the map computed by hand rather than against what the
    # implementation answered: a source of [10, 20] onto a target of [0, 100]
    # puts 12.5 at 25.
    scaled = FrameNormalizer((10.0, 20.0), (0.0, 100.0), torch.float32).apply(
        torch.full((4, 4), 12.5)
    )

    assert scaled.unique().tolist() == pytest.approx([25.0])


def test_an_integer_dtype_rounds_onto_the_target_span():
    scaled = FrameNormalizer((0.0, 1.0), (0.0, 255.0), torch.uint8).apply(
        _ramp(0.0, 1.0)
    )

    assert scaled.dtype == torch.uint8
    assert (int(scaled.min()), int(scaled.max())) == (0, 255)


def test_a_target_narrower_than_the_dtype_is_kept_to():
    scaled = FrameNormalizer((0.0, 1.0), (0.0, 100.0), torch.uint8).apply(
        _ramp(0.0, 1.0)
    )

    assert int(scaled.max()) == 100


def test_a_float_target_may_go_below_zero():
    scaled = FrameNormalizer((0.0, 1.0), (-1.0, 1.0), torch.float32).apply(
        _ramp(0.0, 1.0)
    )

    assert scaled.min().item() == pytest.approx(-1.0)
    assert scaled.max().item() == pytest.approx(1.0)


def test_leading_dimensions_ride_along_on_the_one_pair_of_constants():
    normalizer = FrameNormalizer((0.0, 2.0), (0.0, 1.0), torch.float32)
    batch = torch.stack((_ramp(0.0, 1.0), _ramp(0.0, 2.0)))

    scaled = normalizer.apply(batch)

    assert scaled.shape == batch.shape
    assert scaled[0].max().item() == pytest.approx(0.5)
    assert scaled[1].max().item() == pytest.approx(1.0)


def test_the_input_dtype_says_nothing_about_the_output():
    normalizer = FrameNormalizer((0.0, 255.0), (0.0, 1.0), torch.float32)

    scaled = normalizer.apply(torch.full((4, 4), 255, dtype=torch.uint8))

    assert scaled.dtype == torch.float32
    assert scaled.unique().tolist() == pytest.approx([1.0])


def test_something_that_is_not_a_frame_is_refused():
    normalizer = FrameNormalizer((0.0, 1.0), (0.0, 1.0), torch.float32)

    with pytest.raises(Exception, match=r"\[8\]"):
        normalizer.apply(torch.zeros(8))


# ========================== #
#         Refusals           #
# ========================== #


def test_an_empty_span_is_refused_by_the_name_it_was_given_under():
    with pytest.raises(ValueError, match="empty source"):
        FrameNormalizer((1.0, 1.0), (0.0, 1.0), torch.float32)

    with pytest.raises(ValueError, match="empty target"):
        FrameNormalizer((0.0, 1.0), (1.0, 0.0), torch.float32)


def test_a_target_an_integer_dtype_cannot_hold_is_refused():
    with pytest.raises(ValueError, match="overflows"):
        FrameNormalizer((0.0, 1.0), (-1.0, 1.0), torch.uint8)


def test_a_float_dtype_bounds_no_target():
    # Nothing to overflow, so a span an integer dtype would refuse is fine.
    assert FrameNormalizer((0.0, 1.0), (-1e9, 1e9), torch.float32).target == (
        -1e9,
        1e9,
    )


# ========================== #
#           Config           #
# ========================== #


def test_a_level_nobody_offers_is_refused():
    with pytest.raises(ValueError, match="level"):
        NormalizerConfig(level="perframe")  # type: ignore[invalid-argument-type]


def test_the_given_level_needs_the_span_it_scales_from():
    with pytest.raises(ValueError, match="'given' scales from `source`"):
        NormalizerConfig(level="given")


def test_a_measured_level_refuses_a_span_of_its_own():
    # Otherwise a run would carry two ranges and scale by one of them without
    # saying which.
    with pytest.raises(ValueError, match="'dataset' is measured"):
        NormalizerConfig(level="dataset", source=(0.0, 1.0))


def test_a_span_the_config_carries_is_checked_where_it_is_written():
    with pytest.raises(ValueError, match="empty source"):
        NormalizerConfig(level="given", source=(1.0, 1.0))

    with pytest.raises(ValueError, match="empty target"):
        NormalizerConfig(target=(1.0, 0.0))


def test_a_measured_level_scales_from_the_range_the_document_holds():
    built = NormalizerConfig(level="sequence").build(torch.uint8, (-3.0, 5.0))

    assert built == FrameNormalizer((-3.0, 5.0), (0.0, 255.0), torch.uint8)


def test_a_measured_level_handed_no_range_is_refused():
    with pytest.raises(ValueError, match="'dataset' scales from the range"):
        NormalizerConfig().build(torch.uint8)


def test_the_given_level_refuses_a_measured_range_beside_its_own():
    config = NormalizerConfig(level="given", source=(0.0, 1.0))

    with pytest.raises(ValueError, match="'given' brings its own range"):
        config.build(torch.uint8, (0.0, 2.0))


def test_no_target_takes_the_output_dtype_span():
    assert NormalizerConfig(level="given", source=(0.0, 1.0)).build(
        torch.uint8
    ).target == (0.0, 255.0)

    assert NormalizerConfig(level="given", source=(0.0, 1.0)).build(
        torch.float32
    ).target == (0.0, 1.0)


def test_a_target_the_config_carries_beats_the_dtype_span():
    built = NormalizerConfig(level="given", source=(0.0, 1.0), target=(0.0, 100.0))

    assert built.build(torch.uint8).target == (0.0, 100.0)


def test_a_config_that_could_not_scale_is_refused_where_it_is_built():
    # The dtype is not known until `build`, so a target no integer dtype can
    # hold is only refusable there.
    config = NormalizerConfig(level="given", source=(0.0, 1.0), target=(-1.0, 1.0))

    with pytest.raises(ValueError, match="overflows"):
        config.build(torch.uint8)
