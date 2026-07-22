from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch

from iivs_cardio.data.preprocessing.normalization import FrameNormalizer


def _ramp(low: float, high: float, h: int = 8, w: int = 8) -> torch.Tensor:
    # A float32 frame spanning exactly [low, high], so the encoding is exact.
    values = np.linspace(low, high, h * w, dtype=np.float32).reshape(h, w)
    return torch.as_tensor(values)


# --------------------------- perframe / pairwise ------------------------ #


def test_perframe_scales_each_frame_by_its_own_range():
    # Different spans, identical output -- which is exactly why the mode breaks
    # brightness constancy between the two frames.
    out1, out2 = FrameNormalizer("perframe").apply(_ramp(0.0, 1.0), _ramp(100.0, 300.0))
    assert (out1.min().item(), out1.max().item()) == pytest.approx((0.0, 1.0))
    assert torch.allclose(out1, out2)


def test_pairwise_scales_both_frames_by_the_joint_range():
    # frame2 spans the pair, so it keeps the full range while frame1 occupies the
    # lower half -- the pair stays comparable, unlike perframe.
    out1, out2 = FrameNormalizer("pairwise").apply(_ramp(0.0, 1.0), _ramp(0.0, 2.0))
    assert out2.max().item() == pytest.approx(1.0)
    assert out1.max().item() == pytest.approx(0.5)  # 1.0 of a [0, 2] range
    assert not torch.allclose(out1, out2)


def test_perframe_rejects_a_uniform_frame():
    # max == min makes the scaling undefined; unguarded it becomes nan, then 0.
    uniform = torch.full((8, 8), 7.0)
    with pytest.raises(ValueError, match="uniform"):
        FrameNormalizer("perframe").apply(uniform, _ramp(0.0, 1.0))


def test_perframe_rejects_a_source_range():
    with pytest.raises(ValueError, match="source_range is for"):
        FrameNormalizer("perframe").source_range = (0.0, 1.0)


# --------------------------------- injected ------------------------------- #


def test_injected_range_is_shared_by_every_frame():
    normalizer = FrameNormalizer("injected")
    normalizer.source_range = (0.0, 2.0)
    out1, out2 = normalizer.apply(_ramp(0.0, 1.0), _ramp(0.0, 2.0))
    assert out1.max().item() == pytest.approx(0.5)  # half the injected span
    assert out2.max().item() == pytest.approx(1.0)


def test_injected_mode_requires_a_range():
    with pytest.raises(ValueError, match="has no source_range"):
        FrameNormalizer("injected").apply(_ramp(0.0, 1.0), _ramp(0.0, 1.0))


def test_injected_range_broadcasts_over_a_batch():
    # The injected range is a scalar pair, not the `(*dim, 1, 1)` tensors a
    # measured range produces, so it has to broadcast over the batch as is. The
    # payoff is what perframe/pairwise cannot do: batch elements stay on one
    # common scale instead of each being stretched to its own extremes.
    normalizer = FrameNormalizer("injected", source_range=(0.0, 4.0))
    frames = torch.stack([_ramp(0.0, 1.0), _ramp(0.0, 2.0)])
    out1, out2 = normalizer.apply(frames, frames)

    assert out1.shape == frames.shape
    assert out1[0].max().item() == pytest.approx(0.25)  # 1.0 of the [0, 4] span
    assert out1[1].max().item() == pytest.approx(0.5)  # 2.0 of the same span
    assert torch.equal(out1, out2)


def test_reset_forgets_the_injected_range():
    normalizer = FrameNormalizer("injected")
    normalizer.source_range = (0.0, 1.0)
    normalizer.apply(_ramp(0.0, 1.0), _ramp(0.0, 1.0))  # fine while injected
    normalizer.reset()
    with pytest.raises(ValueError, match="has no source_range"):
        normalizer.apply(_ramp(0.0, 1.0), _ramp(0.0, 1.0))


def test_source_range_rejects_an_empty_span():
    with pytest.raises(ValueError, match="empty source_range"):
        FrameNormalizer("injected").source_range = (5.0, 5.0)


def test_values_outside_the_injected_range_are_clamped_not_wrapped():
    # The failure this guards: without the clamp, a value just below the injected
    # minimum wraps to near-white in uint8 (-0.02 -> 251) with no error at all.
    normalizer = FrameNormalizer("injected", dtype=torch.uint8)
    normalizer.source_range = (0.0, 1.0)
    below, above = torch.full((4, 4), -0.02), torch.full((4, 4), 1.5)
    out_below, out_above = normalizer.apply(below, above)
    assert out_below.unique().tolist() == [0]
    assert out_above.unique().tolist() == [255]


# ---------------------------- dtype and batching ------------------------- #


def test_dtype_defaults_to_each_input_dtype():
    # `None` keeps whatever came in, per frame, so a float pipeline stays float;
    # the estimator path opts into uint8 explicitly instead.
    float_frame = _ramp(0.0, 1.0)
    uint8_frame = _ramp(0.0, 255.0).to(torch.uint8)

    out1, out2 = FrameNormalizer("perframe").apply(float_frame, uint8_frame)
    assert out1.dtype == torch.float32
    assert out2.dtype == torch.uint8  # each output follows its own input
    assert out1.max().item() == pytest.approx(1.0)
    assert out2.max().item() == 255


def test_explicit_dtype_overrides_the_input():
    out1, _ = FrameNormalizer("perframe", dtype=torch.uint8).apply(
        _ramp(0.0, 1.0), _ramp(0.0, 1.0)
    )
    assert out1.dtype == torch.uint8
    assert (out1.min().item(), out1.max().item()) == (0, 255)


def test_integer_dtype_spans_the_full_iinfo_range():
    out1, _ = FrameNormalizer("perframe", dtype=torch.int16).apply(
        _ramp(0.0, 1.0), _ramp(0.0, 1.0)
    )
    info = torch.iinfo(torch.int16)
    assert out1.dtype == torch.int16
    assert out1.min().item() == info.min
    assert out1.max().item() == info.max


def test_batched_pairwise_pairs_elementwise():
    # Pair i is (frame1[i], frame2[i]); each pair gets its own joint range.
    frame1 = torch.stack([_ramp(0.0, 1.0), _ramp(0.0, 10.0)])
    frame2 = torch.stack([_ramp(0.0, 2.0), _ramp(0.0, 10.0)])
    out1, out2 = FrameNormalizer("pairwise").apply(frame1, frame2)

    assert out1.shape == frame1.shape
    assert out1[0].max().item() == pytest.approx(0.5)  # 1.0 of [0, 2]
    assert out1[1].max().item() == pytest.approx(1.0)  # pair 1 shares one range
    assert torch.allclose(out1[1], out2[1])


def test_apply_rejects_mismatched_frames():
    with pytest.raises(Exception, match=r"\[2,8,8\]"):
        FrameNormalizer("perframe").apply(_ramp(0.0, 1.0), torch.zeros(2, 8, 8))


# ------------------------------- target_range ----------------------------- #


def test_target_range_maps_onto_a_symmetric_float_span():
    # The tanh-head convention: [-1, 1] instead of the [0, 1] default.
    out1, _ = FrameNormalizer("perframe", target_range=(-1.0, 1.0)).apply(
        _ramp(0.0, 1.0), _ramp(0.0, 1.0)
    )
    assert out1.min().item() == pytest.approx(-1.0)
    assert out1.max().item() == pytest.approx(1.0)
    assert out1.median().item() == pytest.approx(0.0, abs=0.05)  # midpoint maps to 0


def test_target_range_defaults_stay_dtype_derived():
    # `None` must reproduce the old hard-coded policy exactly, per dtype.
    frames = (_ramp(0.0, 1.0), _ramp(0.0, 1.0))
    explicit = FrameNormalizer("perframe", dtype=torch.uint8, target_range=(0.0, 255.0))
    assert torch.equal(
        FrameNormalizer("perframe", dtype=torch.uint8).apply(*frames)[0],
        explicit.apply(*frames)[0],
    )


def test_target_range_can_reserve_headroom_in_an_integer_dtype():
    # A sub-span of the dtype is legitimate -- 0..100 inside uint8, not 0..255.
    out1, _ = FrameNormalizer(
        "perframe", dtype=torch.uint8, target_range=(0.0, 100.0)
    ).apply(_ramp(0.0, 1.0), _ramp(0.0, 1.0))
    assert out1.dtype == torch.uint8
    assert (out1.min().item(), out1.max().item()) == (0, 100)


def test_target_range_rejects_a_span_the_integer_dtype_cannot_hold():
    # Unguarded, [-1, 1] on uint8 clamps to essentially binary output.
    with pytest.raises(ValueError, match="overflows"):
        FrameNormalizer("perframe", dtype=torch.uint8, target_range=(-1.0, 1.0))


def test_target_range_rejects_an_empty_span():
    with pytest.raises(ValueError, match="empty target_range"):
        FrameNormalizer("perframe", dtype=torch.float32, target_range=(1.0, 1.0))


def test_target_range_is_checked_against_each_input_dtype_when_dtype_is_none():
    # `dtype=None` defers the check to `apply`, where the input dtype is known.
    normalizer = FrameNormalizer("perframe", target_range=(-1.0, 1.0))
    with pytest.raises(ValueError, match="overflows"):
        normalizer.apply(_ramp(0.0, 255.0).to(torch.uint8), _ramp(0.0, 1.0))


# ------------------------------- source_range ----------------------------- #


def test_source_range_is_usable_straight_from_the_constructor():
    normalizer = FrameNormalizer("injected", source_range=(0.0, 2.0))
    out1, _ = normalizer.apply(_ramp(0.0, 1.0), _ramp(0.0, 2.0))
    assert out1.max().item() == pytest.approx(0.5)  # half the injected span


def test_source_range_survives_a_pickle_round_trip():
    # A DataLoader worker gets the normalizer by pickle and never sets the range itself.
    # S301 guards against untrusted input; this revives an object built one line up.
    blob = pickle.dumps(FrameNormalizer("injected", source_range=(0.0, 2.0)))
    revived = pickle.loads(blob)  # noqa: S301
    out1, _ = revived.apply(_ramp(0.0, 1.0), _ramp(0.0, 2.0))
    assert out1.max().item() == pytest.approx(0.5)


def test_source_range_is_rejected_for_a_measuring_mode():
    with pytest.raises(ValueError, match="source_range is for"):
        FrameNormalizer("perframe", source_range=(0.0, 1.0))


def test_reset_returns_to_the_constructed_source_range():
    # Back-to-construction, not back-to-empty: assignment is per-sequence, the
    # constructor value is the baseline that survives it.
    normalizer = FrameNormalizer("injected", source_range=(0.0, 2.0))
    normalizer.source_range = (0.0, 4.0)
    assert normalizer.apply(_ramp(0.0, 1.0), _ramp(0.0, 1.0))[0].max().item() == (
        pytest.approx(0.25)
    )
    normalizer.reset()
    assert normalizer.apply(_ramp(0.0, 1.0), _ramp(0.0, 1.0))[0].max().item() == (
        pytest.approx(0.5)
    )


def test_ranges_read_back_through_their_properties():
    # The getters are the public record of what a normalizer will do and what
    # `reset` restores, so a pipeline can log them instead of re-deriving them.
    normalizer = FrameNormalizer(
        "injected",
        dtype=torch.float32,
        source_range=(0.0, 2.0),
        target_range=(-1.0, 1.0),
    )
    assert normalizer.source_range == (0.0, 2.0)
    assert normalizer.target_range == (-1.0, 1.0)

    normalizer.source_range = (0.0, 4.0)
    assert normalizer.source_range == (0.0, 4.0)
    normalizer.reset()
    assert normalizer.source_range == (0.0, 2.0)  # back to the constructed baseline

    assert FrameNormalizer("perframe").source_range is None
    assert FrameNormalizer("perframe").target_range is None
