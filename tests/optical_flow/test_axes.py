from __future__ import annotations

import pytest
import torch

from iivs_cardio.common.warp import backward_warp
from iivs_cardio.optical_flow.estimators import FarnebackConfig
from iivs_cardio.optical_flow.metrics import (
    flow_magnitude,
    forward_backward_error,
    identity_ssim,
    warp_consistency,
)

SIZE = 128


def _textured() -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(SIZE).float(), torch.arange(SIZE).float(), indexing="ij"
    )
    weave = (
        128
        + 60 * torch.sin(2 * torch.pi * x / 17)
        + 50 * torch.sin(2 * torch.pi * y / 23)
    )
    return weave.clamp(0, 255).to(torch.uint8)


def _shifted(frame: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    offset = torch.zeros((2, SIZE, SIZE))
    offset[0] = -dx
    offset[1] = -dy
    return backward_warp(frame, offset)


def _uniform_flow(dx: float, dy: float) -> torch.Tensor:
    flow = torch.zeros((2, SIZE, SIZE))
    flow[0] = dx
    flow[1] = dy
    return flow


def test_the_floor_is_the_pair_compared_without_a_warp():
    # Sampling at `grid + 0` gives the frame back, so the floor is `frame2`
    # scored against `frame1` as they stand. Computing it through the warp
    # would be the same answer for a longer route.
    first = _textured()
    second = _shifted(first, 1.0, 0.6)

    through_warp = warp_consistency(first, second, torch.zeros((2, SIZE, SIZE)))

    assert identity_ssim(first, second) == pytest.approx(
        float(through_warp["ssim"]), abs=1e-6
    )


def test_the_floor_is_what_a_score_has_to_be_read_against():
    # Sub-pixel motion leaves the two frames nearly alike, so a raw score says
    # almost nothing and two flows an order apart in gain look the same.
    first = _textured()
    near = _shifted(first, 0.35, 0.2)
    far = _shifted(first, 1.0, 0.6)

    of = FarnebackConfig().build("cpu")
    near_score = float(warp_consistency(first, near, of.calc(first, near))["ssim"])
    far_score = float(warp_consistency(first, far, of.calc(first, far))["ssim"])

    assert near_score == pytest.approx(far_score, abs=0.01)  # raw: indistinguishable

    near_gain = near_score - float(identity_ssim(first, near))
    far_gain = far_score - float(identity_ssim(first, far))

    assert far_gain > near_gain * 3  # gain: an order apart


def test_identical_frames_leave_no_gain_to_be_earned():
    frame = _textured()

    assert float(identity_ssim(frame, frame)) == pytest.approx(1.0)


def test_a_consistent_pair_of_flows_cancels():
    # Following a correspondence forward and back returns where it started, so
    # a uniform flow and its negation leave nothing behind.
    forward = _uniform_flow(2.0, -1.0)

    error = forward_backward_error(forward, -forward)

    assert float(error) == pytest.approx(0.0, abs=1e-4)


def test_a_flow_that_does_not_come_back_is_measured_in_pixels():
    # The backward field points the same way as the forward one, so following
    # both lands twice as far out as the flow itself is long.
    forward = _uniform_flow(3.0, 4.0)

    error = forward_backward_error(forward, forward)

    assert float(error) == pytest.approx(10.0, rel=1e-3)  # 2 * |(3, 4)|


def test_a_zero_flow_is_perfectly_self_consistent():
    # Which is why this axis cannot be read alone: doing nothing scores best.
    zero = torch.zeros((2, SIZE, SIZE))

    assert float(forward_backward_error(zero, zero)) == pytest.approx(0.0)


def test_a_real_estimate_comes_back_to_within_a_fraction_of_a_pixel():
    first = _textured()
    second = _shifted(first, 1.0, 0.6)
    of = FarnebackConfig().build("cpu")

    error = forward_backward_error(of.calc(first, second), of.calc(second, first))

    assert 0.0 < float(error) < 0.5


def test_magnitude_is_the_length_of_the_flow_in_pixels():
    assert float(flow_magnitude(_uniform_flow(3.0, 4.0))) == pytest.approx(5.0)
    assert float(flow_magnitude(torch.zeros((2, SIZE, SIZE)))) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("measure", "args"),
    (
        pytest.param(forward_backward_error, 2, id="forward_backward_error"),
        pytest.param(flow_magnitude, 1, id="flow_magnitude"),
    ),
)
def test_a_batch_scores_each_pair_and_averages_to_the_same_thing(measure, args):
    # `reduce=False` keeps one value per pair, and `reduce=True` is exactly
    # their mean -- the property the document's fold rests on.
    batch = torch.stack([_uniform_flow(1.0, 0.0), _uniform_flow(3.0, 4.0)])
    given = (batch,) * args

    per_pair = measure(*given, reduce=False)

    assert per_pair.shape == (2,)
    assert float(measure(*given)) == pytest.approx(float(per_pair.mean()))
