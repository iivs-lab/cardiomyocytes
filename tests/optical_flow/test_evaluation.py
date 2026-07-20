from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from iivs_cardio.optical_flow.estimators import Farneback
from iivs_cardio.optical_flow.evaluation import (
    FlowMetrics,
    MetricsAccumulator,
    OpticalFlowEvaluator,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="no CUDA-capable GPU detected",
)


def _textured() -> torch.Tensor:
    # A smooth sinusoidal texture (structure in both axes) — the same frame the
    # estimator tests use, so SSIM is well defined (non-constant).
    y, x = np.mgrid[0:64, 0:64]
    tex = 128 + 60 * np.sin(2 * np.pi * x / 16) + 60 * np.sin(2 * np.pi * y / 16)
    return torch.as_tensor(tex.astype(np.uint8))


def _shifted(frame: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(np.roll(frame.numpy(), shift=(2, 3), axis=(0, 1)))


def _zero_flow(h: int = 64, w: int = 64) -> torch.Tensor:
    return torch.zeros((h, w, 2), dtype=torch.float32)


def _uniform_flow(dx: float, dy: float, h: int = 64, w: int = 64) -> torch.Tensor:
    flow = torch.zeros((h, w, 2), dtype=torch.float32)
    flow[..., 0] = dx
    flow[..., 1] = dy
    return flow


# --------------------------------- warp ---------------------------------- #


def test_warp_with_zero_flow_is_identity():
    img = _textured()
    warped = OpticalFlowEvaluator.warp(img, _zero_flow())
    assert warped.shape == img.shape
    assert warped.dtype == torch.uint8
    assert torch.equal(warped, img)


def test_warp_recovers_known_shift():
    # Forward flow (dx, dy) = (3, 2) => curr[y, x] == prev[y - 2, x - 3]; the
    # backward warp (grid - flow) samples prev there, reconstructing
    # np.roll(prev, (2, 3)). Check the interior against an independent np.roll.
    prev = _textured()
    warped = OpticalFlowEvaluator.warp(prev, _uniform_flow(3.0, 2.0)).numpy()
    expected = np.roll(prev.numpy(), shift=(2, 3), axis=(0, 1))
    assert np.array_equal(warped[8:56, 8:56], expected[8:56, 8:56])


# -------------------------------- score ---------------------------------- #


def test_score_is_perfect_when_frames_match():
    # prev == curr and zero flow -> warped == curr: zero error, SSIM 1, PSNR inf.
    img = _textured()
    m = OpticalFlowEvaluator().score(img, img, _zero_flow())
    assert m.mse == 0.0
    assert m.mae == 0.0
    assert m.ssim == pytest.approx(1.0)
    assert math.isinf(m.psnr)
    assert m.lpips is None  # off by default


def test_score_mse_mae_match_independent_torch():
    # Zero flow -> warped == prev, so the residual is prev - curr; check MSE/MAE
    # against a torch computation done independently of the evaluator.
    prev = _textured()
    curr = _shifted(prev)
    m = OpticalFlowEvaluator().score(prev, curr, _zero_flow())

    diff = prev.float() - curr.float()
    assert m.mse == pytest.approx(float((diff * diff).mean()))
    assert m.mae == pytest.approx(float(diff.abs().mean()))


def test_score_rewards_the_matching_flow_over_zero_flow():
    prev = _textured()
    curr = _shifted(prev)
    ev = OpticalFlowEvaluator()

    matching = ev.score(prev, curr, _uniform_flow(3.0, 2.0))
    identity = ev.score(prev, curr, _zero_flow())

    assert matching.mae < identity.mae
    assert matching.ssim > identity.ssim


def test_score_with_real_estimator_flow_is_consistent():
    # End-to-end: a real Farneback flow warps prev onto curr well. Also guards
    # the warp direction — the wrong sign collapses SSIM to ~-0.23.
    prev = _textured()
    curr = _shifted(prev)
    flow = Farneback(device="cpu").calc(prev, curr)

    m = OpticalFlowEvaluator().score(prev, curr, flow)
    assert m.ssim > 0.9
    assert m.mae < 15.0


def test_score_rejects_non_uint8_frame():
    ev = OpticalFlowEvaluator()
    bad = torch.zeros((64, 64), dtype=torch.float32)
    with pytest.raises(Exception, match=r"f32\[64,64\]"):
        ev.score(bad, _textured(), _zero_flow())


def test_score_rejects_wrong_flow_shape():
    ev = OpticalFlowEvaluator()
    bad_flow = torch.zeros((64, 64, 3), dtype=torch.float32)  # last dim must be 2
    with pytest.raises(Exception, match=r"\[64,64,3\]"):
        ev.score(_textured(), _textured(), bad_flow)


@requires_cuda
def test_score_stays_on_cuda():
    prev = _textured().cuda()
    curr = _shifted(_textured()).cuda()
    flow = _uniform_flow(3.0, 2.0).cuda()

    # warp is grid_sample only (no cuDNN) — must stay on-device, no host offload.
    assert OpticalFlowEvaluator.warp(prev, flow).device.type == "cuda"
    try:
        m = OpticalFlowEvaluator().score(prev, curr, flow)
    except RuntimeError as exc:  # SSIM's GPU conv needs a working torch cuDNN
        if "CUDNN" in str(exc).upper():
            pytest.skip(f"torch cuDNN unavailable for GPU conv (environment): {exc}")
        raise
    assert math.isfinite(m.mae)
    assert m.ssim > 0.9


# ---------------------------------- LPIPS -------------------------------- #


def test_lpips_opt_in_returns_a_float():
    ev = OpticalFlowEvaluator(lpips=True, lpips_net="squeeze")  # smallest backbone
    prev = _textured()
    curr = _shifted(prev)
    try:
        m = ev.score(prev, curr, _uniform_flow(3.0, 2.0))
    except Exception as exc:  # noqa: BLE001 - backbone weights need a one-time download
        pytest.skip(f"LPIPS backbone weights unavailable: {exc}")
    assert m.lpips is not None
    assert math.isfinite(m.lpips)


# --------------------------- MetricsAccumulator -------------------------- #


def test_accumulator_means_metrics():
    acc = MetricsAccumulator()
    acc.add(FlowMetrics(psnr=10.0, ssim=0.5, mse=4.0, mae=2.0))
    acc.add(FlowMetrics(psnr=20.0, ssim=0.7, mse=6.0, mae=4.0))

    assert acc.count == 2
    assert acc.mean() == FlowMetrics(psnr=15.0, ssim=0.6, mse=5.0, mae=3.0, lpips=None)


def test_accumulator_averages_lpips_only_when_present():
    acc = MetricsAccumulator()
    acc.add(FlowMetrics(psnr=10.0, ssim=0.5, mse=4.0, mae=2.0, lpips=0.2))
    acc.add(FlowMetrics(psnr=20.0, ssim=0.7, mse=6.0, mae=4.0, lpips=0.4))
    assert acc.mean().lpips == pytest.approx(0.3)


def test_accumulator_empty_mean_raises():
    with pytest.raises(ValueError, match="no metrics accumulated"):
        MetricsAccumulator().mean()
