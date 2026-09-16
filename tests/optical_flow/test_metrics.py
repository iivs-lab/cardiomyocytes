from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from iivs_cardio.common.warp import backward_warp
from iivs_cardio.optical_flow.metrics import (
    WarpConsistency,
    identity_ssim,
    warp_consistency,
)

METRICS = {"ssim", "psnr", "mse", "mae"}

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="no CUDA-capable GPU detected",
)


def _textured() -> torch.Tensor:
    # A smooth sinusoidal texture (structure in both axes), so SSIM is well
    # defined (a constant frame would degenerate it).
    y, x = np.mgrid[0:64, 0:64]
    tex = 128 + 60 * np.sin(2 * np.pi * x / 16) + 60 * np.sin(2 * np.pi * y / 16)
    return torch.as_tensor(tex.astype(np.uint8))


def _shifted(frame: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(np.roll(frame.numpy(), shift=(2, 3), axis=(0, 1)))


def _zero_flow(h: int = 64, w: int = 64) -> torch.Tensor:
    return torch.zeros((2, h, w), dtype=torch.float32)


def _uniform_flow(dx: float, dy: float, h: int = 64, w: int = 64) -> torch.Tensor:
    flow = torch.zeros((2, h, w), dtype=torch.float32)
    flow[0] = dx
    flow[1] = dy
    return flow


def test_warp_consistency_returns_metric_tensors():
    frame1 = _textured()
    out = warp_consistency(frame1, _shifted(frame1), _uniform_flow(3.0, 2.0))
    assert set(out) == METRICS
    assert all(isinstance(v, torch.Tensor) for v in out.values())
    assert all(v.ndim == 0 for v in out.values())  # 0-d scalars, not floats


def test_warp_consistency_is_perfect_when_the_warp_reconstructs_the_frame():
    # frame1 == frame2 with zero flow -> the warp reproduces the frame exactly.
    frame = _textured()
    out = warp_consistency(frame, frame, _zero_flow())
    assert out["mse"].item() == 0.0
    assert out["mae"].item() == 0.0
    assert out["ssim"].item() == pytest.approx(1.0)
    assert torch.isinf(out["psnr"])


def test_warp_consistency_mse_mae_match_independent_torch():
    # Zero flow -> the warp is the identity, so the residual is frame2 - frame1;
    # check MSE/MAE against a torch computation independent of the metric functions.
    frame1 = _textured()
    frame2 = _shifted(frame1)
    out = warp_consistency(frame1, frame2, _zero_flow())

    diff = frame1.float() - frame2.float()
    assert out["mse"].item() == pytest.approx(float((diff * diff).mean()))
    assert out["mae"].item() == pytest.approx(float(diff.abs().mean()))


def test_warp_consistency_rewards_the_matching_flow():
    frame1 = _textured()
    frame2 = _shifted(frame1)  # np.roll by (2, 3) => forward flow (dx, dy) = (3, 2)

    matching = warp_consistency(frame1, frame2, _uniform_flow(3.0, 2.0))
    identity = warp_consistency(frame1, frame2, _zero_flow())

    assert matching["mae"].item() < identity["mae"].item()
    assert matching["mse"].item() < identity["mse"].item()
    assert matching["ssim"].item() > identity["ssim"].item()
    assert matching["psnr"].item() > identity["psnr"].item()


def test_warp_consistency_samples_frame2_along_the_flow():
    """Pin the warp direction: sample `frame2` at `grid + flow`, score on `frame1`.

    The flow is non-uniform on purpose: under a uniform translation the two
    directions coincide, so every other test in this file passes either way. A
    flipped *sign* is a different error, and those tests do catch it -- this one
    covers the direction they cannot see.
    """
    frame1 = _textured()
    frame2 = _shifted(frame1)

    flow = torch.zeros((2, 64, 64), dtype=torch.float32)
    flow[0] = torch.linspace(0.0, 4.0, 64)  # varies along x
    flow[1] = torch.linspace(0.0, 2.0, 64)[:, None]  # varies along y

    mse = warp_consistency(frame1, frame2, flow)["mse"].item()

    along = backward_warp(frame2.float(), flow) - frame1.float()
    assert mse == pytest.approx(float((along * along).mean()))

    # The reverse direction scores differently here -- that is the whole point.
    reversed_ = backward_warp(frame1.float(), -flow) - frame2.float()
    assert float((reversed_ * reversed_).mean()) != pytest.approx(mse)


def test_warp_consistency_respects_padding_mode():
    # A large flow samples off-grid at the border, where `zeros` (black) and
    # `border` (replicate) reconstruct differently, so the scores disagree.
    frame1 = _textured()
    frame2 = _shifted(frame1)
    far = _uniform_flow(-40.0, 0.0)
    assert (
        warp_consistency(frame1, frame2, far, padding_mode="zeros")["mae"].item()
        != warp_consistency(frame1, frame2, far, padding_mode="border")["mae"].item()
    )


def test_warp_consistency_accepts_float_frames():
    # FrameType is `Real`, so float frames work; state `data_range` to match.
    frame = _textured().float() / 255.0
    out = warp_consistency(frame, frame, _zero_flow(), data_range=1.0)
    assert out["mse"].item() == pytest.approx(0.0, abs=1e-6)
    assert out["ssim"].item() == pytest.approx(1.0)


def test_warp_consistency_infers_data_range_from_the_integer_dtype():
    # Omitted `data_range` derives uint8's 0..255 span, matching an explicit 255.
    frame1 = _textured()
    frame2 = _shifted(frame1)
    flow = _uniform_flow(3.0, 2.0)
    derived = warp_consistency(frame1, frame2, flow)
    explicit = warp_consistency(frame1, frame2, flow, data_range=255.0)
    for key in METRICS:
        assert torch.equal(derived[key], explicit[key])


def test_warp_consistency_requires_data_range_for_float_frames():
    # A float dtype has no intrinsic range; guessing one silently inflates PSNR
    # and saturates SSIM, so it must be passed rather than assumed.
    frame = _textured().float()
    with pytest.raises(ValueError, match="data_range cannot be inferred"):
        warp_consistency(frame, frame, _zero_flow())


def test_warp_consistency_accepts_batched_frames():
    # Shared leading dims: `(N, H, W)` frames + `(N, 2, H, W)` flow reduce to one
    # scalar per metric, matching the single pair when the batch repeats it.
    frame1 = _textured()
    frame2 = _shifted(frame1)
    flow = _uniform_flow(3.0, 2.0)

    batched = warp_consistency(
        torch.stack([frame1, frame1]),
        torch.stack([frame2, frame2]),
        torch.stack([flow, flow]),
    )
    single = warp_consistency(frame1, frame2, flow)

    assert set(batched) == METRICS
    assert all(v.ndim == 0 for v in batched.values())
    for key in METRICS:
        assert batched[key].item() == pytest.approx(single[key].item(), rel=1e-5)


def test_warp_consistency_reduce_false_keeps_one_score_per_pair():
    # `reduce=False` drops only the batch reduction: `(*dim, H, W)` -> `(*dim)`,
    # and an unbatched pair still yields 0-d (its `*dim` is empty).
    frame1 = torch.stack([_textured(), _textured().flip(0)])
    frame2 = torch.stack([_shifted(_textured()), _textured()])
    flow = torch.stack([_uniform_flow(3.0, 2.0), _zero_flow()])

    per_pair = warp_consistency(frame1, frame2, flow, reduce=False)
    assert all(v.shape == (2,) for v in per_pair.values())
    assert all(
        warp_consistency(frame1[0], frame2[0], flow[0], reduce=False)[k].ndim == 0
        for k in METRICS
    )


def test_warp_consistency_reduce_true_is_the_mean_over_pairs():
    # The flag is pure aggregation: reducing equals averaging the per-pair scores
    # for every metric, PSNR included (not the batch-pooled PSNR).
    frame1 = torch.stack([_textured(), _textured().flip(0)])
    frame2 = torch.stack([_shifted(_textured()), _textured()])
    flow = torch.stack([_uniform_flow(3.0, 2.0), _zero_flow()])

    reduced = warp_consistency(frame1, frame2, flow)
    per_pair = warp_consistency(frame1, frame2, flow, reduce=False)
    for key in METRICS:
        assert reduced[key].ndim == 0
        assert reduced[key].item() == pytest.approx(per_pair[key].mean().item())


def test_warp_consistency_rejects_wrong_flow_shape():
    frame = _textured()
    bad = torch.zeros((3, 64, 64), dtype=torch.float32)  # channel dim must be 2
    with pytest.raises(Exception, match=r"\[3,64,64\]"):
        warp_consistency(frame, frame, bad)


def test_warp_consistency_rejects_non_real_frame():
    # `Real` excludes complex/bool, which the warp cannot handle meaningfully.
    bad = torch.zeros((64, 64), dtype=torch.complex64)
    with pytest.raises(Exception, match=r"c64\[64,64\]"):
        warp_consistency(bad, _textured(), _zero_flow())


# ----------------------------- WarpConsistency ---------------------------- #


def test_warp_consistency_module_matches_function():
    frame1 = _textured()
    frame2 = _shifted(frame1)
    flow = _uniform_flow(3.0, 2.0)

    from_module = WarpConsistency()(frame1, frame2, flow)
    from_function = warp_consistency(frame1, frame2, flow)

    assert set(from_module) == set(from_function)
    for key in from_function:
        assert torch.equal(from_module[key], from_function[key])


def test_warp_consistency_module_reuses_its_warp_grid(monkeypatch):
    # The module holds a BackwardWarp, whose (H, W) grid is built once and reused
    # across same-size calls. Spy on torch.meshgrid to prove it.
    real_meshgrid = torch.meshgrid
    calls = 0

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_meshgrid(*args, **kwargs)

    monkeypatch.setattr(torch, "meshgrid", counting)
    module = WarpConsistency()
    frame1 = _textured()
    frame2 = _shifted(frame1)
    flow = _uniform_flow(3.0, 2.0)
    for _ in range(3):
        module(frame1, frame2, flow)
    assert calls == 1  # grid built once, reused across the 3 evaluations


def test_warp_consistency_module_honours_padding_mode():
    frame1 = _textured()
    frame2 = _shifted(frame1)
    far = _uniform_flow(-40.0, 0.0)
    assert torch.equal(
        WarpConsistency(padding_mode="zeros")(frame1, frame2, far)["mae"],
        warp_consistency(frame1, frame2, far, padding_mode="zeros")["mae"],
    )


@requires_cuda
def test_warp_consistency_stays_on_device():
    frame1 = _textured().cuda()
    frame2 = _shifted(_textured()).cuda()
    flow = _uniform_flow(3.0, 2.0).cuda()
    try:
        out = warp_consistency(frame1, frame2, flow)
    except RuntimeError as exc:  # SSIM's GPU conv needs a working torch cuDNN
        if "CUDNN" in str(exc).upper():
            pytest.skip(f"torch cuDNN unavailable for GPU conv (environment): {exc}")
        raise
    assert all(v.device.type == "cuda" for v in out.values())


def test_psnr_is_the_ratio_the_formula_gives():
    # Worked out from the residual rather than read back from torchmetrics: the
    # pooled form (one log over the batch's total error) is a different quantity
    # and would pass a comparison against the library with itself.
    frame1 = _textured()
    frame2 = _shifted(frame1)

    out = warp_consistency(frame1, frame2, _zero_flow())

    residual = frame2.numpy().astype(np.float64) - frame1.numpy()
    mse = float(np.mean(residual**2))

    assert out["psnr"].item() == pytest.approx(
        10.0 * math.log10(255.0**2 / mse), rel=1e-6
    )


def _gaussian_ssim(first: np.ndarray, second: np.ndarray) -> float:
    """Return SSIM the way this project scores it, written out rather than called."""
    taps = np.exp(-0.5 * (np.arange(-5, 6) / 1.5) ** 2)
    taps /= taps.sum()

    def blur(field: np.ndarray) -> np.ndarray:
        for axis in (0, 1):
            width = [(5, 5) if a == axis else (0, 0) for a in (0, 1)]
            padded = np.pad(field, width, mode="reflect")
            taken = [
                np.take(padded, range(start, start + field.shape[axis]), axis=axis)
                for start in range(len(taps))
            ]
            field = sum(w * t for w, t in zip(taps, taken, strict=True))

        return field

    stable_1, stable_2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
    mean_1, mean_2 = blur(first), blur(second)
    var_1 = blur(first * first) - mean_1**2
    var_2 = blur(second * second) - mean_2**2
    covariance = blur(first * second) - mean_1 * mean_2

    top = (2 * mean_1 * mean_2 + stable_1) * (2 * covariance + stable_2)
    bottom = (mean_1**2 + mean_2**2 + stable_1) * (var_1 + var_2 + stable_2)

    return float((top / bottom).mean())


def test_ssim_is_the_gaussian_pass_this_project_scores_by():
    # Two conventions at once: an 11-tap gaussian of sigma 1.5, and the mean
    # over the *whole* image rather than over the interior a windowed SSIM
    # keeps. The second is why these numbers sit a few thousandths from
    # skimage's, which a reader holding one of the two has to know.
    frame1 = _textured()
    frame2 = _shifted(frame1)

    scored = identity_ssim(frame1, frame2).item()

    expected = _gaussian_ssim(
        frame1.numpy().astype(np.float64), frame2.numpy().astype(np.float64)
    )
    assert scored == pytest.approx(expected, abs=2e-5)


def test_an_integer_pair_is_scored_as_the_same_pair_given_as_float():
    # A warp handed an integer frame rounds its samples back to that dtype, and
    # the floor a score is read against warps nothing. Rounding on one side only
    # charged the rounding to the flow, most heavily where motion is sub-pixel,
    # which is where this project's is. The flow here lands between pixels.
    frame1 = _textured()
    frame2 = _shifted(frame1)
    flow = _uniform_flow(3.0, 2.0)
    flow[0] += torch.linspace(0.0, 0.7, 64)

    as_integers = warp_consistency(frame1, frame2, flow)
    as_floats = warp_consistency(frame1.float(), frame2.float(), flow, data_range=255.0)

    for metric in ("mse", "mae", "psnr", "ssim"):
        assert as_integers[metric].item() == pytest.approx(
            as_floats[metric].item(), rel=1e-6
        ), metric
