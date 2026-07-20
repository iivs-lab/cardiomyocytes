from __future__ import annotations

__all__ = ("FlowMetrics", "MetricsAccumulator", "OpticalFlowEvaluator")

from typing import Literal, NamedTuple, cast

import torch
from beartype import beartype
from jaxtyping import Float32, UInt8, jaxtyped
from torch import Tensor
from torch.nn.functional import grid_sample
from torchmetrics.functional import mean_absolute_error, mean_squared_error
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

FrameType = UInt8[Tensor, "H W"]
FlowType = Float32[Tensor, "H W 2"]
LpipsNet = Literal["alex", "vgg", "squeeze"]


class FlowMetrics(NamedTuple):
    """Warp-consistency metrics for one flow field.

    Lower `mse` / `mae` and higher `psnr` / `ssim` mean the flow warps the
    previous frame onto the current one more faithfully; `psnr` is `inf` on an
    exact match. `lpips` (learned perceptual distance, lower = better) is `None`
    unless the evaluator was built with `lpips=True`.
    """

    psnr: float
    ssim: float
    mse: float
    mae: float
    lpips: float | None = None


class OpticalFlowEvaluator:
    """Warp-consistency scoring of a dense flow field, on `torch.Tensor`s.

    With no ground-truth flow, warp consistency is the standard proxy: warp the
    previous frame by the flow to synthesize a fake current frame, and score it
    against the real one. Everything stays on the input tensor's device
    (`grid_sample` warp + torchmetrics), so a CUDA flow is scored on the GPU with
    no host transfer and a CPU flow on the CPU.

    Args:
        data_range: Value range of the frames (255 for uint8), for PSNR/SSIM.
        padding_mode: `grid_sample` out-of-bounds policy (`border` replicates
            edges; `zeros` pads black; `reflection` mirrors).
        lpips: Include a learned perceptual distance (LPIPS). Opt-in because it
            runs a CNN per pair and its ImageNet-trained backbone is
            out-of-domain for grayscale phase images — treat it as exploratory.
        lpips_net: LPIPS backbone (`alex` cheapest, then `squeeze`, `vgg`).
    """

    def __init__(
        self,
        data_range: float = 255.0,
        *,
        padding_mode: str = "border",
        lpips: bool = False,
        lpips_net: LpipsNet = "alex",
    ) -> None:
        self.data_range = data_range
        self.padding_mode = padding_mode
        self.use_lpips = lpips
        self.lpips_net = lpips_net
        self._lpips_metric = None

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def warp(
        image: FrameType, flow: FlowType, *, padding_mode: str = "border"
    ) -> FrameType:
        """Backward-warp `image` by `flow` to the next frame's grid, bilinearly.

        The estimators return the forward flow (`prev -> curr`), so the current
        frame is reconstructed by sampling `prev` at ``grid - flow`` -- NOT
        ``grid + flow`` (empirically the correct sign recovers a known shift at
        SSIM ~0.98 vs ~-0.23). Runs on `image`'s device.
        """
        height, width = image.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=image.device),
            torch.arange(width, dtype=torch.float32, device=image.device),
            indexing="ij",
        )
        sample_x = grid_x - flow[..., 0]  # grid - flow (backward warp of forward flow)
        sample_y = grid_y - flow[..., 1]
        norm_x = 2.0 * sample_x / (width - 1) - 1.0  # -> [-1, 1], align_corners=True
        norm_y = 2.0 * sample_y / (height - 1) - 1.0
        grid = torch.stack((norm_x, norm_y), dim=-1)[None]  # (1, H, W, 2), (x, y) order
        sampled = grid_sample(
            image.float()[None, None],
            grid,
            mode="bilinear",
            padding_mode=padding_mode,
            align_corners=True,
        )
        return sampled[0, 0].round().clamp(0, 255).to(torch.uint8)

    @jaxtyped(typechecker=beartype)
    def score(self, prev: FrameType, curr: FrameType, flow: FlowType) -> FlowMetrics:
        """Metrics comparing `prev` warped by `flow` against `curr`."""
        with torch.no_grad():
            pred = self.warp(prev, flow, padding_mode=self.padding_mode)
            a = pred.float()[None, None]  # (1, 1, H, W)
            b = curr.float()[None, None]
            psnr = float(peak_signal_noise_ratio(a, b, data_range=self.data_range))
            ssim_out = structural_similarity_index_measure(
                a, b, data_range=self.data_range
            )
            ssim = float(cast("Tensor", ssim_out))  # SSIM returns a scalar Tensor here
            mse = float(mean_squared_error(a.flatten(), b.flatten()))
            mae = float(mean_absolute_error(a.flatten(), b.flatten()))
            lpips = self._lpips_value(pred, curr) if self.use_lpips else None
        return FlowMetrics(psnr=psnr, ssim=ssim, mse=mse, mae=mae, lpips=lpips)

    def _lpips_value(self, pred: Tensor, curr: Tensor) -> float:
        # Lazy import: keep torchvision (the LPIPS backbone) off the default,
        # no-LPIPS path. Weights download on first construction.
        if self._lpips_metric is None:
            from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

            self._lpips_metric = LearnedPerceptualImagePatchSimilarity(
                net_type=self.lpips_net, normalize=True
            ).eval()
        metric = self._lpips_metric.to(pred.device)
        a = (pred.float() / 255.0)[None, None].repeat(
            1, 3, 1, 1
        )  # (1, 3, H, W) in [0,1]
        b = (curr.float() / 255.0)[None, None].repeat(1, 3, 1, 1)
        return float(metric(a, b))


class MetricsAccumulator:
    """Streaming mean of `FlowMetrics` over a sequence, one pair at a time.

    A sequence of `N` frames yields `N - 1` flows; `add` each pair's metrics and
    read the running `mean`. `lpips` is averaged only over the pairs that carry
    it (`None` if none do).
    """

    def __init__(self) -> None:
        self._psnr = 0.0
        self._ssim = 0.0
        self._mse = 0.0
        self._mae = 0.0
        self._lpips = 0.0
        self._count = 0
        self._lpips_count = 0

    def add(self, metrics: FlowMetrics) -> None:
        """Fold one pair's metrics into the running mean."""
        self._psnr += metrics.psnr
        self._ssim += metrics.ssim
        self._mse += metrics.mse
        self._mae += metrics.mae
        if metrics.lpips is not None:
            self._lpips += metrics.lpips
            self._lpips_count += 1
        self._count += 1

    def mean(self) -> FlowMetrics:
        """The mean metrics so far.

        Raises:
            ValueError: If no metrics have been added yet.
        """
        if self._count == 0:
            msg = "no metrics accumulated; call add() at least once before mean()"
            raise ValueError(msg)
        n = self._count
        lpips = self._lpips / self._lpips_count if self._lpips_count else None
        return FlowMetrics(
            psnr=self._psnr / n,
            ssim=self._ssim / n,
            mse=self._mse / n,
            mae=self._mae / n,
            lpips=lpips,
        )

    @property
    def count(self) -> int:
        """How many pairs have been accumulated."""
        return self._count
