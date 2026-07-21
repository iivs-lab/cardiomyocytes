from __future__ import annotations

__all__ = ("WarpConsistency", "warp_consistency")

from typing import cast

import torch
from beartype import beartype
from jaxtyping import Float32, Real, jaxtyped
from torch import Tensor, nn
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

from iivs_cardio.common.warp import BackwardWarp, PaddingMode, backward_warp

FrameType = Real[Tensor, "*dim H W"]
FlowType = Float32[Tensor, "*dim 2 H W"]


def _resolve_data_range(frame: Tensor, data_range: float | None) -> float:
    """The PSNR/SSIM value range: `data_range` if given, else `frame`'s dtype range.

    A float dtype has no intrinsic range, and guessing one silently corrupts both
    metrics, so float frames must state it explicitly.
    """
    if data_range is not None:
        return data_range

    if frame.dtype.is_floating_point:
        msg = "data_range cannot be inferred from a float dtype; pass it explicitly"
        raise ValueError(msg)

    info = torch.iinfo(frame.dtype)
    return float(info.max - info.min)


def _metrics(
    frame1: Tensor, frame2: Tensor, data_range: float | None, *, reduce: bool
) -> dict[str, Tensor]:
    # Always score per sample, then optionally average, so that `reduce=True` is
    # exactly the mean of `reduce=False` for every metric -- including PSNR, whose
    # pooled form (one log over the batch's total error) is a different quantity.
    data_range = _resolve_data_range(frame1, data_range)
    *dim, height, width = frame1.shape

    frame1 = frame1.reshape(-1, 1, height, width).float()
    frame2 = frame2.reshape(-1, 1, height, width).float()

    keepdims = (1, 2, 3)  # channel, height, width
    residual = frame1 - frame2
    mse = residual.square().mean(dim=keepdims)
    mae = residual.abs().mean(dim=keepdims)

    ssim = structural_similarity_index_measure(
        frame1, frame2, data_range=data_range, reduction="none"
    )
    psnr = peak_signal_noise_ratio(
        frame1, frame2, data_range=data_range, reduction="none", dim=keepdims
    )

    per_sample = {
        "ssim": cast("Tensor", ssim),
        "psnr": psnr,
        "mse": mse,
        "mae": mae,
    }

    if reduce:
        return {name: value.mean() for name, value in per_sample.items()}

    return {name: value.reshape(dim) for name, value in per_sample.items()}


@jaxtyped(typechecker=beartype)
def warp_consistency(
    frame1: FrameType,
    frame2: FrameType,
    flow: FlowType,
    *,
    data_range: float | None = None,
    padding_mode: PaddingMode = "border",
    reduce: bool = True,
) -> dict[str, Tensor]:
    """Warp-consistency metrics of `flow`: warp `frame2` back, score it on `frame1`.

    The standard proxy when there is no ground-truth flow. Returns `{"ssim",
    "psnr", "mse", "mae"}` on the frames' device; a perfect match gives mse/mae 0,
    ssim 1, psnr inf.

    **Direction.** `flow` is the forward flow `frame1 -> frame2`, so it is defined
    on `frame1`'s grid: the material point at `x` in `frame1` sits at `x +
    flow(x)` in `frame2`. Sampling `frame2` there reconstructs `frame1`
    **exactly** -- the output grid *is* the grid the flow is defined on, so no
    inverse is needed. Going the other way, reconstructing `frame2` from
    `frame1`, requires inverting the map, and `x - flow(x)` only approximates it,
    with an error growing as `|flow| * |grad flow|`.

    The two agree exactly under a uniform translation, which makes that case
    useless for telling them apart -- and is how a warp of the wrong *sign*
    survives review. Test the direction with a non-uniform flow.

    Gradients reach `flow` for float frames, so this doubles as a photometric
    training loss -- also the form the unsupervised-flow literature uses. Integer
    frames break the graph (the warp rounds and clamps them back to their dtype),
    so training must use float frames.

    Args:
        frame1: `(*dim, H, W)` frame(s) to score against, any real dtype.
        frame2: `(*dim, H, W)` frame(s) to warp back onto `frame1`.
        flow: `(*dim, 2, H, W)` float32 forward flow `frame1 -> frame2`.
        data_range: PSNR/SSIM value range; inferred from the frame dtype when
            omitted, required for float frames.
        padding_mode: `grid_sample` out-of-bounds policy. Sampling at
            `grid + flow` leaves the frame wherever the flow diverges, so this
            decides what those pixels contribute.
        reduce: average over the batch to a 0-d scalar per metric. `False` keeps
            one score per pair, shaped `(*dim)`.
    """
    # `backward_warp` samples at `grid - transform`, so the transform that samples
    # *along* the flow -- at `grid + flow` -- is its negation.
    warped = backward_warp(frame2, -flow, padding_mode=padding_mode)
    return _metrics(warped, frame1, data_range, reduce=reduce)


class WarpConsistency(nn.Module):
    """Warp-consistency scoring with a cached warp grid.

    `forward(frame1, frame2, flow)` takes two `(*dim, H, W)` frames of any real
    dtype and a `(*dim, 2, H, W)` float32 forward flow `frame1 -> frame2`, samples
    `frame2` at `grid + flow` to reconstruct `frame1`, and scores the two,
    returning `{"ssim", "psnr", "mse", "mae"}` on the frames' device. See
    `warp_consistency` for why that direction and not the reverse. The warp grid
    is built once and reused across same-size calls, so scoring a fixed-size
    sequence skips the rebuild.

    Args:
        data_range: PSNR/SSIM value range; inferred from the frame dtype when
            omitted, required for float frames.
        padding_mode: `grid_sample` out-of-bounds policy for the warp.
        reduce: average over the batch to a 0-d scalar per metric. `False` keeps
            one score per pair, shaped `(*dim)`.
    """

    def __init__(
        self,
        *,
        data_range: float | None = None,
        padding_mode: PaddingMode = "border",
        reduce: bool = True,
    ) -> None:
        super().__init__()
        self.data_range = data_range
        self.reduce = reduce
        self._warp = BackwardWarp(padding_mode=padding_mode)

    def forward(
        self, frame1: Tensor, frame2: Tensor, flow: Tensor
    ) -> dict[str, Tensor]:
        """Return the warp-consistency metrics of `flow`, reusing the cached grid."""
        warped = self._warp(frame2, -flow)  # sample along the flow: `grid + flow`
        return _metrics(warped, frame1, self.data_range, reduce=self.reduce)
