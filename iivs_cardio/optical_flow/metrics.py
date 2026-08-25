from __future__ import annotations

__all__ = (
    "WarpConsistency",
    "flow_magnitude",
    "forward_backward_error",
    "identity_ssim",
    "warp_consistency",
)

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
    # exactly the mean of `reduce=False` for every metric, PSNR included, whose
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

    The standard proxy when there is no ground-truth flow. Returns
    `{"ssim", "psnr", "mse", "mae"}` on the frames' device; a perfect match
    gives mse/mae 0, ssim 1, psnr inf.

    **Direction.** `flow` is the forward flow `frame1 -> frame2`, so it is
    defined on `frame1`'s grid: the material point at `x` in `frame1` sits at
    `x + flow(x)` in `frame2`. Sampling `frame2` there reconstructs `frame1`
    **exactly**: the output grid *is* the grid the flow is defined on, so no
    inverse is needed. Going the other way, reconstructing `frame2` from
    `frame1`, requires inverting the map, and `x - flow(x)` only approximates it,
    with an error growing as `|flow| * |grad flow|`.

    Under a *uniform* flow the two coincide, `x - flow(x)` inverting a constant
    map exactly, so no amount of testing on a rigid translation can tell them
    apart. Pin this choice with a non-uniform flow instead. (A flipped *sign* is
    a different error, and a uniform translation does catch that one.)

    Gradients reach `flow` for float frames, so this doubles as a photometric
    training loss, which is also the form the unsupervised-flow literature
    uses. Integer
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
    warped = backward_warp(frame2, flow, padding_mode=padding_mode)
    return _metrics(warped, frame1, data_range, reduce=reduce)


@jaxtyped(typechecker=beartype)
def identity_ssim(
    frame1: FrameType,
    frame2: FrameType,
    *,
    data_range: float | None = None,
    reduce: bool = True,
) -> Tensor:
    """SSIM of a zero flow: the floor every real flow's score is read against.

    Inter-frame motion here is sub-pixel, so two consecutive frames are already
    nearly alike and a flow of exactly zero scores around 0.95. Raw SSIM
    therefore says almost nothing, and what a search compares is the gain above
    this.

    No warp is done. Sampling at `grid + 0` gives the frame back unchanged, so
    the floor is `frame2` scored against `frame1` as they stand.

    Args:
        frame1: `(*dim, H, W)` frame(s) to score against, any real dtype.
        frame2: `(*dim, H, W)` frame(s) that a flow would have been warped from.
        data_range: SSIM value range; inferred from the frame dtype when
            omitted, required for float frames.
        reduce: average over the batch to a 0-d scalar. `False` keeps one score
            per pair, shaped `(*dim)`.

    Returns:
        The floor, which is `1` where the two frames are identical. A duplicated
        frame reaches that exactly, so a caller folding these has a value that
        no gain can be earned above rather than one that is merely high.
    """
    return _metrics(frame2, frame1, data_range, reduce=reduce)["ssim"]


@jaxtyped(typechecker=beartype)
def forward_backward_error(
    forward: FlowType,
    backward: FlowType,
    *,
    padding_mode: PaddingMode = "border",
    reduce: bool = True,
) -> Tensor:
    """Mean `|f_fwd(x) + f_bwd(x + f_fwd(x))|` in pixels; `0` for a consistent flow.

    Following a correspondence forward and then back should return where it
    started, and the residual says how far it does not. This is what a flow that
    won its photometric score by fitting noise fails: the noise it latched onto
    is not a correspondence, so following it back lands somewhere else.

    It cannot be read alone either. **A zero flow scores a perfect `0`**, having
    nothing to be inconsistent about, so this and the gain over
    `identity_ssim` are read together or neither is worth reading.

    `backward_warp` samples at `grid + offset`, so warping the backward field by
    the forward one evaluates it exactly where the forward field claims the
    pixel went. The forward field is broadcast over the backward field's own two
    channels, which warp together.

    Args:
        forward: `(*dim, 2, H, W)` float32 flow `frame1 -> frame2`.
        backward: `(*dim, 2, H, W)` float32 flow `frame2 -> frame1`, which is
            the same pair the other way round rather than a neighbouring pair.
        padding_mode: `grid_sample` out-of-bounds policy for the warp.
        reduce: average over the batch to a 0-d scalar. `False` keeps one error
            per pair, shaped `(*dim)`.

    Returns:
        The error in pixels, which is the unit the flow itself is in.
    """
    lead = forward.shape[:-3]
    spread = forward.unsqueeze(-4).expand(*lead, 2, *forward.shape[-3:])
    residual = forward + backward_warp(backward, spread, padding_mode=padding_mode)

    return _reduce_field(residual, reduce=reduce)


@jaxtyped(typechecker=beartype)
def flow_magnitude(flow: FlowType, *, reduce: bool = True) -> Tensor:
    """Mean `|flow|` in pixels, which is how much motion was found.

    On its own this says only that something moved. What it is for is the
    spread of it across a sequence: a beating cell's displacement rises and
    falls, and a filter reaching too far through time flattens that while
    leaving the photometric score intact.

    Args:
        flow: `(*dim, 2, H, W)` float32 flow.
        reduce: average over the batch to a 0-d scalar. `False` keeps one
            magnitude per pair, shaped `(*dim)`.
    """
    return _reduce_field(flow, reduce=reduce)


def _reduce_field(field: Tensor, *, reduce: bool) -> Tensor:
    """Mean length of a `(*dim, 2, H, W)` vector field, per pair or over all."""
    lengths = field.square().sum(dim=-3).sqrt()
    per_pair = lengths.mean(dim=(-2, -1))

    return per_pair.mean() if reduce else per_pair


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

    Attributes:
        data_range: The value range PSNR and SSIM are scored against, or `None`
            to take it from the frame dtype.
        reduce: Whether each call averages over the batch.
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
        warped = self._warp(frame2, flow)
        return _metrics(warped, frame1, self.data_range, reduce=self.reduce)
