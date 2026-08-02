from __future__ import annotations

__all__ = ("finite_range",)

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def finite_range(frame: Tensor) -> tuple[float, float] | None:
    """The `(min, max)` of `frame`'s finite values, or None when it has none.

    Non-finite values are ignored rather than propagated, so a masked frame's NaN
    background does not swallow the range. An empty frame and an entirely
    non-finite one both answer `None`: whether that is a frame to skip or a run
    to refuse is the caller's to decide, which is why it is not an exception.
    """
    if frame.numel() == 0:
        return None

    # One fused pass, which also betrays a non-finite value: NaN propagates
    # through both bounds and an infinity lands on the one it belongs to. Only
    # then is the mask worth its second allocation and compacting copy -- at
    # 512x512, taking that route for every frame measured about six times this.
    low, high = torch.aminmax(frame)
    if low.isfinite() and high.isfinite():
        return float(low), float(high)

    finite = frame[torch.isfinite(frame)]
    if finite.numel() == 0:  # `.numel()`: a Tensor's `size` is a method, never 0
        return None
    return float(finite.min()), float(finite.max())
