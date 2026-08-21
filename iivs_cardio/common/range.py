from __future__ import annotations

__all__ = ("all_finite", "finite_range")

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def all_finite(frame: Tensor) -> bool:
    """Test whether every value in `frame` is finite, in one fused pass.

    `aminmax` reads the frame once and betrays a non-finite value on the way,
    the same way `finite_range` uses it: NaN propagates through both bounds and
    an infinity lands on the one it belongs to. `isfinite().all()` reaches the
    same answer through a mask the size of the frame, which on a hot path is
    the allocation this avoids. An empty frame holds nothing to be non-finite,
    and `aminmax` has no bounds to take from one.
    """
    if frame.numel() == 0:
        return True

    low, high = torch.aminmax(frame)

    return bool(low.isfinite() and high.isfinite())


def finite_range(frame: Tensor) -> tuple[float, float] | None:
    """Return the `(min, max)` of `frame`'s finite values, or `None` if it has none.

    Non-finite values are ignored rather than propagated. That tolerance is this
    function's, not the pipeline's, which refuses a phase frame holding one on
    the way in, a NaN there meaning the frame came out broken. An empty frame and
    an entirely non-finite one both answer `None`: whether that is a frame to
    skip or a run to refuse is the caller's to decide, which is why it is not an
    exception.

    The bounds come back as Python floats, so an integer frame holding values a
    float cannot tell apart answers with them rounded together. Phase frames are
    float32 by the time they arrive here, which is what makes that no one's
    problem in practice rather than a check worth paying for per frame.
    """
    if frame.numel() == 0:
        return None

    # One fused pass, which also betrays a non-finite value: NaN propagates
    # through both bounds and an infinity lands on the one it belongs to. Only
    # then is the mask worth its second allocation and compacting copy.
    low, high = torch.aminmax(frame)
    if low.isfinite() and high.isfinite():
        return float(low), float(high)

    finite = frame[torch.isfinite(frame)]
    if finite.numel() == 0:  # `.numel()`: a Tensor's `size` is a method, never 0
        return None
    return float(finite.min()), float(finite.max())
