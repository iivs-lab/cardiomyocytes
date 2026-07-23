from __future__ import annotations

__all__ = ("KernelShape", "MedianKernel", "MedianParams")

from dataclasses import dataclass
from itertools import product
from typing import Literal, override

import torch
from beartype import beartype
from jaxtyping import jaxtyped
from torch.nn.functional import pad

from iivs_cardio.data.transforms.filtering.kernel.base import (
    FilterKernel,
    FrameType,
    KernelParams,
    RadiusLike,
    RadiusType,
    WindowType,
)

KernelShape = Literal["ellipsoid", "cuboid"]

# Sample counts where CUDA's `topk` beats its `sort`. Both leave the same
# shared-memory path once they must order more than 32 elements, and both step
# about 3x when they do -- `sort` orders every sample, so it steps at 33, while
# `topk` orders `samples // 2 + 1`, so it steps at 64. Between those two steps
# only `sort` is paying. Measured on one GPU, but both bounds turn on the same
# threshold, which is why this is a range rather than a fitted cutoff.
_CUDA_TOPK_SAMPLES = range(33, 64)


class MedianKernel(FilterKernel):
    """A 3D median over a discrete neighbourhood, robust to isolated spikes.

    Dropping out-of-range neighbours shortens the sample list rather than
    biasing it, and with an even number left the median averages the middle two
    -- which is why `torch.median`, returning the lower, cannot serve here.

    Args:
        radius: half-extent per axis; `0` disables that axis. Left required
            because there is no safe default: `rz` counts frames but damage
            tracks the time a window spans, so it has to follow the frame rate
            rather than a constant -- which is also why `(r_spatial, r_temporal)`
            is usually the form to reach for over a bare `r`.
        shape: `ellipsoid` weighs the axes against their radii together, taking
            33 offsets at radius `(2, 2, 2)`; `cuboid` takes the whole box, 125.

    Raises:
        ValueError: If any radius is negative.
    """

    def __init__(self, radius: RadiusLike, *, shape: KernelShape = "ellipsoid") -> None:
        super().__init__(radius)
        self.shape = shape
        self._offsets = self._build_offsets()

    @property
    def offsets(self) -> tuple[RadiusType, ...]:
        """The `(dx, dy, dz)` offsets sampled at each pixel, the centre included."""
        return self._offsets

    def _build_offsets(self) -> tuple[RadiusType, ...]:
        """Enumerate the offsets `shape` admits, in scan order.

        An axis with radius `0` contributes only `0`, disabling it. `ellipsoid`
        keeps those satisfying `(dx/rx)^2 + (dy/ry)^2 + (dz/rz)^2 <= 1`.
        """
        rx, ry, rz = self.radius
        box = product(range(-rx, rx + 1), range(-ry, ry + 1), range(-rz, rz + 1))

        if self.shape == "cuboid":
            return tuple(box)

        def inside(offset: RadiusType) -> bool:
            axes = zip(offset, self.radius, strict=True)
            return sum((d / r) ** 2 for d, r in axes if r) <= 1.0

        return tuple(filter(inside, box))

    @jaxtyped(typechecker=beartype)
    @override
    def apply(self, window: WindowType, target: int) -> FrameType:
        """Take the median over each pixel's in-range neighbours.

        Args:
            window: `(T, H, W)` consecutive float32 frames.
            target: index in `window` of the frame to filter.

        Returns:
            The `(H, W)` filtered frame, each pixel the median of however many
            of its neighbours fell inside the window and the frame.

        Raises:
            ValueError: If `target` is not an index into `window`.
        """
        self._validate_target(window, target)
        frames, height, width = window.shape
        rx, ry = self.spatial_radius

        padded = pad(window, (rx, rx, ry, ry), value=float("nan"))

        gathered = torch.stack(
            [
                padded[
                    target + dz, ry + dy : ry + dy + height, rx + dx : rx + dx + width
                ]
                for dx, dy, dz in self._offsets
                if 0 <= target + dz < frames
            ]
        )

        # Only the two central order statistics are read, so the lower half is
        # enough: however many samples are valid, neither rank reaches past
        # `samples // 2`. `topk` supplies it more cheaply than a full sort
        # except on CUDA outside `_CUDA_TOPK_SAMPLES`.
        samples = gathered.shape[0]
        if gathered.is_cuda and samples not in _CUDA_TOPK_SAMPLES:
            ordered = gathered.sort(dim=0).values
        else:
            ordered = gathered.topk(
                samples // 2 + 1, dim=0, largest=False, sorted=True
            ).values

        # NaNs order last either way, so the valid samples occupy `[0, valid)`
        # and the median is one element for an odd count, the middle two to
        # average for an even one -- which is why `torch.median` cannot serve.
        # An odd count needs no case of its own: the two ranks coincide, and
        # halving the sample added to itself only moves the exponent, so it
        # comes back bit-for-bit.
        valid = (~gathered.isnan()).sum(dim=0)  # >= 1: the centre never drops
        pair = ordered.gather(0, torch.stack(((valid - 1) // 2, valid // 2)))

        return (pair[0] + pair[1]) / 2


@dataclass(frozen=True, slots=True)
class MedianParams(KernelParams):
    radius: RadiusLike
    shape: KernelShape = "ellipsoid"

    @override
    def build(self) -> MedianKernel:
        return MedianKernel(self.radius, shape=self.shape)
