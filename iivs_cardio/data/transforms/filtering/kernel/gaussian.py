from __future__ import annotations

__all__ = ("GaussianConfig", "GaussianKernel", "SigmaLike", "SigmaType")

from dataclasses import dataclass
from typing import ClassVar, override

import torch
from beartype import beartype
from jaxtyping import jaxtyped
from torch import Tensor
from torch.nn.functional import conv2d

from iivs_cardio.data.transforms.filtering.kernel.base import (
    FilterKernel,
    FrameType,
    KernelConfig,
    WindowType,
    _normalize_triple,
)

# The stored form is always the triple; `SigmaLike` is what a caller may write.
# `int` is accepted where `float` is annotated and coerced on the way in.
SigmaType = tuple[float, float, float]
SigmaLike = float | tuple[float, float] | SigmaType


def _normalize_sigma(sigma: SigmaLike) -> SigmaType:
    """Expand `sigma` to the validated `(sx, sy, sz)` of standard deviations.

    Mirrors a radius in shape, taking a scalar, `(s_spatial, s_temporal)`, or the
    full triple. The axes carry a width rather than a sample count, so an `int`
    is coerced to `float` rather than required.

    Raises:
        ValueError: If `sigma` is none of those shapes, holds a non-number, or
            holds a negative axis.
    """
    triple = _normalize_triple(sigma, name="sigma")
    if not all(isinstance(s, int | float) and not isinstance(s, bool) for s in triple):
        msg = f"invalid sigma {triple}: each axis must be a number"
        raise ValueError(msg)

    values = (float(triple[0]), float(triple[1]), float(triple[2]))
    if any(s < 0 for s in values):
        msg = f"negative sigma {values}: each axis needs 0 or more (0 disables it)"
        raise ValueError(msg)
    return values


class GaussianKernel(FilterKernel):
    """A separable 3D Gaussian, renormalized over whatever neighbours survive.

    Dropping out-of-range neighbours would darken every border, so the weights
    that landed inside are summed and divided out, giving a weighted mean over
    the surviving support rather than a partial sum. Because the weights are
    separable and that division happens once at the end, the result equals the
    full 3D normalized Gaussian exactly, not a per-axis approximation of it.

    Where a `MedianKernel` deletes an isolated spike, this spreads it across the
    neighbourhood; the two are not interchangeable.

    Args:
        sigma: standard deviation per axis, in samples; `0` disables that axis.
            Written as `s`, `(s_spatial, s_temporal)`, or `(sx, sy, sz)`, like a
            radius, and for the same reason the two-value form is usual, since
            `sz` spans frames and tracks the frame rate.
        truncate: how many standard deviations the window spans, so each radius
            is `int(truncate * sigma + 0.5)`, which is `scipy.ndimage`'s rule and
            name. It is distinct from the border policy, which is always to drop.

    Raises:
        ValueError: If any sigma is negative, or `truncate` is not positive.
    """

    def __init__(self, sigma: SigmaLike, *, truncate: float = 4.0) -> None:
        sigma = _normalize_sigma(sigma)
        if truncate <= 0:
            msg = f"truncate must be positive, got {truncate}"
            raise ValueError(msg)

        radius = (
            int(truncate * sigma[0] + 0.5),
            int(truncate * sigma[1] + 0.5),
            int(truncate * sigma[2] + 0.5),
        )
        super().__init__(radius)

        self.sigma = sigma
        self.truncate = truncate
        self._weights = tuple(self._build_weights(axis) for axis in range(3))

    def weights(self, axis: int, device: torch.device | None = None) -> Tensor:
        """The `2r + 1` normalized weights along `axis` (`0`=x, `1`=y, `2`=z).

        A disabled axis (`sigma` `0`) yields the single weight `[1.0]`, so it
        contributes a pass-through. Precomputed on the CPU; moved to `device` on
        each call, which is cheap against the convolution it feeds.
        """
        return self._weights[axis].to(device)

    def _build_weights(self, axis: int) -> Tensor:
        """The normalized 1D Gaussian for `axis`, or `[1.0]` when it is disabled."""
        radius = self.radius[axis]
        if radius == 0:
            return torch.ones(1)

        sigma = self.sigma[axis]
        steps = torch.arange(-radius, radius + 1, dtype=torch.float32)
        weights = torch.exp(-steps.square() / (2.0 * sigma * sigma))
        return weights / weights.sum()

    @jaxtyped(typechecker=beartype)
    @override
    def apply(self, window: WindowType, target: int) -> FrameType:
        """Take the Gaussian-weighted mean over each pixel's in-range neighbours.

        Args:
            window: `(T, H, W)` consecutive float32 frames.
            target: index in `window` of the frame to filter.

        Returns:
            The `(H, W)` filtered frame, each pixel divided by the weight that
            actually reached it, so borders keep their brightness.

        Raises:
            ValueError: If `target` is not an index into `window`.
        """
        self._validate_target(window, target)
        frames = window.shape[0]
        rx, ry, rz = self.radius
        device = window.device

        # Time collapses first: which frames are in range is the same for every
        # pixel, so their weights fold into one plane and one scalar mass.
        offsets = [dz for dz in range(-rz, rz + 1) if 0 <= target + dz < frames]
        wz = self.weights(2, device)[[dz + rz for dz in offsets]]
        plane = (window[[target + dz for dz in offsets]] * wz[:, None, None]).sum(0)
        temporal_mass = wz.sum()

        # Space: blur the plane and a field of ones together, zero-padded so an
        # out-of-frame neighbour adds to neither. The weights are symmetric, so
        # cross-correlation is convolution.
        stacked = torch.stack((plane, torch.ones_like(plane)))[:, None]
        stacked = conv2d(
            stacked, self.weights(1, device).view(1, 1, -1, 1), padding=(ry, 0)
        )
        stacked = conv2d(
            stacked, self.weights(0, device).view(1, 1, 1, -1), padding=(0, rx)
        )
        numerator, spatial_mass = stacked[:, 0]

        # Divide once at the very end. The surviving weight at a corner is the
        # product of what survived on each axis, so dividing per axis would bias
        # exactly those pixels.
        return numerator / (spatial_mass * temporal_mass)


@dataclass(frozen=True, slots=True)
class GaussianConfig(KernelConfig):
    """The settings of a `GaussianKernel`, as one recordable value.

    Attributes:
        kind: what a record says this filter was.
        sigma: standard deviation per axis, in samples.
        truncate: how many standard deviations the window spans.
    """

    kind: ClassVar[str] = "gaussian"

    sigma: SigmaLike
    truncate: float = 4.0

    @override
    def build(self) -> GaussianKernel:
        """Build the kernel these settings describe."""
        return GaussianKernel(self.sigma, truncate=self.truncate)
