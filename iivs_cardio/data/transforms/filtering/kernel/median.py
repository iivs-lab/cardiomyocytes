from __future__ import annotations

__all__ = ("KernelShape", "MedianConfig", "MedianKernel")

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, ClassVar, Final, Literal, get_args, override

import torch
from beartype import beartype
from jaxtyping import jaxtyped
from torch.nn.functional import pad

from iivs_cardio.data.transforms.filtering.kernel.base import (
    FilterKernel,
    FrameType,
    KernelConfig,
    RadiusLike,
    RadiusType,
    WindowType,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

KernelShape = Literal["ellipsoid", "cuboid"]
KERNEL_SHAPES: Final[tuple[KernelShape, ...]] = get_args(KernelShape)

# Where CUDA's `sort` and `topk` each leave a faster shared-memory path.
_SHARED_TIERS: Final = (32, 128)

# Stacked neighbourhood one pass may hold; every temporary scales with it.
_TILE_BYTES: Final = 32 << 20


def _prefers_topk(samples: int) -> bool:
    """Whether `topk` beats a full `sort`, which it does across a tier boundary."""

    def tier(count: int) -> int:
        return sum(count > bound for bound in _SHARED_TIERS)

    return tier(samples) > tier(samples // 2 + 1)


def _tile_rows(samples: int, width: int, itemsize: int) -> int:
    """How many rows of the frame one pass may take, at least one."""
    return max(1, _TILE_BYTES // (samples * width * itemsize))


class MedianKernel(FilterKernel):
    """A 3D median over a discrete neighbourhood, robust to isolated spikes.

    Dropping out-of-range neighbours shortens the sample list rather than
    biasing it, and with an even number left the median averages the middle two.
    That is why `torch.median`, which returns the lower, cannot serve here.

    Args:
        radius: half-extent per axis; `0` disables that axis. Left required
            because there is no safe default: `rz` counts frames but damage
            tracks the time a window spans, so it has to follow the frame rate
            rather than a constant. That is also why `(r_spatial, r_temporal)` is
            usually the form to reach for over a bare `r`.
        shape: `ellipsoid` weighs the axes against their radii together, taking
            33 offsets at radius `(2, 2, 2)`; `cuboid` takes the whole box, 125.

    Raises:
        ValueError: If any radius is negative, or `shape` is neither name.
    """

    def __init__(self, radius: RadiusLike, *, shape: KernelShape = "ellipsoid") -> None:
        if shape not in KERNEL_SHAPES:
            listed = ", ".join(repr(name) for name in KERNEL_SHAPES)
            msg = f"unsupported shape {shape!r}: expected {listed}"
            raise ValueError(msg)

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
        offsets = [o for o in self._offsets if 0 <= target + o[2] < frames]

        rx, ry = self.spatial_radius
        padded = pad(window, (rx, rx, ry, ry), value=float("nan"))

        rows = _tile_rows(len(offsets), width, window.element_size())
        if rows >= height:
            return self._median(padded, offsets, target, 0, height, width)

        filtered = torch.empty_like(window[0])
        for start in range(0, height, rows):
            stop = min(start + rows, height)
            filtered[start:stop] = self._median(
                padded, offsets, target, start, stop - start, width
            )

        return filtered

    def _median(
        self,
        padded: Tensor,
        offsets: Sequence[RadiusType],
        target: int,
        start: int,
        rows: int,
        width: int,
    ) -> Tensor:
        """Take the median for `rows` rows of the frame, beginning at `start`."""
        rx, ry = self.spatial_radius

        gathered = torch.stack(
            [
                padded[
                    target + dz,
                    ry + dy + start : ry + dy + start + rows,
                    rx + dx : rx + dx + width,
                ]
                for dx, dy, dz in offsets
            ]
        )

        samples = gathered.shape[0]
        if gathered.is_cuda and _prefers_topk(samples):
            ordered = gathered.topk(samples // 2 + 1, dim=0, largest=False).values
        else:
            ordered = gathered.sort(dim=0).values

        valid = (~gathered.isnan()).sum(dim=0)
        pair = ordered.gather(0, torch.stack(((valid - 1) // 2, valid // 2)))

        return (pair[0] + pair[1]) / 2


@dataclass(frozen=True, slots=True)
class MedianConfig(KernelConfig):
    """The settings of a `MedianKernel`, as one recordable value.

    Attributes:
        kind: what a record says this filter was.
        radius: half-extent per axis, in samples.
        shape: which neighbours inside that extent are read.
    """

    kind: ClassVar[str] = "median"

    radius: RadiusLike
    shape: KernelShape = "ellipsoid"

    @override
    def build(self) -> MedianKernel:
        """Build the kernel these settings describe."""
        return MedianKernel(self.radius, shape=self.shape)
