from __future__ import annotations

__all__ = (
    "Kernel",
    "KernelShape",
    "MedianKernel",
    "MedianParams",
    "Radius",
    "RadiusLike",
)

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, override

import torch
from beartype import beartype
from jaxtyping import Float32, jaxtyped
from torch import Tensor
from torch.nn.functional import pad

KernelShape = Literal["ellipsoid", "cuboid"]

# The stored form is always the triple; `RadiusLike` is what a caller may write.
Radius = tuple[int, int, int]
RadiusLike = int | tuple[int, int] | Radius

FrameType = Float32[Tensor, "H W"]
WindowType = Float32[Tensor, "T H W"]


def _normalize_radius(radius: RadiusLike) -> Radius:
    """Expand `radius` to the `(rx, ry, rz)` every kernel stores.

    The two-value form exists because the in-plane axes are almost always equal
    while `rz` is not free to follow them: it counts frames, so it tracks the
    frame rate rather than the spatial extent. Writing `(2, 5)` says that
    directly, where `(2, 2, 5)` leaves a reader checking whether the repetition
    was meant.

    Args:
        radius: `r` for every axis, `(r_spatial, r_temporal)` to set the two
            in-plane axes together, or an explicit `(rx, ry, rz)`.

    Returns:
        The half-extent per axis, in `(rx, ry, rz)` order.

    Raises:
        ValueError: If `radius` is none of those three forms.
    """
    match radius:
        case int():
            return radius, radius, radius
        case (spatial, temporal):
            return spatial, spatial, temporal
        case (rx, ry, rz):
            return rx, ry, rz
        case _:
            msg = f"invalid radius {radius!r}: expected r, (r_xy, r_z), or (rx, ry, rz)"
            raise ValueError(msg)


class Kernel(ABC):
    """The neighbourhood a 3D filter reads, and what it reduces it to.

    Holds the sampling geometry only, never frames, so one kernel serves any
    number of sequences and `FilteredSequence` owns the reading and buffering.

    Out-of-range neighbours -- past a sequence end in time, past an edge in
    space -- are **dropped, not padded**, in every subclass. A pixel near a
    border is therefore reduced over fewer samples, and each subclass says what
    that means for its own reduction.

    `FilteredSequence` is written against this type rather than a concrete
    kernel, so a new reduction is written here and leaves the reading, the
    buffering, and the window arithmetic untouched.

    Args:
        radius: half-extent per axis, so an axis spans `2r + 1` samples and `0`
            disables it. Written as `r`, `(r_spatial, r_temporal)`, or an
            explicit `(rx, ry, rz)`; stored normalized to the triple. Subclasses
            may derive it rather than take it directly.

    Raises:
        ValueError: If `radius` is not one of those forms, or any axis is
            negative.
    """

    def __init__(self, radius: RadiusLike) -> None:
        radii = _normalize_radius(radius)
        if min(radii) < 0:
            msg = f"negative radius {radii}: each axis needs 0 or more (0 disables it)"
            raise ValueError(msg)

        self.radius = radii

    @property
    def spatial_radius(self) -> tuple[int, int]:
        """The in-plane half-extent, `(rx, ry)`.

        A kernel pads for these itself, so they never reach `FilteredSequence`
        -- which is the asymmetry with `temporal_radius`, not an oversight.
        """
        rx, ry, _ = self.radius
        return rx, ry

    @property
    def temporal_radius(self) -> int:
        """How many frames either side of the centre a window must carry.

        The time-axis half-extent, in frames -- `rz` here, but a subclass
        deriving its footprint reports whatever its own reduction needs.
        `FilteredSequence` reads this alone to size a window, which is why it
        is the one axis the base class promises.
        """
        return self.radius[2]

    @abstractmethod
    def apply(self, window: WindowType, center: int) -> FrameType:
        """Reduce the neighbourhood of each pixel of frame `center` in `window`.

        A pure function of its arguments, so a caller holding a whole sequence
        gets exactly what the streaming pass would produce for the same frame.

        Args:
            window: `(T, H, W)` consecutive float32 frames.
            center: index in `window` of the frame to filter.

        Returns:
            The `(H, W)` filtered frame.

        Raises:
            ValueError: If `center` is not an index into `window`.
        """

    def _validate_center(self, window: Tensor, center: int) -> None:
        """Raise if `center` does not index a frame of `window`."""
        frames = window.shape[0]
        if not 0 <= center < frames:
            msg = f"center {center} is not an index into a {frames}-frame window"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class MedianParams:
    """The arguments a `MedianKernel` is built from, as one value.

    Separate from the kernel so a config file, a CLI, or the cache sidecar can
    carry the settings without holding a live object -- and so what a later run
    reconstructs is exactly what was recorded.

    `radius` is normalized to the `(rx, ry, rz)` triple on construction, so two
    records describing one kernel compare equal however each was written. That
    matters here and not on the kernel: this is the value a cache is looked up
    by, and `MedianParams(2) != MedianParams((2, 2, 2))` would miss a cache the
    same settings had already built. Whether the axes make a usable kernel is
    still the kernel's to say, and is raised by `build`.

    Attributes:
        radius: the normalized `(rx, ry, rz)`, whichever form was passed.
        shape: the footprint the offsets are drawn from.
    """

    radius: RadiusLike
    shape: KernelShape = "ellipsoid"

    def __post_init__(self) -> None:
        object.__setattr__(self, "radius", _normalize_radius(self.radius))

    def build(self) -> MedianKernel:
        """Construct the kernel these describe.

        Raises:
            ValueError: If any axis of `radius` is negative.
        """
        return MedianKernel(self.radius, shape=self.shape)


class MedianKernel(Kernel):
    """A 3D median over a discrete neighbourhood, robust to isolated spikes.

    Dropping out-of-range neighbours shortens the sample list rather than
    biasing it, and with an even number left the median averages the middle two
    -- which is why `torch.median`, returning the lower, cannot serve here.

    Args:
        radius: half-extent per axis; `0` disables that axis. Left required
            because there is no safe default: `rz` counts frames but damage
            tracks the time a window spans, so it has to follow the frame rate
            rather than a constant -- which is also why `(r_spatial,
            r_temporal)` is usually the form to reach for over a bare `r`.
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
    def offsets(self) -> tuple[Radius, ...]:
        """The `(dx, dy, dz)` offsets sampled at each pixel, the centre included."""
        return self._offsets

    def _build_offsets(self) -> tuple[Radius, ...]:
        """Enumerate the offsets `shape` admits, in scan order.

        An axis with radius `0` contributes only `0`, disabling it. `ellipsoid`
        keeps those satisfying `(dx/rx)^2 + (dy/ry)^2 + (dz/rz)^2 <= 1`.
        """
        rx, ry, rz = self.radius

        offsets = []
        for dx in range(-rx, rx + 1):
            for dy in range(-ry, ry + 1):
                for dz in range(-rz, rz + 1):
                    axes = zip((dx, dy, dz), self.radius, strict=True)
                    inside = sum((d / r) ** 2 for d, r in axes if r) <= 1.0
                    if self.shape == "cuboid" or inside:
                        offsets.append((dx, dy, dz))

        return tuple(offsets)

    @jaxtyped(typechecker=beartype)
    @override
    def apply(self, window: WindowType, center: int) -> FrameType:
        """Take the median over each pixel's in-range neighbours.

        Args:
            window: `(T, H, W)` consecutive float32 frames.
            center: index in `window` of the frame to filter.

        Returns:
            The `(H, W)` filtered frame, each pixel the median of however many
            of its neighbours fell inside the window and the frame.

        Raises:
            ValueError: If `center` is not an index into `window`.
        """
        self._validate_center(window, center)
        _, height, width = window.shape
        rx, ry = self.spatial_radius

        # NaN marks "no sample here", so an edge offset drops out of the median
        # instead of contributing a padded value.
        padded = pad(window, (rx, rx, ry, ry), value=float("nan"))

        gathered = torch.stack(
            [
                padded[
                    center + dz, ry + dy : ry + dy + height, rx + dx : rx + dx + width
                ]
                for dx, dy, dz in self._offsets
                if 0 <= center + dz < window.shape[0]
            ]
        )

        # Sorting puts the NaNs last, so the valid samples occupy `[0, valid)`
        # and the median sits at `(valid - 1) // 2` and `valid // 2` -- the same
        # element for an odd count, the middle two to average for an even one.
        ordered = gathered.sort(dim=0).values
        valid = (~ordered.isnan()).sum(dim=0)  # >= 1: the centre offset never drops

        lower = ordered.gather(0, ((valid - 1) // 2).unsqueeze(0))
        upper = ordered.gather(0, (valid // 2).unsqueeze(0))

        return ((lower + upper) / 2).squeeze(0)
