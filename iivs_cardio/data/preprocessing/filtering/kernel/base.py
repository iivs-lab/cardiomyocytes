from __future__ import annotations

__all__ = ("Kernel", "KernelParams", "RadiusLike", "RadiusType")

from abc import ABC, abstractmethod

from jaxtyping import Float32
from torch import Tensor

# The stored form is always the triple; `RadiusLike` is what a caller may write.
RadiusType = tuple[int, int, int]
RadiusLike = int | tuple[int, int] | RadiusType

FrameType = Float32[Tensor, "H W"]
WindowType = Float32[Tensor, "T H W"]


def _normalize_triple[T](
    value: T | tuple[T, T] | tuple[T, T, T], *, name: str
) -> tuple[T, T, T]:
    """Expand a scalar / `(spatial, temporal)` / `(x, y, z)` to the `(x, y, z)` triple.

    Shape only -- element validation is the caller's, since a radius and a sigma
    admit different values. Any two- or three-element sequence is accepted, so
    the lists a config parser produces pass as readily as tuples.

    Raises:
        ValueError: If `value` is none of the three shapes.
    """
    if isinstance(value, tuple | list):
        match list(value):
            case [x, y, z]:
                return x, y, z
            case [spatial, temporal]:
                return spatial, spatial, temporal
    elif not isinstance(value, str):
        return value, value, value

    msg = f"invalid {name} {value!r}: expected a scalar, (spatial, temporal), or (x, y, z)"
    raise ValueError(msg)


def _normalize_radius(radius: RadiusLike) -> RadiusType:
    """Expand `radius` to the validated `(rx, ry, rz)` every kernel stores.

    The two-value form is usually the one to reach for: the in-plane axes are
    almost always equal, while `rz` is not free to follow them because it counts
    frames and so tracks the frame rate.

    Args:
        radius: `r` for every axis, `(r_spatial, r_temporal)` to set the two
            in-plane axes together, or an explicit `(rx, ry, rz)`.

    Returns:
        The half-extent per axis, in `(rx, ry, rz)` order.

    Raises:
        ValueError: If `radius` is none of those shapes, holds a non-`int`, or
            holds a negative axis.
    """
    triple = _normalize_triple(radius, name="radius")
    if not all(isinstance(r, int) and not isinstance(r, bool) for r in triple):
        msg = f"invalid radius {triple}: each axis must be a whole number of samples"
        raise ValueError(msg)
    if any(r < 0 for r in triple):
        msg = f"negative radius {triple}: each axis needs 0 or more (0 disables it)"
        raise ValueError(msg)
    return triple


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
        ValueError: If `radius` is not one of those shapes, holds a non-`int`,
            or holds a negative axis.
    """

    def __init__(self, radius: RadiusLike) -> None:
        self.radius = _normalize_radius(radius)

    @property
    def spatial_radius(self) -> tuple[int, int]:
        return self.radius[:2]

    @property
    def temporal_radius(self) -> int:
        return self.radius[2]

    @abstractmethod
    def apply(self, window: WindowType, target: int) -> FrameType:
        """Reduce the neighbourhood of each pixel of frame `target` in `window`.

        A pure function of its arguments, so a caller holding a whole sequence
        gets exactly what the streaming pass would produce for the same frame.

        Args:
            window: `(T, H, W)` consecutive float32 frames.
            target: index in `window` of the frame to filter.

        Returns:
            The `(H, W)` filtered frame.

        Raises:
            ValueError: If `target` is not an index into `window`.
        """

    def _validate_target(self, window: Tensor, target: int) -> None:
        """Raise if `target` does not index a frame of `window`."""
        frames = window.shape[0]
        if not 0 <= target < frames:
            msg = f"target {target} is not an index into a {frames}-frame window"
            raise ValueError(msg)


class KernelParams(ABC):
    """A kernel's constructor arguments as one value, buildable into the kernel.

    Separate from `Kernel` so a config, a CLI, or the cache sidecar carries the
    settings without a live object -- and what a later run reconstructs is
    exactly what was recorded. Closed at the same family as `Kernel`; each
    concrete params is a plain frozen record that neither expands nor validates
    its fields, leaving the kernel the one place that interprets them.
    """

    @abstractmethod
    def build(self) -> Kernel:
        """Construct the kernel these describe."""
