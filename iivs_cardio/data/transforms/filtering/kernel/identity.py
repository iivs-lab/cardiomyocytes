from __future__ import annotations

__all__ = ("IdentityConfig", "IdentityKernel")

from dataclasses import dataclass
from typing import ClassVar, override

from beartype import beartype
from jaxtyping import jaxtyped

from iivs_cardio.data.transforms.filtering.kernel.base import (
    FilterKernel,
    FrameType,
    KernelConfig,
    WindowType,
)


class IdentityKernel(FilterKernel):
    """The kernel that reduces a pixel to itself, so a sequence reads unfiltered.

    Exists so that "no filtering" is a kernel rather than a missing one: a
    caller, a config group, and `FilteredSequence` all take the same shape
    whether or not a run filters, and `None` stops meaning anything special.

    Its radius is `0` on every axis, so `FilteredSequence` reads a one-frame
    window and hands it straight back. That path costs one buffered read per
    frame and no copy, which is what the unfiltered case would cost anyway.
    """

    def __init__(self) -> None:
        super().__init__(0)

    @jaxtyped(typechecker=beartype)
    @override
    def apply(self, window: WindowType, target: int) -> FrameType:
        """Return frame `target` of `window` unchanged.

        Args:
            window: The `(T, H, W)` consecutive float32 frames to read.
            target: The index in `window` of the frame to return.

        Returns:
            That frame, copied out of `window` rather than viewed: no filtering
            still means a frame of the caller's own, since the one behind it is
            the buffer the next window is built from.

        Raises:
            ValueError: If `target` is not an index into `window`.
        """
        self._validate_target(window, target)

        return window[target].clone()


@dataclass(frozen=True, slots=True)
class IdentityConfig(KernelConfig):
    """The settings of an `IdentityKernel`, which has none of its own.

    Attributes:
        kind: The name a record gives this filter.
    """

    kind: ClassVar[str] = "identity"

    @override
    def build(self) -> IdentityKernel:
        """Build the kernel these settings describe."""
        return IdentityKernel()
