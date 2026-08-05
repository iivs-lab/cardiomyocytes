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
            window: `(T, H, W)` consecutive float32 frames.
            target: index in `window` of the frame to return.

        Returns:
            That frame, as a view rather than a copy.

        Raises:
            ValueError: If `target` is not an index into `window`.
        """
        self._validate_target(window, target)

        return window[target]


@dataclass(frozen=True, slots=True)
class IdentityConfig(KernelConfig):
    kind: ClassVar[str] = "identity"

    @override
    def build(self) -> IdentityKernel:
        return IdentityKernel()
