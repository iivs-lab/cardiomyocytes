from __future__ import annotations

__all__ = ("FrameSequence",)

import math
from functools import cached_property, partial
from typing import TYPE_CHECKING, Any, override

import numpy as np
import torch
from kaparoo.data.sequences import DataSequence, TransformedSequence
from torch import Tensor

from iivs_cardio.common.device import resolve_device
from iivs_cardio.data.transforms.filtering import FilteredSequence

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

    from iivs_cardio.data.transforms.filtering import FilterKernel, KernelParams

type NumPyRealDType = np.floating | np.integer
"""Any real numpy dtype. Complex is excluded: casting one drops its phase."""

type IndexLike = int | slice | Iterable[int] | None
"""What a caller may write to select frames: one, a span, a set, or None for all."""


def _as_tensor(frame: NDArray[Any], device: torch.device) -> Tensor:
    """Read one frame as float32 on `device`, the form filtering also returns."""
    return torch.from_numpy(frame).to(device, torch.float32)


def _finite_range(frame: Tensor) -> tuple[float, float] | None:
    """The `(min, max)` of `frame`'s finite values, or None when it has none.

    Non-finite values are dropped rather than propagated, so a masked frame's NaN
    background does not swallow the range. `torch.isfinite` is used over numpy's:
    a `Tensor`'s `size` is a method, so the emptiness test upstream libraries
    write as `finite.size == 0` is never true here.
    """
    finite = frame[torch.isfinite(frame)]
    if finite.numel() == 0:
        return None
    return float(finite.min()), float(finite.max())


class FrameSequence[M, T: NumPyRealDType = np.float32](DataSequence[Tensor, M]):
    """Frame access over one source, raw or filtered, always as float32 tensors.

    `kernel` chooses the view that stands in front of `source`: a `FilteredSequence`
    when given, a plain conversion when not. Both are `DataSequence[Tensor, M]`, so
    reads take one path and the consumer never sees which.

    Type Parameters:
        M: the source's per-frame metadata, which neither view changes.
        T: the source's numpy dtype; frames are read as float32 whatever it is.

    Args:
        source: The frames to read, already open.
        kernel: The neighbourhood to filter with, or None to pass frames through.
        device: Where frames are placed, and where filtering runs.

    Raises:
        ValueError: If `device` names an unsupported device kind.
    """

    def __init__(
        self,
        source: DataSequence[NDArray[T], M],
        kernel: FilterKernel | None = None,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = resolve_device(device)
        self._source: DataSequence[Tensor, M]

        if kernel is None:
            transform = partial(_as_tensor, device=self.device)
            self._source = TransformedSequence(source, transform)
        else:
            self._source = FilteredSequence(source, kernel, device=self.device)

    @classmethod
    def from_params(
        cls,
        source: DataSequence[NDArray[T], M],
        params: KernelParams | None = None,
        *,
        device: str | torch.device = "cpu",
    ) -> FrameSequence[M, T]:
        """Read `source`, filtered by the kernel `params` describes when there is one.

        `params` is optional where `FilteredSequence.from_params` requires it, since
        a filter is a config group a run may leave out.

        Args:
            source: The frames to read, already open.
            params: Which kernel to filter with, or None to pass frames through.
            device: Where frames are placed, and where filtering runs.
        """
        kernel = None if params is None else params.build()
        return cls(source, kernel, device=device)

    @property
    def source(self) -> DataSequence[Tensor, M]:
        """The view this reads from, filtering or converting the frames given."""
        return self._source

    def __len__(self) -> int:
        return len(self._source)

    @override
    def get_item(self, index: int) -> Tensor:
        """Return frame `index` as a float32 tensor on `device`."""
        return self._source.get_item(index)

    @override
    def get_meta(self, index: int) -> M:
        """Return the source's metadata for `index`, which neither view changes."""
        return self._source.get_meta(index)

    def value_range(self, index: IndexLike = None) -> tuple[float, float]:
        """The `(min, max)` over the frames `index` names, ignoring non-finite values.

        Ranges over what this sequence yields, so a filtered sequence reports the
        filtered values. Only the whole-sequence range is cached, since it is the
        one that has to read everything; a subset is recomputed each call.

        Args:
            index: One index, a `slice`, any iterable of indices (a `range`
                included), or None for the whole sequence. Negative indices count
                from the end.

        Returns:
            The lowest and highest finite value across the selected frames.

        Raises:
            IndexError: If an index is outside `[-len(self), len(self))`.
            ValueError: If the selection is empty, or holds no finite value.
        """
        if index is None:
            return self._global_value_range

        selected: Iterable[int]
        if isinstance(index, int):
            selected = (index,)
        elif isinstance(index, slice):
            selected = range(*index.indices(len(self)))
        else:
            selected = index

        return self._range_over(tuple(self._normalize_index(i) for i in selected))

    @cached_property
    def _global_value_range(self) -> tuple[float, float]:
        """The whole sequence's range, read once and kept for this instance."""
        return self._range_over(tuple(range(len(self))))

    def _range_over(self, indices: tuple[int, ...]) -> tuple[float, float]:
        """Fold `_finite_range` across `indices`, one frame held at a time.

        Raises:
            ValueError: If `indices` is empty, or no frame holds a finite value.
        """
        if not indices:
            msg = "value range is undefined for an empty selection"
            raise ValueError(msg)

        minimum, maximum = math.inf, -math.inf
        for index in indices:
            found = _finite_range(self.get_item(index))
            if found is not None:
                minimum = min(minimum, found[0])
                maximum = max(maximum, found[1])

        if minimum > maximum:  # nothing contributed, so the bounds never crossed
            msg = "value range is undefined: the selection holds no finite value"
            raise ValueError(msg)
        return minimum, maximum
