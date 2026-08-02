from __future__ import annotations

__all__ = ("FrameSequence",)

import math
from functools import cached_property
from typing import TYPE_CHECKING, override

import numpy as np
import torch
from kaparoo.data.sequences import DataSequence, SlicedSequence
from torch import Tensor

from iivs_cardio.data.transforms.filtering import FilteredSequence

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

    from iivs_cardio.common.device import Device, DeviceLike
    from iivs_cardio.data.transforms.filtering import FilterKernel, KernelParams

type NumPyRealDType = np.floating | np.integer
"""Any real numpy dtype. Complex is excluded: casting one drops its phase."""

type IndexLike = int | slice | Iterable[int] | None
"""What a caller may write to select frames: one, a span, a set, or None for all."""


def _finite_range(frame: Tensor) -> tuple[float, float] | None:
    """The `(min, max)` of `frame`'s finite values, or None when it has none.

    Non-finite values are dropped rather than propagated, so a masked frame's NaN
    background does not swallow the range. `torch.isfinite` is used over numpy's:
    a `Tensor`'s `size` is a method, so the emptiness test upstream libraries
    write as `finite.size == 0` is never true here.

    `aminmax` answers in one pass and betrays a non-finite value by returning
    one: NaN propagates through both bounds, and an infinity lands on the bound
    it belongs to. Only then is the mask worth building, which costs a second
    full-size allocation and a compacting copy -- measured at 512x512, taking
    that route for every frame is about six times the fused pass.
    """
    if frame.numel() == 0:
        return None

    low, high = torch.aminmax(frame)
    if low.isfinite() and high.isfinite():
        return float(low), float(high)

    finite = frame[torch.isfinite(frame)]
    if finite.numel() == 0:
        return None
    return float(finite.min()), float(finite.max())


class FrameSequence[M, T: NumPyRealDType = np.float32](DataSequence[Tensor, M]):
    """Frame access over one source, always as float32 tensors on one device.

    Every frame goes through a kernel, `IdentityKernel` included, so reading is
    one path whether or not a run filters and no caller carries a "no filter"
    case. `step` is applied *before* filtering: taking every Nth frame first is
    what makes a strided run measure the frame rate it claims to, where filtering
    first would fold the dropped frames into the kept ones.

    Striding renumbers the sequence -- `len` and every index, `get_meta`
    included, count kept frames -- so `self[1]` at `step=2` is the source's frame
    2. A selection `step` cannot express is written by slicing `source` first;
    the stride then applies to whatever is passed.

    Type Parameters:
        M: the source's per-frame metadata, which neither striding nor filtering
            changes.
        T: the source's numpy dtype; frames are read as float32 whatever it is.

    Args:
        source: The frames to read, already open.
        kernel: The neighbourhood to filter with; `IdentityKernel` to read the
            frames as stored.
        step: Keep every `step`-th frame, counting from the first.
        device: Where frames are placed, and where filtering runs.

    Raises:
        ValueError: If `step` is below 1, or `device` names an unsupported kind.
    """

    def __init__(
        self,
        source: DataSequence[NDArray[T], M],
        kernel: FilterKernel,
        *,
        step: int = 1,
        device: DeviceLike = "cpu",
    ) -> None:
        if step < 1:
            msg = f"invalid frame step {step}: expected 1 or more"
            raise ValueError(msg)

        kept = SlicedSequence(source, range(0, len(source), step))
        self._source = FilteredSequence(kept, kernel, device=device)

    @classmethod
    def from_params(
        cls,
        source: DataSequence[NDArray[T], M],
        params: KernelParams,
        *,
        step: int = 1,
        device: DeviceLike = "cpu",
    ) -> FrameSequence[M, T]:
        """Read `source` through the kernel `params` describes.

        Args:
            source: The frames to read, already open.
            params: Which kernel to filter with.
            step: Keep every `step`-th frame, counting from the first.
            device: Where frames are placed, and where filtering runs.
        """
        return cls(source, params.build(), step=step, device=device)

    @property
    def source(self) -> FilteredSequence[M, T]:
        """The filtered view this reads from, over the strided source."""
        return self._source

    @property
    def device(self) -> Device:
        """Where frames are placed, owned by the view rather than duplicated here.

        Reassignable, and takes effect from the next read; see
        `FilteredSequence.device`.
        """
        return self._source.device

    @device.setter
    def device(self, value: DeviceLike) -> None:
        self._source.device = value

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
