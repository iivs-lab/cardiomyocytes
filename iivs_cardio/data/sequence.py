from __future__ import annotations

__all__ = ("FrameSequence",)

from functools import partial
from typing import TYPE_CHECKING, Any, override

import numpy as np
import torch
from kaparoo.data.sequences import DataSequence, TransformedSequence
from torch import Tensor

from iivs_cardio.common.device import resolve_device
from iivs_cardio.data.transforms.filtering import FilteredSequence

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from iivs_cardio.data.transforms.filtering import FilterKernel, KernelParams

type NumPyRealDType = np.floating | np.integer
"""Any real numpy dtype. Complex is excluded: casting one drops its phase."""


def _as_tensor(frame: NDArray[Any], device: torch.device) -> Tensor:
    """Read one frame as float32 on `device`, the form filtering also returns."""
    return torch.from_numpy(frame).to(device, torch.float32)


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
