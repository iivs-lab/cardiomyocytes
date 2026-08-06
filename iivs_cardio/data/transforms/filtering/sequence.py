from __future__ import annotations

__all__ = ("FilteredSequence",)

from typing import TYPE_CHECKING, Any, cast, override

import numpy as np
import torch
from kaparoo.data.sequences import DataSequence, SlicedSequence

from iivs_cardio.common.device import Device

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch import Tensor

    from iivs_cardio.common.device import DeviceLike
    from iivs_cardio.data.transforms.filtering.kernel import FilterKernel, KernelConfig

type NumPyRealDType = np.floating | np.integer
"""Any real numpy dtype. Complex is excluded: casting one drops its phase."""


class FilteredSequence[S: DataSequence[Any, Any], M, T: NumPyRealDType = np.float32](
    DataSequence["Tensor", M]
):
    def __init__(
        self,
        source: S,
        kernel: FilterKernel,
        *,
        step: int = 1,
        device: DeviceLike = "cpu",
    ) -> None:
        if step < 1:
            msg = f"invalid frame step {step}: expected 1 or more"
            raise ValueError(msg)

        self._origin = cast("DataSequence[NDArray[T], M]", source)

        self._source = self._origin
        if step > 1:
            indices = range(0, len(self._origin), step)
            self._source = SlicedSequence(self._origin, indices)

        self._buffer: dict[int, Tensor] = {}
        self._device = Device.resolve(device)
        self.kernel = kernel

    @property
    def device(self) -> Device:
        return self._device

    @device.setter
    def device(self, value: DeviceLike) -> None:
        device = Device.resolve(value)
        if device != self._device:
            self._buffer.clear()
        self._device = device

    @classmethod
    def from_config(
        cls,
        source: S,
        config: KernelConfig,
        *,
        step: int = 1,
        device: DeviceLike = "cpu",
    ) -> FilteredSequence[S, M, T]:
        return cls(source, config.build(), step=step, device=device)

    @property
    def origin(self) -> S:
        return cast("S", self._origin)

    @override
    def __len__(self) -> int:
        return len(self._source)

    @override
    def get_item(self, index: int) -> Tensor:
        index = self._normalize_index(index)
        radius = self.kernel.temporal_radius

        start = max(0, index - radius)
        stop = min(len(self), index + radius + 1)

        window = self._window(range(start, stop))
        return self.kernel.apply(window, index - start)

    @override
    def get_meta(self, index: int) -> M:
        return self._source.get_meta(self._normalize_index(index))

    def _window(self, indices: range) -> Tensor:
        self._buffer = {i: f for i, f in self._buffer.items() if i in indices}

        device = self.device.as_torch

        missing = [i for i in indices if i not in self._buffer]
        for i, frame in zip(missing, self._source.get_items(missing), strict=True):
            if not np.isfinite(frame).all():
                msg = f"non-finite value in {self._source.get_meta(i)}"
                raise ValueError(msg)

            self._buffer[i] = torch.from_numpy(frame).to(device, torch.float32)

        if len(indices) == 1:
            return self._buffer[indices[0]].unsqueeze(0)

        return torch.stack([self._buffer[i] for i in indices])
