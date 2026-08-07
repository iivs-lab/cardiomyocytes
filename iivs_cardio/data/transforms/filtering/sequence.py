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

# Any real numpy dtype. Complex is excluded: casting one drops its phase.
type NumPyRealDType = np.floating | np.integer


class FilteredSequence[S: DataSequence[Any, Any], M, T: NumPyRealDType = np.float32](
    DataSequence["Tensor", M]
):
    """A filtered view over a sequence, which is itself a sequence.

    Asking for item `i` applies the kernel to the frames around `i`, so what
    comes back depends on `i` alone, whatever order the items are asked for.
    Frames at the two ends are filtered over a shorter window rather than a
    padded one, so every kept frame gives exactly one output.

    A small buffer holds the frames of the current window, which makes a walk
    in order cost one read of the source per frame. Reading out of order stays
    correct and only misses the buffer more often.

    `step` is applied before filtering, not after: taking every Nth frame first
    is what makes a strided read measure the frame rate it claims to. It also
    renumbers the view, so `len` and every index count kept frames, and item 1
    at `step=2` is the source's frame 2.

    Type Parameters:
        S: the source's own type, which `origin` gives back unchanged.
        M: the source's per frame metadata, which filtering passes through.
        T: the source's numpy dtype, any real kind. Frames are read as float32,
            since that is what a kernel reduces and what the output carries.

    Args:
        source: the sequence to read frames from.
        kernel: the reduction to apply over each window.
        step: take every `step`th frame of the source, before filtering.
        device: where the frames are filtered.

    Raises:
        ValueError: If `step` is less than one.
    """

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
        """Where the frames are filtered, and where the output comes back on.

        Setting it to a different device drops whatever is buffered, since
        those frames are on the device it was set away from.
        """
        return self._device

    @device.setter
    def device(self, value: DeviceLike) -> None:
        device = Device.resolve(value)
        if device != self._device:
            self._buffer.clear()
        self._device = device

    def release(self) -> None:
        """Drop the buffered window, leaving the view usable.

        Nothing else lets go of it: the window is whatever the last read needed,
        so a view that has been walked to the end keeps that much on its device
        for as long as anything holds the view. A caller that has finished with
        a sequence but not with the object says so here, and one that reads
        again simply pays for the frames a second time.
        """
        self._buffer.clear()

    @classmethod
    def from_config(
        cls,
        source: S,
        config: KernelConfig,
        *,
        step: int = 1,
        device: DeviceLike = "cpu",
    ) -> FilteredSequence[S, M, T]:
        """Build a filtered view from a kernel's settings rather than a kernel.

        Returns:
            The view, with a kernel built from `config`.
        """
        return cls(source, config.build(), step=step, device=device)

    @property
    def origin(self) -> S:
        """The source as it was given, unstrided and unfiltered."""
        return cast("S", self._origin)

    @override
    def __len__(self) -> int:
        """The number of frames kept, which the stride has already narrowed."""
        return len(self._source)

    @override
    def get_item(self, index: int) -> Tensor:
        """Return frame `index` of the view, filtered over the frames near it.

        Raises:
            IndexError: If `index` is outside the frames this view keeps.
            ValueError: If a frame it has to read holds a non finite value.
        """
        index = self._normalize_index(index)
        radius = self.kernel.temporal_radius

        start = max(0, index - radius)
        stop = min(len(self), index + radius + 1)

        window = self._window(range(start, stop))
        return self.kernel.apply(window, index - start)

    @override
    def get_meta(self, index: int) -> M:
        """Return the source's own metadata for the frame at view `index`.

        It names the frame the way the source does, so a message about a frame
        points at the file it came from rather than at a renumbered position.

        Raises:
            IndexError: If `index` is outside the frames this view keeps.
        """
        return self._source.get_meta(self._normalize_index(index))

    def _window(self, indices: range) -> Tensor:
        """Read the frames the window needs, keeping only those still in it.

        This is the one place every source frame passes through, and the last
        one before they leave the host, so it is where a non finite value is
        refused.
        """
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
