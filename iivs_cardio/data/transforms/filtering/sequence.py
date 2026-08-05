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
    """A filtered view over a source sequence, itself a sequence.

    Wraps `source` rather than consuming it, so `filtered[i]` is the kernel
    applied at `source[i - rz .. i + rz]` -- determined by `i` alone, whatever
    order the items are asked for. That is the property a delay line cannot
    offer: owning the source is what makes indexed access well defined, since
    the window can always be re-read.

    Every kept frame yields one output. The ends are filtered on a truncated
    window rather than a padded one, so `len` matches what `step` kept.

    `step` is applied *before* filtering: taking every Nth frame first is what
    makes a strided read measure the frame rate it claims to, where filtering
    first would fold the dropped frames into the kept ones. Striding renumbers
    the view -- `len` and every index, `get_meta` included, count kept frames --
    so `self[1]` at `step=2` is the source's frame 2. A selection `step` cannot
    express is written by slicing the source first.

    A small buffer holds the frames of the current window, so walking the
    sequence in order costs one source read per frame instead of `2 * rz + 1`.
    Out-of-order access stays correct and simply misses the buffer more often.
    It is sized for that sequential pass -- building the filtered cache -- not
    for a shuffled `DataLoader`, which should read the finished cache instead.

    Type Parameters:
        S: the source's own type, which `source` gives back unchanged, so a
            caller that needs what only that type offers does not have to carry
            a second reference to the same object.
        M: the source's per-frame metadata, which filtering passes through.
        T: the source's numpy dtype, any real (integer or floating) kind; each
            frame is read as float32, since that is what a kernel reduces and
            what the output carries. Defaults to `float32`, the usual source.

    Args:
        source: the frames to filter, all the same shape.
        kernel: the neighbourhood to reduce, and the reduction.
        step: keep every `step`-th frame, counting from the first.
        device: where filtering runs and the returned tensors live.

    Raises:
        ValueError: If `step` is below 1, or `device` names an unsupported kind.
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
        """Where filtering runs and the returned tensors live.

        Reassignable: frames are placed as they are read, so a later device
        takes effect from the next read. Buffered frames sit on the old one and
        are dropped rather than moved, which costs one window of re-reads.
        """
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
        """Build the kernel `config` describes, and filter `source` with it.

        Args:
            source: the frames to filter.
            config: which kernel to build, and with what.
            step: keep every `step`-th frame, counting from the first.
            device: where filtering runs and the returned tensors live.
        """
        return cls(source, config.build(), step=step, device=device)

    @property
    def origin(self) -> S:
        """The unfiltered sequence this was opened over, as the type it was given.

        The stride between it and here is this class's own, so it hands the
        source back rather than making a caller reach through it.
        """
        return cast("S", self._origin)

    @override
    def __len__(self) -> int:
        return len(self._source)

    @override
    def get_item(self, index: int) -> Tensor:
        """Return source frame `index` filtered against its neighbours."""
        index = self._normalize_index(index)
        radius = self.kernel.temporal_radius

        start = max(0, index - radius)
        stop = min(len(self), index + radius + 1)

        window = self._window(range(start, stop))
        return self.kernel.apply(window, index - start)

    @override
    def get_meta(self, index: int) -> M:
        """Return the source's metadata for `index`, which filtering leaves alone."""
        return self._source.get_meta(self._normalize_index(index))

    def _window(self, indices: range) -> Tensor:
        """Stack the source frames at `indices` as float32, reading past the buffer.

        Reads only frames not already buffered, and casts each to float32 on the
        way in -- the dtype a kernel reduces, whatever the source stored. A
        one-frame window is unsqueezed rather than stacked, since `torch.stack`
        copies even a single frame while `unsqueeze` returns a view; every
        `rz = 0` kernel reads through this path once per frame.
        """
        self._buffer = {i: f for i, f in self._buffer.items() if i in indices}

        missing = [i for i in indices if i not in self._buffer]
        for i, frame in zip(missing, self._source.get_items(missing), strict=True):
            self._buffer[i] = torch.from_numpy(frame).to(
                self.device.as_torch, torch.float32
            )

        if len(indices) == 1:
            return self._buffer[indices[0]].unsqueeze(0)

        return torch.stack([self._buffer[i] for i in indices])
