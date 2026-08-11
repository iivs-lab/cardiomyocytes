from __future__ import annotations

__all__ = ("FilteredSequence", "frame_indices")

from typing import TYPE_CHECKING, Any, cast, override

import numpy as np
import torch
from kaparoo.data.sequences import DataSequence, SlicedSequence

from iivs_cardio.common.device import Device
from iivs_cardio.common.range import all_finite

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch import Tensor

    from iivs_cardio.common.device import DeviceLike
    from iivs_cardio.data.transforms.filtering.kernel import FilterKernel, KernelConfig

# Any real numpy dtype. Complex is excluded: casting one drops its phase.
type NumPyRealDType = np.floating | np.integer


def frame_indices(
    total: int, *, start: int = 0, step: int = 1, count: int | None = None
) -> range:
    """Return which of `total` source frames a run takes, in order.

    The one place the three settings become positions: a run reads its frames
    through a sequence and lists them again when it says what it covered, so a
    difference between the two shows up as every output being stale.

    A selection overrunning the source is clamped rather than refused, since
    how short a sequence may come is the caller's policy. `total` alone is
    unchecked, being measured from a source where the rest are read from a
    config: measure it rather than pick a length that shapes the answer.

    Args:
        total: How many frames the source holds.
        start: The first source frame to take. Defaults to 0.
        step: The stride to take them at. Defaults to 1.
        count: How many to take once the stride has been applied. Defaults to
            `None`, which takes them all.

    Returns:
        The source positions to read, ascending.

    Raises:
        ValueError: If `start` is negative, or `step` or `count` is below one.
    """
    if start < 0:
        msg = f"invalid frame start {start}: expected 0 or more"
        raise ValueError(msg)

    if step < 1:
        msg = f"invalid frame step {step}: expected 1 or more"
        raise ValueError(msg)

    if count is not None and count < 1:
        msg = f"invalid frame count {count}: expected 1 or more, or None"
        raise ValueError(msg)

    stop = total if count is None else min(total, start + count * step)

    return range(start, stop, step)


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

    The frames are chosen before filtering, not after: taking every Nth first
    is what makes a strided read measure the frame rate it claims to. It also
    renumbers the view, so `len` and every index count kept frames, and item 1
    at `step=2` is the source's frame 2.

    Type Parameters:
        S: The source's own type, which `origin` gives back unchanged.
        M: The source's per frame metadata, which filtering passes through.
        T: The source's numpy dtype, any real kind. Frames are read as float32,
            since that is what a kernel reduces and what the output carries.

    Args:
        source: The sequence to read frames from.
        kernel: The reduction to apply over each window.
        start: The first source frame to take. Defaults to 0.
        step: The stride to read the source at. Defaults to 1.
        count: How many frames to take once the stride has been applied.
            Defaults to `None`, which takes them all.
        device: The device the frames are filtered on. Defaults to `"cpu"`.

    Raises:
        ValueError: If `start` is negative, or `step` or `count` is below one.
    """

    def __init__(
        self,
        source: S,
        kernel: FilterKernel,
        *,
        start: int = 0,
        step: int = 1,
        count: int | None = None,
        device: DeviceLike = "cpu",
    ) -> None:
        self._origin = cast("DataSequence[NDArray[T], M]", source)

        whole = len(self._origin)
        indices = frame_indices(whole, start=start, step=step, count=count)

        self._source = self._origin
        if indices != range(whole):
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
        start: int = 0,
        step: int = 1,
        count: int | None = None,
        device: DeviceLike = "cpu",
    ) -> FilteredSequence[S, M, T]:
        """Build a filtered view from a kernel's settings rather than a kernel.

        Returns:
            The view, with a kernel built from `config`.
        """
        return cls(
            source, config.build(), start=start, step=step, count=count, device=device
        )

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
        refused. The check follows the cast to float32, since a value a wider
        source holds may be finite there and infinite once narrowed, and comes
        before the move to the device, so the answer costs no synchronisation.

        What is buffered owns its memory rather than viewing the source's, so
        neither side can change the other: a source handing back a slice of an
        array it keeps is the ordinary case, and without the copy a float32 one
        would be buffered, filtered, and handed to a caller by reference.
        """
        self._buffer = {i: f for i, f in self._buffer.items() if i in indices}

        device = self.device.as_torch

        if missing := [i for i in indices if i not in self._buffer]:
            for i, frame in zip(missing, self._source.get_items(missing), strict=True):
                held = torch.from_numpy(frame).to(torch.float32, copy=True)
                if not all_finite(held):
                    msg = f"non-finite value in {self._source.get_meta(i)}"
                    raise ValueError(msg)

                self._buffer[i] = held.to(device)

        if len(indices) == 1:
            return self._buffer[indices[0]].unsqueeze(0)

        return torch.stack([self._buffer[i] for i in indices])
