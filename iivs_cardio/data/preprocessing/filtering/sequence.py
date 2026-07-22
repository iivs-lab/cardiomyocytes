from __future__ import annotations

__all__ = ("FilteredSequence",)

from typing import TYPE_CHECKING, override

import torch
from kaparoo.data.sequences import DataSequence

from iivs_cardio.common.device import resolve_device

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from torch import Tensor

    from iivs_cardio.data.preprocessing.filtering.kernel import Kernel, MedianParams


class FilteredSequence[M](DataSequence["Tensor", M]):
    """A filtered view over a phase sequence, itself a sequence.

    Wraps `source` rather than consuming it, so `filtered[i]` is the kernel
    applied at `source[i - rz .. i + rz]` -- determined by `i` alone, whatever
    order the items are asked for. That is the property a delay line cannot
    offer: owning the source is what makes indexed access well defined, since
    the window can always be re-read.

    Every source frame yields one output. The ends are filtered on a truncated
    window rather than a padded one, so `len` matches the source exactly.

    A small buffer holds the frames of the current window, so walking the
    sequence in order costs one source read per frame instead of `2 * rz + 1`.
    Out-of-order access stays correct and simply misses the buffer more often.
    It is sized for that sequential pass -- building the filtered cache -- not
    for a shuffled `DataLoader`, which should read the finished cache instead.

    Args:
        source: the float32 phase frames to filter, all the same shape.
        kernel: the neighbourhood to reduce, and the reduction.
        device: where filtering runs and the returned tensors live.

    Raises:
        ValueError: If `device` names an unsupported device kind.
    """

    def __init__(
        self,
        source: DataSequence[NDArray[np.float32], M],
        kernel: Kernel,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self._source = source
        self._buffer: dict[int, Tensor] = {}
        self.kernel = kernel
        self.device = resolve_device(device)

    @classmethod
    def from_params(
        cls,
        source: DataSequence[NDArray[np.float32], M],
        params: MedianParams,
        *,
        device: str | torch.device = "cpu",
    ) -> FilteredSequence[M]:
        """Build the kernel `params` describes, and filter `source` with it.

        The entry point for configuration-driven callers -- a CLI, a config
        file, a cache sidecar being replayed -- which hold settings rather than
        a live kernel.

        Args:
            source: the float32 phase frames to filter.
            params: which kernel to build, and with what.
            device: where filtering runs and the returned tensors live.
        """
        return cls(source, params.build(), device=device)

    @property
    def source(self) -> DataSequence[NDArray[np.float32], M]:
        """The unfiltered sequence this reads from."""
        return self._source

    @override
    def __len__(self) -> int:
        return len(self._source)

    @override
    def get_item(self, index: int) -> Tensor:
        """Return source frame `index` filtered against its neighbours."""
        index = self._resolve(index)
        radius = self.kernel.temporal_radius

        start = max(0, index - radius)
        stop = min(len(self), index + radius + 1)

        return self.kernel.apply(self._window(range(start, stop)), index - start)

    @override
    def get_meta(self, index: int) -> M:
        """Return the source's metadata for `index`, which filtering leaves alone."""
        return self._source.get_meta(self._resolve(index))

    def _resolve(self, index: int) -> int:
        """Normalize a possibly-negative index and bounds-check it."""
        length = len(self)
        if not -length <= index < length:
            msg = f"index {index} out of range for {length} frames"
            raise IndexError(msg)
        return index % length

    def _window(self, indices: range) -> Tensor:
        """Stack the source frames at `indices`, reading only what is not buffered."""
        self._buffer = {i: f for i, f in self._buffer.items() if i in indices}

        missing = [i for i in indices if i not in self._buffer]
        for i, frame in zip(missing, self._source.get_items(missing), strict=True):
            self._buffer[i] = torch.from_numpy(frame).to(self.device)

        return torch.stack([self._buffer[i] for i in indices])
