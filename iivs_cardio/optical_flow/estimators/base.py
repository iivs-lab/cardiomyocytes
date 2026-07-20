from __future__ import annotations

__all__ = ("OpticalFlowEstimator",)

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from iivs_cardio.common.device import DEVICE_KINDS, resolve_device

if TYPE_CHECKING:
    import torch
    from torch import Tensor

    from iivs_cardio.common.device import DeviceKind


class OpticalFlowEstimator(ABC):
    """Stateful dense optical-flow estimator over a stream of frames.

    Frames are pushed one at a time as `torch.Tensor`s. The tensor carries its
    own device, so CPU and CUDA estimators share a single interface; a CUDA
    estimator keeps its whole round trip on the device (no host transfer), so
    its output can chain into the next GPU stage or be offloaded with `.cpu()`.

    Each `push` returns the flow from the previously pushed frame to the current
    one, so the first call returns `None` (N frames yield N-1 flows). The
    estimator retains only the previous frame, so its memory is O(1) regardless
    of sequence length — the caller consumes or offloads each returned flow
    rather than accumulating them. `reset` starts a new sequence and `calc` is a
    stateless one-shot for a single pair.

    Subclasses pin the concrete dtype/shape of frames and flow (e.g. the OpenCV
    backend takes `(H, W)` uint8 frames and returns `(H, W, 2)` float32 flow).
    """

    SUPPORTED_DEVICES: ClassVar[frozenset[DeviceKind]] = DEVICE_KINDS

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = resolve_device(device, self.SUPPORTED_DEVICES)

    @property
    def is_cuda(self) -> bool:
        """Whether this estimator runs on a CUDA device."""
        return self.device.type == "cuda"

    @abstractmethod
    def reset(self) -> None:
        """Forget the retained previous frame to start a new sequence."""

    @abstractmethod
    def push(self, frame: Tensor) -> Tensor | None:
        """Return the flow from the previous frame to `frame`, `None` if first."""

    @abstractmethod
    def push_chunk(self, frames: Tensor) -> Tensor:
        """Stream a chunk of `N` consecutive frames, returning stacked flows.

        Continues the sequence: the retained previous frame pairs with the first of the
        chunk, so `N` frames yield `N` flows (or `N - 1` on the first chunk). Bound the
        chunk size to bound the output memory.
        """

    @abstractmethod
    def calc(self, prev: Tensor, curr: Tensor) -> Tensor:
        """Compute the dense flow `prev -> curr` in one shot (stateless)."""

    @abstractmethod
    def calc_batch(self, prev: Tensor, curr: Tensor) -> Tensor:
        """Compute the flow for a batch of independent pairs `prev[i] -> curr[i]`.

        `prev` and `curr` are `(N, ...)`; returns `(N, ...)` stacked flows.
        """
