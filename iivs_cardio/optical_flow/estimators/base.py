from __future__ import annotations

__all__ = ("EstimatorConfig", "OpticalFlowEstimator")

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from iivs_cardio.common.device import DEVICE_KINDS, Device

if TYPE_CHECKING:
    import torch
    from torch import Tensor

    from iivs_cardio.common.device import DeviceKind, DeviceLike


class OpticalFlowEstimator(ABC):
    """Stateful dense optical-flow estimator over a stream of frames.

    Frames are pushed one at a time as `torch.Tensor`s. The tensor carries its
    own device, so CPU and CUDA estimators share a single interface; a CUDA
    estimator keeps its whole round trip on the device (no host transfer), so
    its output can chain into the next GPU stage or be offloaded with `.cpu()`.

    Each `push` returns the flow from the previously pushed frame to the current
    one, so the first call returns `None` (N frames yield N-1 flows). The
    estimator retains only the previous frame, so its memory is O(1) whatever the
    sequence length: the caller consumes or offloads each returned flow rather
    than accumulating them. `reset` starts a new sequence and `calc` is a
    stateless one-shot for a single pair.

    A subclass pins the concrete dtype and shape of a frame and of a flow, which
    is the one thing this contract leaves open.

    Attributes:
        device: The device this estimator runs on.
        is_cuda: Whether that device is a CUDA one.
    """

    def __init__(self, device: DeviceLike = "cpu") -> None:
        self.device = Device.resolve(device)

    @property
    def is_cuda(self) -> bool:
        """Whether this estimator runs on a CUDA device."""
        return self.device.is_cuda

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


class EstimatorConfig(ABC):
    """An estimator's constructor arguments as one value, buildable into it.

    Separate from `OpticalFlowEstimator` so a config, a CLI, or a process-pool
    recipe carries the settings without a live estimator, which cannot cross a
    process boundary while it holds a library object that does not pickle.
    `build` reconstructs one inside the worker, and takes the device because an
    estimator is bound to one: a cv2 CUDA algorithm is allocated on whichever
    device was current when it was made.

    Which devices an algorithm runs on is declared here rather than on the
    estimator: an algorithm with no implementation for one is a fact about the
    algorithm, not about the machinery that streams frames through it.

    What frames the built estimator takes is declared here for the same
    reason: nothing upstream can read the dtype off an estimator that does not
    exist yet, and a normalizer has to scale onto it before one is built.

    Attributes:
        SUPPORTED_DEVICES: The device kinds this algorithm has an
            implementation for. Defaults to every kind.
        FRAME_DTYPE: The dtype the built estimator takes its frames in, which
            is what a normalizer upstream scales onto. Declared without a
            default, since a wrong one would be found in a shape error deep in
            a library rather than here.
    """

    SUPPORTED_DEVICES: ClassVar[frozenset[DeviceKind]] = DEVICE_KINDS
    FRAME_DTYPE: ClassVar[torch.dtype]

    @abstractmethod
    def build(self, device: DeviceLike = "cpu") -> OpticalFlowEstimator:
        """Construct the estimator these describe, on `device`.

        Raises:
            ValueError: If `device` is not one of `SUPPORTED_DEVICES`.
        """
