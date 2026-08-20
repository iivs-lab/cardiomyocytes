from __future__ import annotations

__all__ = ("DenseAlgorithm", "OpenCVConfig", "OpenCVEstimator")

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, override

import cv2
import torch
from beartype import beartype
from cv2.cuda import GpuMat
from jaxtyping import Float32, UInt8, jaxtyped
from torch import Tensor

from iivs_cardio.common.cuda_utils import gpumat_to_cupy, tensor_to_gpumat
from iivs_cardio.common.device import Device
from iivs_cardio.optical_flow.estimators.base import (
    EstimatorConfig,
    OpticalFlowEstimator,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from iivs_cardio.common.device import DeviceLike

DenseAlgorithm = cv2.DenseOpticalFlow | cv2.cuda.DenseOpticalFlow

FrameType = UInt8[Tensor, "H W"]
FlowType = Float32[Tensor, "2 H W"]

BatchFrameType = UInt8[Tensor, "N H W"]
BatchFlowType = Float32[Tensor, "N 2 H W"]
ChunkFlowType = Float32[Tensor, "M 2 H W"]


class OpenCVConfig(EstimatorConfig, ABC):
    """The settings of one cv2 flow algorithm, and how to make it on a device.

    Every OpenCV estimator is built through this one `build`, so an algorithm is
    added here rather than beside `OpenCVEstimator`: a subclass carries the
    parameters, makes the algorithm that reads them, and narrows
    `SUPPORTED_DEVICES` where cv2 has no implementation for a device.

    Attributes:
        SUPPORTED_DEVICES: As `EstimatorConfig`, narrowed by a subclass whose
            algorithm does not run everywhere.
    """

    @abstractmethod
    def _create(self, device: Device) -> DenseAlgorithm:
        """Make the cv2 algorithm these settings describe, for `device`.

        Called with `device` resolved and already current, so an implementation
        asks cv2 for the factory it wants rather than binding anything itself.
        """

    def _backend(self, device: DeviceLike = "cpu") -> Backend:
        """Return the backend that runs the algorithm these settings describe.

        Args:
            device: The device to make it for, in any form a caller may write.

        Returns:
            A backend of its own, so two estimators built from one config share
            no state.
        """
        device = Device.resolve(device, self.SUPPORTED_DEVICES)
        device.activate()

        algorithm = self._create(device)

        on_cuda = isinstance(algorithm, cv2.cuda.DenseOpticalFlow)
        if on_cuda != device.is_cuda:
            made = "cuda" if on_cuda else "cpu"
            msg = f"{type(algorithm).__name__} was made for {made}, not for {device}"
            raise ValueError(msg)

        if isinstance(algorithm, cv2.cuda.DenseOpticalFlow):
            return CUDABackend(algorithm, device)

        return CPUBackend(algorithm)

    @override
    def build(self, device: DeviceLike = "cpu") -> OpenCVEstimator:
        """Build the estimator these settings describe, on `device`.

        Args:
            device: The device to build for, in any form a caller may write.

        Raises:
            ValueError: If `device` is not one of `SUPPORTED_DEVICES`.
        """
        backend = self._backend(device)  # raises if device is unsupported
        return OpenCVEstimator(backend)


# ========================== #
#          Backends          #
# ========================== #


class Backend(ABC):
    """The flow calls of one device, once a frame is in a form cv2 reads.

    The two implementations differ in where a frame is put, how cv2 is called on
    it, and how the answer comes back. Everything above them, the validation and
    the streaming and the batching, is one.

    Attributes:
        algorithm: The cv2 algorithm this calls, which is the one an estimator
            reads its settings back from.
        device: Where it runs. Carried here because an algorithm does not say
            which device cv2 made it on, and whoever runs it has to know.
        retained: Whether a frame is retained, so the next `push` yields a flow.
    """

    # Declared, not abstract: an attribute an `__init__` sets does not clear an
    # `abstractmethod`. A subclass re-declares `algorithm` as the concrete cv2
    # type it calls, so a call needs no second word on which of the two it is.
    algorithm: DenseAlgorithm
    device: Device

    @property
    @abstractmethod
    def retained(self) -> bool:
        """Whether a frame is retained, so the next `push` yields a flow."""

    @abstractmethod
    def push(self, frame: Tensor, out: Tensor | None = None) -> Tensor | None:
        """Return the flow from the retained frame, and retain `frame`.

        Args:
            frame: The frame to retain, copied, so a caller may write over its
                own afterwards.
            out: Where to put the flow, sparing the allocation a caller that
                already has somewhere to put it would only copy out of.

        Returns:
            The flow, or `None` where nothing was retained yet, in which case
            `out` is left untouched.
        """

    @abstractmethod
    def calc(self, prev: Tensor, curr: Tensor, out: Tensor | None = None) -> Tensor:
        """Return the flow `prev -> curr`, leaving the retained frame alone.

        Args:
            prev: The frame to flow from.
            curr: The frame to flow to.
            out: Where to put the flow, as `push`.
        """

    @abstractmethod
    def reset(self) -> None:
        """Forget the retained frame."""

    def _as_flow(self, flow: Tensor, out: Tensor | None) -> Tensor:
        """Return cv2's `(H, W, 2)` flow as the `(2, H, W)` torch ops consume.

        Args:
            flow: The flow as cv2 wrote it, which may be a view of a buffer the
                next call writes over.
            out: Where to put it, or `None` to allocate.

        Returns:
            A flow of the caller's own either way, so it outlives the buffer.
        """
        channels_first = flow.permute(2, 0, 1)
        if out is None:
            return channels_first.contiguous()

        return out.copy_(channels_first)


class CPUBackend(Backend):
    """The CPU calls, over `numpy` views of frames the backend copied.

    Attributes:
        algorithm: The cv2 algorithm to call.
        device: The CPU, which is not asked for: there is one of it, where a
            host with several GPUs has a CUDA device to choose between.
        retained: As `Backend`.
    """

    algorithm: cv2.DenseOpticalFlow
    device = Device("cpu")

    def __init__(self, algorithm: cv2.DenseOpticalFlow) -> None:
        self.algorithm = algorithm
        self._prev: Tensor | None = None

    @property
    @override
    def retained(self) -> bool:
        return self._prev is not None

    @override
    def push(self, frame: Tensor, out: Tensor | None = None) -> Tensor | None:
        # Copied, as the CUDA backend copies into a `GpuMat` of its own: a
        # caller refilling one buffer would otherwise overwrite the retained
        # frame, and a frame taken from a chunk is a view pinning the batch.
        prev, self._prev = self._prev, frame.clone()
        if prev is None:
            return None
        return self.calc(prev, self._prev, out)

    @override
    def calc(self, prev: Tensor, curr: Tensor, out: Tensor | None = None) -> Tensor:
        prev_np: NDArray[np.uint8] = prev.contiguous().numpy()
        curr_np: NDArray[np.uint8] = curr.contiguous().numpy()
        flow: NDArray[np.float32] = self.algorithm.calc(prev_np, curr_np, None)  # ty: ignore[no-matching-overload]
        return self._as_flow(torch.from_numpy(flow), out)

    @override
    def reset(self) -> None:
        self._prev = None


class CUDABackend(Backend):
    """The CUDA calls, over `GpuMat`s the backend owns and reuses.

    Attributes:
        algorithm: The cv2 algorithm to call.
        device: The device it and the buffers live on.
        retained: As `Backend`. Held as a flag of its own rather than read off
            an empty buffer, so `reset` keeps the buffers it has.
    """

    algorithm: cv2.cuda.DenseOpticalFlow

    def __init__(self, algorithm: cv2.cuda.DenseOpticalFlow, device: Device) -> None:
        self.algorithm = algorithm
        self.device = device
        self._flow_buffer = GpuMat()
        self._calc_buffers = (GpuMat(), GpuMat())
        self._push_buffers = (GpuMat(), GpuMat())
        self._push_slot = 0
        self._retained = False

    @property
    @override
    def retained(self) -> bool:
        return self._retained

    @override
    def push(self, frame: Tensor, out: Tensor | None = None) -> Tensor | None:
        self.device.activate()  # the GpuMat/CuPy calls below read the global device
        buffer_prev = self._push_buffers[self._push_slot]
        buffer_curr = self._push_buffers[self._push_slot ^ 1]
        tensor_to_gpumat(frame, out=buffer_curr)
        self._push_slot ^= 1
        retained, self._retained = self._retained, True
        if not retained:
            return None
        return self._flow_between(buffer_prev, buffer_curr, out)

    @override
    def calc(self, prev: Tensor, curr: Tensor, out: Tensor | None = None) -> Tensor:
        self.device.activate()
        buffer_prev, buffer_curr = self._calc_buffers
        tensor_to_gpumat(prev, out=buffer_prev)
        tensor_to_gpumat(curr, out=buffer_curr)
        return self._flow_between(buffer_prev, buffer_curr, out)

    @override
    def reset(self) -> None:
        self._retained = False

    def _flow_between(self, prev: GpuMat, curr: GpuMat, out: Tensor | None) -> Tensor:
        if self._flow_buffer.size() != prev.size():
            self._flow_buffer = GpuMat(prev.size(), cv2.CV_32FC2)
        self._flow_buffer = self.algorithm.calc(prev, curr, self._flow_buffer)
        flow = torch.as_tensor(gpumat_to_cupy(self._flow_buffer))
        return self._as_flow(flow, out)


# ========================== #
#         Estimator          #
# ========================== #


class OpenCVEstimator(OpticalFlowEstimator):
    """Optical-flow estimation backed by one OpenCV `cv2` / `cv2.cuda` algorithm.

    Takes `(H, W)` uint8 frames and returns `(2, H, W)` float32 flow (channel 0 =
    dx, channel 1 = dy) as `torch.Tensor`s on `self.device`. cv2 computes flow in
    `(H, W, 2)`; the output is transposed once to the channel-first layout that
    torch spatial ops (`grid_sample`, `conv2d`) consume natively. A CUDA estimator
    keeps the whole computation on the device, so its output chains into the next
    GPU stage without a host transfer.

    One class serves every algorithm and every device, what differs between
    them being the backend's business rather than the streaming this holds.
    Build one through `OpenCVConfig.build`.

    Separate from `OpticalFlowEstimator` so a future PyTorch (`nn.Module`)
    backend can extend the neutral base directly.

    Args:
        backend: What runs the flow calls, holding the cv2 algorithm and the
            device it was made on. `OpenCVConfig.build` is what makes one.

    Attributes:
        algorithm: The cv2 algorithm itself, which is where the settings it was
            made with can be read back from.
        device: As `OpticalFlowEstimator`, the device the algorithm was made on.
        is_cuda: As `OpticalFlowEstimator`.
    """

    def __init__(self, backend: Backend) -> None:
        super().__init__(backend.device)
        self._backend = backend

    @property
    def algorithm(self) -> DenseAlgorithm:
        """The cv2 algorithm this estimator streams through."""
        return self._backend.algorithm

    def validate_device(self, frame: Tensor) -> None:
        """Raise if `frame` is not on this estimator's device.

        Raises:
            ValueError: If `frame` sits on another device. The algorithm names
                itself in the refusal, the estimator being one class for all of
                them.
        """
        if frame.device != self.device.as_torch:
            name = type(self.algorithm).__name__
            msg = f"{name} expects a {self.device} tensor, got one on {frame.device}"
            raise ValueError(msg)

    @override
    def reset(self) -> None:
        """Forget the retained frame, restarting the sequence."""
        self._backend.reset()

    @jaxtyped(typechecker=beartype)
    @override
    def push(self, frame: FrameType) -> FlowType | None:
        """Return the flow from the retained frame, or `None` on the first frame."""
        self.validate_device(frame)

        return self._backend.push(frame)

    @jaxtyped(typechecker=beartype)
    @override
    def push_chunk(self, frames: BatchFrameType) -> ChunkFlowType:
        """Stream a chunk of frames, returning stacked flows continuing the sequence.

        Each flow is written into the batch as it comes, rather than collected
        and stacked afterwards, which would hold the whole chunk twice over.
        """
        self.validate_device(frames)

        count = len(frames) if self._backend.retained else max(len(frames) - 1, 0)
        flows = self._flow_batch(count, frames)

        index = 0
        for frame in frames:
            # No row to write only on the frame a fresh sequence spends retaining.
            out = flows[index] if index < count else None
            if self._backend.push(frame, out=out) is not None:
                index += 1

        return flows

    @jaxtyped(typechecker=beartype)
    @override
    def calc(self, prev: FrameType, curr: FrameType) -> FlowType:
        """Compute the flow `prev -> curr` in one shot, leaving no retained state."""
        self.validate_device(prev)
        self.validate_device(curr)

        return self._backend.calc(prev, curr)

    @jaxtyped(typechecker=beartype)
    @override
    def calc_batch(self, prev: BatchFrameType, curr: BatchFrameType) -> BatchFlowType:
        """Compute the flow for each independent pair `prev[i] -> curr[i]`, stacked."""
        self.validate_device(prev)
        self.validate_device(curr)

        flows = self._flow_batch(len(prev), prev)
        for index, (p, c) in enumerate(zip(prev, curr, strict=True)):
            self._backend.calc(p, c, out=flows[index])

        return flows

    def _flow_batch(self, count: int, frames: Tensor) -> Tensor:
        """An uninitialized `(count, 2, H, W)` float32 batch beside `frames`.

        Sized to the flows that will be written rather than to `frames`, so no
        row of it is returned still holding whatever `new_empty` picked up.
        """
        return frames.new_empty((count, 2, *frames.shape[1:]), dtype=torch.float32)
