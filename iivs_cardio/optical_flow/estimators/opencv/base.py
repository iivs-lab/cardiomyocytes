from __future__ import annotations

__all__ = ("DenseAlgorithm", "OpenCVAlgorithm", "OpenCVConfig", "OpenCVEstimator")

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, override

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


@dataclass(frozen=True, slots=True)
class OpenCVAlgorithm:
    """A cv2 flow algorithm, and the device it was created on.

    A CUDA algorithm is allocated on whichever device was current when cv2 was
    asked for it, and nothing on the object says which that was. Pairing the two
    is what lets whoever holds one know where it runs, and the pair is checked
    rather than taken on trust, so an algorithm cannot be presented as another
    device's.

    Attributes:
        algorithm: The cv2 algorithm, CPU or CUDA.
        device: The device it was created on.

    Raises:
        ValueError: If the algorithm's backend is not `device`'s.
    """

    algorithm: DenseAlgorithm
    device: Device

    def __post_init__(self) -> None:
        """Refuse a pairing cv2 could not have produced.

        Which backend an algorithm came from is the one thing it does say,
        through its type, where the device it was made on is not recoverable
        from it at all.
        """
        on_cuda = isinstance(self.algorithm, cv2.cuda.DenseOpticalFlow)
        if on_cuda != self.device.is_cuda:
            made = "cuda" if on_cuda else "cpu"
            name = type(self.algorithm).__name__
            msg = f"{name} was made for {made}, not for {self.device}"
            raise ValueError(msg)


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

        Called with `device` already current, so an implementation asks cv2 for
        the factory it wants rather than binding anything itself.

        Args:
            device: The device to make it for, resolved and activated.

        Returns:
            The algorithm, which `build` pairs with `device`.
        """

    @override
    def build(self, device: DeviceLike = "cpu") -> OpenCVEstimator:
        """Build the estimator these settings describe, on `device`.

        Args:
            device: The device to build for, in any form a caller may write.

        Returns:
            The estimator, holding an algorithm made for `device` and carrying
            it, so the two cannot be paired wrongly by whoever receives them.

        Raises:
            ValueError: If `device` is not one of `SUPPORTED_DEVICES`.
        """
        resolved = Device.resolve(device, self.SUPPORTED_DEVICES)
        resolved.activate()

        return OpenCVEstimator(OpenCVAlgorithm(self._create(resolved), resolved))


def _stack_flows(flows: list[Tensor], frames: Tensor) -> Tensor:
    """Stack the flows, or an empty `(0, 2, H, W)` float32 when there are none."""
    if not flows:
        return frames.new_empty((0, 2, *frames.shape[1:]), dtype=torch.float32)
    return torch.stack(flows)


def _as_flow(channels_last: Tensor) -> Tensor:
    """Return cv2's `(H, W, 2)` flow as the `(2, H, W)` torch ops consume.

    The copy `contiguous` makes after the permute is also what frees the result
    from the buffer cv2 wrote it into, which the next call writes over.
    """
    return channels_last.permute(2, 0, 1).contiguous()


# ========================== #
#          Backends          #
# ========================== #


class Backend(Protocol):
    """The flow calls of one device, once a frame is in a form cv2 reads.

    The two implementations differ in where a frame is put, how cv2 is called on
    it, and how the answer comes back. Everything above them, the validation and
    the streaming and the batching, is one.

    Attributes:
        algorithm: The cv2 algorithm this calls, which is the one an estimator
            reads its settings back from.
    """

    @property
    def algorithm(self) -> DenseAlgorithm:
        """The cv2 algorithm this backend calls."""
        ...

    def push(self, frame: Tensor) -> Tensor | None:
        """Return the flow from the retained frame, and retain `frame`.

        Returns:
            The flow, or `None` where nothing was retained yet. `frame` is taken
            in a form of this backend's own, so a caller may write over its own
            afterwards.
        """
        ...

    def calc(self, prev: Tensor, curr: Tensor) -> Tensor:
        """Return the flow `prev -> curr`, leaving the retained frame alone."""
        ...

    def reset(self) -> None:
        """Forget the retained frame."""
        ...


class CPUBackend:
    """The CPU calls, over `numpy` views of frames the backend copied.

    Attributes:
        algorithm: The cv2 algorithm to call.
    """

    def __init__(self, algorithm: cv2.DenseOpticalFlow) -> None:
        self.algorithm = algorithm
        self._prev: Tensor | None = None

    def push(self, frame: Tensor) -> Tensor | None:
        # Copied, as the CUDA backend copies into a `GpuMat` of its own: a
        # caller refilling one buffer would otherwise overwrite the retained
        # frame, and a frame taken from a chunk is a view pinning the batch.
        prev, self._prev = self._prev, frame.clone()
        if prev is None:
            return None
        return self.calc(prev, self._prev)

    def calc(self, prev: Tensor, curr: Tensor) -> Tensor:
        prev_np: NDArray[np.uint8] = prev.contiguous().numpy()
        curr_np: NDArray[np.uint8] = curr.contiguous().numpy()
        flow: NDArray[np.float32] = self.algorithm.calc(prev_np, curr_np, None)  # ty: ignore[no-matching-overload]
        return _as_flow(torch.from_numpy(flow))

    def reset(self) -> None:
        self._prev = None


class CUDABackend:
    """The CUDA calls, over `GpuMat`s the backend owns and reuses.

    Streaming keeps a pair it alternates between, so the frame retained from one
    call is the one the next reads back. One-shot calls take a pair of their own
    rather than borrowing that one, which is what leaves the stream undisturbed
    by a `calc` between two pushes.

    Attributes:
        algorithm: The cv2 algorithm to call.
        device: The device it and the buffers live on.
    """

    def __init__(self, algorithm: cv2.cuda.DenseOpticalFlow, device: Device) -> None:
        self.algorithm = algorithm
        self.device = device
        self._flow = GpuMat()
        self._streamed = (GpuMat(), GpuMat())
        self._one_shot = (GpuMat(), GpuMat())
        self._slot = 0

    def push(self, frame: Tensor) -> Tensor | None:
        self.device.activate()  # the GpuMat/CuPy calls below read the global device
        prev = self._streamed[self._slot]
        curr = self._streamed[self._slot ^ 1]
        tensor_to_gpumat(frame, out=curr)
        self._slot ^= 1
        if prev.empty():
            return None
        return self._flow_between(prev, curr)

    def calc(self, prev: Tensor, curr: Tensor) -> Tensor:
        self.device.activate()
        first, second = self._one_shot
        return self._flow_between(
            tensor_to_gpumat(prev, out=first),
            tensor_to_gpumat(curr, out=second),
        )

    def reset(self) -> None:
        self._streamed = (GpuMat(), GpuMat())
        self._slot = 0

    def _flow_between(self, prev: GpuMat, curr: GpuMat) -> Tensor:
        if self._flow.size() != prev.size():
            self._flow = GpuMat(prev.size(), cv2.CV_32FC2)
        # Taken back rather than assumed written in place: cv2 returns the flow,
        # and a call that resized would leave the buffer here holding the last.
        self._flow = self.algorithm.calc(prev, curr, self._flow)
        return _as_flow(torch.as_tensor(gpumat_to_cupy(self._flow)))


def _backend_for(paired: OpenCVAlgorithm) -> Backend:
    """Return the backend that runs `paired` where it was made.

    The test is the one `OpenCVAlgorithm` already made, so the two cannot come
    to different answers, and it narrows the algorithm to the concrete type its
    backend calls rather than the union either could have been.
    """
    algorithm = paired.algorithm
    if isinstance(algorithm, cv2.cuda.DenseOpticalFlow):
        return CUDABackend(algorithm, paired.device)

    return CPUBackend(algorithm)


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

    One class serves every algorithm and every device: which algorithm runs
    changes the settings, and which device runs it changes where a frame is put
    and how the answer comes back, neither of which is the streaming this holds.
    Build one through `OpenCVConfig.build`, which is what pairs an algorithm with
    the device it was made on.

    Separate from `OpticalFlowEstimator` so a future PyTorch (`nn.Module`)
    backend can extend the neutral base directly.

    Args:
        algorithm: The cv2 algorithm to stream through, with the device it was
            created on.

    Attributes:
        algorithm: The cv2 algorithm itself, which is where the settings it was
            made with can be read back from. Held by the backend that calls it
            rather than beside it, so a spy put on one is seen by both.
        device: As `OpticalFlowEstimator`, taken from `algorithm`.
        is_cuda: As `OpticalFlowEstimator`.
    """

    def __init__(self, algorithm: OpenCVAlgorithm) -> None:
        super().__init__(algorithm.device)
        self._backend = _backend_for(algorithm)

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
        """Forget the retained frame, restarting the sequence.

        The output buffer stays, being scratch the next call sizes for itself.
        """
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
        """Stream a chunk of frames, returning stacked flows continuing the sequence."""
        self.validate_device(frames)

        flows: list[Tensor] = []
        for frame in frames:
            flow = self._backend.push(frame)
            if flow is not None:
                flows.append(flow)

        return _stack_flows(flows, frames)

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

        flows = [self._backend.calc(p, c) for p, c in zip(prev, curr, strict=True)]

        return _stack_flows(flows, prev)
