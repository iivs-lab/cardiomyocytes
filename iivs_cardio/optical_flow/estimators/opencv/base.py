from __future__ import annotations

__all__ = ("DenseAlgorithm", "OpenCVAlgorithm", "OpenCVConfig", "OpenCVEstimator")

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast, override

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
    asked for it, and nothing on the object says which that was. Carrying it
    beside the algorithm is what lets an estimator hold the pair rather than
    trust a caller to have activated the right one first: the two are set
    together, by the one place that activates and creates in that order.

    Attributes:
        algorithm: The cv2 algorithm, CPU or CUDA.
        device: The device it was created on.
    """

    algorithm: DenseAlgorithm
    device: Device


class OpenCVConfig(EstimatorConfig, ABC):
    """The settings of one cv2 flow algorithm, and how to make it on a device.

    Every OpenCV estimator is built through this one `build`. A subclass adds
    the parameters, the factory that takes them, and `SUPPORTED_DEVICES` where
    the algorithm has no CUDA implementation. There is deliberately no class
    per algorithm below `OpenCVEstimator`: what differs between them is the
    parameters and which factory reads them, not the machinery that streams
    frames through, so the algorithm itself fills the seat `FilterKernel` fills
    on the filtering side.
    """

    @abstractmethod
    def create(self, device: Device) -> DenseAlgorithm:
        """Create the cv2 algorithm these settings describe, for `device`.

        Called with `device` already current, so an implementation chooses the
        CUDA factory rather than binding anything itself.

        Args:
            device: The device to create for, resolved and activated.

        Returns:
            The algorithm, which `build` pairs with the device it was made on.
        """

    @override
    def build(self, device: DeviceLike = "cpu") -> OpenCVEstimator:
        """Build the estimator these settings describe, on `device`.

        The order is what this exists to hold: the device is checked against
        what the algorithm has an implementation for, made current, and only
        then asked for one. A caller assembling the pair itself would carry
        that order in its head, and nothing would say so when it did not.

        Raises:
            ValueError: If `device` is not one of `SUPPORTED_DEVICES`.
        """
        resolved = Device.resolve(device, self.SUPPORTED_DEVICES)
        resolved.activate()

        return OpenCVEstimator(OpenCVAlgorithm(self.create(resolved), resolved))


def _stack_flows(flows: list[Tensor], frames: Tensor) -> Tensor:
    """Stack the flows, or an empty `(0, 2, H, W)` float32 when there are none."""
    if not flows:
        return frames.new_empty((0, 2, *frames.shape[1:]), dtype=torch.float32)
    return torch.stack(flows)


class OpenCVEstimator(OpticalFlowEstimator):
    """Optical-flow estimators backed by OpenCV's `cv2` / `cv2.cuda` algorithms.

    Takes `(H, W)` uint8 frames and returns `(2, H, W)` float32 flow (channel 0 =
    dx, channel 1 = dy) as `torch.Tensor`s on `self.device`. cv2 computes flow in
    `(H, W, 2)`; the output is transposed once to the channel-first layout that
    torch spatial ops (`grid_sample`, `conv2d`) consume natively. A CUDA estimator
    keeps the whole computation on the device, so its output chains into the next
    GPU stage without a host transfer.

    One class serves every algorithm: which one it streams is a value it is
    given rather than a subclass of it, the way `FilteredSequence` takes the
    kernel it reduces with. `OpenCVConfig.build` is what makes the pair, so an
    algorithm and the device it was created on cannot arrive apart.

    Separate from `OpticalFlowEstimator` so a future PyTorch (`nn.Module`)
    backend can extend the neutral base directly.

    Args:
        algorithm: The cv2 algorithm to stream through, with the device it was
            created on.

    Attributes:
        algorithm: The cv2 algorithm itself, public the way `FilteredSequence`
            leaves its kernel public: what is plugged in is worth reading, and
            it is the one place a caller can ask cv2 what it was configured
            with.
        device: The device it runs on, as `OpticalFlowEstimator` holds it.
    """

    def __init__(self, algorithm: OpenCVAlgorithm) -> None:
        super().__init__(algorithm.device)
        self.algorithm = algorithm.algorithm

        if self.is_cuda:
            self._flow_buffer = GpuMat()
            self._frame_buffers = (GpuMat(), GpuMat())
            self._prev_slot = 0
        else:
            self._prev_frame: Tensor | None = None

    def validate_device(self, frame: Tensor) -> None:
        """Raise if `frame` is not on this estimator's device.

        The algorithm names itself in the refusal rather than the estimator,
        which is one class for all of them and would say the same thing however
        the run was configured.
        """
        if frame.device != self.device.as_torch:
            name = type(self.algorithm).__name__
            msg = f"{name} expects a {self.device} tensor, got one on {frame.device}"
            raise ValueError(msg)

    @override
    def reset(self) -> None:
        """Forget the retained frame and CUDA buffers, restarting the sequence."""
        if self.is_cuda:
            self._frame_buffers = (GpuMat(), GpuMat())
            self._prev_slot = 0
        else:
            self._prev_frame = None

    @jaxtyped(typechecker=beartype)
    @override
    def push(self, frame: FrameType) -> FlowType | None:
        """Return the flow from the retained frame, or `None` on the first frame."""
        self.validate_device(frame)
        push = self._push_cuda if self.is_cuda else self._push_cpu
        return push(frame)

    @jaxtyped(typechecker=beartype)
    @override
    def push_chunk(self, frames: BatchFrameType) -> ChunkFlowType:
        """Stream a chunk of frames, returning stacked flows continuing the sequence."""
        self.validate_device(frames)
        push = self._push_cuda if self.is_cuda else self._push_cpu
        flows: list[Tensor] = []
        for frame in frames:
            flow = push(frame)
            if flow is not None:
                flows.append(flow)
        return _stack_flows(flows, frames)

    @jaxtyped(typechecker=beartype)
    @override
    def calc(self, prev: FrameType, curr: FrameType) -> FlowType:
        """Compute the flow `prev -> curr` in one shot, leaving no retained state."""
        self.validate_device(prev)
        self.validate_device(curr)
        calc = self._calc_cuda if self.is_cuda else self._calc_cpu
        return calc(prev, curr)

    @jaxtyped(typechecker=beartype)
    @override
    def calc_batch(self, prev: BatchFrameType, curr: BatchFrameType) -> BatchFlowType:
        """Compute the flow for each independent pair `prev[i] -> curr[i]`, stacked."""
        self.validate_device(prev)
        self.validate_device(curr)
        calc = self._calc_cuda if self.is_cuda else self._calc_cpu
        flows = [calc(p, c) for p, c in zip(prev, curr, strict=True)]
        return _stack_flows(flows, prev)

    # ----------------------------- cpu (numpy) ----------------------------- #

    def _push_cpu(self, frame: Tensor) -> Tensor | None:
        prev, self._prev_frame = self._prev_frame, frame
        if prev is None:
            return None
        return self._calc_cpu(prev, frame)

    def _calc_cpu(self, prev: Tensor, curr: Tensor) -> Tensor:
        prev_np: NDArray[np.uint8] = prev.contiguous().numpy()
        curr_np: NDArray[np.uint8] = curr.contiguous().numpy()
        algorithm = cast("cv2.DenseOpticalFlow", self.algorithm)
        flow: NDArray[np.float32] = algorithm.calc(prev_np, curr_np, None)  # ty: ignore[no-matching-overload]
        return torch.from_numpy(flow).permute(2, 0, 1).contiguous()

    # -------------------- cuda (GpuMat via cuda_utils) --------------------- #

    def _push_cuda(self, frame: Tensor) -> Tensor | None:
        self.device.activate()  # the GpuMat/CuPy calls below read the global device
        prev = self._frame_buffers[self._prev_slot]
        curr = self._frame_buffers[self._prev_slot ^ 1]
        tensor_to_gpumat(frame, out=curr)
        self._prev_slot ^= 1
        if prev.empty():
            return None
        return self._calc_cuda_core(prev, curr)

    def _calc_cuda(self, prev: Tensor, curr: Tensor) -> Tensor:
        self.device.activate()
        prev_cv = tensor_to_gpumat(prev)
        curr_cv = tensor_to_gpumat(curr)
        return self._calc_cuda_core(prev_cv, curr_cv)

    def _calc_cuda_core(self, prev: GpuMat, curr: GpuMat) -> Tensor:
        if self._flow_buffer.size() != prev.size():
            self._flow_buffer = GpuMat(prev.size(), cv2.CV_32FC2)
        algorithm = cast("cv2.cuda.DenseOpticalFlow", self.algorithm)
        algorithm.calc(prev, curr, self._flow_buffer)
        flow = torch.as_tensor(gpumat_to_cupy(self._flow_buffer))
        return flow.permute(2, 0, 1).contiguous()
