from __future__ import annotations

__all__ = ("OpenCVAlgorithm", "OpenCVEstimator")

from abc import abstractmethod
from typing import TYPE_CHECKING, cast, override

import cv2
import torch
from beartype import beartype
from cv2.cuda import GpuMat
from jaxtyping import Float32, UInt8, jaxtyped
from torch import Tensor

from iivs_cardio.common.cuda_utils import gpumat_to_cupy, tensor_to_gpumat
from iivs_cardio.optical_flow.estimators.base import OpticalFlowEstimator

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

OpenCVAlgorithm = cv2.DenseOpticalFlow | cv2.cuda.DenseOpticalFlow

FrameType = UInt8[Tensor, "H W"]
FlowType = Float32[Tensor, "2 H W"]

BatchFrameType = UInt8[Tensor, "N H W"]
BatchFlowType = Float32[Tensor, "N 2 H W"]
ChunkFlowType = Float32[Tensor, "M 2 H W"]


class OpenCVEstimator(OpticalFlowEstimator):
    """Optical-flow estimators backed by OpenCV's `cv2` / `cv2.cuda` algorithms.

    Takes `(H, W)` uint8 frames and returns `(2, H, W)` float32 flow (channel 0 =
    dx, channel 1 = dy) as `torch.Tensor`s on `self.device`. cv2 computes flow in
    `(H, W, 2)`; the output is transposed once to the channel-first layout that
    torch spatial ops (`grid_sample`, `conv2d`) consume natively. A CUDA estimator
    keeps the whole computation on the device, so its output chains into the next
    GPU stage without a host transfer. Subclasses supply only `_create_algorithm`,
    choosing the concrete cv2 algorithm for the device.

    Separate from `OpticalFlowEstimator` so a future PyTorch (`nn.Module`)
    backend can extend the neutral base directly.
    """

    def __init__(self, device: str | torch.device = "cpu") -> None:
        super().__init__(device)
        self.select_device()
        self._algorithm: OpenCVAlgorithm = self._create_algorithm()

        if self.is_cuda:
            self._flow_buffer = GpuMat()
            self._frame_buffers = (GpuMat(), GpuMat())
            self._prev_slot = 0
        else:
            self._prev_frame: Tensor | None = None

    @abstractmethod
    def _create_algorithm(self) -> OpenCVAlgorithm:
        """Create the cv2 flow algorithm for `self.device`."""

    def select_device(self) -> None:
        """Pin cv2.cuda's process-global current device to this estimator's GPU."""
        if self.is_cuda and self.device.index is not None:
            cv2.cuda.setDevice(self.device.index)

    def validate_device(self, frame: Tensor) -> None:
        """Raise if `frame` is not on this estimator's device."""
        if frame.device != self.device:
            name = type(self).__name__
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
        return self._stack_flows(flows, frames)

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
        return self._stack_flows(flows, prev)

    @staticmethod
    def _stack_flows(flows: list[Tensor], frames: Tensor) -> Tensor:
        """Stack the flows, or an empty `(0, 2, H, W)` float32 when there are none."""
        if not flows:
            return frames.new_empty((0, 2, *frames.shape[1:]), dtype=torch.float32)
        return torch.stack(flows)

    # ----------------------------- cpu (numpy) ----------------------------- #

    def _push_cpu(self, frame: Tensor) -> Tensor | None:
        prev, self._prev_frame = self._prev_frame, frame
        if prev is None:
            return None
        return self._calc_cpu(prev, frame)

    def _calc_cpu(self, prev: Tensor, curr: Tensor) -> Tensor:
        prev_np: NDArray[np.uint8] = prev.contiguous().numpy()
        curr_np: NDArray[np.uint8] = curr.contiguous().numpy()
        algorithm = cast("cv2.DenseOpticalFlow", self._algorithm)
        flow: NDArray[np.float32] = algorithm.calc(prev_np, curr_np, None)  # ty: ignore[no-matching-overload]
        return torch.from_numpy(flow).permute(2, 0, 1).contiguous()

    # -------------------- cuda (GpuMat via cuda_utils) --------------------- #

    def _push_cuda(self, frame: Tensor) -> Tensor | None:
        self.select_device()
        prev = self._frame_buffers[self._prev_slot]
        curr = self._frame_buffers[self._prev_slot ^ 1]
        tensor_to_gpumat(frame, out=curr)
        self._prev_slot ^= 1
        if prev.empty():
            return None
        return self._calc_cuda_core(prev, curr)

    def _calc_cuda(self, prev: Tensor, curr: Tensor) -> Tensor:
        self.select_device()
        prev_cv = tensor_to_gpumat(prev)
        curr_cv = tensor_to_gpumat(curr)
        return self._calc_cuda_core(prev_cv, curr_cv)

    def _calc_cuda_core(self, prev: GpuMat, curr: GpuMat) -> Tensor:
        if self._flow_buffer.size() != prev.size():
            self._flow_buffer = GpuMat(prev.size(), cv2.CV_32FC2)
        algorithm = cast("cv2.cuda.DenseOpticalFlow", self._algorithm)
        algorithm.calc(prev, curr, self._flow_buffer)
        flow = torch.as_tensor(gpumat_to_cupy(self._flow_buffer))
        return flow.permute(2, 0, 1).contiguous()
