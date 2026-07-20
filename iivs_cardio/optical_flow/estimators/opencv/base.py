from __future__ import annotations

__all__ = ("DenseOpticalFlow", "OpenCVEstimator")

from abc import abstractmethod
from typing import TYPE_CHECKING, cast, override

import cv2
import numpy as np
import torch
from beartype import beartype
from cv2.cuda import GpuMat
from jaxtyping import Float32, UInt8, jaxtyped
from torch import Tensor

from iivs_cardio.common.cuda_utils import gpumat_to_tensor, tensor_to_gpumat
from iivs_cardio.optical_flow.estimators.base import OpticalFlowEstimator

if TYPE_CHECKING:
    from numpy.typing import NDArray

DenseOpticalFlow = cv2.DenseOpticalFlow | cv2.cuda.DenseOpticalFlow

FrameType = UInt8[Tensor, "H W"]
FlowType = Float32[Tensor, "H W 2"]

BatchFrameType = UInt8[Tensor, "N H W"]
BatchFlowType = Float32[Tensor, "N H W 2"]
ChunkFlowType = Float32[Tensor, "M H W 2"]


class OpenCVEstimator(OpticalFlowEstimator):
    """Optical-flow estimators backed by OpenCV's `cv2` / `cv2.cuda` algorithms.

    Takes `(H, W)` uint8 frames and returns `(H, W, 2)` float32 flow as
    `torch.Tensor`s on `self.device`. A CUDA estimator keeps the whole
    computation on the device, so its output chains into the next GPU stage
    without a host transfer. Subclasses supply only `_create_algorithm`,
    choosing the concrete cv2 algorithm for the device.

    Separate from `OpticalFlowEstimator` so a future PyTorch (`nn.Module`)
    backend can extend the neutral base directly.
    """

    def __init__(self, device: str | torch.device = "cpu") -> None:
        super().__init__(device)
        self.select_device()
        self._algorithm = self._create_algorithm()

        if self.is_cuda:
            self._flow_buffer = GpuMat()
            self._frame_buffers = (GpuMat(), GpuMat())
            self._prev_slot = 0
        else:
            self._prev_frame: Tensor | None = None

    @abstractmethod
    def _create_algorithm(self) -> DenseOpticalFlow:
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
        if self.is_cuda:
            self._frame_buffers = (GpuMat(), GpuMat())
            self._prev_slot = 0
        else:
            self._prev_frame = None

    @jaxtyped(typechecker=beartype)
    @override
    def push(self, frame: FrameType) -> FlowType | None:
        self.validate_device(frame)
        if self.is_cuda:
            return self._push_cuda(frame)
        return self._push_cpu(frame)

    @jaxtyped(typechecker=beartype)
    @override
    def push_chunk(self, frames: BatchFrameType) -> ChunkFlowType:
        self.validate_device(frames)
        push = self._push_cuda if self.is_cuda else self._push_cpu
        flows = []
        for frame in frames:
            flow = push(frame)
            if flow is not None:
                flows.append(flow)
        if not flows:
            return frames.new_empty((0, *frames.shape[1:], 2), dtype=torch.float32)
        return torch.stack(flows)

    @jaxtyped(typechecker=beartype)
    @override
    def calc(self, prev: FrameType, curr: FrameType) -> FlowType:
        self.validate_device(prev)
        self.validate_device(curr)
        if self.is_cuda:
            return self._calc_cuda(prev, curr)
        return self._calc_cpu(prev, curr)

    @jaxtyped(typechecker=beartype)
    @override
    def calc_batch(self, prev: BatchFrameType, curr: BatchFrameType) -> BatchFlowType:
        self.validate_device(prev)
        self.validate_device(curr)
        if prev.shape[0] == 0:
            return prev.new_empty((0, *prev.shape[1:], 2), dtype=torch.float32)
        calc = self._calc_cuda if self.is_cuda else self._calc_cpu
        return torch.stack([calc(p, c) for p, c in zip(prev, curr, strict=True)])

    # ----------------------------- cpu (numpy) ----------------------------- #

    def _push_cpu(self, frame: Tensor) -> Tensor | None:
        prev, self._prev_frame = self._prev_frame, frame
        if prev is None:
            return None
        return self._calc_cpu(prev, frame)

    def _calc_cpu(self, prev: Tensor, curr: Tensor) -> Tensor:
        prev_np: NDArray[np.uint8] = prev.contiguous().numpy()
        curr_np: NDArray[np.uint8] = curr.contiguous().numpy()
        flow_np: NDArray[np.float32] = np.empty((*curr_np.shape, 2), np.float32)
        algorithm = cast("cv2.DenseOpticalFlow", self._algorithm)
        algorithm.calc(prev_np, curr_np, flow_np)
        return torch.from_numpy(flow_np)

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
        return gpumat_to_tensor(self._flow_buffer)
