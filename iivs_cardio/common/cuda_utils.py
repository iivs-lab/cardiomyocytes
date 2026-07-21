from __future__ import annotations

__all__ = ("cupy_to_gpumat", "gpumat_to_cupy", "gpumat_to_tensor", "tensor_to_gpumat")

from typing import TYPE_CHECKING, cast

import cupy as cp
import cv2
import torch

if TYPE_CHECKING:
    from cupy.cuda import MemoryPointer

# cv2.cuda.GpuMat exposes `cudaPtr()` (a cudawarped extension) and a padded row
# `step`, so a GpuMat is wrapped as a strided CuPy view rather than copied.
_DEPTH_TO_DTYPE: dict[int, type] = {cv2.CV_8U: cp.uint8, cv2.CV_32F: cp.float32}
_DTYPE_CH_TO_CVTYPE: dict[tuple[type, int], int] = {
    (cp.uint8, 1): cv2.CV_8UC1,
    (cp.float32, 1): cv2.CV_32FC1,
    (cp.float32, 2): cv2.CV_32FC2,
}


def _cupy_dtype(depth: int) -> type:
    """The CuPy scalar dtype for a GpuMat `depth`, or raise on an unsupported one."""
    try:
        return _DEPTH_TO_DTYPE[depth]
    except KeyError:
        msg = f"unsupported GpuMat depth {depth!r}: expected CV_8U or CV_32F"
        raise ValueError(msg) from None


def _cv_type(dtype: type, channels: int) -> int:
    """The cv2 type code for a `(dtype, channels)` pair, or raise on an unsupported one."""
    try:
        return _DTYPE_CH_TO_CVTYPE[dtype, channels]
    except KeyError:
        msg = (
            f"unsupported (dtype, channels) ({dtype.__name__}, {channels}): expected "
            "(uint8, 1), (float32, 1), or (float32, 2)"
        )
        raise ValueError(msg) from None


def gpumat_to_cupy(gm: cv2.cuda.GpuMat) -> cp.ndarray:
    """Zero-copy view of a `cv2.cuda.GpuMat` as a CuPy array.

    The GpuMat's device memory is wrapped (not copied); its row padding (`step`)
    is honored through the CuPy strides. The view stays valid only while `gm`
    lives — `gm` is held as the memory's owner to keep it alive.
    """
    width, height = gm.size()
    channels = gm.channels()
    dtype = _cupy_dtype(gm.depth())
    itemsize = cp.dtype(dtype).itemsize
    step = gm.step  # bytes per row, padded for alignment

    memory = cp.cuda.UnownedMemory(gm.cudaPtr(), step * height, owner=gm)
    pointer = cast("MemoryPointer", cp.cuda.MemoryPointer(memory, 0))
    if channels == 1:
        return cp.ndarray(
            (height, width), dtype, memptr=pointer, strides=(step, itemsize)
        )
    return cp.ndarray(
        (height, width, channels),
        dtype,
        memptr=pointer,
        strides=(step, channels * itemsize, itemsize),
    )


def cupy_to_gpumat(arr: cp.ndarray) -> cv2.cuda.GpuMat:
    """Copy a CuPy array into a fresh `cv2.cuda.GpuMat`, device-to-device.

    Allocates a GpuMat of matching shape/dtype and copies `arr` into it on the
    device (no host round-trip). Accepts `(H, W)` or `(H, W, C)` arrays.
    """
    height, width = arr.shape[:2]
    channels = 1 if arr.ndim == 2 else arr.shape[2]
    gm = cv2.cuda.GpuMat(height, width, _cv_type(arr.dtype.type, channels))
    gpumat_to_cupy(gm)[...] = arr
    return gm


def tensor_to_gpumat(
    tensor: torch.Tensor, out: cv2.cuda.GpuMat | None = None
) -> cv2.cuda.GpuMat:
    """Copy a CUDA `torch.Tensor` into a `cv2.cuda.GpuMat`, device-to-device.

    Copies into `out` in place when given — sizing it to `tensor` (a no-op when it
    already matches), so a reused buffer skips a per-call allocation — otherwise
    allocates a fresh GpuMat. Accepts `(H, W)` or `(H, W, C)` tensors; `tensor`
    must live on a CUDA device (else `cp.asarray` would silently host->device copy).
    """
    if not tensor.is_cuda:
        msg = f"tensor_to_gpumat expects a CUDA tensor, got one on {tensor.device}"
        raise ValueError(msg)
    arr = cp.ascontiguousarray(cp.asarray(tensor))
    height, width = arr.shape[:2]
    channels = 1 if arr.ndim == 2 else arr.shape[2]
    if out is None:
        out = cv2.cuda.GpuMat()
    out.create(height, width, _cv_type(arr.dtype.type, channels))
    gpumat_to_cupy(out)[...] = arr
    return out


def gpumat_to_tensor(gm: cv2.cuda.GpuMat) -> torch.Tensor:
    """Copy a `cv2.cuda.GpuMat` into a CUDA `torch.Tensor` that owns its memory.

    The GpuMat is viewed as a CuPy array (zero-copy) and cloned into a tensor on
    the same device, so the result stays valid after `gm` is freed or reused.
    """
    return torch.as_tensor(gpumat_to_cupy(gm)).clone()
