from __future__ import annotations

__all__ = ("cupy_to_gpumat", "gpumat_to_cupy", "gpumat_to_tensor", "tensor_to_gpumat")

import importlib
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch

if TYPE_CHECKING:
    from types import ModuleType

# `cupy` is an optional CUDA-only dependency (the `cuda` extra). It is imported
# lazily so this module — and everything that imports it (e.g. the OpenCV flow
# estimators, whose CPU path never touches CUDA) — stays importable without it.
# Its arrays are typed `Any` so type-checking never needs the package either.
CupyArray = Any

# cv2.cuda.GpuMat exposes `cudaPtr()` (a cudawarped extension) and a padded row
# `step`, so a GpuMat is wrapped as a strided CuPy view rather than copied.
# Keyed by NumPy dtypes (cupy reuses them via `array.dtype.type`) so the tables
# need no cupy import at module load.
_DEPTH_TO_DTYPE = {cv2.CV_8U: np.uint8, cv2.CV_32F: np.float32}
_DTYPE_CH_TO_CVTYPE = {
    (np.uint8, 1): cv2.CV_8UC1,
    (np.float32, 1): cv2.CV_32FC1,
    (np.float32, 2): cv2.CV_32FC2,
}


def _require_cupy() -> ModuleType:
    # Imported via importlib (not a static `import cupy`) so type-checking never
    # tries to resolve the optional package on a CPU-only install / CI.
    try:
        return importlib.import_module("cupy")
    except ImportError as exc:
        msg = (
            "cupy is required for CUDA GpuMat interop but is not installed; "
            "install the 'cuda' extra on a CUDA machine (uv sync --extra cuda)"
        )
        raise ImportError(msg) from exc


def gpumat_to_cupy(gm: cv2.cuda.GpuMat) -> CupyArray:
    """Zero-copy view of a `cv2.cuda.GpuMat` as a CuPy array.

    The GpuMat's device memory is wrapped (not copied); its row padding (`step`)
    is honored through the CuPy strides. The view stays valid only while `gm`
    lives — `gm` is held as the memory's owner to keep it alive.
    """
    cp = _require_cupy()
    width, height = gm.size()
    channels = gm.channels()
    dtype = _DEPTH_TO_DTYPE[gm.depth()]
    itemsize = np.dtype(dtype).itemsize
    step = gm.step  # bytes per row, padded for alignment

    memory = cp.cuda.UnownedMemory(gm.cudaPtr(), step * height, owner=gm)
    pointer = cp.cuda.MemoryPointer(memory, 0)
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


def cupy_to_gpumat(arr: CupyArray) -> cv2.cuda.GpuMat:
    """Copy a CuPy array into a fresh `cv2.cuda.GpuMat`, device-to-device.

    Allocates a GpuMat of matching shape/dtype and copies `arr` into it on the
    device (no host round-trip). Accepts `(H, W)` or `(H, W, C)` arrays.
    """
    height, width = arr.shape[:2]
    channels = 1 if arr.ndim == 2 else arr.shape[2]
    gm = cv2.cuda.GpuMat(height, width, _DTYPE_CH_TO_CVTYPE[arr.dtype.type, channels])
    gpumat_to_cupy(gm)[...] = arr
    return gm


def tensor_to_gpumat(
    tensor: torch.Tensor, out: cv2.cuda.GpuMat | None = None
) -> cv2.cuda.GpuMat:
    """Copy a CUDA `torch.Tensor` into a `cv2.cuda.GpuMat`, device-to-device.

    Copies into `out` in place when given — sizing it to `tensor` (a no-op when it
    already matches), so a reused buffer skips a per-call allocation — otherwise
    allocates a fresh GpuMat. Accepts `(H, W)` or `(H, W, C)` tensors; `tensor`
    must live on a CUDA device.
    """
    cp = _require_cupy()
    arr = cp.ascontiguousarray(cp.asarray(tensor))
    height, width = arr.shape[:2]
    channels = 1 if arr.ndim == 2 else arr.shape[2]
    if out is None:
        out = cv2.cuda.GpuMat()
    out.create(height, width, _DTYPE_CH_TO_CVTYPE[arr.dtype.type, channels])
    gpumat_to_cupy(out)[...] = arr
    return out


def gpumat_to_tensor(gm: cv2.cuda.GpuMat) -> torch.Tensor:
    """Copy a `cv2.cuda.GpuMat` into a CUDA `torch.Tensor` that owns its memory.

    The GpuMat is viewed as a CuPy array (zero-copy) and cloned into a tensor on
    the same device, so the result stays valid after `gm` is freed or reused.
    """
    return torch.as_tensor(gpumat_to_cupy(gm)).clone()
