from __future__ import annotations

import numpy as np
import pytest

# A bare import: the OpenCV CUDA wheel resolves its own CUDA runtime, and cuDNN
# is expected to live in the CUDA toolkit's bin (see the Windows setup script).
# Skip the whole module where the CUDA build is not installed/importable (e.g. a
# GPU-less CI runner, or Windows before the cuDNN setup) rather than erroring at
# collection. A missing cuDNN surfaces as `ImportError: DLL load failed`.
try:
    import cv2
except (ImportError, OSError):
    cv2 = None

pytestmark = pytest.mark.skipif(
    cv2 is None,
    reason="OpenCV (CUDA build) is not importable in this environment",
)


def _cuda_device_count() -> int:
    return 0 if cv2 is None else cv2.cuda.getCudaEnabledDeviceCount()


# GPU-dependent checks skip the device work when no CUDA GPU is visible, so the
# suite still passes on CPU-only machines that can import cv2.
requires_gpu = pytest.mark.skipif(
    _cuda_device_count() < 1,
    reason="no CUDA-capable GPU detected in this environment",
)


def test_opencv_version() -> None:
    assert cv2.__version__.startswith("4.13")


def test_cuda_module_is_exposed() -> None:
    assert hasattr(cv2, "cuda")


@requires_gpu
def test_cuda_device_is_visible() -> None:
    assert _cuda_device_count() >= 1


@requires_gpu
def test_gpu_mat_upload_download_roundtrip() -> None:
    host = np.arange(256, dtype=np.uint8).reshape(16, 16)

    gpu = cv2.cuda.GpuMat()
    gpu.upload(host)
    result = gpu.download()

    assert result.shape == host.shape
    assert np.array_equal(result, host)


@requires_gpu
def test_gpu_cvtcolor_matches_cpu() -> None:
    rng = np.random.default_rng(0)
    host = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

    gpu = cv2.cuda.GpuMat()
    gpu.upload(host)
    gpu_gray = cv2.cuda.cvtColor(gpu, cv2.COLOR_BGR2GRAY).download()

    cpu_gray = cv2.cvtColor(host, cv2.COLOR_BGR2GRAY)

    assert gpu_gray.shape == cpu_gray.shape
    # The GPU and CPU colour-conversion paths must agree within rounding.
    max_delta = int(np.abs(gpu_gray.astype(np.int16) - cpu_gray.astype(np.int16)).max())
    assert max_delta <= 1
