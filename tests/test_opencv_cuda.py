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
def test_build_information_reports_cuda() -> None:
    # Confirms the installed wheel is genuinely CUDA-compiled, not a plain
    # (CPU) OpenCV that happens to expose an empty `cv2.cuda` namespace.
    cuda_lines = [
        line for line in cv2.getBuildInformation().splitlines() if "NVIDIA CUDA" in line
    ]
    assert cuda_lines, "build information reports no NVIDIA CUDA entry"
    assert "YES" in cuda_lines[0]


@requires_gpu
def test_cuda_device_is_visible() -> None:
    assert _cuda_device_count() >= 1


@requires_gpu
def test_device_info_reports_compatible_gpu() -> None:
    info = cv2.cuda.DeviceInfo(0)

    assert info.isCompatible()
    assert info.majorVersion() >= 1  # compute-capability major version
    assert info.totalMemory() > 0
    assert info.multiProcessorCount() > 0


@requires_gpu
def test_gpu_mat_upload_download_roundtrip() -> None:
    host = np.arange(256, dtype=np.uint8).reshape(16, 16)

    gpu = cv2.cuda.GpuMat()
    gpu.upload(host)

    # GpuMat metadata reflects the uploaded array.
    assert gpu.size() == (host.shape[1], host.shape[0])  # (width, height)
    assert gpu.channels() == 1
    assert gpu.type() == cv2.CV_8UC1

    result = gpu.download()

    assert result.shape == host.shape
    assert result.dtype == host.dtype
    assert np.array_equal(result, host)


@requires_gpu
def test_gpu_add_saturates_like_numpy() -> None:
    a = np.array([[0, 100, 200], [50, 150, 255]], dtype=np.uint8)
    b = np.array([[0, 100, 100], [10, 150, 1]], dtype=np.uint8)

    gpu_a = cv2.cuda.GpuMat()
    gpu_a.upload(a)
    gpu_b = cv2.cuda.GpuMat()
    gpu_b.upload(b)
    result = cv2.cuda.add(gpu_a, gpu_b).download()

    # uint8 addition saturates at 255; check against an independently computed
    # value, not another OpenCV call.
    expected = np.clip(a.astype(np.int16) + b.astype(np.int16), 0, 255).astype(np.uint8)
    assert np.array_equal(result, expected)


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
