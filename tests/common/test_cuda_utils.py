from __future__ import annotations

import cv2
import numpy as np
import pytest

# cupy and the interop module are imported inside the tests so that collection
# never imports cupy on a machine without a CUDA runtime.
requires_cuda = pytest.mark.skipif(
    cv2.cuda.getCudaEnabledDeviceCount() < 1,
    reason="no CUDA-capable GPU detected",
)


@requires_cuda
def test_gpumat_to_cupy_is_zero_copy():
    import cupy as cp

    from iivs_cardio.common.cuda_utils import gpumat_to_cupy

    host = np.arange(16 * 16, dtype=np.uint8).reshape(16, 16)
    gm = cv2.cuda.GpuMat()
    gm.upload(host)

    view = gpumat_to_cupy(gm)
    assert view.shape == (16, 16)
    assert view.dtype == cp.uint8
    assert int(view.data.ptr) == gm.cudaPtr()  # same device memory, no copy
    assert np.array_equal(cp.asnumpy(view), host)


@requires_cuda
def test_cupy_to_gpumat_roundtrips_uint8():
    import cupy as cp

    from iivs_cardio.common.cuda_utils import cupy_to_gpumat

    host = np.arange(16 * 16, dtype=np.uint8).reshape(16, 16)
    gm = cupy_to_gpumat(cp.asarray(host))
    assert np.array_equal(gm.download(), host)


@requires_cuda
def test_roundtrips_two_channel_flow():
    import cupy as cp

    from iivs_cardio.common.cuda_utils import cupy_to_gpumat, gpumat_to_cupy

    flow = np.random.default_rng(0).random((16, 16, 2)).astype(np.float32)
    gm = cupy_to_gpumat(cp.asarray(flow))  # 32FC2, padded rows
    assert np.allclose(cp.asnumpy(gpumat_to_cupy(gm)), flow)


@requires_cuda
def test_tensor_to_gpumat_roundtrips_uint8():
    import torch

    from iivs_cardio.common.cuda_utils import tensor_to_gpumat

    host = np.arange(16 * 16, dtype=np.uint8).reshape(16, 16)
    gm = tensor_to_gpumat(torch.as_tensor(host, device="cuda"))
    assert np.array_equal(gm.download(), host)


@requires_cuda
def test_tensor_to_gpumat_out_reuses_buffer():
    import torch

    from iivs_cardio.common.cuda_utils import tensor_to_gpumat

    gm = cv2.cuda.GpuMat()
    host = np.arange(16 * 16, dtype=np.uint8).reshape(16, 16)
    tensor_to_gpumat(torch.as_tensor(host, device="cuda"), out=gm)
    ptr = gm.cudaPtr()
    assert np.array_equal(gm.download(), host)

    # a second frame reuses the same buffer in place (no realloc)
    host2 = (host + 1).astype(np.uint8)
    tensor_to_gpumat(torch.as_tensor(host2, device="cuda"), out=gm)
    assert gm.cudaPtr() == ptr
    assert np.array_equal(gm.download(), host2)


@requires_cuda
def test_gpumat_to_tensor_owns_its_memory():
    from iivs_cardio.common.cuda_utils import gpumat_to_tensor

    host = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
    gm = cv2.cuda.GpuMat()
    gm.upload(host)

    tensor = gpumat_to_tensor(gm)
    assert tensor.device.type == "cuda"
    assert np.array_equal(tensor.cpu().numpy(), host)
    # clone(): the tensor owns a copy, not a view aliasing the GpuMat's memory.
    assert tensor.data_ptr() != gm.cudaPtr()


@requires_cuda
def test_tensor_gpumat_tensor_roundtrips_flow():
    import torch

    from iivs_cardio.common.cuda_utils import gpumat_to_tensor, tensor_to_gpumat

    flow = torch.as_tensor(
        np.random.default_rng(0).random((16, 16, 2)).astype(np.float32), device="cuda"
    )
    out = gpumat_to_tensor(tensor_to_gpumat(flow))
    assert out.shape == (16, 16, 2)
    assert torch.allclose(out.cpu(), flow.cpu())
