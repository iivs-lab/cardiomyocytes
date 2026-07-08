"""Check the CUDA compute environment and PyTorch / TorchVision / OpenCV.

Run inside the project so it verifies the project's own environment:

    uv run scripts/compute_env/check_compute_env.py

It carries no PEP 723 inline metadata on purpose — `uv run` then resolves
`torch` / `torchvision` / `opencv-contrib-python` / `numpy` from the project
(`pyproject.toml` + `uv.lock`), so the check tests the exact pinned versions
the project uses instead of drifting to the newest wheels in a throwaway env.

CUDA is optional: each package must pass a CPU baseline (that is the exit-code
contract), and GPU work is verified only where a CUDA device is present. A
package fails only if it cannot import or its CPU baseline is wrong; a plain
CPU-only machine reports "No CUDA" and still exits 0. The GPU hardware is
enumerated once up front so the per-package sections need not repeat it.
"""

from __future__ import annotations

import numpy as np


def print_gpu_hardware() -> None:
    # Prefer torch (it exposes the device name); fall back to OpenCV's probe so
    # the hardware still shows when only OpenCV sees CUDA. Printed once so the
    # per-package sections carry only their own CUDA-linkage facts.
    try:
        import torch
    except (ImportError, OSError):
        torch = None

    if torch is not None and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            major, minor = torch.cuda.get_device_capability(i)
            vram = props.total_memory / (1024**3)
            print(f"   GPU {i}: {props.name} (cc {major}.{minor}, {vram:.2f} GB)")
        return

    try:
        import cv2
    except (ImportError, OSError):
        cv2 = None

    has_cv2_cuda = cv2 is not None and hasattr(cv2, "cuda")
    device_count = cv2.cuda.getCudaEnabledDeviceCount() if has_cv2_cuda else 0
    if device_count >= 1:
        for i in range(device_count):
            info = cv2.cuda.DeviceInfo(i)
            vram = info.totalMemory() / (1024**3)
            print(
                f"   GPU {i}: cc {info.majorVersion()}.{info.minorVersion()}, {vram:.2f} GB"
            )
        return

    print("   No CUDA GPU detected (CPU mode)")


def check_pytorch() -> bool:
    try:
        import torch
    except (ImportError, OSError) as exc:
        print(f"[!] PyTorch import failed: {exc}")
        return False

    cuda = torch.cuda.is_available()
    if cuda:
        linkage = f"CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}"
    else:
        linkage = "no CUDA (CPU mode)"
    print(f"PyTorch {torch.__version__} - {linkage}")

    # CPU baseline: matmul checked against a hand-computed value, not another
    # torch call. [[0,1,2],[3,4,5]] @ its transpose == [[5,14],[14,50]].
    mat = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    expected = torch.tensor([[5.0, 14.0], [14.0, 50.0]])
    cpu_out = mat @ mat.T
    ok = bool(torch.allclose(cpu_out, expected))
    print(f"   CPU matmul matches expected ... {'PASS' if ok else 'FAIL'}")

    if cuda:
        gpu_out = (mat.cuda() @ mat.cuda().T).cpu()
        gpu_ok = bool(torch.allclose(gpu_out, cpu_out))
        print(f"   CUDA matmul matches CPU ... {'PASS' if gpu_ok else 'FAIL'}")
        ok = ok and gpu_ok

    return ok


def check_torchvision() -> bool:
    try:
        import torch
        import torchvision
        from torchvision.ops import nms
    except (ImportError, OSError) as exc:
        print(f"[!] TorchVision import failed: {exc}")
        return False

    print(f"TorchVision {torchvision.__version__}")

    # CPU baseline: NMS exercises TorchVision's compiled ops. The first two
    # boxes overlap (IoU ~0.68 > 0.5) so the lower-scoring one is dropped; the
    # distant third box survives -> kept indices {0, 2}.
    boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [50.0, 50.0, 60.0, 60.0]]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    ok = sorted(nms(boxes, scores, iou_threshold=0.5).tolist()) == [0, 2]
    print(f"   CPU nms keeps expected boxes ... {'PASS' if ok else 'FAIL'}")

    if torch.cuda.is_available():
        kept_gpu = nms(boxes.cuda(), scores.cuda(), iou_threshold=0.5).cpu()
        gpu_ok = sorted(kept_gpu.tolist()) == [0, 2]
        print(f"   CUDA nms keeps expected boxes ... {'PASS' if gpu_ok else 'FAIL'}")
        ok = ok and gpu_ok

    return ok


def check_opencv() -> bool:
    try:
        import cv2
    except (ImportError, OSError) as exc:
        print(f"[!] OpenCV import failed: {exc}")
        print(
            "    On Windows, run scripts/compute_env/setup-opencv-cuda.ps1 as Administrator first."
        )
        return False

    # A plain CPU OpenCV can expose an empty `cv2.cuda` namespace, so read the
    # build information rather than trusting the module's presence.
    cuda_lines = [
        line for line in cv2.getBuildInformation().splitlines() if "NVIDIA CUDA" in line
    ]
    built_with_cuda = bool(cuda_lines) and "YES" in cuda_lines[0]
    device_count = cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, "cuda") else 0
    print(
        f"OpenCV {cv2.__version__} - built with CUDA: "
        f"{'Yes' if built_with_cuda else 'No'}, devices visible: {device_count}"
    )

    # CPU baseline: BGR->GRAY weights blue at 0.114, so pure blue (255 in the B
    # channel) maps to round(255 * 0.114) == 29.
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[:] = (255, 0, 0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ok = gray.shape == (2, 2) and int(gray[0, 0]) == 29
    print(f"   CPU cvtColor matches expected ... {'PASS' if ok else 'FAIL'}")

    if device_count >= 1:
        host = np.arange(256, dtype=np.uint8).reshape(16, 16)
        gpu = cv2.cuda.GpuMat()
        gpu.upload(host)
        roundtrip_ok = np.array_equal(gpu.download(), host)
        print(
            f"   GpuMat upload/download round-trip ... {'PASS' if roundtrip_ok else 'FAIL'}"
        )

        gpu_image = cv2.cuda.GpuMat()
        gpu_image.upload(image)
        gpu_gray = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY).download()
        cvt_ok = int(gpu_gray[0, 0]) == 29
        print(f"   CUDA cvtColor matches CPU ... {'PASS' if cvt_ok else 'FAIL'}")

        ok = ok and roundtrip_ok and cvt_ok

    return ok


print("=" * 60)
print("CUDA compute environment check")
print("=" * 60)

print("\n[ GPU hardware ]")
print_gpu_hardware()

results = {}
for name, check in (
    ("PyTorch", check_pytorch),
    ("TorchVision", check_torchvision),
    ("OpenCV", check_opencv),
):
    print(f"\n[ {name} ]")
    results[name] = check()

print("\n" + "=" * 60)
for name, ok in results.items():
    print(f"  {name:<12} {'OK' if ok else 'FAIL'}")
print("=" * 60)

raise SystemExit(0 if all(results.values()) else 1)
