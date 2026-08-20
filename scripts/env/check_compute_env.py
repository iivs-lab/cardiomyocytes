import sys
import textwrap
from dataclasses import dataclass

import numpy as np

_RULE = "=" * 60
_STEP = "   "
_DETAIL = "      "

_CONV_LABEL = "CUDA conv2d (cuDNN) matches CPU"

_SETUP_SCRIPT = "scripts/env/setup-opencv-cuda.ps1"

# One soname is loaded once per process, so a foreign cuDNN reaches torch the
# same way on either platform. Where it comes from, and what takes it back, is
# what differs.
if sys.platform == "win32":
    _CUDNN_FAULT = f"""\
Works with cuDNN disabled, so cuDNN itself is the cause. Usually a
foreign cuDNN shadowing the one torch bundles in torch/lib. On
Windows the OpenCV setup is the usual source -- preview and repair:
  {_SETUP_SCRIPT} -DryRun
  {_SETUP_SCRIPT}     (as Administrator)"""

    _OPENCV_FIX = f"    On Windows, run {_SETUP_SCRIPT} as Administrator first."
else:
    _CUDNN_FAULT = """\
Works with cuDNN disabled, so cuDNN itself is the cause. Usually a
foreign libcudnn reaching the process before the one torch carries in
torch/lib, where the first to arrive is the one every consumer gets.
Look for a system copy, and for one LD_LIBRARY_PATH puts ahead of it:
  ldconfig -p | grep libcudnn
  echo $LD_LIBRARY_PATH"""

    _OPENCV_FIX = "    Check the loader reaches CUDA: ldconfig -p | grep libcudart"

_DRIVER_FAULT = """\
Fails with cuDNN disabled too, so cuDNN is not the cause --
check the NVIDIA driver, the CUDA runtime, or the torch build."""


@dataclass(frozen=True, slots=True)
class Check:
    label: str
    passed: bool
    detail: str = ""  # written under the line, indented, when there is any


@dataclass(frozen=True, slots=True)
class Section:
    name: str
    headline: str
    checks: tuple[Check, ...] = ()
    reached: bool = True  # False where the package could not be read at all

    @property
    def passed(self) -> bool:
        return self.reached and all(check.passed for check in self.checks)


def _unreachable(name: str, headline: str) -> Section:
    return Section(name, headline, reached=False)


def _import_failed(name: str, exc: BaseException, fix: str = "") -> Section:
    headline = f"[!] {name} import failed: {exc}"

    return _unreachable(name, f"{headline}\n{fix}" if fix else headline)


def _render(section: Section) -> None:
    print(f"\n[ {section.name} ]")
    print(section.headline)

    for check in section.checks:
        print(f"{_STEP}{check.label} ... {'PASS' if check.passed else 'FAIL'}")
        if check.detail:
            print(textwrap.indent(check.detail, _DETAIL))


def _torch_gpus() -> list[str]:
    # Returning out of the `except` keeps the name bound on every path that reads
    # it, where a `torch = None` fallback would give one name two types.
    try:
        import torch
    except (ImportError, OSError):
        return []

    if not torch.cuda.is_available():
        return []

    lines = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        major, minor = torch.cuda.get_device_capability(i)
        vram = props.total_memory / (1024**3)
        lines.append(f"GPU {i}: {props.name} (cc {major}.{minor}, {vram:.2f} GB)")

    return lines


def _opencv_gpus() -> list[str]:
    try:
        import cv2
    except (ImportError, OSError):
        return []

    lines = []
    for i in range(_opencv_device_count()):
        info = cv2.cuda.DeviceInfo(i)
        capability = f"{info.majorVersion()}.{info.minorVersion()}"
        vram = info.totalMemory() / (1024**3)
        lines.append(f"GPU {i}: cc {capability}, {vram:.2f} GB")

    return lines


def _opencv_device_count() -> int:
    # A CPU-only OpenCV can still expose an empty `cv2.cuda`, so ask before using it.
    import cv2

    return cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, "cuda") else 0


def print_gpu_hardware() -> None:
    # torch first, since it is the one that names the device; OpenCV's probe is
    # the fallback, so the hardware still shows when only OpenCV sees CUDA.
    lines = _torch_gpus() or _opencv_gpus() or ["No CUDA GPU detected (CPU mode)"]

    print("\n[ GPU hardware ]")
    for line in lines:
        print(f"{_STEP}{line}")


def _conv_works_without_cudnn() -> bool:
    import torch

    image = torch.zeros(1, 1, 3, 3, device="cuda")
    box = torch.ones(1, 1, 2, 2, device="cuda")
    torch.backends.cudnn.enabled = False
    try:
        torch.nn.functional.conv2d(image, box)
        torch.cuda.synchronize()
    except RuntimeError:
        return False
    finally:
        torch.backends.cudnn.enabled = True
    return True


def check_pytorch() -> Section:
    try:
        import torch
    except (ImportError, OSError) as exc:
        return _import_failed("PyTorch", exc)

    cuda = torch.cuda.is_available()
    if cuda:
        linkage = f"CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}"
    else:
        linkage = "no CUDA (CPU mode)"

    headline = f"PyTorch {torch.__version__} - {linkage}"

    # [[0,1,2],[3,4,5]] @ its transpose == [[5,14],[14,50]], worked out by hand
    # rather than taken from another torch call.
    mat = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    expected = torch.tensor([[5.0, 14.0], [14.0, 50.0]])
    cpu_out = mat @ mat.T
    matmul_ok = bool(torch.allclose(cpu_out, expected))
    checks = [Check("CPU matmul matches expected", passed=matmul_ok)]

    if cuda:
        gpu_out = (mat.cuda() @ mat.cuda().T).cpu()
        matches = bool(torch.allclose(gpu_out, cpu_out))
        checks.append(Check("CUDA matmul matches CPU", passed=matches))

    # A 3x3 ramp under a 2x2 box sums each window: [[8,12],[20,24]].
    image = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
    box = torch.ones(1, 1, 2, 2)
    cpu_conv = torch.nn.functional.conv2d(image, box)[0, 0]
    conv_ok = bool(torch.allclose(cpu_conv, torch.tensor([[8.0, 12.0], [20.0, 24.0]])))
    checks.append(Check("CPU conv2d matches expected", passed=conv_ok))

    if cuda:
        # The only op checked here that goes through cuDNN, so it is what
        # catches one that is broken or shadowed by a foreign build.
        try:
            gpu_conv = torch.nn.functional.conv2d(image.cuda(), box.cuda()).cpu()[0, 0]
            matches = bool(torch.allclose(gpu_conv, cpu_conv))
            checks.append(Check(_CONV_LABEL, passed=matches))
        except RuntimeError as exc:
            excerpt = str(exc).splitlines()[0][:88]
            fault = _CUDNN_FAULT if _conv_works_without_cudnn() else _DRIVER_FAULT
            detail = f"{excerpt}\n{fault}"
            checks.append(Check(_CONV_LABEL, passed=False, detail=detail))

    return Section("PyTorch", headline, tuple(checks))


def check_torchvision() -> Section:
    try:
        import torch
        import torchvision
        from torchvision.ops import nms
    except (ImportError, OSError) as exc:
        return _import_failed("TorchVision", exc)

    headline = f"TorchVision {torchvision.__version__}"

    # NMS exercises TorchVision's compiled ops. The first two boxes overlap
    # (IoU ~0.68 > 0.5) and the third is distant, so indices {0, 2} survive.
    boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [50.0, 50.0, 60.0, 60.0]]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    kept = sorted(nms(boxes, scores, iou_threshold=0.5).tolist())
    checks = [Check("CPU nms keeps expected boxes", passed=kept == [0, 2])]

    if torch.cuda.is_available():
        on_gpu = sorted(nms(boxes.cuda(), scores.cuda(), iou_threshold=0.5).tolist())
        checks.append(Check("CUDA nms keeps expected boxes", passed=on_gpu == [0, 2]))

    return Section("TorchVision", headline, tuple(checks))


def check_opencv() -> Section:
    try:
        import cv2
    except (ImportError, OSError) as exc:
        return _import_failed("OpenCV", exc, _OPENCV_FIX)

    # A CPU-only OpenCV can expose an empty `cv2.cuda`, so read the build itself.
    cuda_lines = [
        line for line in cv2.getBuildInformation().splitlines() if "NVIDIA CUDA" in line
    ]
    built = "Yes" if cuda_lines and "YES" in cuda_lines[0] else "No"
    device_count = _opencv_device_count()
    linkage = f"built with CUDA: {built}, devices visible: {device_count}"
    headline = f"OpenCV {cv2.__version__} - {linkage}"

    # BGR->GRAY weights blue at 0.114, so pure blue maps to round(255 * 0.114) == 29.
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[:] = (255, 0, 0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cvt_ok = gray.shape == (2, 2) and int(gray[0, 0]) == 29
    checks = [Check("CPU cvtColor matches expected", passed=cvt_ok)]

    if device_count >= 1:
        host = np.arange(256, dtype=np.uint8).reshape(16, 16)
        gpu = cv2.cuda.GpuMat()
        gpu.upload(host)
        returned = bool(np.array_equal(gpu.download(), host))
        checks.append(Check("GpuMat upload/download round-trip", passed=returned))

        gpu_image = cv2.cuda.GpuMat()
        gpu_image.upload(image)
        gpu_gray = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY).download()
        gpu_cvt_ok = int(gpu_gray[0, 0]) == 29
        checks.append(Check("CUDA cvtColor matches CPU", passed=gpu_cvt_ok))

    return Section("OpenCV", headline, tuple(checks))


def _cupy_torch_interop() -> list[Check]:
    # `common/cuda_utils.py` wraps a torch tensor as a CuPy array and back
    # without copying, which holds only while both read one
    # `__cuda_array_interface__`. A pointer that differs means a silent copy.
    import cupy as cp

    try:
        import torch
    except (ImportError, OSError):
        return []

    if not torch.cuda.is_available():
        return []

    tensor = torch.arange(6, dtype=torch.float32, device="cuda")
    view = cp.asarray(tensor)
    wrapped = view.data.ptr == tensor.data_ptr()

    back = torch.as_tensor(view)
    returned = back.data_ptr() == view.data.ptr and bool(torch.equal(back, tensor))

    return [
        Check("torch tensor wraps as a CuPy view", passed=wrapped),
        Check("CuPy array wraps back as a torch tensor", passed=returned),
    ]


def check_cupy() -> Section:
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        return _import_failed("CuPy", exc)

    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        return _unreachable("CuPy", f"[!] CuPy reached no CUDA runtime: {exc}")

    if device_count < 1:
        missing = "[!] CuPy sees no CUDA device, and has no CPU build to fall back on."
        return _unreachable("CuPy", missing)

    version = cp.cuda.runtime.runtimeGetVersion()
    runtime = f"{version // 1000}.{version % 1000 // 10}"
    linkage = f"CUDA {runtime}, devices visible: {device_count}"
    headline = f"CuPy {cp.__version__} - {linkage}"

    # CuPy is CUDA-only, so there is no CPU side to hold these against: each is
    # checked against a value worked out by hand, as the CPU baselines above are.

    # 0 + 1 + 4 + 9 + 16 + 25 == 55, over kernels CuPy compiles on first use.
    squares = cp.arange(6, dtype=cp.float32) ** 2
    sum_ok = float(squares.sum()) == 55.0
    checks = [Check("CUDA elementwise sum matches expected", passed=sum_ok)]

    # The same matmul as above, through cuBLAS, read back so the copy out goes too.
    mat = cp.arange(6, dtype=cp.float32).reshape(2, 3)
    expected = np.array([[5.0, 14.0], [14.0, 50.0]], dtype=np.float32)
    matmul_ok = bool(np.allclose(cp.asnumpy(mat @ mat.T), expected))
    checks.append(Check("CUDA matmul matches expected", passed=matmul_ok))

    checks.extend(_cupy_torch_interop())

    return Section("CuPy", headline, tuple(checks))


def main() -> None:
    print(_RULE)
    print("CUDA compute environment check")
    print(_RULE)

    print_gpu_hardware()

    sections = []
    for check in (check_pytorch, check_torchvision, check_opencv, check_cupy):
        section = check()
        _render(section)
        sections.append(section)

    print(f"\n{_RULE}")
    for section in sections:
        print(f"  {section.name:<12} {'OK' if section.passed else 'FAIL'}")
    print(_RULE)

    raise SystemExit(0 if all(section.passed for section in sections) else 1)


if __name__ == "__main__":
    main()
