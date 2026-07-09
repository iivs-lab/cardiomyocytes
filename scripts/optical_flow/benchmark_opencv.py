"""Benchmark CPU vs CUDA optical flow to confirm the GPU builds pay off.

Runs Farneback and Dual TV-L1 on both CPU and CUDA, plus DeepFlow on CPU only
(OpenCV ships no CUDA DeepFlow). The goal is to confirm each CUDA backend is
*faster* than its CPU counterpart while reconstructing the reference frame about
as *accurately* (SSIM within tolerance).

Fairness:

- Farneback uses identical parameters on CPU and CUDA. Dual TV-L1 uses each
  backend's defaults, which already agree on the shared parameters (tau, lambda,
  theta, scales, warps, epsilon), so both do equivalent work. TV-L1 is fed
  float32 in [0, 255] (its default parameters assume an 8-bit intensity scale;
  a [0, 1] range starves the data term and cripples the result).
- Both backends see the same seeded synthetic scene: a textured background under
  a known translation plus a disk moving by its own vector, so an accurate flow
  can reconstruct the source and SSIM reflects real accuracy (not a noise floor).
- Two CUDA times are reported: end-to-end (host<->device transfer + kernel) and
  compute-only (kernel on resident data, stream-synchronised). CPU and GPU are
  both warmed up, and each reported time is the median of several runs. The
  pass/fail verdict uses the honest end-to-end time.

Provisional: the synthetic scene is a placeholder (see TODO.md). Treat the
numbers — Dual TV-L1's especially — as preliminary until the benchmark is
reworked on real cardiomyocyte data with the parameters used in prior work.

Run inside the project so it uses the project's pinned OpenCV / NumPy /
scikit-image. It carries no PEP 723 inline metadata on purpose — the
dependencies come from pyproject.toml + uv.lock and cannot drift from the
versions the project actually uses:

    uv run scripts/optical_flow/benchmark_opencv.py --size 900 --repeats 3

Exit code is 0 only if every CUDA backend is both faster and within the SSIM
tolerance of its CPU counterpart; DeepFlow is reported as a CPU-only reference.
"""

from __future__ import annotations

import argparse
import statistics
import time
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity

if TYPE_CHECKING:
    from collections.abc import Callable

# Farneback parameters, applied identically to the CPU and CUDA backends.
FB_PYR_SCALE = 0.5
FB_LEVELS = 5
FB_WINSIZE = 15
FB_ITERS = 3
FB_POLY_N = 5
FB_POLY_SIGMA = 1.2
FB_FLAGS = 0

SSIM_TOLERANCE = 0.02  # A CUDA backend's SSIM may sit at most this far below CPU.


class Result(NamedTuple):
    algorithm: str
    backend: str
    seconds: float  # end-to-end; includes host<->device transfer for CUDA
    compute_seconds: float  # pure compute (kernel only); == seconds for CPU
    rmse: float
    psnr: float
    ssim: float


def build_scene(size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Build two float32 [0, 1] frames related by a known motion.

    The background is a textured image translated by a fixed vector; a bright
    disk moves by that translation plus its own offset. Because frame2 is a
    genuine motion of frame1 (not independent noise), a correct flow can
    reconstruct frame1 and the quality metrics measure real flow accuracy.
    """
    rng = np.random.default_rng(seed)
    base = rng.random((size, size), dtype=np.float32) * 0.5

    # Static blobs give the flow structure to track beyond raw noise; they
    # translate with the background, so they stay consistent across the frames.
    for _ in range(12):
        center = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        radius = int(rng.integers(20, 70))
        cv2.circle(base, center, radius, float(rng.random()), -1)

    bg_dx, bg_dy = 8.0, 5.0  # background translation
    disk_dx, disk_dy = 6.0, -4.0  # extra disk motion on top of the background
    disk_radius = size // 8
    c1 = (size // 2, size // 2)
    c2 = (int(c1[0] + bg_dx + disk_dx), int(c1[1] + bg_dy + disk_dy))

    frame1 = base.copy()
    cv2.circle(frame1, c1, disk_radius, 1.0, -1)

    translation = np.array([[1.0, 0.0, bg_dx], [0.0, 1.0, bg_dy]], dtype=np.float32)
    frame2 = cv2.warpAffine(
        base, translation, (size, size), borderMode=cv2.BORDER_REFLECT101
    )
    cv2.circle(frame2, c2, disk_radius, 1.0, -1)

    return np.clip(frame1, 0.0, 1.0), np.clip(frame2, 0.0, 1.0)


def warp_back(target: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Reconstruct the source frame by sampling `target` along `flow`.

    For flow computed as calc(source, target), `target[x + flow]` recovers the
    source, so comparing the result to the source measures flow accuracy in the
    correct direction.
    """
    height, width = target.shape
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(
        target, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101
    )


def quality(
    reference: np.ndarray, reconstructed: np.ndarray
) -> tuple[float, float, float]:
    """Return (RMSE, PSNR, SSIM) of `reconstructed` against `reference`."""
    mse = float(np.mean((reference - reconstructed) ** 2))
    rmse = float(np.sqrt(mse))
    psnr = float("inf") if mse == 0 else float(20.0 * np.log10(1.0 / rmse))
    ssim = float(structural_similarity(reference, reconstructed, data_range=1.0))
    return rmse, psnr, ssim


def median_seconds(compute: Callable[[], object], repeats: int) -> float:
    """Median wall-clock of `repeats` calls; assumes `compute` is pre-warmed."""
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        compute()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def measure(
    algorithm: str,
    backend: str,
    compute: Callable[[], np.ndarray],
    reference: np.ndarray,
    target: np.ndarray,
    repeats: int,
    *,
    compute_only: Callable[[], object] | None = None,
) -> Result:
    """Warm up, time, and score one backend, returning its `Result`.

    `compute` is the end-to-end path (for CUDA, upload + calc + download) and
    also yields the flow used for quality. `compute_only`, when given, times just
    the kernel on already-resident device data (synchronised by the caller), so
    the transfer-free GPU cost is reported alongside the end-to-end one.
    """
    flow = compute()  # warm up and capture a flow field for the quality metrics
    seconds = median_seconds(compute, repeats)

    if compute_only is not None:
        compute_only()  # warm up the transfer-free path on resident device data
        compute_seconds = median_seconds(compute_only, repeats)
    else:
        compute_seconds = seconds

    rmse, psnr, ssim = quality(reference, warp_back(target, flow))
    return Result(algorithm, backend, seconds, compute_seconds, rmse, psnr, ssim)


def make_tvl1_cpu() -> object:
    """Create the CPU Dual TV-L1 solver across OpenCV layouts."""
    try:
        return cv2.optflow.createOptFlow_DualTVL1()
    except AttributeError:
        return cv2.createOptFlow_DualTVL1()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=900, help="frame edge length in px")
    parser.add_argument("--repeats", type=int, default=3, help="timed runs per backend")
    parser.add_argument("--seed", type=int, default=42, help="scene RNG seed")
    args = parser.parse_args()

    if cv2.cuda.getCudaEnabledDeviceCount() < 1:
        print(
            "[!] No CUDA device — this benchmark compares CPU against CUDA and needs a GPU."
        )
        return 1

    frame1, frame2 = build_scene(args.size, args.seed)
    # Farneback and DeepFlow take 8-bit input. Dual TV-L1's default parameters
    # assume an 8-bit intensity scale too, so feed it float32 in [0, 255] (not
    # [0, 1], which starves its data term and wrecks the flow).
    u8_1 = (frame1 * 255).astype(np.uint8)
    u8_2 = (frame2 * 255).astype(np.uint8)
    f255_1 = (frame1 * 255.0).astype(np.float32)
    f255_2 = (frame2 * 255.0).astype(np.float32)

    print(
        f"Scene {args.size}x{args.size}, seed {args.seed}, "
        f"{args.repeats} timed run(s) per backend (median reported)."
    )
    print("CUDA times are end-to-end and include host<->device transfer.")
    print("Warming up and benchmarking (TV-L1 and DeepFlow are slow) ...\n")

    # A dedicated stream lets the compute-only closures synchronise on just the
    # kernel (via waitForCompletion) without a device->host copy.
    stream = cv2.cuda_Stream()

    # -- Farneback: matched parameters on both backends --------------------
    def farneback_cpu() -> np.ndarray:
        return cv2.calcOpticalFlowFarneback(
            u8_1,
            u8_2,
            None,
            FB_PYR_SCALE,
            FB_LEVELS,
            FB_WINSIZE,
            FB_ITERS,
            FB_POLY_N,
            FB_POLY_SIGMA,
            FB_FLAGS,
        )

    farn_gpu = cv2.cuda_FarnebackOpticalFlow.create(
        FB_LEVELS,
        FB_PYR_SCALE,
        False,  # noqa: FBT003 - fastPyramids; positional to match the C++ factory
        FB_WINSIZE,
        FB_ITERS,
        FB_POLY_N,
        FB_POLY_SIGMA,
        FB_FLAGS,
    )
    g8_1 = cv2.cuda_GpuMat()
    g8_2 = cv2.cuda_GpuMat()
    d_flow_fb = cv2.cuda_GpuMat()

    def farneback_gpu() -> np.ndarray:
        g8_1.upload(u8_1)
        g8_2.upload(u8_2)
        return farn_gpu.calc(g8_1, g8_2, None).download()

    def farneback_gpu_compute() -> None:
        farn_gpu.calc(g8_1, g8_2, d_flow_fb, stream)
        stream.waitForCompletion()

    # -- Dual TV-L1: fed float32 [0, 255]; backend defaults share parameters
    tvl1_cpu_algo = make_tvl1_cpu()

    def tvl1_cpu() -> np.ndarray:
        return tvl1_cpu_algo.calc(f255_1, f255_2, None)

    tvl1_gpu_algo = cv2.cuda_OpticalFlowDual_TVL1.create()
    g32_1 = cv2.cuda_GpuMat()
    g32_2 = cv2.cuda_GpuMat()
    d_flow_tv = cv2.cuda_GpuMat()

    def tvl1_gpu() -> np.ndarray:
        g32_1.upload(f255_1)
        g32_2.upload(f255_2)
        return tvl1_gpu_algo.calc(g32_1, g32_2, None).download()

    def tvl1_gpu_compute() -> None:
        tvl1_gpu_algo.calc(g32_1, g32_2, d_flow_tv, stream)
        stream.waitForCompletion()

    # -- DeepFlow: CPU only (no CUDA build exists) -------------------------
    try:
        deepflow_algo = cv2.optflow.createOptFlow_DeepFlow()
    except AttributeError:
        deepflow_algo = None

    def deepflow_cpu() -> np.ndarray:
        return deepflow_algo.calc(u8_1, u8_2, None)

    func = partial(measure, reference=frame1, target=frame2, repeats=args.repeats)

    results = [
        func("Farneback", "CPU", farneback_cpu),
        func("Farneback", "CUDA", farneback_gpu, compute_only=farneback_gpu_compute),
        func("Dual TV-L1", "CPU", tvl1_cpu),
        func("Dual TV-L1", "CUDA", tvl1_gpu, compute_only=tvl1_gpu_compute),
    ]
    if deepflow_algo is not None:
        results.append(func("DeepFlow", "CPU", deepflow_cpu))
    else:
        print(
            "[!] cv2.optflow.createOptFlow_DeepFlow unavailable — skipping DeepFlow.\n"
        )

    header = (
        f"{'Algorithm':<11} {'Backend':<7} {'E2E (s)':>9} {'Compute (s)':>12} "
        f"{'RMSE':>9} {'PSNR (dB)':>10} {'SSIM':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.algorithm:<11} {r.backend:<7} {r.seconds:>9.4f} {r.compute_seconds:>12.4f} "
            f"{r.rmse:>9.5f} {r.psnr:>10.2f} {r.ssim:>8.5f}"
        )
    print("(CPU has no host<->device transfer, so its E2E and Compute are identical.)")

    by_key = {(r.algorithm, r.backend): r for r in results}
    all_ok = True
    print("\nVerdict — CUDA must be faster end-to-end and keep SSIM within tolerance:")
    for algorithm in ("Farneback", "Dual TV-L1"):
        cpu = by_key[algorithm, "CPU"]
        gpu = by_key[algorithm, "CUDA"]
        e2e_speedup = cpu.seconds / gpu.seconds if gpu.seconds > 0 else float("inf")
        compute_speedup = (
            cpu.compute_seconds / gpu.compute_seconds
            if gpu.compute_seconds > 0
            else float("inf")
        )
        faster = gpu.seconds < cpu.seconds
        quality_ok = gpu.ssim >= cpu.ssim - SSIM_TOLERANCE
        passed = faster and quality_ok
        all_ok = all_ok and passed
        mark = "PASS" if passed else "FAIL"
        print(
            f"  {algorithm:<11} {mark}: CUDA {e2e_speedup:>4.1f}x end-to-end / "
            f"{compute_speedup:>4.1f}x compute-only, "
            f"SSIM {gpu.ssim:.4f} vs CPU {cpu.ssim:.4f} (delta {gpu.ssim - cpu.ssim:+.4f})"
        )

    deep = by_key.get(("DeepFlow", "CPU"))
    if deep is not None:
        print(
            f"  {'DeepFlow':<11} REF : CPU-only reference, {deep.seconds:.4f}s, SSIM {deep.ssim:.4f}"
        )

    return 0 if all_ok else 1


raise SystemExit(main())
