"""Benchmark CPU vs CUDA optical flow through the project's own estimators.

Runs Farneback and Dual TV-L1 on both CPU and CUDA, plus DeepFlow on CPU only
(OpenCV ships no CUDA DeepFlow). The goal is to confirm each CUDA backend is
*faster* than its CPU counterpart while reconstructing the other frame about as
*accurately* (SSIM within tolerance).

Everything goes through `iivs_cardio.optical_flow` rather than raw `cv2`, so the
numbers describe the code the project actually runs:

- `estimators` take `(H, W)` uint8 frames and return `(2, H, W)` float32 flow.
  Both timed paths are measured: `calc` (stateless one-shot; on CUDA that is
  upload + kernel + download) and `push` (streaming, which uploads each frame
  once and reuses its device buffers).
- `evaluation.warp_consistency` scores the flow, so the benchmark and the
  library agree on the warp direction and the metrics by construction.

Fairness: one `FarnebackParams` / `DualTVL1Params` object feeds both devices, so
the shared parameters are identical by construction -- the earlier version had to
keep two parameter lists in sync and let TV-L1 fall back to each backend's own
defaults. TV-L1's iteration counts still differ because the CPU API exposes
inner/outer iterations where CUDA exposes a single count. All algorithms now see
the same uint8 frames (the earlier version fed TV-L1 float32 `[0, 255]`).

Provisional: the synthetic scene is a placeholder (see TODO.md). Treat the
numbers -- Dual TV-L1's especially -- as preliminary until the benchmark is
reworked on real cardiomyocyte data with the parameters used in prior work.

Run inside the project so it uses the project's pinned dependencies:

    uv run scripts/optical_flow/benchmark_opencv.py --size 900 --repeats 3

Exit code is 0 only if every CUDA backend is both faster and within the SSIM
tolerance of its CPU counterpart; DeepFlow is reported as a CPU-only reference.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import TYPE_CHECKING, NamedTuple

import cv2
import numpy as np
import torch

from iivs_cardio.optical_flow.estimators import (
    DeepFlow,
    DualTVL1,
    DualTVL1Params,
    Farneback,
    FarnebackParams,
    OpticalFlowEstimator,
)
from iivs_cardio.optical_flow.evaluation import warp_consistency

if TYPE_CHECKING:
    from collections.abc import Callable

SSIM_TOLERANCE = 0.02  # A CUDA backend's SSIM may sit at most this far below CPU.

# (label, factory, devices) -- one params object per algorithm feeds every device.
ALGORITHMS: tuple[
    tuple[str, Callable[[str], OpticalFlowEstimator], tuple[str, ...]], ...
] = (
    (
        "Farneback",
        lambda device: Farneback(FarnebackParams(), device=device),
        ("cpu", "cuda"),
    ),
    (
        "Dual TV-L1",
        lambda device: DualTVL1(DualTVL1Params(), device=device),
        ("cpu", "cuda"),
    ),
    ("DeepFlow", lambda device: DeepFlow(device=device), ("cpu",)),
)


class Result(NamedTuple):
    algorithm: str
    device: str
    calc_seconds: float  # stateless one-shot; on CUDA includes host<->device copies
    push_seconds: float  # streaming, per pair; one upload per frame, buffers reused
    mae: float
    psnr: float
    ssim: float


def build_scene(size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build two uint8 frames related by a known motion, as `(H, W)` tensors.

    The background is a textured image translated by a fixed vector; a bright
    disk moves by that translation plus its own offset. Because frame2 is a
    genuine motion of frame1 (not independent noise), a correct flow can
    reconstruct one from the other and the metrics measure real accuracy.
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

    def to_u8(frame: np.ndarray) -> torch.Tensor:
        return torch.as_tensor((np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8))

    return to_u8(frame1), to_u8(frame2)


def median_seconds(compute: Callable[[], object], repeats: int, device: str) -> float:
    """Median wall-clock of `repeats` calls; assumes `compute` is pre-warmed.

    CUDA work is queued asynchronously, so the device is synchronised inside the
    timed region -- otherwise the measurement would stop before the GPU does.
    """
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        compute()
        if device == "cuda":
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def measure(
    algorithm: str,
    device: str,
    factory: Callable[[str], OpticalFlowEstimator],
    frame1: torch.Tensor,
    frame2: torch.Tensor,
    repeats: int,
) -> Result:
    """Time an estimator's one-shot and streaming paths, and score its flow."""
    estimator = factory(device)
    first, second = frame1.to(device), frame2.to(device)

    flow = estimator.calc(first, second)  # warm up, and keep a flow to score
    calc_seconds = median_seconds(
        lambda: estimator.calc(first, second), repeats, device
    )

    # Streaming: alternate the two frames so every push computes a real motion,
    # then charge each timed block with the two flows it produced.
    def push_pair() -> None:
        estimator.push(second)
        estimator.push(first)

    estimator.reset()
    estimator.push(first)  # prime the retained frame
    push_pair()  # warm up the streaming path
    push_seconds = median_seconds(push_pair, repeats, device) / 2

    # Score on the host so CPU and CUDA rows are compared by an identical path.
    metrics = warp_consistency(frame1, frame2, flow.cpu())
    return Result(
        algorithm,
        device,
        calc_seconds,
        push_seconds,
        float(metrics["mae"]),
        float(metrics["psnr"]),
        float(metrics["ssim"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=900, help="frame edge length in px")
    parser.add_argument("--repeats", type=int, default=3, help="timed runs per backend")
    parser.add_argument("--seed", type=int, default=42, help="scene RNG seed")
    args = parser.parse_args()

    if cv2.cuda.getCudaEnabledDeviceCount() < 1:
        print("[!] No CUDA device -- this benchmark compares CPU against CUDA.")
        return 1

    frame1, frame2 = build_scene(args.size, args.seed)

    print(
        f"Scene {args.size}x{args.size}, seed {args.seed}, "
        f"{args.repeats} timed run(s) per backend (median reported)."
    )
    print(
        "calc = one-shot (CUDA: upload + kernel + download); push = streaming, per pair."
    )
    print("Warming up and benchmarking (TV-L1 and DeepFlow are slow) ...\n")

    results = [
        measure(name, device, factory, frame1, frame2, args.repeats)
        for name, factory, devices in ALGORITHMS
        for device in devices
    ]

    header = (
        f"{'Algorithm':<11} {'Device':<7} {'calc (s)':>9} {'push (s)':>9} "
        f"{'MAE':>8} {'PSNR (dB)':>10} {'SSIM':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.algorithm:<11} {r.device:<7} {r.calc_seconds:>9.4f} "
            f"{r.push_seconds:>9.4f} {r.mae:>8.4f} {r.psnr:>10.2f} {r.ssim:>8.5f}"
        )
    print(
        "(On CPU there is no transfer, so calc and push differ only by buffer reuse.)"
    )

    by_key = {(r.algorithm, r.device): r for r in results}
    all_ok = True
    print("\nVerdict -- CUDA must be faster and keep SSIM within tolerance:")
    for algorithm, _, devices in ALGORITHMS:
        if "cuda" not in devices:
            reference = by_key[algorithm, "cpu"]
            print(
                f"  {algorithm:<11} REF : CPU-only reference, "
                f"{reference.calc_seconds:.4f}s, SSIM {reference.ssim:.4f}"
            )
            continue

        cpu = by_key[algorithm, "cpu"]
        gpu = by_key[algorithm, "cuda"]
        calc_speedup = (
            cpu.calc_seconds / gpu.calc_seconds
            if gpu.calc_seconds > 0
            else float("inf")
        )
        push_speedup = (
            cpu.push_seconds / gpu.push_seconds
            if gpu.push_seconds > 0
            else float("inf")
        )
        passed = gpu.calc_seconds < cpu.calc_seconds and (
            gpu.ssim >= cpu.ssim - SSIM_TOLERANCE
        )
        all_ok = all_ok and passed
        print(
            f"  {algorithm:<11} {'PASS' if passed else 'FAIL'}: CUDA "
            f"{calc_speedup:>4.1f}x calc / {push_speedup:>4.1f}x push, "
            f"SSIM {gpu.ssim:.4f} vs CPU {cpu.ssim:.4f} "
            f"(delta {gpu.ssim - cpu.ssim:+.4f})"
        )

    return 0 if all_ok else 1


raise SystemExit(main())
