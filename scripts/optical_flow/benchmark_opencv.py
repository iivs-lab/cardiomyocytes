"""Benchmark CPU vs CUDA optical flow on the real DHM fixtures.

Runs Farneback and Dual TV-L1 on both CPU and CUDA, plus DeepFlow on CPU only
(OpenCV ships no CUDA DeepFlow), over consecutive frame pairs of a real Koala
cardiomyocyte time-lapse. The goal is to confirm each CUDA backend is *faster*
than its CPU counterpart while producing about as *accurate* a flow.

Everything goes through `iivs_cardio.optical_flow` rather than raw `cv2`, so the
numbers describe the code the project actually runs. Both timed paths are
measured: `calc` (stateless one-shot; on CUDA that is upload + kernel + download)
and `push` (streaming, which uploads each frame once and reuses device buffers).

Reading the quality columns -- this is the part that is easy to get wrong:

- Real inter-frame motion here is **sub-pixel** (median ~0.3-0.4 px), so the two
  frames are already nearly identical and a flow of exactly zero scores SSIM
  ~0.95. Raw SSIM therefore says almost nothing. Every row is reported as a
  **gain over the identity baseline**, which is the row for a zero flow.
- SSIM alone is gameable: a flow with more freedom reconstructs the frame better by
  fitting *noise*. Measured on this data, parameters that double the SSIM gain
  degrade forward-backward consistency 8-20x. So each row also carries **FB-err**
  -- `|f_fwd(x) + f_bwd(x + f_fwd(x))|`, which a noise-fitted flow fails because
  the noise it latched onto is not a real correspondence.
- FB-err alone is gameable too, in the opposite direction: a zero flow is
  perfectly self-consistent. **Read the two together** -- a good flow earns SSIM
  gain *and* stays self-consistent.

Preprocessing follows the legacy pipeline -- read, 3D median filter, normalize to
uint8, then flow -- and now runs on `iivs_cardio.data.preprocessing`, which the
prototype that used to live here was written to specify. `--filter-radius none`
and `--normalize` switch the stages off or between modes, so their effect on flow
quality is measurable rather than assumed.

One number moved in the handover: the prototype truncated when converting to
uint8, as the legacy does, while `FrameNormalizer` rounds. Half the pixels shift
by a single level, which removes a systematic downward bias rather than adding
noise. Quality columns are not comparable across that change at the fifth
decimal.

That matters for the parameter defaults. Measured on *raw* frames, no parameter
change improved both quality axes at once, which argued for keeping the
`cardio-force-legacy` values. But raw frames were never what those values were
tuned for: the median filter exists to remove exactly the noise the tuning was
overfitting to, and pairwise normalization gives each pair the full 256 levels.
Re-run the sweep through this pipeline before trusting that conclusion.

    uv run scripts/optical_flow/benchmark_opencv.py --sample 10hz_tif

The fixtures are the private `iivs-lab/iivs-lib-fixtures` release; extract the
samples under `fixtures/` (or point `--fixtures` at a checkout that has them,
such as `iivs-lib/tests/fixtures`):

    gh release download v1 -R iivs-lab/iivs-lib-fixtures -D fixtures

Exit code is 0 only if every CUDA backend is both faster and within the SSIM
tolerance of its CPU counterpart; DeepFlow is reported as a CPU-only reference.

**WIP.** Quality conclusions from this script are provisional: the fixtures are
20-frame excerpts, which is 2 s at 10 Hz and 1 s at 20 Hz -- around one beat.
Settle algorithm and parameter choices on a full dataset.
"""

from __future__ import annotations

import argparse
import statistics
import time
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import cv2
import numpy as np
import torch
from iivs.dhm.data.phase import PhaseBinFolder

from iivs_cardio.common.warp import backward_warp
from iivs_cardio.data.preprocessing.filtering import FilteredSequence, MedianKernel
from iivs_cardio.data.preprocessing.normalization import FrameNormalizer
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

    from iivs_cardio.data.preprocessing.filtering import Kernel
    from iivs_cardio.data.preprocessing.normalization import NormalizationMode

SSIM_TOLERANCE = 0.02  # A CUDA backend's SSIM may sit at most this far below CPU.

# (label, factory, devices) -- one params object per algorithm feeds every device,
# so the shared parameters are identical by construction. TV-L1's iteration counts
# still differ because the CPU API exposes inner/outer where CUDA exposes one count.
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
    ssim: float
    ssim_gain: float  # over the identity baseline -- the meaningful column
    fb_error: float  # px; forward-backward inconsistency, lower is better
    magnitude: float  # mean |flow| in px


# --------------------------------------------------------------------------- #
#                                Preprocessing                                  #
# --------------------------------------------------------------------------- #
# The legacy pipeline -- read -> 3D median filter -> normalize to uint8 -> flow
# -- now assembled from `iivs_cardio.data.preprocessing`, which this script's
# prototype was written to specify. Everything below is wiring: the semantics
# live in the modules.


def load_frames(sample: Path, kernel: Kernel | None, device: str) -> list[torch.Tensor]:
    """A time-lapse as float32 frames, filtered when a `kernel` is given.

    `PhaseBinFolder` is already the sequence `FilteredSequence` reads, so the
    filter wraps it rather than a volume held in memory.
    """
    source = PhaseBinFolder(sample / "Phase" / "Float" / "Bin")
    if kernel is None:
        return [torch.as_tensor(np.asarray(frame)) for frame in source]

    return [frame.cpu() for frame in FilteredSequence(source, kernel, device=device)]


def normalize_pairs(
    frames: list[torch.Tensor], mode: NormalizationMode
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Consecutive frame pairs as uint8, each scaled by `mode`.

    Pairs rather than frames because `pairwise` scales a frame by the joint
    range of whichever pair it is in, so it appears twice with two encodings and
    no single normalized frame list exists. `injected` is the sequence-wide
    scope: one range measured over every frame, which is the pass this does up
    front.
    """
    normalizer = FrameNormalizer(mode, dtype=torch.uint8)
    if mode == "injected":
        low = min(float(frame.min()) for frame in frames)
        high = max(float(frame.max()) for frame in frames)
        normalizer.source_range = (low, high)

    return [normalizer.apply(prev, curr) for prev, curr in pairwise(frames)]


def forward_backward_error(forward: torch.Tensor, backward: torch.Tensor) -> float:
    """Mean `|f_fwd(x) + f_bwd(x + f_fwd(x))|` in px; 0 for a consistent flow.

    `backward_warp` samples at `grid + flow`, so warping the backward flow by the
    forward one evaluates it exactly where the forward flow claims the pixel went.
    The flow is broadcast over the backward field's own two channels.
    """
    sampled = backward_warp(backward, forward.unsqueeze(0).expand(2, -1, -1, -1))
    residual = forward + sampled
    return float(torch.sqrt(residual[0] ** 2 + residual[1] ** 2).mean())


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


def identity_ssim(pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    """Mean warp-consistency SSIM of a zero flow -- the floor every row sits on."""
    zero = torch.zeros(2, *pairs[0][0].shape)
    return float(
        np.mean([float(warp_consistency(a, b, zero)["ssim"]) for a, b in pairs])
    )


def measure(
    algorithm: str,
    device: str,
    factory: Callable[[str], OpticalFlowEstimator],
    pairs: list[tuple[torch.Tensor, torch.Tensor]],
    baseline: float,
    repeats: int,
) -> Result:
    """Time an estimator's one-shot and streaming paths, and score its flow."""
    estimator = factory(device)
    first, second = pairs[0][0].to(device), pairs[0][1].to(device)

    estimator.calc(first, second)  # warm up
    calc_seconds = median_seconds(
        lambda: estimator.calc(first, second), repeats, device
    )

    def push_pair() -> None:
        estimator.push(second)
        estimator.push(first)

    estimator.reset()
    estimator.push(first)  # prime the retained frame
    push_pair()  # warm up the streaming path
    push_seconds = median_seconds(push_pair, repeats, device) / 2

    # Quality over every pair, scored on the host so CPU and CUDA rows are
    # compared by an identical path.
    ssims, errors, magnitudes = [], [], []
    for prev, curr in pairs:
        p, c = prev.to(device), curr.to(device)
        forward = estimator.calc(p, c).cpu()
        backward = estimator.calc(c, p).cpu()
        ssims.append(float(warp_consistency(prev, curr, forward)["ssim"]))
        errors.append(forward_backward_error(forward, backward))
        magnitudes.append(float(torch.sqrt(forward[0] ** 2 + forward[1] ** 2).mean()))

    ssim = float(np.mean(ssims))
    return Result(
        algorithm,
        device,
        calc_seconds,
        push_seconds,
        ssim,
        ssim - baseline,
        float(np.mean(errors)),
        float(np.mean(magnitudes)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures", type=Path, default=Path("fixtures"), help="fixture root"
    )
    parser.add_argument("--sample", default="10hz_tif", help="time-lapse folder name")
    parser.add_argument(
        "--pairs", type=int, default=0, help="consecutive pairs to score (0 = all)"
    )
    parser.add_argument("--repeats", type=int, default=3, help="timed runs per backend")
    parser.add_argument(
        "--filter-radius",
        default="2,2,2",
        help="3D median filter radius: r, r_xy,r_z, or x,y,z; 'none' to skip",
    )
    parser.add_argument(
        "--footprint", default="ellipsoid", choices=("ellipsoid", "cuboid")
    )
    parser.add_argument(
        "--normalize",
        default="pairwise",
        choices=("pairwise", "perframe", "injected"),
        help="uint8 normalization mode; 'injected' is sequence-wide (legacy: pairwise)",
    )
    args = parser.parse_args()

    if cv2.cuda.getCudaEnabledDeviceCount() < 1:
        print("[!] No CUDA device -- this benchmark compares CPU against CUDA.")
        return 1

    sample = args.fixtures / args.sample
    if not (sample / "Phase" / "Float" / "Bin").is_dir():
        print(f"[!] No phase sequence at {sample}.")
        print("    gh release download v1 -R iivs-lab/iivs-lib-fixtures -D fixtures")
        return 1

    kernel = None
    filter_note = "no filter"
    if args.filter_radius != "none":
        try:
            values = tuple(int(value) for value in args.filter_radius.split(","))
            kernel = MedianKernel(
                values[0] if len(values) == 1 else values, shape=args.footprint
            )
        except ValueError as error:  # the kernel names the forms it accepts
            print(f"[!] --filter-radius {args.filter_radius!r}: {error}")
            return 1
        filter_note = (
            f"3D median r={kernel.radius} {args.footprint} "
            f"({len(kernel.offsets)} samples)"
        )

    # The filter is preprocessing, not a benchmarked path, so run it wherever it
    # is fastest and hand the estimators an identical result either way.
    where = "cuda" if torch.cuda.is_available() else "cpu"
    frames = load_frames(sample, kernel, where)
    frame_count = len(frames)
    height, width = frames[0].shape

    pairs = normalize_pairs(frames, args.normalize)
    if args.pairs:
        pairs = pairs[: args.pairs]
    baseline = identity_ssim(pairs)

    print(
        f"Sample {args.sample}: {frame_count} frames of {height}x{width}, "
        f"{len(pairs)} pair(s) scored, {args.repeats} timed run(s) per backend."
    )
    print(f"Preprocessing: {filter_note}, {args.normalize} normalization.")
    print(
        "calc = one-shot (CUDA: upload + kernel + download); push = streaming, per pair."
    )
    print(
        f"Identity baseline (zero flow) SSIM {baseline:.5f} -- read 'gain', not 'SSIM'."
    )
    print("Warming up and benchmarking (TV-L1 and DeepFlow are slow) ...\n")

    results = [
        measure(name, device, factory, pairs, baseline, args.repeats)
        for name, factory, devices in ALGORITHMS
        for device in devices
    ]

    header = (
        f"{'Algorithm':<11} {'Device':<7} {'calc (s)':>9} {'push (s)':>9} "
        f"{'SSIM':>8} {'gain':>9} {'FB-err':>8} {'|flow|':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.algorithm:<11} {r.device:<7} {r.calc_seconds:>9.4f} "
            f"{r.push_seconds:>9.4f} {r.ssim:>8.5f} {r.ssim_gain:>+9.5f} "
            f"{r.fb_error:>8.4f} {r.magnitude:>7.3f}"
        )
    print("(gain = SSIM above the identity baseline; FB-err in px, lower is better.)")
    print("(A flow is good only if it earns gain AND stays self-consistent.)")

    by_key = {(r.algorithm, r.device): r for r in results}
    all_ok = True
    print("\nVerdict -- CUDA must be faster and keep SSIM within tolerance:")
    for algorithm, _, devices in ALGORITHMS:
        if "cuda" not in devices:
            reference = by_key[algorithm, "cpu"]
            print(
                f"  {algorithm:<11} REF : CPU-only reference, "
                f"{reference.calc_seconds:.4f}s, gain {reference.ssim_gain:+.5f}"
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
