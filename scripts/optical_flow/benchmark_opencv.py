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
uint8, then flow -- on `iivs_cardio.data.preprocessing`. `preprocess=raw` and
`preprocess.normalize` switch the stages off or between modes, so their effect on
flow quality is measurable rather than assumed.

Note that `FrameNormalizer` rounds where the legacy truncated. Half the pixels
shift by a single level of 256, which removes a systematic downward bias rather
than adding noise -- but quality columns from before that change are not
comparable at the fifth decimal.

That matters for the parameter defaults. Measured on *raw* frames, no parameter
change improved both quality axes at once, which argued for keeping the
`cardio-force-legacy` values. But raw frames were never what those values were
tuned for: the median filter exists to remove exactly the noise the tuning was
overfitting to, and pairwise normalization gives each pair the full 256 levels.
Re-run the sweep through this pipeline before trusting that conclusion.

`hydra` supplies the configuration, so every knob is a config override and a
sweep is the same command with `--multirun`:

    uv run scripts/optical_flow/benchmark_opencv.py sample=10hz_tif
    uv run scripts/optical_flow/benchmark_opencv.py --multirun \\
        preprocess=legacy,raw \\
        algorithms.farneback.estimator.params.win_size=9,15,21

Each run writes `benchmark.json` -- every row plus the configuration that
produced it -- next to its log under `outputs/`, so a sweep can be collected
afterwards rather than read off the terminal.

The two modes end differently. A single run is a gate: exit code 0 only if every
CUDA backend is both faster and within the SSIM tolerance of its CPU counterpart.
A sweep is collecting combinations, so a failing one is a finding rather than a
broken run and the exit code reports only whether the runs executed. DeepFlow has
no CUDA backend and is reported as a CPU-only reference either way.

The fixtures are the private `iivs-lab/iivs-lib-fixtures` release; extract the
samples under `fixtures/` (or point `fixtures=` at a checkout that has them,
such as `iivs-lib/tests/fixtures`):

    gh release download v1 -R iivs-lab/iivs-lib-fixtures -D fixtures

**WIP.** Quality conclusions from this script are provisional: the fixtures are
20-frame excerpts, which is 2 s at 10 Hz and 1 s at 20 Hz -- around one beat.
Settle algorithm and parameter choices on a full dataset.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import cv2
import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from hydra.utils import instantiate
from iivs.dhm.data.phase import PhaseBinFolder
from omegaconf import OmegaConf

from iivs_cardio.common.warp import backward_warp
from iivs_cardio.data.preprocessing.filtering import FilteredSequence, MedianKernel
from iivs_cardio.data.preprocessing.normalization import FrameNormalizer
from iivs_cardio.optical_flow.evaluation import warp_consistency

if TYPE_CHECKING:
    from collections.abc import Callable

    from omegaconf import DictConfig

    from iivs_cardio.data.preprocessing.filtering import Kernel
    from iivs_cardio.data.preprocessing.normalization import NormalizationMode
    from iivs_cardio.optical_flow.estimators import OpticalFlowEstimator

LOGGER = logging.getLogger(__name__)


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


def render(results: list[Result], baseline: float) -> list[str]:
    """The result table, as lines, so it can be both printed and logged."""
    header = (
        f"{'Algorithm':<11} {'Device':<7} {'calc (s)':>9} {'push (s)':>9} "
        f"{'SSIM':>8} {'gain':>9} {'FB-err':>8} {'|flow|':>7}"
    )
    lines = [
        f"Identity baseline (zero flow) SSIM {baseline:.5f} -- read 'gain', not 'SSIM'.",
        header,
        "-" * len(header),
    ]
    lines += [
        f"{r.algorithm:<11} {r.device:<7} {r.calc_seconds:>9.4f} "
        f"{r.push_seconds:>9.4f} {r.ssim:>8.5f} {r.ssim_gain:>+9.5f} "
        f"{r.fb_error:>8.4f} {r.magnitude:>7.3f}"
        for r in results
    ]
    lines.append(
        "(gain = SSIM above the identity baseline; FB-err in px, lower is better.)"
    )
    lines.append("(A flow is good only if it earns gain AND stays self-consistent.)")

    return lines


def verdicts(results: list[Result], tolerance: float) -> tuple[list[str], bool]:
    """Per-algorithm CPU-vs-CUDA lines, and whether every comparable pair passed.

    An algorithm with no CUDA backend is reported and skipped rather than
    counted, so a CPU-only reference cannot fail the run.
    """
    by_name: dict[str, dict[str, Result]] = {}
    for result in results:
        by_name.setdefault(result.algorithm, {})[result.device] = result

    lines, passed = [], True
    for name, devices in by_name.items():
        cpu, gpu = devices.get("cpu"), devices.get("cuda")
        if cpu is None or gpu is None:
            reference = next(iter(devices.values()))  # non-empty by construction
            lines.append(
                f"  {name:<11} REF : {reference.device}-only reference, "
                f"{reference.calc_seconds:.4f}s, gain {reference.ssim_gain:+.5f}"
            )
            continue

        ok = gpu.calc_seconds < cpu.calc_seconds and gpu.ssim >= cpu.ssim - tolerance
        passed = passed and ok
        lines.append(
            f"  {name:<11} {'PASS' if ok else 'FAIL'}: CUDA "
            f"{cpu.calc_seconds / gpu.calc_seconds:>4.1f}x calc / "
            f"{cpu.push_seconds / gpu.push_seconds:>4.1f}x push, "
            f"SSIM {gpu.ssim:.4f} vs CPU {cpu.ssim:.4f} "
            f"(delta {gpu.ssim - cpu.ssim:+.4f})"
        )

    return lines, passed


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Score every configured backend, then record the run.

    Raises:
        SystemExit: In a single run whose verdict failed. A sweep leaves the exit
            code alone, since one bad combination is a finding rather than a
            broken run.
    """
    if cv2.cuda.getCudaEnabledDeviceCount() < 1:
        LOGGER.error("No CUDA device -- this benchmark compares CPU against CUDA.")
        raise SystemExit(1)

    sample = Path(cfg.fixtures) / cfg.sample
    if not (sample / "Phase" / "Float" / "Bin").is_dir():
        LOGGER.error("No phase sequence at %s.", sample)
        LOGGER.error(
            "  gh release download v1 -R iivs-lab/iivs-lib-fixtures -D fixtures"
        )
        raise SystemExit(1)

    kernel: Kernel | None = instantiate(cfg.preprocess.filter)
    where = "cuda" if torch.cuda.is_available() else "cpu"
    frames = load_frames(sample, kernel, where)

    pairs = normalize_pairs(frames, cfg.preprocess.normalize)
    if cfg.pairs:
        pairs = pairs[: cfg.pairs]
    baseline = identity_ssim(pairs)

    filter_note = (
        f"3D median r={kernel.radius} ({len(kernel.offsets)} samples)"
        if isinstance(kernel, MedianKernel)
        else "no filter"
    )
    LOGGER.info(
        "Sample %s: %d frames of %dx%d, %d pair(s) scored, %d timed run(s) per backend.",
        cfg.sample,
        len(frames),
        *frames[0].shape,
        len(pairs),
        cfg.repeats,
    )
    LOGGER.info(
        "Preprocessing: %s, %s normalization.", filter_note, cfg.preprocess.normalize
    )
    LOGGER.info("Warming up and benchmarking (TV-L1 and DeepFlow are slow) ...")

    results = [
        measure(
            name, device, instantiate(entry.estimator), pairs, baseline, cfg.repeats
        )
        for name, entry in cfg.algorithms.items()
        for device in entry.devices
    ]

    for line in render(results, baseline):
        LOGGER.info("%s", line)

    lines, passed = verdicts(results, cfg.ssim_tolerance)
    LOGGER.info("Verdict -- CUDA must be faster and keep SSIM within tolerance:")
    for line in lines:
        LOGGER.info("%s", line)

    output = Path(HydraConfig.get().runtime.output_dir) / "benchmark.json"
    output.write_text(
        json.dumps(
            {
                "config": OmegaConf.to_container(cfg, resolve=True),
                "baseline_ssim": baseline,
                "passed": passed,
                "results": [r._asdict() for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    LOGGER.info("Wrote %s", output)

    # A sweep is collecting combinations, so a failing one is data, not an error.
    if not passed and HydraConfig.get().mode is RunMode.RUN:
        raise SystemExit(1)


main()
