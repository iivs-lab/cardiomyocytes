from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

from iivs.dhm.data.phase import search_phase_bin_folders
from kaparoo.utils.aggregate import Aggregator, Mean

from iivs_cardio.data.preprocessing.normalization import FrameNormalizer
from iivs_cardio.optical_flow.estimators import OpticalFlowEstimator
from iivs_cardio.optical_flow.evaluation import warp_consistency

if TYPE_CHECKING:
    from kaparoo.data.sequences import DataSequence
    from torch import Tensor


def run_sequence(
    sequence: DataSequence[Tensor],
    estimator: OpticalFlowEstimator,
    normalizer: FrameNormalizer,
) -> Aggregator:
    estimator.reset()
    normalizer.reset()

    aggregator = Aggregator(
        overrides={"ssim": Mean(), "psnr": Mean(), "mse": Mean(), "mae": Mean()}
    )

    for frame1, frame2 in pairwise(sequence):
        frame1, frame2 = normalizer.apply(frame1, frame2)
        flow = estimator.calc(frame1, frame2)
        metrics = warp_consistency(frame1, frame2, flow)
        aggregator.update({k: v.item() for k, v in metrics.items()})

    return aggregator


def main() -> None:
    sequences = search_phase_bin_folders(Path("data/phase"), recursive=True)
    estimator = OpticalFlowEstimator()
    normalizer = FrameNormalizer()
    for sequence in sequences:
        run_sequence(sequence, estimator, normalizer)


if __name__ == "__main__":
    main()
