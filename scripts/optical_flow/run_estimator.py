from __future__ import annotations

import os
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np
import torch
from dotenv import load_dotenv
from iivs.dhm.data.phase import search_phase_bin_folders
from kaparoo.data.sequences import TransformedSequence
from kaparoo.filesystem import ensure_dir_exists

from iivs_cardio.common.device import Device
from iivs_cardio.data.transforms.filtering import FilteredSequence

if TYPE_CHECKING:
    from collections.abc import Sequence

    from kaparoo.data.sequences import DataSequence
    from numpy.typing import NDArray
    from omegaconf import DictConfig

    from iivs_cardio.common.device import DeviceLike
    from iivs_cardio.data.transforms.filtering.kernel import FilterKernel
    from iivs_cardio.data.transforms.normalization import FrameNormalizer
    from iivs_cardio.optical_flow.estimators import (
        EstimatorConfig,
        OpticalFlowEstimator,
    )

load_dotenv()

CONFIG_PATH = os.environ["CONFIGS_ROOT"]
CONFIG_NAME = "optical_flow/estimators/config"


def prepare_sequences(root_dir: Path) -> Sequence[DataSequence[NDArray[np.float32]]]:
    return search_phase_bin_folders(root_dir)


def process_sequence(
    sequence: DataSequence[NDArray[np.float32]],
    normalizer: FrameNormalizer,
    estimator: OpticalFlowEstimator,
) -> None:
    normalizer.reset()
    estimator.reset()

    for i, (frame1, frame2) in enumerate(pairwise(sequence)):
        normalized1, normalized2 = normalizer.apply(frame1, frame2)
        flow_1to2 = estimator.calc(normalized1, normalized2)
        flow_2to1 = estimator.calc(normalized2, normalized1)


def process_sequences(
    samples: Sequence[DataSequence[NDArray[np.float32]]],
    kernel: FilterKernel | None,
    normalizer: FrameNormalizer,
    estimator_params: EstimatorConfig,
    device: DeviceLike = "cpu",
) -> None:
    device = Device.resolve(device)
    estimator = estimator_params.build(device=device)

    normalizer.reset()
    estimator.reset()

    for sequence in samples:
        if kernel is not None:
            sequence = FilteredSequence(sequence, kernel, device=device)
        else:
            sequence = TransformedSequence(sequence, torch.from_numpy)

        process_sequence(sequence, normalizer, estimator)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    source_root = ensure_dir_exists(cfg.source_root)

    sequences = prepare_sequences(source_root)
    if not sequences:
        msg = f"No sequences found in {source_root}"
        raise SystemExit(msg)


if __name__ == "__main__":
    main()
