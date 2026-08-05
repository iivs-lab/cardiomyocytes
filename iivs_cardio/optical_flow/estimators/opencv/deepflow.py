from __future__ import annotations

__all__ = ("DeepFlow", "DeepFlowConfig")

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

import cv2

from iivs_cardio.optical_flow.estimators.base import EstimatorConfig
from iivs_cardio.optical_flow.estimators.opencv.base import OpenCVEstimator

if TYPE_CHECKING:
    from iivs_cardio.common.device import DeviceLike


class DeepFlow(OpenCVEstimator):
    SUPPORTED_DEVICES = frozenset({"cpu"})  # CPU only; OpenCV ships no CUDA DeepFlow

    @override
    def _create_algorithm(self) -> cv2.DenseOpticalFlow:
        return cv2.optflow.createOptFlow_DeepFlow()  # no tunable parameters


@dataclass(frozen=True, slots=True)
class DeepFlowConfig(EstimatorConfig):
    """DeepFlow's (empty) recipe: it exposes no tunable parameters.

    Held anyway so every estimator has a buildable `EstimatorConfig`, letting a
    worker construct any of them through the one `build` interface.
    """

    @override
    def build(self, device: DeviceLike = "cpu") -> DeepFlow:
        return DeepFlow(device=device)
