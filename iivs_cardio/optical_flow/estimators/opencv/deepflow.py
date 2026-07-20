from __future__ import annotations

__all__ = ("DeepFlow",)

from typing import override

import cv2

from iivs_cardio.optical_flow.estimators.opencv.base import OpenCVEstimator


class DeepFlow(OpenCVEstimator):
    SUPPORTED_DEVICES = frozenset({"cpu"})  # CPU only; OpenCV ships no CUDA DeepFlow

    @override
    def _create_algorithm(self) -> cv2.DenseOpticalFlow:
        return cv2.optflow.createOptFlow_DeepFlow()  # no tunable parameters
