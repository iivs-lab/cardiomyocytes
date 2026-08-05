from __future__ import annotations

__all__ = ("Farneback", "FarnebackConfig")

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

import cv2
from kaparoo.utils.optional import unwrap_or_factory

from iivs_cardio.optical_flow.estimators.base import EstimatorConfig
from iivs_cardio.optical_flow.estimators.opencv.base import OpenCVEstimator

if TYPE_CHECKING:
    from iivs_cardio.common.device import DeviceLike
    from iivs_cardio.optical_flow.estimators.opencv.base import OpenCVAlgorithm


@dataclass(frozen=True, slots=True)
class FarnebackConfig(EstimatorConfig):
    num_levels: int = 3
    pyr_scale: float = 0.5
    fast_pyramids: bool = False
    win_size: int = 15
    num_iters: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0

    @override
    def build(self, device: DeviceLike = "cpu") -> Farneback:
        return Farneback(self, device=device)


class Farneback(OpenCVEstimator):
    def __init__(
        self,
        config: FarnebackConfig | None = None,
        *,
        device: DeviceLike = "cpu",
    ) -> None:
        self.config = unwrap_or_factory(config, FarnebackConfig)
        super().__init__(device)

    @override
    def _create_algorithm(self) -> OpenCVAlgorithm:
        config = self.config

        if self.is_cuda:
            return cv2.cuda.FarnebackOpticalFlow.create(
                numLevels=config.num_levels,
                pyrScale=config.pyr_scale,
                fastPyramids=config.fast_pyramids,
                winSize=config.win_size,
                numIters=config.num_iters,
                polyN=config.poly_n,
                polySigma=config.poly_sigma,
                flags=config.flags,
            )

        return cv2.FarnebackOpticalFlow.create(
            numLevels=config.num_levels,
            pyrScale=config.pyr_scale,
            fastPyramids=config.fast_pyramids,
            winSize=config.win_size,
            numIters=config.num_iters,
            polyN=config.poly_n,
            polySigma=config.poly_sigma,
            flags=config.flags,
        )
