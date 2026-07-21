from __future__ import annotations

__all__ = ("Farneback", "FarnebackParams")

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

import cv2
from kaparoo.utils.optional import unwrap_or_factory

from iivs_cardio.optical_flow.estimators.opencv.base import OpenCVEstimator

if TYPE_CHECKING:
    import torch

    from iivs_cardio.optical_flow.estimators.opencv.base import OpenCVAlgorithm


@dataclass(frozen=True, slots=True)
class FarnebackParams:
    num_levels: int = 3
    pyr_scale: float = 0.5
    fast_pyramids: bool = False
    win_size: int = 15
    num_iters: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0


class Farneback(OpenCVEstimator):
    def __init__(
        self,
        params: FarnebackParams | None = None,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.params = unwrap_or_factory(params, FarnebackParams)
        super().__init__(device)

    @override
    def _create_algorithm(self) -> OpenCVAlgorithm:
        params = self.params

        if self.is_cuda:
            return cv2.cuda.FarnebackOpticalFlow.create(
                numLevels=params.num_levels,
                pyrScale=params.pyr_scale,
                fastPyramids=params.fast_pyramids,
                winSize=params.win_size,
                numIters=params.num_iters,
                polyN=params.poly_n,
                polySigma=params.poly_sigma,
                flags=params.flags,
            )

        return cv2.FarnebackOpticalFlow.create(
            numLevels=params.num_levels,
            pyrScale=params.pyr_scale,
            fastPyramids=params.fast_pyramids,
            winSize=params.win_size,
            numIters=params.num_iters,
            polyN=params.poly_n,
            polySigma=params.poly_sigma,
            flags=params.flags,
        )
