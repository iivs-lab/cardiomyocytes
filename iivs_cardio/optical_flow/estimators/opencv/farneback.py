from __future__ import annotations

__all__ = ("FarnebackConfig",)

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

import cv2

from iivs_cardio.optical_flow.estimators.opencv.base import DenseAlgorithm, OpenCVConfig

if TYPE_CHECKING:
    from iivs_cardio.common.device import Device


@dataclass(frozen=True, slots=True)
class FarnebackConfig(OpenCVConfig):
    """Farneback's settings, every one of which either device reads.

    Attributes:
        SUPPORTED_DEVICES: As `EstimatorConfig`: cv2 implements Farneback for
            both.
        num_levels: The pyramid levels to build.
        pyr_scale: The scale between one level and the next.
        fast_pyramids: Whether to build the pyramid the cheaper way.
        win_size: The averaging window, in pixels.
        num_iters: The iterations run at each level.
        poly_n: The neighbourhood the polynomial is fitted over.
        poly_sigma: The gaussian weighting that fit uses.
        flags: cv2's own flag word, passed through.
    """

    num_levels: int = 3
    pyr_scale: float = 0.5
    fast_pyramids: bool = False
    win_size: int = 15
    num_iters: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0

    @override
    def _algorithm(self, device: Device) -> DenseAlgorithm:
        factory = cv2.FarnebackOpticalFlow
        if device.is_cuda:
            factory = cv2.cuda.FarnebackOpticalFlow

        return factory.create(
            numLevels=self.num_levels,
            pyrScale=self.pyr_scale,
            fastPyramids=self.fast_pyramids,
            winSize=self.win_size,
            numIters=self.num_iters,
            polyN=self.poly_n,
            polySigma=self.poly_sigma,
            flags=self.flags,
        )
