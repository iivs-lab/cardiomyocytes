from __future__ import annotations

__all__ = ("DualTVL1", "DualTVL1Config")

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
class DualTVL1Config(EstimatorConfig):
    tau: float = 0.25
    lambda_: float = 0.05
    theta: float = 0.3
    nscales: int = 3
    warps: int = 3
    epsilon: float = 0.005
    scale_step: float = 0.8
    gamma: float = 0.0
    # CPU-only (ignored on CUDA):
    inner_iterations: int = 20
    outer_iterations: int = 5
    median_filtering: int = 5
    # CUDA-only (ignored on CPU):
    iterations: int = 300

    @override
    def build(self, device: DeviceLike = "cpu") -> DualTVL1:
        return DualTVL1(self, device=device)


class DualTVL1(OpenCVEstimator):
    def __init__(
        self,
        config: DualTVL1Config | None = None,
        *,
        device: DeviceLike = "cpu",
    ) -> None:
        self.config = unwrap_or_factory(config, DualTVL1Config)
        super().__init__(device)

    @override
    def _create_algorithm(self) -> OpenCVAlgorithm:
        config = self.config

        if self.is_cuda:
            return cv2.cuda.OpticalFlowDual_TVL1.create(
                tau=config.tau,
                lambda_=config.lambda_,
                theta=config.theta,
                nscales=config.nscales,
                warps=config.warps,
                epsilon=config.epsilon,
                iterations=config.iterations,
                scaleStep=config.scale_step,
                gamma=config.gamma,
                useInitialFlow=False,
            )

        return cv2.optflow.DualTVL1OpticalFlow.create(
            tau=config.tau,
            lambda_=config.lambda_,
            theta=config.theta,
            nscales=config.nscales,
            warps=config.warps,
            epsilon=config.epsilon,
            innnerIterations=config.inner_iterations,  # OpenCV's parameter name (triple n)
            outerIterations=config.outer_iterations,
            scaleStep=config.scale_step,
            gamma=config.gamma,
            medianFiltering=config.median_filtering,
            useInitialFlow=False,
        )
