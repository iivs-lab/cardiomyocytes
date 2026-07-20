from __future__ import annotations

__all__ = ("DualTVL1", "DualTVL1Params")

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

import cv2
from kaparoo.utils.optional import unwrap_or_factory

from iivs_cardio.optical_flow.estimators.opencv.base import OpenCVEstimator

if TYPE_CHECKING:
    import torch

    from iivs_cardio.optical_flow.estimators.opencv.base import DenseOpticalFlow


@dataclass(frozen=True, slots=True)
class DualTVL1Params:
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


class DualTVL1(OpenCVEstimator):
    def __init__(
        self,
        params: DualTVL1Params | None = None,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.params = unwrap_or_factory(params, DualTVL1Params)
        super().__init__(device)

    @override
    def _create_algorithm(self) -> DenseOpticalFlow:
        params = self.params

        if self.is_cuda:
            return cv2.cuda.OpticalFlowDual_TVL1.create(
                tau=params.tau,
                lambda_=params.lambda_,
                theta=params.theta,
                nscales=params.nscales,
                warps=params.warps,
                epsilon=params.epsilon,
                iterations=params.iterations,
                scaleStep=params.scale_step,
                gamma=params.gamma,
                useInitialFlow=False,
            )

        return cv2.optflow.DualTVL1OpticalFlow.create(
            tau=params.tau,
            lambda_=params.lambda_,
            theta=params.theta,
            nscales=params.nscales,
            warps=params.warps,
            epsilon=params.epsilon,
            innnerIterations=params.inner_iterations,  # OpenCV's parameter name (triple n)
            outerIterations=params.outer_iterations,
            scaleStep=params.scale_step,
            gamma=params.gamma,
            medianFiltering=params.median_filtering,
            useInitialFlow=False,
        )
