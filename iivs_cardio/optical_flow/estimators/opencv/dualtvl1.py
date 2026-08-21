from __future__ import annotations

__all__ = ("DualTVL1Config",)

import logging
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Final, override

import cv2

from iivs_cardio.optical_flow.estimators.opencv.base import DenseAlgorithm, OpenCVConfig

if TYPE_CHECKING:
    from iivs_cardio.common.device import Device

_logger = logging.getLogger(__name__)

# The settings cv2 hands to one implementation only. One config carries both
# sets, so on either device four of them go nowhere.
_CPU_ONLY: Final = ("inner_iterations", "outer_iterations", "median_filtering")
_CUDA_ONLY: Final = ("iterations",)


@dataclass(frozen=True, slots=True)
class DualTVL1Config(OpenCVConfig):
    """TV-L1's settings, the last four of which only one device each reads.

    cv2 offers no way to ask an algorithm what it ignored, so a sweep over one of
    those four on the other device would run to the end and report no difference.
    Building one logs a warning naming any that were changed and will not be
    read, which is all that can be done: the defaults carry both devices' values,
    so the untouched ones say nothing about what a caller meant.

    Attributes:
        SUPPORTED_DEVICES: As `EstimatorConfig`: cv2 implements TV-L1 for both.
        tau: The time step of the dual ascent.
        lambda_: The weight the data term carries against smoothness.
        theta: The tightness coupling the two variables.
        nscales: The pyramid levels to build.
        warps: The warpings run at each level.
        epsilon: The stopping threshold.
        scale_step: The scale between one level and the next.
        gamma: The weight on the gradient constancy term.
        inner_iterations: The inner loop's iterations. **CPU only.**
        outer_iterations: The outer loop's iterations. **CPU only.**
        median_filtering: The median filter's size, 1 to disable. **CPU only.**
        iterations: The iterations run at each warping. **CUDA only.**
    """

    tau: float = 0.25
    lambda_: float = 0.05
    theta: float = 0.3
    nscales: int = 3
    warps: int = 3
    epsilon: float = 0.005
    scale_step: float = 0.8
    gamma: float = 0.0
    inner_iterations: int = 20
    outer_iterations: int = 5
    median_filtering: int = 5
    iterations: int = 300

    def _warn_unread(self, device: Device) -> None:
        """Warn about settings changed for the device that is not being built.

        Only the changed ones: every config carries both devices' settings at
        their defaults, so naming those would fire on every build and say
        nothing. A caller who wrote one down meant it to do something.
        """
        defaults = {field.name: field.default for field in fields(self)}
        unread = _CPU_ONLY if device.is_cuda else _CUDA_ONLY
        changed = [name for name in unread if getattr(self, name) != defaults[name]]
        if not changed:
            return

        other = "cpu" if device.is_cuda else "cuda"
        _logger.warning(
            "TV-L1 on %s never reads %s, which is the %s implementation's",
            device,
            ", ".join(changed),
            other,
        )

    @override
    def _algorithm(self, device: Device) -> DenseAlgorithm:
        self._warn_unread(device)

        if device.is_cuda:
            return cv2.cuda.OpticalFlowDual_TVL1.create(
                tau=self.tau,
                lambda_=self.lambda_,
                theta=self.theta,
                nscales=self.nscales,
                warps=self.warps,
                epsilon=self.epsilon,
                iterations=self.iterations,
                scaleStep=self.scale_step,
                gamma=self.gamma,
                useInitialFlow=False,
            )

        return cv2.optflow.DualTVL1OpticalFlow.create(
            tau=self.tau,
            lambda_=self.lambda_,
            theta=self.theta,
            nscales=self.nscales,
            warps=self.warps,
            epsilon=self.epsilon,
            innnerIterations=self.inner_iterations,  # OpenCV's own name (triple n)
            outerIterations=self.outer_iterations,
            scaleStep=self.scale_step,
            gamma=self.gamma,
            medianFiltering=self.median_filtering,
            useInitialFlow=False,
        )
