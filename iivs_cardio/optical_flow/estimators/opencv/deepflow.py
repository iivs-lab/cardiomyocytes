from __future__ import annotations

__all__ = ("DeepFlowConfig",)

from dataclasses import dataclass
from typing import ClassVar, override

import cv2

from iivs_cardio.common.device import Device, DeviceKind
from iivs_cardio.optical_flow.estimators.opencv.estimator import (
    OpenCVAlgorithm,
    OpenCVConfig,
)


@dataclass(frozen=True, slots=True)
class DeepFlowConfig(OpenCVConfig):
    """DeepFlow's settings, of which it exposes none.

    Held anyway so every algorithm is built the same way, from a value that
    crosses a process boundary where a live algorithm cannot.

    Attributes:
        SUPPORTED_DEVICES: CPU alone, cv2 shipping no CUDA DeepFlow.
    """

    SUPPORTED_DEVICES: ClassVar[frozenset[DeviceKind]] = frozenset({"cpu"})

    @override
    def _algorithm(self, device: Device) -> OpenCVAlgorithm:
        """Make the DeepFlow algorithm, which takes no settings.

        Raises:
            ValueError: If `device` is a CUDA one. `build` refuses that first,
                so this answers whoever reached past it, who would otherwise
                hold a CPU algorithm labelled CUDA.
        """
        if device.is_cuda:
            msg = "DeepFlow is not available on CUDA devices"
            raise ValueError(msg)

        return cv2.optflow.createOptFlow_DeepFlow()
