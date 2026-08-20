from __future__ import annotations

__all__ = ("DeepFlowConfig",)

from dataclasses import dataclass
from typing import ClassVar, override

import cv2

from iivs_cardio.common.device import Device, DeviceKind
from iivs_cardio.optical_flow.estimators.opencv.base import DenseAlgorithm, OpenCVConfig


@dataclass(frozen=True, slots=True)
class DeepFlowConfig(OpenCVConfig):
    """DeepFlow's (empty) recipe: it exposes no tunable parameters.

    Held anyway so every algorithm has a buildable config, letting a worker
    construct any of them through the one `build`.

    Attributes:
        SUPPORTED_DEVICES: CPU alone, since OpenCV ships no CUDA DeepFlow.
    """

    SUPPORTED_DEVICES: ClassVar[frozenset[DeviceKind]] = frozenset({"cpu"})

    @override
    def create(self, device: Device) -> DenseAlgorithm:
        """Create the DeepFlow algorithm, which takes neither settings nor a device.

        `device` is the contract's, not this algorithm's: `SUPPORTED_DEVICES`
        has already refused everything but the one it runs on.
        """
        return cv2.optflow.createOptFlow_DeepFlow()
