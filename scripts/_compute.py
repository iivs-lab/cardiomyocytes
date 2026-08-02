from __future__ import annotations

__all__ = ("ComputeConfig", "plan_devices")

import os
from dataclasses import dataclass, field

from kaparoo.utils.optional import unwrap_or_default

from iivs_cardio.common.device import Device

DEFAULT_WORKERS = os.cpu_count() or 1


@dataclass
class ComputeConfig:
    device: str = "cpu"
    workers: int | None = None
    gpu_ids: list[int] | None = field(default_factory=lambda: [0])


def plan_devices(compute: ComputeConfig) -> tuple[Device, ...]:
    """One device per worker, in the order the workers will claim them.

    A single entry means this process does the work itself, so "sequential / many
    processes / many GPUs" is one length check downstream rather than three cases.

    Raises:
        ValueError: If the worker count is negative, or CUDA is asked for and the
            driver reports no device.
    """
    if not Device.resolve(compute.device).is_cuda:
        workers = unwrap_or_default(compute.workers, DEFAULT_WORKERS)
        if workers < 0:
            msg = f"invalid worker count {workers}: expected 0 or more, or null"
            raise ValueError(msg)

        return Device.resolve_all(["cpu"] * max(workers, 1))

    if not compute.gpu_ids:
        devices = Device.visible_cuda()
        if not devices:
            msg = "no CUDA device is visible: set `compute=cpu`, or check the driver"
            raise ValueError(msg)

        return devices

    return Device.resolve_all(f"cuda:{index}" for index in compute.gpu_ids)
