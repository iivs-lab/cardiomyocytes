from __future__ import annotations

__all__ = ("ComputeConfig", "plan_devices", "report_insights")

import os
from dataclasses import dataclass, field
from typing import Any

from kaparoo.utils.optional import unwrap_or_default

from iivs_cardio.common.device import Device

DEFAULT_WORKERS = os.cpu_count() or 1


@dataclass
class ComputeConfig:
    device: str = "cpu"
    workers: int | None = None
    gpu_ids: list[int] | None = field(default_factory=lambda: [0])
    progress_bar: bool = True
    insights: bool = False
    worker_lifespan: int | None = None


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


def report_insights(insights: dict[str, Any]) -> None:
    """Print what each worker spent its time on, as `mpire` measured it.

    Written to stdout, which hydra captures into the job's own log, so a long run
    leaves behind the timings that say *why* it took what it took -- a sweep that
    finishes silently can only be re-run to find out.

    Args:
        insights: What `WorkerPool.get_insights` returned. Its per-worker entries
            are lists indexed by worker id, and its times are preformatted
            strings rather than numbers.
    """
    print(
        f"insights: {insights['total_time']} total, "
        f"{insights['working_ratio']:.1%} working, "
        f"{insights['waiting_ratio']:.1%} waiting"
    )
    rows = zip(insights["n_completed_tasks"], insights["working_time"], strict=True)
    for worker_id, (completed, working) in enumerate(rows):
        print(f"  worker {worker_id}: {completed} sequences, {working} working")
