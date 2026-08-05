from __future__ import annotations

__all__ = (
    "ComputeConfig",
    "pin_threads",
    "plan_devices",
    "report_insights",
    "run_all",
)

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from kaparoo.utils.optional import unwrap_or_default
from mpire import WorkerPool
from tqdm import trange

from iivs_cardio.common.device import Device

if TYPE_CHECKING:
    from iivs_cardio.common.pipeline import StageFactory

DEFAULT_WORKERS = os.cpu_count() or 1

# Read before anything pins, so a share divides what torch would have taken on
# its own. A spawned worker re-imports this module and so reads its own default,
# which tracks physical cores rather than the logical count above.
_UNPINNED_THREADS = torch.get_num_threads()


@dataclass
class ComputeConfig:
    device: str = "cpu"
    workers: int | None = None
    gpu_ids: list[int] | None = None
    progress_bar: bool = True
    insights: bool = False
    worker_lifespan: int | None = None


def plan_devices(config: ComputeConfig) -> tuple[Device, ...]:
    """One device per worker, in the order the workers will claim them.

    A single entry means this process does the work itself, so "sequential / many
    processes / many GPUs" is one length check downstream rather than three cases.

    Each device reads one knob and ignores the other's, so setting the other's is
    refused rather than dropped: a `workers` a CUDA run cannot honour is a wall
    clock several times what the caller planned for, with nothing saying why.

    Raises:
        ValueError: If a knob belongs to the other device -- `workers` under
            CUDA, `gpu_ids` under CPU -- or the worker count is negative, or
            CUDA is asked for and the driver reports no device.
    """
    if not Device.resolve(config.device).is_cuda:
        if config.gpu_ids is not None:
            msg = "`gpu_ids` has no effect on cpu: drop it, or set `compute=cuda`"
            raise ValueError(msg)

        workers = unwrap_or_default(config.workers, DEFAULT_WORKERS)
        if workers < 0:
            msg = f"invalid worker count {workers}: expected 0 or more, or null"
            raise ValueError(msg)

        return Device.resolve_all(["cpu"] * max(workers, 1))

    if config.workers is not None:
        msg = "`workers` has no effect on cuda: use `gpu_ids` to pick the devices"
        raise ValueError(msg)

    if not config.gpu_ids:
        devices = Device.visible_cuda()
        if not devices:
            # `compute=cpu` names the hydra config group, not this parameter.
            msg = "no CUDA device is visible: set `compute=cpu`, or check the driver"
            raise ValueError(msg)

        return devices

    return Device.resolve_all(f"cuda:{index}" for index in config.gpu_ids)


def pin_threads(workers: int) -> None:
    """Hold this worker to its share of the machine's intra-op threads.

    torch sizes its thread pool to the machine in every process, so workers each
    claiming all of it contend rather than parallelise. Measured on 64 cores,
    sixteen unpinned workers ran 2.7x slower than no pool at all, while
    sixty-four pinned to one thread each beat the sequential path by 1.35x.

    A lone worker is left alone: it has the machine to itself, and the same
    measurement puts one unpinned process ahead of every pinned pool it tried
    below sixty-four workers. Only that widest point is measured -- the share
    between is this policy, not a result -- and it moves torch alone, so a stage
    that leaves torch for numpy is not covered.

    Args:
        workers: How many processes are sharing this machine.
    """
    if workers <= 1:
        return

    torch.set_num_threads(max(1, _UNPINNED_THREADS // workers))


def _run_on_worker(
    worker_id: int, shared: tuple[tuple[Device, ...], StageFactory], index: int
) -> None:
    devices, stages = shared

    device = devices[worker_id]
    device.activate()
    pin_threads(len(devices))

    stages.run_one(index, device)


def run_all(
    stages: StageFactory,
    config: ComputeConfig,
    *,
    desc: str = "running",
    unit: str = "it",
) -> None:
    """Run everything `stages` offers, sequentially or across a worker pool.

    The lone path goes through the same call the pool does, with worker `0` being
    this process: `plan_devices` answers one device for it, so the two differ in
    where the loop lives rather than in what an item gets.

    Args:
        stages: What to run, and what to hold open around the run.
        config: Which devices to divide the work across, and how to report it.
        desc: What the progress bar calls this run.
        unit: What the progress bar calls one item of it.
    """
    num_stages = len(stages)

    devices = plan_devices(config)[:num_stages]
    shared = (devices, stages)
    pbar_options = {"desc": desc, "unit": unit}

    with stages.running():
        if (num_workers := len(devices)) == 1:
            indices = trange(
                num_stages, disable=not config.progress_bar, **pbar_options
            )
            for index in indices:
                _run_on_worker(0, shared, index)
            return

        with WorkerPool(
            n_jobs=num_workers,
            shared_objects=shared,
            pass_worker_id=True,
            enable_insights=config.insights,
        ) as pool:
            pool.map(
                _run_on_worker,
                range(num_stages),
                chunk_size=1,
                worker_lifespan=config.worker_lifespan,
                progress_bar=config.progress_bar,
                progress_bar_options=pbar_options,
            )


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
