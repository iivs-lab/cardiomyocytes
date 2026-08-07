from __future__ import annotations

__all__ = (
    "ComputeConfig",
    "IncompleteRunError",
    "SharedContext",
    "WorkerLogFolder",
    "log_compute_config",
    "log_insights",
    "pin_threads",
    "plan_devices",
    "run_all",
)

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from kaparoo.filesystem import ensure_dir_exists
from kaparoo.utils import Timer, unwrap_or_default
from mpire import WorkerPool
from tqdm import trange

from iivs_cardio.common.device import Device
from iivs_cardio.common.logging import log_indented

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from logging import Logger
    from pathlib import Path

    from kaparoo.filesystem.types import StrPath

    from iivs_cardio.common.pipeline import StageFactory

DEFAULT_WORKERS = unwrap_or_default(os.cpu_count(), 1)

_UNPINNED_THREADS = torch.get_num_threads()


@dataclass
class ComputeConfig:
    device: str = "cpu"
    workers: int | list[int] | None = None
    tasks_per_worker: int | None = None
    show_progress: bool = True
    measure_workers: bool = False


class IncompleteRunError(RuntimeError):
    def __init__(self, failed: Mapping[str, str], total: int) -> None:
        self.failed = dict(failed)
        self.total = total
        super().__init__(f"{len(self.failed)} of {total} failed")


class WorkerLogFolder:
    STEM: ClassVar[str] = "worker"

    _FORMAT: ClassVar[str] = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"

    def __init__(self, root: StrPath) -> None:
        self.root = ensure_dir_exists(root)

    def path_for(self, worker_id: int, num_workers: int) -> Path:
        width = len(str(num_workers - 1))
        return self.root / f"{self.STEM}{worker_id:0{width}d}.log"

    def clear(self) -> None:
        for stale in self.root.glob(f"{self.STEM}*.log"):
            stale.unlink()

    def configure_worker(
        self, worker_id: int, num_workers: int, level: int = logging.INFO
    ) -> None:
        log_file = self.path_for(worker_id, num_workers)

        handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter(self._FORMAT))

        logger = logging.getLogger()
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(level)


@dataclass(frozen=True, slots=True)
class SharedContext:
    devices: tuple[Device, ...]
    stages: StageFactory
    name: str
    log_folder: WorkerLogFolder | None
    log_level: int = logging.INFO


def log_compute_config(config: ComputeConfig, logger: Logger) -> None:
    log_indented(logger, "compute: %s", config.device, depth=0)

    if config.tasks_per_worker is not None:
        log_indented(
            logger, "replacing a worker after %d tasks", config.tasks_per_worker
        )

    if config.measure_workers:
        log_indented(logger, "reporting how busy each worker was")

    if not config.show_progress:
        log_indented(logger, "showing no progress bar")


def plan_devices(config: ComputeConfig) -> tuple[Device, ...]:
    workers = config.workers

    if not Device.resolve(config.device).is_cuda:
        if workers is not None and not isinstance(workers, int):
            msg = "`workers` on cpu is a count: use `compute=cuda` to name gpu ids"
            raise TypeError(msg)

        count = unwrap_or_default(workers, DEFAULT_WORKERS)
        if count < 0:
            msg = f"invalid worker count {count}: expected 0 or more, or null"
            raise ValueError(msg)

        return Device.resolve_all(["cpu"] * max(count, 1))

    if isinstance(workers, int):
        msg = f"`workers` on cuda names gpu ids: use [{workers}] rather than {workers}"
        raise TypeError(msg)

    if not workers:
        devices = Device.visible_cuda()
        if not devices:
            msg = "no CUDA device is visible: set `compute=cpu`, or check the driver"
            raise ValueError(msg)

        return devices

    return Device.resolve_all(f"cuda:{index}" for index in workers)


def pin_threads(max_workers: int) -> None:
    if max_workers <= 1:
        return

    torch.set_num_threads(max(1, _UNPINNED_THREADS // max_workers))


def _init_worker(worker_id: int, context: SharedContext) -> None:
    if context.log_folder is not None:
        num_workers = len(context.devices)
        context.log_folder.configure_worker(worker_id, num_workers, context.log_level)


def _run_on_worker(
    worker_id: int, context: SharedContext, index: int
) -> tuple[int, str] | None:
    devices = context.devices
    stages = context.stages
    device = devices[worker_id]
    device.activate()
    pin_threads(len(devices))

    try:
        stages.run_stage(index, device)
    except Exception as error:
        logging.getLogger(context.name).exception("%s failed", stages.get_name(index))
        return index, f"{type(error).__name__}: {error}"

    return None


def _watch(
    outcomes: Iterable[tuple[int, str] | None],
    stages: StageFactory,
    logger: logging.Logger,
    total: int,
) -> list[tuple[int, str]]:
    failed: list[tuple[int, str]] = []

    for index, outcome in enumerate(outcomes):
        if outcome is not None:
            failed.append(outcome)

        verdict = "done" if outcome is None else "failed"
        logger.info("%s %s (%d/%d)", stages.get_name(index), verdict, index + 1, total)

    return failed


def log_insights(insights: dict[str, Any], name: str, *, unit: str = "it") -> None:
    logger = logging.getLogger(name)

    if not insights:
        logger.warning("nothing to report: the pool collected no insights")
        return

    summary = (
        "workers spent %s: %.1f%% working, %.1f%% waiting, %.1f%% starting/stopping"
    )
    working = insights["working_ratio"] * 100
    waiting = insights["waiting_ratio"] * 100
    idle = 100 - working - waiting
    logger.info(summary, insights["total_time"], working, waiting, idle)

    per_worker = "worker %d completed %d %s in %s"
    rows = zip(insights["n_completed_tasks"], insights["working_time"], strict=True)
    for worker_id, (completed, spent) in enumerate(rows):
        log_indented(logger, per_worker, worker_id, completed, unit, spent)


def run_all(
    stages: StageFactory,
    config: ComputeConfig,
    *,
    unit: str = "it",
    log_folder: WorkerLogFolder | None = None,
) -> None:
    name = stages.name
    logger = logging.getLogger(name)

    log_compute_config(config, logger)

    num_stages = len(stages)
    devices = plan_devices(config)[:num_stages]
    num_workers = len(devices)
    one_worker = num_workers == 1

    log_level = logger.getEffectiveLevel()
    context = SharedContext(devices, stages, name, log_folder, log_level)

    if config.measure_workers and one_worker:
        logger.warning("not measuring workers: a lone worker runs no pool")

    show_progress = config.show_progress and num_stages > 1
    pbar_options = {"desc": name, "unit": unit}

    stages_str = f"{num_stages} {unit}"
    workers_str = f"{num_workers} worker{'' if one_worker else 's'}"
    devices_str = ", ".join(str(device) for device in dict.fromkeys(devices))

    logger.info("running %s across %s on %s", stages_str, workers_str, devices_str)

    with Timer("s") as timer, stages.running():
        if one_worker:
            indices = trange(num_stages, disable=not show_progress, **pbar_options)
            outcomes = (_run_on_worker(0, context, index) for index in indices)
            failed = _watch(outcomes, stages, logger, num_stages)
        else:
            with WorkerPool(
                n_jobs=num_workers,
                shared_objects=context,
                pass_worker_id=True,
                enable_insights=config.measure_workers,
            ) as pool:
                outcomes = pool.imap(
                    _run_on_worker,
                    range(num_stages),
                    chunk_size=1,
                    worker_init=_init_worker,
                    worker_lifespan=config.tasks_per_worker,
                    progress_bar=show_progress,
                    progress_bar_options=pbar_options,
                )
                failed = _watch(outcomes, stages, logger, num_stages)

                if config.measure_workers:
                    log_insights(pool.get_insights(), name, unit=unit)

    completed = num_stages - len(failed)
    logger.info("%d of %d done in %.1fs", completed, num_stages, timer.elapsed)

    if failed:
        named = {stages.get_name(index): why for index, why in failed}
        for stage, why in named.items():
            logger.error("%s: %s", stage, why)

        raise IncompleteRunError(named, num_stages)
