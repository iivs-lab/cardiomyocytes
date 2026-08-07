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
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Final

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

# Affinity-aware, so `taskset`, a cpuset, or a scheduler's allocation is
# followed. A cgroup cpu quota is not visible here and is not followed.
DEFAULT_WORKERS: Final = unwrap_or_default(os.process_cpu_count(), 1)

# Named rather than left to `mpire`, which forks where the platform allows it.
# A worker cannot inherit this process's CUDA context, only start its own.
START_METHOD: Final = "spawn"

_UNPINNED_THREADS: Final = torch.get_num_threads()


@dataclass
class ComputeConfig:
    """What a job is told about where to run and what to report.

    Attributes:
        device: the kind of device to run on, `cpu` or `cuda`.
        workers: how the work is spread, in the shape the device reads. On cpu
            a count, where `0` stays in this process; on cuda the gpu ids to
            take one worker each. `None` lets the machine answer: every core on
            cpu, every visible gpu on cuda.
        tasks_per_worker: how many items a worker takes before it is replaced,
            or `None` to keep it for the whole run.
        measure_workers: whether to ask the pool how busy each worker was.
        show_progress: whether to draw a progress bar, when there is a terminal
            to draw it on. A redirected run says so and leaves it undrawn.
    """

    device: str = "cpu"
    workers: int | list[int] | None = None
    tasks_per_worker: int | None = None
    measure_workers: bool = False
    show_progress: bool = True


class IncompleteRunError(RuntimeError):
    """Raised once a run has finished, when some of its items did not.

    What failed is carried whole rather than folded into the message, so a
    caller can retry exactly those items or record which of them are missing.

    Attributes:
        failed: why each failed item failed, keyed by its name.
        total: how many items the run was given.
    """

    def __init__(self, failed: Mapping[str, str], total: int) -> None:
        self.failed = dict(failed)
        self.total = total
        super().__init__(f"{len(self.failed)} of {total} failed")


class WorkerLogFolder:
    """The folder a run's workers write their own log files into.

    One file per worker rather than one shared file: several processes appending
    to the same file interleave, and on Windows they tear. A worker keeps its id
    across a restart, so its file is appended to rather than replaced, and
    clearing is the job's to do once before the run.

    Args:
        root: an existing folder to write the files into.
    """

    STEM: ClassVar[str] = "worker"

    _FORMAT: ClassVar[str] = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"

    def __init__(self, root: StrPath) -> None:
        self.root = ensure_dir_exists(root)

    def path_for(self, worker_id: int, num_workers: int) -> Path:
        """Return where worker `worker_id` of `num_workers` writes.

        The number is padded to the width the largest id needs, so the files of
        one run sort the way their workers are numbered.
        """
        width = len(str(num_workers - 1))
        return self.root / f"{self.STEM}{worker_id:0{width}d}.log"

    def clear(self) -> None:
        """Delete the worker files an earlier job left in this folder."""
        for stale in self.root.glob(f"{self.STEM}*.log"):
            stale.unlink()

    def configure_worker(
        self, worker_id: int, num_workers: int, level: int = logging.INFO
    ) -> None:
        """Send everything this process logs to its own file, at `level`.

        This replaces the process's root handlers, so that every module's lines
        land in the worker's file and not only those of one logger. A worker
        process starts with none of the parent's logging, which is why the
        level has to be given rather than inherited.
        """
        log_file = self.path_for(worker_id, num_workers)

        handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter(self._FORMAT))

        logger = logging.getLogger()
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(level)


@dataclass(frozen=True, slots=True)
class SharedContext:
    """Everything a worker is handed, sent out once when the pool starts.

    It travels one way. A worker gets its own copy, so what it changes there is
    never seen again by the parent or by any other worker.

    Attributes:
        name: what the run is called, and what its log lines are filed under.
        stages: the items to run, and how to run one.
        devices: one device per worker, indexed by worker id.
        log_folder: where a worker writes its own file, or `None` to leave the
            process's logging alone.
        log_level: the level a worker logs at, taken from the parent.
    """

    name: str
    stages: StageFactory
    devices: tuple[Device, ...]
    log_folder: WorkerLogFolder | None
    log_level: int = logging.INFO


def log_compute_config(config: ComputeConfig, logger: Logger) -> None:
    """Log the compute settings a run was given, before it resolves them.

    Only settings that were moved get a line, so a run that changed nothing
    beyond its device says only that. What the run then actually planned is
    reported by `run_all`, which is the one that knows it.
    """
    log_indented(logger, "compute: %s", config.device, depth=0)

    if (tasks := config.tasks_per_worker) is not None:
        log_indented(logger, "replacing a worker after %d tasks", tasks)

    if config.measure_workers:
        log_indented(logger, "reporting how busy each worker was")

    if not config.show_progress:
        log_indented(logger, "showing no progress bar")


def plan_devices(config: ComputeConfig) -> tuple[Device, ...]:
    """Turn the compute settings into one device per worker.

    There is always at least one, since a run with no worker has nowhere to
    happen. A gpu id may repeat, which puts two workers on that device.

    Returns:
        The device each worker will take, in worker order.

    Raises:
        TypeError: If `workers` is not the shape the device reads: a count on
            cpu, gpu ids on cuda.
        ValueError: If the count is negative, or no CUDA device is visible when
            one was asked for.
    """
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
    """Hold this process to its share of the machine's compute threads.

    Every process otherwise sizes its thread pool to the whole machine, and
    they then contend. A lone worker keeps the machine to itself.
    """
    if max_workers <= 1:
        return

    torch.set_num_threads(max(1, _UNPINNED_THREADS // max_workers))


def _init_worker(worker_id: int, context: SharedContext) -> None:
    """Give a freshly started worker its own log file, once per process."""
    if context.log_folder is not None:
        num_workers = len(context.devices)
        context.log_folder.configure_worker(worker_id, num_workers, context.log_level)


def _run_on_worker(
    worker_id: int, context: SharedContext, index: int
) -> tuple[int, str | None]:
    """Run one item, returning why it failed instead of raising.

    A raised task tears the pool down and takes every item still to come with
    it, so the failure comes back as a value. The item's own index comes back
    with it, which is what lets a result be recognised whatever order it
    arrives in.
    """
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

    return index, None


def _collect_failures(
    outcomes: Iterable[tuple[int, str | None]],
    stages: StageFactory,
    logger: logging.Logger,
) -> dict[str, str]:
    """Log a verdict for each result as it arrives, and gather the failures.

    Returns:
        Why each failed item failed, keyed by its name.
    """
    failed: dict[str, str] = {}
    total = len(stages)

    for returned, (index, reason) in enumerate(outcomes, start=1):
        name = stages.get_name(index)
        if reason is not None:
            failed[name] = reason

        verdict = "done" if reason is None else "failed"
        logger.info("%s %s (%d/%d)", name, verdict, returned, total)

    return failed


def log_insights(insights: dict[str, Any], name: str, *, unit: str = "it") -> None:
    """Log how the pool's workers spent their time, and what each finished.

    The shares are of a worker's whole life, so what is left after working and
    waiting is the cost of being a worker at all: starting, setting up, and
    stopping.

    Args:
        insights: what the pool collected, empty if it was not asked to.
        name: what the run is called, so the lines are filed with the rest.
        unit: what one item is called, for the per worker counts.
    """
    logger = logging.getLogger(name)

    if not insights:
        logger.warning("nothing to report: the pool collected no insights")
        return

    summary = "workers spent %s: %.1f%% working, %.1f%% waiting, %.1f%% overhead"
    working = insights["working_ratio"] * 100
    waiting = insights["waiting_ratio"] * 100
    overhead = 100 - (working + waiting)
    logger.info(summary, insights["total_time"], working, waiting, overhead)

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
    """Run every item a job offers, and report what got through.

    A single worker runs here rather than in a pool, since a pool of one only
    costs a process. Whatever has to outlive one item is opened around the whole
    run, and a failure while closing it does not take the verdict with it: once
    every item has been seen, the run still says which of them failed.

    Args:
        stages: the items to run, and how to run one.
        config: where to run them and what to report.
        unit: what one item is called, for the progress bar and the summary.
        log_folder: where workers write their own files, or `None` to leave
            their logging alone.

    Raises:
        IncompleteRunError: If any item failed, raised once the rest have
            finished.
    """
    name = stages.name
    logger = logging.getLogger(name)

    log_compute_config(config, logger)

    num_stages = len(stages)
    devices = plan_devices(config)[:num_stages]
    num_workers = len(devices)
    in_process = num_workers <= 1

    workers = f"{num_workers} worker{'' if in_process else 's'}"
    where = ", ".join(str(device) for device in dict.fromkeys(devices))
    logger.info("running %d %s across %s on %s", num_stages, unit, workers, where)

    if config.measure_workers and in_process:
        logger.warning("not measuring workers: a lone worker runs no pool")

    log_level = logger.getEffectiveLevel()
    context = SharedContext(name, stages, devices, log_folder, log_level)

    watched = sys.stderr.isatty()
    if config.show_progress and not watched:
        logger.warning("not drawing progress: stderr is not a terminal")

    show_progress = config.show_progress and watched and num_stages > 1

    failed: dict[str, str] = {}

    with Timer("s") as timer:
        try:
            with stages.running():
                if in_process:
                    bar = trange(
                        num_stages, desc=name, unit=unit, disable=not show_progress
                    )
                    outcomes = (_run_on_worker(0, context, index) for index in bar)
                    failed = _collect_failures(outcomes, stages, logger)
                else:
                    with WorkerPool(
                        n_jobs=num_workers,
                        shared_objects=context,
                        pass_worker_id=True,
                        enable_insights=config.measure_workers,
                        start_method=START_METHOD,
                    ) as pool:
                        outcomes = pool.imap(
                            _run_on_worker,
                            range(num_stages),
                            chunk_size=1,
                            worker_init=_init_worker,
                            worker_lifespan=config.tasks_per_worker,
                            progress_bar=show_progress,
                            progress_bar_options={"desc": name, "unit": unit},
                        )
                        failed = _collect_failures(outcomes, stages, logger)

                        if config.measure_workers:
                            log_insights(pool.get_insights(), name, unit=unit)
        except Exception:
            if not failed:
                raise

            logger.exception("every item was seen, but the run could not be closed")

    completed = num_stages - len(failed)
    logger.info("%d of %d done in %.1fs", completed, num_stages, timer.elapsed)

    if failed:
        for stage, reason in failed.items():
            logger.error("%s: %s", stage, reason)

        raise IncompleteRunError(failed, num_stages)
