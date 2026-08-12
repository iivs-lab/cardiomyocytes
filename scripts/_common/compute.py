from __future__ import annotations

__all__ = (
    "ComputeConfig",
    "IncompleteRunError",
    "Outcome",
    "RunRecord",
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
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Final, NamedTuple

import torch
from kaparoo.filesystem import ensure_dir_exists
from kaparoo.utils import Timer, quantify, unwrap_or_default
from mpire import WorkerPool
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from iivs_cardio.common.device import Device, DeviceKind
from iivs_cardio.common.logging import log_indented
from iivs_cardio.common.pipeline import StageFactory

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from contextlib import AbstractContextManager
    from logging import Logger
    from pathlib import Path

    from kaparoo.filesystem.types import StrPath


# ========================== #
#          Settings          #
# ========================== #


_START_METHOD: Final = "spawn"


_DEFAULT_WORKERS: Final = unwrap_or_default(os.process_cpu_count(), 1)


_UNPINNED_THREADS: Final = torch.get_num_threads()


@dataclass
class ComputeConfig:
    """What a job is told about where to run and what to report.

    Attributes:
        device: The kind of device to run on, `"cpu"` or `"cuda"`.
        workers: The division of work, in the shape the device reads. A count
            on cpu, where `0` stays in this process; the gpu ids to take one
            worker each on cuda. Defaults to None, which lets the machine
            answer: every core on cpu, every visible gpu on cuda.
        tasks_per_worker: How many items a worker takes before it is replaced.
            Defaults to None, which keeps it for the whole run.
        measure_workers: Whether to ask the pool how each worker spent its time.
            Defaults to False.
        show_progress: Whether to draw a progress bar, when there is a terminal to draw
            it on. A redirected run says so and leaves it undrawn. Defaults to True.
    """

    device: DeviceKind = "cpu"
    workers: int | list[int] | None = None
    tasks_per_worker: int | None = None
    measure_workers: bool = False
    show_progress: bool = True


# ========================== #
#          Results           #
# ========================== #


class IncompleteRunError(RuntimeError):
    """Raised once a run has finished, when some of its items did not.

    What failed is carried whole rather than folded into the message, so a caller can
    retry exactly those items or record which of them are missing.

    Attributes:
        failed: The reason each failed item failed, keyed by its name.
        total: How many items the run was given.
    """

    def __init__(self, failed: Mapping[str, str], total: int) -> None:
        self.failed = dict(failed)
        self.total = total
        super().__init__(f"{len(self.failed)} of {total} failed")


class Outcome(NamedTuple):
    """What one item came back with.

    Attributes:
        index: The item this outcome belongs to. Carried back so a result can
            be recognised whatever order it arrives in.
        reason: The reason the item failed, or `None` if it did not.
        computed: Whether anything was read and computed for it. Defaults to
            False, which is also what an item that failed comes back with.
    """

    index: int
    reason: str | None
    computed: bool = False


@dataclass(slots=True)
class RunRecord:
    """What a run has learned about its items, as they come back.

    Filled as the results arrive rather than returned at the end: a pool that
    dies part way never returns, and what came back before it did is the
    grounds for a retry.

    Attributes:
        returned: The items that came back at all, which separates one nobody
            ran from one that ran and failed.
        unchanged: The items this run did not compute.
        failed: The reason each failed item failed, keyed by its name.
    """

    returned: set[str] = field(default_factory=set)
    unchanged: set[str] = field(default_factory=set)
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def ready(self) -> int:
        """How many items have an output to show for them."""
        return len(self.returned) - len(self.failed)

    def add(self, name: str, outcome: Outcome) -> None:
        """Take in what one item came back with."""
        self.returned.add(name)

        if outcome.reason is not None:
            self.failed[name] = outcome.reason
        elif not outcome.computed:
            self.unchanged.add(name)


# ========================== #
#          Workers           #
# ========================== #


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

        count = unwrap_or_default(workers, _DEFAULT_WORKERS)
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


class WorkerLogFolder:
    """The folder a run's workers write their own log files into.

    One file per worker rather than one shared file: several processes appending
    to the same file interleave, and on Windows they tear. A worker keeps its id
    across a restart, so its file is appended to rather than replaced, and
    clearing is the job's to do once before the run.

    The files are named for the run, as `<name>.worker0.log` beside the parent's
    own `<name>.log`. Two runs may be pointed at one folder, and the deliberate
    pairing is why they must not be pointed at one file: a stage that filters
    and a stage that estimates hold different configurations, so what a reader
    goes looking for is one run's lines rather than both in the order they
    happened.

    Args:
        root: An existing folder to write the files into.
        name: The run's name, which the files are named after and which the
            parent's own lines are filed under.
    """

    STEM: ClassVar[str] = "worker"

    # Kept the same as `hydra.job_logging` so a worker file and the parent's read
    # alike. The level is padded to the longest of them, or the message column
    # would step in and out as the level changed.
    _FORMAT: ClassVar[str] = "[%(asctime)s][%(name)s][%(levelname)-8s] - %(message)s"

    def __init__(self, root: StrPath, name: str) -> None:
        self.root = ensure_dir_exists(root)
        self.name = name

    def path_for(self, worker_id: int, num_workers: int) -> Path:
        """Return where worker `worker_id` of `num_workers` writes.

        The number is padded to the width the largest id needs, so the files of
        one run sort the way their workers are numbered.
        """
        width = len(str(num_workers - 1))
        return self.root / f"{self.name}.{self.STEM}{worker_id:0{width}d}.log"

    def list_logs(self) -> list[Path]:
        """Return the files this run's workers would write, that are here now.

        Only this run's are listed, so a folder holding another run's files
        reports none of them: they are that run's to clear.
        """
        return sorted(self.root.glob(f"{self.name}.{self.STEM}*.log"))

    def clear(self) -> None:
        """Delete the worker files an earlier job left under this run's name."""
        for stale in self.list_logs():
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
        name: The run's name, which its log lines are filed under.
        stages: The items to run, and how to run one.
        devices: One device per worker, indexed by worker id.
        log_folder: The folder a worker writes its own file into.
            Defaults to None, which leaves the process's logging alone.
        log_level: The level a worker logs at, taken from the parent.
            Defaults to `logging.INFO`.
    """

    name: str
    stages: StageFactory
    devices: tuple[Device, ...]
    log_folder: WorkerLogFolder | None
    log_level: int = logging.INFO


def _init_worker(worker_id: int, context: SharedContext) -> None:
    """Give a freshly started worker its own log file, once per process."""
    if context.log_folder is not None:
        num_workers = len(context.devices)
        context.log_folder.configure_worker(worker_id, num_workers, context.log_level)


def _run_on_worker(worker_id: int, context: SharedContext, index: int) -> Outcome:
    """Run one item on this worker and report what happened.

    A raised task tears the pool down and takes every item still to come with
    it, so anything that goes wrong here comes back as a value instead. That
    covers binding the device as well as running the item: binding happens per
    task rather than once, so its failure belongs to the task it happened on.

    Args:
        worker_id: The worker this is running on, which is how it picks its device.
        context: The state the pool handed every worker when it started.
        index: The item of the run to carry out.

    Returns:
        The outcome of the item.
    """
    devices = context.devices
    device = devices[worker_id]

    stages = context.stages

    try:
        device.activate()
        pin_threads(len(devices))
        computed = stages.run_stage(index, device)
    except Exception as error:
        logging.getLogger(context.name).exception("%s failed", stages.get_name(index))
        return Outcome(index, f"{type(error).__name__}: {error}")

    return Outcome(index, None, computed=computed)


# ========================== #
#          Running           #
# ========================== #


def log_compute_config(config: ComputeConfig, logger: Logger) -> None:
    """Log the compute settings a run was given, before it resolves them.

    Only settings that were moved get a line, so a run that changed nothing
    beyond its device says only that. What the run then actually planned is
    reported by `run_all`, which is the one that knows it.
    """
    log_indented(logger, "compute: %s", config.device, depth=0)

    if (tasks := config.tasks_per_worker) is not None:
        log_indented(logger, "starting a fresh worker every %d tasks", tasks)

    if config.measure_workers:
        log_indented(logger, "measuring how each worker spends its time")

    if not config.show_progress:
        log_indented(logger, "drawing no progress bar")


def log_insights(insights: dict[str, Any], name: str, *, unit: str = "it") -> None:
    """Log how the pool's workers spent their time, and what each finished.

    The shares are of a worker's whole life, so what is left after working and waiting
    is the cost of being a worker at all: starting, setting up, and stopping.

    Args:
        insights: The measurements the pool collected, empty if it was not
            asked for them.
        name: The run's name, so the lines are filed with the rest.
        unit: The name for one item, used in the per worker counts. Defaults to `"it"`.
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


def _collect_outcomes(
    outcomes: Iterable[Outcome],
    stages: StageFactory,
    logger: logging.Logger,
    record: RunRecord,
) -> None:
    """Log a verdict for each result as it arrives, and take it into `record`.

    Args:
        outcomes: The outcome of each item, in the order they arrive.
        stages: The run's items, for naming an index.
        logger: The logger the per item verdict goes to.
        record: The record to fill as the results arrive. Given rather than returned,
            so a pool that dies part way leaves behind what came back before it did.
    """
    total = len(stages)

    for count, outcome in enumerate(outcomes, start=1):
        name = stages.get_name(outcome.index)
        record.add(name, outcome)

        verdict = "unchanged"
        if outcome.reason is not None:
            verdict = "failed"
        elif outcome.computed:
            verdict = "computed"

        logger.info("%s %s (%d/%d)", name, verdict, count, total)


def _drawing(*, progress: bool) -> AbstractContextManager[None]:
    """Return a context in which a log line does not tear the bar it lands on.

    Console handlers are routed through `tqdm.write`, which clears the bar
    before the line and draws it again after. File handlers are left alone, so
    what reaches the log on disk is unchanged. With no bar there is nothing to
    tear, and nothing is done.
    """
    return logging_redirect_tqdm() if progress else nullcontext()


def _tracked(
    outcomes: Iterable[Outcome], context: SharedContext, *, unit: str, drawn: bool
) -> Iterable[Outcome]:
    """Return `outcomes`, advancing a bar as each one is taken.

    Both run paths draw through here, so the bar counts one thing: the results
    this process has in hand. Left to `mpire`, the pool's own bar counts what
    the workers reported finishing, which is a different clock and reaches the
    end while the parent is still draining the queue.

    The bar therefore follows collection rather than computation, and `imap`
    hands results back in order, so one slow item holds the bar behind workers
    that have already moved on. It catches up by the end, which is where the
    two clocks disagreed.
    """
    total = len(context.stages)

    return tqdm(outcomes, total=total, desc=context.name, unit=unit, disable=not drawn)


def _run_in_process(
    context: SharedContext, record: RunRecord, *, unit: str, show_progress: bool
) -> None:
    """Run every item in this process, one after another.

    A pool of one only costs a process, so a lone worker stays here.

    Args:
        context: The state a worker would have been handed.
        record: The record to fill as the items come back.
        unit: The name for one item, used in the progress bar.
        show_progress: Whether to draw the progress bar.
    """
    stages = context.stages
    logger = logging.getLogger(context.name)

    outcomes = (_run_on_worker(0, context, index) for index in range(len(stages)))
    tracked = _tracked(outcomes, context, unit=unit, drawn=show_progress)

    _collect_outcomes(tracked, stages, logger, record)


def _run_in_pool(
    context: SharedContext,
    config: ComputeConfig,
    record: RunRecord,
    *,
    unit: str,
    show_progress: bool,
) -> None:
    """Run the items across a pool of worker processes.

    Args:
        context: The state handed to every worker when the pool starts.
        config: The settings saying what to report and how long a worker
            lives.
        record: The record to fill as the items come back.
        unit: The name for one item, used in the progress bar and the insights.
        show_progress: Whether to draw the progress bar.
    """
    stages = context.stages
    logger = logging.getLogger(context.name)

    with WorkerPool(
        n_jobs=len(context.devices),
        shared_objects=context,
        pass_worker_id=True,
        enable_insights=config.measure_workers,
        start_method=_START_METHOD,
    ) as pool:
        outcomes = pool.imap(
            _run_on_worker,
            range(len(stages)),
            chunk_size=1,
            worker_init=_init_worker,
            worker_lifespan=config.tasks_per_worker,
        )
        tracked = _tracked(outcomes, context, unit=unit, drawn=show_progress)

        _collect_outcomes(tracked, stages, logger, record)

        if config.measure_workers:
            log_insights(pool.get_insights(), context.name, unit=unit)


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
        stages: The items to run, and how to run one.
        config: The device to run them on, and what to report.
        unit: The name for one item, used in the progress bar and the summary.
            Defaults to `"it"`.
        log_folder: The folder workers write their own files into, named for
            this same run. Defaults to None, which leaves their logging alone.

    Raises:
        ValueError: If the log folder is named for another run. Its files would
            then be filed under one name and the parent's own under another,
            which no reader could pair up again.
        IncompleteRunError: If any item failed, raised once the rest have
            finished.
    """
    name = stages.name
    logger = logging.getLogger(name)

    if log_folder is not None and log_folder.name != name:
        msg = f"log folder is named for {log_folder.name!r}: expected {name!r}"
        raise ValueError(msg)

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

    if log_folder is not None and (stale := log_folder.list_logs()):
        left = quantify(len(stale), "worker log")
        listed = ", ".join(path.name for path in stale)
        logger.warning("%s left by an earlier run: %s", left, listed)

    log_level = logger.getEffectiveLevel()
    context = SharedContext(name, stages, devices, log_folder, log_level)

    watched = sys.stderr.isatty()
    if config.show_progress and not watched:
        logger.warning("not drawing the progress bar: stderr is not a terminal")

    progress = config.show_progress and watched and num_stages > 1

    record = RunRecord()

    try:
        with _drawing(progress=progress), Timer("s") as timer, stages.running():
            if in_process:
                _run_in_process(context, record, unit=unit, show_progress=progress)
            else:
                _run_in_pool(context, config, record, unit=unit, show_progress=progress)
    except Exception:
        if not record.failed:
            raise

        logger.exception("every item was seen, but the run could not be closed")

    ready = record.ready
    unchanged = len(record.unchanged)
    counted = f"{ready - unchanged} computed, {unchanged} unchanged"
    split = f" ({counted})" if unchanged else ""

    logger.info("%d of %d ready in %.1fs%s", ready, num_stages, timer.elapsed, split)

    if (missing := num_stages - len(record.returned)) > 0:
        logger.error("%d never came back: the pool went down with them", missing)

    if record.failed:
        for stage, reason in record.failed.items():
            logger.error("%s: %s", stage, reason)

        raise IncompleteRunError(record.failed, num_stages)
