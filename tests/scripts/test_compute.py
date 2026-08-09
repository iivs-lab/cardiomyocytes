from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
import torch
from mpire import WorkerPool

from iivs_cardio.common.device import Device
from scripts import _compute as compute
from scripts._compute import (
    _DEFAULT_WORKERS,
    ComputeConfig,
    IncompleteRunError,
    Outcome,
    RunRecord,
    SharedContext,
    WorkerLogFolder,
    _collect_outcomes,
    _drawing,
    _init_worker,
    log_compute_config,
    log_insights,
    pin_threads,
    plan_devices,
    run_all,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path


class _Stages:
    """A stage factory that records what ran, through the filesystem.

    Through files rather than an attribute, because the pool hands each worker
    its own copy: what a worker did to itself never comes home.
    """

    def __init__(
        self,
        count: int,
        dest: Path,
        explode_at: Iterable[int] = (),
        reuse_at: Iterable[int] = (),
    ) -> None:
        self._count = count
        self._dest = dest
        self._explode_at = frozenset(explode_at)
        self._reuse_at = frozenset(reuse_at)

    @property
    def name(self) -> str:
        return "run"

    def __len__(self) -> int:
        return self._count

    def get_name(self, index: int) -> str:
        return f"item{index}"

    def run_stage(self, index: int, device: Device) -> bool:
        if index in self._explode_at:
            msg = f"item {index} gave up"
            raise ValueError(msg)

        if index in self._reuse_at:
            return False

        (self._dest / f"{index:03d}.done").write_text("", encoding="utf-8")

        return True

    @contextmanager
    def running(self) -> Iterator[_Stages]:
        self._dest.mkdir(parents=True, exist_ok=True)
        yield self


def _done(dest: Path) -> list[int]:
    return sorted(int(path.stem) for path in dest.glob("*.done"))


def _compute(workers: int) -> ComputeConfig:
    return ComputeConfig(device="cpu", workers=workers, show_progress=False)


@pytest.mark.parametrize("workers", (0, 2))
def test_a_clean_run_finishes_quietly(tmp_path, workers):
    dest = tmp_path / "done"

    run_all(_Stages(4, dest), _compute(workers))

    assert _done(dest) == [0, 1, 2, 3]


@pytest.mark.parametrize("workers", (0, 2))
def test_one_item_failing_does_not_take_the_rest_with_it(tmp_path, workers):
    # `mpire` re-raises a task's exception in the parent and tears the pool down,
    # so a run that let one through would lose every item still to come: hours
    # of finished sequences, at dataset scale.
    dest = tmp_path / "done"

    with pytest.raises(IncompleteRunError, match=r"1 of 4 failed"):
        run_all(_Stages(4, dest, explode_at=[1]), _compute(workers))

    assert _done(dest) == [0, 2, 3]


@pytest.mark.parametrize("workers", (0, 2))
def test_what_failed_stays_whole_rather_than_folded_into_the_message(tmp_path, workers):
    # A caller acts on this, with a report saying it covers a subset or with a
    # retry, and a dataset's worth of failures would not fit one line anyway.
    dest = tmp_path / "done"

    with pytest.raises(IncompleteRunError) as failure:
        run_all(_Stages(4, dest, explode_at=[0, 3]), _compute(workers))

    assert str(failure.value) == "2 of 4 failed"
    assert failure.value.total == 4
    assert failure.value.failed == {
        "item0": "ValueError: item 0 gave up",
        "item3": "ValueError: item 3 gave up",
    }


def test_the_run_is_bracketed_before_anything_says_it_failed(tmp_path):
    # A side branch that gathers across the run has to commit what did finish
    # before the summary raises, or a failure would throw the whole document away.
    dest = tmp_path / "done"
    closed: list[int] = []

    class _Gathering(_Stages):
        @contextmanager
        def running(self) -> Iterator[_Stages]:
            self._dest.mkdir(parents=True, exist_ok=True)
            yield self
            closed.append(len(_done(self._dest)))

    with pytest.raises(IncompleteRunError, match=r"1 of 3 failed"):
        run_all(_Gathering(3, dest, explode_at=[2]), _compute(0))

    assert closed == [2]


class _Unclosable(_Stages):
    @contextmanager
    def running(self) -> Iterator[_Stages]:
        self._dest.mkdir(parents=True, exist_ok=True)
        yield self
        msg = "value_range.json already exists"
        raise FileExistsError(msg)


def test_a_branch_that_cannot_commit_does_not_bury_what_failed(tmp_path, caplog):
    # The verdict is what a retry is built from, and a side branch raising on its
    # way out would otherwise replace it with a file error naming no sequence.
    dest = tmp_path / "done"

    with caplog.at_level(logging.INFO), pytest.raises(IncompleteRunError) as failure:
        run_all(_Unclosable(3, dest, explode_at=[1]), _compute(0))

    assert failure.value.failed == {"item1": "ValueError: item 1 gave up"}

    logged = [record.getMessage() for record in caplog.records]
    assert "2 of 3 ready" in " ".join(logged)
    assert any("could not be closed" in message for message in logged)


def test_a_branch_that_cannot_commit_is_the_verdict_when_nothing_failed(tmp_path):
    # With every item through, the broken output is the only thing left to report.
    with pytest.raises(FileExistsError, match=r"already exists"):
        run_all(_Unclosable(3, tmp_path / "done"), _compute(0))


@pytest.mark.parametrize("workers", (0, 2))
def test_a_run_with_no_items_starts_no_pool(tmp_path, monkeypatch, workers):
    # `plan_devices` is capped at the item count, so an empty run plans no worker
    # at all and must not reach `WorkerPool`, which cannot be asked for none.
    # "It did not raise" only caught the narrowest way back, so the pool itself
    # is watched: reaching it at all is the failure, whatever it then did.
    def refuse(*args, **kwargs):
        pytest.fail("an empty run reached the pool")

    monkeypatch.setattr("scripts._compute.WorkerPool", refuse)

    run_all(_Stages(0, tmp_path / "done"), _compute(workers))


def test_the_pool_starts_its_workers_fresh_rather_than_forked(tmp_path, monkeypatch):
    # Spied rather than observed, because the platform this runs on has no fork
    # to get wrong. `mpire` takes it where it can, and this run has already asked
    # the driver about the GPUs by the time the pool starts, so a forked worker
    # would inherit a CUDA context it cannot use and fail every item it took.
    started = {}

    def spy(*args, **kwargs):
        started.update(kwargs)
        return WorkerPool(*args, **kwargs)

    monkeypatch.setattr("scripts._compute.WorkerPool", spy)
    run_all(_Stages(2, tmp_path / "done"), _compute(2))

    assert started["start_method"] == "spawn"


@pytest.fixture()
def restored_thread_count():
    """Put torch's thread count back, since pinning changes it process-wide."""
    before = torch.get_num_threads()

    yield before

    torch.set_num_threads(before)


def test_a_lone_worker_keeps_the_machine_to_itself(restored_thread_count):
    pin_threads(1)

    assert torch.get_num_threads() == restored_thread_count


def test_workers_take_a_share_of_the_machine_each(restored_thread_count):
    # Measured on 64 cores: sixteen unpinned workers ran 2.7x slower than no
    # pool at all, since every process sizes its pool to the whole machine. The
    # share is what the pool path runs, and the pool runs it in a subprocess --
    # so nothing here had ever seen it happen.
    pin_threads(4)

    assert torch.get_num_threads() == max(1, restored_thread_count // 4)


class _Talkative(_Stages):
    """A factory whose items say something only a low level lets through."""

    def run_stage(self, index: int, device: Device) -> bool:
        logging.getLogger(self.name).debug("item %d had something to say", index)

        return super().run_stage(index, device)


@pytest.mark.usefixtures("restored_root_logger")
def test_the_level_the_parent_runs_at_reaches_the_worker_files(tmp_path):
    # A spawned worker starts at `WARNING` with no handlers, so the level has to
    # travel with the rest of the context. The half below covers the arrival;
    # this covers the departure, which nothing held: `run_all` could carry a
    # constant and every other test would still pass, leaving the per sequence
    # lines out of the worker files exactly when a long run needs reading.
    dest = tmp_path / "done"
    logging.getLogger("run").setLevel(logging.DEBUG)

    run_all(
        _Talkative(2, dest),
        _compute(2),
        log_folder=WorkerLogFolder(tmp_path),
    )

    written = "".join(
        path.read_text(encoding="utf-8") for path in tmp_path.glob("worker*.log")
    )
    assert "had something to say" in written


@pytest.mark.usefixtures("restored_root_logger")
def test_a_starting_worker_takes_the_level_the_parent_was_at(tmp_path):
    # A spawned worker starts with no handlers at `WARNING`, so a level not
    # carried across leaves its file without a single line of the run. Nothing
    # held that: every other test names the level when it configures a worker,
    # so the field could be dropped and they would all still pass.
    folder = WorkerLogFolder(tmp_path)
    context = SharedContext("run", _Stages(2, tmp_path), (Device("cpu"),) * 2, folder)

    _init_worker(1, replace(context, log_level=logging.DEBUG))
    logging.getLogger("run").debug("a line only DEBUG lets through")

    assert "only DEBUG" in (tmp_path / "worker1.log").read_text(encoding="utf-8")


def test_a_worker_with_nowhere_to_write_keeps_the_logging_it_started_with(tmp_path):
    # A driver may run without a log folder, such as a test or a caller doing its
    # own logging, and a worker then has no file to open. It leaves the process's
    # logging alone rather than replacing it with nothing.
    context = SharedContext("run", _Stages(1, tmp_path), (Device("cpu"),), None)
    before = logging.getLogger().handlers[:]

    _init_worker(0, context)

    assert logging.getLogger().handlers == before
    assert not list(tmp_path.glob("worker*.log"))


@pytest.fixture()
def restored_root_logger():
    """Put the root logger back, since configuring a worker replaces it."""
    logger = logging.getLogger()
    handlers, level = logger.handlers[:], logger.level

    yield logger

    logger.handlers.clear()
    logger.handlers.extend(handlers)
    logger.setLevel(level)


@pytest.mark.usefixtures("restored_root_logger")
def test_the_sequences_one_worker_takes_share_its_file(tmp_path):
    # `worker_init` runs once per worker process, not once per task, so a second
    # sequence on the same worker writes through the handler the first opened.
    WorkerLogFolder(tmp_path).configure_worker(0, 2, logging.INFO)
    for line in ("TL_00", "TL_02"):
        logging.getLogger("preprocess").info(line)

    written = (tmp_path / "worker0.log").read_text(encoding="utf-8")
    assert "TL_00" in written
    assert "TL_02" in written


@pytest.mark.usefixtures("restored_root_logger")
def test_a_worker_that_restarts_keeps_what_it_already_wrote(tmp_path):
    # `tasks_per_worker` retires a worker and starts a fresh one under the same
    # id, so truncating would take everything the retired one wrote with it.
    for line in ("before", "after"):
        WorkerLogFolder(tmp_path).configure_worker(0, 2, logging.INFO)
        logging.getLogger("preprocess").info(line)

    written = (tmp_path / "worker0.log").read_text(encoding="utf-8")
    assert "before" in written
    assert "after" in written


@pytest.mark.usefixtures("restored_root_logger")
def test_two_stages_of_one_job_land_in_the_same_file(tmp_path):
    # Which stage wrote a line is the logger's name to say; splitting the files
    # by stage would only make a job that runs both read out of order.
    for name in ("preprocess", "optical_flow"):
        WorkerLogFolder(tmp_path).configure_worker(0, 2, logging.INFO)
        logging.getLogger(name).info("%s here", name)

    written = (tmp_path / "worker0.log").read_text(encoding="utf-8")
    assert [path.name for path in tmp_path.glob("*.log")] == ["worker0.log"]
    assert "[preprocess]" in written
    assert "[optical_flow]" in written


def test_clearing_drops_what_an_earlier_job_left(tmp_path):
    stale = tmp_path / "worker9.log"
    stale.write_text("from a job that is over", encoding="utf-8")

    WorkerLogFolder(tmp_path).clear()

    assert not stale.exists()


@pytest.mark.parametrize(("workers", "expected"), ((0, 1), (2, 0)))
def test_insights_nobody_will_collect_are_said_out_loud(
    tmp_path, caplog, workers, expected
):
    # `mpire` gathers them, so a run with no pool gathers nothing, and asking
    # for a measurement that never arrives should not look like a quiet result.
    config = ComputeConfig(
        device="cpu", workers=workers, show_progress=False, measure_workers=True
    )

    with caplog.at_level(logging.WARNING):
        run_all(_Stages(2, tmp_path / "done"), config)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == expected
    assert all("no pool" in r.getMessage() for r in warnings)


@pytest.mark.parametrize(
    ("workers", "expected"),
    (
        (2, "worker0.log"),
        (10, "worker0.log"),
        (11, "worker00.log"),
        (128, "worker000.log"),
    ),
)
def test_a_worker_file_is_padded_to_the_pool_it_belongs_to(tmp_path, workers, expected):
    # A fixed width misaligns the moment a pool outgrows it, and 64 cores is a
    # size this project has already run at. The width follows the highest id
    # rather than the count, so ten workers, ids 0 to 9, still take one.
    assert WorkerLogFolder(tmp_path).path_for(0, workers).name == expected


@pytest.mark.parametrize(
    ("config", "error", "refused"),
    (
        ({"device": "cpu", "workers": [0, 1]}, TypeError, r"on cpu is a count"),
        ({"device": "cuda", "workers": 2}, TypeError, r"on cuda names gpu ids"),
        ({"device": "cpu", "workers": -1}, ValueError, r"invalid worker count -1"),
    ),
)
def test_a_worker_setting_the_device_cannot_read_is_refused(config, error, refused):
    with pytest.raises(error, match=refused):
        plan_devices(ComputeConfig(**config))


@pytest.mark.parametrize(("workers", "count"), ((0, 1), (1, 1), (3, 3)))
def test_a_cpu_run_plans_one_device_per_worker_and_never_none(workers, count):
    planned = plan_devices(ComputeConfig(device="cpu", workers=workers))

    assert len(planned) == count
    assert all(not device.is_cuda for device in planned)


def test_a_failure_is_named_by_the_index_it_carries_not_by_when_it_returned(
    tmp_path, caplog
):
    stages = _Stages(3, tmp_path)
    outcomes = [
        Outcome(2, "boom"),
        Outcome(0, None, computed=True),
        Outcome(1, None, computed=False),
    ]
    record = RunRecord()

    with caplog.at_level(logging.INFO):
        _collect_outcomes(iter(outcomes), stages, logging.getLogger("run"), record)

    assert record.failed == {"item2": "boom"}
    assert record.unchanged == {"item1"}
    assert record.returned == {"item0", "item1", "item2"}
    assert record.ready == 2
    assert [record.getMessage() for record in caplog.records] == [
        "item2 failed (1/3)",
        "item0 computed (2/3)",
        "item1 unchanged (3/3)",
    ]


def test_what_came_back_before_the_pool_died_is_kept(tmp_path, caplog):
    # The collections belong to the caller, so a pool that dies part way leaves
    # what it already said behind. Owning them here lost the grounds for a
    # retry exactly when a run most needs them.
    def outcomes():
        yield Outcome(0, "boom")
        yield Outcome(1, None, computed=False)
        msg = "Worker-1 died unexpectedly"
        raise RuntimeError(msg)

    stages = _Stages(3, tmp_path)
    record = RunRecord()

    with pytest.raises(RuntimeError, match="died unexpectedly"):
        _collect_outcomes(outcomes(), stages, logging.getLogger("run"), record)

    assert record.failed == {"item0": "boom"}
    assert record.unchanged == {"item1"}
    assert record.returned == {"item0", "item1"}  # item2 never came back


def test_items_the_pool_took_down_with_it_are_not_counted_ready(
    tmp_path, caplog, monkeypatch
):
    # `ready` was every item the run had no failure for, which counts the ones
    # that never ran: a pool dying after two of four said "3 of 4 ready". The
    # difference between those two numbers is what a retry is built from.
    # Patched at the worker, since what tears down is the pool rather than an
    # item: an item that raises comes back as a failure and is counted.
    outcomes = iter([Outcome(0, "boom"), Outcome(1, None, computed=True)])

    def vanish(worker_id, context, index):
        try:
            return next(outcomes)
        except StopIteration:
            msg = "Worker-1 died unexpectedly"
            raise RuntimeError(msg) from None

    monkeypatch.setattr(compute, "_run_on_worker", vanish)

    with (
        caplog.at_level(logging.INFO),
        pytest.raises(IncompleteRunError, match=r"1 of 4 failed"),
    ):
        run_all(_Stages(4, tmp_path / "done"), _compute(0))

    said = " ".join(caplog.messages)
    assert "1 of 4 ready" in said
    assert "2 never came back" in said


def test_an_unset_worker_count_falls_back_to_the_machine(monkeypatch):
    # Comparing against `_DEFAULT_WORKERS` could not fail, whatever that constant
    # came to mean. What the fallback owes is the machine's own answer, so the
    # machine is what it is checked against, and moving that answer has to
    # move the plan with it.
    monkeypatch.setattr(compute, "_DEFAULT_WORKERS", 3)

    assert len(plan_devices(ComputeConfig(device="cpu"))) == 3
    assert len(plan_devices(ComputeConfig(device="cpu", workers=None))) == 3
    assert len(plan_devices(ComputeConfig(device="cpu", workers=5))) == 5


def test_the_default_worker_count_follows_this_process_s_own_affinity():
    # `os.cpu_count()` answers for the machine, not for what this process may
    # use, so a run under `taskset` or a scheduler's allocation would start a
    # worker per core of the host and have them contend over its own few.
    # The two agree on windows whatever the affinity mask, so only a linux run
    # tells the constants apart, which is where the pool is actually sized.
    assert os.process_cpu_count() == _DEFAULT_WORKERS


# ------------------------------ the progress bar -------------------------- #


def _redirected(monkeypatch, *, tty: bool) -> None:
    monkeypatch.setattr(sys.stderr, "isatty", lambda: tty, raising=False)


def test_a_redirected_run_draws_no_progress_bar(tmp_path, monkeypatch, caplog):
    # `tqdm` renders with a carriage return, so a redirected pool writes one
    # long line of fragments into the log it was pointed at, at the timer's
    # own rate, not the run's.
    _redirected(monkeypatch, tty=False)
    drawn = []
    monkeypatch.setattr(
        compute, "trange", lambda *a, **kw: drawn.append(kw) or range(*a)
    )

    config = replace(_compute(0), show_progress=True)
    with caplog.at_level(logging.WARNING):
        run_all(_Stages(4, tmp_path / "done"), config)

    assert drawn
    assert drawn[0]["disable"] is True
    assert "stderr is not a terminal" in caplog.text


def test_a_watched_run_still_draws_one(tmp_path, monkeypatch, caplog):
    # The other half: the check must not disable the bar everywhere, which is
    # what an assertion on the redirected case alone would let through.
    _redirected(monkeypatch, tty=True)
    drawn = []
    monkeypatch.setattr(
        compute, "trange", lambda *a, **kw: drawn.append(kw) or range(*a)
    )

    config = replace(_compute(0), show_progress=True)
    with caplog.at_level(logging.WARNING):
        run_all(_Stages(4, tmp_path / "done"), config)

    assert drawn
    assert drawn[0]["disable"] is False
    assert "stderr is not a terminal" not in caplog.text


def test_a_run_told_not_to_draw_says_nothing_about_the_terminal(
    tmp_path, monkeypatch, caplog
):
    # The warning is for a run that asked and cannot have it. One that never
    # asked would only be told something it did not want to know.
    _redirected(monkeypatch, tty=False)

    with caplog.at_level(logging.WARNING):
        run_all(_Stages(4, tmp_path / "done"), _compute(0))

    assert "stderr is not a terminal" not in caplog.text


def test_a_drawn_bar_takes_console_logging_over(restored_root_logger):
    # `tqdm` draws with a carriage return, so a handler writing straight to the
    # stream lands on the bar's own line. The bar is left behind as a frozen
    # copy and a fresh one appears below, once per line the run logs.
    handler = logging.StreamHandler()
    restored_root_logger.addHandler(handler)

    with _drawing(progress=True):
        during = list(restored_root_logger.handlers)

    assert handler not in during
    assert handler in restored_root_logger.handlers


def test_a_run_with_no_bar_leaves_console_logging_alone(restored_root_logger):
    # Routing a run that draws nothing would move its output through `tqdm` for
    # no reason, and a redirected run is exactly the one whose log is read.
    handler = logging.StreamHandler()
    restored_root_logger.addHandler(handler)

    with _drawing(progress=False):
        assert handler in restored_root_logger.handlers

    assert handler in restored_root_logger.handlers


@pytest.mark.parametrize("tty", (True, False))
def test_the_bar_and_the_console_agree_on_whether_one_is_drawn(
    tmp_path, monkeypatch, tty
):
    # The two answers come from one decision, so a run cannot end up routing
    # its console around a bar it never drew, or drawing one it then tears.
    _redirected(monkeypatch, tty=tty)
    drawn, routed = [], []
    monkeypatch.setattr(
        compute, "trange", lambda *a, **kw: drawn.append(kw) or range(*a)
    )
    monkeypatch.setattr(
        compute,
        "_drawing",
        lambda *, progress: routed.append(progress) or nullcontext(),
    )

    config = replace(_compute(0), show_progress=True)
    run_all(_Stages(4, tmp_path / "done"), config)

    assert routed == [tty]
    assert drawn[0]["disable"] is not tty


# ---------------------------- the configuration log ----------------------- #


def _logged(caplog, **config):
    with caplog.at_level(logging.INFO):
        log_compute_config(ComputeConfig(**config), logging.getLogger("run"))

    return [record.getMessage() for record in caplog.records]


@pytest.mark.parametrize(
    ("config", "head"),
    (
        ({}, "compute: cpu"),
        ({"workers": 4}, "compute: cpu"),
        ({"device": "cuda"}, "compute: cuda"),
        ({"device": "cuda", "workers": [0, 2]}, "compute: cuda"),
    ),
)
def test_the_head_says_the_device_and_leaves_the_count_to_the_run(caplog, config, head):
    # `run_all` names the workers it resolved, and a count guessed from the
    # configuration would only say the same thing less reliably.
    assert _logged(caplog, **config)[0] == head


def test_a_default_run_says_nothing_beyond_its_head(caplog):
    # `run_all` reports the plan it resolved, so a configuration that moved
    # nothing else has nothing to add.
    assert _logged(caplog) == ["compute: cpu"]


@pytest.mark.parametrize(
    ("config", "said"),
    (
        ({"tasks_per_worker": 8}, "  replacing a worker after 8 tasks"),
        ({"measure_workers": True}, "  reporting how busy each worker was"),
        ({"show_progress": False}, "  showing no progress bar"),
    ),
)
def test_a_setting_a_run_moved_gets_a_line(caplog, config, said):
    assert said in _logged(caplog, **config)


@pytest.mark.parametrize(
    "config", ({"tasks_per_worker": None}, {"measure_workers": False})
)
def test_a_setting_left_alone_gets_none(caplog, config):
    # Silence is the default, which `run_all` then answers for itself.
    assert _logged(caplog, **config) == ["compute: cpu"]


def test_a_device_that_cannot_be_resolved_is_still_logged(caplog):
    # Refusing it here would cost the block that shows what was refused, so the
    # spec is taken verbatim and `plan_devices` is left to turn it down.
    assert _logged(caplog, device="tpu") == ["compute: tpu"]

    with pytest.raises(ValueError, match=r"invalid device spec"):
        plan_devices(ComputeConfig(device="tpu"))


# ------------------------------- the insights ----------------------------- #


_INSIGHTS = {
    "total_time": "0:00:10",
    "working_ratio": 0.5,
    "waiting_ratio": 0.2,
    "n_completed_tasks": [3, 1],
    "working_time": ["0:00:03.5", "0:00:01.5"],
}


def test_the_shares_reported_account_for_the_whole(caplog):
    # `mpire` splits a worker's life five ways and the ratios are of their sum,
    # so naming two of them leaves a reader asking where the rest went: 98% of
    # it, on a run short enough for process start-up to dominate. The other three
    # are start-up, `worker_init` and `worker_exit`, which no single name covers.
    with caplog.at_level(logging.INFO):
        log_insights(_INSIGHTS, "run", unit="seq")

    (summary, *_) = [record.getMessage() for record in caplog.records]

    assert summary == (
        "workers spent 0:00:10: 50.0% working, 20.0% waiting, 30.0% overhead"
    )


def test_each_worker_says_what_it_finished_and_in_what(caplog):
    with caplog.at_level(logging.INFO):
        log_insights(_INSIGHTS, "run", unit="seq")

    logged = [record.getMessage() for record in caplog.records]

    assert logged[1:] == [
        "  worker 0 completed 3 seq in 0:00:03.5",
        "  worker 1 completed 1 seq in 0:00:01.5",
    ]


def test_a_pool_that_collected_nothing_says_so_rather_than_raising(caplog):
    # `get_insights()` answers `{}` when a pool was told to collect none, and
    # reading its keys would be a `KeyError` at the end of a finished run.
    with caplog.at_level(logging.INFO):
        log_insights({}, "run")

    assert [record.getMessage() for record in caplog.records] == [
        "nothing to report: the pool collected no insights"
    ]


def test_the_report_is_filed_under_the_name_it_is_given(caplog):
    # No default: `run_all` always knows the stage's name, and a default would
    # only ever apply where a caller forgot, filing the run's lines elsewhere.
    with caplog.at_level(logging.INFO):
        log_insights(_INSIGHTS, "reconstruct")

    assert {record.name for record in caplog.records} == {"reconstruct"}
