from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest

from scripts._compute import (
    ComputeConfig,
    IncompleteRunError,
    WorkerLogFolder,
    log_compute_config,
    log_insights,
    plan_devices,
    run_all,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

    from iivs_cardio.common.device import Device


class _Stages:
    """A stage factory that records what ran, through the filesystem.

    Through files rather than an attribute, because the pool hands each worker
    its own copy: what a worker did to itself never comes home.
    """

    def __init__(self, count: int, dest: Path, explode_at: Iterable[int] = ()) -> None:
        self._count = count
        self._dest = dest
        self._explode_at = frozenset(explode_at)

    @property
    def name(self) -> str:
        return "run"

    def __len__(self) -> int:
        return self._count

    def get_name(self, index: int) -> str:
        return f"item{index}"

    def run_stage(self, index: int, device: Device) -> None:
        if index in self._explode_at:
            msg = f"item {index} gave up"
            raise ValueError(msg)

        (self._dest / f"{index:03d}.done").write_text("", encoding="utf-8")

    @contextmanager
    def running(self) -> Iterator[_Stages]:
        self._dest.mkdir(parents=True, exist_ok=True)
        yield self


def _done(dest: Path) -> list[int]:
    return sorted(int(path.stem) for path in dest.glob("*.done"))


def _compute(workers: int) -> ComputeConfig:
    return ComputeConfig(device="cpu", workers=workers, progress_bar=False)


@pytest.mark.parametrize("workers", (0, 2))
def test_a_clean_run_finishes_quietly(tmp_path, workers):
    dest = tmp_path / "done"

    run_all(_Stages(4, dest), _compute(workers))

    assert _done(dest) == [0, 1, 2, 3]


@pytest.mark.parametrize("workers", (0, 2))
def test_one_item_failing_does_not_take_the_rest_with_it(tmp_path, workers):
    # `mpire` re-raises a task's exception in the parent and tears the pool down,
    # so a run that let one through would lose every item still to come -- hours
    # of finished sequences, at dataset scale.
    dest = tmp_path / "done"

    with pytest.raises(IncompleteRunError, match=r"1 of 4 failed"):
        run_all(_Stages(4, dest, explode_at=[1]), _compute(workers))

    assert _done(dest) == [0, 2, 3]


@pytest.mark.parametrize("workers", (0, 2))
def test_what_failed_stays_whole_rather_than_folded_into_the_message(tmp_path, workers):
    # A caller acts on this -- a report saying it covers a subset, a retry -- and
    # a dataset's worth of failures would not fit one line anyway.
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
    # `lifespan` retires a worker and starts a fresh one under the same
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
    # `mpire` gathers them, so a run with no pool gathers nothing -- and asking
    # for a measurement that never arrives should not look like a quiet result.
    config = ComputeConfig(
        device="cpu", workers=workers, progress_bar=False, log_insights=True
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
    # rather than the count, so ten workers -- ids 0 to 9 -- still take one.
    assert WorkerLogFolder(tmp_path).path_for(0, workers).name == expected


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
        ({"device": "cuda", "gpu_ids": [0, 2]}, "compute: cuda"),
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
        ({"lifespan": 8}, "  replacing a worker after 8 tasks"),
        ({"log_insights": True}, "  reporting how busy each worker was"),
        ({"progress_bar": False}, "  showing no progress bar"),
    ),
)
def test_a_setting_a_run_moved_gets_a_line(caplog, config, said):
    assert said in _logged(caplog, **config)


@pytest.mark.parametrize("config", ({"lifespan": None}, {"log_insights": False}))
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
    # so naming two of them leaves a reader asking where the rest went -- 98% of
    # it, on a run short enough for process start-up to dominate.
    with caplog.at_level(logging.INFO):
        log_insights(_INSIGHTS, "run", unit="seq")

    (summary, *_) = [record.getMessage() for record in caplog.records]

    assert summary == (
        "workers spent 0:00:10: 50.0% working, 20.0% waiting, 30.0% starting/stopping"
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
    # only ever apply where a caller forgot -- filing the run's lines elsewhere.
    with caplog.at_level(logging.INFO):
        log_insights(_INSIGHTS, "reconstruct")

    assert {record.name for record in caplog.records} == {"reconstruct"}
