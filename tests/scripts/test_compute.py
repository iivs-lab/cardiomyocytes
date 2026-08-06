from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest

from scripts._compute import (
    ComputeConfig,
    IncompleteRunError,
    WorkerLogFolder,
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

    with pytest.raises(IncompleteRunError, match=r"1 of 4 items failed"):
        run_all(_Stages(4, dest, explode_at=[1]), _compute(workers))

    assert _done(dest) == [0, 2, 3]


@pytest.mark.parametrize("workers", (0, 2))
def test_what_failed_stays_whole_rather_than_folded_into_the_message(tmp_path, workers):
    # A caller acts on this -- a report saying it covers a subset, a retry -- and
    # a dataset's worth of failures would not fit one line anyway.
    dest = tmp_path / "done"

    with pytest.raises(IncompleteRunError) as failure:
        run_all(_Stages(4, dest, explode_at=[0, 3]), _compute(workers))

    assert str(failure.value) == "2 of 4 items failed"
    assert failure.value.total == 4
    assert failure.value.failed == (
        (0, "ValueError: item 0 gave up"),
        (3, "ValueError: item 3 gave up"),
    )


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

    with pytest.raises(IncompleteRunError, match=r"1 of 3 items failed"):
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
    # `worker_lifespan` retires a worker and starts a fresh one under the same
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
        device="cpu", workers=workers, progress_bar=False, insights=True
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
