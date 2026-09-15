from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Self, override

from iivs_cardio.common.device import Device
from iivs_cardio.common.pipeline import Stage, StageRun

if TYPE_CHECKING:
    from iivs_cardio.common.pipeline import Step


class _Numbers(Stage[int]):
    """A source stage over the first `count` integers."""

    def __init__(self, count: int) -> None:
        super().__init__()
        self._count = count

    def __len__(self) -> int:
        return self._count

    @override
    def _compute(self, index: int) -> int:
        return index


class _Doubled(Stage[int]):
    """A stage built over another, so a run has a chain to walk."""

    def __init__(self, source: Stage[int]) -> None:
        super().__init__(source)
        self._source = source

    def __len__(self) -> int:
        return len(self._source)

    @override
    def _compute(self, index: int) -> int:
        return self._source[index].require() * 2


class _Reporting:
    """A hook that has one line to say once its stage is done."""

    def __init__(self, line: str) -> None:
        self._line = line

    def __call__(self, step: Step[int, None]) -> None:
        return None

    def report(self) -> str:
        return self._line


class _Item:
    name = "TL_00"

    def release(self) -> None:
        return None


class _Run(StageRun[_Item]):
    """A run handing out one prebuilt stage."""

    def __init__(self, stage: Stage[int]) -> None:
        super().__init__([_Item()], name="demo")
        self._stage = stage

    @override
    def get_stage(self, index: int, device: Device) -> Stage[int]:
        return self._stage

    @override
    def _work_label(self, index: int) -> str:
        return "doubling"


def _said(caplog) -> list[str]:
    return [record.getMessage().strip() for record in caplog.records]


def test_a_hook_on_a_stage_further_down_reports_too(caplog):
    # The walk opens the whole chain's hooks and commits every one of them, so
    # asking only the top stage's left one that committed with nothing said.
    source = _Numbers(2).register_hooks(_Reporting("below"))
    stage = _Doubled(source).register_hooks(_Reporting("above"))

    with caplog.at_level(logging.INFO):
        assert _Run(stage).run_stage(0, Device("cpu"))

    said = _said(caplog)
    assert "above" in said
    assert "below" in said


def test_a_hook_on_two_stages_reports_once(caplog):
    # It is one hook that committed once, so it is one line.
    shared = _Reporting("shared")
    source = _Numbers(2).register_hooks(shared)
    stage = _Doubled(source).register_hooks(shared)

    with caplog.at_level(logging.INFO):
        _Run(stage).run_stage(0, Device("cpu"))

    assert _said(caplog).count("shared") == 1


class _Traced:
    """A hook that notes when it is opened, handed a step, closed, and asked."""

    def __init__(self, where: str, events: list[str]) -> None:
        self._where = where
        self._events = events

    def __enter__(self) -> Self:
        self._events.append(f"open {self._where}")
        return self

    def __call__(self, step: Step[int, None]) -> None:
        self._events.append(f"call {self._where} {step.index}")

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._events.append(f"close {self._where}")

    def report(self) -> str:
        self._events.append(f"report {self._where}")
        return self._where


def test_a_walk_opens_down_the_chain_and_does_the_rest_up_it():
    # A value is computed before the one built from it, so a stage further down
    # sees each step first and closes first, and its account is read first too.
    # Only opening goes the other way, from the stage the walk asks.
    events: list[str] = []
    source = _Numbers(2).register_hooks(_Traced("below", events))
    stage = _Doubled(source).register_hooks(_Traced("above", events))

    _Run(stage).run_stage(0, Device("cpu"))

    assert events == [
        "open above",
        "open below",
        "call below 0",
        "call above 0",
        "call below 1",
        "call above 1",
        "close below",
        "close above",
        "report below",
        "report above",
    ]


def test_hooks_of_one_stage_report_in_the_order_they_were_registered(caplog):
    # Up and down are between stages. Within one the order stays the one the
    # hooks were registered in, which is the order a run built its branches in.
    stage = _Numbers(1).register_hooks(_Reporting("first"), _Reporting("second"))

    with caplog.at_level(logging.INFO):
        _Run(stage).run_stage(0, Device("cpu"))

    said = _said(caplog)
    assert said.index("first") < said.index("second")
