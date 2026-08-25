from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from iivs_cardio.common.pipeline import SequenceStage, Step
from iivs_cardio.optical_flow.estimators import FarnebackConfig
from iivs_cardio.optical_flow.pipeline import EvaluationDocument
from tests.optical_flow.pipeline.helpers import frame_stage

if TYPE_CHECKING:
    from pathlib import Path

SETTINGS = {"estimator": "farneback", "filter": "identity"}


@dataclass
class _Source:
    """A sequence as the document meets it: a name and the frames it was read
    over."""

    name: str
    frames: SequenceStage


def _names(count: int) -> tuple[str, ...]:
    return tuple(f"{index:05d}_phase.bin" for index in range(count))


def _contents(*counts: int) -> dict[str, tuple[str, ...]]:
    return {f"plate/TL_{i:02d}": _names(count) for i, count in enumerate(counts)}


def _score(document: EvaluationDocument, source: _Source) -> bool:
    """Score one sequence the way a flow stage would, or say it was reused."""
    meter = document.get_hook(source)
    if meter is None:
        return False

    of = FarnebackConfig().build("cpu")
    frames = source.frames

    with meter:
        for index in range(len(frames) - 1):
            flow = of.calc(frames[index].require(), frames[index + 1].require())
            meter(Step(index, flow, frames[index].extra))

    return True


def _run(path: Path, counts: dict[str, tuple[str, ...]], **kwargs) -> list[str]:
    """Run a whole dataset through one document, naming what it measured."""
    measured = []

    with EvaluationDocument(path, "nexel", counts, settings=SETTINGS, **kwargs) as doc:
        for name, frames in counts.items():
            if _score(doc, _Source(name, frame_stage(len(frames)))):
                measured.append(name)

    return measured


def test_a_run_folds_every_sequence_into_one_document(tmp_path):
    contents = _contents(5, 4)

    _run(tmp_path / "evaluation", contents)

    written = json.loads((tmp_path / "evaluation.json").read_text("utf-8"))

    assert written["settings"] == SETTINGS
    assert written["coverage"]["covered"] == 2
    assert written["dataset"]["source"] == "nexel"
    assert written["dataset"]["pairs"] == 7
    assert len(written["dataset"]["sequences"]) == 2


def test_a_part_covers_a_pair_short_of_the_frames_it_was_read_over(tmp_path):
    # The whole point of `_expected` here: a part names one source fewer than
    # the sequence holds, and a document expecting every frame would call every
    # part stale and measure the dataset again on each run.
    contents = _contents(5)

    _run(tmp_path / "evaluation", contents, if_present="overwrite")
    again = _run(tmp_path / "evaluation", contents, if_present="reuse")

    assert again == []

    written = json.loads((tmp_path / "evaluation.json").read_text("utf-8"))

    assert written["coverage"]["reused"] == 1
    assert written["coverage"]["covered"] == 1
    assert len(written["dataset"]["sequences"][0]["frames"]) == 4


def test_a_part_left_under_other_settings_is_measured_again(tmp_path):
    contents = _contents(4)
    path = tmp_path / "evaluation"

    _run(path, contents, if_present="overwrite")

    with EvaluationDocument(
        path, "nexel", contents, settings={"filter": "median"}, if_present="reuse"
    ) as doc:
        stale = doc.get_hook(_Source("plate/TL_00", frame_stage(4)))

    assert stale is not None


def test_a_sequence_that_grew_since_its_part_was_written_is_measured_again(tmp_path):
    # The other half of `_still_describes`: the settings held but the source
    # did not, and the part now covers a prefix of the sequence rather than it.
    path = tmp_path / "evaluation"

    _run(path, _contents(4), if_present="overwrite")

    with EvaluationDocument(
        path, "nexel", _contents(6), settings=SETTINGS, if_present="reuse"
    ) as doc:
        stale = doc.get_hook(_Source("plate/TL_00", frame_stage(6)))

    assert stale is not None


def test_the_estimator_reaches_the_meters_the_document_hands_out(tmp_path):
    # Without one the forward-backward axis is not measured at all, so a
    # document that dropped it on the way through would read as absent rather
    # than as an error.
    of = FarnebackConfig().build("cpu")
    contents = _contents(4)

    _run(tmp_path / "with", contents, estimator=of)
    _run(tmp_path / "without", contents)

    scored = json.loads((tmp_path / "with.json").read_text("utf-8"))
    absent = json.loads((tmp_path / "without.json").read_text("utf-8"))

    assert scored["dataset"]["metrics"]["fb_error"]["scored"] == 3
    assert absent["dataset"]["metrics"]["fb_error"]["scored"] == 0


def test_a_sequence_nothing_measured_is_named_rather_than_folded(tmp_path):
    contents = _contents(4, 4)
    path = tmp_path / "evaluation"

    with EvaluationDocument(path, "nexel", contents, settings=SETTINGS) as doc:
        _score(doc, _Source("plate/TL_00", frame_stage(4)))

    written = json.loads(path.with_suffix(".json").read_text("utf-8"))

    assert written["coverage"]["covered"] == 1
    assert written["coverage"]["skipped"] == ["plate/TL_01"]


def test_the_report_names_the_axis_the_document_exists_for(tmp_path):
    document = EvaluationDocument(
        tmp_path / "evaluation", "nexel", _contents(4), settings=SETTINGS
    )

    with document as doc:
        _score(doc, _Source("plate/TL_00", frame_stage(4)))

    said = document.report()

    assert said is not None
    assert said.startswith("wrote evaluation.json from 1 sequence: SSIM ")


def test_nothing_written_reports_nothing(tmp_path):
    document = EvaluationDocument(tmp_path / "evaluation", "nexel", _contents(4))

    assert document.report() is None


def test_a_document_over_no_sequence_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no sequence to cover"):
        EvaluationDocument(tmp_path / "evaluation", "nexel", {})
