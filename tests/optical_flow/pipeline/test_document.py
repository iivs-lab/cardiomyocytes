from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from iivs_cardio.common.pipeline import Step
from iivs_cardio.optical_flow.estimators import FarnebackConfig
from iivs_cardio.optical_flow.pipeline import EvaluationDocument, FlowSource
from tests.optical_flow.pipeline.helpers import frame_stage

if TYPE_CHECKING:
    from pathlib import Path

    from iivs_cardio.optical_flow.estimators import OpticalFlowEstimator

SETTINGS = {"estimator": "farneback", "filter": "identity"}


def _names(count: int) -> tuple[str, ...]:
    return tuple(f"{index:05d}_phase.bin" for index in range(count))


def _contents(*counts: int) -> dict[str, tuple[str, ...]]:
    return {f"plate/TL_{i:02d}": _names(count) for i, count in enumerate(counts)}


def _score(document: EvaluationDocument, source: FlowSource) -> bool:
    """Score one sequence the way a flow stage would, or say it was reused."""
    writer = document.get_hook(source)
    if writer is None:
        return False

    of = FarnebackConfig().build("cpu")
    frames = source.frames

    with writer:
        for index in range(len(frames) - 1):
            flow = of.calc(frames[index].require(), frames[index + 1].require())
            writer(Step(index, flow, frames[index].extra))

    return True


def _run(
    path: Path,
    counts: dict[str, tuple[str, ...]],
    estimator: OpticalFlowEstimator | None = None,
    **kwargs,
) -> list[str]:
    """Run a whole dataset through one document, naming what it measured."""
    measured = []

    with EvaluationDocument(path, "nexel", counts, settings=SETTINGS, **kwargs) as doc:
        for name, frames in counts.items():
            source = FlowSource(name, frame_stage(len(frames)), estimator)
            if _score(doc, source):
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
    # The whole point of `_expected` here: a result names one source fewer than
    # the sequence holds, and a document expecting every frame would call every
    # result stale and measure the dataset again on each run.
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
        stale = doc.get_hook(FlowSource("plate/TL_00", frame_stage(4)))

    assert stale is not None


def test_a_sequence_that_grew_since_its_part_was_written_is_measured_again(tmp_path):
    # The other half of `_still_describes`: the settings held but the source
    # did not, and the result now covers a prefix of the sequence rather than it.
    path = tmp_path / "evaluation"

    _run(path, _contents(4), if_present="overwrite")

    with EvaluationDocument(
        path, "nexel", _contents(6), settings=SETTINGS, if_present="reuse"
    ) as doc:
        stale = doc.get_hook(FlowSource("plate/TL_00", frame_stage(6)))

    assert stale is not None


def test_the_estimator_a_sequence_carries_reaches_the_meter_made_for_it(tmp_path):
    # It rides on the sequence rather than on the document because it is bound
    # to a device, and a run may work its sequences on more than one. Without
    # one the forward-backward axis is not measured, which reads as absent
    # rather than as an error.
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
        _score(doc, FlowSource("plate/TL_00", frame_stage(4)))

    written = json.loads(path.with_suffix(".json").read_text("utf-8"))

    assert written["coverage"]["covered"] == 1
    assert written["coverage"]["skipped"] == ["plate/TL_01"]


def test_the_report_names_the_axis_the_document_exists_for(tmp_path):
    document = EvaluationDocument(
        tmp_path / "evaluation", "nexel", _contents(4), settings=SETTINGS
    )

    with document as doc:
        _score(doc, FlowSource("plate/TL_00", frame_stage(4)))

    said = document.report()

    assert said is not None
    assert said.startswith("wrote evaluation.json from 1 sequence: SSIM ")


def test_nothing_written_reports_nothing(tmp_path):
    document = EvaluationDocument(tmp_path / "evaluation", "nexel", _contents(4))

    assert document.report() is None


def test_a_document_over_no_sequence_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no sequence to cover"):
        EvaluationDocument(tmp_path / "evaluation", "nexel", {})
