from __future__ import annotations

import json
from pathlib import PurePath
from typing import TYPE_CHECKING

import pytest
import torch

from iivs_cardio.common.pipeline import SequenceStage, Step
from iivs_cardio.common.warp import backward_warp
from iivs_cardio.optical_flow.estimators import FarnebackConfig
from iivs_cardio.optical_flow.pipeline import SequenceEvaluation, SequenceEvaluator

if TYPE_CHECKING:
    from pathlib import Path

SIZE = 96
NAME = "plate/TL_00"


class _Frames:
    """A `DataSequence` of frames, which is all a stage asks of a source."""

    def __init__(self, frames: list[torch.Tensor]) -> None:
        self._frames = frames

    def __len__(self) -> int:
        return len(self._frames)

    def get_item(self, index: int) -> torch.Tensor:
        return self._frames[index]

    def get_meta(self, index: int):
        return PurePath(f"{index:05d}_phase.bin")


def _textured() -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(SIZE).float(), torch.arange(SIZE).float(), indexing="ij"
    )
    weave = (
        128
        + 60 * torch.sin(2 * torch.pi * x / 17)
        + 50 * torch.sin(2 * torch.pi * y / 23)
    )
    return weave.clamp(0, 255).to(torch.uint8)


def _shifted(frame: torch.Tensor, dx: float) -> torch.Tensor:
    offset = torch.zeros((2, SIZE, SIZE))
    offset[0] = -dx
    offset[1] = -dx * 0.6
    return backward_warp(frame, offset)


def _stage(count: int, *, duplicate: int | None = None) -> SequenceStage:
    base = _textured()
    frames = [_shifted(base, i * 0.8) for i in range(count)]
    if duplicate is not None:
        frames[duplicate] = frames[duplicate - 1].clone()

    return SequenceStage(_Frames(frames), window=2)


def _run(tmp_path: Path, frames: SequenceStage, *, estimator=None, **kwargs):
    """Score a whole sequence, as a flow stage's hook would."""
    of = FarnebackConfig().build("cpu")
    evaluator = SequenceEvaluator(tmp_path, NAME, frames, estimator=estimator, **kwargs)

    with evaluator:
        for index in range(len(frames) - 1):
            first = frames[index].require()
            second = frames[index + 1].require()
            flow = of.calc(first, second)
            evaluator(Step(index, flow, frames[index].extra))

    return evaluator


def test_a_sequence_of_n_frames_is_scored_n_minus_one_times(tmp_path):
    folded = _run(tmp_path, _stage(5)).to_evaluation()

    assert folded.source == NAME
    assert folded.pairs == 4
    assert folded.metrics["ssim"].scored == 4


def test_a_score_names_the_frame_its_flow_starts_from(tmp_path):
    # Start labelling, so the frame with nothing to pair with is the last.
    _run(tmp_path, _stage(4))

    written = json.loads((tmp_path / f"{NAME}.json").read_text("utf-8"))
    names = [frame["source"] for frame in written["frames"]]

    assert names == ["00000_phase.bin", "00001_phase.bin", "00002_phase.bin"]


def test_a_part_carries_the_pairs_it_was_folded_from(tmp_path):
    # What lets a chunked run be folded again from its parts, and what a reader
    # goes to when a mean is not the whole story.
    _run(tmp_path, _stage(4))

    written = json.loads((tmp_path / f"{NAME}.json").read_text("utf-8"))
    read = SequenceEvaluation.from_dict(written)

    assert len(read) == 3
    assert read.metrics["ssim"].mean == pytest.approx(
        written["metrics"]["ssim"]["mean"]
    )


def test_a_real_flow_gains_on_the_floor_it_is_read_against(tmp_path):
    folded = _run(tmp_path, _stage(4)).to_evaluation()

    gain = folded.metrics["ssim"].mean - folded.metrics["ssim_floor"].mean
    assert gain > 0.0
    assert folded.metrics["magnitude"].mean > 0.0


def test_without_an_estimator_the_reverse_axis_is_not_measured(tmp_path):
    folded = _run(tmp_path, _stage(4)).to_evaluation()

    assert folded.metrics["fb_error"].scored == 0
    assert folded.dropped("fb_error") == 3


def test_with_an_estimator_the_flow_is_asked_to_come_back(tmp_path):
    of = FarnebackConfig().build("cpu")
    folded = _run(tmp_path, _stage(4), estimator=of).to_evaluation()

    assert folded.metrics["fb_error"].scored == 3
    assert 0.0 < folded.metrics["fb_error"].mean < 1.0


def test_a_duplicated_frame_is_counted_rather_than_rounded_away(tmp_path):
    # Identical frames and a flow of exactly zero reconstruct exactly, so PSNR
    # is infinite. Every other metric still takes that pair, and the difference
    # between `pairs` and `scored` is how many such frames the sequence holds.
    # (DualTVL1 and DeepFlow answer identical frames with an exactly zero flow;
    # Farneback answers 4.6e-02, so which estimator ran decides whether the
    # sequence reaches this at all.)
    frames = _stage(4, duplicate=2)
    evaluator = SequenceEvaluator(tmp_path, NAME, frames)
    of = FarnebackConfig().build("cpu")

    with evaluator:
        for index in range(len(frames) - 1):
            first = frames[index].require()
            second = frames[index + 1].require()
            exact = torch.equal(first, second)
            flow = torch.zeros((2, SIZE, SIZE)) if exact else of.calc(first, second)
            evaluator(Step(index, flow, frames[index].extra))

    folded = evaluator.to_evaluation()

    assert folded.pairs == 3
    assert folded.dropped("psnr") == 1
    assert folded.dropped("ssim") == 0
    assert folded.dropped("mse") == 0


def test_the_part_is_written_on_a_clean_close(tmp_path):
    _run(tmp_path, _stage(4), settings={"estimator": "farneback"})

    written = json.loads((tmp_path / f"{NAME}.json").read_text("utf-8"))

    assert written["settings"] == {"estimator": "farneback"}
    assert written["source"] == NAME
    assert written["pairs"] == 3


def test_a_sequence_that_ended_badly_leaves_no_part(tmp_path):
    # A part on disk stands for a sequence that finished.
    frames = _stage(4)
    evaluator = SequenceEvaluator(tmp_path, NAME, frames)

    died = RuntimeError("the worker died")
    with pytest.raises(RuntimeError), evaluator:
        raise died

    assert not (tmp_path / f"{NAME}.json").exists()


def test_a_part_goes_back_when_the_sequence_it_stood_for_did_not(tmp_path):
    evaluator = _run(tmp_path, _stage(4))
    assert (tmp_path / f"{NAME}.json").exists()

    evaluator.revert()

    assert not (tmp_path / f"{NAME}.json").exists()


def test_a_frame_stage_that_is_not_the_one_it_was_computed_from_is_caught(tmp_path):
    # Asking for `i + 1` past the end says the stage is a different length, so
    # it is not the stage these flows came from.
    frames = _stage(3)
    evaluator = SequenceEvaluator(tmp_path, NAME, frames)
    flow = torch.zeros((2, SIZE, SIZE))

    with pytest.raises(IndexError):
        evaluator(Step(2, flow, PurePath("00002_phase.bin")))


def test_nothing_scored_reports_nothing(tmp_path):
    assert SequenceEvaluator(tmp_path, NAME, _stage(3)).report() is None


def test_a_sequence_that_answered_no_pair_is_refused_rather_than_filed(tmp_path):
    # A part standing for it would count as covered while saying nothing.
    evaluator = SequenceEvaluator(tmp_path, NAME, _stage(3))

    with pytest.raises(ValueError, match="answered no pair"):
        evaluator.to_evaluation()


def test_a_report_says_the_gain_and_how_far_the_flow_came_back(tmp_path):
    of = FarnebackConfig().build("cpu")

    said = _run(tmp_path, _stage(4), estimator=of).report()

    assert said is not None
    assert "gaining +" in said
    assert "px apart" in said
