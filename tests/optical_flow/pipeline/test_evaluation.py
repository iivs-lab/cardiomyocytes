from __future__ import annotations

import math

import pytest

from iivs_cardio.optical_flow.pipeline import (
    DatasetEvaluation,
    FrameEvaluation,
    Measured,
    SequenceEvaluation,
    Spread,
)
from iivs_cardio.optical_flow.pipeline.evaluation import METRICS


def _frame(index: int, **scores: float | None) -> FrameEvaluation:
    given: dict[str, float | None] = {
        "ssim": 0.99,
        "ssim_floor": 0.95,
        "psnr": 30.0,
        "mse": 40.0,
        "mae": 3.0,
        "magnitude": 0.4,
        "fb_error": 0.01,
    }
    given.update(scores)

    return FrameEvaluation(source=f"{index:05d}_phase.bin", **given)  # type: ignore[arg-type]


def _sequence(name: str, count: int, **scores: float | None) -> SequenceEvaluation:
    return SequenceEvaluation(name, tuple(_frame(i, **scores) for i in range(count)))


def test_the_gain_is_what_the_score_rose_above_doing_nothing():
    assert _frame(0, ssim=0.99, ssim_floor=0.95).gain == pytest.approx(0.04)


def test_a_score_nobody_measures_is_refused_by_name():
    with pytest.raises(ValueError, match="unsupported metric 'gain'"):
        _frame(0).score("gain")


def test_a_fold_leaves_out_what_did_not_come_back_finite():
    # A duplicated frame reaches `mse` zero exactly and sends `psnr` to
    # infinity, which is a fact about the dataset rather than a defect here.
    frames = [_frame(0), _frame(1, psnr=math.inf), _frame(2)]
    folded = SequenceEvaluation("plate/TL_00", tuple(frames))

    assert folded.pairs == 3
    assert folded.metrics["psnr"].scored == 2
    assert folded.metrics["ssim"].scored == 3
    assert folded.dropped("psnr") == 1
    assert folded.dropped("ssim") == 0


def test_a_metric_that_was_never_measured_scores_none():
    # No estimator, no reverse flow, no forward-backward error. It reads as
    # absent rather than as zero everywhere, which is a different claim.
    folded = _sequence("plate/TL_00", 4, fb_error=None)

    assert folded.metrics["fb_error"] == Measured(0, 0.0)
    assert folded.dropped("fb_error") == 4


def test_a_mean_over_nothing_cannot_be_a_number():
    with pytest.raises(ValueError, match="over nothing scored"):
        Measured(0, 0.5)


def test_two_levels_of_folding_answer_what_one_would_have():
    # The property the whole document rests on: weighting each sequence by what
    # it scored makes the dataset mean exactly the mean over every finite score,
    # so splitting a run into chunks cannot move the answer.
    long = _sequence("plate/TL_00", 1200, ssim=0.99)
    short = _sequence("plate/TL_01", 600, ssim=0.90)

    folded = DatasetEvaluation("nexel", (long, short))

    flat = (1200 * 0.99 + 600 * 0.90) / 1800
    assert folded.metrics["ssim"].mean == pytest.approx(flat)
    assert folded.metrics["ssim"].scored == 1800
    assert folded.pairs == 1800


def test_weighting_by_pairs_instead_would_have_been_a_different_answer():
    # Which is why the weight is `scored`: the unweighted mean of the two means
    # counts the shorter sequence's frames twice over.
    long = _sequence("plate/TL_00", 1200, ssim=0.99)
    short = _sequence("plate/TL_01", 600, ssim=0.90)

    folded = DatasetEvaluation("nexel", (long, short))

    assert folded.metrics["ssim"].mean != pytest.approx((0.99 + 0.90) / 2)


def test_the_ends_name_the_sequences_that_reached_them():
    # A mean that rose while one sequence collapsed is the shape this search
    # produces, and the name is what settles whether the setting or the
    # sequence is at fault.
    sequences = [
        _sequence("plate/TL_00", 10, ssim=0.99),
        _sequence("plate/TL_01", 10, ssim=0.40),
        _sequence("plate/TL_02", 10, ssim=0.97),
    ]

    spread = DatasetEvaluation("nexel", tuple(sequences)).metrics["ssim"]

    assert spread.minimum == pytest.approx(0.40)
    assert spread.min_source == "plate/TL_01"
    assert spread.maximum == pytest.approx(0.99)
    assert spread.max_source == "plate/TL_00"


def test_a_sequence_that_scored_none_is_left_out_of_the_ends():
    # Otherwise a metric nobody measured reads as zero everywhere, and zero is
    # the worst a score can be for half of them.
    scored = _sequence("plate/TL_00", 10, ssim=0.99)
    absent = _sequence("plate/TL_01", 10, fb_error=None)

    spread = DatasetEvaluation("nexel", (scored, absent)).metrics["fb_error"]

    assert spread.min_source == "plate/TL_00"
    assert spread.scored == 10


def test_a_metric_nobody_measured_reads_as_absent():
    folded = DatasetEvaluation("nexel", (_sequence("a", 5, fb_error=None),))

    assert folded.metrics["fb_error"] == Spread(0, 0.0, 0.0, 0.0, "", "")


def test_a_sequence_filed_twice_is_refused():
    # One of them would be left out of every fold without saying so, since the
    # ends and the weights are both taken by name.
    twice = [_sequence("plate/TL_00", 5), _sequence("plate/TL_00", 5)]

    with pytest.raises(ValueError, match="appears twice"):
        DatasetEvaluation("nexel", tuple(twice))


@pytest.mark.parametrize("metric", METRICS)
def test_every_metric_is_folded_at_both_levels(metric):
    # A metric added to the tuple and nowhere else would be dropped in silence.
    sequence = _sequence("plate/TL_00", 3)

    assert metric in sequence.metrics
    assert metric in DatasetEvaluation("nexel", (sequence,)).metrics


@pytest.mark.parametrize(
    "value",
    (
        pytest.param(_frame(0), id="frame"),
        pytest.param(_sequence("plate/TL_00", 3), id="sequence"),
        pytest.param(
            DatasetEvaluation("nexel", (_sequence("plate/TL_00", 3),)), id="dataset"
        ),
    ),
)
def test_what_is_written_reads_back_as_what_it_was(value):
    assert type(value).from_dict(value.to_dict()) == value


def test_a_document_that_is_not_one_is_refused_by_the_key_that_gave_it_away():
    with pytest.raises(ValueError, match="'frames' is None"):
        SequenceEvaluation.from_dict({"source": "plate/TL_00"})


def test_a_fold_is_taken_again_rather_than_read_back():
    # A document whose numbers were edited by hand cannot disagree with the
    # pairs under them.
    folded = _sequence("plate/TL_00", 3, ssim=0.99)
    edited = folded.to_dict()
    edited["metrics"]["ssim"]["mean"] = 0.10

    assert SequenceEvaluation.from_dict(edited).metrics["ssim"].mean == pytest.approx(
        0.99
    )


def test_an_evaluation_of_nothing_is_refused():
    with pytest.raises(ValueError, match="answered no pair"):
        SequenceEvaluation("plate/TL_00", ())

    with pytest.raises(ValueError, match="holds no sequence"):
        DatasetEvaluation("nexel", ())


def test_a_score_that_is_not_finite_is_not_read_back():
    # The fold that writes a part leaves them out, so one here means the
    # document was written by something else.
    written = _frame(0).to_dict() | {"psnr": math.inf}

    with pytest.raises(ValueError, match="'psnr' is inf"):
        FrameEvaluation.from_dict(written)
