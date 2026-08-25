from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, override

import pytest
import torch

from iivs_cardio.common.device import Device
from iivs_cardio.common.pipeline import SequenceStage, Step
from iivs_cardio.data.transforms.normalization import FrameNormalizer
from iivs_cardio.optical_flow.estimators import EstimatorConfig, FarnebackConfig
from iivs_cardio.optical_flow.pipeline import (
    FlowSource,
    FlowStage,
    FlowStageFactory,
    NormalizedFrameStage,
)
from tests.optical_flow.pipeline.helpers import SIZE, shifted, textured

if TYPE_CHECKING:
    from iivs_cardio.optical_flow.estimators import OpticalFlowEstimator

# Phase arrives as float and leaves as the uint8 the estimator takes, so the
# source range is the one the frames were built to fill.
SOURCE = (0.0, 2.55)
NORMALIZER = FrameNormalizer(SOURCE, (0.0, 255.0), torch.uint8)


def _phase(count: int, *, duplicate: int | None = None) -> list[torch.Tensor]:
    """Return `count` float phase frames drifting steadily across the field."""
    base = textured()
    frames = [shifted(base, index * 0.8).float() / 100 for index in range(count)]
    if duplicate is not None:
        frames[duplicate] = frames[duplicate - 1].clone()

    return frames


class _Sequence:
    """A phase sequence as a job meets it: named, sized, and releasable."""

    def __init__(self, name: str, frames: list[torch.Tensor]) -> None:
        self.name = name
        self.device: Device | None = None
        self.released = 0
        self._frames = frames

    def __len__(self) -> int:
        return len(self._frames)

    def get_item(self, index: int) -> torch.Tensor:
        return self._frames[index]

    def get_meta(self, index: int) -> Path:
        return Path(f"{index:05d}_phase.bin")

    def release(self) -> None:
        self.released += 1


class _CountingNormalizer:
    """A normalizer that says how many frames it was actually asked to scale."""

    def __init__(self) -> None:
        self.applied = 0

    def apply(self, frame: torch.Tensor) -> torch.Tensor:
        self.applied += 1
        return NORMALIZER.apply(frame)


class _CountingConfig(EstimatorConfig):
    """A config that says how many estimators were built from it."""

    FRAME_DTYPE = torch.uint8

    def __init__(self) -> None:
        self.builds = 0
        self._inner = FarnebackConfig()

    @override
    def build(self, device="cpu") -> OpticalFlowEstimator:
        self.builds += 1
        return self._inner.build(device)


class _Watching:
    """A branch whose hook reads the pair back, as the evaluation branch does."""

    def __init__(self, *, wanted: bool = True) -> None:
        self.wanted = wanted
        self.sources: list[FlowSource] = []
        self.seen: list[tuple[int, str]] = []

    def get_hook(self, source: FlowSource):
        self.sources.append(source)
        if not self.wanted:
            return None

        def watch(step: Step[torch.Tensor, Path]) -> None:
            source.frames[step.index].require()
            source.frames[step.index + 1].require()
            self.seen.append((step.index, step.require_extra().name))

        return watch


def _frames(count: int, normalizer=NORMALIZER) -> NormalizedFrameStage:
    sequence = SequenceStage(_Sequence("plate/TL_00", _phase(count)))
    return NormalizedFrameStage(sequence, normalizer)


# ========================== #
#         Normalizing        #
# ========================== #


def test_the_frames_come_out_as_the_estimator_takes_them():
    stage = _frames(3)

    frame = stage[0].require()

    assert frame.dtype == torch.uint8
    assert frame.shape == (SIZE, SIZE)


def test_a_frame_keeps_the_path_it_was_read_under():
    assert _frames(3)[1].require_extra().name == "00001_phase.bin"


def test_every_frame_of_the_source_answers_once():
    assert len(_frames(5)) == 5


def test_one_pair_of_constants_covers_the_whole_sequence():
    # Not each frame by its own range: a frame half as bright has to come out
    # half as bright. Scaling each by its own extremes would send both to the
    # top of the span, which is the brightness constancy the estimator above
    # reads motion from being destroyed underneath it.
    bright = _phase(1)[0]
    sequence = _Sequence("plate/TL_00", [bright, bright / 2])
    stage = NormalizedFrameStage(SequenceStage(sequence), NORMALIZER)

    top = int(stage[0].require().max())
    half = int(stage[1].require().max())

    assert half == pytest.approx(top / 2, abs=1)


def test_a_pair_read_twice_over_is_scaled_once():
    # The window is two so the branch's read lands on what the flow stage just
    # put there. A window of one would let go of `i` in between and scale it
    # again, silently.
    counting = _CountingNormalizer()
    frames = _frames(4, counting)

    for index in range(3):
        frames[index].require()
        frames[index + 1].require()

    assert counting.applied == 4


# ========================== #
#           Flowing          #
# ========================== #


def _flows(count: int, **kwargs) -> FlowStage:
    return FlowStage(_frames(count), FarnebackConfig().build("cpu"), **kwargs)


def test_n_frames_make_one_flow_fewer():
    assert len(_flows(5)) == 4


def test_a_flow_is_named_after_the_frame_it_starts_from():
    # Start labelling, so the frame with nothing to pair with is the last.
    stage = _flows(4)

    assert [step.require_extra().name for step in stage] == [
        "00000_phase.bin",
        "00001_phase.bin",
        "00002_phase.bin",
    ]


def test_a_flow_runs_forward_from_its_own_index():
    # The frames drift one way, so a flow read the other way round would come
    # back with the sign flipped.
    flow = _flows(3)[0].require()

    assert flow.shape == (2, SIZE, SIZE)
    assert flow[0].mean().item() > 0.0


def test_an_index_read_again_answers_what_it_answered_before():
    # The contract `_compute` is held to, and the reason the estimator is asked
    # for `calc`: what `push` answers depends on what has been pushed into it,
    # so re-reading index 0 after a pass would come back sign-flipped.
    stage = _flows(4)

    first = stage[0].require().clone()
    for index in range(len(stage)):
        stage[index].require()

    assert torch.equal(stage[0].require(), first)


def test_a_hook_fires_once_for_each_flow():
    stage = _flows(4)
    seen: list[int] = []

    stage.register_hooks(lambda step: seen.append(step.index))
    stage.run()

    assert seen == [0, 1, 2]


def test_a_flow_that_is_not_finite_is_refused_by_the_frame_it_starts_from():
    # Every metric would take it: `grid_sample` reads a non-finite coordinate
    # as out of bounds and the padding fills the hole, so a broken flow scores
    # as a slightly poor reconstruction rather than as broken.
    class _Broken:
        def calc(self, prev, curr):
            return torch.full((2, SIZE, SIZE), float("nan"))

    stage = FlowStage(_frames(3), _Broken())  # type: ignore[invalid-argument-type]

    with pytest.raises(ValueError, match=r"non-finite flow from 00000_phase\.bin"):
        stage[0].require()


# ========================== #
#            Job             #
# ========================== #


def _job(*sequences: _Sequence, branches=(), config=None) -> FlowStageFactory:
    normalizers = {sequence.name: NORMALIZER for sequence in sequences}

    return FlowStageFactory(
        sequences,  # type: ignore[invalid-argument-type]
        normalizers,
        config or FarnebackConfig(),
        *branches,
        name="flow",
    )


def _sequences(*counts: int) -> tuple[_Sequence, ...]:
    return tuple(
        _Sequence(f"plate/TL_{index:02d}", _phase(count))
        for index, count in enumerate(counts)
    )


def test_a_job_with_nothing_to_write_reads_nothing():
    sequences = _sequences(4)

    assert _job(*sequences).get_stage(0, Device("cpu")) is None
    assert sequences[0].device is None


def test_a_branch_is_asked_with_the_flows_source_rather_than_the_sequence():
    watching = _Watching()
    (sequence,) = _sequences(4)

    _job(sequence, branches=(watching,)).get_stage(0, Device("cpu"))

    (source,) = watching.sources
    assert source.name == sequence.name
    assert source.estimator is not None


def test_a_branch_reads_the_same_frames_the_flows_were_computed_from():
    # The whole reason the normalized frames are a stage: a branch reading a
    # stage of its own would scale every frame a second time, and score against
    # an encoding the flow never saw.
    counting = _CountingNormalizer()
    watching = _Watching()
    (sequence,) = _sequences(4)

    job = FlowStageFactory(
        (sequence,),  # type: ignore[invalid-argument-type]
        {sequence.name: counting},  # type: ignore[invalid-argument-type]
        FarnebackConfig(),
        watching,
        name="flow",
    )
    stage = job.get_stage(0, Device("cpu"))
    assert stage is not None
    stage.run()

    assert counting.applied == 4
    assert watching.seen == [
        (0, "00000_phase.bin"),
        (1, "00001_phase.bin"),
        (2, "00002_phase.bin"),
    ]


def test_one_estimator_is_built_for_a_device_however_many_sequences_run():
    config = _CountingConfig()
    sequences = _sequences(3, 3, 3)
    job = _job(*sequences, branches=(_Watching(),), config=config)

    for index in range(len(sequences)):
        job.get_stage(index, Device("cpu"))

    assert config.builds == 1


def test_the_sequence_lets_go_of_its_window_once_it_has_run():
    (sequence,) = _sequences(4)
    job = _job(sequence, branches=(_Watching(),))

    assert job.run_stage(0, Device("cpu"))
    assert sequence.released == 1
    assert sequence.device == Device("cpu")


def test_a_sequence_no_branch_wants_is_not_run():
    (sequence,) = _sequences(4)
    job = _job(sequence, branches=(_Watching(wanted=False),))

    assert not job.run_stage(0, Device("cpu"))
    assert sequence.released == 0


def test_a_sequence_with_no_normalizer_is_refused_by_name():
    (sequence,) = _sequences(4)

    with pytest.raises(ValueError, match="no normalizer for 'plate/TL_00'"):
        FlowStageFactory(
            (sequence,),  # type: ignore[invalid-argument-type]
            {},
            FarnebackConfig(),
            name="flow",
        )


def test_a_sequence_too_short_to_make_a_pair_is_refused_before_the_run():
    # It would answer no flow at all, and a document standing for it would
    # count it as covered while saying nothing.
    (sequence,) = _sequences(1)

    with pytest.raises(ValueError, match="holds 1 frames: a flow needs two"):
        _job(sequence)


def test_the_job_says_what_it_is_about_to_do():
    (sequence,) = _sequences(6)

    assert _job(sequence)._describe_work(0) == "computing 5 flows"  # noqa: SLF001
