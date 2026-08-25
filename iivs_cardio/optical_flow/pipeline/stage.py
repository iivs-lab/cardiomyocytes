from __future__ import annotations

__all__ = ("FlowSource", "FlowStage", "FlowStageFactory", "NormalizedFrameStage")

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, override

from torch import Tensor

from iivs_cardio.common.pipeline import SequenceStage, Stage, StageJob
from iivs_cardio.common.range import all_finite
from iivs_cardio.optical_flow.estimators import OpticalFlowEstimator

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from iivs_cardio.common.device import Device
    from iivs_cardio.common.pipeline import SideBranch
    from iivs_cardio.data.phase import PhaseFilteredSequence
    from iivs_cardio.data.transforms.normalization import FrameNormalizer
    from iivs_cardio.optical_flow.estimators import EstimatorConfig


class NormalizedFrameStage(Stage[Tensor, Path]):
    """The frames of one sequence as the estimator downstream takes them.

    A stage of its own rather than a step inside the flow stage's `_compute`,
    because two consumers need the same frames: the flow stage reads `i` and
    `i + 1` to answer once, and the evaluation branch reads that same pair back
    to score the answer against it. Normalising inside `_compute` would leave
    those frames as local variables, and a branch normalising again would be a
    second definition of the same thing.

    The window is two for the same reason. The branch reads the pair right
    after the flow stage did, so anything shorter lets go of `i` in between and
    scales it a second time. That is not a wrong answer, since a hook does not
    fire twice for one index, but it is a silent slowdown.

    Args:
        frames: The stage the frames are read from, one per index.
        normalizer: The scaling every frame goes through.
        window: How many recent indices to keep, as for any stage. Defaults to
            2, which is what a consumer reading a pair needs.
    """

    def __init__(
        self,
        frames: Stage[Tensor, Path],
        normalizer: FrameNormalizer,
        *,
        window: int = 2,
    ) -> None:
        super().__init__(frames, window=window)

        self._frames = frames
        self._normalizer = normalizer

    def __len__(self) -> int:
        """The number of frames the source holds, each of which answers once."""
        return len(self._frames)

    @override
    def _compute(self, index: int) -> Tensor:
        return self._normalizer.apply(self._frames[index].require())

    @override
    def _describe(self, index: int) -> Path:
        return self._frames[index].require_extra()


class FlowStage(Stage[Tensor, Path]):
    """The flows of one sequence, one for each pair of the frames beneath it.

    `flow[i]` runs from frame `i` to frame `i + 1`, so `N` frames answer `N - 1`
    times and the frame with nothing to pair with is the last. That is the
    labelling this project gives every interval quantity, and it is why the
    estimator is asked for `calc` rather than `push`: `_compute` has to be a
    pure function of the index, and what `push` answers depends on what has
    been pushed into it instead.

    A flow that is not finite everywhere is refused here. Nothing downstream
    would catch it: `grid_sample` reads a non-finite coordinate as out of
    bounds and the padding fills the hole, so every metric comes back with an
    ordinary number and a broken flow reads as a slightly poor reconstruction.
    Refusing here covers the cache too, and one written past this point would
    be met by a later run with no way back to where it came from.

    Args:
        frames: The stage the pairs are read from.
        estimator: The estimator to ask for each pair.
        window: How many recent indices to keep, as for any stage. Defaults to
            1, since a flow is written and scored where it is computed and
            nothing reads it back.
    """

    def __init__(
        self,
        frames: Stage[Tensor, Path],
        estimator: OpticalFlowEstimator,
        *,
        window: int = 1,
    ) -> None:
        super().__init__(frames, window=window)

        self._frames = frames
        self._estimator = estimator

    def __len__(self) -> int:
        """The pairs the frames beneath make, which is one fewer than they are."""
        return max(len(self._frames) - 1, 0)

    @override
    def _compute(self, index: int) -> Tensor:
        first = self._frames[index].require()
        second = self._frames[index + 1].require()

        flow = self._estimator.calc(first, second)
        if not all_finite(flow):
            name = self._frames[index].require_extra().name
            msg = f"non-finite flow from {name}: drop the sequence, or repair it"
            raise ValueError(msg)

        return flow

    @override
    def _describe(self, index: int) -> Path:
        return self._frames[index].require_extra()


@dataclass(frozen=True, slots=True)
class FlowSource:
    """One sequence as the side branches of a flow stage meet it.

    Not the phase sequence itself. What a branch needs is the name to file its
    output under, the frames the flows were computed from, and the estimator
    they came from, and the last two are made per run and per device rather
    than being properties of the sequence.

    Attributes:
        name: The name the sequence has in its dataset.
        frames: The frames the flows were computed from, held so that a branch
            reads that same computation rather than making one of its own.
        estimator: The estimator the flows came from, for a branch with another
            question about the same pair. Defaults to `None`, which is what a
            run reading flows from a cache has.
    """

    name: str
    frames: Stage[Tensor, Path]
    estimator: OpticalFlowEstimator | None = None


class FlowStageFactory(StageJob["PhaseFilteredSequence"]):
    """The job of computing one dataset's flows, one stage graph per sequence.

    Three stages deep: the sequence reads and filters a phase frame, the frames
    above it are scaled onto what the estimator takes, and the flows above those
    are what the branches watch. The middle one exists so a branch can reach the
    very frames a flow was computed from, which is what makes a score a
    statement about that flow rather than about a second encoding of the pair.

    Args:
        sequences: The sequences to run, in the order they will be offered.
        normalizers: The scaling each sequence's frames go through, by name. One
            entry may be shared by every sequence, which is what a dataset-wide
            range comes to; a per-sequence range gives each its own.
        estimator: The settings the estimator is built from, built once per
            device rather than once per sequence.
        branches: The branches to watch each sequence's flows with. Each is
            asked for a hook with the `FlowSource`, not with the sequence.
        name: The run's name.

    Raises:
        ValueError: If a sequence has no normalizer, or holds too few frames to
            make a pair.
    """

    def __init__(
        self,
        sequences: Sequence[PhaseFilteredSequence],
        normalizers: Mapping[str, FrameNormalizer],
        estimator: EstimatorConfig,
        *branches: SideBranch[FlowSource, Tensor, Path],
        name: str,
    ) -> None:
        super().__init__(sequences, *branches, name=name)

        for sequence in sequences:
            if sequence.name not in normalizers:
                msg = f"no normalizer for {sequence.name!r}: give one per sequence"
                raise ValueError(msg)

            if len(sequence) < 2:
                held = len(sequence)
                msg = f"{sequence.name!r} holds {held} frames: a flow needs two"
                raise ValueError(msg)

        self._normalizers = normalizers
        self._config = estimator
        self._estimators: dict[Device, OpticalFlowEstimator] = {}

    def _get_estimator(self, device: Device) -> OpticalFlowEstimator:
        """Return this job's estimator for `device`, building one on first use.

        One per device rather than one per sequence: a cv2 algorithm is
        allocated on whichever device was current when it was made, so an
        estimator is bound to one, and a run that made one per sequence would
        pay that for every sequence it has.
        """
        if (estimator := self._estimators.get(device)) is None:
            estimator = self._config.build(device)
            self._estimators[device] = estimator

        return estimator

    @override
    def get_stage(self, index: int, device: Device) -> FlowStage | None:
        sequence = self._items[index]
        estimator = self._get_estimator(device)

        frames = NormalizedFrameStage(
            SequenceStage(sequence), self._normalizers[sequence.name]
        )
        flows = FlowStage(frames, estimator)

        source = FlowSource(sequence.name, frames, estimator)
        made = (branch.get_hook(source) for branch in self._branches)
        hooks = [hook for hook in made if hook is not None]
        if not hooks:
            return None

        sequence.device = device

        return flows.register_hooks(*hooks)

    @override
    def _describe_work(self, index: int) -> str:
        return f"computing {len(self._items[index]) - 1} flows"
