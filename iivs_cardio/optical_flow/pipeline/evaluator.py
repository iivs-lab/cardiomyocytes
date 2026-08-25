from __future__ import annotations

__all__ = ("SequenceEvaluator",)

from typing import TYPE_CHECKING, override

from kaparoo.utils import quantify

from iivs_cardio.common.pipeline.document import PartMeter
from iivs_cardio.optical_flow.metrics import (
    WarpConsistency,
    flow_magnitude,
    forward_backward_error,
    identity_ssim,
)
from iivs_cardio.optical_flow.pipeline.evaluation import (
    FrameEvaluation,
    SequenceEvaluation,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from kaparoo.filesystem.types import StrPath
    from torch import Tensor

    from iivs_cardio.common.pipeline import Stage, Step
    from iivs_cardio.common.warp import PaddingMode
    from iivs_cardio.optical_flow.estimators import OpticalFlowEstimator


class SequenceEvaluator(PartMeter[SequenceEvaluation]):
    """Score every flow of one sequence, then write the result.

    This is the hook an evaluation document hands to a sequence. Unlike the
    range meter it is not a consumer of what the step carries: warp consistency
    wants the two frames the flow was computed from, and a step carries the flow
    alone. So it holds the stage those frames came from and pulls `i` and `i+1`
    when it fires, which costs nothing where the flow stage has just read them.

    Both frames come from that stage rather than from anywhere else, so what is
    scored is what the flow was computed on. Normalising a second time here
    would be a second definition of the same thing, and two that agreed only by
    coincidence.

    An estimator is what the reverse flow needs, and without one the
    forward-backward axis is simply not measured: a run reading flows from a
    cache has no estimator, and so cannot have that axis at all.

    Args:
        root: As `PartMeter`.
        source: As `PartMeter`.
        frames: The stage the flows were computed from, held rather than
            rebuilt so that both consumers of an index share one computation.
        estimator: The estimator to take the reverse flow from, which doubles
            what a pair costs. Defaults to `None`, which leaves that axis out.
        settings: As `PartMeter`.
        overwrite: As `PartMeter`.
        data_range: The value range SSIM and PSNR are scored against; taken from
            the frame dtype when omitted, which a float frame has none to give.
        padding_mode: `grid_sample` out-of-bounds policy for both warps.
    """

    def __init__(
        self,
        root: StrPath,
        source: str,
        frames: Stage[Tensor, Path],
        estimator: OpticalFlowEstimator | None = None,
        settings: Mapping[str, object] | None = None,
        *,
        overwrite: bool = False,
        data_range: float | None = None,
        padding_mode: PaddingMode = "border",
    ) -> None:
        super().__init__(root, source, settings, overwrite=overwrite)

        self._frames = frames
        self._estimator = estimator
        self._data_range = data_range
        self._padding_mode = padding_mode
        self._consistency = WarpConsistency(
            data_range=data_range, padding_mode=padding_mode
        )
        self._scored: list[FrameEvaluation] = []

    def __call__(self, step: Step[Tensor, Path]) -> None:
        """Score `step`, so the evaluator can be registered as a hook directly."""
        self.evaluate(step)

    def evaluate(self, step: Step[Tensor, Path]) -> None:
        """Score the flow in `step` against the pair it was computed from.

        The pair is `i` and `i + 1` of the frame stage, which is the labelling a
        flow carries: `flow[i]` runs from frame `i`, so the frame with nothing
        to pair with is the last rather than the first.

        Every score is kept as it came. A duplicated frame reconstructs exactly
        and sends PSNR to infinity, which the fold leaves out and counts rather
        than something here quietly rounding away.

        Raises:
            ValueError: If the step carries no flow or no path.
            IndexError: If the frame stage has nothing at `i + 1`, which means
                it is not the stage this flow was computed from.
        """
        flow = step.require()
        path = step.require_extra()

        first = self._frames[step.index].require()
        second = self._frames[step.index + 1].require()

        scored = self._consistency(first, second, flow)
        floor = identity_ssim(first, second, data_range=self._data_range)

        self._scored.append(
            FrameEvaluation(
                source=path.name,
                ssim=float(scored["ssim"]),
                ssim_floor=float(floor),
                psnr=float(scored["psnr"]),
                mse=float(scored["mse"]),
                mae=float(scored["mae"]),
                magnitude=float(flow_magnitude(flow)),
                fb_error=self._reverse(first, second, flow),
            )
        )

    def _reverse(self, first: Tensor, second: Tensor, flow: Tensor) -> float | None:
        """How far the flow fails to come back, or `None` with no estimator.

        The reverse is the same pair the other way round rather than a
        neighbouring flow, so there is nothing to reuse and it costs a second
        call. Taken from the estimator the flow itself came from: measuring one
        estimator's self-consistency with another's answer is a different
        question.
        """
        if self._estimator is None:
            return None

        backward = self._estimator.calc(second, first)
        error = forward_backward_error(flow, backward, padding_mode=self._padding_mode)

        return float(error)

    @override
    def _fold(self) -> SequenceEvaluation:
        return self.to_evaluation()

    def to_evaluation(self) -> SequenceEvaluation:
        """Fold what has been scored so far into one evaluation of the sequence.

        Raises:
            ValueError: If no pair has been scored yet.
        """
        return SequenceEvaluation(self._source, tuple(self._scored))

    def report(self) -> str | None:
        """Return one line naming what was scored, or `None` if nothing was."""
        if not self._scored:
            return None

        folded = self.to_evaluation()
        gain = folded.metrics["ssim"].mean - folded.metrics["ssim_floor"].mean
        pairs = quantify(folded.pairs, "pair")
        said = f"scored {pairs}, gaining {gain:+.4f} SSIM"

        error = folded.metrics["fb_error"]

        return f"{said}, {error.mean:.4f} px apart" if error.scored else said
