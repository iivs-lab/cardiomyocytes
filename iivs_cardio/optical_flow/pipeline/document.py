from __future__ import annotations

__all__ = ("Evaluated", "EvaluationDocument")

from typing import TYPE_CHECKING, Any, Protocol, override

from iivs_cardio.common.pipeline.base import Named
from iivs_cardio.common.pipeline.document import DocumentBranch
from iivs_cardio.optical_flow.pipeline.evaluation import (
    DatasetEvaluation,
    SequenceEvaluation,
)
from iivs_cardio.optical_flow.pipeline.evaluator import EvaluationWriter

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from kaparoo.filesystem.types import StrPath
    from torch import Tensor

    from iivs_cardio.common.pipeline import PresentPolicy, Stage, UnsourcedPolicy
    from iivs_cardio.common.warp import PaddingMode
    from iivs_cardio.optical_flow.estimators import OpticalFlowEstimator


class Evaluated(Named, Protocol):
    """Whatever an evaluation document needs of a sequence.

    Its name, to file the result under; the frames its flows were computed from, which
    no step carries and nothing here could rebuild, since reading them a second time
    would scale them a second time and two definitions of one thing agree only by
    coincidence; and the estimator those flows came from, which is bound to a device and
    so cannot be settled once for a whole run.
    """

    @property
    def frames(self) -> Stage[Tensor, Path]: ...

    @property
    def estimator(self) -> OpticalFlowEstimator | None: ...


class EvaluationDocument(
    DocumentBranch[Evaluated, SequenceEvaluation, DatasetEvaluation, EvaluationWriter]
):
    """The document a flow stage writes, gathering what every sequence scored.

    Args:
        path: The file to write the document to, given `.json` if it has none.
        source: The dataset root the run read, recorded so two documents can be told
            apart before anyone merges them.
        contents: Every sequence the source holds, mapped to the frames it was read over
            rather than the flows: what a sequence owes is worked out from them, so the
            same contents describes both branches of a stage. The whole dataset rather
            than the run's own selection, since a document may combine results an
            earlier run left and coverage counted against the selection would call that
            complete.
        settings: The block a later run would compare against this one. Defaults to
            `None`, which records nothing and so can never be reused.
        selected: The sequences of the contents this run was given to cover. Repeats
            count once. Defaults to `None`, which takes all of them.
        if_present: The policy for a sequence that already has a result here. Defaults
            to `"error"`.
        if_unsourced: The policy for a result whose sequence the source has lost.
            Defaults to `"keep"`.
        data_range: The value range SSIM and PSNR are scored against; taken from the
            frame dtype when omitted, which a float frame has none to give. The reverse
            flow each writer measures comes from the estimator its own sequence carries,
            since that is what is bound to the device the sequence ran on.
        padding_mode: `grid_sample` out-of-bounds policy for every warp.

    Attributes:
        RESULTS_SUFFIX: What the folder of results beside the document is called.
        path: The document itself, extension included.
        results_root: The folder each sequence's evaluation is written into.
        source: The dataset root the run read, recorded so two documents can be told
            apart before anyone merges them.
        contents: Every sequence the source holds, each mapped to the frames it was read
            over. A pair answers one flow, so a sequence is owed a score for every name
            here but the last.
        settings: The block a later run would compare against this one, `None` where
            nothing was recorded and so nothing can be reused.
        selected: The sequences of the contents this run was given to cover, repeats
            counted once.
        if_present: The policy for a sequence that already has an evaluation here.
            `"reuse"` keeps one still describing this run and scores the rest.
        if_unsourced: The policy for an evaluation whose sequence the source has lost.

    Raises:
        ValueError: If `if_present` or `if_unsourced` is not a policy a document offers,
            if `contents` is empty, since coverage would then have nothing to be
            measured against, or if `selected` names something the contents does not
            hold.
    """

    def __init__(
        self,
        path: StrPath,
        source: str,
        contents: Mapping[str, Sequence[str]],
        settings: Mapping[str, object] | None = None,
        *,
        selected: Sequence[str] | None = None,
        if_present: PresentPolicy = "error",
        if_unsourced: UnsourcedPolicy = "keep",
        data_range: float | None = None,
        padding_mode: PaddingMode = "border",
    ) -> None:
        super().__init__(
            path,
            source,
            contents,
            settings,
            selected=selected,
            if_present=if_present,
            if_unsourced=if_unsourced,
        )

        self._data_range = data_range
        self._padding_mode = padding_mode

    @override
    def _make_writer(self, source: Evaluated) -> EvaluationWriter:
        return EvaluationWriter(
            self.results_root,
            source.name,
            source.frames,
            source.estimator,
            self.settings,
            overwrite=self._replacing,
            data_range=self._data_range,
            padding_mode=self._padding_mode,
        )

    @override
    def _parse(self, document: Mapping[str, Any]) -> SequenceEvaluation:
        return SequenceEvaluation.from_dict(document)

    @override
    def _combine(self, results: tuple[SequenceEvaluation, ...]) -> DatasetEvaluation:
        return DatasetEvaluation(self.source, results)

    @override
    def _expected(self, names: Sequence[str]) -> Sequence[str]:
        """Every frame but the last, a pair being two frames in and one out.

        Start labelling, so the frame with nothing to pair with is the last rather than
        the first.
        """
        return names[:-1]
