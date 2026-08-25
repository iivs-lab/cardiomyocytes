from __future__ import annotations

__all__ = ("Evaluated", "EvaluationDocument")

from typing import TYPE_CHECKING, Any, Protocol, override

from iivs_cardio.common.pipeline.branch import Named
from iivs_cardio.common.pipeline.document import DocumentBranch
from iivs_cardio.optical_flow.pipeline.evaluation import (
    DatasetEvaluation,
    SequenceEvaluation,
)
from iivs_cardio.optical_flow.pipeline.evaluator import SequenceEvaluator

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

    Its name, to file the part under; the frames its flows were computed from,
    which no step carries and nothing here could rebuild, since reading them a
    second time would scale them a second time and two definitions of one thing
    agree only by coincidence; and the estimator those flows came from, which
    is bound to a device and so cannot be settled once for a whole run.
    """

    @property
    def frames(self) -> Stage[Tensor, Path]: ...

    @property
    def estimator(self) -> OpticalFlowEstimator | None: ...


class EvaluationDocument(
    DocumentBranch[Evaluated, SequenceEvaluation, DatasetEvaluation, SequenceEvaluator]
):
    """The document a flow stage writes, gathering what every sequence scored.

    Args:
        path: As `DocumentBranch`.
        source: As `DocumentBranch`.
        contents: As `DocumentBranch`, holding the frames each sequence was
            read over rather than the flows: what a sequence owes is worked out
            from them, so the same contents describes both branches of a stage.
        settings: As `DocumentBranch`.
        selected: As `DocumentBranch`.
        if_present: As `DocumentBranch`.
        if_unsourced: As `DocumentBranch`.
        data_range: The value range SSIM and PSNR are scored against; taken
            from the frame dtype when omitted, which a float frame has none to
            give. The reverse flow each meter measures comes from the
            estimator its own sequence carries, since that is what is bound to
            the device the sequence ran on.
        padding_mode: `grid_sample` out-of-bounds policy for every warp.

    Attributes:
        PARTS_SUFFIX: As `DocumentBranch`.
        path: As `DocumentBranch`.
        parts_root: As `DocumentBranch`.
        source: As `DocumentBranch`.
        contents: As `DocumentBranch`.
        settings: As `DocumentBranch`.
        selected: As `DocumentBranch`.
        if_present: As `DocumentBranch`.
        if_unsourced: As `DocumentBranch`.

    Raises:
        ValueError: As `DocumentBranch`.
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
    def _make_meter(self, source: Evaluated) -> SequenceEvaluator:
        return SequenceEvaluator(
            self.parts_root,
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
    def _fold(self, parts: tuple[SequenceEvaluation, ...]) -> DatasetEvaluation:
        return DatasetEvaluation(self.source, parts)

    @override
    def _expected(self, names: Sequence[str]) -> Sequence[str]:
        """Every frame but the last, a pair being two frames in and one out.

        Start labelling, so the frame with nothing to pair with is the last
        rather than the first.
        """
        return names[:-1]
