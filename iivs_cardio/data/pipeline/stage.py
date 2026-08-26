from __future__ import annotations

__all__ = ("SequenceStageFactory",)

from typing import TYPE_CHECKING, override

from iivs_cardio.common.pipeline import SequenceStage, StageRun

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from torch import Tensor

    from iivs_cardio.common.device import Device
    from iivs_cardio.common.pipeline import SideBranch
    from iivs_cardio.data.phase import PhaseFilteredSequence


class SequenceStageFactory(StageRun["PhaseFilteredSequence"]):
    """The job of filtering one dataset's sequences, one stage per sequence.

    A sequence filters itself as it is read, so the stage is the sequence and
    the branches watch it directly.

    Args:
        sequences: As `StageRun`.
        branches: As `StageRun`, each asked for a hook with the sequence itself.
        name: As `StageRun`.
    """

    def __init__(
        self,
        sequences: Sequence[PhaseFilteredSequence],
        *branches: SideBranch[PhaseFilteredSequence, Tensor, Path],
        name: str,
    ) -> None:
        super().__init__(sequences, *branches, name=name)

    @override
    def get_stage(
        self, index: int, device: Device
    ) -> SequenceStage[Tensor, Path] | None:
        sequence = self._items[index]
        made = (branch.get_hook(sequence) for branch in self._branches)
        hooks = [hook for hook in made if hook is not None]
        if not hooks:
            return None

        sequence.device = device

        return SequenceStage(sequence).register_hooks(*hooks)

    @override
    def _describe_work(self, index: int) -> str:
        return f"filtering {len(self._items[index])} frames"
