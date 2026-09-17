from __future__ import annotations

__all__ = ("SequenceStageRun",)

from typing import TYPE_CHECKING, override

from iivs_cardio.common.pipeline import SequenceStage, StageRun

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from torch import Tensor

    from iivs_cardio.common.device import Device
    from iivs_cardio.common.pipeline import SideBranch
    from iivs_cardio.data.phase import PhaseFilteredSequence


class SequenceStageRun(StageRun["PhaseFilteredSequence"]):
    """The run that filters one dataset's sequences, one stage per sequence.

    A sequence filters itself as it is read, so the stage is the sequence and the
    branches watch it directly.

    Args:
        sequences: The sequences to run, in the order they will be offered.
        branches: The branches to watch each sequence with, such as a writer or a meter.
            Each is asked for a hook with the sequence itself, a sequence being its own
            stage here.
        name: The run's name, which every line of it is filed under.
    """

    def __init__(
        self,
        sequences: Sequence[PhaseFilteredSequence],
        *branches: SideBranch[PhaseFilteredSequence, Tensor, Path],
        name: str,
    ) -> None:
        super().__init__(sequences, *branches, name=name)

    @override
    def build_stage(self, index: int, device: Device) -> SequenceStage[Tensor, Path]:
        return SequenceStage(self._items[index])

    @override
    def build_source(self, index: int, device: Device) -> PhaseFilteredSequence:
        return self._items[index]

    @override
    def get_stage(
        self, index: int, device: Device
    ) -> SequenceStage[Tensor, Path] | None:
        sequence = self.build_source(index, device)
        made = (branch.get_hook(sequence) for branch in self._branches)
        hooks = [hook for hook in made if hook is not None]
        if not hooks:
            return None

        sequence.device = device

        return self.build_stage(index, device).register_hooks(*hooks)

    @override
    def _work_label(self, index: int) -> str:
        return f"filtering {len(self._items[index])} frames"
