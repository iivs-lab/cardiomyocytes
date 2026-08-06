from __future__ import annotations

__all__ = ("PhaseFilteredSequence", "PhaseStageFactory")

import logging
from contextlib import AbstractContextManager, ExitStack, contextmanager
from typing import TYPE_CHECKING, Self

from iivs.dhm.data.phase import PhaseFileFolder
from kaparoo.filesystem import stringify_path
from kaparoo.utils.timer import Timer

from iivs_cardio.common.pipeline import Reporting, SequenceStage, SideBranch
from iivs_cardio.data.transforms.filtering import FilteredSequence

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from pathlib import Path

    from torch import Tensor

    from iivs_cardio.common.device import Device
    from iivs_cardio.data.transforms.filtering.kernel import FilterKernel


_INDENT = "  "


class PhaseFilteredSequence(FilteredSequence[PhaseFileFolder, "Path"]):
    def __init__(
        self,
        source: PhaseFileFolder,
        kernel: FilterKernel,
        *,
        root: str,
        subpath: str,
        step: int = 1,
    ) -> None:
        super().__init__(source, kernel, step=step)
        self._name = stringify_path(source.root, after=root, before=subpath)

    @property
    def name(self) -> str:
        return self._name


class PhaseStageFactory:
    def __init__(
        self,
        sequences: Sequence[PhaseFilteredSequence],
        *branches: SideBranch[PhaseFilteredSequence, Tensor, Path],
        name: str,
    ) -> None:
        self._sequences = sequences
        self._branches = branches
        self._name = name

        self._logger = logging.getLogger(name)

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(self._sequences)

    def get_name(self, index: int) -> str:
        return self._sequences[index].name

    def get_stage(self, index: int, device: Device) -> SequenceStage[Tensor, Path]:
        sequence = self._sequences[index]
        sequence.device = device
        hooks = [branch.get_hook(sequence) for branch in self._branches]
        return SequenceStage(sequence).register_hooks(*hooks)

    def _log(self, message: str, *args: object, head: bool = False) -> None:
        if not head:
            message = f"{_INDENT}{message}"
        self._logger.info(message, *args)

    def run_stage(self, index: int, device: Device) -> None:
        sequence = self._sequences[index]

        self._log("%s", sequence.name, head=True)
        self._log("filtering %d frames on %s", len(sequence), device)

        with Timer("s") as timer:
            stage = self.get_stage(index, device)
            stage.run()

        for line in _reports(stage.hooks):
            self._log("%s", line)

        self._log("done in %.1fs", timer.elapsed)

    @contextmanager
    def running(self) -> Iterator[Self]:
        with ExitStack() as stack:
            for branch in self._branches:
                if isinstance(branch, AbstractContextManager):
                    stack.enter_context(branch)

            yield self

        for line in _reports(self._branches):
            self._log("%s", line, head=True)


def _reports(candidates: Iterable[object]) -> Iterator[str]:
    for candidate in candidates:
        if not isinstance(candidate, Reporting):
            continue

        if (line := candidate.report()) is not None:
            yield line
