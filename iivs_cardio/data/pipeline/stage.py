from __future__ import annotations

__all__ = ("PhaseFilteredSequence", "PhaseStageFactory")

import logging
from contextlib import AbstractContextManager, ExitStack, contextmanager
from typing import TYPE_CHECKING, Self

from iivs.dhm.data.phase import PhaseFileFolder
from kaparoo.filesystem import stringify_path
from kaparoo.utils.timer import Timer

from iivs_cardio.common.logging import log_indented
from iivs_cardio.common.pipeline import Reporting, SequenceStage, SideBranch
from iivs_cardio.data.transforms.filtering import FilteredSequence

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from pathlib import Path

    from torch import Tensor

    from iivs_cardio.common.device import Device
    from iivs_cardio.data.transforms.filtering.kernel import FilterKernel


class PhaseFilteredSequence(FilteredSequence[PhaseFileFolder, "Path"]):
    """A filtered phase sequence that knows what it is called in its dataset.

    The name is taken from where the folder sits under the dataset root, so a
    side branch filing something under it lands where the frames came from.

    Args:
        source: the phase folder to read.
        kernel: the reduction to apply over each window.
        root: the dataset root the name is measured from.
        subpath: the part of the folder's path that is the same for every
            sequence, and so is left out of the name.
        step: take every `step`th frame of the source, before filtering.
    """

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
        """What this sequence is called in the dataset it belongs to."""
        return self._name


class PhaseStageFactory:
    """The sequences of one job, and how to run and report on each of them.

    The name is the job's to give rather than the factory's to assume: the same
    filtering run is preprocessing under one pipeline and postprocessing behind
    another, so a machine that named itself would be lying in the second case.
    Every line of the run is filed under it.

    Args:
        sequences: the sequences to run, in the order they will be offered.
        branches: what to watch each sequence with, such as a writer or a
            meter. Each is asked for a hook per sequence.
        name: what the run is called.
    """

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
        """What the run is called, and what its log lines are filed under."""
        return self._name

    def __len__(self) -> int:
        """The number of sequences this run was given."""
        return len(self._sequences)

    def get_name(self, index: int) -> str:
        """Return what the sequence at `index` is called."""
        return self._sequences[index].name

    def get_stage(self, index: int, device: Device) -> SequenceStage[Tensor, Path]:
        """Build the stage for the sequence at `index`, running on `device`.

        Every branch is asked for a hook first, so a branch that cannot make
        one refuses before any frame is read.
        """
        sequence = self._sequences[index]
        sequence.device = device
        hooks = [branch.get_hook(sequence) for branch in self._branches]
        return SequenceStage(sequence).register_hooks(*hooks)

    def _log(self, message: str, *args: object, nested: bool = True) -> None:
        """Log under this run's name, indented unless it heads a block."""
        log_indented(self._logger, message, *args, depth=int(nested))

    def run_stage(self, index: int, device: Device) -> None:
        """Filter the sequence at `index` on `device`, and log what happened.

        The sequence's name heads a block and everything else hangs under it,
        so a reader skimming the left margin sees one entry per sequence. Every
        branch that has something to say says it after it committed.

        The sequence lets go of its window afterwards, whether it finished or
        gave up. Every sequence of the run is held for the whole of it, so a
        window kept past the item it belongs to is held to the end -- once per
        sequence, on the device, and again in each worker's own copy.
        """
        sequence = self._sequences[index]

        self._log("%s", sequence.name, nested=False)
        self._log("filtering %d frames on %s", len(sequence), device)

        try:
            with Timer("s") as timer:
                stage = self.get_stage(index, device)
                stage.run()
        finally:
            sequence.release()

        for line in _reports(stage.hooks):
            self._log("%s", line)

        self._log("done in %.1fs", timer.elapsed)

    @contextmanager
    def running(self) -> Iterator[Self]:
        """Open the branches that outlive one sequence, for the whole run.

        A branch that gathers across the dataset commits when this closes, and
        says what it committed afterwards. One whose work ends with the
        sequence it watched needs nothing here.
        """
        with ExitStack() as stack:
            for branch in self._branches:
                if isinstance(branch, AbstractContextManager):
                    stack.enter_context(branch)

            yield self

        for line in _reports(self._branches):
            self._log("%s", line, nested=False)


def _reports(candidates: Iterable[object]) -> Iterator[str]:
    """Yield a line from each candidate that can report and has something."""
    for candidate in candidates:
        if not isinstance(candidate, Reporting):
            continue

        if (line := candidate.report()) is not None:
            yield line
