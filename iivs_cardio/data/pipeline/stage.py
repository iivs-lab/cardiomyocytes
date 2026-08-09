from __future__ import annotations

__all__ = ("PhaseStageFactory",)

import logging
from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING, Any, Self

from kaparoo.utils import quantify
from kaparoo.utils.timer import Timer

from iivs_cardio.common.logging import log_indented
from iivs_cardio.common.pipeline import (
    Holding,
    Reporting,
    SequenceStage,
    SideBranch,
    close_together,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from pathlib import Path

    from torch import Tensor

    from iivs_cardio.common.device import Device
    from iivs_cardio.data.phase import PhaseFilteredSequence


class PhaseStageFactory:
    """The sequences of one job, and how to run and report on each of them.

    The name is the job's to give rather than the factory's to assume: the same
    filtering run is preprocessing under one pipeline and postprocessing behind
    another, so a machine that named itself would be lying in the second case.
    Every line of the run is filed under it.

    Args:
        sequences: The sequences to run, in the order they will be offered.
        branches: The branches to watch each sequence with, such as a writer or
            a meter. Each is asked for a hook per sequence.
        name: The run's name.
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
        """The run's name, which every line of it is filed under."""
        return self._name

    def __len__(self) -> int:
        """The number of sequences this run was given."""
        return len(self._sequences)

    def get_name(self, index: int) -> str:
        """Return the name of the sequence at `index`.

        Args:
            index: The sequence to name.

        Returns:
            The name it has in its dataset.
        """
        return self._sequences[index].name

    def get_stage(
        self, index: int, device: Device
    ) -> SequenceStage[Tensor, Path] | None:
        """Build the stage for the sequence at `index`, running on `device`.

        Every branch is asked for a hook first, so a branch that cannot make one
        refuses before any frame is read, and one that has nothing to do for
        this sequence says so before it costs anything.

        Args:
            index: The sequence to build the stage for.
            device: The device the sequence is to be filtered on.

        Returns:
            The stage, or `None` when no branch wants this sequence. Reading it
            would then produce nothing anyone had asked for, so the device is
            left alone too.
        """
        sequence = self._sequences[index]
        made = (branch.get_hook(sequence) for branch in self._branches)
        hooks = [hook for hook in made if hook is not None]
        if not hooks:
            return None

        sequence.device = device

        return SequenceStage(sequence).register_hooks(*hooks)

    def _log(self, message: str, *args: object, nested: bool = True) -> None:
        """Log under this run's name, indented unless it heads a block.

        Args:
            message: The format string to log.
            *args: What it interpolates, left to the logger to apply.
            nested: Whether to indent the line under a block. Defaults to True.
        """
        log_indented(self._logger, message, *args, depth=int(nested))

    def _nothing_to_do(self) -> str:
        """Say why a sequence is being passed over, which is not always reuse.

        A run given no target has no branch to ask, so nothing is held and
        nothing was declined: there is simply nothing this run wants. Saying
        the branches already hold it would name a cause that is not there.
        """
        if not self._branches:
            return "nothing to do: this run writes nothing"

        return "nothing to compute: every branch already holds this sequence"

    def run_stage(self, index: int, device: Device) -> bool:
        """Filter the sequence at `index` on `device`, and log what happened.

        The sequence's name heads a block and everything else hangs under it,
        so a reader skimming the left margin sees one entry per sequence. Every
        branch that has something to say says it after it committed.

        The sequence lets go of its window afterwards, whether it finished or
        gave up. Every sequence of the run is held for the whole of it, so a
        window kept past the item it belongs to is held to the end: once per
        sequence, on the device, and again in each worker's own copy.

        Args:
            index: The sequence to carry out.
            device: The device to filter it on.

        Returns:
            Whether the sequence was computed. One that no branch wants a hook
            for is not read at all, and the frames that would have cost are the
            whole point of asking first.
        """
        sequence = self._sequences[index]

        self._log("%s", sequence.name, nested=False)

        stage = self.get_stage(index, device)
        if stage is None:
            self._log("%s", self._nothing_to_do())
            return False

        self._log("filtering %d frames on %s", len(sequence), device)

        try:
            with Timer("s") as timer:
                stage.run()
        finally:
            sequence.release()

        for line in _reports(stage.hooks):
            self._log("%s", line)

        self._log("done in %.1fs", timer.elapsed)

        return True

    def _log_unsourced(self) -> None:
        """Say which outputs have no sequence behind them any more, once each.

        Said whatever the branch then does with them, and before the run rather
        than after: a dataset that shrank and a share that came up half read
        the same from here, and only whoever started the run can tell them
        apart. Waiting until the end would say it after the frames were spent.
        """
        named = {
            name
            for branch in self._branches
            if isinstance(branch, Holding)
            for name in branch.list_unsourced()
        }
        if not named:
            return

        listed = ", ".join(sorted(named))
        outputs = quantify(len(named), "output")

        self._log("%s with no source: %s", outputs, listed, nested=False)

    @contextmanager
    def running(self) -> Iterator[Self]:
        """Open the branches that outlive one sequence, for the whole run.

        A branch that gathers across the dataset commits when this closes, and
        says what it committed afterwards. One whose work ends with the
        sequence it watched needs nothing here.

        Each branch is closed against the run's own outcome and never against
        what another raised, which is the rule the hooks of one sequence are
        closed by a level down. What committed still says so even when the next
        one could not, since a branch that committed nothing reports nothing
        anyway.
        """
        opened: list[AbstractContextManager[Any]] = []

        try:
            for branch in self._branches:
                if isinstance(branch, AbstractContextManager):
                    branch.__enter__()
                    opened.append(branch)

            self._log_unsourced()
            yield self
        except BaseException as error:
            close_together(opened, error)
            raise

        try:
            close_together(opened, None)
        finally:
            for line in _reports(self._branches):
                self._log("%s", line, nested=False)


def _reports(candidates: Iterable[object]) -> Iterator[str]:
    """Yield a line from each candidate that can report and has something."""
    for candidate in candidates:
        if not isinstance(candidate, Reporting):
            continue

        if (line := candidate.report()) is not None:
            yield line
