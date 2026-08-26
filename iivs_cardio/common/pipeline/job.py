from __future__ import annotations

__all__ = ("Held", "StageJob")

import logging
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING, Any, Protocol, Self

from kaparoo.utils import quantify
from kaparoo.utils.timer import Timer

from iivs_cardio.common.logging import log_indented
from iivs_cardio.common.pipeline.base import (
    SupportsReport,
    SupportsUnsourced,
    close_together,
)
from iivs_cardio.common.pipeline.branch import Named

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from iivs_cardio.common.device import Device
    from iivs_cardio.common.pipeline.base import SideBranch, Stage


class Held(Named, Protocol):
    """Whatever a job needs of one item: a name, and a hold it can be told to drop.

    Every item of a run is held for the whole of it, so anything an item keeps
    while it is being worked on would otherwise be kept to the end.
    """

    def release(self) -> None: ...


class StageJob[S: Held](ABC):
    """The items of one job, and how to run and report on each of them.

    The name is the job's to give rather than the machinery's to assume: the
    same filtering run is preprocessing under one pipeline and postprocessing
    behind another, so a machine that named itself would be lying in the second
    case. Every line of the run is filed under it.

    A subclass says two things and inherits the rest: what stage graph to build
    for an item, and what one line to say about the work before it starts.

    Type Parameters:
        S: The type of one item, which is one sequence of a dataset.

    Args:
        items: The items to run, in the order they will be offered.
        branches: The branches to watch each item with, such as a writer or a
            meter. Each is asked for a hook per item, which is the subclass's
            to do since only it knows what a branch is given.
        name: The run's name.
    """

    def __init__(
        self,
        items: Sequence[S],
        *branches: SideBranch[Any, Any, Any],
        name: str,
    ) -> None:
        self._items = items
        self._branches = branches
        self._name = name

        self._logger = logging.getLogger(name)

    @abstractmethod
    def get_stage(self, index: int, device: Device) -> Stage[Any, Any] | None:
        """Build the stage for the item at `index`, running on `device`.

        Every branch is asked for a hook, so a branch that cannot make one
        refuses before any frame is read, and one that has nothing to do for
        this item says so before it costs anything.

        Args:
            index: The item to build the stage for.
            device: The device the item is to be computed on.

        Returns:
            The stage, or `None` when no branch wants this item. Reading it
            would then produce nothing anyone had asked for, so the device is
            left alone too.
        """

    @abstractmethod
    def _describe_work(self, index: int) -> str:
        """Return what this job does to the item at `index`, without the device.

        Heads the item's block, so it reads as a phrase rather than a sentence:
        `"filtering 40 frames"`, which the device is appended to.
        """

    @property
    def name(self) -> str:
        """The run's name, which every line of it is filed under."""
        return self._name

    def __len__(self) -> int:
        """The number of items this run was given."""
        return len(self._items)

    def get_name(self, index: int) -> str:
        """Return the name of the item at `index`.

        Args:
            index: The item to name.

        Returns:
            The name it has in its dataset.
        """
        return self._items[index].name

    def _log(self, message: str, *args: object, nested: bool = True) -> None:
        """Log under this run's name, indented unless it heads a block.

        Args:
            message: The format string to log.
            *args: What it interpolates, left to the logger to apply.
            nested: Whether to indent the line under a block. Defaults to True.
        """
        log_indented(self._logger, message, *args, depth=int(nested))

    def _nothing_to_do(self) -> str:
        """Say why an item is being passed over, which is not always reuse.

        A run given no target has no branch to ask, so nothing is held and
        nothing was declined: there is simply nothing this run wants. Saying
        the branches already hold it would name a cause that is not there.
        """
        if not self._branches:
            return "nothing to do: this run writes nothing"

        return "nothing to compute: every branch already holds this sequence"

    def run_stage(self, index: int, device: Device) -> bool:
        """Carry out the item at `index` on `device`, and log what happened.

        The item's name heads a block and everything else hangs under it, so a
        reader skimming the left margin sees one entry per item. Every branch
        that has something to say says it after it committed.

        The item lets go of what it held afterwards, whether it finished or
        gave up. Every item of the run is held for the whole of it, so a window
        kept past the item it belongs to is held to the end: once per item, on
        the device, and again in each worker's own copy.

        Args:
            index: The item to carry out.
            device: The device to carry it out on.

        Returns:
            Whether the item was computed. One that no branch wants a hook for
            is not read at all, and the frames that would have cost are the
            whole point of asking first.
        """
        item = self._items[index]

        self._log("%s", item.name, nested=False)

        stage = self.get_stage(index, device)
        if stage is None:
            self._log("%s", self._nothing_to_do())
            return False

        self._log("%s on %s", self._describe_work(index), device)

        try:
            with Timer("s") as timer:
                stage.run()
        finally:
            item.release()

        for line in _reports(stage.hooks):
            self._log("%s", line)

        self._log("done in %.1fs", timer.elapsed)

        return True

    def _log_unsourced(self) -> None:
        """Say which outputs have no item behind them any more, once each.

        Said whatever the branch then does with them, and before the run rather
        than after: a dataset that shrank and a share that came up half read
        the same from here, and only whoever started the run can tell them
        apart. Waiting until the end would say it after the frames were spent.
        """
        named = {
            name
            for branch in self._branches
            if isinstance(branch, SupportsUnsourced)
            for name in branch.list_unsourced()
        }
        if not named:
            return

        listed = ", ".join(sorted(named))
        outputs = quantify(len(named), "output")

        self._log("%s with no source: %s", outputs, listed, nested=False)

    @contextmanager
    def running(self) -> Iterator[Self]:
        """Open the branches that outlive one item, for the whole run.

        A branch that gathers across the dataset commits when this closes, and
        says what it committed afterwards. One whose work ends with the item it
        watched needs nothing here.

        Each branch is closed against the run's own outcome and never against
        what another raised, which is the rule the hooks of one item are closed
        by a level down. What committed still says so even when the next one
        could not, since a branch that committed nothing reports nothing anyway.
        """
        opened: list[AbstractContextManager[object]] = []

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
        if not isinstance(candidate, SupportsReport):
            continue

        if (line := candidate.report()) is not None:
            yield line
