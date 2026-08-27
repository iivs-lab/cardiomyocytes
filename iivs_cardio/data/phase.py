from __future__ import annotations

__all__ = ("PhaseFilteredSequence",)

from pathlib import Path
from typing import TYPE_CHECKING

from iivs.dhm.data.phase import PhaseFileFolder
from kaparoo.filesystem import stringify_path

from iivs_cardio.data.transforms.filtering import FilteredSequence

if TYPE_CHECKING:
    from iivs_cardio.data.transforms.filtering.kernel import FilterKernel


class PhaseFilteredSequence(FilteredSequence[PhaseFileFolder, Path]):
    """A filtered phase sequence that knows what it is called in its dataset.

    The name is taken from where the folder sits under the dataset root, so a
    side branch filing something under it lands where the frames came from.

    Args:
        source: The phase folder to read.
        kernel: The reduction to apply over each window.
        root: The dataset root the name is measured from.
        subpath: The part of the folder's path that is the same for every
            sequence, and so is left out of the name.
        start: The first source frame to take. Defaults to 0.
        step: Take every `step`th frame of the source, before filtering.
            Defaults to 1.
        count: How many frames to take once the stride has been applied.
            Defaults to `None`, which takes them all.
    """

    def __init__(
        self,
        source: PhaseFileFolder,
        kernel: FilterKernel,
        *,
        root: str,
        subpath: str,
        start: int = 0,
        step: int = 1,
        count: int | None = None,
    ) -> None:
        super().__init__(source, kernel, start=start, step=step, count=count)
        self._name = stringify_path(source.root, after=root, before=subpath)

    @property
    def name(self) -> str:
        """The name this sequence has in the dataset it belongs to."""
        return self._name
