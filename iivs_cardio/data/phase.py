from __future__ import annotations

__all__ = ("PhaseFilteredSequence", "phase_frame_writer")

from typing import TYPE_CHECKING

from iivs.dhm.data.phase import (
    PhaseBinFolder,
    PhaseFileFolder,
    PhaseUnit,
    save_phase_bin,
)
from kaparoo.filesystem import stringify_path

from iivs_cardio.data.transforms.filtering import FilteredSequence
from iivs_cardio.data.writer import KoalaFrameWriter

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from kaparoo.filesystem.types import StrPath
    from torch import Tensor

    from iivs_cardio.data.transforms.filtering.kernel import FilterKernel


class PhaseFilteredSequence(FilteredSequence[PhaseFileFolder, "Path"]):
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
        limit: How many frames to take once the stride has been applied.
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
        limit: int | None = None,
    ) -> None:
        super().__init__(source, kernel, start=start, step=step, limit=limit)
        self._name = stringify_path(source.root, after=root, before=subpath)

    @property
    def name(self) -> str:
        """The name this sequence has in the dataset it belongs to."""
        return self._name


def phase_frame_writer(
    dest: StrPath,
    *,
    pixel_size: float,
    height_scale: float,
    unit: PhaseUnit = PhaseUnit.RADIANS,
    overwrite: bool = False,
    record: Mapping[str, object] | None = None,
) -> KoalaFrameWriter[Tensor]:
    """Build a writer that saves frames as a phase folder under `dest`.

    The pixel size, height scale and unit are written into each file, so what
    a later run needs to read them back travels with the frames rather than
    beside them. A non finite value is refused rather than written, since
    what is written here is what a later run will take as its source.

    A phase header carries no time and no source name, and the folder is
    renumbered from zero, so nothing in the frames themselves says which
    acquisition they came from. That is what `record` is for.

    Args:
        dest: The folder the finished frames go to.
        pixel_size: The size one pixel covers, stamped into each frame.
        height_scale: The scale that turns phase into height.
        unit: The meaning the values carry. Defaults to `PhaseUnit.RADIANS`.
        overwrite: Whether an existing folder may be replaced. Defaults to
            `False`.
        record: The block the folder should carry about itself. Defaults to
            `None`, which files nothing.

    Returns:
        A writer ready to be registered as a hook.
    """

    def save(path: Path, frame: Tensor) -> None:
        save_phase_bin(
            path,
            frame.cpu().numpy(),
            pixel_size=pixel_size,
            height_scale=height_scale,
            unit=unit,
            on_nonfinite="raise",
        )

    return KoalaFrameWriter(
        dest,
        save,
        stem=PhaseBinFolder.FILE_STEM,
        ext=PhaseBinFolder.FILE_EXT,
        overwrite=overwrite,
        record=record,
    )
