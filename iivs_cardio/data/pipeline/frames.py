from __future__ import annotations

__all__ = ("FrameTree", "phase_frame_writer")

from typing import TYPE_CHECKING, override

from iivs.dhm.data.koala import koala_frame_name
from iivs.dhm.data.phase import PhaseBinFolder, PhaseUnit, save_phase_bin
from kaparoo.utils.optional import unwrap_or_default

from iivs_cardio.common.pipeline.frames import (
    RECORD_FILE,
    FrameBranch,
    FrameWriter,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from kaparoo.filesystem.types import StrPath
    from torch import Tensor

    from iivs_cardio.data.phase import PhaseFilteredSequence


class FrameTree(FrameBranch["PhaseFilteredSequence", "Tensor"]):
    """The frame tree of a phase stage, which answers one frame per source.

    Each writer takes the pixel size, height scale and unit from the sequence it
    was made for, since a phase file carries them and a frame alone does not.

    Attributes:
        root: As `FrameBranch`.
        subpath: As `FrameBranch`.
        contents: As `FrameBranch`.
        settings: As `FrameBranch`.
        selected: As `FrameBranch`.
        record_file: As `FrameBranch`.
        if_present: As `FrameBranch`.
        if_unsourced: As `FrameBranch`.
    """

    @override
    def _make_writer(
        self,
        dest: Path,
        source: PhaseFilteredSequence,
        *,
        overwrite: bool,
        record: Mapping[str, object] | None,
    ) -> FrameWriter[Tensor]:
        origin = source.origin
        header = origin.header

        return phase_frame_writer(
            dest,
            pixel_size=header.pixel_size,
            height_scale=header.height_scale,
            unit=unwrap_or_default(origin.target_unit, header.unit),
            overwrite=overwrite,
            record=record,
            record_file=self.record_file,
        )

    @override
    def _expected(self, names: Sequence[str]) -> Sequence[str]:
        """Every source frame, filtering being one frame in and one frame out."""
        return names


def phase_frame_writer(
    dest: StrPath,
    *,
    pixel_size: float,
    height_scale: float,
    unit: PhaseUnit = PhaseUnit.RADIANS,
    overwrite: bool = False,
    record: Mapping[str, object] | None = None,
    record_file: str = RECORD_FILE,
) -> FrameWriter[Tensor]:
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
        record_file: The name that block is filed under. Defaults to
            `RECORD_FILE`.

    Returns:
        A writer ready to be registered as a hook.
    """

    def save_fn(folder: Path, index: int, frame: Tensor) -> None:
        name = koala_frame_name(
            index,
            stem=PhaseBinFolder.FILE_STEM,
            ext=PhaseBinFolder.FILE_EXT,
        )

        save_phase_bin(
            folder / name,
            frame.cpu().numpy(),
            pixel_size=pixel_size,
            height_scale=height_scale,
            unit=unit,
            on_nonfinite="raise",
        )

    def source_fn(source: Path) -> str:
        return source.name

    return FrameWriter(
        dest,
        save_fn,
        source_fn,
        overwrite=overwrite,
        record=record,
        record_file=record_file,
    )
