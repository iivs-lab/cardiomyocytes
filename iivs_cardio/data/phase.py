from __future__ import annotations

__all__ = ("phase_frame_writer",)

from typing import TYPE_CHECKING

from iivs.dhm.data.phase import PhaseBinFolder, PhaseUnit, save_phase_bin

from iivs_cardio.data.writer import KoalaFrameWriter

if TYPE_CHECKING:
    from pathlib import Path

    from kaparoo.filesystem.types import StrPath
    from torch import Tensor


def phase_frame_writer(
    dest: StrPath,
    *,
    pixel_size: float,
    height_scale: float,
    unit: PhaseUnit = PhaseUnit.RADIANS,
    overwrite: bool = False,
) -> KoalaFrameWriter[Tensor]:
    """Build a writer that saves frames as a phase folder under `dest`.

    The pixel size, height scale and unit are written into each file, so what
    a later run needs to read them back travels with the frames rather than
    beside them. A non finite value is refused rather than written, since
    what is written here is what a later run will take as its source.

    Args:
        dest: where the finished folder goes.
        pixel_size: the size one pixel covers, stamped into each frame.
        height_scale: the scale that turns phase into height.
        unit: what the values mean.
        overwrite: whether an existing folder may be replaced.

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
    )
