from __future__ import annotations

__all__ = ("phase_frame_writer",)

from typing import TYPE_CHECKING

from iivs.dhm.data.phase import PhaseBinFolder, PhaseUnit, save_phase_bin

from iivs_cardio.common.writer import KoalaFrameWriter

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
