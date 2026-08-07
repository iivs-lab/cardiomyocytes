from __future__ import annotations

__all__ = ("FrameTree",)

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from kaparoo.utils.optional import unwrap_or_default

from iivs_cardio.data.phase import phase_frame_writer

if TYPE_CHECKING:
    from kaparoo.filesystem.types import StrPath
    from torch import Tensor

    from iivs_cardio.data.pipeline.stage import PhaseFilteredSequence
    from iivs_cardio.data.writer import KoalaFrameWriter


@dataclass(frozen=True, slots=True)
class FrameTree:
    root: StrPath
    subpath: str
    overwrite: bool = field(default=False, kw_only=True)

    def get_hook(self, source: PhaseFilteredSequence) -> KoalaFrameWriter[Tensor]:
        origin = source.origin
        header = origin.header

        return phase_frame_writer(
            Path(self.root, source.name, self.subpath),
            pixel_size=header.pixel_size,
            height_scale=header.height_scale,
            unit=unwrap_or_default(origin.target_unit, header.unit),
            overwrite=self.overwrite,
        )
