from __future__ import annotations

__all__ = ("FrameTree",)

from typing import TYPE_CHECKING, override

from kaparoo.utils.optional import unwrap_or_default

from iivs_cardio.common.pipeline.frames import FrameBranch
from iivs_cardio.data.phase import phase_frame_writer

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from torch import Tensor

    from iivs_cardio.common.pipeline.frames import FrameWriter
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

    __slots__ = ()

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
