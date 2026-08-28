from __future__ import annotations

__all__ = ("FlowTree", "flow_frame_writer")

from typing import TYPE_CHECKING, override

from iivs.dhm.data.koala import koala_frame_name

from iivs_cardio.common.pipeline.frames import RECORD_FILE, FrameBranch, FrameWriter
from iivs_cardio.optical_flow.data.folder import (
    OpticalFlowFolder,
    save_flow_npy,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from kaparoo.filesystem.types import StrPath
    from torch import Tensor

    from iivs_cardio.common.pipeline import Named


class FlowTree(FrameBranch["Named", "Tensor"]):
    """The frame tree of a flow stage, which answers once per pair of sources.

    A flow file carries the field alone, so a writer needs nothing off the
    sequence but the name the tree files it under. What a phase writer takes
    from its source, the pixel size and the height scale, is what a flow needs
    to become a velocity and is not carried here either.

    Attributes:
        root: As `FrameBranch`.
        subpath: As `FrameBranch`.
        contents: As `FrameBranch`, holding the phase frames each sequence was
            read from rather than the flows written back.
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
        source: Named,
        *,
        overwrite: bool,
        record: Mapping[str, object] | None,
    ) -> FrameWriter[Tensor]:
        return flow_frame_writer(
            dest, overwrite=overwrite, record=record, record_file=self.record_file
        )

    @override
    def _expected(self, names: Sequence[str]) -> Sequence[str]:
        """Every source frame but the last, which has nothing to pair with.

        A flow is labelled by the frame it starts from, so `N` frames answer
        `N - 1` times and the one left out is at the end. Comparing a record
        against the source's own frames instead would find one name too many
        every time, and refuse to reuse a folder this tree itself wrote.
        """
        return names[:-1]


def flow_frame_writer(
    dest: StrPath,
    *,
    overwrite: bool = False,
    record: Mapping[str, object] | None = None,
    record_file: str = RECORD_FILE,
) -> FrameWriter[Tensor]:
    """Build a writer that saves flows as an `OpticalFlowFolder` under `dest`.

    Header-less, so nothing a later run needs travels with the fields: neither
    the pixel size and frame interval that turn a flow into a velocity, nor
    which frames it came from. That is what `record` is for.

    A non finite value is refused rather than written, as it is on the phase
    side and for the same reason: what is written here is what a later run will
    take as its source, and one that got through could not be traced back.

    Args:
        dest: The folder the finished flows go to.
        overwrite: Whether an existing folder may be replaced. Defaults to
            `False`.
        record: The block the folder should carry about itself. Defaults to
            `None`, which files nothing.
        record_file: The name that block is filed under. Defaults to
            `RECORD_FILE`.
    """

    def save_fn(folder: Path, index: int, flow: Tensor) -> None:
        name = koala_frame_name(
            index, stem=OpticalFlowFolder.FILE_STEM, ext=OpticalFlowFolder.FILE_EXT
        )
        save_flow_npy(folder / name, flow.cpu().numpy(), on_nonfinite="raise")

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
