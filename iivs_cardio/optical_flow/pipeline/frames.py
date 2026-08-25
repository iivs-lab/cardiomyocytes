from __future__ import annotations

__all__ = ("FlowTree",)

from typing import TYPE_CHECKING, override

from iivs_cardio.data.pipeline import FrameBranch
from iivs_cardio.optical_flow.data.folder import flow_frame_writer

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from torch import Tensor

    from iivs_cardio.common.pipeline import Named
    from iivs_cardio.data.writer import KoalaFrameWriter


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

    __slots__ = ()

    @override
    def _make_writer(
        self,
        dest: Path,
        source: Named,
        *,
        overwrite: bool,
        record: Mapping[str, object] | None,
    ) -> KoalaFrameWriter[Tensor]:
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
