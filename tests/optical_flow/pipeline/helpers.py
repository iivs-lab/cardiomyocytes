from __future__ import annotations

__all__ = ("SIZE", "frame_stage", "shifted", "textured")

from pathlib import PurePath

import torch

from iivs_cardio.common.pipeline import SequenceStage
from iivs_cardio.common.warp import backward_warp

SIZE = 96


class _Frames:
    """A `DataSequence` of frames, which is all a stage asks of a source."""

    def __init__(self, frames: list[torch.Tensor]) -> None:
        self._frames = frames

    def __len__(self) -> int:
        return len(self._frames)

    def get_item(self, index: int) -> torch.Tensor:
        return self._frames[index]

    def get_meta(self, index: int) -> PurePath:
        return PurePath(f"{index:05d}_phase.bin")


def textured() -> torch.Tensor:
    """Return one frame a dense estimator can actually track across."""
    y, x = torch.meshgrid(
        torch.arange(SIZE).float(), torch.arange(SIZE).float(), indexing="ij"
    )
    weave = (
        128
        + 60 * torch.sin(2 * torch.pi * x / 17)
        + 50 * torch.sin(2 * torch.pi * y / 23)
    )
    return weave.clamp(0, 255).to(torch.uint8)


def shifted(frame: torch.Tensor, dx: float) -> torch.Tensor:
    """Return `frame` moved by `dx` across and a little less down."""
    offset = torch.zeros((2, SIZE, SIZE))
    offset[0] = -dx
    offset[1] = -dx * 0.6
    return backward_warp(frame, offset)


def frame_stage(count: int, *, duplicate: int | None = None) -> SequenceStage:
    """Return a stage of `count` frames drifting steadily across the field.

    Args:
        count: How many frames the stage holds.
        duplicate: The frame to replace with the one before it, which is what
            sends a reconstruction exact and PSNR to infinity. Defaults to
            `None`, which leaves every frame its own.
    """
    base = textured()
    frames = [shifted(base, i * 0.8) for i in range(count)]
    if duplicate is not None:
        frames[duplicate] = frames[duplicate - 1].clone()

    return SequenceStage(_Frames(frames), window=2)
