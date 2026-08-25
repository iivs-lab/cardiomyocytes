from __future__ import annotations

__all__ = ("FRAMES", "SEQUENCES", "SIZE", "phase_tree", "range_document")

from typing import TYPE_CHECKING

import numpy as np
from iivs.dhm.data.koala import PHASE_FLOAT_BIN
from iivs.dhm.data.phase import save_phase_bin

from iivs_cardio.common.pipeline import save_document
from iivs_cardio.data.pipeline import DatasetRange, FrameRange, SequenceRange

if TYPE_CHECKING:
    from pathlib import Path

# Big enough for a dense estimator to build its pyramid over, which the shared
# 4x5 tree is not.
SIZE = 48
SEQUENCES = 2
FRAMES = 4

PIXEL_SIZE = 1.5e-7
HEIGHT_SCALE = 2.0e-7


def _drifting(index: int, sequence: int) -> np.ndarray:
    """Return one phase frame of a sequence that drifts steadily across."""
    y, x = np.meshgrid(
        np.arange(SIZE, dtype=np.float32),
        np.arange(SIZE, dtype=np.float32),
        indexing="ij",
    )
    shift = index * 0.8
    weave = np.sin(2 * np.pi * (x - shift) / 17) + np.sin(
        2 * np.pi * (y - shift * 0.6) / 23
    )

    return ((weave + 2.0) * (sequence + 1)).astype(np.float32)


def phase_tree(root: Path) -> Path:
    """Write a dataset of drifting phase sequences, and return its root."""
    for sequence in range(SEQUENCES):
        folder = root / f"TL_{sequence:02d}" / PHASE_FLOAT_BIN
        folder.mkdir(parents=True)
        for frame in range(FRAMES):
            save_phase_bin(
                folder / f"{frame:05d}_phase.bin",
                _drifting(frame, sequence),
                pixel_size=PIXEL_SIZE,
                height_scale=HEIGHT_SCALE,
            )

    return root


def range_document(path: Path, spans: dict[str, tuple[float, float]]) -> Path:
    """Write the document a measuring run would have left for `spans`."""
    sequences = tuple(
        SequenceRange(
            name,
            tuple(
                FrameRange(f"{index:05d}_phase.bin", *span) for index in range(FRAMES)
            ),
        )
        for name, span in spans.items()
    )

    return save_document(path, DatasetRange("src", sequences), overwrite=True)


def names(count: int = FRAMES) -> tuple[str, ...]:
    """The frame names a sequence of the tree holds."""
    return tuple(f"{index:05d}_phase.bin" for index in range(count))
