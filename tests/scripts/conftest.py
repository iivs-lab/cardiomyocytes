from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from iivs.dhm.data.koala import PHASE_FLOAT_BIN
from iivs.dhm.data.phase import save_phase_bin

# `preprocess` reads this at import, and normally takes it from `.env`. Setting it
# here keeps the tests runnable in a clone that has not generated one yet.
CONFIGS_ROOT = str(Path(__file__).resolve().parents[2] / "configs")
os.environ.setdefault("CONFIGS_ROOT", CONFIGS_ROOT)

PIXEL_SIZE = 1.5e-7
HEIGHT_SCALE = 2.0e-7
SEQUENCES = 3
FRAMES = 4


@pytest.fixture()
def phase_tree(tmp_path: Path) -> Path:
    """A dataset root holding `SEQUENCES` time-lapses of `FRAMES` phase frames."""
    rng = np.random.default_rng(0)
    root = tmp_path / "src"

    for sequence in range(SEQUENCES):
        folder = root / f"TL_{sequence:02d}" / PHASE_FLOAT_BIN
        folder.mkdir(parents=True)
        for frame in range(FRAMES):
            save_phase_bin(
                folder / f"{frame:05d}_phase.bin",
                rng.random((4, 5), dtype=np.float32) * (sequence + 1),
                pixel_size=PIXEL_SIZE,
                height_scale=HEIGHT_SCALE,
            )

    return root
