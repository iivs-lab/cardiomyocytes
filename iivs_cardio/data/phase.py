from __future__ import annotations

__all__ = ("save_phase_bin_folder",)

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from iivs.dhm.data.koala import save_koala_frames
from iivs.dhm.data.phase import PhaseBinFolder, PhaseUnit, save_phase_bin

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np
    from iivs.common.data import OnNonFinite
    from kaparoo.filesystem.types import StrPath
    from numpy.typing import NDArray


def save_phase_bin_folder(
    dest: StrPath,
    frames: Iterable[NDArray[np.float32]],
    *,
    pixel_size: float,
    height_scale: float,
    unit: PhaseUnit = PhaseUnit.RADIANS,
    overwrite: bool = False,
    on_nonfinite: OnNonFinite = "ignore",
) -> None:
    """Write `frames` into `dest` as a numbered folder `PhaseBinFolder` reads back.

    `frames` is consumed one at a time, so a whole sequence is never held in
    memory, and the folder is built atomically: a failure part-way leaves any
    existing `dest` untouched rather than a half-written folder. Frames are
    numbered from `0` in the order given, so a strided or filtered read writes a
    dense sequence rather than carrying the source's gaps.

    The scale arguments describe the frames being written, not the ones they came
    from -- a converted unit has to be recorded as the unit the data is now in, or
    a later reader scales it twice.

    Args:
        dest: The folder to create and fill.
        frames: The frames to write, in order.
        pixel_size: Sample spacing in metres, for the header of every frame.
        height_scale: Metres per radian. Not optional -- every `.bin` header
            carries one, so it comes from the source's.
        unit: The unit `frames` hold their values in. `NANOMETERS` is converted
            to `METERS` on the way out, since the header cannot hold it, so a
            caller states the unit rather than pre-converting.
        overwrite: Whether to replace `dest` if it already exists.
        on_nonfinite: What to do about NaN / inf in a frame. Defaults to
            `"ignore"` where `save_flow_folder` warns: a filter reduces over a
            dropped neighbourhood and may legitimately produce NaN, and at
            dataset scale a per-frame warning buries the log -- the range
            document is what reports them.

    Raises:
        ValueError: If `frames` is empty.
        FileExistsError: If `dest` exists and `overwrite` is False.
    """
    # `save_koala_frames` stages into a sibling of `dest`, so the parent has to
    # exist before it starts; it takes no `make_parents` of its own.
    Path(dest).parent.mkdir(parents=True, exist_ok=True)

    save = partial(
        save_phase_bin,
        pixel_size=pixel_size,
        height_scale=height_scale,
        unit=unit,
        overwrite=overwrite,
        on_nonfinite=on_nonfinite,
    )
    save_koala_frames(
        dest,
        frames,
        save,
        stem=PhaseBinFolder.FILE_STEM,
        ext=PhaseBinFolder.FILE_EXT,
        kind="phase",
        overwrite=overwrite,
    )
