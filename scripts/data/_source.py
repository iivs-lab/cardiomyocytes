from __future__ import annotations

__all__ = ("sequence_name",)

from typing import TYPE_CHECKING

from kaparoo.filesystem import stringify_path

if TYPE_CHECKING:
    from iivs.dhm.data.phase import PhaseFileFolder


def sequence_name(origin: PhaseFileFolder, root: str, subpath: str) -> str:
    """What names a sequence inside the dataset it was found in.

    A path relative to `root` with `subpath` cut off its end, so the same
    sequence answers the same name to every side branch -- the range document
    records it, the frame writer lays its output out under it, and a reader
    matching the two has only this rule to agree with.
    """
    return stringify_path(origin.root, after=root, before=subpath)
