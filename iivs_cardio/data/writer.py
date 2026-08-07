from __future__ import annotations

__all__ = ("KoalaFrameWriter",)

from typing import TYPE_CHECKING, Any, Self

from iivs.dhm.data.koala import koala_frame_name
from kaparoo.filesystem import StagedDirectory

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from types import TracebackType

    from kaparoo.filesystem.types import StrPath

    from iivs_cardio.common.pipeline import Step


class KoalaFrameWriter[T]:
    """A hook that writes the frames it is given as a Koala folder.

    Frames must arrive in order and without a gap, since the names they are
    written under are counted from zero rather than taken from the source. A
    step carrying no frame is passed over, which lets a sequence end early
    without failing, but one missing in the middle is refused.

    Nothing is moved into place until the writer closes cleanly. Closing after
    an error, or with no frame written at all, leaves the destination as it was.

    Type Parameters:
        T: what a frame is, as `save` expects it.

    Args:
        root: where the finished folder goes.
        save: writes one frame to one path.
        stem: the part of a frame's name after its number.
        ext: the extension a frame is written with.
        overwrite: whether an existing folder may be replaced.

    Raises:
        FileExistsError: If the destination is there and `overwrite` is not set.
    """

    def __init__(
        self,
        root: StrPath,
        save: Callable[[Path, T], object],
        *,
        stem: str,
        ext: str,
        overwrite: bool = False,
    ) -> None:
        self._root = StagedDirectory(root, overwrite=overwrite, make_parents=True)
        self._save = save
        self._stem = stem
        self._ext = ext
        self._written = -1

    def write(self, step: Step[T, Any]) -> None:
        """Write the frame in `step`, numbered by the index it came from.

        Raises:
            ValueError: If the frame does not follow the last one written.
        """
        if step.value is None:
            return

        if step.index != (expected := self._written + 1):
            msg = f"non-contiguous frame {step.index}: expected {expected}"
            raise ValueError(msg)

        name = koala_frame_name(step.index, stem=self._stem, ext=self._ext)
        self._save(self._root.workdir / name, step.value)
        self._written = step.index

    def __call__(self, step: Step[T, Any]) -> None:
        """Write `step`, so the writer can be registered as a hook directly."""
        self.write(step)

    def report(self) -> str | None:
        """Return one line naming how many frames were written, or `None`."""
        if self._written < 0:
            return None

        count = self._written + 1

        return f"wrote {count} frame{'s' if count != 1 else ''}"

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Move the folder into place, unless nothing was written or it failed.

        Raises:
            ValueError: If the sequence ended without a single frame, since
                there is then nothing to move and an empty folder would read as
                a finished one.
        """
        if exc_type is not None:
            self._root.abort()
            return

        if self._written < 0:
            self._root.abort()
            msg = f"no frame was written: nothing to commit at {self._root.path}"
            raise ValueError(msg)

        self._root.commit()
