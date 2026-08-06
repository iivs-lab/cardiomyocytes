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
        if step.value is None:
            return

        if step.index != (expected := self._written + 1):
            msg = f"non-contiguous frame {step.index}: expected {expected}"
            raise ValueError(msg)

        name = koala_frame_name(step.index, stem=self._stem, ext=self._ext)
        self._save(self._root.workdir / name, step.value)
        self._written = step.index

    def __call__(self, step: Step[T, Any]) -> None:
        self.write(step)

    def report(self) -> str | None:
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
        if exc_type is not None:
            self._root.abort()
            return

        if self._written < 0:
            self._root.abort()
            msg = f"no frame was written: nothing to commit at {self._root.path}"
            raise ValueError(msg)

        self._root.commit()
