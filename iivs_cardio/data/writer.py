from __future__ import annotations

__all__ = ("RECORD_FILE", "KoalaFrameWriter")

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Self

from iivs.dhm.data.koala import koala_frame_name
from kaparoo.filesystem import StagedDirectory
from kaparoo.utils import quantify

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import TracebackType

    from kaparoo.filesystem.types import StrPath

    from iivs_cardio.common.pipeline import Step

# What a written folder says about itself, filed inside it. The readers select
# frames by extension, so a name of another kind sits there unnoticed.
RECORD_FILE: Final = "source.json"


class KoalaFrameWriter[T]:
    """A hook that writes the frames it is given as a Koala folder.

    Names are counted from the first frame that arrives rather than taken from
    the source, so a stream that has nothing to say until its second step still
    writes a folder numbered from zero. That is the ordinary shape of a stage
    that needs two frames to produce one.

    What is refused is a gap between frames: renumbering would close it, and the
    folder would then read as an unbroken sequence of whatever it does hold. A
    step carrying no frame is passed over while none has arrived and after the
    last one, which is what lets a sequence start late or end early.

    Nothing is moved into place until the writer closes cleanly. Closing after
    an error, or with no frame written at all, leaves the destination as it was.

    Renumbering is what makes the folder readable again, and it is also what
    loses the source. Given a `record`, the writer files what it was told plus
    the source name of every frame it took, in the order it took them, so the
    nth frame here can be traced back to the one it was made from.

    Type Parameters:
        T: The type of one frame, as `save` expects it.

    Args:
        root: The folder the finished frames go into.
        save: A function writing one frame to one path.
        stem: The part of a frame's name after its number.
        ext: The extension a frame is written with.
        overwrite: Whether an existing folder may be replaced. Defaults to
            False.
        record: The block the folder should carry about itself, beside the
            source names the writer collects. Defaults to `None`, which files
            nothing and asks nothing of the steps.

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
        record: Mapping[str, object] | None = None,
    ) -> None:
        self._made = [path for path in Path(root).parents if not path.is_dir()]
        self._root = StagedDirectory(root, overwrite=overwrite, make_parents=True)
        self._save = save
        self._stem = stem
        self._ext = ext
        self._record = record
        self._taken: list[str] = []
        self._written = -1
        self._source = -1
        self._committed = False

    def write(self, step: Step[T, Any]) -> None:
        """Write the frame in `step`, numbered after the last one written.

        Args:
            step: The step to write. One carrying no frame is passed over.

        Raises:
            ValueError: If a frame does not follow the one before it at the
                source, since the numbering here would close the gap, or if a
                `record` was asked for and the step says nothing about where
                its frame came from.
        """
        if step.value is None:
            return

        last = self._source
        if self._written >= 0 and step.index != last + 1:
            msg = f"frame {step.index} does not follow {last}: expected {last + 1}"
            raise ValueError(msg)

        if self._record is not None:
            self._taken.append(Path(str(step.require_extra())).name)

        name = koala_frame_name(self._written + 1, stem=self._stem, ext=self._ext)
        self._save(self._root.workdir / name, step.value)

        self._written += 1
        self._source = step.index

    def __call__(self, step: Step[T, Any]) -> None:
        """Write `step`, so the writer can be registered as a hook directly.

        Args:
            step: The step to write.
        """
        self.write(step)

    def report(self) -> str | None:
        """Return one line naming how many frames landed.

        Frames are staged and moved into place together, so there is nothing to
        report until that move: a sequence that was interrupted or gave up has
        no folder to point at, however many frames it had staged by then.

        Returns:
            The line, or `None` before the folder reached its destination.
        """
        if not self._committed:
            return None

        return f"wrote {quantify(self._written + 1, 'frame')}"

    def _file_record(self) -> None:
        """Write what the folder says about itself, into the staged folder.

        Into the staged folder rather than the committed one, so the move that
        makes the frames visible makes this visible with them: a record written
        afterwards could be missing from a folder that is otherwise finished,
        and a reader has no way to tell that from one written without a record.
        """
        if self._record is None:
            return

        document = {**self._record, "frames": self._taken}
        written = json.dumps(document, allow_nan=False)
        (self._root.workdir / RECORD_FILE).write_text(written, encoding="utf-8")

    def _abort(self) -> None:
        """Drop the staged folder, and the ones opening it brought into being.

        Staging needs somewhere to sit, so the destination's parents are made
        before a single frame is written. Leaving them behind would put an
        empty `<sequence>/...` in the output tree for every sequence that gave
        up, which reads as a sequence that is there. Only folders that were
        absent when this writer opened are removed, and the climb stops at the
        first one that will not come away: another sequence landed in it
        meanwhile, or a worker sharing the ancestor took it first.
        """
        self._root.abort()

        for path in self._made:
            try:
                path.rmdir()
            except OSError:
                return

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Move the folder into place, unless nothing was written or it failed.

        A move that fails takes the staged folder with it. It sits beside the
        destination under a hidden name, so leaving it there puts a `.tmp` in
        the output tree for a sequence that has none of its frames, and the
        only other reference to it dies with the process.

        Args:
            exc_type: The type of what the walk ended with, or `None` if it
                finished.
            exc: The exception the walk ended with, or `None` if it
                finished.
            traceback: The place it was raised, or `None` if it finished.

        Raises:
            ValueError: If the sequence ended without a single frame, since
                there is then nothing to move and an empty folder would read as
                a finished one.
        """
        if exc_type is not None:
            self._abort()
            return

        if self._written < 0:
            self._abort()
            msg = f"no frame was written: nothing to commit at {self._root.path}"
            raise ValueError(msg)

        try:
            self._file_record()
            self._root.commit()
        except BaseException:
            self._abort()
            raise

        self._committed = True
