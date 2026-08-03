from __future__ import annotations

__all__ = ("FieldWriter",)

from typing import TYPE_CHECKING, Self

from iivs.dhm.data.koala import koala_frame_name
from kaparoo.filesystem import StagedDirectory

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from types import TracebackType

    from kaparoo.filesystem.types import StrPath

    from iivs_cardio.common.pipeline import Slot


class FieldWriter[T]:
    """Write one field per step into a folder, as a pipeline hook.

    A hook is handed one step at a time, so this takes one field per call where
    the folder-at-a-time writers take an iterable and drain it -- which only ever
    worked while writing was the single consumer of a traversal. What varies
    between formats is one call, so `save` carries it along with whatever
    per-file settings it needs bound in; the shapes do not have to agree between
    writers, since a step is `(H, W)` for phase and `(2, H, W)` for a flow.

    Numbering follows `Slot.index`, not arrival. The two agree until a step goes
    missing, and there arrival order would close the gap and shift every later
    field one place away from the index the range document reports it under.
    Indexing by the slot instead leaves the hole where it happened, and a hole is
    then **refused**: `koala_frame_name` builds a contiguous name and the folder
    readers discover by it, so a folder with a gap is one no reader agrees with.

    An absent slot writes nothing. Under the forward convention those fall at the
    tail, so the folder simply ends earlier than the sequence it came from.

    It stages into a temporary directory moved into place on a clean exit, so a
    reader never sees a half-written folder and a failure leaves any existing
    `dest` untouched. Attaching it to a node is enough -- `Node.run` opens and
    closes every managed hook in the chain, together -- and it is a plain context
    manager outside one. Construction already creates the staging directory,
    following `StagedDirectory`'s own shape.

    Args:
        dest: The folder to create and fill.
        save: Writes one field to one path. Where the format lives, and where a
            device-to-host transfer belongs if the fields arrive as tensors.
        stem: The `<index>_<stem>.<ext>` stem the matching reader expects.
        ext: That reader's extension, without the dot.
        overwrite: Whether to replace `dest` if it already exists.

    Type Parameters:
        T: What a slot holds -- whatever `save` accepts.

    Example:
        ```python
        node.attach(FieldWriter(dest, save_flow_npy, stem="flow", ext="npy"))
        node.run()
        ```
    """

    def __init__(
        self,
        dest: StrPath,
        save: Callable[[Path, T], object],
        *,
        stem: str,
        ext: str,
        overwrite: bool = False,
    ) -> None:
        self._staged = StagedDirectory(dest, overwrite=overwrite, make_parents=True)
        self._save = save
        self._stem = stem
        self._ext = ext
        self._written = -1

    def write(self, slot: Slot[T]) -> None:
        """Write `slot`'s field under its own index, or nothing if it has none.

        Raises:
            ValueError: If `slot.index` does not continue the folder. Both a gap
                and an out-of-order step land here, since neither produces a
                folder a reader can discover.
        """
        if slot.value is None:
            return

        if slot.index != (expected := self._written + 1):
            msg = f"non-contiguous field {slot.index}: expected {expected}"
            raise ValueError(msg)

        name = koala_frame_name(slot.index, stem=self._stem, ext=self._ext)
        self._save(self._staged.workdir / name, slot.value)
        self._written = slot.index

    def __call__(self, slot: Slot[T]) -> None:
        """`write`, so a writer attaches to a node as the hook it is.

        Attaching the writer rather than its bound method is what lets the node
        recognise something it must open and close around the traversal.
        """
        self.write(slot)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            self._staged.abort()
            return

        if self._written < 0:
            self._staged.abort()
            msg = f"no field was written: nothing to commit at {self._staged.path}"
            raise ValueError(msg)

        self._staged.commit()
