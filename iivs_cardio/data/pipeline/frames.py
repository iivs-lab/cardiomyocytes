from __future__ import annotations

__all__ = ("FRAME_POLICIES", "FrameTree")

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self

from kaparoo.filesystem import dir_exists, search_dirs
from kaparoo.utils.optional import unwrap_or_default

from iivs_cardio.common.pipeline.branch import (
    STAGING,
    counted,
    find_unsourced,
    prune_above,
    read_policy,
)
from iivs_cardio.data.phase import phase_frame_writer

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import TracebackType

    from kaparoo.filesystem.types import StrPath
    from torch import Tensor

    from iivs_cardio.common.pipeline.branch import (
        ExistingOutputPolicy,
        UnsourcedOutputPolicy,
    )
    from iivs_cardio.data.phase import PhaseFilteredSequence
    from iivs_cardio.data.writer import KoalaFrameWriter

# A tree is written whole and carries no record of how it was made, so a later
# run cannot tell whether one already there still describes it.
FRAME_POLICIES: Final[tuple[ExistingOutputPolicy, ...]] = ("error", "overwrite")


@dataclass(frozen=True, slots=True)
class FrameTree:
    """The side branch that writes each sequence back out under a new root.

    A written sequence keeps the name and the layout it had in the source, so
    the result can be read by whatever reads the source. Each writer takes the
    pixel size, height scale and unit from the sequence it was made for.

    The tree has a lifetime as well as its writers, because two things outlive
    any one of them. A writer clears up after itself only while it is alive, so
    a worker killed part way leaves a staged folder and the empty folders above
    it; and a sequence the dataset has dropped leaves a folder no writer will
    ever be made for. Both are found by looking at the tree, which is what
    closing it does.

    Attributes:
        root: The directory the tree is written under.
        subpath: The path to a sequence's frames inside its own folder.
        contents: Every sequence the source holds, which is what tells a folder
            with no sequence behind it from one this run simply did not take.
            Given rather than defaulted, since an empty one leaves every folder
            here unsourced and `if_sources_gone` would then take the whole tree.
        if_frames_exist: The policy for a sequence that already has a folder
            here. `"reuse"` is refused until a tree can be added to. Defaults
            to `"error"`.
        if_sources_gone: The policy for a folder whose sequence the source has
            lost. Defaults to `"keep"`.

    Raises:
        ValueError: If `if_frames_exist` is `"reuse"`.
    """

    root: StrPath
    subpath: str
    contents: Mapping[str, Sequence[str]]
    if_frames_exist: ExistingOutputPolicy = field(default="error", kw_only=True)
    if_sources_gone: UnsourcedOutputPolicy = field(default="keep", kw_only=True)
    _dropped: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        read_policy(self.if_frames_exist, FRAME_POLICIES, "if_frames_exist")

    def get_hook(self, source: PhaseFilteredSequence) -> KoalaFrameWriter[Tensor]:
        """Return the writer for `source`, placed where the source sits."""
        origin = source.origin
        header = origin.header

        return phase_frame_writer(
            Path(self.root, source.name, self.subpath),
            pixel_size=header.pixel_size,
            height_scale=header.height_scale,
            unit=unwrap_or_default(origin.target_unit, header.unit),
            overwrite=self.if_frames_exist == "overwrite",
        )

    def list_sequences(self) -> list[str]:
        """Return every sequence this tree already holds frames for, sorted.

        A sequence is named by where its own folder sits under the root, and is
        recognised by holding `subpath` rather than by the walk reaching it, so
        nothing below one is ever listed. A folder inside a sequence is
        therefore never taken for one itself.
        """
        root = Path(self.root)
        if not root.is_dir():
            return []

        subpath = self.subpath
        found = search_dirs(
            root,
            predicate=lambda folder: dir_exists(folder / subpath),
            exclude=lambda folder: dir_exists(folder.parent / subpath),
            ordered=False,
        )

        return sorted(folder.relative_to(root).as_posix() for folder in found)

    def list_unsourced(self) -> list[str]:
        """Return the sequences this tree holds that the source has lost, sorted.

        A source that looks smaller than it is reads the same from here, so
        acting on the list is `if_sources_gone`'s to decide and naming it is not.
        """
        return find_unsourced(self.list_sequences(), self.contents)

    def drop_unsourced(self) -> list[str]:
        """Remove the folders of sequences the source has lost, and name them.

        The folders a removal empties go with it, so a sequence dropped from a
        nested dataset does not leave the path down to it standing.

        Returns:
            The sequences whose folders were removed, in the order they were
            found. They are kept for `report`, so a removal is said to have
            happened and not only to have been due.
        """
        root = Path(self.root)
        dropped = []

        for name in self.list_unsourced():
            folder = root / name
            shutil.rmtree(folder)
            prune_above(folder.parent, root)
            dropped.append(name)

        self._dropped.extend(dropped)

        return dropped

    def clear_staging(self) -> None:
        """Drop what a writer that did not live to clear up after itself left.

        A staged folder sits under a hidden name ending in `.tmp`, next to the
        destination it is to become, and the parents made to hold it are left
        empty when the move never happens. The writer undoes both while it is
        alive; a worker killed outright leaves them here. Only that shape is
        taken, and only the folders it leaves empty, since a run writes into
        the directory that also keeps its configuration and its logs.
        """
        root = Path(self.root)
        if not root.is_dir():
            return

        for folder in search_dirs(root, name_filter=STAGING):
            shutil.rmtree(folder, ignore_errors=True)
            prune_above(folder.parent, root)

    def report(self) -> str | None:
        """Return one line naming what was removed, or `None` if nothing was.

        Only removals: the frames a run writes are counted by the document that
        indexes them, and a tree that wrote every one of them and took nothing
        away has nothing here that is not said better elsewhere.
        """
        if not self._dropped:
            return None

        return f"removed {counted(len(self._dropped), 'folder')} with no source"

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Clear up after the writers, whether or not the run reached the end.

        Debris always goes, since it is this code's own unfinished business and
        nothing else will collect it. Whole folders go only where the policy
        says so.
        """
        self.clear_staging()

        if self.if_sources_gone == "delete":
            self.drop_unsourced()
