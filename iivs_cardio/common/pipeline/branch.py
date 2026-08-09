from __future__ import annotations

__all__ = (
    "EXISTING_OUTPUT_POLICIES",
    "STAGING",
    "UNSOURCED_OUTPUT_POLICIES",
    "ExistingOutputPolicy",
    "UnsourcedOutputPolicy",
    "counted",
    "find_unsourced",
    "prune_above",
    "read_policy",
)

from typing import TYPE_CHECKING, Final, Literal, cast, get_args

from kaparoo.filters import And, EndsWith, StartsWith

if TYPE_CHECKING:
    from collections.abc import Container, Iterable, Sequence
    from pathlib import Path

type ExistingOutputPolicy = Literal["error", "overwrite", "reuse"]
type UnsourcedOutputPolicy = Literal["keep", "delete"]

EXISTING_OUTPUT_POLICIES: Final[tuple[ExistingOutputPolicy, ...]] = get_args(
    ExistingOutputPolicy.__value__
)
UNSOURCED_OUTPUT_POLICIES: Final[tuple[UnsourcedOutputPolicy, ...]] = get_args(
    UnsourcedOutputPolicy.__value__
)

STAGING: Final = And((StartsWith("."), EndsWith(".tmp")))


def counted(count: int, noun: str) -> str:
    """Return `count` of `noun`, pluralised for every count but one.

    Shared so that the lines the branches report, which a run prints one after
    another, count the same way.
    """
    return f"{count} {noun}{'s' if count != 1 else ''}"


def read_policy[T: str](value: str, allowed: Sequence[T], key: str) -> T:
    """Read a policy a caller wrote as plain text, refusing one nobody offers.

    Configuration arrives as strings whatever the field is annotated as, so
    this is where a value becomes one of the policies the code branches on.

    Args:
        value: The text the caller wrote.
        allowed: The policies this key takes, in the order to list them.
        key: The setting's own name, so a refusal says where to go and fix it.

    Returns:
        The value, as one of `allowed`.

    Raises:
        ValueError: If the value is not one of `allowed`.
    """
    if value not in allowed:
        offered = ", ".join(repr(policy) for policy in allowed)
        msg = f"unsupported {key} {value!r}: expected {offered}"
        raise ValueError(msg)

    return cast("T", value)


def find_unsourced(present: Iterable[str], contents: Container[str]) -> list[str]:
    """Return the names on disk that the source no longer holds, sorted.

    A name here belongs to a sequence the dataset has since dropped, and that
    is all this can tell: a source that looks smaller than it is, because a
    mount is half up or a subpath is misspelt, produces exactly the same list.
    Deciding what to do with them is a policy, and saying they are there is not.

    Args:
        present: The names of the outputs there are.
        contents: The names the source holds.

    Returns:
        The names in `present` that `contents` does not hold, sorted.
    """
    return sorted(name for name in present if name not in contents)


def prune_above(folder: Path, stop: Path) -> None:
    """Remove `folder` and each parent it leaves empty, up to but not `stop`.

    The climb ends at the first folder something else is still in, so one an
    operator put here is not taken along with the sequence that happened to sit
    next to it. A `folder` that is not under `stop` is left alone rather than
    climbed from, since the guard would then be a name this walk never meets and
    the climb would go on to the root of the filesystem.

    Args:
        folder: The folder to remove if it is empty, and to climb from.
        stop: The folder to climb no further than, and never to remove. Nothing
            happens unless `folder` is under it.
    """
    if stop not in folder.parents:
        return

    while folder != stop:
        try:
            folder.rmdir()
        except OSError:  # not empty, not there, or not ours to remove
            return

        folder = folder.parent
