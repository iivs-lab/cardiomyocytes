from __future__ import annotations

__all__ = (
    "JSON_EXT",
    "PRESENT_POLICIES",
    "STAGING",
    "UNSOURCED_POLICIES",
    "Named",
    "PresentPolicy",
    "UnsourcedPolicy",
    "as_read_back",
    "ensure_json_name",
    "ensure_policy",
    "find_unsourced",
)

import json
from pathlib import PurePath
from typing import TYPE_CHECKING, Final, Literal, Protocol

from kaparoo.filesystem import ensure_file_extension
from kaparoo.filters import And, EndsWith, StartsWith
from kaparoo.utils import ensure_one_of, literal_values

if TYPE_CHECKING:
    from collections.abc import Container, Iterable, Mapping, Sequence

type PresentPolicy = Literal["error", "overwrite", "reuse"]
type UnsourcedPolicy = Literal["keep", "delete"]

PRESENT_POLICIES: Final[tuple[PresentPolicy, ...]] = literal_values(PresentPolicy)
UNSOURCED_POLICIES: Final[tuple[UnsourcedPolicy, ...]] = literal_values(UnsourcedPolicy)

STAGING: Final = And((StartsWith("."), EndsWith(".tmp")))


class Named(Protocol):
    """Something with a name, which is what an output or a log line is filed under."""

    @property
    def name(self) -> str: ...


JSON_EXT: Final = ".json"


def ensure_json_name(name: str) -> str:
    """Return `name` as a JSON file, given `.json` if it carries no extension.

    A branch takes the name of what it files from configuration, so the name it
    is given has to be one it can write beside the output rather than anywhere
    a path could reach.

    Args:
        name: The name the setting holds.

    Returns:
        The file name, which the folder holding it always contains.

    Raises:
        ValueError: If `name` carries a directory part.
        UnsupportedExtensionError: If it carries an extension of another kind.
    """
    if PurePath(name).name != name:
        msg = f"invalid file name {name!r}: expected a name, no directory part"
        raise ValueError(msg)

    return ensure_file_extension(name, JSON_EXT, add=True).name


def as_read_back(settings: Mapping[str, object] | None) -> object:
    """Return `settings` as they will read back out of an output on disk.

    Writing changes nothing: `json.dumps` puts a tuple down as an array
    faithfully. Reading is where the asymmetry appears, since that array comes
    back a list, and a list is not equal to the tuple that went in. Comparing
    what is held against what was written finds every output stale, this run's
    own included, and nothing is ever reused.

    Shared because both branches record the same block and both compare it
    back, so a difference between the two ways of reading it would show up as
    one output reusing what the other rewrote.
    """
    if settings is None:
        return None

    return json.loads(json.dumps(dict(settings), allow_nan=False))


def ensure_policy[T: str](value: str, allowed: Sequence[T], key: str) -> T:
    """Read a policy a caller wrote as plain text, refusing one nobody offers.

    Configuration arrives as strings whatever the field is annotated as, so
    this is where a value becomes one of the policies the code branches on.

    The check is `ensure_one_of`'s. What is kept here is the shape of the
    refusal, which every other one in this package matches and which names the
    setting rather than the argument that carried it.

    Args:
        value: The text the caller wrote.
        allowed: The policies this key takes, in the order to list them.
        key: The setting's own name, so a refusal says where to go and fix it.

    Returns:
        The value, as one of `allowed`.

    Raises:
        ValueError: If the value is not one of `allowed`.
    """
    try:
        return ensure_one_of(value, allowed, name=key)
    except ValueError:
        offered = ", ".join(repr(policy) for policy in allowed)
        msg = f"unsupported {key} {value!r}: expected {offered}"
        raise ValueError(msg) from None


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
