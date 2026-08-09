from __future__ import annotations

__all__ = (
    "EXISTING_OUTPUT_POLICIES",
    "SHORT_INPUT_POLICIES",
    "STAGING",
    "UNSOURCED_OUTPUT_POLICIES",
    "ExistingOutputPolicy",
    "ShortInputPolicy",
    "UnsourcedOutputPolicy",
    "as_read_back",
    "ensure_policy",
    "find_unsourced",
)

import json
from typing import TYPE_CHECKING, Final, Literal

from kaparoo.filters import And, EndsWith, StartsWith
from kaparoo.utils import ensure_one_of, literal_values

if TYPE_CHECKING:
    from collections.abc import Container, Iterable, Mapping, Sequence

type ExistingOutputPolicy = Literal["error", "overwrite", "reuse"]
type UnsourcedOutputPolicy = Literal["keep", "delete"]
type ShortInputPolicy = Literal["take", "error"]

EXISTING_OUTPUT_POLICIES: Final[tuple[ExistingOutputPolicy, ...]] = literal_values(
    ExistingOutputPolicy
)
UNSOURCED_OUTPUT_POLICIES: Final[tuple[UnsourcedOutputPolicy, ...]] = literal_values(
    UnsourcedOutputPolicy
)
SHORT_INPUT_POLICIES: Final[tuple[ShortInputPolicy, ...]] = literal_values(
    ShortInputPolicy
)

STAGING: Final = And((StartsWith("."), EndsWith(".tmp")))


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
