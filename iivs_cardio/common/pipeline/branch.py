from __future__ import annotations

__all__ = (
    "JSON_EXT",
    "PRESENT_POLICIES",
    "STAGING",
    "UNSOURCED_POLICIES",
    "DatasetBranch",
    "PresentPolicy",
    "UnsourcedPolicy",
    "as_json_value",
    "ensure_branch_policies",
    "ensure_json_name",
    "ensure_policy",
)

import json
from abc import ABC, abstractmethod
from pathlib import PurePath
from typing import TYPE_CHECKING, Final, Literal

from kaparoo.filesystem import ensure_file_extension
from kaparoo.filters import And, EndsWith, StartsWith
from kaparoo.utils import ensure_one_of, literal_values
from kaparoo.utils.optional import unwrap_or_default

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

type PresentPolicy = Literal["error", "overwrite", "reuse"]
type UnsourcedPolicy = Literal["keep", "delete"]

PRESENT_POLICIES: Final[tuple[PresentPolicy, ...]] = literal_values(PresentPolicy)
UNSOURCED_POLICIES: Final[tuple[UnsourcedPolicy, ...]] = literal_values(UnsourcedPolicy)

STAGING: Final = And((StartsWith("."), EndsWith(".tmp")))


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


def as_json_value(settings: Mapping[str, object] | None) -> object:
    """Return `settings` as the value JSON would give them back as.

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


def ensure_branch_policies(
    if_present: PresentPolicy,
    if_unsourced: UnsourcedPolicy,
) -> tuple[PresentPolicy, UnsourcedPolicy]:
    """Read the pair of policies every branch takes, refusing either unknown.

    Returns:
        The two, in the order they were given.
    """
    return (
        ensure_policy(if_present, PRESENT_POLICIES, "if_present"),
        ensure_policy(if_unsourced, UNSOURCED_POLICIES, "if_unsourced"),
    )


class DatasetBranch(ABC):
    """What a side branch judging a whole dataset holds, whatever it writes.

    A branch settles what to write again by measuring what it was given against
    what an earlier run left. The inputs that measurement rests on are the same
    whether the branch writes frames or a document, so they are read here and
    the two cannot drift apart.

    Args:
        contents: Every sequence the source holds, each mapped to the frames it
            covers and kept as a tuple. The whole dataset rather than the run's
            own selection.
        settings: The settings a later run would compare against this one.
            Defaults to `None`, which records nothing.
        selected: The sequences of the contents this run was given, repeats
            dropped. Defaults to `None`, which takes all of them.
        if_present: The policy for a sequence this branch already holds
            something for. Defaults to `"error"`.
        if_unsourced: The policy for what the branch holds and the source has
            lost. Defaults to `"keep"`.

    Raises:
        ValueError: If `if_present` or `if_unsourced` is not a policy a branch
            offers, or if `selected` names something the contents does not hold.
    """

    def __init__(
        self,
        contents: Mapping[str, Sequence[str]],
        settings: Mapping[str, object] | None = None,
        *,
        selected: Sequence[str] | None = None,
        if_present: PresentPolicy = "error",
        if_unsourced: UnsourcedPolicy = "keep",
    ) -> None:
        self.if_present, self.if_unsourced = ensure_branch_policies(
            if_present, if_unsourced
        )

        self.contents = {name: tuple(frames) for name, frames in contents.items()}
        self.settings = settings

        names = unwrap_or_default(selected, tuple(self.contents))
        if unknown := [name for name in names if name not in self.contents]:
            msg = f"selected {unknown[0]!r}, which the source does not hold"
            raise ValueError(msg)
        self.selected = tuple(dict.fromkeys(names))

        self._recorded = as_json_value(settings)

    @property
    def _replacing(self) -> bool:
        """Whether what a branch already holds may be written over.

        `"reuse"` replaces as readily as `"overwrite"`: what it keeps it keeps
        by never making a writer for it.
        """
        return self.if_present != "error"

    @abstractmethod
    def _expected(self, names: Sequence[str]) -> Sequence[str]:
        """Return the sources of what this stage owes for `names`.

        A stage answering once per source returns what it was given; one reading
        a pair to answer once returns fewer, and saying so is what lets what it
        wrote be recognised again.
        """
