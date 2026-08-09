from __future__ import annotations

from pathlib import Path
from typing import get_args

import pytest

from iivs_cardio.common.pipeline.branch import (
    EXISTING_OUTPUT_POLICIES,
    UNSOURCED_OUTPUT_POLICIES,
    ExistingOutputPolicy,
    UnsourcedOutputPolicy,
    find_unsourced,
    prune_above,
    read_policy,
)


@pytest.mark.parametrize(
    ("listed", "alias"),
    (
        (EXISTING_OUTPUT_POLICIES, ExistingOutputPolicy),
        (UNSOURCED_OUTPUT_POLICIES, UnsourcedOutputPolicy),
    ),
)
def test_the_policies_listed_are_the_ones_the_alias_stands_for(listed, alias):
    # Taken off the alias rather than written twice, and `get_args` reads a
    # `TypeAliasType` as having no arguments of its own: asked about the alias
    # instead of about its value it answers `()`, and every policy is then
    # refused.
    assert listed == get_args(alias.__value__)
    assert listed


@pytest.mark.parametrize(
    ("allowed", "value"),
    ((EXISTING_OUTPUT_POLICIES, "reuse"), (UNSOURCED_OUTPUT_POLICIES, "delete")),
)
def test_a_policy_a_branch_offers_comes_back_as_it_was(allowed, value):
    assert read_policy(value, allowed, "if_ranges_exist") == value


def test_a_policy_nobody_offers_is_refused_by_the_key_it_came_from():
    # Configuration arrives as text whatever the field says, so this is the only
    # place a typo can be caught, and the message has to name both the setting
    # to go and fix and what it takes, since neither is guessable from the other.
    with pytest.raises(ValueError, match=r"unsupported if_ranges_exist 'sync'"):
        read_policy("sync", EXISTING_OUTPUT_POLICIES, "if_ranges_exist")

    with pytest.raises(ValueError, match=r"expected 'error', 'overwrite', 'reuse'"):
        read_policy("", EXISTING_OUTPUT_POLICIES, "if_ranges_exist")


def test_what_the_contents_does_not_hold_is_unsourced_whatever_the_order():
    assert find_unsourced(["c", "a", "b"], {"b"}) == ["a", "c"]
    assert find_unsourced([], {"b"}) == []
    assert find_unsourced(["b"], {"b"}) == []


# ------------------------------ climbing back ----------------------------- #


def _tree(root: Path, *names: str) -> None:
    for name in names:
        (root / name).mkdir(parents=True, exist_ok=True)


def test_the_climb_takes_the_folders_a_removal_emptied(tmp_path):
    _tree(tmp_path, "plate/2026.03.11/gone")

    prune_above(tmp_path / "plate" / "2026.03.11" / "gone", tmp_path)

    assert not (tmp_path / "plate").exists()
    assert tmp_path.is_dir()  # `stop` itself is never removed


def test_the_climb_ends_at_the_first_folder_something_else_is_in(tmp_path):
    _tree(tmp_path, "plate/2026.03.11/gone", "plate/2026.03.12/kept")

    prune_above(tmp_path / "plate" / "2026.03.11" / "gone", tmp_path)

    assert not (tmp_path / "plate" / "2026.03.11").exists()
    assert (tmp_path / "plate" / "2026.03.12" / "kept").is_dir()


def test_a_folder_that_is_not_under_the_stop_is_left_alone(tmp_path):
    # The guard is "under `stop`", not "is not `stop`": a name the walk never
    # meets would let the climb go on past it to the root of the filesystem,
    # and what this function does at each step is remove a folder.
    _tree(tmp_path, "keep_me/a/b/c", "somewhere_else")

    prune_above(tmp_path / "keep_me" / "a" / "b" / "c", tmp_path / "somewhere_else")

    assert (tmp_path / "keep_me" / "a" / "b" / "c").is_dir()


def test_a_folder_that_cannot_be_removed_stops_the_climb(tmp_path, monkeypatch):
    # `rmdir` is the one that refuses, so every reason it refuses is the same
    # stopping rule: not empty, not there, and not ours all arrive as `OSError`.
    _tree(tmp_path, "plate/2026.03.11/gone")
    locked = tmp_path / "plate" / "2026.03.11"
    real = Path.rmdir

    def refuse_the_locked_one(self: Path) -> None:
        if self == locked:
            raise PermissionError(13, "in use")
        real(self)

    monkeypatch.setattr(Path, "rmdir", refuse_the_locked_one)

    prune_above(locked / "gone", tmp_path)

    assert not (locked / "gone").exists()  # the climb got this far
    assert locked.is_dir()  # and the refusal ended it here, unraised
    assert (tmp_path / "plate").is_dir()
