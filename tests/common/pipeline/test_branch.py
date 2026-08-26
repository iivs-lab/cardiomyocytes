from __future__ import annotations

from typing import get_args

import pytest

from iivs_cardio.common.pipeline.branch import (
    PRESENT_POLICIES,
    UNSOURCED_POLICIES,
    PresentPolicy,
    UnsourcedPolicy,
    ensure_json_name,
    ensure_policy,
)


@pytest.mark.parametrize(
    ("listed", "alias"),
    (
        (PRESENT_POLICIES, PresentPolicy),
        (UNSOURCED_POLICIES, UnsourcedPolicy),
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
    ((PRESENT_POLICIES, "reuse"), (UNSOURCED_POLICIES, "delete")),
)
def test_a_policy_a_branch_offers_comes_back_as_it_was(allowed, value):
    assert ensure_policy(value, allowed, "if_present") == value


def test_a_policy_nobody_offers_is_refused_by_the_key_it_came_from():
    # Configuration arrives as text whatever the field says, so this is the only
    # place a typo can be caught, and the message has to name both the setting
    # to go and fix and what it takes, since neither is guessable from the other.
    with pytest.raises(ValueError, match=r"unsupported if_present 'sync'"):
        ensure_policy("sync", PRESENT_POLICIES, "if_present")

    with pytest.raises(ValueError, match=r"expected 'error', 'overwrite', 'reuse'"):
        ensure_policy("", PRESENT_POLICIES, "if_present")


@pytest.mark.parametrize(
    ("named", "settled"),
    (
        ("source", "source.json"),
        ("source.json", "source.json"),
        ("value_range", "value_range.json"),
    ),
)
def test_a_name_a_branch_files_under_comes_back_carrying_json(named, settled):
    # The extension is added rather than demanded, so a setting reads as the
    # name of a thing rather than as a file the writer happens to open.
    assert ensure_json_name(named) == settled


@pytest.mark.parametrize("named", ("../up", "sub/down", "/rooted"))
def test_a_name_that_is_a_path_is_refused_before_anything_is_written(named):
    # It is joined onto the folder the output sits in, so a directory part puts
    # the file somewhere nothing looks for it and a `..` outside the tree.
    with pytest.raises(ValueError, match=r"expected a name, no directory part"):
        ensure_json_name(named)


def test_a_name_carrying_another_extension_is_refused():
    # The frame readers select by extension, so a record named like a frame
    # would be read back as one.
    with pytest.raises(ValueError, match=r"unsupported extension 'bin'"):
        ensure_json_name("source.bin")

    # A dot is what an extension is, so `a.b` is a name carrying one rather
    # than a name with a dot in it, and nothing here guesses which was meant.
    with pytest.raises(ValueError, match=r"unsupported extension 'b'"):
        ensure_json_name("a.b")
