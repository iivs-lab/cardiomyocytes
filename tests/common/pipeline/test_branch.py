from __future__ import annotations

from typing import get_args

import pytest

from iivs_cardio.common.pipeline.branch import (
    EXISTING_OUTPUT_POLICIES,
    UNSOURCED_OUTPUT_POLICIES,
    ExistingOutputPolicy,
    UnsourcedOutputPolicy,
    ensure_policy,
    find_unsourced,
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
    assert ensure_policy(value, allowed, "if_ranges_exist") == value


def test_a_policy_nobody_offers_is_refused_by_the_key_it_came_from():
    # Configuration arrives as text whatever the field says, so this is the only
    # place a typo can be caught, and the message has to name both the setting
    # to go and fix and what it takes, since neither is guessable from the other.
    with pytest.raises(ValueError, match=r"unsupported if_ranges_exist 'sync'"):
        ensure_policy("sync", EXISTING_OUTPUT_POLICIES, "if_ranges_exist")

    with pytest.raises(ValueError, match=r"expected 'error', 'overwrite', 'reuse'"):
        ensure_policy("", EXISTING_OUTPUT_POLICIES, "if_ranges_exist")


def test_what_the_contents_does_not_hold_is_unsourced_whatever_the_order():
    assert find_unsourced(["c", "a", "b"], {"b"}) == ["a", "c"]
    assert find_unsourced([], {"b"}) == []
    assert find_unsourced(["b"], {"b"}) == []
