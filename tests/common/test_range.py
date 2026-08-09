from __future__ import annotations

import math

import pytest
import torch

from iivs_cardio.common.range import all_finite, finite_range


@pytest.mark.parametrize("intruder", (math.nan, math.inf, -math.inf))
def test_one_bad_value_anywhere_makes_a_frame_not_finite(intruder):
    # Every source frame passes this on the way in, so all three have to be
    # caught by the one fused pass: NaN reaches both bounds, an infinity only
    # the one it belongs to.
    assert all_finite(torch.tensor([[-2.5, 1.0], [7.25, 1.0]]))
    assert not all_finite(torch.tensor([[-2.5, intruder], [7.25, 1.0]]))


def test_a_frame_with_no_values_has_none_that_are_not_finite():
    # `aminmax` has no bounds to take from an empty frame and raises, where the
    # mask it replaced answered the vacuous truth.
    assert all_finite(torch.zeros(0, 0))


def test_a_value_a_wider_source_held_is_caught_once_it_is_narrowed():
    # The check follows the cast for this: `1e300` is an ordinary float64 and
    # an infinity in float32, and it is the narrowed frame that gets filtered.
    wide = torch.tensor([[1e300, 1.0]], dtype=torch.float64)

    assert all_finite(wide)
    assert not all_finite(wide.to(torch.float32))


def test_the_range_spans_the_whole_frame():
    frame = torch.tensor([[-2.5, 0.0], [7.25, 1.0]])

    assert finite_range(frame) == (-2.5, 7.25)


def test_a_constant_frame_answers_the_same_bound_twice():
    assert finite_range(torch.full((3, 3), 4.0)) == (4.0, 4.0)


@pytest.mark.parametrize("intruder", (math.nan, math.inf, -math.inf))
def test_a_non_finite_value_is_left_out_rather_than_propagated(intruder):
    # The fast path reads it through `aminmax`, so each of the three has to be
    # caught there: NaN reaches both bounds, an infinity only the one it belongs
    # to. A masked frame's NaN background would otherwise swallow the range.
    frame = torch.tensor([[-2.5, intruder], [7.25, 1.0]])

    assert finite_range(frame) == (-2.5, 7.25)


def test_a_frame_holding_no_finite_value_answers_none():
    frame = torch.tensor([[math.nan, math.inf], [-math.inf, math.nan]])

    assert finite_range(frame) is None


def test_an_empty_frame_answers_none():
    assert finite_range(torch.empty(0, 4)) is None


def test_an_integer_frame_is_read_as_it_stands():
    # `isfinite` accepts an integer tensor, so the slow path stays reachable, but
    # nothing there can be non-finite, so the fast path has to answer.
    assert finite_range(torch.tensor([[3, -1], [8, 0]], dtype=torch.int32)) == (-1, 8)
