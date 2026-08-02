from __future__ import annotations

import math

import pytest
import torch

from iivs_cardio.common.range import finite_range


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
    # nothing there can be non-finite -- the fast path has to answer.
    assert finite_range(torch.tensor([[3, -1], [8, 0]], dtype=torch.int32)) == (-1, 8)
