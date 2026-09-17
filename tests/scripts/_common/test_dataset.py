from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

from scripts._common.dataset import FrameSelectConfig, SourceConfig


@dataclass
class _Layout(SourceConfig):
    """A stage's own source config, standing in for whichever reads the tree."""

    DEFAULT_SUBPATH: ClassVar[str] = "Phase/Float/Bin"


@pytest.mark.parametrize("subpath", ("/elsewhere", "../raw", "Phase/../../raw"))
def test_a_layout_no_sequence_could_hold_is_refused_where_it_is_read(subpath):
    # A layout is compared and joined as it stands, so one that walks out of the
    # sequence folder lands wherever it points. Refused as the field is read, not
    # once something comes to use it.
    with pytest.raises(ValueError, match=r"invalid subpath"):
        SourceConfig(root="/dataset", subpath=subpath)


def test_a_layout_a_config_is_handed_is_refused_as_its_own_would_be():
    # `resolve_subpath` keeps the check for the layout it follows, which belongs
    # to the other end of the stage and never passes through this field.
    with pytest.raises(ValueError, match=r"invalid subpath"):
        _Layout(root="/dataset").resolve_subpath("../raw")

    assert _Layout(root="/dataset").resolve_subpath() == "Phase/Float/Bin"


@pytest.mark.parametrize(
    ("settings", "refusal"),
    (
        ({"start": -1}, "invalid frame start -1"),
        ({"step": 0}, "invalid frame step 0"),
        ({"count": 0}, "invalid frame count 0"),
    ),
)
def test_a_frame_selection_nothing_could_read_is_refused_where_it_is_read(
    settings, refusal
):
    # Left to `indices` these three arrived in the middle of the search, after
    # the configuration had been logged and a tree of headers read.
    with pytest.raises(ValueError, match=refusal):
        FrameSelectConfig(**settings)


def test_a_frame_selection_is_taken_as_it_stands_once_it_reads():
    config = FrameSelectConfig(start=1, step=2, count=3)

    assert config.indices(10) == range(1, 7, 2)
    assert config.indices(4) == range(1, 4, 2)
