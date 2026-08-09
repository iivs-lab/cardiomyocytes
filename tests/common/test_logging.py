from __future__ import annotations

import logging

import pytest

from iivs_cardio.common.logging import log_indented


@pytest.mark.parametrize(("depth", "margin"), ((0, ""), (1, "  "), (3, "      ")))
def test_each_depth_costs_one_indent(caplog, depth, margin):
    # A block's head and the lines hanging under it share this helper, so `depth`
    # has to reach 0 as well: the head is the same call with no margin.
    with caplog.at_level(logging.INFO):
        log_indented(logging.getLogger("stage"), "filtering %d frames", 5, depth=depth)

    assert caplog.records[0].getMessage() == f"{margin}filtering 5 frames"


@pytest.mark.parametrize(("indent", "margin"), ((0, ""), (2, "    "), (4, "        ")))
def test_the_indent_is_the_width_of_one_level(caplog, indent, margin):
    with caplog.at_level(logging.INFO):
        log_indented(
            logging.getLogger("stage"), "wrote %d frames", 5, indent=indent, depth=2
        )

    assert caplog.records[0].getMessage() == f"{margin}wrote 5 frames"


def test_the_arguments_stay_unformatted_until_a_handler_asks(caplog):
    # Indenting rewrites the template, so it has to leave the `%s` standing rather
    # than fold the arguments in, since `ruff`'s `G` rules refuse eager formatting
    # because it costs the call even when the level is off.
    with caplog.at_level(logging.INFO):
        log_indented(logging.getLogger("stage"), "done in %.1fs", 3.5)

    record = caplog.records[0]
    assert record.msg == "  done in %.1fs"
    assert record.args == (3.5,)
    assert record.getMessage() == "  done in 3.5s"


def test_the_level_is_the_callers(caplog):
    with caplog.at_level(logging.DEBUG):
        log_indented(
            logging.getLogger("stage"), "step %d refilled", 4, level=logging.DEBUG
        )

    assert caplog.records[0].levelno == logging.DEBUG


def test_the_logger_is_the_callers(caplog):
    # The stage names every line of its run, so the helper must not reach for a
    # logger of its own.
    with caplog.at_level(logging.INFO):
        log_indented(logging.getLogger("reconstruct"), "TL_00", depth=0)

    assert caplog.records[0].name == "reconstruct"


def test_a_negative_indent_is_refused():
    with pytest.raises(ValueError, match="invalid indent -1: expected 0 or more"):
        log_indented(logging.getLogger("stage"), "TL_00", indent=-1)


def test_a_negative_depth_is_refused():
    with pytest.raises(ValueError, match="invalid depth -1: expected 0 or more"):
        log_indented(logging.getLogger("stage"), "TL_00", depth=-1)
