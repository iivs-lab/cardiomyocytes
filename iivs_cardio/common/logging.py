from __future__ import annotations

__all__ = ("log_indented",)

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logging import Logger


def log_indented(
    logger: Logger,
    message: str,
    *args: object,
    indent: int = 2,
    depth: int = 1,
    level: int = logging.INFO,
) -> None:
    """Log a line set in from the left, so a block of lines reads as a block.

    The message keeps its placeholders and the arguments stay separate, so
    nothing is formatted when the line is not going to be written.

    Args:
        logger: where the line goes.
        message: the line, with placeholders for `args`.
        args: what fills those placeholders.
        indent: how many spaces one step in is worth.
        depth: how many steps in the line sits, where zero heads a block.
        level: the logging severity, such as `logging.INFO`.

    Raises:
        ValueError: If `indent` or `depth` is negative.
    """
    if indent < 0:
        msg = f"invalid indent {indent}: expected 0 or more"
        raise ValueError(msg)

    if depth < 0:
        msg = f"invalid depth {depth}: expected 0 or more"
        raise ValueError(msg)

    nested = f"{' ' * indent * depth}{message}"
    logger.log(level, nested, *args)
