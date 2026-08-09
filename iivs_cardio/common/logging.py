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
        logger: The logger the line goes to.
        message: The line, with placeholders for `args`.
        args: The values those placeholders take.
        indent: The number of spaces one step in is worth. Defaults to 2.
        depth: The number of steps in the line sits, where zero heads a block.
            Defaults to 1.
        level: The severity to log at. Defaults to `logging.INFO`.

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
