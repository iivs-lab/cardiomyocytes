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
    if indent < 0:
        msg = f"invalid indent {indent}: expected 0 or more"
        raise ValueError(msg)

    if depth < 0:
        msg = f"invalid depth {depth}: expected 0 or more"
        raise ValueError(msg)

    nested = f"{' ' * indent * depth}{message}"
    logger.log(level, nested, *args)
