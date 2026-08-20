from __future__ import annotations

__all__ = ("Field", "FlowInput", "Need", "Node", "PhaseInput", "Source")

from dataclasses import dataclass
from inspect import Parameter, signature
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor


def _positional(compute: Callable[..., object]) -> int | None:
    positional = (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    taken = 0

    for parameter in list(signature(compute).parameters.values())[1:]:
        if parameter.kind is Parameter.VAR_POSITIONAL:
            return None
        if parameter.kind in positional:
            taken += 1

    return taken


class Node:
    NEEDS: ClassVar[tuple[Need, ...]] = ()


@dataclass(frozen=True, slots=True)
class Need:
    node: type[Node]
    prev: int = 0
    next: int = 0

    @property
    def width(self) -> int:
        return 1 + self.prev + self.next


class Source(Node):
    TRIM: ClassVar[tuple[int, int]] = (0, 0)


class PhaseInput(Source):
    pass


class FlowInput(Source):
    TRIM: ClassVar[tuple[int, int]] = (0, 1)


class Field(Node):
    compute: Callable[..., Tensor | None]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        if not cls.NEEDS:
            return

        compute = getattr(cls, "compute", None)
        if compute is None:
            msg = f"{cls.__name__} has needs but no `compute`"
            raise TypeError(msg)

        given = sum(need.width for need in cls.NEEDS)
        taken = _positional(compute)

        if taken is not None and taken != given:
            msg = f"{cls.__name__}.compute takes {taken} frames, its needs give {given}"
            raise TypeError(msg)
