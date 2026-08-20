from __future__ import annotations

__all__ = ("FieldGraph", "GraphNode", "build_graph", "run")

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from iivs_cardio.beating_profile.estimation.base import Field, Node, Source

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from torch import Tensor


@dataclass(frozen=True, slots=True)
class GraphNode:
    kind: type[Field]
    field: Field


@dataclass(frozen=True, slots=True)
class FieldGraph:
    ordered: tuple[GraphNode, ...]
    wanted: tuple[type[Field], ...]
    spans: Mapping[type[Node], tuple[int, int]]

    def indices(self, total: int) -> range:
        lead = max((self.spans[kind][0] for kind in self.wanted), default=0)
        trail = max((self.spans[kind][1] for kind in self.wanted), default=0)

        return range(lead, max(total - trail, lead))


def build_graph(
    wanted: Sequence[type[Field]],
    make: Callable[[type[Field]], Field],
) -> FieldGraph:
    spans: dict[type[Node], tuple[int, int]] = {}
    order: list[GraphNode] = []

    def visit(kind: type[Node]) -> None:
        if kind in spans:
            return

        if issubclass(kind, Source):
            spans[kind] = kind.TRIM
            return

        if not issubclass(kind, Field):
            msg = f"{kind.__name__} is neither a source nor a field: nothing builds it"
            raise TypeError(msg)

        field = make(kind)
        needs = type(field).NEEDS

        ahead = (need for need in needs if need.next)
        if any(not issubclass(need.node, Source) for need in ahead):
            fix = "only a source can be read ahead of"
            msg = f"{kind.__name__} reads ahead of a field it rests on: {fix}"
            raise ValueError(msg)

        for need in needs:
            visit(need.node)

        spans[kind] = (
            max((spans[n.node][0] + n.prev for n in needs), default=0),
            max((spans[n.node][1] + n.next for n in needs), default=0),
        )
        order.append(GraphNode(kind, field))

    asked = tuple(dict.fromkeys(wanted))
    for kind in asked:
        visit(kind)

    return FieldGraph(tuple(order), asked, spans)


def run(
    graph: FieldGraph,
    sources: Mapping[type[Source], Sequence[Tensor]],
) -> Iterator[tuple[int, dict[type[Field], Tensor | None]]]:
    held: dict[type[Node], dict[int, Tensor | None]] = defaultdict(dict)
    total = min(len(frames) for frames in sources.values())
    asked = graph.indices(total)

    behind = max(
        (need.prev for node in graph.ordered for need in type(node.field).NEEDS),
        default=0,
    )

    def frame(node: type[Node], index: int) -> Tensor | None:
        if issubclass(node, Source) and index not in held[node]:
            held[node][index] = sources[node][index]

        return held[node][index]

    for index in range(total):
        for node in graph.ordered:
            lead, trail = graph.spans[node.kind]
            if not lead <= index < total - trail:
                continue

            window = [
                frame(need.node, j)
                for need in type(node.field).NEEDS
                for j in range(index - need.prev, index + need.next + 1)
            ]
            whole = all(value is not None for value in window)
            held[node.kind][index] = node.field.compute(*window) if whole else None

        if index in asked:
            yield index, {kind: held[kind][index] for kind in graph.wanted}

        for kept in held.values():
            for stale in [j for j in kept if j < index - behind]:
                del kept[stale]
