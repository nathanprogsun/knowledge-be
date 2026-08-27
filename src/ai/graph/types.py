"""Graph domain models and the graph-retrieval repository interface.

``GraphData`` carries a graph payload (source text plus nodes and
relations); ``NameSpace`` scopes a graph to a knowledge base + knowledge
pair. ``RetrieveGraphRepository`` is the repository interface consumed by
the knowledge-graph build / search pipeline — the shape mirrors the
upstream contract field-for-field, including JSON serialization names.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    """A single node inside a graph."""

    name: str = ""
    chunks: list[str] = Field(default_factory=list)
    attributes: list[str] = Field(default_factory=list)


class GraphRelation(BaseModel):
    """A directed relationship between two nodes."""

    node1: str = ""
    node2: str = ""
    type: str = ""


class GraphData(BaseModel):
    """A graph payload: source text plus nodes and relations."""

    text: str = ""
    node: list[GraphNode] = Field(default_factory=list)
    relation: list[GraphRelation] = Field(default_factory=list)


class NameSpace(BaseModel):
    """Scopes a graph to a knowledge base and one knowledge entry."""

    knowledge_base: str = ""
    knowledge: str = ""

    def labels(self) -> list[str]:
        """Return the non-empty scope segments as label candidates.

        Both segments are used as Neo4j labels when present, which lets a
        graph live under a knowledge base alone or under a knowledge base
        + knowledge pair.
        """
        labels: list[str] = []
        if self.knowledge_base:
            labels.append(self.knowledge_base)
        if self.knowledge:
            labels.append(self.knowledge)
        return labels


@runtime_checkable
class RetrieveGraphRepository(Protocol):
    """Graph repository: write, delete and search node graphs.

    A repository may be *disabled* (no backing graph store): writes and
    deletes then no-op, and ``search_node`` returns ``None``.
    """

    async def add_graph(self, namespace: NameSpace, graphs: list[GraphData]) -> None:
        """Persist every graph under ``namespace``."""
        ...

    async def del_graph(self, namespaces: list[NameSpace]) -> None:
        """Delete the graphs under every given namespace."""
        ...

    async def search_node(self, namespace: NameSpace, nodes: list[str]) -> GraphData | None:
        """Return the subgraph matching any of ``nodes``, or ``None`` when disabled."""
        ...
