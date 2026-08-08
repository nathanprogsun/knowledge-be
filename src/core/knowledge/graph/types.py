"""Graph node/edge domain types.

``Entity`` and ``Relationship`` mirror the upstream graph contract
field-for-field: the LLM-authored fields serialize on the wire
(``title`` / ``type`` / ``description`` for entities; ``source`` /
``target`` / ``description`` / ``strength`` for relationships) while the
aggregate bookkeeping (ids, chunk refs, degree, weight) stays internal,
matching the upstream ``json:"-"`` split.

``ChunkInput`` is the structural shape the builder consumes: both the
wire chunk contract and the ``chunks`` table model satisfy it, so the
builder composes with either without importing a storage layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class Entity(BaseModel):
    """A knowledge-graph node extracted from document chunks.

    ``title`` / ``type`` / ``description`` are the LLM-authored fields and
    are the only ones serialized; ``id`` / ``chunk_ids`` / ``frequency`` /
    ``degree`` are builder bookkeeping excluded from the wire shape.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default="", exclude=True)
    chunk_ids: list[str] = Field(default_factory=list, exclude=True)
    frequency: int = Field(default=0, exclude=True)
    degree: int = Field(default=0, exclude=True)
    title: str = ""
    type: str = ""
    description: str = ""


class Relationship(BaseModel):
    """A directed knowledge-graph edge between two entities.

    ``source`` / ``target`` / ``description`` / ``strength`` are the
    LLM-authored fields and are the only ones serialized; the rest is
    builder bookkeeping excluded from the wire shape.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default="", exclude=True)
    chunk_ids: list[str] = Field(default_factory=list, exclude=True)
    combined_degree: int = Field(default=0, exclude=True)
    weight: float = Field(default=0.0, exclude=True)
    source: str = ""
    target: str = ""
    description: str = ""
    strength: int = 0


@dataclass(frozen=True)
class ChunkRelation:
    """A weighted chunk-to-chunk edge in the chunk relation graph."""

    weight: float
    degree: int


class ChunkInput(Protocol):
    """Structural shape of the chunk rows the builder consumes.

    Both the wire chunk contract and the ``chunks`` table model satisfy
    this shape. Members are declared as read-only properties so frozen
    implementations (immutable dataclasses and frozen Pydantic models)
    are accepted.
    """

    @property
    def id(self) -> str: ...

    @property
    def content(self) -> str: ...

    @property
    def start_at(self) -> int | None: ...

    @property
    def end_at(self) -> int | None: ...

    @property
    def chunk_index(self) -> int: ...


class GraphBuildResult(BaseModel):
    """Snapshot of a completed graph build."""

    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    chunk_count: int = 0
    elapsed_seconds: float = 0.0


__all__ = [
    "ChunkInput",
    "ChunkRelation",
    "Entity",
    "GraphBuildResult",
    "Relationship",
]
