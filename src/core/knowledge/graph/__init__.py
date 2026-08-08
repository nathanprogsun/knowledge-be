"""Knowledge graph domain: the graph builder and its node/edge types.

The graph builder is a standalone service module: given a document's text
chunks and an injectable chat client, it extracts entities and
relationships, then computes the weighted chunk-relation graph used by
retrieval-time expansion. The web layer composes it with the document and
chunk services and a concrete provider later.
"""

from __future__ import annotations

from src.core.knowledge.graph.builder import (
    DEFAULT_EXTRACT_ENTITIES_PROMPT,
    DEFAULT_EXTRACT_RELATIONSHIPS_PROMPT,
    GraphBuilder,
    merge_text_chunks,
    parse_llm_json_response,
    render_prompt_placeholders,
)
from src.core.knowledge.graph.types import (
    ChunkInput,
    ChunkRelation,
    Entity,
    GraphBuildResult,
    Relationship,
)

__all__ = [
    "DEFAULT_EXTRACT_ENTITIES_PROMPT",
    "DEFAULT_EXTRACT_RELATIONSHIPS_PROMPT",
    "ChunkInput",
    "ChunkRelation",
    "Entity",
    "GraphBuildResult",
    "GraphBuilder",
    "Relationship",
    "merge_text_chunks",
    "parse_llm_json_response",
    "render_prompt_placeholders",
]
