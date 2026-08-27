"""Shared helpers for the Elasticsearch v7/v8 engine repositories.

The upstream contract defines a ``VectorEmbedding`` document shape and two
conversion helpers (``ToDBVectorEmbedding`` / ``FromDBVectorEmbeddingWithScore``)
used by both the v7 and v8 drivers. ``resolve_index_name`` mirrors the
index-name resolution priority (config > env > default). The storage-size
estimator and filter builder are shared so both drivers produce identical
DSL shapes.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from src.ai.retrieval.types import (
    IndexConfig,
    IndexInfo,
    IndexSaveParams,
    IndexWithScore,
    MatchType,
    RetrieveParams,
    SourceType,
)

#: Fixed metadata overhead per ES document (IDs, timestamps, etc.).
_METADATA_SIZE_BYTES: int = 250

#: ES index expansion factor applied to content + vector size (1.5x).
_INDEX_OVERHEAD_NUMERATOR: int = 5
_INDEX_OVERHEAD_DENOMINATOR: int = 10

#: Default ES index name when neither config nor env supplies one.
_DEFAULT_INDEX_NAME: str = "xwrag_default"

#: Fields whose query name may carry a ``.keyword`` suffix depending on the
#: index mapping's detected field type.
_ID_FIELDS: tuple[str, ...] = (
    "chunk_id",
    "source_id",
    "knowledge_id",
    "knowledge_base_id",
    "tag_id",
)


def resolve_index_name(index_config: IndexConfig | None, env_key: str, default: str) -> str:
    """Resolve the index name from config, env var, or default.

    Priority: ``index_config.index_name`` > env var > ``default``.
    """
    if index_config is not None and index_config.index_name:
        return index_config.index_name
    env_val = os.getenv(env_key, "")
    if env_val:
        return env_val
    return default


def id_field(name: str, use_keyword_suffix: bool) -> str:
    """Return the query field name, appending ``.keyword`` when the index
    uses text-type mappings with keyword sub-fields."""
    if use_keyword_suffix:
        return f"{name}.keyword"
    return name


def to_db_vector_embedding(info: IndexInfo, params: IndexSaveParams) -> dict[str, Any]:
    """Convert an ``IndexInfo`` to the ES document dict.

    The embedding vector is looked up from ``params["embedding"]`` keyed by
    ``info.source_id``; ``is_enabled`` may be overridden by
    ``params["chunk_enabled"]`` keyed by ``info.chunk_id``.
    """
    doc: dict[str, Any] = {
        "content": info.content,
        "source_id": info.source_id,
        "source_type": int(info.source_type),
        "chunk_id": info.chunk_id,
        "knowledge_id": info.knowledge_id,
        "knowledge_base_id": info.knowledge_base_id,
        "tag_id": info.tag_id,
        "is_enabled": info.is_enabled,
        "is_recommended": info.is_recommended,
    }
    embedding_map = params.get("embedding") if params else None
    if isinstance(embedding_map, dict):
        vec = embedding_map.get(info.source_id)
        if vec:
            doc["embedding"] = list(vec)
    chunk_enabled_map = params.get("chunk_enabled") if params else None
    if isinstance(chunk_enabled_map, dict) and info.chunk_id in chunk_enabled_map:
        doc["is_enabled"] = bool(chunk_enabled_map[info.chunk_id])
    return doc


def from_db_vector_embedding_with_score(
    doc_id: str,
    source: dict[str, Any],
    score: float,
    match_type: MatchType,
) -> IndexWithScore:
    """Build an ``IndexWithScore`` from an ES hit ``_source``."""
    return IndexWithScore(
        id=doc_id,
        source_id=source.get("source_id", ""),
        source_type=SourceType(source.get("source_type", 0)),
        chunk_id=source.get("chunk_id", ""),
        knowledge_id=source.get("knowledge_id", ""),
        knowledge_base_id=source.get("knowledge_base_id", ""),
        tag_id=source.get("tag_id", ""),
        content=source.get("content", ""),
        score=score,
        match_type=match_type,
        is_enabled=source.get("is_enabled", False),
    )


def calculate_storage_size(doc: dict[str, Any]) -> int:
    """Estimate the storage size in bytes for a single ES document.

    Formula: content bytes + vector bytes + metadata overhead + index
    overhead (50% of content + vector).
    """
    content_size = len(doc.get("content", ""))
    vec = doc.get("embedding")
    vector_size = len(vec) * 4 if vec else 0
    index_overhead = (
        (content_size + vector_size) * _INDEX_OVERHEAD_NUMERATOR // _INDEX_OVERHEAD_DENOMINATOR
    )
    return content_size + vector_size + _METADATA_SIZE_BYTES + index_overhead


def build_base_conds(
    params: RetrieveParams, id_field_fn: Callable[[str], str]
) -> list[dict[str, Any]]:
    """Build the filter clause list for ES queries.

    Returns a list containing a single ``bool`` query with ``must``
    (positive filters) and ``must_not`` (negative filters). An empty
    filter list still includes the ``is_enabled != false`` exclusion.
    """
    must: list[dict[str, Any]] = []
    if params.knowledge_base_ids:
        must.append({"terms": {id_field_fn("knowledge_base_id"): list(params.knowledge_base_ids)}})
    if params.knowledge_ids:
        must.append({"terms": {id_field_fn("knowledge_id"): list(params.knowledge_ids)}})
    if params.tag_ids:
        must.append({"terms": {id_field_fn("tag_id"): list(params.tag_ids)}})

    must_not: list[dict[str, Any]] = [{"term": {"is_enabled": False}}]
    if params.exclude_knowledge_ids:
        must_not.append(
            {"terms": {id_field_fn("knowledge_id"): list(params.exclude_knowledge_ids)}}
        )
    if params.exclude_chunk_ids:
        must_not.append({"terms": {id_field_fn("chunk_id"): list(params.exclude_chunk_ids)}})
    return [{"bool": {"must": must, "must_not": must_not}}]


def parse_search_hits(
    response: dict[str, Any],
    match_type: MatchType,
) -> list[IndexWithScore]:
    """Extract ``IndexWithScore`` instances from an ES search response."""
    hits_obj = response.get("hits", {})
    hits = hits_obj.get("hits", [])
    results: list[IndexWithScore] = []
    for hit in hits:
        doc_id = hit.get("_id", "")
        score = hit.get("_score", 0.0)
        source = hit.get("_source", {})
        if not isinstance(source, dict):
            continue
        results.append(from_db_vector_embedding_with_score(doc_id, source, score, match_type))
    return results


__all__ = [
    "build_base_conds",
    "calculate_storage_size",
    "from_db_vector_embedding_with_score",
    "id_field",
    "parse_search_hits",
    "resolve_index_name",
    "to_db_vector_embedding",
]
