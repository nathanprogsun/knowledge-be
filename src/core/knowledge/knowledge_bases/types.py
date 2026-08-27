"""Internal DTOs and constants for the knowledge-base domain.

The wire projections live in ``src/core/contracts/knowledge.py``
(frozen); this module carries the service-side carrier and the
type / enum constants that mirror the upstream contract.

``KnowledgeBaseInfo`` is the service-output projection of a
``knowledge_bases`` row. The count fields (``knowledge_count``,
``chunk_count``, ``is_processing``, ``processing_count``,
``share_count``) and the per-caller pin fields (``is_pinned``,
``pinned_at``) are not stored — the service fills them per query. The
legacy ``cos_config`` column surfaces under its wire name
``storage_config``.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict

from src.common.json import JsonObject
from src.db.models.knowledge_base import KnowledgeBase

# ── Knowledge-base type ──────────────────────────────────────────────

KNOWLEDGE_BASE_TYPE_DOCUMENT = "document"
KNOWLEDGE_BASE_TYPE_FAQ = "faq"
KNOWLEDGE_BASE_TYPE_WIKI = "wiki"

KNOWLEDGE_BASE_TYPES: frozenset[str] = frozenset(
    {
        KNOWLEDGE_BASE_TYPE_DOCUMENT,
        KNOWLEDGE_BASE_TYPE_FAQ,
        KNOWLEDGE_BASE_TYPE_WIKI,
    }
)

# ── FAQ index modes ──────────────────────────────────────────────────

FAQ_INDEX_MODE_QUESTION_ONLY = "question_only"
FAQ_INDEX_MODE_QUESTION_ANSWER = "question_answer"

# ── FAQ question index modes ─────────────────────────────────────────

FAQ_QUESTION_INDEX_MODE_COMBINED = "combined"
FAQ_QUESTION_INDEX_MODE_SEPARATE = "separate"


def _decode_json(raw: JsonObject | str | None) -> JsonObject | None:
    """Decode a JSON-backed column.

    Accepts both a parsed ``dict`` and a raw JSON string (some dialects
    persist JSON columns as text). ``None`` / empty / unparseable input
    yields ``None``.
    """
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return decoded if isinstance(decoded, dict) else None
    return raw


class KnowledgeBaseInfo(BaseModel):
    """Service-side projection of a ``knowledge_bases`` row.

    Mirrors the domain entity: every stored column plus the response-only
    enrichments the service computes per query. JSON config blobs are
    carried as raw ``JsonObject`` values — typed parsing is a service
    concern.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    type: str = KNOWLEDGE_BASE_TYPE_DOCUMENT
    is_temporary: bool = False
    description: str | None = None
    tenant_id: int
    creator_id: str | None = None
    chunking_config: JsonObject | None = None
    image_processing_config: JsonObject | None = None
    embedding_model_id: str = ""
    summary_model_id: str = ""
    vlm_config: JsonObject | None = None
    asr_config: JsonObject | None = None
    storage_provider_config: JsonObject | None = None
    storage_backend_id: str | None = None
    storage_config: JsonObject | None = None
    vector_store_id: str | None = None
    extract_config: JsonObject | None = None
    faq_config: JsonObject | None = None
    question_generation_config: JsonObject | None = None
    wiki_config: JsonObject | None = None
    indexing_strategy: JsonObject | None = None
    is_pinned: bool = False
    pinned_at: datetime | None = None
    knowledge_count: int = 0
    chunk_count: int = 0
    is_processing: bool = False
    processing_count: int = 0
    share_count: int = 0
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None

    @classmethod
    def map_from_db(cls, db: KnowledgeBase) -> Self:
        """Project a storage row onto the service shape.

        JSON columns are decoded leniently (a stored text blob or a
        parsed dict); the legacy ``cos_config`` column is surfaced under
        its wire name ``storage_config``. Response-only fields keep
        their zero defaults — the service fills them per query.
        """
        record = db.model_dump()
        for column in (
            "chunking_config",
            "image_processing_config",
            "vlm_config",
            "asr_config",
            "storage_provider_config",
            "extract_config",
            "faq_config",
            "question_generation_config",
            "wiki_config",
            "indexing_strategy",
        ):
            record[column] = _decode_json(record.get(column))
        record["storage_config"] = _decode_json(record.pop("cos_config"))
        return cls.model_validate(record)

    @classmethod
    def from_json(cls, raw: JsonObject | str | None) -> JsonObject | None:
        """Decode a single JSON column. Aliased to ``_decode_json``."""
        return _decode_json(raw)


__all__ = [
    "FAQ_INDEX_MODE_QUESTION_ANSWER",
    "FAQ_INDEX_MODE_QUESTION_ONLY",
    "FAQ_QUESTION_INDEX_MODE_COMBINED",
    "FAQ_QUESTION_INDEX_MODE_SEPARATE",
    "KNOWLEDGE_BASE_TYPES",
    "KNOWLEDGE_BASE_TYPE_DOCUMENT",
    "KNOWLEDGE_BASE_TYPE_FAQ",
    "KNOWLEDGE_BASE_TYPE_WIKI",
    "KnowledgeBaseInfo",
]
