"""Apply the SPA's KB model-config payload onto a knowledge-base row.

The editor's save-and-close path sends model ids, chunking, extract,
question-generation, and storage fields. This module turns that payload
into a repository patch. Callers validate model ids and document count
before invoking ``apply_model_config``.
"""

from __future__ import annotations

from src.common.exception import ValidationError
from src.common.json import JsonObject
from src.db.models.knowledge_base import KnowledgeBase


def apply_model_config(
    existing: KnowledgeBase,
    *,
    summary_model_id: str,
    embedding_model_id: str | None,
    chunking_config: JsonObject | None,
    vlm_config: JsonObject | None,
    asr_config: JsonObject | None,
    extract_config: JsonObject | None,
    question_generation_config: JsonObject | None,
    storage_backend_id: str | None,
    storage_provider_config: JsonObject | None,
    document_count: int,
) -> dict[str, JsonObject | str | None]:
    """Build the mutable-column patch for a model-config save.

    Changing the embedding model is refused once the knowledge base
    already has documents — existing vectors would become unreadable.
    """
    _reject_embedding_change(
        existing=existing,
        embedding_model_id=embedding_model_id,
        document_count=document_count,
    )
    patch: dict[str, JsonObject | str | None] = {
        "summary_model_id": summary_model_id,
    }
    if embedding_model_id:
        patch["embedding_model_id"] = embedding_model_id
    if chunking_config is not None:
        patch["chunking_config"] = chunking_config
    if vlm_config is not None:
        patch["vlm_config"] = vlm_config
    if asr_config is not None:
        patch["asr_config"] = asr_config
    if extract_config is not None:
        patch["extract_config"] = extract_config
    if question_generation_config is not None:
        patch["question_generation_config"] = question_generation_config
    if storage_backend_id is not None:
        patch["storage_backend_id"] = storage_backend_id or None
    if storage_provider_config is not None:
        patch["storage_provider_config"] = storage_provider_config
    return patch


def _reject_embedding_change(
    *,
    existing: KnowledgeBase,
    embedding_model_id: str | None,
    document_count: int,
) -> None:
    """Refuse an embedding-model swap when documents already exist."""
    incoming = (embedding_model_id or "").strip()
    current = (existing.embedding_model_id or "").strip()
    if current and incoming and current != incoming and document_count > 0:
        raise ValidationError(
            code="knowledge_base.embedding_model_locked",
            message="Cannot change the embedding model after documents exist",
        )


__all__ = ["apply_model_config"]
