"""Unit tests for ``apply_model_config``.

The helper builds the repository patch for the editor's save-and-close
path. Embedding-model swaps are refused once documents exist so existing
vectors stay readable.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.common.exception import ValidationError
from src.core.knowledge.knowledge_bases.service.kb_model_config import (
    apply_model_config,
)
from src.db.models.knowledge_base import KnowledgeBase

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _row(*, embedding_model_id: str = "") -> KnowledgeBase:
    return KnowledgeBase(
        id="kb-1",
        name="docs",
        tenant_id=7,
        embedding_model_id=embedding_model_id,
        created_at=_NOW,
        updated_at=_NOW,
    )


def test_patch_includes_models_and_chunking() -> None:
    patch = apply_model_config(
        _row(),
        summary_model_id="llm-1",
        embedding_model_id="emb-1",
        chunking_config={"chunk_size": 256},
        vlm_config=None,
        asr_config=None,
        extract_config=None,
        question_generation_config=None,
        storage_backend_id=None,
        storage_provider_config={"provider": "local"},
        document_count=0,
    )

    assert patch["summary_model_id"] == "llm-1"
    assert patch["embedding_model_id"] == "emb-1"
    assert patch["chunking_config"] == {"chunk_size": 256}
    assert patch["storage_provider_config"] == {"provider": "local"}
    assert "vlm_config" not in patch


def test_refuses_embedding_change_after_documents() -> None:
    with pytest.raises(ValidationError) as excinfo:
        apply_model_config(
            _row(embedding_model_id="emb-old"),
            summary_model_id="llm-1",
            embedding_model_id="emb-new",
            chunking_config=None,
            vlm_config=None,
            asr_config=None,
            extract_config=None,
            question_generation_config=None,
            storage_backend_id=None,
            storage_provider_config=None,
            document_count=1,
        )

    assert excinfo.value.code == "knowledge_base.embedding_model_locked"


def test_same_embedding_is_allowed_with_documents() -> None:
    patch = apply_model_config(
        _row(embedding_model_id="emb-1"),
        summary_model_id="llm-1",
        embedding_model_id="emb-1",
        chunking_config={"chunk_size": 128},
        vlm_config=None,
        asr_config=None,
        extract_config=None,
        question_generation_config=None,
        storage_backend_id=None,
        storage_provider_config=None,
        document_count=4,
    )

    assert patch["embedding_model_id"] == "emb-1"
    assert patch["chunking_config"] == {"chunk_size": 128}
