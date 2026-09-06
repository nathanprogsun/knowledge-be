"""Wire-shape conversion for the knowledge-base endpoints.

``KnowledgeBaseInfo`` is the service-side projection of a
``knowledge_bases`` row; the wire shape is the frozen ``KnowledgeBase``
in ``src/core/contracts/knowledge.py``. ``knowledge_base_to_contract``
performs the boundary translation, re-emitting the row onto the wire
contract with the JSON config blobs parsed onto their typed models.

Response-only enrichment fields without a backing service in this layer
(``creator_name``, the ``vector_store_*`` display block,
``my_permission``) are emitted as ``None``; the frozen contract types
them as nullable.
"""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from src.common.json import JsonObject
from src.core.contracts.knowledge import (
    ASRConfig,
    ChunkingConfig,
    ExtractConfig,
    FAQConfig,
    ImageProcessingConfig,
    IndexingStrategy,
    KnowledgeBase,
    KnowledgeCopyResponse,
    KnowledgeDuplicateResponse,
    KnowledgeSearchHit,
    LegacyStorageConfig,
    QuestionGenerationConfig,
    StorageProviderConfig,
    VLMConfig,
    WikiConfig,
)
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo

_Parseable = TypeVar("_Parseable", bound=BaseModel)


class KnowledgeBaseEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` - single-KB responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: KnowledgeBase


class KnowledgeBaseListEnvelope(BaseModel):
    """``{"success": true, "data": [...]}`` - list responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: list[KnowledgeBase]


class DeleteKnowledgeBaseResponse(BaseModel):
    """``{"success": true, "message": "..."}`` - delete acknowledgement."""

    model_config = ConfigDict(frozen=True)

    success: bool
    message: str


class KnowledgeCopyEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` - copy acknowledgement."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: KnowledgeCopyResponse


class KnowledgeDuplicateEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` - duplicate acknowledgement."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: KnowledgeDuplicateResponse


class KnowledgeBasePinData(BaseModel):
    """Inner payload of the pin toggle."""

    model_config = ConfigDict(frozen=True)

    is_pinned: bool


class KnowledgeBasePinEnvelope(BaseModel):
    """``{"success": true, "data": {"is_pinned": ...}}``."""

    model_config = ConfigDict(frozen=True)

    success: bool = True
    data: KnowledgeBasePinData


class HybridSearchEnvelope(BaseModel):
    """``{"success": true, "data": [...]}`` - hybrid-search responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: list[KnowledgeSearchHit]


def _parse_optional(
    model: type[_Parseable],
    raw: JsonObject | None,
) -> _Parseable | None:
    """Parse a JSON config blob onto its typed contract, leniently.

    A stored blob whose field set does not match the contract yields
    ``None`` rather than failing the whole response.
    """
    if raw is None:
        return None
    try:
        return model.model_validate(raw)
    except ValidationError:
        return None


def knowledge_base_to_contract(info: KnowledgeBaseInfo) -> KnowledgeBase:
    """Project the service DTO onto the frozen wire contract.

    JSON config columns are parsed onto their typed models; enrichment
    fields without a backing service in this layer stay ``None``.
    """
    return KnowledgeBase(
        id=info.id,
        name=info.name,
        description=info.description,
        type=info.type,
        is_temporary=info.is_temporary,
        tenant_id=info.tenant_id,
        creator_id=info.creator_id,
        creator_name=None,
        chunking_config=_parse_optional(ChunkingConfig, info.chunking_config),
        image_processing_config=_parse_optional(
            ImageProcessingConfig, info.image_processing_config
        ),
        embedding_model_id=info.embedding_model_id,
        summary_model_id=info.summary_model_id,
        vlm_config=_parse_optional(VLMConfig, info.vlm_config),
        asr_config=_parse_optional(ASRConfig, info.asr_config),
        storage_provider_config=_parse_optional(
            StorageProviderConfig, info.storage_provider_config
        ),
        storage_backend_id=info.storage_backend_id,
        storage_config=_parse_optional(LegacyStorageConfig, info.storage_config),
        extract_config=_parse_optional(ExtractConfig, info.extract_config),
        faq_config=_parse_optional(FAQConfig, info.faq_config),
        question_generation_config=_parse_optional(
            QuestionGenerationConfig, info.question_generation_config
        ),
        wiki_config=_parse_optional(WikiConfig, info.wiki_config),
        indexing_strategy=_parse_optional(IndexingStrategy, info.indexing_strategy),
        vector_store_id=info.vector_store_id,
        vector_store_name=None,
        vector_store_source=None,
        vector_store_engine_type=None,
        vector_store_status=None,
        is_pinned=info.is_pinned,
        pinned_at=info.pinned_at,
        knowledge_count=info.knowledge_count,
        chunk_count=info.chunk_count,
        processing_count=info.processing_count,
        is_processing=info.is_processing,
        share_count=info.share_count,
        my_permission=None,
        created_at=info.created_at,
        updated_at=info.updated_at,
        deleted_at=info.deleted_at,
    )


def knowledge_base_envelope(info: KnowledgeBaseInfo) -> KnowledgeBaseEnvelope:
    """Wrap one knowledge base in the success envelope."""
    return KnowledgeBaseEnvelope(success=True, data=knowledge_base_to_contract(info))


def knowledge_base_list_envelope(
    infos: list[KnowledgeBaseInfo],
) -> KnowledgeBaseListEnvelope:
    """Wrap a list of knowledge bases in the success envelope."""
    return KnowledgeBaseListEnvelope(
        success=True,
        data=[knowledge_base_to_contract(info) for info in infos],
    )


__all__ = [
    "DeleteKnowledgeBaseResponse",
    "HybridSearchEnvelope",
    "KnowledgeBaseEnvelope",
    "KnowledgeBaseListEnvelope",
    "KnowledgeBasePinData",
    "KnowledgeBasePinEnvelope",
    "KnowledgeCopyEnvelope",
    "KnowledgeDuplicateEnvelope",
    "knowledge_base_envelope",
    "knowledge_base_list_envelope",
    "knowledge_base_to_contract",
]
