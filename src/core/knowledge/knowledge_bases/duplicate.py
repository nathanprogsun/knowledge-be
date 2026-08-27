"""Knowledge-base duplicate — standalone extension of the knowledge-base service.

``duplicate_kb`` creates a new knowledge base from a source's settings
only. Runtime/content state is deliberately reset, so knowledge entries,
chunks, FAQ content, wiki pages, indexes, shares and pins are never
copied. The new name is the source name plus a locale-aware suffix,
deduplicated against the workspace's existing knowledge bases.

The module is a thin composition over ``KBService``; it owns no session.
The web layer composes the pieces, mirroring the upstream service contract.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.common.exception import ValidationError
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo

_EMPTY_ID_CODE = "knowledge_base.id_required"


def _duplicate_suffix(locale: str) -> str:
    """Locale-aware "copy" suffix appended to the duplicate's name."""
    code = (locale or "").strip().lower()
    if code.startswith("zh"):
        return " 副本"
    if code.startswith("ko"):
        return " 사본"
    if code.startswith("ru"):
        return " копия"
    return " Copy"


def _default_base_name(locale: str) -> str:
    """Fallback base name when the source has no usable name."""
    code = (locale or "").strip().lower()
    if code.startswith("zh"):
        return "知识库"
    if code.startswith("ko"):
        return "지식베이스"
    if code.startswith("ru"):
        return "База знаний"
    return "Knowledge Base"


async def _build_duplicate_name(
    *,
    service: KBService,
    tenant_id: int,
    source_name: str,
    locale: str,
) -> str:
    """Build a non-colliding name: base + suffix, then numbered variants.

    The existing-name probe is best-effort: a failing listing falls back
    to the bare suffixed name, matching the upstream resilience.
    """
    base = source_name.strip()
    if not base:
        base = _default_base_name(locale)
    suffix = _duplicate_suffix(locale)
    try:
        existing = {info.name for info in await service.list_knowledge_bases(tenant_id=tenant_id)}
    except Exception:
        return base + suffix
    candidate = base + suffix
    if candidate not in existing:
        return candidate
    counter = 2
    while True:
        candidate = f"{base}{suffix} {counter}"
        if candidate not in existing:
            return candidate
        counter += 1


async def duplicate_kb(
    *,
    service: KBService,
    session: AsyncSession,
    tenant_id: int,
    source_kb_id: str,
    creator_id: str | None = None,
    locale: str = "en",
) -> KnowledgeBaseInfo:
    """Duplicate a knowledge base from its settings only.

    ``session`` is the caller's shared session; it is reserved for the
    deferred audit seam and kept for signature consistency with ``copy_kb``.
    """
    source_kb_id = source_kb_id.strip()
    if not source_kb_id:
        raise ValidationError(
            code=_EMPTY_ID_CODE,
            message="source knowledge base ID cannot be empty",
        )
    source = await service.get_knowledge_base_by_id_and_tenant(
        tenant_id=tenant_id,
        knowledge_base_id=source_kb_id,
    )
    name = await _build_duplicate_name(
        service=service,
        tenant_id=tenant_id,
        source_name=source.name,
        locale=locale,
    )
    return await service.create_knowledge_base(
        tenant_id=tenant_id,
        name=name,
        kb_type=source.type,
        description=source.description,
        creator_id=creator_id,
        is_temporary=False,
        chunking_config=source.chunking_config,
        image_processing_config=source.image_processing_config,
        embedding_model_id=source.embedding_model_id,
        summary_model_id=source.summary_model_id,
        vlm_config=source.vlm_config,
        asr_config=source.asr_config,
        storage_provider_config=source.storage_provider_config,
        storage_backend_id=source.storage_backend_id,
        storage_config=source.storage_config,
        extract_config=source.extract_config,
        faq_config=source.faq_config,
        question_generation_config=source.question_generation_config,
        wiki_config=source.wiki_config,
        indexing_strategy=source.indexing_strategy,
        vector_store_id=source.vector_store_id,
    )


__all__ = ["duplicate_kb"]
