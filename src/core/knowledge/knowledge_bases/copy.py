"""Knowledge-base copy — standalone extension of the knowledge-base service.

``copy_kb`` clones a knowledge base either into an existing target (a
settings-level clone that leaves the target row untouched) or into a
freshly created knowledge base whose configuration mirrors the source.
Cross-tenant access is rejected: a source or target owned by another
workspace reads as absent and surfaces as ``NotFoundError``.

The module is a thin composition over ``KBService`` plus a lenient read
of the workspace's storage defaults; it owns no session. The web layer
composes the pieces, mirroring the upstream service contract.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.common.exception import ValidationError
from src.common.json import JsonObject
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.db.dao.tenants_repository import TenantRepository

_EMPTY_ID_CODE = "knowledge_base.id_required"

# Sentinel provider written before a concrete storage instance is selected.
_PENDING_PROVIDER = "__pending_env__"


def _require_knowledge_base_id(knowledge_base_id: str) -> None:
    """Reject an empty knowledge-base id."""
    if not knowledge_base_id.strip():
        raise ValidationError(
            code=_EMPTY_ID_CODE,
            message="knowledge base ID cannot be empty",
        )


def _normalise_vector_store_id(vector_store_id: str | None) -> str | None:
    """Fold an empty-string binding into ``None`` for comparison."""
    return vector_store_id if vector_store_id else None


def _shares_vector_store(left: str | None, right: str | None) -> bool:
    """Whether two bindings refer to the same vector store.

    ``None`` and the empty string are the same "no binding" signal; two
    nil bindings compare equal (both fall back to the workspace default
    engines). One nil and one concrete binding never match.
    """
    a = _normalise_vector_store_id(left)
    b = _normalise_vector_store_id(right)
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return a == b


def _require_same_embedding_model(
    source: KnowledgeBaseInfo,
    target: KnowledgeBaseInfo,
) -> None:
    """Reject a clone whose source and target use different embeddings.

    Mixing incompatible vector spaces would silently corrupt search
    results, so the bindings must match exactly.
    """
    if source.embedding_model_id != target.embedding_model_id:
        raise ValidationError(
            code="knowledge_base.copy_embedding_mismatch",
            message=(
                "source and target knowledge bases use different embedding models; "
                "clone into a target with the same embedding model"
            ),
        )


def _require_same_vector_store(
    source: KnowledgeBaseInfo,
    target: KnowledgeBaseInfo,
) -> None:
    """Reject a clone across vector stores.

    Cross-store cloning would require copying physical vector data
    between stores, which is not supported.
    """
    if not _shares_vector_store(source.vector_store_id, target.vector_store_id):
        raise ValidationError(
            code="knowledge_base.copy_vector_store_mismatch",
            message=(
                "source and target knowledge bases are bound to different vector stores; "
                "cross-store cloning is not yet supported"
            ),
        )


async def _load_tenant_storage_defaults(
    session: AsyncSession,
    tenant_id: int,
) -> tuple[str | None, str] | None:
    """Read the workspace's storage defaults, or ``None`` when absent.

    The storage-instance defense is gated on workspace configuration
    being resolvable — a missing row means the check is skipped,
    mirroring the upstream optional-tenant guard.
    """
    tenant = await TenantRepository(session).find_by_primary_key({"id": tenant_id})
    if tenant is None:
        return None
    provider = ""
    config = tenant.storage_engine_config
    if isinstance(config, dict):
        raw = config.get("default_provider")
        provider = str(raw).strip().lower() if raw else ""
    return tenant.default_storage_backend_id, provider


def _provider_from_config(config: JsonObject | None) -> str:
    """Normalised provider from the modern storage-provider config.

    The ``__pending_env__`` sentinel is not a real binding and is ignored.
    """
    if not isinstance(config, dict):
        return ""
    raw = config.get("provider")
    provider = str(raw).strip().lower() if raw else ""
    if provider and provider != _PENDING_PROVIDER:
        return provider
    return ""


def _legacy_storage_provider(info: KnowledgeBaseInfo) -> str:
    """Normalised provider from the legacy storage config column."""
    config = info.storage_config
    if not isinstance(config, dict):
        return ""
    raw = config.get("provider")
    return str(raw).strip().lower() if raw else ""


def _effective_storage_provider(info: KnowledgeBaseInfo, default_provider: str) -> str:
    """Resolve the effective provider, falling back to the workspace default."""
    provider = _provider_from_config(info.storage_provider_config)
    if provider:
        return provider
    legacy = _legacy_storage_provider(info)
    if legacy:
        return legacy
    return (default_provider or "").strip().lower()


def _shares_storage_backend(
    source: KnowledgeBaseInfo,
    target: KnowledgeBaseInfo,
    default_backend_id: str | None,
    default_provider: str,
) -> bool:
    """Whether two KBs resolve to the same concrete storage instance.

    Concrete backend ids win; when neither pins one, the effective
    providers are compared. Comparing only provider names would
    incorrectly allow clones between distinct instances of the same
    provider.
    """

    def _effective_backend_id(info: KnowledgeBaseInfo) -> str:
        backend_id = (info.storage_backend_id or "").strip()
        return backend_id or (default_backend_id or "").strip()

    left_id = _effective_backend_id(source)
    right_id = _effective_backend_id(target)
    if left_id or right_id:
        return bool(left_id) and left_id == right_id
    return _effective_storage_provider(source, default_provider) == _effective_storage_provider(
        target, default_provider
    )


async def _require_same_storage_instance(
    session: AsyncSession,
    tenant_id: int,
    source: KnowledgeBaseInfo,
    target: KnowledgeBaseInfo,
) -> None:
    """Reject a clone across storage instances when workspace config resolves."""
    defaults = await _load_tenant_storage_defaults(session, tenant_id)
    if defaults is None:
        return
    default_backend_id, default_provider = defaults
    if not _shares_storage_backend(source, target, default_backend_id, default_provider):
        raise ValidationError(
            code="knowledge_base.copy_storage_mismatch",
            message=(
                "source and target knowledge bases use different storage instances; "
                "cross-storage-backend cloning is not supported"
            ),
        )


async def _create_from_source(
    *,
    service: KBService,
    tenant_id: int,
    source: KnowledgeBaseInfo,
    creator_id: str | None,
) -> KnowledgeBaseInfo:
    """Create a new knowledge base carrying the source's settings.

    The new row is owned by the caller's tenant and shares the source's
    physical vector store so both land on the same index.
    """
    return await service.create_knowledge_base(
        tenant_id=tenant_id,
        name=source.name,
        kb_type=source.type,
        description=source.description,
        creator_id=creator_id,
        chunking_config=source.chunking_config,
        image_processing_config=source.image_processing_config,
        embedding_model_id=source.embedding_model_id,
        summary_model_id=source.summary_model_id,
        vlm_config=source.vlm_config,
        storage_provider_config=source.storage_provider_config,
        storage_backend_id=source.storage_backend_id,
        storage_config=source.storage_config,
        faq_config=source.faq_config,
        vector_store_id=source.vector_store_id,
    )


async def copy_kb(
    *,
    service: KBService,
    session: AsyncSession,
    tenant_id: int,
    source_kb_id: str,
    target_kb_id: str | None = None,
    creator_id: str | None = None,
) -> tuple[KnowledgeBaseInfo, KnowledgeBaseInfo]:
    """Copy a knowledge base, returning ``(source, target)``.

    With ``target_kb_id`` the clone is settings-level only: the target
    row is left untouched and no content is copied. The source and target
    must share the same embedding model, vector-store binding and storage
    instance, or a ``ValidationError`` is raised. Without ``target_kb_id``
    a new knowledge base is created from the source's configuration.
    """
    _require_knowledge_base_id(source_kb_id)
    source = await service.get_knowledge_base_by_id_and_tenant(
        tenant_id=tenant_id,
        knowledge_base_id=source_kb_id,
    )
    if target_kb_id is not None and target_kb_id.strip():
        target = await service.get_knowledge_base_by_id_and_tenant(
            tenant_id=tenant_id,
            knowledge_base_id=target_kb_id,
        )
        _require_same_embedding_model(source, target)
        _require_same_vector_store(source, target)
        await _require_same_storage_instance(session, tenant_id, source, target)
        return source, target
    target = await _create_from_source(
        service=service,
        tenant_id=tenant_id,
        source=source,
        creator_id=creator_id,
    )
    return source, target


__all__ = ["copy_kb"]
