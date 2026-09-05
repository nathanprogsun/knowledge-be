"""KB-scoped initialization config handlers.

The SPA save-and-close path writes models and chunking through
``PUT /initialization/config/{kb_id}``. Persistence lives in the
knowledge-base service; this module only translates the wire body.
Route decorators stay on ``router.py`` so the feature-map scanner
sees them.
"""

from __future__ import annotations

from src.common.exception import NotFoundError, UnauthorizedError, ValidationError
from src.common.json import JsonObject
from src.core.infra.models.service.model_service import ModelService
from src.core.infra.models.types import ModelInfo
from src.core.infra.storage_backends.service.storage_backend_service import (
    StorageBackendService,
)
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.web.api.infra.initialization.kb_config_views import (
    KBConfigReadEnvelope,
    KBConfigUpdateEnvelope,
    KBModelConfigRequest,
    asr_from_request,
    chunking_from_request,
    config_read_payload,
    extract_from_request,
    question_generation_from_request,
    vlm_from_request,
)
from src.web.deps import (
    KBServiceDep,
    KnowledgeServiceDep,
    ModelServiceDep,
    StorageBackendServiceDep,
)


def _require_tenant(tenant_id: int) -> int:
    """Reject a missing workspace context."""
    if tenant_id == 0:
        raise UnauthorizedError(
            code="knowledge_base.tenant_context_missing",
            message="unauthorized: workspace context missing",
        )
    return tenant_id


async def _load_model(
    model_service: ModelService,
    *,
    tenant_id: int,
    model_id: str,
) -> ModelInfo | None:
    """Return the model, or ``None`` when the id is empty or unknown."""
    clean = model_id.strip()
    if not clean:
        return None
    try:
        return await model_service.get_model(tenant_id=tenant_id, model_id=clean)
    except NotFoundError:
        return None


async def get_kb_config(
    kb_id: str,
    kb_service: KBServiceDep,
    knowledge_service: KnowledgeServiceDep,
    model_service: ModelServiceDep,
    tenant_id: int,
) -> KBConfigReadEnvelope:
    """Return the knowledge base's current model and chunking slots."""
    tenant_id = _require_tenant(tenant_id)
    info = await kb_service.get_knowledge_base_by_id_and_tenant(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
    )
    has_files = (
        await knowledge_service.count_documents(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
        )
        > 0
    )
    models = await _load_bound_models(
        model_service, tenant_id=tenant_id, info_ids=_bound_model_ids(info)
    )
    return KBConfigReadEnvelope(
        success=True,
        data=config_read_payload(info, models=models, has_files=has_files),
    )


async def update_kb_config(
    kb_id: str,
    body: KBModelConfigRequest,
    kb_service: KBServiceDep,
    knowledge_service: KnowledgeServiceDep,
    model_service: ModelServiceDep,
    storage_service: StorageBackendServiceDep,
    tenant_id: int,
) -> KBConfigUpdateEnvelope:
    """Bind models and persist chunking / extract / storage on the KB."""
    tenant_id = _require_tenant(tenant_id)
    await kb_service.get_knowledge_base_by_id_and_tenant(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
    )
    await _require_model(
        model_service,
        tenant_id=tenant_id,
        model_id=body.llm_model_id,
        code="knowledge_base.llm_model_not_found",
        message="LLM model not found",
    )
    embedding_id = body.embedding_model_id.strip()
    if embedding_id:
        await _require_model(
            model_service,
            tenant_id=tenant_id,
            model_id=embedding_id,
            code="knowledge_base.embedding_model_not_found",
            message="Embedding model not found",
        )
    document_count = await knowledge_service.count_documents(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
    )
    await _persist_model_config(
        kb_service,
        model_service=model_service,
        storage_service=storage_service,
        kb_id=kb_id,
        tenant_id=tenant_id,
        body=body,
        embedding_id=embedding_id,
        document_count=document_count,
    )
    return KBConfigUpdateEnvelope(success=True, message="配置更新成功")


async def _persist_model_config(
    kb_service: KBService,
    *,
    model_service: ModelService,
    storage_service: StorageBackendService,
    kb_id: str,
    tenant_id: int,
    body: KBModelConfigRequest,
    embedding_id: str,
    document_count: int,
) -> None:
    """Write the translated payload after models have been checked."""
    storage_backend_id, storage_provider_config = await _resolve_storage(
        storage_service,
        tenant_id=tenant_id,
        backend_id=body.storage_backend_id,
        provider=body.storage_provider,
    )
    vlm_ok = await _optional_model_ok(
        model_service,
        tenant_id=tenant_id,
        model_id=body.vlm_config.model_id if body.vlm_config else "",
    )
    asr_ok = await _optional_model_ok(
        model_service,
        tenant_id=tenant_id,
        model_id=body.asr_config.model_id if body.asr_config else "",
    )
    await kb_service.update_model_config(
        knowledge_base_id=kb_id,
        summary_model_id=body.llm_model_id,
        embedding_model_id=embedding_id or None,
        chunking_config=chunking_from_request(body),
        vlm_config=vlm_from_request(body, model_ok=vlm_ok),
        asr_config=asr_from_request(body, model_ok=asr_ok),
        extract_config=extract_from_request(body),
        question_generation_config=question_generation_from_request(body),
        storage_backend_id=storage_backend_id,
        storage_provider_config=storage_provider_config,
        document_count=document_count,
    )


def _bound_model_ids(info: KnowledgeBaseInfo) -> list[str]:
    """Collect non-empty model ids from the KB projection."""
    ids: list[str] = []
    if info.summary_model_id:
        ids.append(info.summary_model_id)
    if info.embedding_model_id:
        ids.append(info.embedding_model_id)
    return ids


async def _load_bound_models(
    model_service: ModelService,
    *,
    tenant_id: int,
    info_ids: list[str],
) -> list[ModelInfo]:
    """Load every bound model, skipping ids that no longer resolve."""
    loaded: list[ModelInfo] = []
    for model_id in info_ids:
        model = await _load_model(model_service, tenant_id=tenant_id, model_id=model_id)
        if model is not None:
            loaded.append(model)
    return loaded


async def _require_model(
    model_service: ModelService,
    *,
    tenant_id: int,
    model_id: str,
    code: str,
    message: str,
) -> ModelInfo:
    """Load a required model or raise ``ValidationError``."""
    model = await _load_model(model_service, tenant_id=tenant_id, model_id=model_id)
    if model is None:
        raise ValidationError(code=code, message=message)
    return model


async def _optional_model_ok(
    model_service: ModelService,
    *,
    tenant_id: int,
    model_id: str,
) -> bool:
    """True when the optional model id is empty or resolves."""
    if not model_id.strip():
        return False
    return (await _load_model(model_service, tenant_id=tenant_id, model_id=model_id)) is not None


async def _resolve_storage(
    storage_service: StorageBackendService,
    *,
    tenant_id: int,
    backend_id: str,
    provider: str,
) -> tuple[str | None, JsonObject]:
    """Resolve an explicit backend id; otherwise keep the provider projection."""
    clean_id = backend_id.strip()
    clean_provider = provider.strip().lower() or "local"
    if clean_id:
        backend = await storage_service.resolve_backend(
            tenant_id=tenant_id,
            backend_id=clean_id,
            provider="",
        )
        if backend is None:
            raise ValidationError(
                code="knowledge_base.storage_backend_unavailable",
                message="Storage backend is unavailable",
            )
        return backend.id, {"provider": backend.provider}
    return None, {"provider": clean_provider}


__all__ = ["get_kb_config", "update_kb_config"]
