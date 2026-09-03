"""Knowledge-base service — CRUD and aggregate counts.

Request-scoped: constructed per request by ``factory.build_kb_service``
with a fresh repository on the shared ``AsyncSession``; the web layer
never imports ``db`` directly. The methods mirror the upstream service
contract for the knowledge-base entity, including the per-query count
enrichment the list path applies.

Scope of this module
--------------------

- ``create_knowledge_base`` stamps id / tenant / timestamps, applies the
  type-specific config defaults, normalises the vector-store binding and
  inserts the row.
- Reads: single (by id / by id-and-tenant), batch, and tenant list with
  best-effort count enrichment.
- ``update_knowledge_base`` applies the mutable fields and the config
  merge; ``delete_knowledge_base`` soft-deletes the row.
- The three aggregate counts (``count_documents`` / ``count_chunks`` /
  ``count_members``) back the non-persisted response fields.

Deferred seams (neutral wording): storage-backend resolution against the
tenant default and the engine registry, vector-store binding validation,
audit recording, and the async heavy-cleanup task that a delete would
enqueue. Those land with the storage / worker / sharing domains.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from src.common.exception import NotFoundError, ValidationError
from src.common.json import JsonObject, JsonValue
from src.core.knowledge.knowledge_bases.types import (
    FAQ_INDEX_MODE_QUESTION_ANSWER,
    FAQ_QUESTION_INDEX_MODE_COMBINED,
    KNOWLEDGE_BASE_TYPE_DOCUMENT,
    KNOWLEDGE_BASE_TYPE_FAQ,
    KNOWLEDGE_BASE_TYPES,
    KnowledgeBaseInfo,
)
from src.db.dao.knowledge_base_repository import KnowledgeBaseRepository
from src.db.models.knowledge_base import KnowledgeBase

_NOT_FOUND_CODE = "knowledge_base.not_found"

# Default indexing strategy for a freshly created knowledge base (vector
# + keyword on, wiki + graph off).
_DEFAULT_INDEXING_STRATEGY: JsonObject = {
    "vector_enabled": True,
    "keyword_enabled": True,
    "wiki_enabled": False,
    "graph_enabled": False,
}

# FAQ-type defaults applied when a FAQ row carries no FAQ config.
_DEFAULT_FAQ_CONFIG: JsonObject = {
    "index_mode": FAQ_INDEX_MODE_QUESTION_ANSWER,
    "question_index_mode": FAQ_QUESTION_INDEX_MODE_COMBINED,
}

# Keys of the indexing strategy JSON; "any enabled" gates the update path.
_INDEXING_KEYS: tuple[str, ...] = (
    "vector_enabled",
    "keyword_enabled",
    "wiki_enabled",
    "graph_enabled",
)


def _now() -> datetime:
    """Return a timezone-aware ``now`` for stamping rows."""
    return datetime.now(UTC)


def _new_id() -> str:
    """Generate a UUID for a freshly created knowledge base."""
    return str(uuid.uuid4())


# ── Boundary validators ──────────────────────────────────────────────


def _require_tenant_id(tenant_id: int) -> None:
    """Reject a non-positive tenant id."""
    if not isinstance(tenant_id, int) or tenant_id <= 0:
        raise ValidationError(
            code="knowledge_base.tenant_required",
            message="tenant ID is required",
        )


def _require_knowledge_base_id(knowledge_base_id: str) -> None:
    """Reject an empty knowledge-base id."""
    if not knowledge_base_id.strip():
        raise ValidationError(
            code="knowledge_base.id_required",
            message="knowledge base ID cannot be empty",
        )


def _require_name(name: str) -> str:
    """Strip and validate a non-empty name; return the cleaned value."""
    clean = name.strip()
    if not clean:
        raise ValidationError(
            code="knowledge_base.name_required",
            message="name is required",
        )
    return clean


def _require_kb_type(kb_type: str) -> None:
    """Reject a knowledge-base type outside the known set."""
    if kb_type not in KNOWLEDGE_BASE_TYPES:
        raise ValidationError(
            code="knowledge_base.type_invalid",
            message=f"unsupported knowledge base type: {kb_type}",
        )


# ── Config / default helpers ─────────────────────────────────────────


def _is_true(value: JsonValue) -> bool:
    """True only for an explicit JSON boolean ``true``."""
    return isinstance(value, bool) and value


def _has_any_indexing(strategy: JsonObject) -> bool:
    """Whether any indexing pipeline is enabled in the strategy."""
    return any(_is_true(strategy.get(key)) for key in _INDEXING_KEYS)


def _normalise_vector_store_id(vector_store_id: str | None) -> str | None:
    """Fold an empty-string binding into ``None``.

    Mirrors the upstream ``Normalize``: the retrieve-engine factory's
    precondition treats ``None`` as "use the tenant's effective engines",
    so an empty string must never reach the storage layer.
    """
    if vector_store_id is not None and vector_store_id == "":
        return None
    return vector_store_id


def _ensure_defaults(row: KnowledgeBase) -> KnowledgeBase:
    """Apply the type / config defaults, returning a new row.

    Non-FAQ rows drop their FAQ config; FAQ rows get the default index
    modes; a missing or all-disabled indexing strategy falls back to the
    default, and an enabled graph extraction syncs ``graph_enabled`` onto
    the strategy. The input row is never mutated.
    """
    updates: dict[str, JsonValue] = {}
    kb_type = row.type if row.type else KNOWLEDGE_BASE_TYPE_DOCUMENT
    if row.type != kb_type:
        updates["type"] = kb_type
    if kb_type != KNOWLEDGE_BASE_TYPE_FAQ:
        if row.faq_config is not None:
            updates["faq_config"] = None
    else:
        faq = dict(row.faq_config or {})
        if not faq.get("index_mode"):
            faq["index_mode"] = FAQ_INDEX_MODE_QUESTION_ANSWER
        if not faq.get("question_index_mode"):
            faq["question_index_mode"] = FAQ_QUESTION_INDEX_MODE_COMBINED
        updates["faq_config"] = faq
    strategy = dict(row.indexing_strategy) if row.indexing_strategy else None
    if strategy is None or not _has_any_indexing(strategy):
        strategy = dict(_DEFAULT_INDEXING_STRATEGY)
        updates["indexing_strategy"] = strategy
    extract = row.extract_config
    if (
        isinstance(extract, dict)
        and _is_true(extract.get("enabled"))
        and not _is_true(strategy.get("graph_enabled"))
    ):
        strategy = {**strategy, "graph_enabled": True}
        updates["indexing_strategy"] = strategy
    if not updates:
        return row
    return row.model_copy(update=updates)


def _as_json_or_none(value: JsonValue) -> JsonObject | None:
    """Narrow an arbitrary JSON value to an object, else ``None``."""
    return value if isinstance(value, dict) else None


def _apply_update_config(
    existing: KnowledgeBase,
    config: JsonObject | None,
) -> dict[str, JsonObject | None]:
    """Compute the config-column updates for an update request.

    Mirrors the upstream ``KnowledgeBaseConfig`` merge: chunking and
    image-processing configs are replaced unconditionally, FAQ and wiki
    configs only when present, and an indexing strategy, when supplied,
    must enable at least one pipeline and syncs graph extraction onto the
    existing extract config. All results are fresh dicts — inputs are
    never mutated.
    """
    if config is None:
        return {}
    fields: dict[str, JsonObject | None] = {}
    fields["chunking_config"] = _as_json_or_none(config.get("chunking_config"))
    fields["image_processing_config"] = _as_json_or_none(config.get("image_processing_config"))
    faq = config.get("faq_config")
    if isinstance(faq, dict):
        fields["faq_config"] = faq
    wiki = config.get("wiki_config")
    if isinstance(wiki, dict):
        fields["wiki_config"] = wiki
    strategy = config.get("indexing_strategy")
    if isinstance(strategy, dict):
        strategy_copy = dict(strategy)
        if not _has_any_indexing(strategy_copy):
            raise ValidationError(
                code="knowledge_base.indexing_required",
                message="at least one indexing strategy must be enabled",
            )
        fields["indexing_strategy"] = strategy_copy
        if _is_true(strategy_copy.get("wiki_enabled")) and "wiki_config" not in fields:
            fields["wiki_config"] = {}
        graph = _is_true(strategy_copy.get("graph_enabled"))
        existing_extract = existing.extract_config
        if isinstance(existing_extract, dict):
            fields["extract_config"] = {**existing_extract, "enabled": graph}
        elif graph:
            fields["extract_config"] = {"enabled": True}
    return fields


# ── Service ──────────────────────────────────────────────────────────


class KBService:
    """Stateless knowledge-base service, constructed per request."""

    def __init__(self, *, kb_repo: KnowledgeBaseRepository) -> None:
        self._kb_repo = kb_repo

    # ── Create ──────────────────────────────────────────────────────

    async def create_knowledge_base(
        self,
        *,
        tenant_id: int,
        name: str,
        kb_type: str = KNOWLEDGE_BASE_TYPE_DOCUMENT,
        description: str | None = None,
        creator_id: str | None = None,
        is_temporary: bool = False,
        chunking_config: JsonObject | None = None,
        image_processing_config: JsonObject | None = None,
        embedding_model_id: str | None = None,
        summary_model_id: str | None = None,
        vlm_config: JsonObject | None = None,
        asr_config: JsonObject | None = None,
        storage_provider_config: JsonObject | None = None,
        storage_backend_id: str | None = None,
        storage_config: JsonObject | None = None,
        extract_config: JsonObject | None = None,
        faq_config: JsonObject | None = None,
        question_generation_config: JsonObject | None = None,
        wiki_config: JsonObject | None = None,
        indexing_strategy: JsonObject | None = None,
        vector_store_id: str | None = None,
    ) -> KnowledgeBaseInfo:
        """Insert a new knowledge base and return its projected shape.

        The service stamps id / tenant / timestamps and applies the
        type-specific defaults; storage-backend resolution and the
        vector-store binding check are deferred with those domains.
        """
        _require_tenant_id(tenant_id)
        clean_name = _require_name(name)
        _require_kb_type(kb_type)
        now = _now()
        row = KnowledgeBase(
            id=_new_id(),
            tenant_id=tenant_id,
            name=clean_name,
            type=kb_type,
            is_temporary=is_temporary,
            description=description,
            creator_id=creator_id,
            chunking_config=chunking_config,
            image_processing_config=image_processing_config,
            embedding_model_id=embedding_model_id or "",
            summary_model_id=summary_model_id or "",
            vlm_config=vlm_config,
            asr_config=asr_config,
            storage_provider_config=storage_provider_config,
            storage_backend_id=storage_backend_id,
            cos_config=storage_config,
            extract_config=extract_config,
            faq_config=faq_config,
            question_generation_config=question_generation_config,
            wiki_config=wiki_config,
            indexing_strategy=indexing_strategy,
            vector_store_id=_normalise_vector_store_id(vector_store_id),
            created_at=now,
            updated_at=now,
        )
        persisted = await self._kb_repo.create(_ensure_defaults(row))
        return KnowledgeBaseInfo.map_from_db(persisted)

    # ── Reads ───────────────────────────────────────────────────────

    async def get_knowledge_base_by_id(self, *, knowledge_base_id: str) -> KnowledgeBaseInfo:
        """Return one knowledge base by id, or raise ``NotFoundError``.

        Reads without a tenant filter; routes gate access at the web
        layer. Defaults are applied before the row is projected.
        """
        _require_knowledge_base_id(knowledge_base_id)
        row = await self._kb_repo.get_by_id_or_none(knowledge_base_id)
        if row is None:
            raise NotFoundError(
                code=_NOT_FOUND_CODE,
                message=f"knowledge base {knowledge_base_id} not found",
            )
        return KnowledgeBaseInfo.map_from_db(_ensure_defaults(row))

    async def get_knowledge_base_by_id_only(self, *, knowledge_base_id: str) -> KnowledgeBaseInfo:
        """Return one knowledge base by id without a tenant filter.

        Intended for cross-tenant shared access where permission is
        checked by the caller; identical in shape to
        ``get_knowledge_base_by_id``.
        """
        return await self.get_knowledge_base_by_id(knowledge_base_id=knowledge_base_id)

    async def get_knowledge_base_by_id_and_tenant(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
    ) -> KnowledgeBaseInfo:
        """Return one knowledge base scoped to the tenant, or raise.

        A row owned by another tenant reads as absent, enforcing tenant
        isolation at the service boundary.
        """
        _require_tenant_id(tenant_id)
        _require_knowledge_base_id(knowledge_base_id)
        row = await self._kb_repo.get_by_id_and_tenant(knowledge_base_id, tenant_id)
        if row is None:
            raise NotFoundError(
                code=_NOT_FOUND_CODE,
                message=f"knowledge base {knowledge_base_id} not found",
            )
        return KnowledgeBaseInfo.map_from_db(_ensure_defaults(row))

    async def get_knowledge_bases_by_ids(self, *, ids: list[str]) -> list[KnowledgeBaseInfo]:
        """Return the knowledge bases for the given ids.

        Missing ids are silently dropped; an empty input yields an empty
        list.
        """
        if not ids:
            return []
        rows = await self._kb_repo.get_by_ids(ids)
        return [KnowledgeBaseInfo.map_from_db(_ensure_defaults(row)) for row in rows]

    async def list_knowledge_bases(self, *, tenant_id: int) -> list[KnowledgeBaseInfo]:
        """Return every live, non-temporary knowledge base of the tenant.

        Newest first; document-type rows carry ``knowledge_count`` and
        FAQ-type rows carry ``chunk_count`` via best-effort enrichment.
        """
        _require_tenant_id(tenant_id)
        rows = await self._kb_repo.list_by_tenant(tenant_id)
        infos: list[KnowledgeBaseInfo] = []
        for row in rows:
            info = KnowledgeBaseInfo.map_from_db(_ensure_defaults(row))
            infos.append(await self._fill_counts(tenant_id=tenant_id, info=info))
        return infos

    async def _fill_counts(
        self,
        *,
        tenant_id: int,
        info: KnowledgeBaseInfo,
    ) -> KnowledgeBaseInfo:
        """Best-effort count enrichment for one listed knowledge base.

        Mirrors the upstream list path: document rows get the document
        count and FAQ rows get the chunk count. A failing sibling-table
        query is ignored (counts are enrichment, not the listing itself),
        leaving the field at its zero default.
        """
        updates: dict[str, int] = {}
        try:
            if info.type == KNOWLEDGE_BASE_TYPE_DOCUMENT:
                updates["knowledge_count"] = await self._kb_repo.count_documents(
                    tenant_id=tenant_id,
                    knowledge_base_id=info.id,
                )
            elif info.type == KNOWLEDGE_BASE_TYPE_FAQ:
                updates["chunk_count"] = await self._kb_repo.count_chunks(
                    tenant_id=tenant_id,
                    knowledge_base_id=info.id,
                )
        except Exception:
            return info
        return info.model_copy(update=updates)

    # ── Update / delete ─────────────────────────────────────────────

    async def update_knowledge_base(
        self,
        *,
        knowledge_base_id: str,
        name: str | None = None,
        description: str | None = None,
        config: JsonObject | None = None,
    ) -> KnowledgeBaseInfo:
        """Partial-update the mutable fields of an existing knowledge base.

        Every parameter is optional — ``None`` means "leave the existing
        value alone". This lets the same request shape serve PUT (full
        body) and PATCH (subset). The vector-store binding is immutable
        by contract and is never part of an update. The supplied
        ``config`` is merged per the upstream ``KnowledgeBaseConfig``
        semantics.
        """
        _require_knowledge_base_id(knowledge_base_id)
        existing = await self._kb_repo.get_by_id_or_none(knowledge_base_id)
        if existing is None:
            raise NotFoundError(
                code=_NOT_FOUND_CODE,
                message=f"knowledge base {knowledge_base_id} not found",
            )
        # Build the patch dict: skip keys whose caller did not supply.
        patch: dict[str, JsonObject | str | datetime | None] = {"updated_at": _now()}
        if name is not None:
            patch["name"] = _require_name(name)
        if description is not None:
            patch["description"] = description
        patch.update(_apply_update_config(existing, config))
        row = _ensure_defaults(existing.model_copy(update=patch))
        persisted = await self._kb_repo.update(row)
        return KnowledgeBaseInfo.map_from_db(persisted)

    async def delete_knowledge_base(self, *, knowledge_base_id: str) -> bool:
        """Soft-delete a knowledge base; return whether a row was removed.

        Unknown or already-deleted knowledge bases raise
        ``NotFoundError`` (the read filters soft-deleted rows). The heavy
        cleanup (embeddings, chunks, files, graph) runs in a deferred
        async task.
        """
        _require_knowledge_base_id(knowledge_base_id)
        existing = await self._kb_repo.get_by_id_or_none(knowledge_base_id)
        if existing is None:
            raise NotFoundError(
                code=_NOT_FOUND_CODE,
                message=f"knowledge base {knowledge_base_id} not found",
            )
        return await self._kb_repo.soft_delete(id=knowledge_base_id, now=_now())

    # ── Aggregate counts ────────────────────────────────────────────

    async def count_documents(self, *, tenant_id: int, knowledge_base_id: str) -> int:
        """Count live document rows of a knowledge base."""
        _require_tenant_id(tenant_id)
        _require_knowledge_base_id(knowledge_base_id)
        return await self._kb_repo.count_documents(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
        )

    async def count_chunks(self, *, tenant_id: int, knowledge_base_id: str) -> int:
        """Count live chunk rows of a knowledge base."""
        _require_tenant_id(tenant_id)
        _require_knowledge_base_id(knowledge_base_id)
        return await self._kb_repo.count_chunks(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
        )

    async def count_members(self, *, tenant_id: int, knowledge_base_id: str) -> int:
        """Count live share rows of a knowledge base."""
        _require_tenant_id(tenant_id)
        _require_knowledge_base_id(knowledge_base_id)
        return await self._kb_repo.count_members(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
        )


__all__ = ["KBService"]
