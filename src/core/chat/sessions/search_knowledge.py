"""Retrieval-only knowledge search for a session (no LLM summarization).

Maps the upstream session ``SearchKnowledge`` contract: given a query
and a set of knowledge targets (whole knowledge bases, specific
knowledge files, and/or tag-constrained scopes), build the unified
search-target list, run the retrieval pipeline stages (search → rerank →
merge → filter), and return the merged hits without generating an
answer.

The service is request-scoped (carries ``tenant_id`` / ``user_id``) and
keeps every heavy dependency behind ``Protocol`` seams so it is
testable without a live pipeline, model registry, or repository. The
target builder mirrors the upstream resolution rules:

- a knowledge-base id yields one whole-KB target (resolving the owning
  tenant, including shared KBs the caller may read);
- a knowledge (file) id yields a per-KB target listing the files;
- a tag scope resolves the files carrying the tags (document KBs) or
  stays a whole-KB target (FAQ KBs).

When a dependency is not injected the corresponding resolution step is
skipped (best-effort), matching the upstream behaviour of logging and
continuing instead of failing the search.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from src.common.exception import ValidationError
from src.core.chat.pipeline import (
    ERR_SEARCH_NOTHING,
    EventManager,
    EventType,
    PipelineContext,
    PluginError,
    SearchResult,
    SearchTarget,
    SearchTargetType,
)
from src.core.chat.service import TagScope
from src.core.contracts.knowledge import Knowledge as KnowledgeInfo
from src.core.infra.models.types import ModelInfo
from src.core.knowledge.knowledge_bases.hybrid_search import RetrievalConfig
from src.core.knowledge.knowledge_bases.types import (
    KNOWLEDGE_BASE_TYPE_FAQ,
    KnowledgeBaseInfo,
)

logger = logging.getLogger(__name__)

#: Model type used to pick a default rerank model for the search run.
_MODEL_TYPE_RERANK = "rerank"
#: Org role granted to a viewer when probing cross-tenant KB access.
_ORG_ROLE_VIEWER = "viewer"

#: Retrieval pipeline stages run by a knowledge search. The LLM
#: summarization stages are deliberately absent — search only returns
#: the merged hits.
_SEARCH_EVENTS: tuple[EventType, ...] = (
    EventType.CHUNK_SEARCH,
    EventType.CHUNK_RERANK,
    EventType.CHUNK_MERGE,
    EventType.FILTER_TOP_K,
)


@runtime_checkable
class KnowledgeBaseResolver(Protocol):
    """Resolves knowledge bases by id (authorization is the caller's job)."""

    async def get_knowledge_bases_by_ids(self, *, ids: list[str]) -> list[KnowledgeBaseInfo]: ...


@runtime_checkable
class KnowledgeResolver(Protocol):
    """Resolves knowledge (file) rows and tag-to-file mappings.

    ``get_documents`` mirrors the tenant-scoped knowledge fetch used by
    the upstream target builder; ``list_knowledge_ids_by_tag_ids``
    resolves the files carrying a tag scope inside one knowledge base.
    """

    async def get_documents(self, *, tenant_id: int, ids: list[str]) -> list[KnowledgeInfo]: ...

    async def list_knowledge_ids_by_tag_ids(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        tag_ids: list[str],
    ) -> list[str]: ...


@runtime_checkable
class ModelResolver(Protocol):
    """Lists the models visible to the caller's tenant."""

    async def list_models(
        self,
        *,
        tenant_id: int,
        model_type: str | None = None,
    ) -> list[ModelInfo]: ...


@runtime_checkable
class KBPermissionChecker(Protocol):
    """Best-effort cross-tenant KB read check (share seams)."""

    async def has_tenant_kb_permission(
        self,
        *,
        knowledge_base_id: str,
        tenant_id: int,
        org_role: str,
    ) -> bool: ...


class SearchKnowledgeService:
    """Per-request retrieval search over knowledge targets.

    Immutable after construction: every method returns new values or
    raises; the pipeline carrier is the only mutable object, and it is
    created fresh inside :meth:`search`.
    """

    def __init__(
        self,
        *,
        tenant_id: int,
        user_id: str,
        event_manager: EventManager,
        knowledge_base_resolver: KnowledgeBaseResolver | None = None,
        knowledge_resolver: KnowledgeResolver | None = None,
        model_resolver: ModelResolver | None = None,
        kb_permission_checker: KBPermissionChecker | None = None,
        retrieval_config: RetrievalConfig | None = None,
        max_rounds: int = 4,
    ) -> None:
        if tenant_id <= 0:
            raise ValidationError(
                code="search.invalid_tenant_id",
                message="tenant_id must be positive",
            )
        self._tenant_id = tenant_id
        self._user_id = user_id
        self._event_manager = event_manager
        self._kb_resolver = knowledge_base_resolver
        self._knowledge_resolver = knowledge_resolver
        self._model_resolver = model_resolver
        self._permission_checker = kb_permission_checker
        self._retrieval_config = retrieval_config
        self._max_rounds = max_rounds

    # ── Search ──────────────────────────────────────────────────────

    async def search(
        self,
        *,
        query: str,
        knowledge_base_ids: list[str] | None = None,
        knowledge_ids: list[str] | None = None,
        tag_scopes: list[TagScope] | None = None,
    ) -> list[SearchResult]:
        """Run a retrieval-only search and return the merged hits.

        An empty target set yields an empty result list (nothing to
        search). A search stage reporting ``search_nothing`` also yields
        an empty list — both match the upstream contract.
        """
        if not query or not query.strip():
            raise ValidationError(
                code="search.query_required",
                message="Query content cannot be empty",
            )

        targets = await self._build_search_targets(
            knowledge_base_ids=knowledge_base_ids or [],
            knowledge_ids=knowledge_ids or [],
            tag_scopes=tag_scopes or [],
        )
        if not targets:
            logger.warning("No search targets available, returning empty results")
            return []

        pipeline_ctx = PipelineContext(
            session_id="",
            user_id=self._user_id,
            query=query.strip(),
            max_rounds=self._max_rounds,
            knowledge_base_ids=list(knowledge_base_ids or []),
            knowledge_ids=list(knowledge_ids or []),
            search_targets=targets,
            tenant_id=self._tenant_id,
            rewrite_query=query.strip(),
        )
        await self._apply_retrieval_config(pipeline_ctx)
        await self._apply_rerank_model(pipeline_ctx)

        for event_type in _SEARCH_EVENTS:
            logger.info("Starting to trigger search event: %s", event_type)
            err = await self._event_manager.trigger(
                pipeline_ctx,
                event_type,
                pipeline_ctx,
            )
            if err is ERR_SEARCH_NOTHING:
                logger.warning("Event %s triggered, search result is empty", event_type)
                return []
            if err is not None:
                raise _to_exception(err, event_type)

        logger.info(
            "Knowledge base search completed, found %d results",
            len(pipeline_ctx.merge_result),
        )
        return list(pipeline_ctx.merge_result)

    # ── Target building (upstream ``buildSearchTargets``) ───────────

    async def _build_search_targets(
        self,
        *,
        knowledge_base_ids: list[str],
        knowledge_ids: list[str],
        tag_scopes: list[TagScope],
    ) -> list[SearchTarget]:
        """Resolve the requested targets into unified search targets."""
        tag_ids_by_kb: dict[str, list[str]] = _merge_tag_scopes_by_kb(tag_scopes)

        # Batch-fetch the knowledge bases involved (best-effort).
        kb_ids_to_fetch = _unique_non_empty([*knowledge_base_ids, *tag_ids_by_kb])
        kb_by_id: dict[str, KnowledgeBaseInfo] = {}
        if kb_ids_to_fetch and self._kb_resolver is not None:
            try:
                kbs = await self._kb_resolver.get_knowledge_bases_by_ids(ids=kb_ids_to_fetch)
            except Exception:
                logger.warning(
                    "Failed to fetch knowledge bases for search targets",
                    exc_info=True,
                )
            else:
                kb_by_id = {kb.id: kb for kb in kbs if kb is not None}

        kb_tenant_map: dict[str, int] = {}

        async def _resolve_kb_tenant(kb_id: str) -> int:
            known = kb_tenant_map.get(kb_id)
            if known is not None:
                return known
            kb = kb_by_id.get(kb_id)
            if kb is None or kb.tenant_id == self._tenant_id:
                tenant = self._tenant_id
            elif self._permission_checker is not None and self._user_id:
                has_access = await self._permission_checker.has_tenant_kb_permission(
                    knowledge_base_id=kb_id,
                    tenant_id=self._tenant_id,
                    org_role=_ORG_ROLE_VIEWER,
                )
                tenant = kb.tenant_id if has_access else self._tenant_id
            else:
                tenant = self._tenant_id
            kb_tenant_map[kb_id] = tenant
            return tenant

        targets: list[SearchTarget] = []

        # Whole-KB targets (unless the KB is already narrowed by a tag scope).
        full_kb_set: set[str] = set()
        for kb_id in knowledge_base_ids:
            if not kb_id:
                continue
            full_kb_set.add(kb_id)
            if tag_ids_by_kb.get(kb_id):
                continue
            targets.append(
                SearchTarget(
                    type=SearchTargetType.KNOWLEDGE_BASE,
                    knowledge_base_id=kb_id,
                    tenant_id=await _resolve_kb_tenant(kb_id),
                )
            )

        # Per-KB targets for explicit knowledge (file) ids.
        kb_to_knowledge_ids: dict[str, list[str]] = {}
        if knowledge_ids and self._knowledge_resolver is not None:
            try:
                knowledge_list = await self._knowledge_resolver.get_documents(
                    tenant_id=self._tenant_id,
                    ids=list(knowledge_ids),
                )
            except Exception:
                logger.warning(
                    "Failed to get knowledge batch for search targets",
                    exc_info=True,
                )
            else:
                for k in knowledge_list:
                    if k is None or not k.knowledge_base_id:
                        continue
                    if kb_tenant_map.get(k.knowledge_base_id) is None:
                        kb_tenant_map[k.knowledge_base_id] = k.tenant_id
                    if k.knowledge_base_id in full_kb_set and not tag_ids_by_kb.get(
                        k.knowledge_base_id
                    ):
                        continue
                    kb_to_knowledge_ids.setdefault(k.knowledge_base_id, []).append(k.id)

            for kb_id, kid_list in kb_to_knowledge_ids.items():
                if tag_ids_by_kb.get(kb_id):
                    continue
                kb_tenant = kb_tenant_map.get(kb_id) or self._tenant_id
                targets.append(
                    SearchTarget(
                        type=SearchTargetType.KNOWLEDGE,
                        knowledge_base_id=kb_id,
                        tenant_id=kb_tenant,
                        knowledge_ids=list(kid_list),
                        disable_recall_thresholds=True,
                    )
                )

        # Tag-scope targets.
        for kb_id, tag_ids in tag_ids_by_kb.items():
            if not kb_id or not tag_ids:
                continue
            kb = kb_by_id.get(kb_id)
            explicit = _unique_non_empty(kb_to_knowledge_ids.get(kb_id, []))
            use_document_resolution = kb is None or kb.type != KNOWLEDGE_BASE_TYPE_FAQ
            if use_document_resolution:
                if self._knowledge_resolver is None:
                    continue
                kb_tenant = await _resolve_kb_tenant(kb_id)
                try:
                    tag_knowledge_ids = (
                        await self._knowledge_resolver.list_knowledge_ids_by_tag_ids(
                            tenant_id=kb_tenant,
                            knowledge_base_id=kb_id,
                            tag_ids=tag_ids,
                        )
                    )
                except Exception:
                    logger.warning(
                        "Failed to resolve knowledge IDs for tag scope kb_id=%s",
                        kb_id,
                        exc_info=True,
                    )
                    continue
                if explicit:
                    tag_knowledge_ids = _intersect(tag_knowledge_ids, explicit)
                tag_knowledge_ids = _unique_non_empty(tag_knowledge_ids)
                if not tag_knowledge_ids:
                    continue
                targets.append(
                    SearchTarget(
                        type=SearchTargetType.KNOWLEDGE,
                        knowledge_base_id=kb_id,
                        tenant_id=kb_tenant,
                        knowledge_ids=list(tag_knowledge_ids),
                        scope_tag_ids=list(tag_ids),
                        disable_recall_thresholds=True,
                    )
                )
                continue

            target = SearchTarget(
                type=SearchTargetType.KNOWLEDGE_BASE,
                knowledge_base_id=kb_id,
                tenant_id=await _resolve_kb_tenant(kb_id),
                tag_ids=list(tag_ids),
                scope_tag_ids=list(tag_ids),
                disable_recall_thresholds=True,
            )
            if explicit:
                target = target.model_copy(
                    update={
                        "type": SearchTargetType.KNOWLEDGE,
                        "knowledge_ids": list(explicit),
                    }
                )
            targets.append(target)

        logger.info(
            "Built %d search targets (%d full KB, %d partial/tag KB)",
            len(targets),
            len(knowledge_base_ids),
            len(targets) - len(knowledge_base_ids),
        )
        return targets

    # ── Config helpers ──────────────────────────────────────────────

    async def _apply_retrieval_config(self, pipeline_ctx: PipelineContext) -> None:
        """Overlay the effective retrieval config onto the carrier."""
        config = self._retrieval_config
        if config is None:
            return
        pipeline_ctx.embedding_top_k = config.embedding_top_k
        pipeline_ctx.vector_threshold = config.vector_threshold
        pipeline_ctx.keyword_threshold = config.keyword_threshold
        pipeline_ctx.rerank_top_k = config.rerank_top_k
        pipeline_ctx.rerank_threshold = config.rerank_threshold

    async def _apply_rerank_model(self, pipeline_ctx: PipelineContext) -> None:
        """Pick the rerank model from the config or the first available."""
        if self._model_resolver is None:
            return
        models = await self._model_resolver.list_models(
            tenant_id=self._tenant_id,
            model_type=_MODEL_TYPE_RERANK,
        )
        for model in models:
            if model is not None and model.type == _MODEL_TYPE_RERANK:
                pipeline_ctx.rerank_model_id = model.id
                return


# ── Helpers ──────────────────────────────────────────────────────────


def _merge_tag_scopes_by_kb(scopes: list[TagScope]) -> dict[str, list[str]]:
    """Group deduplicated tag ids by knowledge base (upstream merge)."""
    by_kb: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for scope in scopes:
        if not scope.knowledge_base_id:
            continue
        bucket = seen.setdefault(scope.knowledge_base_id, set())
        for tag_id in scope.tag_ids:
            if not tag_id or tag_id in bucket:
                continue
            bucket.add(tag_id)
            by_kb.setdefault(scope.knowledge_base_id, []).append(tag_id)
    return by_kb


def _unique_non_empty(values: list[str]) -> list[str]:
    """Return the values in order, dropping blanks and duplicates."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _intersect(left: list[str], right: list[str]) -> list[str]:
    """Return the values of ``left`` present in ``right``, in order."""
    right_set = set(right)
    return [value for value in left if value in right_set]


def _to_exception(err: PluginError, event_type: EventType) -> Exception:
    """Convert a pipeline plugin error into a domain exception.

    The underlying exception (when present) is re-raised as-is so the
    original type and traceback survive; a bare plugin error becomes a
    ``RuntimeError`` carrying its description.
    """
    if isinstance(err.err, Exception):
        return err.err
    message = err.description or f"Pipeline stage {event_type} failed"
    return RuntimeError(message)


__all__ = [
    "KBPermissionChecker",
    "KnowledgeBaseResolver",
    "KnowledgeResolver",
    "ModelResolver",
    "SearchKnowledgeService",
]
