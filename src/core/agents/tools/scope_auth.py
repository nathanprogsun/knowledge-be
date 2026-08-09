"""Scope authorization for agent retrieval tools.

Every agent tool that accepts a model-visible document or chunk id must
resolve the durable row and prove it belongs to the server-owned search
scope for this session — handle decoding is necessary but never
sufficient. This module is the shared authorization boundary:

- whole-knowledge-base targets authorize every document of the KB;
- explicit document whitelists authorize exactly the listed documents;
- tag scopes authorize documents carrying any of the scoped tags;

and inside one target the documents and tags are an intersection (a tag
mentions a document only when the relation table resolves it into the
whitelist), while alternatives across targets remain a union. Any
violation raises a typed application error with a model-facing message.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from src.ai.embedding.base import Context
from src.common.exception import (
    ApplicationError,
    NotFoundError,
    PermissionDeniedError,
    ValidationError,
)
from src.core.agents.tools.search_target import (
    SearchTarget,
    SearchTargets,
    SearchTargetType,
)
from src.core.agents.tools.text_utils import dedup_non_empty_strings
from src.core.contracts.knowledge import Knowledge
from src.core.knowledge.knowledge_bases.hybrid_search import SearchResult
from src.core.knowledge.tags.types import TagInfo
from src.db.models.chunk import Chunk


@runtime_checkable
class KnowledgeLookup(Protocol):
    """Resolves a document row by id without a tenant filter."""

    async def get_document_by_id_only(self, *, id: str) -> Knowledge | None: ...


@runtime_checkable
class ChunkLookup(Protocol):
    """Resolves a chunk row by id without a tenant filter."""

    async def get_chunk_by_id_only(self, *, id: str) -> Chunk | None: ...


class KnowledgeTagsFetcher(Protocol):
    """Resolves the tag bindings of a set of documents."""

    async def get_knowledge_tags(
        self,
        knowledge_ids: list[str],
    ) -> dict[str, list[TagInfo]]: ...


KnowledgeTagsFn = Callable[[list[str]], Awaitable[dict[str, list[TagInfo]]]]


def effective_search_target_tag_ids(target: SearchTarget) -> list[str]:
    """Return the deduplicated tag ids of a target (physical + scope tags)."""
    if target is None:
        return []
    return dedup_non_empty_strings(list(target.tag_ids) + list(target.scope_tag_ids))


def search_target_scope(target: SearchTarget) -> tuple[list[str], list[str]]:
    """Return what a SINGLE target authorizes: ``(knowledge_ids, tag_ids)``.

    Inside one target, knowledge ids and tags are an intersection: tags only
    authorize when the target carries no resolved document whitelist.
    """
    if target is None:
        return [], []
    knowledge_ids = dedup_non_empty_strings(list(target.knowledge_ids))
    if knowledge_ids:
        return knowledge_ids, []
    return [], effective_search_target_tag_ids(target)


def search_target_is_whole_kb(target: SearchTarget) -> bool:
    """Whether a target grants unrestricted access to its knowledge base."""
    if target is None:
        return False
    knowledge_ids, tag_ids = search_target_scope(target)
    return (
        target.type is SearchTargetType.KNOWLEDGE_BASE
        and not knowledge_ids
        and not tag_ids
    )


async def authorize_knowledge_in_search_targets(
    ctx: Context,
    search_targets: SearchTargets,
    knowledge_id: str,
    knowledge_service: KnowledgeLookup | None,
    tag_fetcher: KnowledgeTagsFetcher | None = None,
) -> Knowledge:
    """Resolve ``knowledge_id`` and prove it is within the session scope.

    Raises ``ValidationError`` / ``NotFoundError`` / ``PermissionDeniedError``
    with a model-facing message on any failure.
    """
    knowledge_id = knowledge_id.strip()
    if not knowledge_id:
        raise ValidationError(
            code="tool.knowledge_id_required",
            message="knowledge_id is required",
        )
    if knowledge_service is None:
        raise ValidationError(
            code="tool.knowledge_service_unavailable",
            message="knowledge service is unavailable",
        )
    knowledge = await knowledge_service.get_document_by_id_only(id=knowledge_id)
    if knowledge is None:
        raise NotFoundError(
            code="tool.document_not_found",
            message=f"document {knowledge_id} not found",
        )
    if not search_targets.contains_kb(knowledge.knowledge_base_id):
        raise PermissionDeniedError(
            code="tool.kb_out_of_scope",
            message=(
                f"knowledge base {knowledge.knowledge_base_id} is not within "
                "the current Agent scope"
            ),
        )
    allowed, error = await search_targets_allow_knowledge_id(
        ctx,
        search_targets,
        knowledge.id,
        knowledge.knowledge_base_id,
        tag_fetcher,
    )
    if error is not None:
        raise PermissionDeniedError(
            code="tool.scope_validation_failed",
            message=f"failed to validate document scope: {error.message}",
        ) from error
    if not allowed:
        raise PermissionDeniedError(
            code="tool.document_out_of_mention_scope",
            message=f"document {knowledge.id} is not within the current @mention scope",
        )
    return knowledge


async def authorize_chunk_in_search_targets(
    ctx: Context,
    search_targets: SearchTargets,
    chunk_id: str,
    chunk_service: ChunkLookup | None,
    knowledge_service: KnowledgeLookup | None,
    tag_fetcher: KnowledgeTagsFetcher | None = None,
) -> Chunk:
    """Resolve ``chunk_id`` and prove its owning document is in scope."""
    chunk_id = chunk_id.strip()
    if not chunk_id:
        raise ValidationError(
            code="tool.chunk_id_required",
            message="chunk_id is required",
        )
    if chunk_service is None:
        raise ValidationError(
            code="tool.chunk_service_unavailable",
            message="chunk service is unavailable",
        )
    chunk = await chunk_service.get_chunk_by_id_only(id=chunk_id)
    if chunk is None:
        raise NotFoundError(
            code="tool.chunk_not_found",
            message=f"chunk {chunk_id} not found",
        )
    if not chunk.is_enabled:
        raise PermissionDeniedError(
            code="tool.chunk_disabled",
            message=f"chunk {chunk.id} is disabled",
        )
    if not search_targets.contains_kb(chunk.knowledge_base_id):
        raise PermissionDeniedError(
            code="tool.kb_out_of_scope",
            message=(
                f"knowledge base {chunk.knowledge_base_id} is not within "
                "the current Agent scope"
            ),
        )
    allowed, error = await search_targets_allow_knowledge_id(
        ctx,
        search_targets,
        chunk.knowledge_id,
        chunk.knowledge_base_id,
        tag_fetcher,
    )
    if error is not None:
        raise PermissionDeniedError(
            code="tool.scope_validation_failed",
            message=f"failed to validate chunk scope: {error.message}",
        ) from error
    if not allowed:
        raise PermissionDeniedError(
            code="tool.chunk_out_of_mention_scope",
            message=f"chunk {chunk.id} is not within the current @mention scope",
        )
    return chunk


def validate_knowledge_base_ids_in_search_targets(
    search_targets: SearchTargets,
    kb_ids: list[str],
) -> None:
    """Reject any knowledge-base id that lies outside the session scope."""
    for kb_id in dedup_non_empty_strings(kb_ids):
        if not search_targets.contains_kb(kb_id):
            raise PermissionDeniedError(
                code="tool.kb_out_of_scope",
                message=f"knowledge base {kb_id} is not within the current Agent scope",
            )


async def resolve_authorized_source_refs(
    ctx: Context,
    search_targets: SearchTargets,
    refs: list[str],
    knowledge_service: KnowledgeLookup | None,
    tag_fetcher: KnowledgeTagsFetcher | None = None,
) -> list[str]:
    """Validate Wiki-style ``id|title`` refs and rebuild them from server data.

    The title suffix supplied by the model is never trusted; each ref is
    re-authorized and re-serialized as ``id|title`` from the durable row.
    """
    resolved: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        knowledge_id = ref.split("|", 1)[0].strip()
        if not knowledge_id:
            continue
        knowledge = await authorize_knowledge_in_search_targets(
            ctx,
            search_targets,
            knowledge_id,
            knowledge_service,
            tag_fetcher,
        )
        if knowledge.id in seen:
            continue
        seen.add(knowledge.id)
        title = knowledge.title.strip() if knowledge.title else ""
        if not title and knowledge.file_name:
            title = knowledge.file_name.strip()
        if title:
            resolved.append(f"{knowledge.id}|{title}")
        else:
            resolved.append(knowledge.id)
    return resolved


async def search_targets_allow_knowledge_id(
    ctx: Context,
    search_targets: SearchTargets,
    knowledge_id: str,
    kb_id: str,
    tag_fetcher: KnowledgeTagsFetcher | None,
) -> tuple[bool, ApplicationError | None]:
    """Whether a single document is authorized by the union of targets.

    Returns ``(allowed, None)`` when the outcome is a decision, or
    ``(False, error)`` when the tag lookup itself failed.
    """
    if not knowledge_id or not kb_id:
        return False, None

    tag_ids: list[str] = []
    matched_kb = False
    for target in search_targets:
        if target is None or target.knowledge_base_id != kb_id:
            continue
        matched_kb = True
        if search_target_is_whole_kb(target):
            return True, None
        target_knowledge_ids, target_tag_ids = search_target_scope(target)
        if knowledge_id in target_knowledge_ids:
            return True, None
        tag_ids.extend(target_tag_ids)

    if not matched_kb or not tag_ids or tag_fetcher is None:
        return False, None

    try:
        matches = await knowledge_ids_matching_any_tag(
            ctx,
            [knowledge_id],
            tag_ids,
            tag_fetcher.get_knowledge_tags,
        )
    except ApplicationError as exc:
        return False, exc
    return matches.get(knowledge_id, False), None


async def filter_search_results_in_search_targets(
    ctx: Context,
    search_targets: SearchTargets,
    kb_id: str,
    results: list[SearchResult],
    knowledge_service: KnowledgeLookup | None,
    tag_fetcher: KnowledgeTagsFetcher | None = None,
) -> list[SearchResult]:
    """Apply the whole-KB / document / tag union semantics to batched results.

    Used by tools whose backend can only query by knowledge base; it batches
    tag lookup and rejects results without enough provenance instead of
    turning a narrow mention into whole-KB access.
    """
    explicit_ids: list[str] = []
    tag_ids: list[str] = []
    matched_kb = False
    for target in search_targets:
        if target is None or target.knowledge_base_id != kb_id:
            continue
        matched_kb = True
        if search_target_is_whole_kb(target):
            return results
        target_knowledge_ids, target_tag_ids = search_target_scope(target)
        explicit_ids.extend(target_knowledge_ids)
        tag_ids.extend(target_tag_ids)

    if not matched_kb:
        raise PermissionDeniedError(
            code="tool.kb_out_of_scope",
            message=f"knowledge base {kb_id} is not within the current Agent scope",
        )

    explicit_set = set(dedup_non_empty_strings(explicit_ids))
    remaining_ids: list[str] = []
    for result in results:
        if result is None or not result.knowledge_id:
            continue
        if result.knowledge_base_id and result.knowledge_base_id != kb_id:
            raise PermissionDeniedError(
                code="tool.graph_result_mismatch",
                message=(
                    f"graph result document {result.knowledge_id} belongs to "
                    f"knowledge base {result.knowledge_base_id}, expected {kb_id}"
                ),
            )
        if result.knowledge_id not in explicit_set:
            remaining_ids.append(result.knowledge_id)

    tag_matches: dict[str, bool] = {}
    if tag_ids:
        if tag_fetcher is None:
            raise ValidationError(
                code="tool.knowledge_service_unavailable",
                message="knowledge service is unavailable for tag-scoped graph filtering",
            )
        tag_matches = await knowledge_ids_matching_any_tag(
            ctx,
            remaining_ids,
            tag_ids,
            tag_fetcher.get_knowledge_tags,
        )

    filtered: list[SearchResult] = []
    for result in results:
        if result is None or not result.knowledge_id:
            continue
        if result.knowledge_id in explicit_set or tag_matches.get(result.knowledge_id, False):
            filtered.append(result)
    return filtered


async def knowledge_ids_matching_any_tag(
    ctx: Context,
    knowledge_ids: list[str],
    tag_ids: list[str],
    fetch_tags: KnowledgeTagsFn | None,
) -> dict[str, bool]:
    """Return which documents carry any of the given tag ids.

    ``fetch_tags`` maps a batch of document ids to their tag lists. A
    ``None`` fetcher or empty inputs yields an empty result.
    """
    result: dict[str, bool] = {}
    if not knowledge_ids or not tag_ids or fetch_tags is None:
        return result

    unique_knowledge_ids = dedup_non_empty_strings(knowledge_ids)
    unique_tag_ids = dedup_non_empty_strings(tag_ids)
    if not unique_knowledge_ids or not unique_tag_ids:
        return result

    tag_set = frozenset(unique_tag_ids)
    tag_map = await fetch_tags(unique_knowledge_ids)
    for knowledge_id in unique_knowledge_ids:
        for tag in tag_map.get(knowledge_id, []):
            if tag is not None and tag.id in tag_set:
                result[knowledge_id] = True
                break
    return result


__all__ = [
    "ChunkLookup",
    "KnowledgeLookup",
    "KnowledgeTagsFetcher",
    "KnowledgeTagsFn",
    "authorize_chunk_in_search_targets",
    "authorize_knowledge_in_search_targets",
    "effective_search_target_tag_ids",
    "filter_search_results_in_search_targets",
    "knowledge_ids_matching_any_tag",
    "resolve_authorized_source_refs",
    "search_target_is_whole_kb",
    "search_target_scope",
    "search_targets_allow_knowledge_id",
    "validate_knowledge_base_ids_in_search_targets",
]
