"""Shared routing and scope infrastructure for the wiki agent tools.

The wiki tools all resolve slugs against a server-owned set of knowledge
bases. This module holds the vocabulary they share:

- :class:`WikiScope` — the effective retrieval scope of one wiki KB
  (an optional source-document / tag whitelist layered on top of the KB);
- :class:`WikiRouteResolver` — request-local slug → KB provenance so a
  slug echoed from a page body resolves to the same KB the model read it
  from, without a second round trip;
- the unique-resolution / create-target / issue-resolution helpers that
  refuse ambiguous writes instead of silently mutating the first KB;
- the slug normalisation and incoming-link mutation helpers used by the
  write tools (delete / rename) to keep machine-maintained links coherent
  and to roll them back when a write aborts;
- the structural :class:`WikiPageServiceProtocol` seam the tools execute
  against (satisfied by the merged wiki page service).

Everything here is pure domain logic over injected seams; no tool touches
storage or an LLM directly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from src.ai.embedding.base import Context
from src.common.exception import NotFoundError, ValidationError
from src.core.agents.tools.scope_auth import (
    KnowledgeLookup,
    KnowledgeTagsFetcher,
    knowledge_ids_matching_any_tag,
    search_target_is_whole_kb,
    search_target_scope,
)
from src.core.agents.tools.search_target import SearchTargets
from src.core.agents.tools.text_utils import dedup_non_empty_strings
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.core.knowledge.wiki.issues import (
    WikiPageIssue,
    WikiPageIssueRepository,
    list_issues,
)
from src.core.knowledge.wiki.types import (
    WIKI_PAGE_TYPE_INDEX,
    WIKI_PAGE_TYPE_SUMMARY,
    WikiIndexResponse,
)
from src.db.models.wiki_page import WikiPage


#: Slug that is not found in any allowed knowledge base.
class WikiPageNotFoundInScopeError(Exception):
    """No page matched the slug in any allowed scope (distinct from a raw miss)."""


#: Slug that resolved in more than one allowed knowledge base.
class WikiPageAmbiguousError(Exception):
    """The slug exists in several allowed scopes; a write cannot pick one."""


@runtime_checkable
class WikiPageServiceProtocol(Protocol):
    """Structural seam for the merged wiki page service (mockable).

    Only the methods the wiki agent tools need are declared. ``get_page_by_slug``
    raises ``NotFoundError`` (code ``wiki.page_not_found``) when the page is
    absent in the given knowledge base.
    """

    async def get_page_by_slug(self, *, knowledge_base_id: str, slug: str) -> WikiPage: ...

    async def create_page(
        self,
        *,
        page: WikiPage,
        edit_source: str = "",
        editor_id: str = "",
    ) -> WikiPage: ...

    async def update_page(
        self,
        *,
        page: WikiPage,
        edit_source: str = "",
        editor_id: str = "",
    ) -> WikiPage: ...

    async def update_auto_linked_content(self, *, page: WikiPage) -> WikiPage: ...

    async def delete_page(self, *, knowledge_base_id: str, slug: str) -> None: ...

    async def search_pages(
        self, *, knowledge_base_id: str, query: str, limit: int = 10
    ) -> list[WikiPage]: ...

    async def get_index_view(
        self,
        *,
        knowledge_base_id: str,
        tenant_id: int,
        page_types: list[str] | None = None,
        limit: int = 0,
        cursor: str = "",
    ) -> WikiIndexResponse: ...

    async def inject_cross_links(
        self, *, knowledge_base_id: str, affected_slugs: list[str]
    ) -> int: ...

    async def rebuild_index_page(self, *, knowledge_base_id: str) -> None: ...

    async def repair_content_links(
        self, *, knowledge_base_id: str, self_slug: str, content: str
    ) -> tuple[str, bool]: ...


#: Loads one knowledge-base record so a write can stamp the owning tenant.
@runtime_checkable
class GraphKbLoader(Protocol):
    """Resolves a knowledge-base record by id (``None`` when absent)."""

    async def load(self, *, knowledge_base_id: str) -> KnowledgeBaseInfo | None: ...


@dataclass(frozen=True, slots=True)
class WikiScope:
    """Effective retrieval scope for one wiki knowledge base.

    ``knowledge_ids`` / ``tag_ids`` are optional whitelists: when either is
    non-empty, a wiki page is only surfaced when at least one of its source
    references lies in the whitelist / carries any of the tags. Both are
    server-enforced and never exposed to the model as tool arguments.
    """

    knowledge_base_id: str
    knowledge_ids: list[str] = field(default_factory=list)
    tag_ids: list[str] = field(default_factory=list)


def new_wiki_scopes_from_kb_ids(kb_ids: list[str]) -> list[WikiScope]:
    """Build one unrestricted scope per knowledge-base id.

    Deduplicates defensively: unique-page resolution counts one hit per
    scope as a distinct owner, so a duplicated KB id would misreport a
    unique slug as ambiguous.
    """
    return [WikiScope(knowledge_base_id=kb_id) for kb_id in dedup_non_empty_strings(kb_ids)]


def new_wiki_scopes_from_search_targets(
    search_targets: SearchTargets,
    wiki_kb_ids: list[str],
) -> list[WikiScope]:
    """Merge the search targets of each wiki KB into one scope per KB.

    Search targets are alternatives selected by the user, so document and
    tag constraints are combined as a union; an unrestricted whole-KB
    target supersedes narrower targets for that KB. A malformed empty
    document target is skipped so it can never silently become whole-KB
    authorization.
    """
    allowed = set(dedup_non_empty_strings(wiki_kb_ids))
    unrestricted: set[str] = set()
    accumulated: dict[str, WikiScope] = {}
    for target in search_targets:
        if target is None or target.knowledge_base_id not in allowed:
            continue
        if target.knowledge_base_id in unrestricted:
            continue
        knowledge_ids, tag_ids = search_target_scope(target)
        whole_kb = search_target_is_whole_kb(target)
        if not whole_kb and not knowledge_ids and not tag_ids:
            continue
        if whole_kb:
            unrestricted.add(target.knowledge_base_id)
            continue
        scope = accumulated.get(target.knowledge_base_id)
        if scope is None:
            scope = WikiScope(knowledge_base_id=target.knowledge_base_id)
        accumulated[target.knowledge_base_id] = WikiScope(
            knowledge_base_id=scope.knowledge_base_id,
            knowledge_ids=[*scope.knowledge_ids, *knowledge_ids],
            tag_ids=[*scope.tag_ids, *tag_ids],
        )

    scopes: list[WikiScope] = []
    for kb_id in dedup_non_empty_strings(wiki_kb_ids):
        if kb_id in unrestricted:
            scopes.append(WikiScope(knowledge_base_id=kb_id))
            continue
        scope = accumulated.get(kb_id)
        if scope is None:
            continue
        scopes.append(
            WikiScope(
                knowledge_base_id=kb_id,
                knowledge_ids=dedup_non_empty_strings(scope.knowledge_ids),
                tag_ids=dedup_non_empty_strings(scope.tag_ids),
            )
        )
    return scopes


def scope_knowledge_filter(scope: WikiScope) -> tuple[dict[str, bool], bool]:
    """Return ``(allowed_set, has_filter)`` for the scope's document whitelist.

    When ``has_filter`` is ``False`` no document-level filtering applies to
    pages of this KB.
    """
    ids = dedup_non_empty_strings(scope.knowledge_ids)
    if not ids:
        return {}, False
    return dict.fromkeys(ids, True), True


def extract_source_knowledge_ids(page: WikiPage | None) -> list[str]:
    """Parse ``source_refs`` (``uuid`` or ``uuid|title``) into bare ids."""
    if page is None or not page.source_refs:
        return []
    ids: list[str] = []
    for ref in page.source_refs:
        kid = ref
        pipe_idx = ref.find("|")
        if pipe_idx > 0:
            kid = ref[:pipe_idx]
        if kid:
            ids.append(kid)
    return ids


def is_structural_page(page: WikiPage | None) -> bool:
    """Whether a page is the wiki-level index rather than a content page.

    The index is never filtered by document scope because it describes wiki
    topology, not a specific source.
    """
    return page is not None and page.page_type == WIKI_PAGE_TYPE_INDEX


def register_linked_slugs(
    found_kbs: dict[str, list[str]],
    page: WikiPage | None,
    kb_id: str,
) -> None:
    """Record the owning KB for every slug the page links to / is linked from.

    Wiki links are KB-local, so sharing the KB mapping with neighbours lets
    the frontend resolve a slug echoed from a page body without a guess.
    """
    if page is None or not kb_id:
        return

    def add(slug: str) -> None:
        if not slug:
            return
        if kb_id in found_kbs.get(slug, []):
            return
        found_kbs.setdefault(slug, []).append(kb_id)

    for slug in page.out_links:
        add(slug)
    for slug in page.in_links:
        add(slug)


def page_intersects_knowledge_ids(page: WikiPage, allowed: dict[str, bool]) -> bool:
    """Whether any of the page's source documents is in the whitelist."""
    if not allowed:
        return True
    return any(allowed.get(kid, False) for kid in extract_source_knowledge_ids(page))


async def page_passes_wiki_scope(
    ctx: Context,
    page: WikiPage,
    scope: WikiScope,
    fetch_tags: KnowledgeTagsFetcher | None,
) -> bool:
    """Whether ``page`` may be surfaced under ``scope``.

    Under a document/tag-constrained scope every surfaced page must prove
    provenance: structural pages and uncited pages describe the whole wiki
    and cannot be attributed to the selected subset.
    """
    allowed, has_knowledge_filter = scope_knowledge_filter(scope)
    tag_ids = dedup_non_empty_strings(scope.tag_ids)
    if not has_knowledge_filter and not tag_ids:
        return True
    if is_structural_page(page):
        return False
    source_ids = extract_source_knowledge_ids(page)
    if not source_ids:
        return False
    if has_knowledge_filter and page_intersects_knowledge_ids(page, allowed):
        return True
    if not tag_ids:
        return False
    fetch_fn = fetch_tags.get_knowledge_tags if fetch_tags is not None else None
    matches = await knowledge_ids_matching_any_tag(ctx, source_ids, tag_ids, fetch_fn)
    return bool(matches)


def seen_link_key(kb_id: str, slug: str) -> str:
    """Dedupe key scoped to a knowledge base so equal slugs in different KBs
    are not collapsed into a single already-seen entry."""
    return kb_id + "\x00" + slug


class WikiRouteResolver:
    """Request-local provenance for wiki slugs.

    Server-side routing state, not a model-handle registry: only KBs already
    in the current scope can ever be returned.
    """

    def __init__(self) -> None:
        self._by_slug: dict[str, set[str]] = {}

    def remember(self, slug: str, kb_id: str) -> None:
        """Record ``kb_id`` as an owner of ``slug``."""
        if not slug or not kb_id:
            return
        self._by_slug.setdefault(slug, set()).add(kb_id)

    def forget(self, slug: str, kb_id: str) -> None:
        """Drop ``kb_id`` from ``slug``'s owners."""
        if not slug or not kb_id:
            return
        owners = self._by_slug.get(slug)
        if owners is None:
            return
        owners.discard(kb_id)
        if not owners:
            del self._by_slug[slug]

    def remember_page(self, page: WikiPage, kb_id: str) -> None:
        """Record the page's slug and every neighbour slug under ``kb_id``."""
        if page is None or not kb_id:
            return
        self.remember(page.slug, kb_id)
        for slug in page.out_links:
            self.remember(slug, kb_id)
        for slug in page.in_links:
            self.remember(slug, kb_id)

    def scopes_for_slug(self, slug: str, scopes: list[WikiScope]) -> list[WikiScope]:
        """Return cached owners still present in ``scopes``, preserving order.

        An empty result means the caller should search every scope.
        """
        if not slug or not scopes:
            return []
        owners = self._by_slug.get(slug)
        if not owners:
            return []
        return [scope for scope in scopes if scope.knowledge_base_id in owners]


def scopes_outside_kbs(scopes: list[WikiScope], excluded: list[WikiScope]) -> list[WikiScope]:
    """Return ``scopes`` minus the knowledge bases of ``excluded``."""
    if not excluded:
        return scopes
    excluded_kbs = {scope.knowledge_base_id for scope in excluded}
    return [scope for scope in scopes if scope.knowledge_base_id not in excluded_kbs]


def first_wiki_route(routes: list[WikiRouteResolver] | None) -> WikiRouteResolver:
    """Return the first non-``None`` resolver or a fresh one."""
    if routes and routes[0] is not None:
        return routes[0]
    return WikiRouteResolver()


async def resolve_unique_wiki_page(
    ctx: Context,
    service: WikiPageServiceProtocol,
    slug: str,
    kb_ids: list[str],
    routes: WikiRouteResolver,
) -> tuple[WikiPage, str]:
    """Resolve ``slug`` to exactly one page within the allowed scopes.

    Every allowed KB is checked (cached provenance only affects order) and
    ambiguous slugs are refused instead of silently mutating the first KB.
    Raises :class:`WikiPageNotFoundInScopeError`, :class:`WikiPageAmbiguousError`,
    or a domain error for backend failures.
    """
    slug = slug.strip()
    if not slug:
        raise ValidationError(code="wiki.tool_slug_required", message="slug is required")
    scopes = new_wiki_scopes_from_kb_ids(kb_ids)
    preferred = routes.scopes_for_slug(slug, scopes)
    ordered = [*preferred, *scopes_outside_kbs(scopes, preferred)]
    hits: list[tuple[WikiPage, str]] = []
    for scope in ordered:
        kb_id = scope.knowledge_base_id
        if not kb_id:
            continue
        try:
            page = await service.get_page_by_slug(knowledge_base_id=kb_id, slug=slug)
        except NotFoundError:
            continue
        if page is None:
            continue
        if page.knowledge_base_id and page.knowledge_base_id != kb_id:
            raise ValidationError(
                code="wiki.tool_kb_mismatch",
                message=(
                    f"wiki page {slug} returned knowledge base {page.knowledge_base_id} "
                    f"while resolving allowed scope {kb_id}"
                ),
            )
        hits.append((page, kb_id))
        routes.remember_page(page, kb_id)
    if not hits:
        raise WikiPageNotFoundInScopeError(slug)
    if len(hits) == 1:
        return hits[0][0], hits[0][1]
    owners = [kb_id for _, kb_id in hits]
    raise WikiPageAmbiguousError(slug, owners)


def resolve_wiki_create_kb(
    slug: str,
    kb_ids: list[str],
    routes: WikiRouteResolver,
    *server_hints: str,
) -> str:
    """Select a creation target only when server-side context is unambiguous.

    Accepts one cached provenance owner, or one wiki KB in scope, or the
    source-ref hints when they agree; otherwise raises a domain error.
    """
    scopes = new_wiki_scopes_from_kb_ids(kb_ids)
    preferred = routes.scopes_for_slug(slug.strip(), scopes)
    allowed = {scope.knowledge_base_id for scope in scopes}
    candidates: list[str] = []
    for scope in preferred:
        candidates.append(scope.knowledge_base_id)
    for kb_id in server_hints:
        if kb_id in allowed:
            candidates.append(kb_id)
    candidates = dedup_non_empty_strings(candidates)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValidationError(
            code="wiki.tool_create_kb_conflict",
            message=(
                f"cannot choose a knowledge base for new wiki page {slug}: "
                f"server provenance conflicts across {', '.join(candidates)}"
            ),
        )
    if len(scopes) == 1:
        return scopes[0].knowledge_base_id
    raise ValidationError(
        code="wiki.tool_create_kb_ambiguous",
        message=f"cannot choose a knowledge base for new wiki page {slug} from {len(scopes)} allowed scopes",
    )


async def wiki_knowledge_bases_for_source_refs(
    ctx: Context,
    refs: list[str],
    knowledge_service: KnowledgeLookup | None,
    allowed_kb_ids: list[str],
) -> list[str]:
    """Return the deduplicated KBs that own the given source-document refs."""
    if not refs:
        return []
    if knowledge_service is None:
        raise ValidationError(
            code="wiki.tool_knowledge_service_unavailable",
            message="knowledge service is unavailable",
        )
    allowed = set(dedup_non_empty_strings(allowed_kb_ids))
    kb_ids: list[str] = []
    for ref in refs:
        knowledge_id = ref.split("|", 1)[0].strip()
        if not knowledge_id:
            continue
        knowledge = await knowledge_service.get_document_by_id_only(id=knowledge_id)
        if knowledge is None:
            raise ValidationError(
                code="wiki.tool_source_doc_not_found",
                message=f"failed to resolve source document {knowledge_id}",
            )
        if knowledge.knowledge_base_id not in allowed:
            raise ValidationError(
                code="wiki.tool_source_kb_unauthorized",
                message=(
                    f"source document {knowledge_id} belongs to non-Wiki or "
                    f"unauthorized knowledge base {knowledge.knowledge_base_id}"
                ),
            )
        kb_ids.append(knowledge.knowledge_base_id)
    return dedup_non_empty_strings(kb_ids)


async def resolve_source_refs(
    ctx: Context,
    knowledge_service: KnowledgeLookup | None,
    refs: list[str],
) -> list[str]:
    """Enrich plain knowledge UUIDs to ``uuid|title`` refs.

    Refs already in ``uuid|title`` form are left unchanged; documents that
    cannot be resolved keep their bare id.
    """
    if not refs or knowledge_service is None:
        return list(refs)
    resolved: list[str] = []
    for ref in refs:
        if "|" in ref:
            resolved.append(ref)
            continue
        try:
            knowledge = await knowledge_service.get_document_by_id_only(id=ref)
        except Exception:
            knowledge = None
        if knowledge is None:
            resolved.append(ref)
            continue
        title = knowledge.title or ""
        if not title:
            title = knowledge.file_name or ""
        if title:
            resolved.append(ref + "|" + title)
        else:
            resolved.append(ref)
    return resolved


async def resolve_wiki_issue(
    ctx: Context,
    issue_repo: WikiPageIssueRepository,
    issue_id: str,
    kb_ids: list[str],
) -> WikiPageIssue:
    """Resolve ``issue_id`` to a single issue within the allowed scopes."""
    issue_id = issue_id.strip()
    if not issue_id:
        raise ValidationError(
            code="wiki.tool_issue_id_required",
            message="issue_id is required",
        )
    match: WikiPageIssue | None = None
    for kb_id in dedup_non_empty_strings(kb_ids):
        issues = await list_issues(
            issue_repo=issue_repo,
            knowledge_base_id=kb_id,
            slug="",
            status="",
        )
        for issue in issues:
            if issue is None or issue.id != issue_id:
                continue
            if issue.knowledge_base_id and issue.knowledge_base_id != kb_id:
                raise ValidationError(
                    code="wiki.tool_issue_kb_mismatch",
                    message=(
                        f"issue_id {issue_id} returned knowledge base "
                        f"{issue.knowledge_base_id} while resolving allowed scope {kb_id}"
                    ),
                )
            if match is not None:
                raise ValidationError(
                    code="wiki.tool_issue_ambiguous",
                    message=f"issue_id {issue_id} is ambiguous across current Wiki scopes",
                )
            match = issue
    if match is None:
        raise ValidationError(
            code="wiki.tool_issue_out_of_scope",
            message=f"issue_id {issue_id} is not within the current Wiki scope",
        )
    return match


def normalize_and_validate_wiki_slug(raw: str) -> tuple[str, str]:
    """Normalize a model-supplied slug and reject malformed ones.

    Returns ``(normalized_slug, "")`` on success or ``("", error_message)``.
    A valid slug contains only lowercase ASCII letters, digits, ``-``, ``/``,
    or CJK characters, has no leading/trailing/duplicate ``/``, and is
    non-empty.
    """
    normalized = raw.lower().strip().replace(" ", "-")
    if not normalized:
        return "", "slug is required and must be non-empty"
    if "//" in normalized or normalized.startswith("/") or normalized.endswith("/"):
        return "", f"invalid slug {raw!r}: '/' separators are malformed"
    for char in normalized:
        if char in "abcdefghijklmnopqrstuvwxyz0123456789-/":
            continue
        if 0x4E00 <= ord(char) <= 0x9FFF:
            continue
        return "", (
            f"invalid slug {raw!r}: character {char!r} is not allowed "
            "(use lowercase letters, digits, '-', '/', or CJK)"
        )
    return normalized, ""


def is_summary_namespace(slug: str) -> bool:
    """Whether a slug lives in the system-owned summary namespace."""
    return slug.startswith(WIKI_PAGE_TYPE_SUMMARY + "/")


# ── Incoming-link mutation helpers (delete / rename rollback) ─────────


@dataclass(frozen=True, slots=True)
class AppliedWikiContentChange:
    """A machine-maintained content rewrite already persisted."""

    page: WikiPage
    original_content: str


WikiContentRewrite = Callable[[str], tuple[str, bool]]


async def apply_incoming_wiki_content_rewrite(
    ctx: Context,
    service: WikiPageServiceProtocol,
    kb_id: str,
    in_links: list[str],
    rewrite: WikiContentRewrite,
) -> tuple[list[AppliedWikiContentChange], list[str], str]:
    """Rewrite machine-maintained links in every incoming page.

    Stops on the first failure and returns the already-applied changes so the
    caller can compensate. The third return value is an empty string on
    success or an error message.
    """
    changes: list[AppliedWikiContentChange] = []
    updated_slugs: list[str] = []
    for source_slug in dedup_non_empty_strings(in_links):
        try:
            page = await service.get_page_by_slug(knowledge_base_id=kb_id, slug=source_slug)
        except NotFoundError:
            page = None
        if page is None:
            return changes, updated_slugs, f"load incoming page {source_slug}: empty result"
        updated_content, changed = rewrite(page.content)
        if not changed:
            continue
        original = page.content
        try:
            await service.update_auto_linked_content(
                page=page.model_copy(update={"content": updated_content})
            )
        except Exception as exc:
            return changes, updated_slugs, f"update incoming page {source_slug}: {exc}"
        changes.append(
            AppliedWikiContentChange(page=page, original_content=original)
        )
        updated_slugs.append(source_slug)
    return changes, updated_slugs, ""


async def rollback_wiki_content_changes(
    ctx: Context,
    service: WikiPageServiceProtocol,
    changes: list[AppliedWikiContentChange],
) -> str:
    """Restore the original content of every applied change, newest first.

    Returns an empty string on success or an error message.
    """
    failures: list[str] = []
    for change in reversed(changes):
        try:
            await service.update_auto_linked_content(
                page=change.page.model_copy(update={"content": change.original_content})
            )
        except Exception as exc:
            failures.append(f"{change.page.slug}: {exc}")
    if failures:
        return f"failed to roll back incoming pages: {'; '.join(failures)}"
    return ""


def join_wiki_mutation_errors(primary: str, extras: list[str]) -> str:
    """Join a primary error with optional extra errors."""
    parts = [primary]
    for extra in extras:
        if extra:
            parts.append(extra)
    return "; ".join(parts)


__all__ = [
    "AppliedWikiContentChange",
    "GraphKbLoader",
    "WikiContentRewrite",
    "WikiPageAmbiguousError",
    "WikiPageNotFoundInScopeError",
    "WikiPageServiceProtocol",
    "WikiRouteResolver",
    "WikiScope",
    "apply_incoming_wiki_content_rewrite",
    "extract_source_knowledge_ids",
    "first_wiki_route",
    "is_structural_page",
    "is_summary_namespace",
    "join_wiki_mutation_errors",
    "new_wiki_scopes_from_kb_ids",
    "new_wiki_scopes_from_search_targets",
    "normalize_and_validate_wiki_slug",
    "page_intersects_knowledge_ids",
    "page_passes_wiki_scope",
    "register_linked_slugs",
    "resolve_source_refs",
    "resolve_unique_wiki_page",
    "resolve_wiki_create_kb",
    "resolve_wiki_issue",
    "rollback_wiki_content_changes",
    "scope_knowledge_filter",
    "scopes_outside_kbs",
    "seen_link_key",
    "wiki_knowledge_bases_for_source_refs",
]
