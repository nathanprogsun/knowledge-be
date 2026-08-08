"""Wiki health-check (lint) service — standalone module.

Ports the wiki lint contract: a comprehensive health check over a wiki
knowledge base (orphan pages, broken links, empty content, stale source
refs, and missing cross references), a derived 0-100 health score, and
an auto-fix pass that applies machine-safe repairs to fixable findings.

The module is deliberately standalone: it composes the already-merged
``WikiPageService``, the knowledge-base service, and an injectable
knowledge-liveness resolver instead of owning any repositories, so the
web layer can wire it from request-scoped services. Every page read goes
through the wiki service's streaming cursor walk, so resident memory
stays bounded regardless of knowledge-base size.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, ConfigDict

from src.common.exception import NotFoundError, ValidationError
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.core.knowledge.wiki.page_service import WikiPageService
from src.core.knowledge.wiki.types import (
    WIKI_PAGE_STATUS_ARCHIVED,
    WIKI_PAGE_TYPE_CONCEPT,
    WIKI_PAGE_TYPE_ENTITY,
    WIKI_PAGE_TYPE_INDEX,
    WikiStats,
)
from src.db.models.wiki_page import WikiPage

logger = logging.getLogger(__name__)

# Per-batch window for the streaming page walk. Wiki pages can carry
# multi-KB content blobs, so a bounded window keeps memory bounded while
# running per-page checks.
LINT_CURSOR_BATCH = 200

# A page body shorter than this (after trimming) counts as empty content.
LINT_MIN_CONTENT_LENGTH = 50

# ── Issue vocabulary ─────────────────────────────────────────────────

LINT_ISSUE_ORPHAN_PAGE = "orphan_page"
LINT_ISSUE_BROKEN_LINK = "broken_link"
LINT_ISSUE_STALE_REF = "stale_ref"
LINT_ISSUE_MISSING_CROSS_REF = "missing_cross_ref"
LINT_ISSUE_EMPTY_CONTENT = "empty_content"
LINT_ISSUE_DUPLICATE_SLUG = "duplicate_slug"

SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_ERROR = "error"


class WikiLintIssue(BaseModel):
    """A single lint finding on one page."""

    model_config = ConfigDict(frozen=True)

    type: str
    severity: str
    page_slug: str
    target_slug: str = ""
    description: str
    auto_fixable: bool = False


class WikiLintReport(BaseModel):
    """The complete lint report for one wiki knowledge base."""

    model_config = ConfigDict(frozen=True)

    knowledge_base_id: str
    issues: list[WikiLintIssue]
    health_score: int  # 0-100
    stats: WikiStats
    summary: str


# A resolver that reports whether a knowledge id is still live
# (non-deleted). The stale-ref check is skipped when none is injected.
KnowledgeResolver = Callable[[str], Awaitable[bool]]


def _wiki_enabled(kb: KnowledgeBaseInfo) -> bool:
    """Return whether the KB's indexing strategy enables the wiki pipeline."""
    strategy = kb.indexing_strategy or {}
    return strategy.get("wiki_enabled") is True


def remove_source_ref(refs: list[str], knowledge_id: str) -> list[str]:
    """Return ``refs`` without entries that reference ``knowledge_id``.

    Both the bare id and the legacy "id|title" prefix form are removed.
    The input list is not mutated.
    """
    prefix = knowledge_id + "|"
    return [ref for ref in refs if ref != knowledge_id and not ref.startswith(prefix)]


def compute_health_score(*, stats: WikiStats, issues: list[WikiLintIssue]) -> int:
    """Derive the 0-100 health score from stats and the collected issues.

    Orphan-heavy wikis, broken links, entirely link-less wikis, and empty
    pages each subtract points; the score is clamped at zero.
    """
    score = 100
    if stats.total_pages > 0:
        orphan_pct = stats.orphan_count / stats.total_pages * 100
        if orphan_pct > 50:
            score -= 25
        elif orphan_pct > 25:
            score -= 10

        broken_count = sum(1 for i in issues if i.type == LINT_ISSUE_BROKEN_LINK)
        score -= broken_count * 5

        if stats.total_links == 0 and stats.total_pages > 2:
            score -= 15

        empty_count = sum(1 for i in issues if i.type == LINT_ISSUE_EMPTY_CONTENT)
        score -= empty_count * 3
    return max(0, score)


def build_summary(issues: list[WikiLintIssue]) -> str:
    """Build the one-line report summary from the issue severity counts."""
    if not issues:
        return "Wiki is healthy! No issues found."
    error_count = sum(1 for i in issues if i.severity == SEVERITY_ERROR)
    warning_count = sum(1 for i in issues if i.severity == SEVERITY_WARNING)
    info_count = len(issues) - error_count - warning_count
    return (
        f"Found {len(issues)} issues: {error_count} errors, "
        f"{warning_count} warnings, {info_count} suggestions."
    )


class WikiLintService:
    """Wiki health-check service, composed from request-scoped services."""

    def __init__(
        self,
        *,
        wiki_service: WikiPageService,
        kb_service: KBService,
        knowledge_resolver: KnowledgeResolver | None = None,
    ) -> None:
        self._wiki_service = wiki_service
        self._kb_service = kb_service
        self._knowledge_resolver = knowledge_resolver

    async def run_lint(self, *, knowledge_base_id: str) -> WikiLintReport:
        """Run the full lint pass and return the report.

        Raises ``NotFoundError`` when the knowledge base is absent and
        ``ValidationError`` (``wiki.lint_kb_not_wiki``) when it is not a
        wiki-enabled knowledge base.
        """
        kb = await self._kb_service.get_knowledge_base_by_id_only(
            knowledge_base_id=knowledge_base_id
        )
        if not _wiki_enabled(kb):
            raise ValidationError(
                code="wiki.lint_kb_not_wiki",
                message=f"knowledge base {knowledge_base_id} is not a wiki type",
            )

        stats = await self._wiki_service.get_stats(knowledge_base_id=knowledge_base_id)
        live_slugs = await self._wiki_service.list_all_slugs(knowledge_base_id=knowledge_base_id)
        slug_set = set(live_slugs)

        issues: list[WikiLintIssue] = []
        # slug -> title for entity / concept pages; built in pass 1 so the
        # missing-cross-ref check (inherently O(entities x pages)) never
        # needs a second scan for candidates.
        entity_slugs: dict[str, str] = {}
        # kid -> live; cached across pages so each id is resolved once.
        knowledge_live: dict[str, bool] = {}

        cursor = ""
        while True:
            pages, next_cursor = await self._wiki_service.list_pages_cursor(
                knowledge_base_id=knowledge_base_id,
                cursor=cursor,
                limit=LINT_CURSOR_BATCH,
            )
            if not pages:
                break
            for page in pages:
                if page.page_type in (WIKI_PAGE_TYPE_ENTITY, WIKI_PAGE_TYPE_CONCEPT) and page.title:
                    entity_slugs[page.slug] = page.title

                issues.extend(self._orphan_issues(page))
                issues.extend(self._broken_link_issues(page, slug_set))
                issues.extend(self._empty_content_issues(page))
                if self._knowledge_resolver is not None and page.page_type != WIKI_PAGE_TYPE_INDEX:
                    issues.extend(
                        await self._stale_ref_issues(page, knowledge_live, knowledge_base_id)
                    )
            if not next_cursor:
                break
            cursor = next_cursor

        cursor = ""
        while True:
            pages, next_cursor = await self._wiki_service.list_pages_cursor(
                knowledge_base_id=knowledge_base_id,
                cursor=cursor,
                limit=LINT_CURSOR_BATCH,
            )
            if not pages:
                break
            for page in pages:
                issues.extend(self._missing_cross_ref_issues(page, entity_slugs))
            if not next_cursor:
                break
            cursor = next_cursor

        health_score = compute_health_score(stats=stats, issues=issues)
        summary = build_summary(issues)
        logger.info(
            "wiki lint: KB %s — health score %d/100, %d issues",
            knowledge_base_id,
            health_score,
            len(issues),
        )
        return WikiLintReport(
            knowledge_base_id=knowledge_base_id,
            issues=issues,
            health_score=health_score,
            stats=stats,
            summary=summary,
        )

    async def auto_fix(self, *, knowledge_base_id: str) -> int:
        """Attempt machine-safe fixes for fixable issues.

        Returns how many fixes actually persisted. When at least one fix
        landed, the bidirectional link graph is rebuilt afterwards.
        """
        report = await self.run_lint(knowledge_base_id=knowledge_base_id)

        fixers: dict[str, Callable[[str, WikiLintIssue], Awaitable[bool]]] = {
            LINT_ISSUE_BROKEN_LINK: self._fix_broken_link,
            LINT_ISSUE_EMPTY_CONTENT: self._fix_empty_content,
            LINT_ISSUE_STALE_REF: self._fix_stale_ref,
        }
        fixed = 0
        for issue in report.issues:
            fixer = fixers.get(issue.type)
            if fixer is None or not issue.auto_fixable:
                continue
            if await fixer(knowledge_base_id, issue):
                fixed += 1

        if fixed > 0:
            try:
                await self._wiki_service.rebuild_links(knowledge_base_id=knowledge_base_id)
            except NotFoundError:
                logger.warning(
                    "wiki auto-fix: KB %s — link rebuild aborted",
                    knowledge_base_id,
                )

        logger.info("wiki auto-fix: KB %s — fixed %d issues", knowledge_base_id, fixed)
        return fixed

    # ── Per-check helpers (pass 1) ───────────────────────────────────

    def _orphan_issues(self, page: WikiPage) -> list[WikiLintIssue]:
        """Flag pages with no inbound links, excluding the index page."""
        if page.page_type == WIKI_PAGE_TYPE_INDEX or page.in_links:
            return []
        return [
            WikiLintIssue(
                type=LINT_ISSUE_ORPHAN_PAGE,
                severity=SEVERITY_WARNING,
                page_slug=page.slug,
                description=(
                    f"Page '{page.title}' has no inbound links — it's disconnected from the wiki"
                ),
                auto_fixable=False,
            )
        ]

    def _broken_link_issues(self, page: WikiPage, slug_set: set[str]) -> list[WikiLintIssue]:
        """Flag outlinks pointing at slugs absent from the live set."""
        issues: list[WikiLintIssue] = []
        for out_link in page.out_links:
            if out_link in slug_set:
                continue
            issues.append(
                WikiLintIssue(
                    type=LINT_ISSUE_BROKEN_LINK,
                    severity=SEVERITY_ERROR,
                    page_slug=page.slug,
                    target_slug=out_link,
                    description=(
                        f"Page '{page.title}' links to [[{out_link}]] which does not exist"
                    ),
                    auto_fixable=True,
                )
            )
        return issues

    def _empty_content_issues(self, page: WikiPage) -> list[WikiLintIssue]:
        """Flag pages whose trimmed body is shorter than the minimum."""
        content = page.content.strip()
        if len(content) >= LINT_MIN_CONTENT_LENGTH:
            return []
        return [
            WikiLintIssue(
                type=LINT_ISSUE_EMPTY_CONTENT,
                severity=SEVERITY_WARNING,
                page_slug=page.slug,
                description=(f"Page '{page.title}' has very little content ({len(content)} chars)"),
                auto_fixable=True,
            )
        ]

    async def _stale_ref_issues(
        self,
        page: WikiPage,
        knowledge_live: dict[str, bool],
        knowledge_base_id: str,
    ) -> list[WikiLintIssue]:
        """Flag source refs pointing at soft-deleted knowledge.

        ``knowledge_base_id`` is unused today and kept for call-site
        symmetry; it documents the KB scope the resolver answers for.
        """
        issues: list[WikiLintIssue] = []
        resolver = self._knowledge_resolver
        if resolver is None:
            return issues
        for ref in page.source_refs:
            kid = ref.split("|", 1)[0] if "|" in ref else ref
            if not kid:
                continue
            if kid not in knowledge_live:
                try:
                    knowledge_live[kid] = await resolver(kid)
                except Exception:
                    # A failed lookup reads as dead, matching the reference
                    # behavior of treating query errors as missing rows.
                    logger.warning(
                        "wiki lint: KB %s — knowledge lookup failed for %s",
                        knowledge_base_id,
                        kid,
                    )
                    knowledge_live[kid] = False
            if knowledge_live[kid]:
                continue
            issues.append(
                WikiLintIssue(
                    type=LINT_ISSUE_STALE_REF,
                    severity=SEVERITY_ERROR,
                    page_slug=page.slug,
                    target_slug=kid,
                    description=f"Page '{page.title}' references deleted knowledge {kid}",
                    auto_fixable=True,
                )
            )
        return issues

    # ── Per-check helpers (pass 2) ───────────────────────────────────

    def _missing_cross_ref_issues(
        self, page: WikiPage, entity_slugs: dict[str, str]
    ) -> list[WikiLintIssue]:
        """Flag entity mentions that are not linked as wiki links."""
        if not entity_slugs:
            return []
        lower_content = page.content.lower()
        out_link_set = set(page.out_links)
        issues: list[WikiLintIssue] = []
        for slug, title in entity_slugs.items():
            if slug == page.slug or not title:
                continue
            if title.lower() not in lower_content:
                continue
            if slug in out_link_set:
                continue
            issues.append(
                WikiLintIssue(
                    type=LINT_ISSUE_MISSING_CROSS_REF,
                    severity=SEVERITY_INFO,
                    page_slug=page.slug,
                    target_slug=slug,
                    description=(
                        f"Page '{page.title}' mentions '{title}' but doesn't link to [[{slug}]]"
                    ),
                    auto_fixable=False,
                )
            )
        return issues

    # ── Auto-fix helpers ─────────────────────────────────────────────

    async def _fix_broken_link(self, knowledge_base_id: str, issue: WikiLintIssue) -> bool:
        """Degrade ``[[target]]`` to plain text so it no longer renders as a link."""
        if not issue.target_slug:
            return False
        page = await self._get_page_or_none(knowledge_base_id, issue.page_slug)
        if page is None:
            return False
        target = issue.target_slug
        new_content = page.content.replace("[[" + target + "]]", target)
        try:
            await self._wiki_service.update_auto_linked_content(
                page=page.model_copy(update={"content": new_content})
            )
        except NotFoundError:
            return False
        return True

    async def _fix_empty_content(self, knowledge_base_id: str, issue: WikiLintIssue) -> bool:
        """Archive pages with very little content instead of deleting them."""
        page = await self._get_page_or_none(knowledge_base_id, issue.page_slug)
        if page is None or page.page_type == WIKI_PAGE_TYPE_INDEX:
            return False
        try:
            await self._wiki_service.update_page(
                page=page.model_copy(update={"status": WIKI_PAGE_STATUS_ARCHIVED})
            )
        except NotFoundError:
            return False
        return True

    async def _fix_stale_ref(self, knowledge_base_id: str, issue: WikiLintIssue) -> bool:
        """Strip refs to deleted knowledge; delete the page when none remain."""
        if not issue.target_slug:
            return False
        page = await self._get_page_or_none(knowledge_base_id, issue.page_slug)
        if page is None or page.page_type == WIKI_PAGE_TYPE_INDEX:
            return False
        remaining = remove_source_ref(page.source_refs, issue.target_slug)
        if not remaining:
            try:
                await self._wiki_service.delete_page(
                    knowledge_base_id=knowledge_base_id, slug=page.slug
                )
            except NotFoundError:
                return False
            return True
        if len(remaining) != len(page.source_refs):
            try:
                await self._wiki_service.update_page_meta(
                    page=page.model_copy(update={"source_refs": remaining})
                )
            except NotFoundError:
                return False
            return True
        return False

    async def _get_page_or_none(self, knowledge_base_id: str, slug: str) -> WikiPage | None:
        """Return one page by (KB, slug), or ``None`` when it is gone."""
        try:
            return await self._wiki_service.get_page_by_slug(
                knowledge_base_id=knowledge_base_id, slug=slug
            )
        except NotFoundError:
            return None


__all__ = [
    "LINT_CURSOR_BATCH",
    "LINT_ISSUE_BROKEN_LINK",
    "LINT_ISSUE_DUPLICATE_SLUG",
    "LINT_ISSUE_EMPTY_CONTENT",
    "LINT_ISSUE_MISSING_CROSS_REF",
    "LINT_ISSUE_ORPHAN_PAGE",
    "LINT_ISSUE_STALE_REF",
    "LINT_MIN_CONTENT_LENGTH",
    "SEVERITY_ERROR",
    "SEVERITY_INFO",
    "SEVERITY_WARNING",
    "KnowledgeResolver",
    "WikiLintIssue",
    "WikiLintReport",
    "WikiLintService",
    "build_summary",
    "compute_health_score",
    "remove_source_ref",
]
