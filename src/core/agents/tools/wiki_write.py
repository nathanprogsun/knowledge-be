"""Wiki write tools: create/overwrite, delete, rename, and text replacement.

These tools mutate wiki pages through the merged wiki page service and are
the only agent surface that writes page bodies. They share a strict routing
boundary — a slug that resolves in more than one knowledge base is refused
instead of silently mutating the first — and a strict slug normaliser so a
mangled model-supplied slug can never persist an unreachable page.

Every write is attributed to the agent via ``edit_source`` so revision
history distinguishes agent edits from pipeline/user ones. Delete and
rename also rewrite the incoming links of the affected pages and roll the
machine-maintained link edits back when the mutation aborts.
"""

from __future__ import annotations

import json
import re
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from src.ai.embedding.base import Context
from src.common.exception import ApplicationError
from src.common.json import JsonObject, JsonValue
from src.core.agents.tools.base import ToolDefinition, ToolResult
from src.core.agents.tools.scope_auth import (
    KnowledgeLookup,
    resolve_authorized_source_refs,
)
from src.core.agents.tools.search_target import SearchTargets
from src.core.agents.tools.text_utils import dedup_non_empty_strings
from src.core.agents.tools.wiki_route import (
    GraphKbLoader,
    WikiPageAmbiguousError,
    WikiPageNotFoundInScopeError,
    WikiPageServiceProtocol,
    WikiRouteResolver,
    apply_incoming_wiki_content_rewrite,
    is_summary_namespace,
    join_wiki_mutation_errors,
    normalize_and_validate_wiki_slug,
    resolve_source_refs,
    resolve_unique_wiki_page,
    resolve_wiki_create_kb,
    rollback_wiki_content_changes,
    wiki_knowledge_bases_for_source_refs,
)
from src.core.knowledge.wiki.types import (
    WIKI_EDIT_SOURCE_AGENT,
    WIKI_PAGE_STATUS_PUBLISHED,
    WIKI_PAGE_TYPE_SUMMARY,
)
from src.db.models.wiki_page import WikiPage

WIKI_WRITE_PAGE_TOOL_NAME = "wiki_write_page"
WIKI_DELETE_PAGE_TOOL_NAME = "wiki_delete_page"
WIKI_RENAME_PAGE_TOOL_NAME = "wiki_rename_page"
WIKI_REPLACE_TEXT_TOOL_NAME = "wiki_replace_text"

WIKI_WRITE_PAGE_TOOL_DESCRIPTION = (
    "Create a new Wiki page or completely overwrite an existing one. Automatically handles outbound links."
)

WIKI_WRITE_PAGE_TOOL_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": "The slug of the Wiki page (e.g. 'entity/hunyuan-damoxing')",
            },
            "title": {"type": "string", "description": "The title of the page"},
            "summary": {
                "type": "string",
                "description": "A one-sentence summary for the index listing",
            },
            "content": {
                "type": "string",
                "description": "The FULL, complete Markdown content of the page. Do NOT use placeholders.",
            },
            "page_type": {
                "type": "string",
                "description": "The page type, e.g., 'summary', 'entity', 'concept', 'synthesis', 'comparison'",
            },
            "aliases": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "A list of aliases for the page (optional). If provided, these will "
                    "COMPLETELY REPLACE the existing aliases of the page."
                ),
            },
            "source_refs": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "A list of short dN source document IDs that contributed to this page. If provided, "
                    "these will COMPLETELY REPLACE the existing source_refs of the page."
                ),
            },
        },
        "required": ["slug", "title", "summary", "content", "page_type"],
    },
    ensure_ascii=False,
)

WIKI_DELETE_PAGE_TOOL_DESCRIPTION = (
    "Delete a Wiki page. Automatically cleans up incoming links on other pages to prevent dead links."
)

WIKI_DELETE_PAGE_TOOL_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": "The slug of the Wiki page to delete",
            },
        },
        "required": ["slug"],
    },
    ensure_ascii=False,
)

WIKI_RENAME_PAGE_TOOL_DESCRIPTION = (
    "Rename a Wiki page's slug. Automatically cascades the new slug to all pages that linked to the old one."
)

WIKI_RENAME_PAGE_TOOL_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": "The current slug of the Wiki page",
            },
            "new_slug": {
                "type": "string",
                "description": "The new slug for the page",
            },
        },
        "required": ["slug", "new_slug"],
    },
    ensure_ascii=False,
)

WIKI_REPLACE_TEXT_TOOL_DESCRIPTION = (
    "Replace specific exact text in a Wiki page. Ideal for minor corrections."
)

WIKI_REPLACE_TEXT_TOOL_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "slug": {"type": "string", "description": "The slug of the Wiki page"},
            "old_text": {
                "type": "string",
                "description": "The exact text to find and replace",
            },
            "new_text": {"type": "string", "description": "The new text to insert"},
            "source_refs": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "An optional list of short dN source document IDs that justify this change. "
                    "If provided, these will COMPLETELY REPLACE the existing source_refs of the page."
                ),
            },
        },
        "required": ["slug", "old_text", "new_text"],
    },
    ensure_ascii=False,
)


def build_wiki_write_page_definition() -> ToolDefinition:
    """Return the default tool definition for the wiki-write-page tool."""
    return ToolDefinition(
        name=WIKI_WRITE_PAGE_TOOL_NAME,
        description=WIKI_WRITE_PAGE_TOOL_DESCRIPTION,
        parameters=WIKI_WRITE_PAGE_TOOL_SCHEMA,
    )


def build_wiki_delete_page_definition() -> ToolDefinition:
    """Return the default tool definition for the wiki-delete-page tool."""
    return ToolDefinition(
        name=WIKI_DELETE_PAGE_TOOL_NAME,
        description=WIKI_DELETE_PAGE_TOOL_DESCRIPTION,
        parameters=WIKI_DELETE_PAGE_TOOL_SCHEMA,
    )


def build_wiki_rename_page_definition() -> ToolDefinition:
    """Return the default tool definition for the wiki-rename-page tool."""
    return ToolDefinition(
        name=WIKI_RENAME_PAGE_TOOL_NAME,
        description=WIKI_RENAME_PAGE_TOOL_DESCRIPTION,
        parameters=WIKI_RENAME_PAGE_TOOL_SCHEMA,
    )


def build_wiki_replace_text_definition() -> ToolDefinition:
    """Return the default tool definition for the wiki-replace-text tool."""
    return ToolDefinition(
        name=WIKI_REPLACE_TEXT_TOOL_NAME,
        description=WIKI_REPLACE_TEXT_TOOL_DESCRIPTION,
        parameters=WIKI_REPLACE_TEXT_TOOL_SCHEMA,
    )


@dataclass(frozen=True, slots=True)
class _WritePageInput:
    """Parsed input shared by the write tools."""

    slug: str = ""
    title: str = ""
    summary: str = ""
    content: str = ""
    page_type: str = ""
    aliases: list[str] | None = None
    source_refs: list[str] | None = None
    old_text: str = ""
    new_text: str = ""
    new_slug: str = ""


def _parse_args(args: str) -> JsonObject:
    try:
        raw = json.loads(args)
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def _as_str(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


def _as_str_list(value: JsonValue) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item != ""]
    return None


def _truncate_preview(text: str, max_runes: int) -> str:
    """Truncate ``text`` to ``max_runes`` code points, appending ``...``."""
    if len(text) <= max_runes:
        return text
    return text[:max_runes] + "..."


class WikiWritePageTool:
    """Creates a new wiki page or completely overwrites an existing one."""

    def __init__(
        self,
        *,
        definition: ToolDefinition,
        wiki_service: WikiPageServiceProtocol,
        kb_ids: list[str],
        routes: WikiRouteResolver | None = None,
        knowledge_service: KnowledgeLookup | None = None,
        search_targets: SearchTargets | None = None,
        kb_loader: GraphKbLoader | None = None,
    ) -> None:
        self._definition = definition
        self._wiki_service = wiki_service
        self._kb_ids = dedup_non_empty_strings(kb_ids)
        self._routes = routes if routes is not None else WikiRouteResolver()
        self._knowledge_service = knowledge_service
        self._search_targets = search_targets
        self._kb_loader = kb_loader
        #: Presence of search targets — not their length — enables the agent
        #: authorization boundary for source_refs; an agent turn with no
        #: target must reject every source.

    def name(self) -> str:
        return self._definition.name

    def description(self) -> str:
        return self._definition.description

    def parameters(self) -> str:
        return self._definition.parameters

    async def execute(self, ctx: Context, args: str) -> ToolResult:
        """Create or fully overwrite a wiki page."""
        raw = _parse_args(args)
        params = _WritePageInput(
            slug=_as_str(raw.get("slug")),
            title=_as_str(raw.get("title")),
            summary=_as_str(raw.get("summary")),
            content=_as_str(raw.get("content")),
            page_type=_as_str(raw.get("page_type")),
            aliases=_as_str_list(raw.get("aliases")),
            source_refs=_as_str_list(raw.get("source_refs")),
        )

        if not self._kb_ids:
            return ToolResult(success=False, error="No knowledge bases available for editing")
        if not params.title or not params.page_type or not params.content or not params.summary:
            return ToolResult(
                success=False,
                error="title, summary, content, and page_type are required for write action",
            )

        normalized_slug, slug_error = normalize_and_validate_wiki_slug(params.slug)
        if slug_error:
            return ToolResult(success=False, error=slug_error)
        params = replace(params, slug=normalized_slug)

        resolved_refs: list[str] | None = None
        if params.source_refs is not None:
            resolved_refs = await self._resolve_refs(ctx, params.source_refs)
            if resolved_refs is None:
                return ToolResult(
                    success=False,
                    error="Invalid source_refs: source document not within the current scope",
                )

        existing_page: WikiPage | None = None
        kb_id = ""
        try:
            existing_page, kb_id = await resolve_unique_wiki_page(
                ctx, self._wiki_service, params.slug, self._kb_ids, self._routes
            )
        except WikiPageNotFoundInScopeError:
            existing_page = None
            try:
                source_kb_hints = await wiki_knowledge_bases_for_source_refs(
                    ctx, resolved_refs or [], self._knowledge_service, self._kb_ids
                )
            except ApplicationError as exc:
                return ToolResult(success=False, error=f"Failed to resolve source_refs routing: {exc.message}")
            try:
                kb_id = resolve_wiki_create_kb(
                    params.slug, self._kb_ids, self._routes, *source_kb_hints
                )
            except ApplicationError as exc:
                return ToolResult(success=False, error=f"Failed to resolve wiki target: {exc.message}")
        except WikiPageAmbiguousError as exc:
            return ToolResult(success=False, error=f"Failed to resolve wiki target: {exc}")
        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to resolve wiki target: {exc}")

        if existing_page is None and (
            is_summary_namespace(params.slug) or params.page_type.lower() == WIKI_PAGE_TYPE_SUMMARY
        ):
            return ToolResult(
                success=False,
                error=(
                    "summary pages are generated automatically from source documents and cannot be created manually. "
                    "Use page_type 'synthesis'/'comparison'/'entity'/'concept' for authored pages, "
                    "or target an existing summary page to update it."
                ),
            )

        # Auto-repair dead [[slug]] references in the body before persisting.
        # Best-effort — never block the write.
        repaired = params.content
        try:
            repaired, changed = await self._wiki_service.repair_content_links(
                knowledge_base_id=kb_id,
                self_slug=params.slug,
                content=params.content,
            )
        except Exception:
            changed = False
        if changed:
            params = replace(params, content=repaired)

        if existing_page is not None:
            updates: dict[str, JsonValue] = {
                "title": params.title,
                "summary": params.summary,
                "content": params.content,
                "page_type": params.page_type,
            }
            if params.aliases is not None:
                updates["aliases"] = cast("JsonValue", params.aliases)
            if params.source_refs is not None:
                updates["source_refs"] = cast("JsonValue", resolved_refs)
            page = existing_page.model_copy(update=updates)
            try:
                await self._wiki_service.update_page(
                    page=page, edit_source=WIKI_EDIT_SOURCE_AGENT
                )
            except Exception as exc:
                return ToolResult(success=False, error=f"Failed to update page: {exc}")
            action = "updated"
        else:
            tenant_id = await self._resolve_tenant(kb_id)
            new_page = WikiPage(
                id=str(uuid4()),
                tenant_id=tenant_id,
                knowledge_base_id=kb_id,
                slug=params.slug,
                title=params.title,
                summary=params.summary,
                content=params.content,
                page_type=params.page_type,
                status=WIKI_PAGE_STATUS_PUBLISHED,
                aliases=params.aliases if params.aliases is not None else [],
                source_refs=resolved_refs or [],
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
            try:
                await self._wiki_service.create_page(
                    page=new_page, edit_source=WIKI_EDIT_SOURCE_AGENT
                )
            except Exception as exc:
                return ToolResult(success=False, error=f"Failed to create page: {exc}")
            action = "created"

        self._routes.remember(params.slug, kb_id)
        with suppress(Exception):
            await self._wiki_service.inject_cross_links(
                knowledge_base_id=kb_id, affected_slugs=[params.slug]
            )
        with suppress(Exception):
            await self._wiki_service.rebuild_index_page(knowledge_base_id=kb_id)

        output = (
            f"Successfully {action} page [[{params.slug}]].\n"
            f"- Title: {params.title}\n"
            f"- Type: {params.page_type}\n"
            f"- Summary: {params.summary}\n"
            f"- Content length: {len(params.content)} chars"
        )
        if params.aliases:
            output += f"\n- Aliases: {', '.join(params.aliases)}"
        if params.source_refs is not None:
            output += f"\n- Source refs: {len(resolved_refs or [])} document(s)"

        return ToolResult(
            success=True,
            output=output,
            data={
                "display_type": "wiki_write_page",
                "action": action,
                "slug": params.slug,
                "title": params.title,
                "page_type": params.page_type,
                "summary": params.summary,
            },
        )

    async def _resolve_refs(self, ctx: Context, refs: list[str]) -> list[str] | None:
        """Resolve and authorize source refs, or ``None`` when rejected."""
        if self._search_targets is not None:
            try:
                return await resolve_authorized_source_refs(
                    ctx, self._search_targets, refs, self._knowledge_service
                )
            except ApplicationError:
                return None
        return await resolve_source_refs(ctx, self._knowledge_service, refs)

    async def _resolve_tenant(self, kb_id: str) -> int:
        """Return the owning tenant id of ``kb_id`` when the loader is wired."""
        if self._kb_loader is None:
            return 0
        try:
            kb = await self._kb_loader.load(knowledge_base_id=kb_id)
        except Exception:
            return 0
        return kb.tenant_id if kb is not None else 0


class WikiDeletePageTool:
    """Deletes a wiki page and cleans up its incoming links."""

    def __init__(
        self,
        *,
        definition: ToolDefinition,
        wiki_service: WikiPageServiceProtocol,
        kb_ids: list[str],
        routes: WikiRouteResolver | None = None,
    ) -> None:
        self._definition = definition
        self._wiki_service = wiki_service
        self._kb_ids = dedup_non_empty_strings(kb_ids)
        self._routes = routes if routes is not None else WikiRouteResolver()

    def name(self) -> str:
        return self._definition.name

    def description(self) -> str:
        return self._definition.description

    def parameters(self) -> str:
        return self._definition.parameters

    async def execute(self, ctx: Context, args: str) -> ToolResult:
        """Delete the page and rewrite its incoming links to plain text."""
        raw = _parse_args(args)
        slug = _as_str(raw.get("slug"))
        if not self._kb_ids:
            return ToolResult(success=False, error="No knowledge bases available for editing")
        if not slug:
            return ToolResult(success=False, error="slug is required")
        normalized_slug, slug_error = normalize_and_validate_wiki_slug(slug)
        if slug_error:
            return ToolResult(success=False, error=slug_error)
        slug = normalized_slug

        try:
            existing_page, kb_id = await resolve_unique_wiki_page(
                ctx, self._wiki_service, slug, self._kb_ids, self._routes
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to fetch page to delete: {exc}")
        in_links = list(existing_page.in_links)

        parts = slug.split("/")
        readable_name = parts[-1].replace("-", " ")
        pipe_link = re.compile(r"\[\[" + re.escape(slug) + r"\|([^\]]+)\]\]")

        def rewrite(content: str) -> tuple[str, bool]:
            updated = content.replace(f"[[{slug}]]", readable_name)
            updated = pipe_link.sub(r"\1", updated)
            return updated, updated != content

        changes, updated_slugs, rewrite_error = await apply_incoming_wiki_content_rewrite(
            ctx, self._wiki_service, kb_id, in_links, rewrite
        )
        if rewrite_error:
            rollback_error = await rollback_wiki_content_changes(ctx, self._wiki_service, changes)
            return ToolResult(
                success=False,
                error="Delete aborted while cleaning incoming links: "
                + join_wiki_mutation_errors(rewrite_error, [rollback_error]),
            )

        try:
            await self._wiki_service.delete_page(knowledge_base_id=kb_id, slug=slug)
        except Exception as exc:
            rollback_error = await rollback_wiki_content_changes(ctx, self._wiki_service, changes)
            return ToolResult(
                success=False,
                error="Delete aborted because the page could not be removed: "
                + join_wiki_mutation_errors(str(exc), [rollback_error]),
            )
        self._routes.forget(slug, kb_id)
        updated_count = len(updated_slugs)

        output_msg = f"Successfully deleted page [[{slug}]] and cleaned up {updated_count} incoming links."
        if updated_count > 0:
            output_msg += f"\n- Affected pages: {', '.join(updated_slugs)}"

        return ToolResult(
            success=True,
            output=output_msg,
            data={
                "display_type": "wiki_delete_page",
                "slug": slug,
                "title": existing_page.title,
                "updated_count": updated_count,
                "affected_pages": cast("list[JsonValue]", updated_slugs),
            },
        )


class WikiRenamePageTool:
    """Renames a wiki page's slug and cascades the change to its linkers."""

    def __init__(
        self,
        *,
        definition: ToolDefinition,
        wiki_service: WikiPageServiceProtocol,
        kb_ids: list[str],
        routes: WikiRouteResolver | None = None,
    ) -> None:
        self._definition = definition
        self._wiki_service = wiki_service
        self._kb_ids = dedup_non_empty_strings(kb_ids)
        self._routes = routes if routes is not None else WikiRouteResolver()

    def name(self) -> str:
        return self._definition.name

    def description(self) -> str:
        return self._definition.description

    def parameters(self) -> str:
        return self._definition.parameters

    async def execute(self, ctx: Context, args: str) -> ToolResult:
        """Create the new page, cascade links, then delete the old page."""
        raw = _parse_args(args)
        slug = _as_str(raw.get("slug"))
        new_slug = _as_str(raw.get("new_slug"))
        if not self._kb_ids:
            return ToolResult(success=False, error="No knowledge bases available for editing")
        if not new_slug:
            return ToolResult(success=False, error="new_slug is required")
        slug, slug_error = normalize_and_validate_wiki_slug(slug)
        if slug_error:
            return ToolResult(success=False, error=slug_error)
        new_slug, slug_error = normalize_and_validate_wiki_slug(new_slug)
        if slug_error:
            return ToolResult(success=False, error=slug_error)
        if new_slug == slug:
            return ToolResult(success=False, error="new_slug must be different from old slug")

        try:
            existing_page, kb_id = await resolve_unique_wiki_page(
                ctx, self._wiki_service, slug, self._kb_ids, self._routes
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to resolve page to rename: {exc}")
        in_links = list(existing_page.in_links)

        new_page = existing_page.model_copy(
            update={
                "id": str(uuid4()),
                "slug": new_slug,
                "aliases": list(existing_page.aliases),
                "source_refs": list(existing_page.source_refs),
                "chunk_refs": list(existing_page.chunk_refs),
                "in_links": list(existing_page.in_links),
                "page_metadata": dict(existing_page.page_metadata),
            }
        )
        try:
            await self._wiki_service.create_page(
                page=new_page, edit_source=WIKI_EDIT_SOURCE_AGENT
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to create renamed page: {exc}")

        def rewrite(content: str) -> tuple[str, bool]:
            updated = content.replace(f"[[{slug}]]", f"[[{new_slug}]]")
            updated = updated.replace(f"[[{slug}|", f"[[{new_slug}|")
            return updated, updated != content

        changes, updated_slugs, rewrite_error = await apply_incoming_wiki_content_rewrite(
            ctx, self._wiki_service, kb_id, in_links, rewrite
        )
        if rewrite_error:
            rollback_error = await rollback_wiki_content_changes(ctx, self._wiki_service, changes)
            cleanup_error = await self._delete_cleanup(kb_id, new_slug)
            return ToolResult(
                success=False,
                error="Rename aborted while updating incoming links: "
                + join_wiki_mutation_errors(rewrite_error, [rollback_error, cleanup_error]),
            )
        updated_count = len(updated_slugs)

        try:
            await self._wiki_service.delete_page(knowledge_base_id=kb_id, slug=slug)
        except Exception as exc:
            rollback_error = await rollback_wiki_content_changes(ctx, self._wiki_service, changes)
            cleanup_error = await self._delete_cleanup(kb_id, new_slug)
            return ToolResult(
                success=False,
                error="Rename aborted because the old page could not be deleted: "
                + join_wiki_mutation_errors(str(exc), [rollback_error, cleanup_error]),
            )
        self._routes.forget(slug, kb_id)
        self._routes.remember(new_slug, kb_id)

        with suppress(Exception):
            await self._wiki_service.inject_cross_links(
                knowledge_base_id=kb_id, affected_slugs=[new_slug]
            )
        with suppress(Exception):
            await self._wiki_service.rebuild_index_page(knowledge_base_id=kb_id)

        output_msg = (
            f"Successfully renamed page [[{slug}]] → [[{new_slug}]] and updated {updated_count} incoming links."
        )
        if updated_count > 0:
            output_msg += f"\n- Affected pages: {', '.join(updated_slugs)}"

        return ToolResult(
            success=True,
            output=output_msg,
            data={
                "display_type": "wiki_rename_page",
                "old_slug": slug,
                "new_slug": new_slug,
                "title": existing_page.title,
                "updated_count": updated_count,
                "affected_pages": cast("list[JsonValue]", updated_slugs),
            },
        )

    async def _delete_cleanup(self, kb_id: str, slug: str) -> str:
        """Delete the half-created renamed page, returning an error string."""
        try:
            await self._wiki_service.delete_page(knowledge_base_id=kb_id, slug=slug)
        except Exception as exc:
            return f"cleanup: {exc}"
        return ""


class WikiReplaceTextTool:
    """Replaces specific exact text in a wiki page (minor corrections)."""

    def __init__(
        self,
        *,
        definition: ToolDefinition,
        wiki_service: WikiPageServiceProtocol,
        kb_ids: list[str],
        routes: WikiRouteResolver | None = None,
        knowledge_service: KnowledgeLookup | None = None,
        search_targets: SearchTargets | None = None,
    ) -> None:
        self._definition = definition
        self._wiki_service = wiki_service
        self._kb_ids = dedup_non_empty_strings(kb_ids)
        self._routes = routes if routes is not None else WikiRouteResolver()
        self._knowledge_service = knowledge_service
        self._search_targets = search_targets

    def name(self) -> str:
        return self._definition.name

    def description(self) -> str:
        return self._definition.description

    def parameters(self) -> str:
        return self._definition.parameters

    async def execute(self, ctx: Context, args: str) -> ToolResult:
        """Replace the first occurrence of ``old_text`` in the page body."""
        raw = _parse_args(args)
        slug = _as_str(raw.get("slug"))
        old_text = _as_str(raw.get("old_text"))
        new_text = _as_str(raw.get("new_text"))
        source_refs = _as_str_list(raw.get("source_refs"))

        if not self._kb_ids:
            return ToolResult(success=False, error="No knowledge bases available for editing")
        if not old_text:
            return ToolResult(success=False, error="old_text is required")
        slug, slug_error = normalize_and_validate_wiki_slug(slug)
        if slug_error:
            return ToolResult(success=False, error=slug_error)

        try:
            existing_page, _kb_id = await resolve_unique_wiki_page(
                ctx, self._wiki_service, slug, self._kb_ids, self._routes
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to fetch page {slug}: {exc}")

        if old_text not in existing_page.content:
            return ToolResult(
                success=False,
                error=(
                    "old_text not found in the current page content. "
                    "Ensure you copy it exactly as it appears."
                ),
            )

        content = existing_page.content.replace(old_text, new_text, 1)
        if source_refs is not None:
            if self._search_targets is not None:
                try:
                    resolved_refs = await resolve_authorized_source_refs(
                        ctx, self._search_targets, source_refs, self._knowledge_service
                    )
                except ApplicationError as exc:
                    return ToolResult(success=False, error=f"Invalid source_refs: {exc.message}")
            else:
                resolved_refs = await resolve_source_refs(ctx, self._knowledge_service, source_refs)
            page = existing_page.model_copy(update={"content": content, "source_refs": resolved_refs})
        else:
            page = existing_page.model_copy(update={"content": content})

        try:
            await self._wiki_service.update_page(page=page, edit_source=WIKI_EDIT_SOURCE_AGENT)
        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to update page: {exc}")

        old_preview = _truncate_preview(old_text, 80)
        new_preview = _truncate_preview(new_text, 80)
        output = (
            f"Successfully replaced text on page [[{slug}]].\n"
            f"- Old: {old_preview}\n"
            f"- New: {new_preview}"
        )
        return ToolResult(
            success=True,
            output=output,
            data={
                "display_type": "wiki_replace_text",
                "slug": slug,
                "title": existing_page.title,
                "old_text": old_preview,
                "new_text": new_preview,
            },
        )


__all__ = [
    "WIKI_DELETE_PAGE_TOOL_DESCRIPTION",
    "WIKI_DELETE_PAGE_TOOL_NAME",
    "WIKI_DELETE_PAGE_TOOL_SCHEMA",
    "WIKI_RENAME_PAGE_TOOL_DESCRIPTION",
    "WIKI_RENAME_PAGE_TOOL_NAME",
    "WIKI_RENAME_PAGE_TOOL_SCHEMA",
    "WIKI_REPLACE_TEXT_TOOL_DESCRIPTION",
    "WIKI_REPLACE_TEXT_TOOL_NAME",
    "WIKI_REPLACE_TEXT_TOOL_SCHEMA",
    "WIKI_WRITE_PAGE_TOOL_DESCRIPTION",
    "WIKI_WRITE_PAGE_TOOL_NAME",
    "WIKI_WRITE_PAGE_TOOL_SCHEMA",
    "WikiDeletePageTool",
    "WikiRenamePageTool",
    "WikiReplaceTextTool",
    "WikiWritePageTool",
    "build_wiki_delete_page_definition",
    "build_wiki_rename_page_definition",
    "build_wiki_replace_text_definition",
    "build_wiki_write_page_definition",
]
