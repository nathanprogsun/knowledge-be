"""Wiki issue tools: flag, read, and update review flags on wiki pages.

Issues are lightweight review flags raised against wiki pages for human or
automated maintenance. ``wiki_flag_issue`` pins a pending issue to a page
(the page must resolve uniquely within the allowed scope); ``wiki_read_issue``
returns one issue by id or the pending issues of a slug; ``wiki_update_issue``
moves an issue to ``resolved`` / ``ignored`` / ``pending`` after proving the
issue belongs to an allowed knowledge base.

The tools execute against the merged issue persistence seam (a repository
protocol) plus the wiki page service for unique-page resolution.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

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
    WikiPageServiceProtocol,
    WikiRouteResolver,
    normalize_and_validate_wiki_slug,
    resolve_unique_wiki_page,
    resolve_wiki_issue,
)
from src.core.knowledge.wiki.issues import (
    WIKI_ISSUE_STATUS_PENDING,
    WikiPageIssue,
    WikiPageIssueRepository,
    create_issue,
    list_issues,
    update_issue_status,
)

WIKI_FLAG_ISSUE_TOOL_NAME = "wiki_flag_issue"
WIKI_READ_ISSUE_TOOL_NAME = "wiki_read_issue"
WIKI_UPDATE_ISSUE_TOOL_NAME = "wiki_update_issue"

WIKI_FLAG_ISSUE_TOOL_DESCRIPTION = (
    "Flag a wiki page that contains errors, mixed entities, or outdated information.\n"
    "Use this tool when you or the user identifies that a wiki page is factually incorrect or "
    "wrongly merged (e.g., a page contains information about two different products).\n"
    "This will log an issue for human review or automated maintenance."
)

WIKI_FLAG_ISSUE_TOOL_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": "The slug of the wiki page that has an issue (e.g. 'entity/hunyuan-damoxing')",
            },
            "issue_type": {
                "type": "string",
                "enum": ["mixed_entities", "contradictory_facts", "out_of_date", "other"],
                "description": "The category of the issue",
            },
            "description": {
                "type": "string",
                "description": "A detailed explanation of what is wrong with the page and what should be fixed.",
            },
            "suspected_knowledge_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of short dN document IDs from the <sources> block that you "
                    "suspect are causing the pollution or error."
                ),
            },
        },
        "required": ["slug", "issue_type", "description"],
    },
    ensure_ascii=False,
)

WIKI_READ_ISSUE_TOOL_DESCRIPTION = (
    "Read the details of a specific wiki page issue or list pending issues for a wiki page."
)

WIKI_READ_ISSUE_TOOL_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "issue_id": {
                "type": "string",
                "description": "Optional: The short iN ID of a specific issue from an earlier wiki_read_issue result.",
            },
            "slug": {
                "type": "string",
                "description": "Optional: The slug of the wiki page to list pending issues for.",
            },
        },
        "description": "Provide either issue_id or slug to read issue(s).",
    },
    ensure_ascii=False,
)

WIKI_UPDATE_ISSUE_TOOL_DESCRIPTION = (
    "Update the status of a specific wiki page issue (e.g., set it to 'resolved' or 'ignored')."
)

WIKI_UPDATE_ISSUE_TOOL_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "issue_id": {
                "type": "string",
                "description": "The short iN issue ID from wiki_read_issue.",
            },
            "status": {
                "type": "string",
                "enum": ["resolved", "ignored", "pending"],
                "description": "The new status for the issue.",
            },
        },
        "required": ["issue_id", "status"],
    },
    ensure_ascii=False,
)


def build_wiki_flag_issue_definition() -> ToolDefinition:
    """Return the default tool definition for the flag-issue tool."""
    return ToolDefinition(
        name=WIKI_FLAG_ISSUE_TOOL_NAME,
        description=WIKI_FLAG_ISSUE_TOOL_DESCRIPTION,
        parameters=WIKI_FLAG_ISSUE_TOOL_SCHEMA,
    )


def build_wiki_read_issue_definition() -> ToolDefinition:
    """Return the default tool definition for the read-issue tool."""
    return ToolDefinition(
        name=WIKI_READ_ISSUE_TOOL_NAME,
        description=WIKI_READ_ISSUE_TOOL_DESCRIPTION,
        parameters=WIKI_READ_ISSUE_TOOL_SCHEMA,
    )


def build_wiki_update_issue_definition() -> ToolDefinition:
    """Return the default tool definition for the update-issue tool."""
    return ToolDefinition(
        name=WIKI_UPDATE_ISSUE_TOOL_NAME,
        description=WIKI_UPDATE_ISSUE_TOOL_DESCRIPTION,
        parameters=WIKI_UPDATE_ISSUE_TOOL_SCHEMA,
    )


def _parse_args(args: str) -> JsonObject:
    try:
        raw = json.loads(args)
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def _as_str(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


def _as_str_list(value: JsonValue) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item != ""]
    return []


def _dump_issue(issue: WikiPageIssue) -> str:
    return json.dumps(issue.model_dump(mode="json"), indent=2, ensure_ascii=False)


class WikiFlagIssueTool:
    """Flags a wiki page for human or automated review."""

    def __init__(
        self,
        *,
        definition: ToolDefinition,
        wiki_service: WikiPageServiceProtocol,
        kb_ids: list[str],
        issue_repo: WikiPageIssueRepository,
        routes: WikiRouteResolver | None = None,
        knowledge_service: KnowledgeLookup | None = None,
        search_targets: SearchTargets | None = None,
    ) -> None:
        self._definition = definition
        self._wiki_service = wiki_service
        self._kb_ids = dedup_non_empty_strings(kb_ids)
        self._issue_repo = issue_repo
        self._routes = routes if routes is not None else WikiRouteResolver()
        self._knowledge_service = knowledge_service
        self._search_targets = search_targets
        #: Presence of search targets — not their length — enables the agent
        #: authorization boundary for suspected source documents.

    def name(self) -> str:
        return self._definition.name

    def description(self) -> str:
        return self._definition.description

    def parameters(self) -> str:
        return self._definition.parameters

    async def execute(self, ctx: Context, args: str) -> ToolResult:
        """Pin a pending issue to a uniquely resolved wiki page."""
        raw = _parse_args(args)
        slug = _as_str(raw.get("slug")).strip()
        issue_type = _as_str(raw.get("issue_type"))
        description = _as_str(raw.get("description"))
        suspected = _as_str_list(raw.get("suspected_knowledge_ids"))

        normalized_slug, slug_error = normalize_and_validate_wiki_slug(slug)
        if slug_error:
            return ToolResult(success=False, error=slug_error)
        slug = normalized_slug

        if not self._kb_ids:
            return ToolResult(
                success=False, error="No knowledge bases available for issue tracking"
            )

        try:
            page, kb_id = await resolve_unique_wiki_page(
                ctx, self._wiki_service, slug, self._kb_ids, self._routes
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

        if self._search_targets is not None and suspected:
            try:
                resolved = await resolve_authorized_source_refs(
                    ctx, self._search_targets, suspected, self._knowledge_service
                )
            except ApplicationError as exc:
                return ToolResult(
                    success=False, error=f"Invalid suspected_knowledge_ids: {exc.message}"
                )
            suspected = [ref.split("|", 1)[0] for ref in resolved]

        now = datetime.now(UTC)
        issue = WikiPageIssue(
            id="",
            tenant_id=page.tenant_id,
            knowledge_base_id=kb_id,
            slug=slug,
            issue_type=issue_type,
            description=description,
            suspected_knowledge_ids=suspected,
            reported_by="wiki-researcher-agent",
            status=WIKI_ISSUE_STATUS_PENDING,
            created_at=now,
            updated_at=now,
        )
        try:
            await create_issue(issue_repo=self._issue_repo, issue=issue)
        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to create issue: {exc}")

        return ToolResult(
            success=True,
            output=f"Successfully flagged issue for {slug}. A maintenance ticket has been created for review.",
        )


class WikiReadIssueTool:
    """Reads one issue by id or lists the pending issues of a slug."""

    def __init__(
        self,
        *,
        definition: ToolDefinition,
        issue_repo: WikiPageIssueRepository,
        kb_ids: list[str],
    ) -> None:
        self._definition = definition
        self._issue_repo = issue_repo
        self._kb_ids = dedup_non_empty_strings(kb_ids)

    def name(self) -> str:
        return self._definition.name

    def description(self) -> str:
        return self._definition.description

    def parameters(self) -> str:
        return self._definition.parameters

    async def execute(self, ctx: Context, args: str) -> ToolResult:
        """Read one issue by id, or list the pending issues of a slug."""
        raw = _parse_args(args)
        issue_id = _as_str(raw.get("issue_id")).strip()
        slug = _as_str(raw.get("slug")).strip()

        if not issue_id and not slug:
            return ToolResult(success=False, error="Either issue_id or slug is required")
        if not self._kb_ids:
            return ToolResult(success=False, error="No knowledge bases available")

        if issue_id:
            try:
                issue = await resolve_wiki_issue(ctx, self._issue_repo, issue_id, self._kb_ids)
            except ApplicationError as exc:
                return ToolResult(success=False, error=exc.message)
            return ToolResult(success=True, output=_dump_issue(issue))

        issues: list[WikiPageIssue] = []
        for kb_id in dedup_non_empty_strings(self._kb_ids):
            try:
                kb_issues = await list_issues(
                    issue_repo=self._issue_repo,
                    knowledge_base_id=kb_id,
                    slug=slug,
                    status=WIKI_ISSUE_STATUS_PENDING,
                )
            except ApplicationError as exc:
                return ToolResult(success=False, error=f"Failed to list issues: {exc.message}")
            for issue in kb_issues:
                if (
                    issue is not None
                    and issue.knowledge_base_id
                    and issue.knowledge_base_id != kb_id
                ):
                    return ToolResult(
                        success=False,
                        error=(
                            f"Issue result returned knowledge base {issue.knowledge_base_id} "
                            f"while resolving allowed scope {kb_id}"
                        ),
                    )
            issues.extend(kb_issues)

        if not issues:
            return ToolResult(success=True, output=f"No pending issues found for slug: {slug}")
        return ToolResult(
            success=True,
            output=json.dumps(
                [issue.model_dump(mode="json") for issue in issues],
                indent=2,
                ensure_ascii=False,
            ),
        )


class WikiUpdateIssueTool:
    """Updates the status of a wiki page issue."""

    def __init__(
        self,
        *,
        definition: ToolDefinition,
        issue_repo: WikiPageIssueRepository,
        kb_ids: list[str],
    ) -> None:
        self._definition = definition
        self._issue_repo = issue_repo
        self._kb_ids = dedup_non_empty_strings(kb_ids)

    def name(self) -> str:
        return self._definition.name

    def description(self) -> str:
        return self._definition.description

    def parameters(self) -> str:
        return self._definition.parameters

    async def execute(self, ctx: Context, args: str) -> ToolResult:
        """Set an issue's status after proving it belongs to an allowed KB."""
        raw = _parse_args(args)
        issue_id = _as_str(raw.get("issue_id"))
        status = _as_str(raw.get("status"))

        if not issue_id:
            return ToolResult(success=False, error="issue_id is required")
        if not status:
            return ToolResult(success=False, error="status is required")
        if not self._kb_ids:
            return ToolResult(success=False, error="No knowledge bases available")

        try:
            await resolve_wiki_issue(ctx, self._issue_repo, issue_id, self._kb_ids)
        except ApplicationError as exc:
            return ToolResult(success=False, error=exc.message)

        try:
            await update_issue_status(issue_repo=self._issue_repo, issue_id=issue_id, status=status)
        except ApplicationError as exc:
            return ToolResult(success=False, error=f"Failed to update issue status: {exc.message}")

        return ToolResult(
            success=True,
            output=f"Successfully updated issue {issue_id} to status '{status}'",
        )


__all__ = [
    "WIKI_FLAG_ISSUE_TOOL_DESCRIPTION",
    "WIKI_FLAG_ISSUE_TOOL_NAME",
    "WIKI_FLAG_ISSUE_TOOL_SCHEMA",
    "WIKI_READ_ISSUE_TOOL_DESCRIPTION",
    "WIKI_READ_ISSUE_TOOL_NAME",
    "WIKI_READ_ISSUE_TOOL_SCHEMA",
    "WIKI_UPDATE_ISSUE_TOOL_DESCRIPTION",
    "WIKI_UPDATE_ISSUE_TOOL_NAME",
    "WIKI_UPDATE_ISSUE_TOOL_SCHEMA",
    "WikiFlagIssueTool",
    "WikiReadIssueTool",
    "WikiUpdateIssueTool",
    "build_wiki_flag_issue_definition",
    "build_wiki_read_issue_definition",
    "build_wiki_update_issue_definition",
]
