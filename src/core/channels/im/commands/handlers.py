"""Built-in IM slash-command handlers.

Mirrors the upstream command set: ``/help``, ``/info``, ``/search``,
``/stop``, and ``/clear``. Service dependencies (knowledge-base
listing for ``/info``, retrieval search for ``/search``) are injected
at construction time so the commands stay free of service wiring; a
missing dependency degrades the affected section to the same fallback
text the upstream renders when the lookup fails.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from src.common.json import JsonObject, JsonValue
from src.core.channels.im.commands.registry import (
    Command,
    CommandAction,
    CommandContext,
    CommandRegistry,
    CommandResult,
    CustomAgentLike,
)
from src.core.chat.pipeline.types import SearchResult
from src.core.chat.service import TagScope
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo

logger = logging.getLogger("src.core.channels.im.commands")

#: Cap on the number of search hits rendered in an IM reply.
_SEARCH_MAX_RESULTS = 5
#: Max characters (code points) shown per search hit.
_SEARCH_CONTENT_MAX_LEN = 200

#: Agent-config keys ``/info`` reads (dotted accessor paths into the
#: bound agent's ``config``). The agent-resolution seam wires a concrete
#: object that satisfies these keys.
_CONFIG_PATH_KEYS: frozenset[str] = frozenset(
    {
        "agent_mode",
        "kb_selection_mode",
        "knowledge_base_ids",
        "skills_selection_mode",
        "selected_skills",
        "mcp_selection_mode",
        "mcp_service_ids",
        "web_search_enabled",
    }
)


# ── Service seams (structural protocols) ───────────────────────────────


@runtime_checkable
class KnowledgeBaseLister(Protocol):
    """KB-listing surface ``/info`` and ``/search`` need.

    Structurally satisfied by the knowledge-base service once the
    message pipeline wires it.
    """

    async def list_knowledge_bases(self, *, tenant_id: int) -> list[KnowledgeBaseInfo]: ...


@runtime_checkable
class SearchService(Protocol):
    """Retrieval-search surface ``/search`` needs.

    Structurally satisfied by the message-pipeline search service once
    wired; without it the command returns a not-yet-wired reply.
    """

    async def search(
        self,
        *,
        query: str,
        knowledge_base_ids: list[str] | None = None,
        knowledge_ids: list[str] | None = None,
        tag_scopes: list[TagScope] | None = None,
    ) -> list[SearchResult]: ...


# ── /help ──────────────────────────────────────────────────────────────


class HelpCommand(Command):
    """Implements ``/help [command]``."""

    def __init__(self, registry: CommandRegistry) -> None:
        self._registry = registry

    def name(self) -> str:
        return "help"

    def description(self) -> str:
        return "显示可用指令列表，或查看某个指令的详细用法"

    async def execute(self, ctx: CommandContext, args: list[str]) -> CommandResult:
        if args:
            name = args[0].strip().lower()
            command, _args, ok = self._registry.parse("/" + name)
            if not ok:
                return CommandResult(
                    content=f"未知指令 `{args[0]}`，发送 `/help` 查看所有可用指令。"
                )
            return CommandResult(
                content=f"**/{command.name()}** — {command.description()}"
            )

        commands = sorted(self._registry.all(), key=lambda c: c.name())
        lines = ["**可用指令**", ""]
        lines.extend(f"· `/{c.name()}` — {c.description()}" for c in commands)
        lines.append("")
        lines.append("发送 `/help <指令名>` 查看详细用法")
        return CommandResult(content="\n".join(lines))


# ── /clear ─────────────────────────────────────────────────────────────


class ClearCommand(Command):
    """Implements ``/clear`` — reset the current conversation.

    Requests ``ActionClear`` so the executing layer soft-deletes the
    current session and the next message starts completely fresh.
    """

    def name(self) -> str:
        return "clear"

    def description(self) -> str:
        return "清空对话记忆，下次消息将开始全新会话"

    async def execute(self, ctx: CommandContext, args: list[str]) -> CommandResult:
        return CommandResult(
            content="✅ 对话已清空，下次消息将开始全新会话。",
            action=CommandAction.CLEAR,
        )


# ── /stop ──────────────────────────────────────────────────────────────


class StopCommand(Command):
    """Implements ``/stop`` — abort the in-flight reply.

    Requests ``ActionStop`` so the executing layer cancels the running
    QA request for this user+chat; if none is in progress the command
    simply acknowledges.
    """

    def name(self) -> str:
        return "stop"

    def description(self) -> str:
        return "中止当前正在进行的回答"

    async def execute(self, ctx: CommandContext, args: list[str]) -> CommandResult:
        return CommandResult(
            content="✅ 已请求中止当前回答。",
            action=CommandAction.STOP,
        )


# ── /info ──────────────────────────────────────────────────────────────


class InfoCommand(Command):
    """Implements ``/info`` — show the bound agent's profile and capabilities."""

    def __init__(self, kb_service: KnowledgeBaseLister | None = None) -> None:
        self._kb_service = kb_service

    def name(self) -> str:
        return "info"

    def description(self) -> str:
        return "查看当前智能体的信息与能力"

    async def execute(self, ctx: CommandContext, args: list[str]) -> CommandResult:
        lines: list[str] = []
        name = ctx.agent_name or "未命名智能体"
        lines.append(f"🤖 **{name}**")
        description = _agent_description(ctx)
        if description:
            lines.append(f"> {description}")

        if ctx.custom_agent is None:
            lines.append("")
            lines.append("未绑定智能体，发送 `/help` 查看可用指令。")
            return CommandResult(content="\n".join(lines))

        lines.extend(await self._render_capabilities(ctx))
        lines.append("")
        lines.append("---")
        lines.append("发送 `/help` 查看所有可用指令")
        return CommandResult(content="\n".join(lines))

    async def _render_capabilities(self, ctx: CommandContext) -> list[str]:
        """Render the mode / knowledge-base / skills / MCP / search sections."""
        lines: list[str] = []
        config = _config_map(ctx.custom_agent)

        lines.append("")
        lines.append("🧠 **Agent模式**")
        agent_mode = _config_bool(config, "agent_mode", _agent_default_mode(ctx))
        if agent_mode:
            lines.append("支持多步思考、工具调用（ReAct）")
        else:
            lines.append("基于知识库检索直接回答（RAG）")

        lines.append("")
        lines.append("📚 **知识库**")
        lines.extend(await self._render_knowledge_bases(ctx, config))

        lines.append("")
        lines.append("⚡ **Skills**")
        skills_mode = _config_str(config, "skills_selection_mode")
        skills = _config_str_list(config, "selected_skills")
        if skills_mode == "all":
            lines.append("  全部启用")
        elif skills_mode == "selected" and skills:
            lines.extend(f"  · {skill}" for skill in skills)
        else:
            lines.append("  未配置")

        lines.append("")
        lines.append("🔌 **MCP 服务**")
        mcp_mode = _config_str(config, "mcp_selection_mode")
        mcp_ids = _config_str_list(config, "mcp_service_ids")
        if mcp_mode == "all":
            lines.append("  全部接入")
        elif mcp_mode == "selected" and mcp_ids:
            lines.append(f"  已接入 {len(mcp_ids)} 个服务")
        else:
            lines.append("  未配置")

        lines.append("")
        lines.append("🌐 **网络搜索**")
        if _config_bool(config, "web_search_enabled", False):
            lines.append("  已启用")
        else:
            lines.append("  未启用")

        output_label = "流式输出" if ctx.channel_output_mode != "full" else "完整输出"
        lines.append("")
        lines.append("⚙️ **输出模式**")
        lines.append(f"  {output_label}")
        return lines

    async def _render_knowledge_bases(
        self, ctx: CommandContext, config: JsonObject
    ) -> list[str]:
        """Render the knowledge-base section, mirroring the upstream logic.

        ``KBSelectionMode == "all"`` lists every KB under the tenant;
        ``"selected"`` lists the configured KB ids (labelled by name
        when the KB service is available); anything else renders
        "未配置". A failed or unwired KB lookup falls back to the same
        static text the upstream renders in that case.
        """
        lines: list[str] = []
        kb_mode = _config_str(config, "kb_selection_mode")
        if kb_mode == "all":
            names = await self._list_kb_names(ctx.tenant_id)
            if names:
                lines.extend(f"  · {name}" for name in names)
                lines.append(f"  共 {len(names)} 个（全部启用）")
            else:
                lines.append("  全部启用")
            return lines

        kb_ids = _config_str_list(config, "knowledge_base_ids")
        if not kb_ids:
            lines.append("  未配置")
            return lines
        names = await self._list_kb_names(ctx.tenant_id)
        if names:
            name_map = {name_id: name for name_id, name in names}
            for kb_id in kb_ids:
                lines.append(f"  · {name_map.get(kb_id, kb_id)}")
        else:
            lines.append(f"  已选择 {len(kb_ids)} 个")
        return lines

    async def _list_kb_names(self, tenant_id: int) -> list[tuple[str, str]]:
        """Return ``[(id, name)]`` for the tenant's KBs (empty when unwired)."""
        if self._kb_service is None:
            return []
        try:
            infos = await self._kb_service.list_knowledge_bases(tenant_id=tenant_id)
        except Exception:
            logger.warning("[IM] /info KB listing failed; falling back", exc_info=True)
            return []
        return [(info.id, info.name) for info in infos]


# ── /search ────────────────────────────────────────────────────────────


class SearchCommand(Command):
    """Implements ``/search <query>`` — raw knowledge-base retrieval.

    Runs retrieval against the user's selected knowledge bases and
    returns the matching passages without AI summarisation. Until the
    search service is wired, the command returns a not-yet-wired reply.
    """

    def __init__(
        self,
        search_service: SearchService | None = None,
        kb_service: KnowledgeBaseLister | None = None,
    ) -> None:
        self._search_service = search_service
        self._kb_service = kb_service

    def name(self) -> str:
        return "search"

    def description(self) -> str:
        return "直接检索知识库原文（不经 AI 总结），例如：/search 退款政策"

    async def execute(self, ctx: CommandContext, args: list[str]) -> CommandResult:
        if not args:
            return CommandResult(content="请输入搜索内容，例如：`/search 退款政策`")
        query = " ".join(args).strip()
        if not query:
            return CommandResult(content="请输入搜索内容，例如：`/search 退款政策`")

        if self._search_service is None:
            return CommandResult(content="检索服务尚未接入，请稍后再试。")

        kb_ids = await self._resolve_kb_ids(ctx)
        results = await self._search_service.search(
            query=query,
            knowledge_base_ids=kb_ids,
        )
        return _format_search_results(query, results)

    async def _resolve_kb_ids(self, ctx: CommandContext) -> list[str]:
        """Resolve which KBs to search, mirroring the QA-pipeline scope.

        ``"all"`` uses every KB under the tenant; ``"none"`` disables
        retrieval; ``"selected"`` (and the backward-compatible default)
        uses the configured KB id list. The capability filter the QA
        pipeline applies to the ``"all"`` branch is a deferred seam.
        """
        if ctx.custom_agent is None:
            return []
        config = _config_map(ctx.custom_agent)
        mode = _config_str(config, "kb_selection_mode")
        if mode == "all":
            return await self._list_all_kb_ids(ctx.tenant_id)
        if mode == "none":
            return []
        return _config_str_list(config, "knowledge_base_ids")

    async def _list_all_kb_ids(self, tenant_id: int) -> list[str]:
        """Return every KB id under the tenant (empty when unwired)."""
        if self._kb_service is None:
            return []
        try:
            infos = await self._kb_service.list_knowledge_bases(tenant_id=tenant_id)
        except Exception:
            logger.warning("[IM] /search KB listing failed; searching nothing", exc_info=True)
            return []
        return [info.id for info in infos]


# ── Registry builder ───────────────────────────────────────────────────


def build_default_registry(
    *,
    search_service: SearchService | None = None,
    kb_service: KnowledgeBaseLister | None = None,
) -> CommandRegistry:
    """Build the registry with every built-in command registered.

    Service dependencies are injected when the caller has them; the
    commands degrade gracefully (``/info`` renders fallback sections,
    ``/search`` returns a not-yet-wired reply) until the message
    pipeline wires them.
    """
    registry = CommandRegistry()
    registry.register(HelpCommand(registry))
    registry.register(InfoCommand(kb_service=kb_service))
    registry.register(
        SearchCommand(search_service=search_service, kb_service=kb_service)
    )
    registry.register(StopCommand())
    registry.register(ClearCommand())
    return registry


# ── Result formatting ──────────────────────────────────────────────────


def _format_search_results(query: str, results: list[SearchResult]) -> CommandResult:
    """Render search hits as a compact Markdown list (capped like upstream)."""
    lines = [f"🔍 **搜索「{query}」** — 找到 {len(results)} 条结果", ""]
    shown = results[:_SEARCH_MAX_RESULTS]
    for index, result in enumerate(shown, start=1):
        content = result.content
        suffix = ""
        if len(content) > _SEARCH_CONTENT_MAX_LEN:
            content = content[:_SEARCH_CONTENT_MAX_LEN]
            suffix = "…"
        source = result.knowledge_title or result.knowledge_id
        lines.append(f"**[{index}]** {source}")
        lines.append(f"> {content}{suffix}")
        if result.score > 0:
            lines.append(f"匹配度：{result.score * 100:.0f}%")
        lines.append("")
    if len(results) > _SEARCH_MAX_RESULTS:
        lines.append(f"_（仅显示前 {_SEARCH_MAX_RESULTS} 条，共 {len(results)} 条）_")
    return CommandResult(content="\n".join(lines))


# ── Defensive agent-config access ──────────────────────────────────────


def _agent_description(ctx: CommandContext) -> str:
    """Return the bound agent's description, or ``""``."""
    if ctx.custom_agent is None:
        return ""
    return _as_str(getattr(ctx.custom_agent, "description", ""), "")


def _agent_default_mode(ctx: CommandContext) -> bool:
    """Return the agent's declared mode default (``is_agent_mode()``)."""
    if ctx.custom_agent is None:
        return False
    method = getattr(ctx.custom_agent, "is_agent_mode", None)
    if callable(method):
        return bool(method())
    return False


def _config_map(custom_agent: CustomAgentLike | None) -> JsonObject:
    """Return the bound agent's config as a plain dict with safe defaults."""
    if custom_agent is None:
        return {}
    config = getattr(custom_agent, "config", None)
    if isinstance(config, dict):
        return {k: v for k, v in config.items() if k in _CONFIG_PATH_KEYS}
    if config is None:
        return {}
    return {k: getattr(config, k) for k in _CONFIG_PATH_KEYS if hasattr(config, k)}


def _config_str(config: JsonObject, key: str) -> str:
    value = config.get(key)
    return _as_str(value, "")


def _config_bool(config: JsonObject, key: str, default: bool) -> bool:
    value = config.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    if isinstance(value, (int, float)):
        return value != 0
    return default


def _config_str_list(config: JsonObject, key: str) -> list[str]:
    value = config.get(key)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    if isinstance(value, tuple):
        return [item for item in value if isinstance(item, str)]
    return []


def _as_str(value: JsonValue | None, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return default


__all__ = [
    "ClearCommand",
    "HelpCommand",
    "InfoCommand",
    "KnowledgeBaseLister",
    "SearchCommand",
    "SearchService",
    "StopCommand",
    "build_default_registry",
]
