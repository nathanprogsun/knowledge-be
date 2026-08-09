"""Agent tool base: metadata, result carrier, and shared constants.

``ToolDefinition`` holds the metadata half of a tool (name, description,
JSON parameter schema); ``ToolResult`` is the outcome carrier every tool
returns. Both mirror the upstream contract shapes, and the registry plus
the knowledge-retrieval tools build on these primitives.

The :class:`Tool` protocol is the interface the registry executes against;
concrete tools expose a ``definition`` plus an ``execute`` method. The
relevance / match-type formatters and the tool-name constants live here so
the whole agent layer agrees on one vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from src.ai.embedding.base import Context
from src.ai.retrieval.types import MatchType
from src.common.json import JsonObject

#: Maximum length for a tool/function name imposed by the OpenAI API.
MAX_FUNCTION_NAME_LENGTH = 64

#: Built-in tool names.
TOOL_THINKING = "thinking"
TOOL_TODO_WRITE = "todo_write"
TOOL_GREP_CHUNKS = "grep_chunks"
TOOL_KNOWLEDGE_SEARCH = "knowledge_search"
TOOL_LIST_KNOWLEDGE_CHUNKS = "list_knowledge_chunks"
TOOL_QUERY_KNOWLEDGE_GRAPH = "query_knowledge_graph"
TOOL_GET_DOCUMENT_INFO = "get_document_info"
TOOL_DATABASE_QUERY = "database_query"
TOOL_DATA_ANALYSIS = "data_analysis"
TOOL_DATA_SCHEMA = "data_schema"
TOOL_WEB_SEARCH = "web_search"
TOOL_WEB_FETCH = "web_fetch"
# Skills-related tools (only available when skills are enabled).
TOOL_EXECUTE_SKILL_SCRIPT = "execute_skill_script"
TOOL_READ_SKILL = "read_skill"
# Wiki-related tools (only available when wiki knowledge bases are in scope).
TOOL_WIKI_READ_PAGE = "wiki_read_page"
TOOL_WIKI_WRITE_PAGE = "wiki_write_page"
TOOL_WIKI_REPLACE_TEXT = "wiki_replace_text"
TOOL_WIKI_RENAME_PAGE = "wiki_rename_page"
TOOL_WIKI_DELETE_PAGE = "wiki_delete_page"
TOOL_WIKI_SEARCH = "wiki_search"
TOOL_WIKI_READ_SOURCE_DOC = "wiki_read_source_doc"
TOOL_WIKI_FLAG_ISSUE = "wiki_flag_issue"
TOOL_WIKI_READ_ISSUE = "wiki_read_issue"
TOOL_WIKI_UPDATE_ISSUE = "wiki_update_issue"


@dataclass(frozen=True, slots=True)
class AvailableTool:
    """Tool metadata exposed by the settings APIs."""

    name: str
    label: str
    description: str


def available_tool_definitions() -> list[AvailableTool]:
    """Return the built-in tools exposed to the UI, in registry order.

    Keep this in sync with the tools registered in this package.
    """
    return [
        AvailableTool(TOOL_THINKING, "思考", "动态和反思性的问题解决思考工具"),
        AvailableTool(TOOL_TODO_WRITE, "制定计划", "创建结构化的研究计划"),
        AvailableTool(TOOL_GREP_CHUNKS, "关键词搜索", "快速定位包含特定关键词的文档和分块"),
        AvailableTool(TOOL_KNOWLEDGE_SEARCH, "语义搜索", "理解问题并查找语义相关内容"),
        AvailableTool(TOOL_LIST_KNOWLEDGE_CHUNKS, "查看文档分块", "获取文档完整分块内容"),
        AvailableTool(TOOL_QUERY_KNOWLEDGE_GRAPH, "查询知识图谱", "从知识图谱中查询关系"),
        AvailableTool(TOOL_GET_DOCUMENT_INFO, "获取文档信息", "查看文档元数据"),
        AvailableTool(TOOL_DATABASE_QUERY, "查询数据库", "查询数据库中的信息"),
        AvailableTool(TOOL_DATA_ANALYSIS, "数据分析", "理解数据文件并进行数据分析"),
        AvailableTool(TOOL_DATA_SCHEMA, "查看数据元信息", "获取表格文件的元信息"),
        AvailableTool(TOOL_READ_SKILL, "读取技能", "按需读取技能内容以学习专业能力"),
        AvailableTool(TOOL_EXECUTE_SKILL_SCRIPT, "执行技能脚本", "在沙箱环境中执行技能脚本"),
        AvailableTool(TOOL_WIKI_READ_PAGE, "读取Wiki页面", "读取指定的Wiki页面内容"),
        AvailableTool(TOOL_WIKI_SEARCH, "搜索Wiki", "在Wiki中搜索页面"),
        AvailableTool(TOOL_WIKI_READ_SOURCE_DOC, "精读源文档", "使用知识点深入阅读特定原始文档"),
        AvailableTool(TOOL_WIKI_FLAG_ISSUE, "标记Wiki问题", "标记页面中存在的事实错误或合并冲突问题"),
        AvailableTool(TOOL_WIKI_WRITE_PAGE, "创建/覆盖Wiki", "创建新页面或完全覆盖已有页面"),
        AvailableTool(TOOL_WIKI_REPLACE_TEXT, "局部替换Wiki", "替换Wiki页面中的特定文本"),
        AvailableTool(TOOL_WIKI_RENAME_PAGE, "重命名Wiki", "重命名Wiki页面并自动更新关联链接"),
        AvailableTool(TOOL_WIKI_DELETE_PAGE, "删除Wiki", "删除Wiki页面并自动清理关联死链"),
        AvailableTool(TOOL_WIKI_READ_ISSUE, "查看Wiki问题", "查看特定的Wiki页面问题详情"),
        AvailableTool(TOOL_WIKI_UPDATE_ISSUE, "更新Wiki问题状态", "更新特定的Wiki页面问题状态"),
    ]


def default_allowed_tools() -> list[str]:
    """Return the default allowed-tools list."""
    return [
        TOOL_THINKING,
        TOOL_TODO_WRITE,
        TOOL_KNOWLEDGE_SEARCH,
        TOOL_GREP_CHUNKS,
        TOOL_LIST_KNOWLEDGE_CHUNKS,
        TOOL_QUERY_KNOWLEDGE_GRAPH,
        TOOL_GET_DOCUMENT_INFO,
        TOOL_DATABASE_QUERY,
        TOOL_DATA_ANALYSIS,
        TOOL_DATA_SCHEMA,
    ]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Metadata of one agent tool."""

    name: str
    description: str
    parameters: str = "{}"


def _empty_map() -> JsonObject:
    return {}


def _empty_str_list() -> list[str]:
    return []


@dataclass(frozen=True)
class ToolResult:
    """Outcome of one tool execution."""

    success: bool = False
    output: str = ""
    data: JsonObject = field(default_factory=_empty_map)
    error: str = ""
    images: list[str] = field(default_factory=_empty_str_list)


@dataclass(frozen=True, slots=True)
class FunctionDefinition:
    """LLM function-calling definition for one tool."""

    name: str
    description: str
    parameters: str


@runtime_checkable
class Tool(Protocol):
    """Interface every agent tool implements."""

    def name(self) -> str: ...

    def description(self) -> str: ...

    def parameters(self) -> str: ...

    async def execute(self, ctx: Context, args: str) -> ToolResult: ...


@runtime_checkable
class Cleanable(Protocol):
    """Optional release hook for tools holding session-scoped resources."""

    async def cleanup(self, ctx: Context) -> None: ...


def get_relevance_level(score: float) -> str:
    """Convert a retrieval score to a human-readable relevance level."""
    if score >= 0.8:
        return "High Relevance"
    if score >= 0.6:
        return "Medium Relevance"
    if score >= 0.4:
        return "Low Relevance"
    return "Weak Relevance"


_MATCH_TYPE_LABELS: dict[MatchType, str] = {
    MatchType.EMBEDDING: "Vector Match",
    MatchType.KEYWORDS: "Keyword Match",
    MatchType.NEAR_BY_CHUNK: "Adjacent Chunk Match",
    MatchType.HISTORY: "History Match",
    MatchType.PARENT_CHUNK: "Parent Chunk Match",
    MatchType.RELATION_CHUNK: "Relation Chunk Match",
    MatchType.GRAPH: "Graph Match",
}


def format_match_type(match_type: MatchType) -> str:
    """Convert a match type to a human-readable label."""
    label = _MATCH_TYPE_LABELS.get(match_type)
    if label is not None:
        return label
    return f"Unknown Type({int(match_type)})"


__all__ = [
    "MAX_FUNCTION_NAME_LENGTH",
    "TOOL_DATABASE_QUERY",
    "TOOL_DATA_ANALYSIS",
    "TOOL_DATA_SCHEMA",
    "TOOL_EXECUTE_SKILL_SCRIPT",
    "TOOL_GET_DOCUMENT_INFO",
    "TOOL_GREP_CHUNKS",
    "TOOL_KNOWLEDGE_SEARCH",
    "TOOL_LIST_KNOWLEDGE_CHUNKS",
    "TOOL_QUERY_KNOWLEDGE_GRAPH",
    "TOOL_READ_SKILL",
    "TOOL_THINKING",
    "TOOL_TODO_WRITE",
    "TOOL_WEB_FETCH",
    "TOOL_WEB_SEARCH",
    "TOOL_WIKI_DELETE_PAGE",
    "TOOL_WIKI_FLAG_ISSUE",
    "TOOL_WIKI_READ_ISSUE",
    "TOOL_WIKI_READ_PAGE",
    "TOOL_WIKI_READ_SOURCE_DOC",
    "TOOL_WIKI_RENAME_PAGE",
    "TOOL_WIKI_REPLACE_TEXT",
    "TOOL_WIKI_SEARCH",
    "TOOL_WIKI_UPDATE_ISSUE",
    "TOOL_WIKI_WRITE_PAGE",
    "AvailableTool",
    "Cleanable",
    "FunctionDefinition",
    "Tool",
    "ToolDefinition",
    "ToolResult",
    "available_tool_definitions",
    "default_allowed_tools",
    "format_match_type",
    "get_relevance_level",
]
