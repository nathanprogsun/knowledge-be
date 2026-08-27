# Prompt-editor labels are Chinese with fullwidth punctuation.

"""Prompt placeholder catalog — static definitions for the agent editor.

Maps the upstream placeholder catalog: every placeholder a prompt template
can reference is declared here with its display label and description, and
the ``prompt_placeholder_group`` helper projects the per-field grouping the
editor UI renders. The catalog is static wire data (no persistence), so it
lives beside the agent domain types rather than behind a service.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from src.common.json import JsonObject

# Prompt-field type keys, mirroring the upstream field enum values.
PROMPT_FIELD_SYSTEM_PROMPT: Final[str] = "system_prompt"
PROMPT_FIELD_AGENT_SYSTEM_PROMPT: Final[str] = "agent_system_prompt"
PROMPT_FIELD_CONTEXT_TEMPLATE: Final[str] = "context_template"
PROMPT_FIELD_REWRITE_SYSTEM_PROMPT: Final[str] = "rewrite_system_prompt"
PROMPT_FIELD_REWRITE_PROMPT: Final[str] = "rewrite_prompt"
PROMPT_FIELD_FALLBACK_PROMPT: Final[str] = "fallback_prompt"

#: Every prompt-field key the editor exposes, in wire order.
PROMPT_FIELDS: Final[tuple[str, ...]] = (
    PROMPT_FIELD_SYSTEM_PROMPT,
    PROMPT_FIELD_AGENT_SYSTEM_PROMPT,
    PROMPT_FIELD_CONTEXT_TEMPLATE,
    PROMPT_FIELD_REWRITE_SYSTEM_PROMPT,
    PROMPT_FIELD_REWRITE_PROMPT,
    PROMPT_FIELD_FALLBACK_PROMPT,
)


@dataclass(frozen=True, slots=True)
class PromptPlaceholder:
    """One insertable placeholder in a prompt template."""

    name: str
    label: str
    description: str

    def to_json(self) -> JsonObject:
        """Project onto the wire shape ``{"name", "label", "description"}``."""
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
        }


_PLACEHOLDER_QUERY = PromptPlaceholder(
    name="query",
    label="用户问题",
    description="用户当前的问题或查询内容",
)
_PLACEHOLDER_CONTEXTS = PromptPlaceholder(
    name="contexts",
    label="检索内容",
    description="从知识库检索到的相关内容列表",
)
_PLACEHOLDER_CURRENT_TIME = PromptPlaceholder(
    name="current_time",
    label="当前时间",
    description="当前系统时间（格式：2006-01-02 15:04:05）",
)
_PLACEHOLDER_CURRENT_WEEK = PromptPlaceholder(
    name="current_week",
    label="当前星期",
    description="当前星期几（如：星期一、Monday）",
)
_PLACEHOLDER_CONVERSATION = PromptPlaceholder(
    name="conversation",
    label="历史对话",
    description="格式化的历史对话内容，用于多轮对话改写",
)
_PLACEHOLDER_YESTERDAY = PromptPlaceholder(
    name="yesterday",
    label="昨天日期",
    description="昨天的日期（格式：2006-01-02）",
)
_PLACEHOLDER_ANSWER = PromptPlaceholder(
    name="answer",
    label="助手回答",
    description="助手的回答内容（用于对话历史格式化）",
)
_PLACEHOLDER_KNOWLEDGE_BASES = PromptPlaceholder(
    name="knowledge_bases",
    label="知识库列表",
    description="自动格式化的知识库列表，包含名称、描述、文档数量等信息",
)
_PLACEHOLDER_WEB_SEARCH_STATUS = PromptPlaceholder(
    name="web_search_status",
    label="网络搜索状态",
    description="网络搜索工具是否启用的状态（Enabled 或 Disabled）",
)
_PLACEHOLDER_LANGUAGE = PromptPlaceholder(
    name="language",
    label="用户语言",
    description=(
        "用户界面的语言偏好，如 Chinese (Simplified)、English、Korean 等，用于控制 LLM 回答语言"
    ),
)

_ALL: Final[tuple[PromptPlaceholder, ...]] = (
    _PLACEHOLDER_QUERY,
    _PLACEHOLDER_CONTEXTS,
    _PLACEHOLDER_CURRENT_TIME,
    _PLACEHOLDER_CURRENT_WEEK,
    _PLACEHOLDER_CONVERSATION,
    _PLACEHOLDER_YESTERDAY,
    _PLACEHOLDER_ANSWER,
    _PLACEHOLDER_KNOWLEDGE_BASES,
    _PLACEHOLDER_WEB_SEARCH_STATUS,
    _PLACEHOLDER_LANGUAGE,
)

#: Per-field availability, mirroring the upstream grouping.
_FIELD_PLACEHOLDERS: Final[dict[str, tuple[PromptPlaceholder, ...]]] = {
    PROMPT_FIELD_SYSTEM_PROMPT: (
        _PLACEHOLDER_QUERY,
        _PLACEHOLDER_CONTEXTS,
        _PLACEHOLDER_CURRENT_TIME,
        _PLACEHOLDER_CURRENT_WEEK,
        _PLACEHOLDER_LANGUAGE,
    ),
    PROMPT_FIELD_AGENT_SYSTEM_PROMPT: (
        _PLACEHOLDER_KNOWLEDGE_BASES,
        _PLACEHOLDER_WEB_SEARCH_STATUS,
        _PLACEHOLDER_CURRENT_TIME,
        _PLACEHOLDER_LANGUAGE,
    ),
    PROMPT_FIELD_CONTEXT_TEMPLATE: (
        _PLACEHOLDER_QUERY,
        _PLACEHOLDER_CONTEXTS,
        _PLACEHOLDER_CURRENT_TIME,
        _PLACEHOLDER_CURRENT_WEEK,
        _PLACEHOLDER_LANGUAGE,
    ),
    PROMPT_FIELD_REWRITE_SYSTEM_PROMPT: (
        _PLACEHOLDER_QUERY,
        _PLACEHOLDER_CONVERSATION,
        _PLACEHOLDER_CURRENT_TIME,
        _PLACEHOLDER_YESTERDAY,
        _PLACEHOLDER_LANGUAGE,
    ),
    PROMPT_FIELD_REWRITE_PROMPT: (
        _PLACEHOLDER_QUERY,
        _PLACEHOLDER_CONVERSATION,
        _PLACEHOLDER_CURRENT_TIME,
        _PLACEHOLDER_YESTERDAY,
        _PLACEHOLDER_LANGUAGE,
    ),
    PROMPT_FIELD_FALLBACK_PROMPT: (
        _PLACEHOLDER_QUERY,
        _PLACEHOLDER_LANGUAGE,
    ),
}


def all_placeholders() -> list[JsonObject]:
    """Return every placeholder in the system, in canonical order."""
    return [placeholder.to_json() for placeholder in _ALL]


def placeholders_by_field(field: str) -> list[JsonObject]:
    """Return the placeholders available for one prompt-field type."""
    return [placeholder.to_json() for placeholder in _FIELD_PLACEHOLDERS.get(field, ())]


def prompt_placeholder_group() -> dict[str, list[JsonObject]]:
    """Return the full field → placeholders map the editor renders."""
    return {field: placeholders_by_field(field) for field in PROMPT_FIELDS}


__all__ = [
    "PROMPT_FIELDS",
    "PROMPT_FIELD_AGENT_SYSTEM_PROMPT",
    "PROMPT_FIELD_CONTEXT_TEMPLATE",
    "PROMPT_FIELD_FALLBACK_PROMPT",
    "PROMPT_FIELD_REWRITE_PROMPT",
    "PROMPT_FIELD_REWRITE_SYSTEM_PROMPT",
    "PROMPT_FIELD_SYSTEM_PROMPT",
    "PromptPlaceholder",
    "all_placeholders",
    "placeholders_by_field",
    "prompt_placeholder_group",
]
