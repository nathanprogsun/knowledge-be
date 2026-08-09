"""Chat-domain event types.

``EventType`` enumerates the domain event vocabulary; the string values
are stable routing keys carried by every ``Event`` published on the
in-process event bus. They mirror the upstream event vocabulary.
"""

from __future__ import annotations

from enum import StrEnum


class EventType(StrEnum):
    """Routing key for a domain event (mirrors the upstream ``EventType``)."""

    # Query processing
    QUERY_RECEIVED = "query.received"
    QUERY_VALIDATED = "query.validated"
    QUERY_PREPROCESS = "query.preprocess"
    QUERY_REWRITE = "query.rewrite"
    QUERY_REWRITTEN = "query.rewritten"

    # Retrieval
    RETRIEVAL_START = "retrieval.start"
    RETRIEVAL_VECTOR = "retrieval.vector"
    RETRIEVAL_KEYWORD = "retrieval.keyword"
    RETRIEVAL_ENTITY = "retrieval.entity"
    RETRIEVAL_COMPLETE = "retrieval.complete"

    # Rerank
    RERANK_START = "rerank.start"
    RERANK_COMPLETE = "rerank.complete"

    # Merge
    MERGE_START = "merge.start"
    MERGE_COMPLETE = "merge.complete"

    # Chat completion
    CHAT_START = "chat.start"
    CHAT_COMPLETE = "chat.complete"
    CHAT_STREAM = "chat.stream"

    # Agent lifecycle
    AGENT_QUERY = "agent.query"
    AGENT_PLAN = "agent.plan"
    AGENT_STEP = "agent.step"
    AGENT_TOOL = "agent.tool"
    AGENT_COMPLETE = "agent.complete"

    # Agent streaming / real-time feedback
    AGENT_THOUGHT = "thought"
    AGENT_TOOL_CALL = "tool_call"
    AGENT_TOOL_RESULT = "tool_result"
    AGENT_REFLECTION = "reflection"
    AGENT_REFERENCES = "references"
    AGENT_FINAL_ANSWER = "final_answer"

    # MCP tool human approval
    TOOL_APPROVAL_REQUIRED = "tool_approval_required"
    TOOL_APPROVAL_RESOLVED = "tool_approval_resolved"

    # MCP OAuth in-conversation authorization
    MCP_OAUTH_REQUIRED = "mcp_oauth_required"
    MCP_OAUTH_RESOLVED = "mcp_oauth_resolved"

    # Error / session / control
    ERROR = "error"
    SESSION_TITLE = "session_title"
    STOP = "stop"


__all__ = ["EventType"]
