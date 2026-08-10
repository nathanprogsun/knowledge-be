"""Agent engine types: the shared contract for the ReAct core loop.

``types`` owns the value shapes the loop produces and later phases consume:
the persisted run state (``AgentState`` / ``AgentStep`` / ``ToolCall`` /
``ToolResult``), the runtime configuration (``AgentConfig``), the loop's
control-flow sentinels (``IterOutcome``) and analysis verdict
(``ResponseVerdict``), plus the retry / stop-condition constants and the
engine error hierarchy.

Field names and JSON serialization names mirror the upstream contract
field-for-field (``current_round``, ``round_steps``, ``is_complete``,
``final_answer``, ``knowledge_refs``, ``tool_calls``, ``duration``), because
``AgentState`` is persisted onto assistant messages and replayed by the
history loader on later turns.

All persisted models are frozen: the loop builds each value once and never
mutates a model in place. Immutability keeps the contract safe to share
across phases and to serialize without accidental aliasing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.ai.llm.types import SearchResult, ToolCallMetadata
from src.common.exception import ApplicationError
from src.common.json import JsonObject
from src.core.agents.tools.base import ToolResult as ToolsToolResult

# ── Defaults ──────────────────────────────────────────────────────────

#: Default sampling temperature for the agent model calls.
DEFAULT_AGENT_TEMPERATURE = 0.7
#: Default ceiling on ReAct iterations before the loop finalizes.
DEFAULT_AGENT_MAX_ITERATIONS = 20
#: Default single-LLM-call timeout in seconds (the whole run may exceed it).
DEFAULT_LLM_CALL_TIMEOUT = 120.0
#: Maximum retries for transient LLM errors (rate limits, server errors).
MAX_LLM_RETRIES = 2
#: Maximum retries when the model stops naturally with empty content.
MAX_EMPTY_RESPONSE_RETRIES = 2
#: Consecutive identical no-tool responses tolerated before the loop stops.
MAX_REPEATED_RESPONSE_ROUNDS = 2

#: Substrings that mark an error as transient (retryable).
TRANSIENT_ERROR_MARKERS: tuple[str, ...] = (
    "429",
    "rate limit",
    "500",
    "502",
    "503",
    "504",
    "overloaded",
    "timeout",
    "timed out",
    "connection",
    "server error",
    "temporarily unavailable",
)

#: Nudge message appended when the model stops with empty content.
EMPTY_RESPONSE_NUDGE = "Please provide your complete answer now as plain text."
#: Fallback answer once empty-response retries are exhausted.
EMPTY_RESPONSE_FALLBACK = "I'm sorry, I was unable to generate a response. Please try again."
#: Fallback answer when final synthesis fails after the iteration ceiling.
MAX_ITERATIONS_FALLBACK = "Sorry, I was unable to generate a complete answer."
#: Fallback answer when the model's content filter blocks the turn.
CONTENT_FILTER_FALLBACK = (
    "Sorry, this request was blocked by the content safety policy. "
    "Please try rephrasing your question."
)


def is_transient_error(error: BaseException | None) -> bool:
    """Return whether ``error`` looks transient and worth retrying."""
    if error is None:
        return False
    error_text = str(error).lower()
    return any(marker in error_text for marker in TRANSIENT_ERROR_MARKERS)


def is_natural_stop_finish_reason(reason: str) -> bool:
    """Report whether a provider finish reason ends the turn without tools."""
    return reason.strip().lower() in {"stop", "end_turn", "stop_sequence"}


def generate_event_id(suffix: str) -> str:
    """Build a unique event id carrying a human-readable ``suffix``."""
    return f"{uuid4().hex[:8]}-{suffix}"


# ── Loop control ──────────────────────────────────────────────────────


class IterOutcome(StrEnum):
    """Control-flow sentinel returned by one ReAct iteration.

    ``NEXT`` advances the round counter and loops again; ``CONTINUE`` re-runs
    the loop without advancing (used by the empty-content retry path); ``BREAK``
    exits the loop (final answer, stuck loop, or cancelled).
    """

    NEXT = "next"
    CONTINUE = "continue"
    BREAK = "break"


@dataclass(frozen=True, slots=True)
class ResponseVerdict:
    """Outcome of analysing one LLM response for stop conditions."""

    is_done: bool
    final_answer: str = ""
    empty_content: bool = False
    step: AgentStep | None = None


# ── Error hierarchy ───────────────────────────────────────────────────


class AgentExecutionError(ApplicationError):
    """Base error for agent engine loop failures."""

    code = "agent.execution_failed"
    message = "Agent execution failed"


class AgentLLMError(AgentExecutionError):
    """Raised when the LLM call fails irrecoverably."""

    code = "agent.llm_failed"
    message = "Agent LLM call failed"


class AgentCancelledError(AgentExecutionError):
    """Raised when the run is cancelled before finishing naturally.

    ``state`` carries the salvaged partial result (round steps plus any
    synthesized answer) so callers can persist what completed before the stop.
    """

    code = "agent.cancelled"
    message = "Agent run was cancelled"

    def __init__(self, message: str | None = None, *, state: AgentState) -> None:
        self.state = state
        super().__init__(message)


# ── Context metadata ──────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class KnowledgeBaseInfo:
    """Essential knowledge-base facts used by the prompt and runtime context."""

    id: str
    name: str = ""
    type: str = "document"
    description: str = ""
    doc_count: int = 0
    capabilities: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SelectedDocumentInfo:
    """A user-selected document (@ mention) pinned for this turn."""

    knowledge_id: str = ""
    knowledge_base_id: str = ""
    title: str = ""
    file_name: str = ""
    file_type: str = ""


# ── Configuration ─────────────────────────────────────────────────────


class AgentConfig(BaseModel):
    """Runtime agent configuration (persisted blob plus runtime defaults).

    ``thinking`` / ``citation_enabled`` are tri-state on the wire; a ``None``
    citation value means the legacy default (citations on), mirroring the
    upstream semantics.
    """

    model_config = ConfigDict(frozen=True)

    max_iterations: int = DEFAULT_AGENT_MAX_ITERATIONS
    allowed_tools: list[str] = Field(default_factory=list)
    temperature: float = DEFAULT_AGENT_TEMPERATURE
    knowledge_bases: list[str] = Field(default_factory=list)
    knowledge_ids: list[str] = Field(default_factory=list)
    system_prompt: str = ""
    use_custom_system_prompt: bool = False
    web_search_enabled: bool = False
    web_search_max_results: int = 5
    web_search_provider_id: str = ""
    multi_turn_enabled: bool = False
    history_turns: int = 0
    mcp_selection_mode: str = ""
    mcp_services: list[str] = Field(default_factory=list)
    thinking: bool | None = None
    citation_enabled: bool | None = None
    retrieve_kb_only_when_mentioned: bool = False
    retain_retrieval_history: bool = False
    skills_enabled: bool = False
    skill_dirs: list[str] = Field(default_factory=list)
    allowed_skills: list[str] = Field(default_factory=list)
    llm_call_timeout: float = 0.0
    max_tool_output_chars: int = 0
    max_context_tokens: int = 0
    parallel_tool_calls: bool = False

    def citations_enabled(self) -> bool:
        """Preserve citations for legacy configs with a ``None`` setting."""
        return self.citation_enabled is None or self.citation_enabled

    def resolve_system_prompt(self, web_search_enabled: bool) -> str:
        """Return the configured prompt template for the web-search state."""
        if self.system_prompt:
            return self.system_prompt
        return ""

    def llm_call_timeout_seconds(self) -> float:
        """Return the LLM call timeout, falling back to the default."""
        return self.llm_call_timeout if self.llm_call_timeout > 0 else DEFAULT_LLM_CALL_TIMEOUT


# ── Persisted state ───────────────────────────────────────────────────


class ToolResult(BaseModel):
    """Outcome of one tool execution, in the engine's persisted shape.

    ``output`` is the canonical human-readable text for the model;
    ``data`` carries the structured payload; ``images`` holds optional base64
    data URIs produced by image-returning tools.
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    output: str = ""
    data: JsonObject | None = None
    error: str = ""
    images: list[str] = Field(default_factory=list)

    @classmethod
    def from_tools_result(cls, result: ToolsToolResult | None) -> ToolResult:
        """Map a registry tool result onto the engine carrier.

        Accepts the registry's dataclass result and ``None`` (a hard
        execution failure that surfaced no result object).
        """
        if result is None:
            return cls(success=False, error="tool returned no result")
        return cls(
            success=result.success,
            output=result.output,
            data=result.data or None,
            error=result.error,
            images=list(result.images),
        )


class ToolCall(BaseModel):
    """One tool invocation inside an agent step."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    args: JsonObject = Field(default_factory=dict)
    result: ToolResult | None = None
    reflection: str = ""
    duration: int = 0
    provider_metadata: ToolCallMetadata = Field(default_factory=dict)


class AgentStep(BaseModel):
    """One iteration of the ReAct loop (Think + Act phase output)."""

    model_config = ConfigDict(frozen=True)

    iteration: int
    thought: str = ""
    reasoning_content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def observations(self) -> list[str]:
        """Return the collected tool outputs plus any reflection text."""
        observations: list[str] = []
        for tool_call in self.tool_calls:
            if tool_call.result is not None and tool_call.result.output:
                observations.append(tool_call.result.output)
            if tool_call.reflection:
                observations.append(f"Reflection: {tool_call.reflection}")
        return observations


class AgentState(BaseModel):
    """Execution state of an agent across all rounds."""

    model_config = ConfigDict(frozen=True)

    current_round: int = 0
    round_steps: list[AgentStep] = Field(default_factory=list)
    is_complete: bool = False
    final_answer: str = ""
    knowledge_refs: list[SearchResult] = Field(default_factory=list)

    def total_tool_calls(self) -> int:
        """Return the number of tool calls across every step."""
        return sum(len(step.tool_calls) for step in self.round_steps)


__all__ = [
    "CONTENT_FILTER_FALLBACK",
    "DEFAULT_AGENT_MAX_ITERATIONS",
    "DEFAULT_AGENT_TEMPERATURE",
    "DEFAULT_LLM_CALL_TIMEOUT",
    "EMPTY_RESPONSE_FALLBACK",
    "EMPTY_RESPONSE_NUDGE",
    "MAX_EMPTY_RESPONSE_RETRIES",
    "MAX_ITERATIONS_FALLBACK",
    "MAX_LLM_RETRIES",
    "MAX_REPEATED_RESPONSE_ROUNDS",
    "TRANSIENT_ERROR_MARKERS",
    "AgentCancelledError",
    "AgentConfig",
    "AgentExecutionError",
    "AgentLLMError",
    "AgentState",
    "AgentStep",
    "IterOutcome",
    "KnowledgeBaseInfo",
    "ResponseVerdict",
    "SelectedDocumentInfo",
    "ToolCall",
    "ToolResult",
    "generate_event_id",
    "is_natural_stop_finish_reason",
    "is_transient_error",
]
