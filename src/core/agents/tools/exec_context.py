"""Per-call tool execution context.

Carries session / request identity for a single tool invocation plus the
principal user id used by human-in-the-loop approval gates. The value is
threaded through ``contextvars`` so it is visible to the tool and any
nested async calls without explicit plumbing.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

#: Fallback per-tool execution timeout (seconds) when none is set.
DEFAULT_TOOL_EXEC_TIMEOUT = 60.0


@dataclass(frozen=True, slots=True)
class ToolExecContext:
    """Identity attached to one tool execution."""

    session_id: str = ""
    assistant_message_id: str = ""
    request_id: str = ""
    tool_call_id: str = ""
    user_id: str = ""
    exec_timeout: float = 0.0

    def effective_timeout(self) -> float:
        """Return the execution timeout, falling back to the default."""
        return self.exec_timeout if self.exec_timeout > 0 else DEFAULT_TOOL_EXEC_TIMEOUT


_tool_exec_ctx_var: ContextVar[ToolExecContext | None] = ContextVar(
    "agent_tool_exec_ctx", default=None
)


@contextmanager
def with_tool_exec_context(meta: ToolExecContext | None) -> Iterator[None]:
    """Run the enclosing block with ``meta`` visible to tool execution."""
    token = _tool_exec_ctx_var.set(meta)
    try:
        yield
    finally:
        _tool_exec_ctx_var.reset(token)


def tool_exec_from_context() -> ToolExecContext | None:
    """Return the execution context attached by the engine, if any."""
    return _tool_exec_ctx_var.get()


__all__ = [
    "DEFAULT_TOOL_EXEC_TIMEOUT",
    "ToolExecContext",
    "tool_exec_from_context",
    "with_tool_exec_context",
]
