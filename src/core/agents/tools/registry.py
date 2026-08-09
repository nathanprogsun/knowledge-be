"""Tool registry: registration, lookup, and guarded execution.

The registry is the single place tools are registered and executed. It
enforces a first-wins registration policy so a name collision cannot
hijack tool execution, applies schema-driven parameter casting and
validation before any tool runs, publishes the output ceiling so
budget-aware tools can shape results, and truncates oversized output as a
fallback.

Every execution returns a :class:`ToolResult`; the LLM-facing hint is
appended to soft failures so the model can retry with a different
approach. An unknown tool name raises ``NotFoundError`` (a hard failure
the caller must handle).
"""

from __future__ import annotations

import logging
from dataclasses import replace

from src.ai.embedding.base import Context
from src.common.exception import NotFoundError
from src.core.agents.tools.base import Cleanable, FunctionDefinition, Tool, ToolResult
from src.core.agents.tools.output_budget import (
    DEFAULT_MAX_TOOL_OUTPUT,
    truncate_tool_output,
    with_output_budget,
)
from src.core.agents.tools.param_utils import (
    cast_params,
    format_validation_errors,
    validate_params,
)

logger = logging.getLogger(__name__)

#: Appended to tool error messages to guide the LLM to try another approach.
TOOL_ERROR_HINT = "\n\n[Analyze the error above and try a different approach.]"

#: Error code for an unknown tool name.
TOOL_NOT_FOUND_CODE = "tool.not_found"


class ToolRegistry:
    """Manages the registration and retrieval of tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        # Maximum tool output size; 0 means the default.
        self._max_tool_output_size: int = 0

    def set_max_tool_output_size(self, max_chars: int) -> None:
        """Set the maximum character length for tool output.

        Values ``<= 0`` fall back to the default.
        """
        self._max_tool_output_size = max_chars

    def get_max_tool_output(self) -> int:
        """Return the effective maximum tool output size."""
        if self._max_tool_output_size > 0:
            return self._max_tool_output_size
        return DEFAULT_MAX_TOOL_OUTPUT

    def register_tool(self, tool: Tool) -> None:
        """Register a tool; an existing name is kept (first-wins policy).

        The first-wins policy prevents tool execution hijacking via name
        collision.
        """
        name = tool.name()
        if name in self._tools:
            logger.warning(
                "[ToolRegistry] Duplicate tool registration rejected: %s (first-wins policy)",
                name,
            )
            return
        self._tools[name] = tool

    def get_tool(self, name: str) -> Tool:
        """Return the tool registered under ``name``.

        Raises ``NotFoundError`` when no tool is registered under that name.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise NotFoundError(
                code=TOOL_NOT_FOUND_CODE,
                message=f"tool not found: {name}",
            )
        return tool

    def list_tools(self) -> list[str]:
        """Return all registered tool names sorted alphabetically."""
        return sorted(self._tools)

    def get_function_definitions(self) -> list[FunctionDefinition]:
        """Return LLM function definitions, sorted by tool name.

        The stable order keeps the serialized payload sent to the model
        byte-identical across requests, which is required for providers
        that key prompt caching on a byte-level prefix match.
        """
        return [
            FunctionDefinition(
                name=self._tools[name].name(),
                description=self._tools[name].description(),
                parameters=self._tools[name].parameters(),
            )
            for name in sorted(self._tools)
        ]

    async def execute_tool(self, ctx: Context, name: str, args: str) -> ToolResult:
        """Execute a tool by name with the given JSON arguments.

        Arguments are cast and validated against the tool's schema before
        execution; a validation failure returns a failed result with the
        LLM hint instead of wasting an execution. Output that exceeds the
        ceiling is truncated as a fallback.
        """
        tool = self.get_tool(name)

        schema = tool.parameters()
        casted_args = cast_params(args, schema)
        validation_errors = validate_params(casted_args, schema)
        if validation_errors:
            error_message = format_validation_errors(validation_errors) + TOOL_ERROR_HINT
            return ToolResult(success=False, error=error_message)

        max_output = self.get_max_tool_output()
        with with_output_budget(max_output):
            result = await tool.execute(ctx, casted_args)

        if result is None:
            return ToolResult(success=False, error=f"tool returned no result{TOOL_ERROR_HINT}")

        # Truncate oversized output, counted in code points so multi-byte
        # text is treated fairly.
        if len(result.output) > max_output:
            result = replace(
                result,
                output=truncate_tool_output(result.output, max_output),
            )
        if not result.success and result.error:
            result = replace(result, error=result.error + TOOL_ERROR_HINT)
        return result

    async def cleanup(self, ctx: Context) -> None:
        """Release resources held by tools that opt into cleanup."""
        for tool in self._tools.values():
            if isinstance(tool, Cleanable):
                await tool.cleanup(ctx)


__all__ = ["TOOL_ERROR_HINT", "TOOL_NOT_FOUND_CODE", "ToolRegistry"]
