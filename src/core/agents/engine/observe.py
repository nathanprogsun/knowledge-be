"""Observe phase: tool-result observation and message replay.

After the model's tool calls have been executed, the observe phase writes the
assistant step (thought + tool calls) and the rendered tool results back onto
the in-turn message list, following the provider tool-calling message shape.
The assistant message is only appended when it carries content, tool calls, or
reasoning, and every tool result is rendered through the model-context
registry so durable ids never leak to the model.

The phase is pure with respect to messages: it returns a new list and never
mutates the input.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from src.ai.llm.types import FunctionCall, Message
from src.ai.llm.types import ToolCall as MessageToolCall
from src.common.json import JsonObject
from src.core.agents.engine.modelcontext import Registry
from src.core.agents.engine.modelcontext.model_output import ToolResult as ModelToolResult
from src.core.agents.engine.types import AgentStep, ToolResult


def _dump_args(args: JsonObject) -> str:
    """Serialize tool arguments back to a JSON string for replay."""
    return json.dumps(args, ensure_ascii=False)


def _to_model_output_result(result: ToolResult) -> ModelToolResult:
    """Map the engine's tool result onto the model-context rendering shape."""
    return ModelToolResult(
        success=result.success,
        output=result.output,
        data=result.data,
        error=result.error,
        images=list(result.images),
    )


class ObservePhase:
    """Appends one agent step and its rendered tool results to the messages."""

    def __init__(self, model_context: Registry) -> None:
        self._model_context = model_context

    def append_tool_results(
        self,
        messages: Sequence[Message],
        step: AgentStep,
    ) -> list[Message]:
        """Return a new message list with the assistant step and tool results."""
        out = list(messages)
        if step.thought or step.tool_calls or step.reasoning_content:
            assistant_msg = Message(
                role="assistant",
                content=step.thought,
                reasoning_content=step.reasoning_content,
                tool_calls=[
                    MessageToolCall(
                        id=tc.id,
                        type="function",
                        function=FunctionCall(name=tc.name, arguments=_dump_args(tc.args)),
                        provider_metadata=tc.provider_metadata,
                    )
                    for tc in step.tool_calls
                ],
            )
            out.append(assistant_msg)
        for tc in step.tool_calls:
            result = (
                tc.result if tc.result is not None else ToolResult(success=False, error="no result")
            )
            rendered = self._model_context.model_tool_result_for_tool(
                tc.name, _to_model_output_result(result)
            )
            out.append(
                Message(role="tool", content=rendered, tool_call_id=tc.id, name=tc.name)
            )
        return out


__all__ = ["ObservePhase"]
