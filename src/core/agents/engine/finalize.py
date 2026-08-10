"""Finalize phase: completion synthesis and run summary.

The finalize phase produces the run's terminal output. When the loop exhausts
its iteration ceiling, or when the think phase fails after prior tool results
exist, it synthesizes a final answer from the gathered tool results through a
fresh streaming call, and always emits exactly one completion event carrying
the execution summary for downstream persistence.

The phase composes the think phase for prompt assembly and streaming, and
consumes the merged ``AgentState`` contract. State updates are immutable: each
method returns a new state copy rather than mutating the input.
"""

from __future__ import annotations

import logging
import re
import time

from src.ai.embedding.base import Context
from src.ai.llm.types import ChatOptions, Message, ResponseType, StreamResponse
from src.core.agents.engine.modelcontext import Registry
from src.core.agents.engine.observe import _to_model_output_result
from src.core.agents.engine.think import ThinkPhase, strip_think_blocks
from src.core.agents.engine.types import (
    MAX_ITERATIONS_FALLBACK,
    AgentConfig,
    AgentExecutionError,
    AgentState,
    ToolResult,
    generate_event_id,
)
from src.core.chat.bus import Event, EventBus
from src.core.chat.types import EventType

logger = logging.getLogger(__name__)

#: Markdown image link matcher used by the final-answer image requirement.
_MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def _final_answer_image_requirement(has_retrieved_image: bool) -> str:
    """Return the image-output requirement when retrieved results show images."""
    if not has_retrieved_image:
        return ""
    return (
        "5. Retrieved tool results contain Markdown images. Unless the user "
        "explicitly requested text-only output or every image is clearly "
        "unrelated, the final answer MUST include at least one relevant "
        "Markdown image copied verbatim from the tool results. Preserve its "
        "complete URL exactly. Use ASCII half-width parentheses exactly as "
        "![alt](url) and never use full-width \uff08 or \uff09. Place the image "
        "immediately after the paragraph it supports. When multiple images "
        "support different sections, distribute them across those sections "
        "instead of stopping after the first image.\n"
        "6. Before finishing, silently verify that the answer contains a "
        "Markdown image when requirement 5 applies."
    )


class FinalizePhase:
    """Synthesizes final answers and emits the run completion event."""

    def __init__(
        self,
        config: AgentConfig,
        event_bus: EventBus,
        model_context: Registry,
        think: ThinkPhase,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._model_context = model_context
        self._think = think

    async def synthesize_final_answer(
        self,
        ctx: Context,
        query: str,
        state: AgentState,
        session_id: str,
    ) -> AgentState:
        """Synthesize a final answer from the gathered tool results.

        Returns an updated state carrying ``final_answer``; the caller marks
        the run complete when appropriate.
        """
        system_prompt = self._think.build_system_prompt()
        user_turn = self._think.render_user_turn(session_id, query)
        messages: list[Message] = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_turn),
        ]
        has_retrieved_image = False
        for step in state.round_steps:
            for tool_call in step.tool_calls:
                result = (
                    tool_call.result
                    if tool_call.result is not None
                    else ToolResult(success=False, error="no result")
                )
                if _MARKDOWN_IMAGE_RE.search(result.output):
                    has_retrieved_image = True
                rendered = self._model_context.model_tool_result_for_tool(
                    tool_call.name, _to_model_output_result(result)
                )
                messages.append(
                    Message(role="user", content=f"Tool {tool_call.name} returned: {rendered}")
                )
        image_requirement = _final_answer_image_requirement(has_retrieved_image)
        final_prompt = (
            "Based on the above tool call results, generate a complete answer "
            "for the user's question.\n\n"
            f"User question: {query}\n\n"
            "Requirements:\n"
            "1. Answer based on the actually retrieved content\n"
            "2. Organize the answer in a structured format\n"
            "3. If information is insufficient, honestly state so\n"
            "4. IMPORTANT: Respond in the same language as the user's question\n"
            f"{image_requirement}\n\n"
            "Now generate the final answer:"
        )
        messages.append(Message(role="user", content=final_prompt))

        answer_id = generate_event_id("answer")
        answer_done_emitted = False

        async def emit_final(chunk: StreamResponse, full_content: str) -> None:
            nonlocal answer_done_emitted
            if chunk.response_type is ResponseType.THINKING:
                return
            if chunk.content:
                await self._emit(
                    Event(
                        id=answer_id,
                        type=EventType.AGENT_FINAL_ANSWER,
                        session_id=session_id,
                        data={"content": chunk.content, "done": chunk.done},
                    )
                )
                if chunk.done:
                    answer_done_emitted = True

        llm_result = await self._think.stream_llm_to_events(
            ctx,
            messages,
            ChatOptions(temperature=self._config.temperature),
            session_id,
            emit_func=emit_final,
        )
        if not answer_done_emitted:
            await self._emit(
                Event(
                    id=answer_id,
                    type=EventType.AGENT_FINAL_ANSWER,
                    session_id=session_id,
                    data={"content": "", "done": True},
                )
            )
        return state.model_copy(update={"final_answer": strip_think_blocks(llm_result.content)})

    async def handle_max_iterations(
        self,
        ctx: Context,
        query: str,
        state: AgentState,
        session_id: str,
    ) -> AgentState:
        """Synthesize a final answer when the iteration ceiling was exhausted."""
        try:
            state = await self.synthesize_final_answer(ctx, query, state, session_id)
        except AgentExecutionError:
            state = state.model_copy(update={"final_answer": MAX_ITERATIONS_FALLBACK})
        return state.model_copy(update={"is_complete": True})

    async def emit_completion_event(
        self,
        state: AgentState,
        session_id: str,
        message_id: str,
        start_time: float,
    ) -> None:
        """Emit the exactly-once completion event with the run summary."""
        await self._emit(
            Event(
                id=generate_event_id("complete"),
                type=EventType.AGENT_COMPLETE,
                session_id=session_id,
                data={
                    "session_id": session_id,
                    "total_steps": len(state.round_steps),
                    "final_answer": state.final_answer,
                    "knowledge_refs": [ref.model_dump(mode="json") for ref in state.knowledge_refs],
                    "agent_steps": [step.model_dump(mode="json") for step in state.round_steps],
                    "total_duration_ms": int((time.monotonic() - start_time) * 1000),
                    "message_id": message_id,
                },
            )
        )

    async def _emit(self, event: Event) -> None:
        """Publish one event to the phase's event bus."""
        await self._event_bus.emit(event)


__all__ = ["FinalizePhase"]
