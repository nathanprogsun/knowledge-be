"""Todo-write tool: structured research task-list tracking.

``todo_write`` records the *retrieval* plan for a complex multi-step task —
what to search for, retrieve, or compare — and renders a progress report the
model can update turn-by-turn (pending / in_progress / completed). Synthesis
is deliberately out of scope: summary steps belong to the thinking tool.

The output format mirrors the upstream plan renderer: a task heading, a
numbered step list with status emoji, a progress summary, and a reminder
that all retrieval tasks must finish before conclusions are drawn.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import cast

from src.ai.embedding.base import Context
from src.common.json import JsonObject, JsonValue
from src.core.agents.tools.base import TOOL_TODO_WRITE, ToolResult

logger = logging.getLogger(__name__)

_TODO_WRITE_DESCRIPTION = """Use this tool to create and manage a structured task list for retrieval and research tasks. This helps you track progress, organize complex retrieval operations, and demonstrate thoroughness to the user.

**CRITICAL - Focus on Retrieval Tasks Only**:
- This tool is for tracking RETRIEVAL and RESEARCH tasks (e.g., searching knowledge bases, retrieving documents, gathering information)
- DO NOT include summary or synthesis tasks in todo_write - those are handled by the thinking tool
- Examples of appropriate tasks: "Search for X in knowledge base", "Retrieve information about Y", "Compare A and B"
- Examples of tasks to EXCLUDE: "Summarize findings", "Generate final answer", "Synthesize results" - these are for thinking tool

## When to Use This Tool
Use this tool proactively in these scenarios:

1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
3. User explicitly requests todo list - When the user directly asks you to use the todo list
4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
5. After receiving new instructions - Immediately capture user requirements as todos
6. When you start working on a task - Mark it as in_progress BEFORE beginning work. Ideally you should only have one todo as in_progress at a time
7. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation

## When NOT to Use This Tool

Skip using this tool when:
1. There is only a single, straightforward task
2. The task is trivial and tracking it provides no organizational benefit
3. The task is purely conversational or informational

NOTE that you should not use this tool if there is only one trivial task to do. In this case you are better off just doing the task directly.

## Examples of When to Use the Todo List

<example>
User: Compare this knowledge platform with other RAG frameworks like LangChain and LlamaIndex.
Assistant: I'll help you compare this knowledge platform with other RAG frameworks. Let me create a retrieval plan to gather information about each framework.
*Creates a todo list with retrieval tasks: 1) Search knowledge base for this platform's features and architecture, 2) Use web_search to find LangChain documentation and features, 3) Use web_search to find LlamaIndex documentation and features, 4) Retrieve detailed comparison points for each framework*
*Note: Summary and synthesis will be handled by thinking tool after all retrieval tasks are completed*
Assistant: Let me start by searching the knowledge base for detailed information about this platform.

<reasoning>
The assistant used todo_write tool correctly because:
1. The task requires multiple retrieval operations (searching the knowledge base, web search for LangChain and LlamaIndex)
2. Each retrieval task is specific and actionable
3. The todo list focuses on WHAT to retrieve, not HOW to summarize
4. Summary and synthesis will be handled separately by thinking tool after all retrieval is complete
5. This separation ensures clear task boundaries and avoids duplication
</reasoning>
</example>

<example>
User: I need to research the latest developments in vector databases for RAG applications.
Assistant: I'll help you research vector databases for RAG. Let me create a retrieval plan to gather comprehensive information.
*Creates todo list with retrieval tasks: 1) Search knowledge base for vector database information, 2) Use web_search to find latest vector database technologies, 3) Retrieve information about performance comparisons, 4) Gather information about integration approaches*
*Note: After completing all retrieval tasks, thinking tool will synthesize the findings into a comprehensive answer*
Assistant: Let me start by searching the knowledge base for vector database information.

<reasoning>
The assistant used todo_write tool correctly because:
1. The research task requires multiple retrieval operations (knowledge-base search, web search for latest info)
2. Each task focuses on retrieving specific information
3. The todo list tracks retrieval progress, not synthesis
4. Summary and analysis will be handled by thinking tool after retrieval is complete
5. This approach separates retrieval (todo_write) from synthesis (thinking tool)
</reasoning>
</example>

## Examples of When NOT to Use the Todo List

<example>
User: How do I print 'Hello World' in Python?
Assistant: In Python, you can print "Hello World" with this simple code:

print("Hello World")

This will output the text "Hello World" to the console when executed.</assistant>

<reasoning>
The assistant did not use the todo list because this is a single, trivial task that can be completed in one step. There's no need to track multiple tasks or steps for such a straightforward request.
</reasoning>
</example>

<example>
User: What does the git status command do?
Assistant: The git status command shows the current state of your working directory and staging area. It displays which changes have been staged, which haven't, and which files aren't being tracked by Git. This is a fundamental command for understanding the current state of your repository before making commits.

<reasoning>
The assistant did not use the todo list because this is an informational request with no actual coding task to complete. The user is simply asking for an explanation, not for the assistant to perform multiple steps or tasks.
</reasoning>
</example>

## Task States and Management

1. **Task States**: Use these states to track progress:
  - pending: Task not yet started
  - in_progress: Currently working on (limit to ONE task at a time)
  - completed: Task finished successfully

2. **Task Management**:
  - Update task status in real-time as you work
  - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
  - Only have ONE task in_progress at any time
  - Complete current tasks before starting new ones
  - Remove tasks that are no longer relevant from the list entirely

3. **Task Completion Requirements**:
  - ONLY mark a task as completed when you have FULLY accomplished it
  - If you encounter errors, blockers, or cannot finish, keep the task as in_progress
  - When blocked, create a new task describing what needs to be resolved
  - Never mark a task as completed if:
    - Tests are failing
    - Implementation is partial
    - You encountered unresolved errors
    - You couldn't find necessary files or dependencies

4. **Task Breakdown**:
  - Create specific, actionable RETRIEVAL tasks
  - Break complex retrieval needs into smaller, manageable steps
  - Use clear, descriptive task names focused on what to retrieve or research
  - **DO NOT include summary/synthesis tasks** - those are handled separately by the thinking tool

**Important**: After completing all retrieval tasks in todo_write, use the thinking tool to synthesize findings and generate the final answer. The todo_write tool tracks WHAT to retrieve, while thinking tool handles HOW to synthesize and present the information.

When in doubt, use this tool. Being proactive with task management demonstrates attentiveness and ensures you complete all retrieval requirements successfully."""

_TODO_WRITE_SCHEMA: JsonObject = {
    "type": "object",
    "properties": {
        "task": {
            "type": "string",
            "description": "The complex task or question you need to create a plan for",
        },
        "steps": {
            "type": "array",
            "description": "Array of research plan steps with status tracking",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Unique identifier for this step (e.g., 'step1', 'step2')",
                    },
                    "description": {
                        "type": "string",
                        "description": "Clear description of what to investigate or accomplish in this step",
                    },
                    "status": {
                        "type": "string",
                        "description": (
                            "Current status: pending (not started), in_progress (executing), "
                            "completed (finished)"
                        ),
                    },
                },
                "required": ["id", "description", "status"],
            },
        },
    },
    "required": ["steps"],
}

#: Emoji shown per plan-step status (fallback for unknown statuses: pending).
_STATUS_EMOJI: dict[str, str] = {
    "pending": "⏳",
    "in_progress": "🔄",
    "completed": "✅",
    "skipped": "⏭️",
}

#: Fallback task label when the caller omits ``task``.
_DEFAULT_TASK_LABEL = "No task description provided"


@dataclass(frozen=True, slots=True)
class PlanStep:
    """One step in a research plan."""

    id: str = ""
    description: str = ""
    status: str = "pending"

    def to_json(self) -> JsonObject:
        """Render the step in its JSON wire shape."""
        return {"id": self.id, "description": self.description, "status": self.status}


@dataclass(frozen=True, slots=True)
class TodoWriteInput:
    """Parsed input for the todo_write tool."""

    task: str = ""
    steps: tuple[PlanStep, ...] = field(default_factory=tuple)

    @classmethod
    def from_json(cls, raw: JsonObject) -> TodoWriteInput:
        return cls(
            task=_as_str(raw.get("task")),
            steps=tuple(_parse_steps(raw.get("steps"))),
        )


class TodoWriteTool:
    """Implements a planning tool for complex retrieval tasks."""

    def name(self) -> str:
        return TOOL_TODO_WRITE

    def description(self) -> str:
        return _TODO_WRITE_DESCRIPTION

    def parameters(self) -> str:
        return json.dumps(_TODO_WRITE_SCHEMA, ensure_ascii=False)

    async def execute(self, ctx: Context, args: str) -> ToolResult:
        """Record the plan steps and render a formatted progress report."""
        del ctx
        input_, parse_error = _parse_todo_args(args)
        if parse_error is not None:
            return ToolResult(success=False, error=parse_error)

        task = input_.task if input_.task != "" else _DEFAULT_TASK_LABEL
        plan_steps = list(input_.steps)

        output = generate_plan_output(task, plan_steps)
        steps_json = json.dumps([step.to_json() for step in plan_steps], ensure_ascii=False)

        data: JsonObject = {
            "task": task,
            "steps": cast("list[JsonValue]", [step.to_json() for step in plan_steps]),
            "steps_json": steps_json,
            "total_steps": len(plan_steps),
            "plan_created": True,
            "display_type": "plan",
        }
        return ToolResult(success=True, output=output, data=data)


def generate_plan_output(task: str, steps: list[PlanStep]) -> str:
    """Render a formatted plan output for the model to read back."""
    output = "Plan created\n\n"
    output += f"**Task**: {task}\n\n"

    if not steps:
        output += (
            "Note: No specific steps provided. It is recommended to create 3-7 retrieval "
            "tasks for systematic research.\n\n"
        )
        output += (
            "Suggested retrieval workflow (focused on retrieval tasks, excluding summarization):\n"
        )
        output += "1. Use grep_chunks to search keywords and locate relevant documents\n"
        output += "2. Use knowledge_search for semantic search to retrieve relevant content\n"
        output += "3. Use list_knowledge_chunks to get the full content of key documents\n"
        output += "4. Use web_search to get supplementary information (if needed)\n"
        output += (
            "\nNote: Summarization and synthesis are handled by the thinking tool. "
            "Do not add summarization tasks here.\n"
        )
        return output

    pending_count = sum(1 for step in steps if step.status == "pending")
    in_progress_count = sum(1 for step in steps if step.status == "in_progress")
    completed_count = sum(1 for step in steps if step.status == "completed")
    total_count = len(steps)
    remaining_count = pending_count + in_progress_count

    output += "**Plan Steps**:\n\n"
    for index, step in enumerate(steps, start=1):
        output += format_plan_step(index, step)

    output += "\n=== Task Progress ===\n"
    output += f"Total: {total_count} tasks\n"
    output += f"✅ Completed: {completed_count}\n"
    output += f"🔄 In Progress: {in_progress_count}\n"
    output += f"⏳ Pending: {pending_count}\n"

    output += "\n=== ⚠️ Important Reminder ===\n"
    if remaining_count > 0:
        output += f"**{remaining_count} tasks remaining!**\n\n"
        output += "**All tasks must be completed before summarizing or drawing conclusions.**\n\n"
        output += "Next steps:\n"
        if in_progress_count > 0:
            output += "- Continue completing tasks currently in progress\n"
        if pending_count > 0:
            output += f"- Start processing {pending_count} pending tasks\n"
            output += "- Complete each task in order, do not skip\n"
        output += "- After completing each task, update todo_write to mark it as completed\n"
        output += "- Only generate the final summary after all tasks are completed\n"
    else:
        output += "✅ **All tasks completed!**\n\n"
        output += "You can now:\n"
        output += "- Synthesize findings from all tasks\n"
        output += "- Generate a complete final answer or report\n"
        output += "- Ensure all aspects have been thoroughly researched\n"

    return output


def format_plan_step(index: int, step: PlanStep) -> str:
    """Format one plan step with its status emoji."""
    emoji = _STATUS_EMOJI.get(step.status, "⏳")
    return f"  {index}. {emoji} [{step.status}] {step.description}\n"


def _parse_todo_args(args: str) -> tuple[TodoWriteInput, str | None]:
    """Parse tool args; ``(input, error_message)`` with exactly one set."""
    try:
        parsed = json.loads(args)
    except json.JSONDecodeError as exc:
        return TodoWriteInput(), f"Failed to parse args: {exc}"
    if not isinstance(parsed, dict):
        return TodoWriteInput(), "Failed to parse args: expected a JSON object"
    return TodoWriteInput.from_json(cast(JsonObject, parsed)), None


def _parse_steps(value: JsonValue) -> list[PlanStep]:
    """Parse the ``steps`` array into :class:`PlanStep` rows."""
    if not isinstance(value, list):
        return []
    steps: list[PlanStep] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        steps.append(
            PlanStep(
                id=_as_str(entry.get("id")),
                description=_as_str(entry.get("description")),
                status=_as_str(entry.get("status")) or "pending",
            )
        )
    return steps


def _as_str(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


__all__ = [
    "PlanStep",
    "TodoWriteInput",
    "TodoWriteTool",
    "format_plan_step",
    "generate_plan_output",
]
