"""Skill tools: on-demand skill reading and sandboxed script execution.

``read_skill`` loads a skill's full instructions (``SKILL.md``) or an
additional file from its directory on demand (progressive disclosure).
``execute_skill_script`` runs a utility script bundled with a skill in a
sandboxed environment and reports stdout / stderr / exit code back to the
model.

Both tools depend on a :class:`SkillManager` protocol that mirrors the
upstream skills-manager surface — ``is_enabled``, ``load_skill``,
``read_skill_file``, ``list_skill_files`` and ``execute_script``. The
manager is injected at construction; ``None`` or a disabled manager makes
the tools fail fast with "Skills are not enabled".
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

from src.ai.embedding.base import Context
from src.common.json import JsonObject, JsonValue
from src.core.agents.engine.sandbox.types import ExecuteResult
from src.core.agents.tools.base import (
    TOOL_EXECUTE_SKILL_SCRIPT,
    TOOL_READ_SKILL,
    ToolResult,
)

logger = logging.getLogger(__name__)

#: Canonical skill instruction file name.
SKILL_FILE_NAME = "SKILL.md"

#: File extensions treated as executable scripts.
_SCRIPT_EXTENSIONS: frozenset[str] = frozenset(
    {".py", ".sh", ".bash", ".js", ".ts", ".rb", ".pl", ".php"}
)


@dataclass(frozen=True, slots=True)
class Skill:
    """A loaded skill: metadata plus its on-demand instructions."""

    name: str
    description: str = ""
    base_path: str = ""
    file_path: str = ""
    instructions: str = ""
    loaded: bool = False


def is_script(path: str) -> bool:
    """Return whether ``path``'s extension marks it as an executable script."""
    lowered = path.strip().lower()
    dot = lowered.rfind(".")
    if dot < 0:
        return False
    return lowered[dot:] in _SCRIPT_EXTENSIONS


@runtime_checkable
class SkillManager(Protocol):
    """Skills-manager seam consumed by the skill tools.

    Implemented by the live skills manager (which coordinates filesystem
    loading and the sandbox backend). Tests inject a stub.
    """

    def is_enabled(self) -> bool: ...

    async def load_skill(self, ctx: Context, skill_name: str) -> Skill: ...

    async def read_skill_file(
        self,
        ctx: Context,
        skill_name: str,
        file_path: str,
    ) -> str: ...

    async def list_skill_files(self, ctx: Context, skill_name: str) -> list[str]: ...

    async def execute_script(
        self,
        ctx: Context,
        skill_name: str,
        script_path: str,
        args: list[str],
        stdin: str,
    ) -> ExecuteResult: ...


_READ_SKILL_DESCRIPTION = """Read skill content on demand to learn specialized capabilities.

## Usage
- Use this tool when a user request matches an available skill's description
- Provide the skill_name to load the skill's full instructions (SKILL.md content)
- Optionally provide file_path to read additional files within the skill directory

## When to Use
- When the system prompt shows an available skill that matches the user's request
- Before performing tasks that match a skill's description
- To read additional documentation or reference files within a skill

## Returns
- Skill instructions and guidance for completing the task
- File content if file_path is specified"""

_READ_SKILL_SCHEMA: JsonObject = {
    "type": "object",
    "properties": {
        "skill_name": {"type": "string", "description": "Name of the skill to read"},
        "file_path": {
            "type": "string",
            "description": "Optional relative path to a specific file within the skill directory",
        },
    },
    "required": ["skill_name"],
}

_EXECUTE_SKILL_SCRIPT_DESCRIPTION = """Execute a script from a skill in a sandboxed environment.

## Usage
- Use this tool to run utility scripts bundled with a skill
- Scripts are executed in an isolated sandbox for security
- Only scripts from loaded skills can be executed

## When to Use
- When a skill's instructions reference a utility script (e.g., "Run scripts/analyze_form.py")
- When automation or data processing is needed as part of skill workflow
- For deterministic operations where script execution is more reliable than generating code

## Security
- Scripts run in a sandboxed environment with limited permissions
- Network access is disabled by default
- File access is restricted to the skill directory

## Returns
- Script stdout and stderr output
- Exit code indicating success (0) or failure (non-zero)"""

_EXECUTE_SKILL_SCRIPT_SCHEMA: JsonObject = {
    "type": "object",
    "properties": {
        "skill_name": {
            "type": "string",
            "description": "Name of the skill containing the script",
        },
        "script_path": {
            "type": "string",
            "description": (
                "Relative path to the script within the skill directory (e.g. scripts/analyze.py)"
            ),
        },
        "args": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Optional command-line arguments to pass to the script. Note: if using "
                "--file flag, you must provide an actual file path that exists in the "
                "skill directory. If you have data in memory (not a file), use the "
                "'input' parameter instead."
            ),
        },
        "input": {
            "type": "string",
            "description": (
                "Optional input data to pass to the script via stdin. Use this when you "
                "have data in memory (e.g. JSON string) that the script should process. "
                "This is equivalent to piping data: echo 'data' | python script.py"
            ),
        },
    },
    "required": ["skill_name", "script_path"],
}


class ReadSkillTool:
    """Allows the agent to read skill content on demand."""

    def __init__(self, *, skill_manager: SkillManager | None = None) -> None:
        self._skill_manager = skill_manager

    def name(self) -> str:
        return TOOL_READ_SKILL

    def description(self) -> str:
        return _READ_SKILL_DESCRIPTION

    def parameters(self) -> str:
        return json.dumps(_READ_SKILL_SCHEMA, ensure_ascii=False)

    async def execute(self, ctx: Context, args: str) -> ToolResult:
        """Read a skill file or the skill's main instructions."""
        input_, parse_error = _parse_object_args(args)
        if parse_error is not None:
            return ToolResult(success=False, error=parse_error)
        skill_name = _as_str(input_.get("skill_name"))
        file_path = _as_str(input_.get("file_path"))

        if not skill_name:
            return ToolResult(success=False, error="skill_name is required")

        manager = self._skill_manager
        if manager is None or not manager.is_enabled():
            return ToolResult(success=False, error="Skills are not enabled")

        data: JsonObject
        if file_path:
            try:
                content = await manager.read_skill_file(ctx, skill_name, file_path)
            except Exception as exc:
                logger.error("Failed to read skill file: %s", exc)
                return ToolResult(success=False, error=f"Failed to read skill file: {exc}")
            output = f"=== Skill File: {skill_name}/{file_path} ===\n\n{content}"
            data = {
                "skill_name": skill_name,
                "file_path": file_path,
                "content": content,
                "content_length": len(content),
            }
        else:
            try:
                skill = await manager.load_skill(ctx, skill_name)
            except Exception as exc:
                logger.error("Failed to load skill: %s", exc)
                return ToolResult(success=False, error=f"Failed to load skill: {exc}")
            try:
                files = await manager.list_skill_files(ctx, skill_name)
            except Exception:
                files = []
            output = f"=== Skill: {skill.name} ===\n\n"
            output += f"**Description**: {skill.description}\n\n"
            output += "## Instructions\n\n"
            output += skill.instructions
            if len(files) > 1:  # more than just SKILL.md
                output += "\n\n## Available Files\n\n"
                output += (
                    "The following files are available in this skill directory. "
                    "Use `read_skill` with `file_path` to read them:\n\n"
                )
                for file in files:
                    if file == SKILL_FILE_NAME:
                        continue
                    if is_script(file):
                        output += f"- `{file}` (script - can be executed)\n"
                    else:
                        output += f"- `{file}`\n"
            data = {
                "skill_name": skill.name,
                "description": skill.description,
                "instructions": skill.instructions,
                "instructions_length": len(skill.instructions),
                "files": cast("list[JsonValue]", files),
            }

        logger.info("Successfully read skill: %s", skill_name)
        return ToolResult(success=True, output=output, data=data)


class ExecuteSkillScriptTool:
    """Allows the agent to execute skill scripts in a sandbox."""

    def __init__(self, *, skill_manager: SkillManager | None = None) -> None:
        self._skill_manager = skill_manager

    def name(self) -> str:
        return TOOL_EXECUTE_SKILL_SCRIPT

    def description(self) -> str:
        return _EXECUTE_SKILL_SCRIPT_DESCRIPTION

    def parameters(self) -> str:
        return json.dumps(_EXECUTE_SKILL_SCRIPT_SCHEMA, ensure_ascii=False)

    async def execute(self, ctx: Context, args: str) -> ToolResult:
        """Execute a skill script in the sandbox and format the outcome."""
        input_, parse_error = _parse_execute_input(args)
        if parse_error is not None:
            return ToolResult(success=False, error=parse_error)
        skill_name = _as_str(input_.get("skill_name"))
        script_path = _as_str(input_.get("script_path"))
        args_list = _as_str_list(input_.get("args"))
        stdin = _as_str(input_.get("input"))

        if not skill_name:
            return ToolResult(success=False, error="skill_name is required")
        if not script_path:
            return ToolResult(success=False, error="script_path is required")

        manager = self._skill_manager
        if manager is None or not manager.is_enabled():
            return ToolResult(success=False, error="Skills are not enabled")

        logger.info(
            "Executing script: %s/%s with args: %s, input length: %d",
            skill_name,
            script_path,
            args_list,
            len(stdin),
        )
        try:
            result = await manager.execute_script(
                ctx,
                skill_name,
                script_path,
                args_list,
                stdin,
            )
        except Exception as exc:
            logger.error("Script execution failed: %s", exc)
            return ToolResult(success=False, error=f"Script execution failed: {exc}")

        output = f"=== Script Execution: {skill_name}/{script_path} ===\n\n"
        if args_list:
            output += f"**Arguments**: {args_list}\n"
        output += f"**Exit Code**: {result.exit_code}\n"
        output += f"**Duration**: {_format_duration(result.duration)}\n\n"
        if result.killed:
            output += "**Warning**: Script was terminated (timeout or killed)\n\n"
        if result.stdout:
            output += "## Standard Output\n\n```\n"
            output += result.stdout
            if not result.stdout.endswith("\n"):
                output += "\n"
            output += "```\n\n"
        if result.stderr:
            output += "## Standard Error\n\n```\n"
            output += result.stderr
            if not result.stderr.endswith("\n"):
                output += "\n"
            output += "```\n\n"
        if result.error:
            output += "## Error\n\n"
            output += result.error
            output += "\n"

        success = result.is_success()
        data: JsonObject = {
            "skill_name": skill_name,
            "script_path": script_path,
            "args": cast("list[JsonValue]", args_list),
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_ms": int(result.duration * 1000),
            "killed": result.killed,
        }
        error = ""
        if not success:
            error = result.error if result.error else f"Script exited with code {result.exit_code}"

        logger.info("Script completed with exit code: %d", result.exit_code)
        return ToolResult(success=success, output=output, data=data, error=error)


def _parse_object_args(args: str) -> tuple[JsonObject, str | None]:
    """Parse tool args as a JSON object; ``(input, error_message)`` on failure."""
    try:
        parsed = json.loads(args)
    except json.JSONDecodeError as exc:
        return {}, f"Failed to parse args: {exc}"
    if not isinstance(parsed, dict):
        return {}, "Failed to parse args: expected a JSON object"
    return cast(JsonObject, parsed), None


def _parse_execute_input(args: str) -> tuple[JsonObject, str | None]:
    """Parse ``execute_skill_script`` args, accepting ``args`` as array or string.

    Some model providers emit a single string for a one-element ``args``;
    a string is interpreted as a conventional space-separated command line
    (the schema keeps advertising an array, so well-formed calls are
    unaffected).
    """
    input_, parse_error = _parse_object_args(args)
    if parse_error is not None:
        return {}, parse_error
    raw_args = input_.get("args")
    if raw_args is None:
        return input_, None
    if isinstance(raw_args, list):
        cleaned = [item for item in raw_args if isinstance(item, str)]
        return {**input_, "args": cast("list[JsonValue]", cleaned)}, None
    if isinstance(raw_args, str):
        return {**input_, "args": cast("list[JsonValue]", raw_args.split())}, None
    return {}, "args must be a string or an array of strings"


def _as_str(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


def _as_str_list(value: JsonValue) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _format_duration(duration_seconds: float) -> str:
    """Render a duration in seconds (like Go's ``time.Duration`` seconds)."""
    return f"{duration_seconds:.3f}s"


__all__ = [
    "SKILL_FILE_NAME",
    "TOOL_EXECUTE_SKILL_SCRIPT",
    "TOOL_READ_SKILL",
    "ExecuteSkillScriptTool",
    "ReadSkillTool",
    "Skill",
    "SkillManager",
    "is_script",
]
