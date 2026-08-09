"""Local process-based sandbox.

The default backend when Docker is unavailable. Isolation is basic: a command
whitelist, a working-directory restriction, timeout enforcement, and
environment-variable filtering.
"""

from __future__ import annotations

import os

from src.core.agents.engine.sandbox._runner import run_command
from src.core.agents.engine.sandbox.errors import (
    SandboxExecutionFailedError,
    SandboxInvalidScriptError,
    SandboxScriptNotFoundError,
)
from src.core.agents.engine.sandbox.types import (
    DEFAULT_TIMEOUT,
    Config,
    ExecuteConfig,
    ExecuteResult,
    Sandbox,
    SandboxType,
    default_allowed_commands,
    default_config,
)

#: Environment variables that can override interpreter/library behavior and are
#: therefore excluded from the script's environment.
_DANGEROUS_ENV_VARS = frozenset(
    {
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "NODE_OPTIONS",
        "BASH_ENV",
        "ENV",
        "SHELL",
    }
)

#: Minimal environment the sandboxed script sees; nothing from the host
#: process leaks in.
_BASE_ENV = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "HOME": "/tmp",
    "LANG": "en_US.UTF-8",
    "LC_ALL": "en_US.UTF-8",
}


def interpreter_for_script(script_path: str) -> str:
    """Return the interpreter command for a script based on its extension."""
    ext = os.path.splitext(script_path)[1].lower()
    if ext == ".py":
        return "python3"
    if ext in (".sh", ".bash"):
        return "bash"
    if ext == ".js":
        return "node"
    if ext == ".rb":
        return "ruby"
    if ext == ".pl":
        return "perl"
    if ext == ".php":
        return "php"
    return "sh"


class LocalSandbox(Sandbox):
    """Implements ``Sandbox`` using local process isolation."""

    def __init__(self, config: Config | None = None) -> None:
        self._config = config if config is not None else default_config()

    def sandbox_type(self) -> SandboxType:
        return SandboxType.LOCAL

    async def is_available(self) -> bool:
        """Local sandbox is always available."""
        return True

    async def execute(self, config: ExecuteConfig) -> ExecuteResult:
        if config is None:
            raise SandboxInvalidScriptError()
        self._validate_script(config.script)
        interpreter = self._get_interpreter(config.script)
        if not self._is_allowed_command(interpreter):
            raise SandboxExecutionFailedError(message=f"interpreter not allowed: {interpreter}")
        timeout = config.timeout or self._config.default_timeout or DEFAULT_TIMEOUT
        work_dir = config.work_dir or os.path.dirname(config.script)
        env = self._build_environment(config.env)
        return await run_command(
            argv=[interpreter, config.script, *config.args],
            env=env,
            cwd=work_dir,
            stdin=config.stdin,
            timeout=timeout,
        )

    async def cleanup(self) -> None:
        """Local sandbox keeps no resources to release."""
        return

    def _validate_script(self, script_path: str) -> None:
        """Check the script path is a readable file inside the allowed paths."""
        if not os.path.exists(script_path):
            raise SandboxScriptNotFoundError()
        if os.path.isdir(script_path):
            raise SandboxInvalidScriptError()
        if not os.path.isabs(script_path):
            raise SandboxInvalidScriptError(message=f"script path must be absolute: {script_path}")
        if self._config.allowed_paths:
            abs_path = os.path.abspath(script_path)
            if not any(
                abs_path.startswith(os.path.abspath(allowed))
                for allowed in self._config.allowed_paths
            ):
                raise SandboxInvalidScriptError(
                    message=f"script path not in allowed paths: {script_path}"
                )

    def _get_interpreter(self, script_path: str) -> str:
        return interpreter_for_script(script_path)

    def _is_allowed_command(self, cmd: str) -> bool:
        allowed = self._config.allowed_commands or default_allowed_commands()
        return cmd in allowed

    def _build_environment(self, extra: dict[str, str]) -> dict[str, str]:
        env = dict(_BASE_ENV)
        for key, value in extra.items():
            if key.upper() in _DANGEROUS_ENV_VARS:
                continue
            env[key] = value
        return env


__all__ = ["LocalSandbox", "interpreter_for_script"]
