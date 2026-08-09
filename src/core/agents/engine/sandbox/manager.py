"""Default sandbox manager: backend selection, fallback, and validation.

The manager applies security validation before any execution unless explicitly
skipped, then delegates to the active backend. Validation failures are raised
as ``SandboxSecurityViolationError`` subclasses.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from src.core.agents.engine.sandbox.docker import DockerSandbox
from src.core.agents.engine.sandbox.errors import (
    SandboxArgInjectionError,
    SandboxConfigError,
    SandboxDisabledError,
    SandboxScriptNotFoundError,
    SandboxSecurityViolationError,
    SandboxStdinInjectionError,
)
from src.core.agents.engine.sandbox.local import LocalSandbox
from src.core.agents.engine.sandbox.types import (
    Config,
    ExecuteConfig,
    ExecuteResult,
    Manager,
    Sandbox,
    SandboxType,
    default_config,
    validate_config,
)
from src.core.agents.engine.sandbox.validator import ScriptValidator


def _read_script_content(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise SandboxScriptNotFoundError(
            message=f"failed to read script for validation: {exc}"
        ) from exc


class DisabledSandbox(Sandbox):
    """No-op sandbox that rejects all execution requests."""

    async def execute(self, config: ExecuteConfig) -> ExecuteResult:
        raise SandboxDisabledError()

    async def cleanup(self) -> None:
        return None

    def sandbox_type(self) -> SandboxType:
        return SandboxType.DISABLED

    async def is_available(self) -> bool:
        return False


class DefaultManager(Manager):
    """Default manager handling sandbox selection and fallback logic."""

    def __init__(self, config: Config | None = None) -> None:
        resolved = config if config is not None else default_config()
        validate_config(resolved)
        self._config = resolved
        self._validator = ScriptValidator()
        if resolved.sandbox_type is SandboxType.DISABLED:
            self._sandbox: Sandbox | None = DisabledSandbox()
            self._initialized = True
        elif resolved.sandbox_type is SandboxType.LOCAL:
            self._sandbox = LocalSandbox(resolved)
            self._initialized = True
        else:
            self._sandbox = None
            self._initialized = False

    async def initialize(self) -> None:
        """Resolve the active backend, checking Docker availability on demand."""
        if self._initialized:
            return
        if self._config.sandbox_type is SandboxType.DOCKER:
            docker_sandbox = DockerSandbox(self._config)
            if await docker_sandbox.is_available():
                self._sandbox = docker_sandbox
            elif self._config.fallback_enabled:
                self._sandbox = LocalSandbox(self._config)
            else:
                raise SandboxConfigError(message="docker is not available and fallback is disabled")
            self._initialized = True

    async def execute(self, config: ExecuteConfig) -> ExecuteResult:
        """Run a script through the active backend after security validation."""
        await self.initialize()
        sandbox = self._sandbox
        if sandbox is None or sandbox.sandbox_type() is SandboxType.DISABLED:
            raise SandboxDisabledError()
        if not config.skip_validation:
            await self._validate_execution(config)
        return await sandbox.execute(config)

    async def cleanup(self) -> None:
        await self.initialize()
        if self._sandbox is not None:
            await self._sandbox.cleanup()

    def sandbox(self) -> Sandbox | None:
        return self._sandbox

    def sandbox_type(self) -> SandboxType:
        if self._sandbox is not None:
            return self._sandbox.sandbox_type()
        return SandboxType.DISABLED

    async def _validate_execution(self, config: ExecuteConfig) -> None:
        """Perform comprehensive security validation on the execution config."""
        if self._validator is None:
            return
        script_content = config.script_content
        if script_content == "" and config.script != "":
            script_content = await asyncio.to_thread(_read_script_content, config.script)
        if script_content != "":
            result = self._validator.validate_script(script_content)
            if not result.valid:
                failure = result.errors[0] if result.errors else None
                raise SandboxSecurityViolationError(
                    message=failure.message if failure is not None else "Security validation failed"
                )
        if config.args:
            result = self._validator.validate_args(config.args)
            if not result.valid:
                failure = result.errors[0] if result.errors else None
                raise SandboxArgInjectionError(
                    message=failure.message
                    if failure is not None
                    else "Argument injection detected"
                )
        if config.stdin != "":
            result = self._validator.validate_stdin(config.stdin)
            if not result.valid:
                failure = result.errors[0] if result.errors else None
                raise SandboxStdinInjectionError(
                    message=failure.message if failure is not None else "Stdin injection detected"
                )


def new_manager(config: Config | None = None) -> DefaultManager:
    """Create a sandbox manager with the given configuration."""
    return DefaultManager(config)


def new_manager_from_type(
    sandbox_type: str, fallback_enabled: bool, docker_image: str = ""
) -> DefaultManager:
    """Create a manager from a type string; ``docker_image`` is optional."""
    if sandbox_type == "docker":
        s_type = SandboxType.DOCKER
    elif sandbox_type == "local":
        s_type = SandboxType.LOCAL
    elif sandbox_type in ("disabled", ""):
        s_type = SandboxType.DISABLED
    else:
        raise SandboxConfigError(message=f"unknown sandbox type: {sandbox_type}")
    config = default_config().model_copy(
        update={"sandbox_type": s_type, "fallback_enabled": fallback_enabled}
    )
    if docker_image != "":
        config = config.model_copy(update={"docker_image": docker_image})
    return DefaultManager(config)


def new_disabled_manager() -> DefaultManager:
    """Create a manager that rejects all execution requests."""
    return DefaultManager(
        default_config().model_copy(update={"sandbox_type": SandboxType.DISABLED})
    )


__all__ = [
    "DefaultManager",
    "DisabledSandbox",
    "new_disabled_manager",
    "new_manager",
    "new_manager_from_type",
]
