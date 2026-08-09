"""Sandboxed execution of untrusted scripts.

The default manager selects a backend (local process by default, Docker when
available and configured), applies security validation before every execution,
and enforces timeouts and resource limits.
"""

from __future__ import annotations

from src.core.agents.engine.sandbox.errors import (
    SandboxArgInjectionError,
    SandboxConfigError,
    SandboxDisabledError,
    SandboxError,
    SandboxExecutionFailedError,
    SandboxInvalidScriptError,
    SandboxScriptNotFoundError,
    SandboxSecurityViolationError,
    SandboxStdinInjectionError,
    SandboxTimeoutError,
)
from src.core.agents.engine.sandbox.manager import (
    DefaultManager,
    new_disabled_manager,
    new_manager,
    new_manager_from_type,
)
from src.core.agents.engine.sandbox.types import (
    DEFAULT_CPU_LIMIT,
    DEFAULT_DOCKER_IMAGE,
    DEFAULT_MEMORY_LIMIT,
    DEFAULT_TIMEOUT,
    Config,
    ExecuteConfig,
    ExecuteResult,
    SandboxType,
    default_allowed_commands,
    default_config,
    validate_config,
)
from src.core.agents.engine.sandbox.validator import (
    ScriptValidator,
    ValidationFailure,
    ValidationResult,
)

__all__ = [
    "DEFAULT_CPU_LIMIT",
    "DEFAULT_DOCKER_IMAGE",
    "DEFAULT_MEMORY_LIMIT",
    "DEFAULT_TIMEOUT",
    "Config",
    "DefaultManager",
    "ExecuteConfig",
    "ExecuteResult",
    "SandboxArgInjectionError",
    "SandboxConfigError",
    "SandboxDisabledError",
    "SandboxError",
    "SandboxExecutionFailedError",
    "SandboxInvalidScriptError",
    "SandboxScriptNotFoundError",
    "SandboxSecurityViolationError",
    "SandboxStdinInjectionError",
    "SandboxTimeoutError",
    "SandboxType",
    "ScriptValidator",
    "ValidationFailure",
    "ValidationResult",
    "default_allowed_commands",
    "default_config",
    "new_disabled_manager",
    "new_manager",
    "new_manager_from_type",
    "validate_config",
]
