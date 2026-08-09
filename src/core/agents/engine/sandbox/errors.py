"""Application error subclasses for sandboxed script execution.

All sandbox failures derive from ``SandboxError`` (an ``ApplicationError``)
so the web layer can translate them uniformly. Validation failures are split
by cause but share the ``SandboxSecurityViolationError`` base, mirroring the
reference manager that surfaces every pre-execution rejection as a security
violation.
"""

from __future__ import annotations

from src.common.exception import ApplicationError


class SandboxError(ApplicationError):
    """Base for every sandbox execution failure."""

    code = "sandbox_error"
    message = "Sandbox execution failed"


class SandboxDisabledError(SandboxError):
    """Raised when script execution is disabled."""

    code = "sandbox_disabled"
    message = "Sandbox is disabled"


class SandboxTimeoutError(SandboxError):
    """Raised when a script exceeds its execution timeout."""

    code = "sandbox_timeout"
    message = "Execution timed out"


class SandboxScriptNotFoundError(SandboxError):
    """Raised when the script file does not exist."""

    code = "sandbox_script_not_found"
    message = "Script not found"


class SandboxInvalidScriptError(SandboxError):
    """Raised when the script path is invalid or outside the allowed paths."""

    code = "sandbox_invalid_script"
    message = "Invalid script"


class SandboxExecutionFailedError(SandboxError):
    """Raised when the script cannot be started (e.g. interpreter denied)."""

    code = "sandbox_execution_failed"
    message = "Script execution failed"


class SandboxSecurityViolationError(SandboxError):
    """Raised when pre-execution security validation fails."""

    code = "sandbox_security_violation"
    message = "Security validation failed"


class SandboxArgInjectionError(SandboxSecurityViolationError):
    """Raised when a script argument contains an injection pattern."""

    code = "sandbox_arg_injection"
    message = "Argument injection detected"


class SandboxStdinInjectionError(SandboxSecurityViolationError):
    """Raised when stdin content contains embedded shell commands."""

    code = "sandbox_stdin_injection"
    message = "Stdin injection detected"


class SandboxConfigError(SandboxError):
    """Raised when the sandbox configuration is invalid."""

    code = "sandbox_config"
    message = "Invalid sandbox configuration"


__all__ = [
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
]
