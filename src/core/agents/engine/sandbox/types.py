"""Sandbox domain types, configuration, and the backend interfaces.

The ``Sandbox`` protocol is implemented by the local and Docker backends; the
``Manager`` protocol is implemented by the default manager that selects a
backend and applies security validation before execution.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from src.core.agents.engine.sandbox.errors import SandboxConfigError

DEFAULT_TIMEOUT = 60.0  # seconds
DEFAULT_MEMORY_LIMIT = 256 * 1024 * 1024  # 256MB
DEFAULT_CPU_LIMIT = 1.0  # 1 CPU core
DEFAULT_DOCKER_IMAGE = "wechatopenai/kb-sandbox:latest"


class SandboxType(StrEnum):
    """The execution backend kind."""

    DOCKER = "docker"
    LOCAL = "local"
    DISABLED = "disabled"


class ExecuteConfig(BaseModel):
    """Configuration for one script execution."""

    script: str
    args: list[str] = Field(default_factory=list)
    work_dir: str = ""
    timeout: float = 0.0  # seconds; 0 = use the sandbox default
    env: dict[str, str] = Field(default_factory=dict)
    allowed_cmds: list[str] = Field(default_factory=list)
    allow_network: bool = False
    memory_limit: int = 0
    cpu_limit: float = 0.0
    read_only_rootfs: bool = False
    stdin: str = ""
    skip_validation: bool = False
    script_content: str = ""


class ExecuteResult(BaseModel):
    """Outcome of one script execution."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration: float = 0.0
    killed: bool = False
    error: str = ""

    def is_success(self) -> bool:
        """Return whether the script exited cleanly without being killed."""
        return self.exit_code == 0 and not self.killed and self.error == ""

    def get_output(self) -> str:
        """Return the combined output, preferring stdout."""
        if self.stdout != "":
            return self.stdout
        return self.stderr


class Config(BaseModel):
    """Sandbox manager configuration."""

    sandbox_type: SandboxType = SandboxType.LOCAL
    fallback_enabled: bool = True
    default_timeout: float = DEFAULT_TIMEOUT
    docker_image: str = DEFAULT_DOCKER_IMAGE
    allowed_commands: list[str] = Field(default_factory=lambda: default_allowed_commands())
    allowed_paths: list[str] = Field(default_factory=list)
    max_memory: int = DEFAULT_MEMORY_LIMIT
    max_cpu: float = DEFAULT_CPU_LIMIT


def default_allowed_commands() -> list[str]:
    """Return the default whitelist of safe interpreter commands."""
    return [
        "python",
        "python3",
        "node",
        "bash",
        "sh",
        "cat",
        "echo",
        "head",
        "tail",
        "grep",
        "sed",
        "awk",
        "sort",
        "uniq",
        "wc",
        "cut",
        "tr",
        "ls",
        "pwd",
        "date",
    ]


def default_config() -> Config:
    """Return the default sandbox configuration."""
    return Config(
        sandbox_type=SandboxType.LOCAL,
        fallback_enabled=True,
        default_timeout=DEFAULT_TIMEOUT,
        docker_image=DEFAULT_DOCKER_IMAGE,
        allowed_commands=default_allowed_commands(),
        max_memory=DEFAULT_MEMORY_LIMIT,
        max_cpu=DEFAULT_CPU_LIMIT,
    )


def validate_config(config: Config) -> None:
    """Validate sandbox configuration, raising ``SandboxConfigError`` on failure."""
    if config.sandbox_type not in (SandboxType.DOCKER, SandboxType.LOCAL, SandboxType.DISABLED):
        raise SandboxConfigError(message="invalid sandbox type")
    if config.default_timeout < 0:
        raise SandboxConfigError(message="timeout cannot be negative")
    if config.max_memory < 0:
        raise SandboxConfigError(message="memory limit cannot be negative")
    if config.max_cpu < 0:
        raise SandboxConfigError(message="CPU limit cannot be negative")


@runtime_checkable
class Sandbox(Protocol):
    """Interface for isolated script execution."""

    async def execute(self, config: ExecuteConfig) -> ExecuteResult:
        """Run a script in an isolated environment."""
        ...

    async def cleanup(self) -> None:
        """Release sandbox resources."""
        ...

    def sandbox_type(self) -> SandboxType:
        """Return the sandbox type."""
        ...

    async def is_available(self) -> bool:
        """Report whether this sandbox is available for use."""
        ...


@runtime_checkable
class Manager(Protocol):
    """Unified interface for sandbox operations and fallback logic."""

    async def execute(self, config: ExecuteConfig) -> ExecuteResult:
        """Run a script using the configured sandbox."""
        ...

    async def cleanup(self) -> None:
        """Release all sandbox resources."""
        ...

    def sandbox(self) -> Sandbox | None:
        """Return the active sandbox."""
        ...

    def sandbox_type(self) -> SandboxType:
        """Return the current sandbox type."""
        ...


__all__ = [
    "DEFAULT_CPU_LIMIT",
    "DEFAULT_DOCKER_IMAGE",
    "DEFAULT_MEMORY_LIMIT",
    "DEFAULT_TIMEOUT",
    "Config",
    "ExecuteConfig",
    "ExecuteResult",
    "Manager",
    "Sandbox",
    "SandboxType",
    "default_allowed_commands",
    "default_config",
    "validate_config",
]
