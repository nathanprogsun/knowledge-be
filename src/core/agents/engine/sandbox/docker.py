"""Docker-based sandbox.

An optional injectable backend: containers provide stronger isolation than the
local process sandbox. Requires a working ``docker`` CLI; availability is
checked lazily at initialization so tests and Docker-less hosts can fall back
to the local backend.
"""

from __future__ import annotations

import asyncio
import os

from src.core.agents.engine.sandbox._runner import run_command
from src.core.agents.engine.sandbox.errors import SandboxInvalidScriptError
from src.core.agents.engine.sandbox.local import interpreter_for_script
from src.core.agents.engine.sandbox.types import (
    DEFAULT_DOCKER_IMAGE,
    DEFAULT_TIMEOUT,
    Config,
    ExecuteConfig,
    ExecuteResult,
    Sandbox,
    SandboxType,
    default_config,
)


class DockerSandbox(Sandbox):
    """Implements ``Sandbox`` using Docker containers."""

    def __init__(self, config: Config | None = None) -> None:
        resolved = config if config is not None else default_config()
        if resolved.docker_image == "":
            resolved = resolved.model_copy(update={"docker_image": DEFAULT_DOCKER_IMAGE})
        self._config = resolved

    def sandbox_type(self) -> SandboxType:
        return SandboxType.DOCKER

    async def is_available(self) -> bool:
        """Report whether the ``docker`` CLI responds."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "version",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError:
            return False
        return (await proc.wait()) == 0

    async def image_exists(self) -> bool:
        """Report whether the configured image exists locally."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "image",
                "inspect",
                self._config.docker_image,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError:
            return False
        return (await proc.wait()) == 0

    async def ensure_image(self) -> None:
        """Pull the configured image when it does not exist locally."""
        if await self.image_exists():
            return
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "pull",
                self._config.docker_image,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError:
            return
        await proc.wait()

    async def execute(self, config: ExecuteConfig) -> ExecuteResult:
        if config is None:
            raise SandboxInvalidScriptError()
        timeout = config.timeout or self._config.default_timeout or DEFAULT_TIMEOUT
        args = self._build_docker_args(config)
        return await run_command(
            argv=["docker", *args],
            env=dict(os.environ),
            cwd=os.getcwd(),
            stdin=config.stdin,
            timeout=timeout,
        )

    async def cleanup(self) -> None:
        """The ``--rm`` flag handles container cleanup; nothing to release."""
        return

    def _build_docker_args(self, config: ExecuteConfig) -> list[str]:
        args = ["run", "--rm"]
        args += ["--user", "1000:1000"]
        args += ["--cap-drop", "ALL"]
        if config.read_only_rootfs:
            args += ["--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m"]
        mem_limit = config.memory_limit or self._config.max_memory
        if mem_limit > 0:
            args += ["--memory", str(mem_limit), "--memory-swap", str(mem_limit)]
        cpu_limit = config.cpu_limit or self._config.max_cpu
        if cpu_limit > 0:
            args += ["--cpus", f"{cpu_limit:.2f}"]
        if not config.allow_network:
            args += ["--network", "none"]
        args += ["--pids-limit", "100", "--security-opt", "no-new-privileges"]
        script_dir = os.path.dirname(config.script)
        args += ["-v", f"{script_dir}:/workspace:ro"]
        args += ["-w", "/workspace"]
        for key, value in config.env.items():
            args += ["-e", f"{key}={value}"]
        args += [self._config.docker_image]
        script_name = os.path.basename(config.script)
        args += [interpreter_for_script(script_name), script_name]
        args += list(config.args)
        return args


__all__ = ["DockerSandbox"]
