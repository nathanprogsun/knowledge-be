"""Shared subprocess execution with timeout and process-group cleanup.

The child is started as the leader of a new session/process group (POSIX), so
an expired timeout can kill the whole tree — the script and any processes it
spawned — instead of leaving an orphan running.
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
from contextlib import suppress

from src.core.agents.engine.sandbox.types import ExecuteResult

_POSIX = os.name == "posix"
_TIMEOUT_MESSAGE = "execution timed out"


def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    """Terminate the process and, on POSIX, its whole group."""
    if _POSIX:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    else:
        with suppress(ProcessLookupError):
            process.kill()


async def _read_pipes(process: asyncio.subprocess.Process) -> tuple[bytes, bytes]:
    stdout_bytes = await process.stdout.read() if process.stdout is not None else b""
    stderr_bytes = await process.stderr.read() if process.stderr is not None else b""
    return stdout_bytes, stderr_bytes


async def run_command(
    *,
    argv: list[str],
    env: dict[str, str],
    cwd: str,
    stdin: str,
    timeout: float,
) -> ExecuteResult:
    """Run ``argv`` in ``cwd`` with ``timeout`` seconds; kill the group on expiry.

    A non-zero exit code is captured in the result, not raised; only a timeout
    marks the result as killed.
    """
    stdin_bytes = stdin.encode("utf-8") if stdin else None
    if _POSIX:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            env=env,
            cwd=cwd,
            stdin=asyncio.subprocess.PIPE if stdin_bytes is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    else:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            env=env,
            cwd=cwd,
            stdin=asyncio.subprocess.PIPE if stdin_bytes is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    start = time.monotonic()
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(input=stdin_bytes), timeout=timeout
        )
        if stdout_bytes is None:
            stdout_bytes = b""
        if stderr_bytes is None:
            stderr_bytes = b""
        exit_code = proc.returncode or 0
        killed = False
        error = ""
    except TimeoutError:
        _kill_process_group(proc)
        await proc.wait()
        stdout_bytes, stderr_bytes = await _read_pipes(proc)
        exit_code = -1
        killed = True
        error = _TIMEOUT_MESSAGE
    duration = time.monotonic() - start
    return ExecuteResult(
        stdout=stdout_bytes.decode("utf-8", errors="replace"),
        stderr=stderr_bytes.decode("utf-8", errors="replace"),
        exit_code=exit_code,
        duration=duration,
        killed=killed,
        error=error,
    )
