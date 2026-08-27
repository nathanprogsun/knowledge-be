"""Per-key distributed concurrency governor for model-provider calls.

The shared finite resource is the model provider, so concurrency is capped at
the model-client layer keyed by model ID rather than at the queue layer. The
Redis implementation is a self-healing distributed semaphore built on a sorted
set: each held slot is a ZSET member scored by its lease expiry. Acquire
atomically prunes expired leases, counts live holders, and admits a new one
only while under the limit. A background heartbeat refreshes the lease so long
calls keep their slot; a crashed holder's lease simply expires. Every backend
error fails open so a limiter outage can never halt model traffic.

Only background tasks (see ``background_task_context``) are gated; interactive
chat passes straight through. The ``set_governor`` singleton is wired once at
startup and shared by every model-client layer.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol, cast

from redis.asyncio import Redis

from src.app_logging import logger

#: Lease refresh happens every ``ttl / 3``; a live call never loses its slot.
DEFAULT_LEASE_TTL = timedelta(seconds=30)
#: How often a waiting acquirer re-checks for a free slot.
DEFAULT_POLL_INTERVAL = timedelta(milliseconds=200)
#: Namespaces the semaphore ZSETs in Redis.
KEY_PREFIX = "kb:modelsem:"

_ACQUIRE_LUA = """
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])
local count = redis.call('ZCARD', KEYS[1])
if count < tonumber(ARGV[2]) then
    redis.call('ZADD', KEYS[1], ARGV[1] + ARGV[4], ARGV[3])
    redis.call('PEXPIRE', KEYS[1], ARGV[4] * 2)
    return 1
end
return 0
"""


def _noop() -> None:
    """Release no-op used on the fail-open / passthrough paths."""


@dataclass(frozen=True)
class RuntimeStat:
    """Point-in-time view of one model semaphore."""

    model_id: str
    name: str = ""
    active: int = 0
    waiting: int = 0
    limit: int = 0


@dataclass
class _TrackedSemaphore:
    limit: int = 0
    waiting: int = 0
    active: int = 0
    name: str = ""


class _ScriptRunner(Protocol):
    """Executes the acquire Lua script against a (fake) Redis client."""

    async def __call__(
        self,
        keys: Sequence[object] | None = None,
        args: Sequence[object] | None = None,
    ) -> int: ...


class ModelConcurrencyLimiter(Protocol):
    """Caps concurrent in-flight calls per key across processes."""

    async def acquire(self, key: str, limit: int) -> Callable[[], None]:
        """Block until a slot for ``key`` is available.

        Returns a release function that MUST be invoked to free the slot. On
        any backend error the limiter fails open: the release is a no-op, so
        callers proceed without a slot rather than dropping the call.
        """
        ...


# ── Redis-backed distributed limiter ─────────────────────────────────


class RedisLimiter:
    """Self-healing distributed semaphore backed by a Redis sorted set."""

    def __init__(
        self,
        rdb: Redis | None,  # type: ignore[type-arg]
        *,
        lease_ttl: timedelta = DEFAULT_LEASE_TTL,
        poll_interval: timedelta = DEFAULT_POLL_INTERVAL,
    ) -> None:
        self._rdb = rdb
        self._ttl = lease_ttl
        self._poll_interval = poll_interval
        self._tracked: dict[str, _TrackedSemaphore] = {}
        self._script: _ScriptRunner | None = None
        if rdb is not None:
            self._script = cast(_ScriptRunner, rdb.register_script(_ACQUIRE_LUA))

    async def acquire(self, key: str, limit: int) -> Callable[[], None]:
        rdb = self._rdb
        script = self._script
        if rdb is None or script is None or limit <= 0 or not key:
            return _noop

        zkey = KEY_PREFIX + key
        tracked = self._tracked.get(key)
        if tracked is None:
            tracked = _TrackedSemaphore()
            self._tracked[key] = tracked
        tracked.limit = limit
        tracked.waiting += 1
        try:
            token = str(uuid.uuid4())
            ttl_ms = int(self._ttl.total_seconds() * 1000)
            while True:
                now_ms = int(time.time() * 1000)
                try:
                    admitted = await script(
                        keys=[zkey],
                        args=[now_ms, limit, token, ttl_ms],
                    )
                except Exception as exc:
                    logger.warning(
                        "[ModelLimiter] acquire failed for key={}, failing open: {}", key, exc
                    )
                    return _noop
                if admitted == 1:
                    return self._hold(rdb, zkey, token)
                await asyncio.sleep(self._poll_interval.total_seconds())
        finally:
            tracked.waiting -= 1

    def set_model_name(self, model_id: str, name: str) -> None:
        if not model_id or not name:
            return
        tracked = self._tracked.get(model_id)
        if tracked is None:
            tracked = _TrackedSemaphore()
            self._tracked[model_id] = tracked
        tracked.name = name

    async def runtime_stats(self) -> list[RuntimeStat]:
        rdb = self._rdb
        if rdb is None:
            return []
        now_ms = int(time.time() * 1000)
        stats: list[RuntimeStat] = []
        for model_id, tracked in self._tracked.items():
            try:
                active = await rdb.zcount(KEY_PREFIX + model_id, str(now_ms + 1), "+inf")
            except Exception:
                continue
            stats.append(
                RuntimeStat(
                    model_id=model_id,
                    name=tracked.name,
                    active=active,
                    waiting=tracked.waiting,
                    limit=tracked.limit,
                )
            )
        stats.sort(key=lambda stat: stat.model_id)
        return stats

    def _hold(self, rdb: Redis, zkey: str, token: str) -> Callable[[], None]:  # type: ignore[type-arg]
        """Start a lease-refresh heartbeat and return an idempotent release."""
        interval = self._ttl.total_seconds() / 3
        ttl_ms = int(self._ttl.total_seconds() * 1000)
        key_ttl_ms = int(self._ttl.total_seconds() * 2)

        async def _heartbeat() -> None:
            try:
                while True:
                    await asyncio.sleep(interval)
                    now_ms = int(time.time() * 1000)
                    await rdb.zadd(zkey, {token: float(now_ms + ttl_ms)})
                    await rdb.pexpire(zkey, key_ttl_ms)
            except asyncio.CancelledError:
                pass

        loop = asyncio.get_running_loop()
        task = loop.create_task(_heartbeat())
        released = False
        # Fire-and-forget cleanup task; kept referenced so the event loop does
        # not drop it before the ZRem runs.
        drop_task: asyncio.Task[None] | None = None

        async def _drop() -> None:
            with suppress(Exception):
                await rdb.zrem(zkey, token)

        def release() -> None:
            nonlocal released, drop_task
            if released:
                return
            released = True
            task.cancel()
            drop_task = loop.create_task(_drop())

        return release


# ── In-process (single-node) limiter ──────────────────────────────────


class LocalLimiter:
    """In-process counting semaphore keyed by model ID (Lite mode)."""

    def __init__(self) -> None:
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._tracked: dict[str, _TrackedSemaphore] = {}
        self._lock = threading.Lock()

    async def acquire(self, key: str, limit: int) -> Callable[[], None]:
        if limit <= 0 or not key:
            return _noop
        with self._lock:
            sem = self._sems.get(key)
            tracked = self._tracked.get(key)
            if tracked is None:
                tracked = _TrackedSemaphore()
                self._tracked[key] = tracked
            tracked.limit = limit
            if sem is None:
                sem = asyncio.Semaphore(limit)
                self._sems[key] = sem
        tracked.waiting += 1
        try:
            await sem.acquire()
        except asyncio.CancelledError:
            tracked.waiting -= 1
            raise
        tracked.waiting -= 1
        tracked.active += 1

        released = False

        def release() -> None:
            nonlocal released
            if released:
                return
            released = True
            tracked.active -= 1
            sem.release()

        return release

    def set_model_name(self, model_id: str, name: str) -> None:
        if not model_id or not name:
            return
        with self._lock:
            tracked = self._tracked.get(model_id)
            if tracked is None:
                tracked = _TrackedSemaphore()
                self._tracked[model_id] = tracked
            tracked.name = name

    async def runtime_stats(self) -> list[RuntimeStat]:
        with self._lock:
            stats = [
                RuntimeStat(
                    model_id=model_id,
                    name=tracked.name,
                    active=tracked.active,
                    waiting=tracked.waiting,
                    limit=tracked.limit,
                )
                for model_id, tracked in self._tracked.items()
            ]
        stats.sort(key=lambda stat: stat.model_id)
        return stats


# ── Process-wide governor ─────────────────────────────────────────────


_BACKGROUND_TASK: ContextVar[bool] = ContextVar("ai_background_task", default=False)

_governor: ModelConcurrencyLimiter | None = None
_governor_n: int = 0


@contextmanager
def background_task_context() -> Iterator[None]:
    """Mark the surrounding async scope as a background worker task.

    Only background LLM traffic is throttled by the concurrency governor, so
    interactive chat never waits on a semaphore.
    """
    token = _BACKGROUND_TASK.set(True)
    try:
        yield
    finally:
        _BACKGROUND_TASK.reset(token)


def is_background_task() -> bool:
    """True when the current scope was marked by ``background_task_context``."""
    return _BACKGROUND_TASK.get()


def set_governor(limiter: ModelConcurrencyLimiter | None, limit: int) -> None:
    """Install the process-wide governor and default per-model limit.

    A ``None`` limiter or non-positive limit disables governance.
    """
    global _governor, _governor_n
    _governor = limiter
    _governor_n = limit


def set_global_limit(limit: int) -> None:
    """Retune only the default per-model limit at runtime."""
    global _governor_n
    _governor_n = limit


async def gate(model_id: str) -> Callable[[], None]:
    """Acquire a per-model slot using the process-wide default limit."""
    return await gate_named_n(model_id, "", 0)


async def gate_n(model_id: str, model_limit: int) -> Callable[[], None]:
    """Acquire a per-model slot using ``model_limit`` (0 = default)."""
    return await gate_named_n(model_id, "", model_limit)


async def gate_named_n(model_id: str, model_name: str, model_limit: int) -> Callable[[], None]:
    """Acquire a background per-model slot; always safe to release.

    On the passthrough / fail-open paths the returned release is a no-op. The
    gate never blocks a call permanently — a limiter outage fails open.
    """
    limit = model_limit if model_limit > 0 else _governor_n
    governor = _governor
    if governor is None or limit <= 0 or not is_background_task():
        return _noop
    set_model_name = getattr(governor, "set_model_name", None)
    if callable(set_model_name):
        set_model_name(model_id, model_name)
    release = await governor.acquire(model_id, limit)
    if release is None:
        return _noop
    return release


async def runtime_stats() -> tuple[list[RuntimeStat], bool]:
    """Return observed semaphores and whether the governor exposes stats."""
    governor = _governor
    if governor is None:
        return [], False
    getter = getattr(governor, "runtime_stats", None)
    if not callable(getter):
        return [], False
    stats = await getter()
    return stats, True


__all__ = [
    "DEFAULT_LEASE_TTL",
    "DEFAULT_POLL_INTERVAL",
    "KEY_PREFIX",
    "LocalLimiter",
    "ModelConcurrencyLimiter",
    "RedisLimiter",
    "RuntimeStat",
    "background_task_context",
    "gate",
    "gate_n",
    "gate_named_n",
    "is_background_task",
    "runtime_stats",
    "set_global_limit",
    "set_governor",
]
