"""Per-model concurrency governor for embedding calls (concurrency_wrapper.go).

Embedding is the highest-volume background model call: document ingestion
vectorises every chunk, so a single batch upload can burst the whole worker
pool against one embedding provider. Like chat and vlm, embedding is
governed at the client layer via a shared per-model concurrency governor.
Only background (worker) calls are throttled — interactive query embedding
is never gated.

Placement: the wrapper sits innermost — directly around the real embedder.
``batch_embed_with_pool`` fans a batch out into per-sub-batch
``batch_embed`` calls through the pooler, and the pooler invokes
``batch_embed`` on whichever embedder was threaded down as ``model``.
Sitting innermost routes those per-sub-batch provider round-trips back
through the gate, so the semaphore bounds real concurrent provider calls
rather than one coarse per-document unit.

The governor here is self-contained (an in-process per-key counting
semaphore) so this PR does not depend on the shared distributed limiter;
the ``ConcurrencyLimiter`` protocol documents the seam a Redis-backed
implementation would fill. Every backend error fails OPEN (the call is
allowed) so a limiter outage can never halt model traffic.

The module references the ``Embedder`` protocol through a module-level
import of ``src.ai.embedding.base`` (rather than a from-import) to break
the base↔concurrency import cycle.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import src.ai.embedding.base as base

# A release handle returned by a successful acquire; always safe to call.
Release = Callable[[], None]


def _noop_release() -> None:
    return None


# ── Background-task marker ───────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TaskContext:
    """Default ``Context`` implementation (interactive by default).

    Background ingestion workers construct
    ``TaskContext(is_background_task=True)`` so the governor throttles
    their calls; interactive request-scoped paths leave the flag off.
    """

    is_background_task: bool = False


class ConcurrencyLimiter(Protocol):
    """Per-key concurrency limiter (upstream ``ModelConcurrencyLimiter``).

    ``acquire`` blocks until a slot for ``key`` is available or the task
    is cancelled, then returns a release callable. On any backend error
    it fails open: release is a no-op.
    """

    async def acquire(self, ctx: base.Context, key: str, limit: int) -> Release: ...


class LocalLimiter:
    """In-process per-key counting semaphore (the local limiter mode).

    The single-node counterpart to a distributed limiter: Lite runs a
    single process, so a shared semaphore is neither available nor
    needed — but background ingestion can still burst the whole worker
    pool against one provider, so per-model concurrency is still capped.
    """

    def __init__(self) -> None:
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._guard = asyncio.Lock()

    async def acquire(self, ctx: base.Context, key: str, limit: int) -> Release:
        if key == "" or limit <= 0:
            return _noop_release
        async with self._guard:
            sem = self._sems.get(key)
            if sem is None:
                sem = asyncio.Semaphore(limit)
                self._sems[key] = sem
        try:
            await sem.acquire()
        except asyncio.CancelledError:
            return _noop_release
        return sem.release


# ── Process-wide governor ────────────────────────────────────────────

_governor_lock = threading.Lock()
_governor: ConcurrencyLimiter | None = None
_governor_default_limit = 0


def set_governor(limiter: ConcurrencyLimiter | None, limit: int) -> None:
    """Install the process-wide governor and the default per-model limit.

    Passing ``None`` or a non-positive ``limit`` disables governance
    (all calls pass through). Safe to call at startup.
    """
    global _governor, _governor_default_limit
    with _governor_lock:
        _governor = limiter
        _governor_default_limit = limit


def set_global_limit(limit: int) -> None:
    """Update ONLY the process-wide default per-model limit.

    Leaves the installed limiter backend intact. A non-positive value
    disables the default (models that carry their own ``max_concurrency``
    still honour it).
    """
    global _governor_default_limit
    with _governor_lock:
        _governor_default_limit = limit


def _read_governor() -> tuple[ConcurrencyLimiter | None, int]:
    with _governor_lock:
        return _governor, _governor_default_limit


async def gate_named_n(
    ctx: base.Context,
    model_id: str,
    model_name: str,
    model_limit: int,
) -> Release:
    """Acquire a per-model concurrency slot for a background call.

    Equivalent to the upstream ``GateNamedN``: only background tasks are
    gated, and only when a governor is installed with a positive limit.
    ``model_limit`` is the model's own cap; ``<= 0`` falls back to the
    process-wide default. The returned release is always safe to call.
    """
    limiter, default_limit = _read_governor()
    limit = model_limit if model_limit > 0 else default_limit
    if limiter is None or limit <= 0 or not ctx.is_background_task:
        return _noop_release
    try:
        release = await limiter.acquire(ctx, model_id, limit)
    except Exception:
        return _noop_release
    if release is None:
        return _noop_release
    return release


# ── Concurrency wrapper ──────────────────────────────────────────────


class ConcurrencyEmbedder:
    """Governs a wrapped embedder's provider round-trips (upstream wrapper).

    Each ``embed`` / ``batch_embed`` call acquires a per-model slot
    before delegating; ``batch_embed_with_pool`` threads THIS wrapper down
    as the model so the pooler's per-sub-batch callbacks land on the
    gated ``batch_embed`` above rather than on the raw embedder.
    """

    def __init__(self, inner: base.Embedder, limit: int) -> None:
        self._inner = inner
        self._limit = limit

    async def embed(self, ctx: base.Context, text: str) -> list[float]:
        release = await gate_named_n(
            ctx,
            self._inner.get_model_id(),
            self._inner.get_model_name(),
            self._limit,
        )
        try:
            return await self._inner.embed(ctx, text)
        finally:
            release()

    async def batch_embed(self, ctx: base.Context, texts: list[str]) -> list[list[float]]:
        release = await gate_named_n(
            ctx,
            self._inner.get_model_id(),
            self._inner.get_model_name(),
            self._limit,
        )
        try:
            return await self._inner.batch_embed(ctx, texts)
        finally:
            release()

    async def batch_embed_with_pool(
        self,
        ctx: base.Context,
        model: base.Embedder,
        texts: list[str],
    ) -> list[list[float]]:
        return await self._inner.batch_embed_with_pool(ctx, self, texts)

    def get_model_name(self) -> str:
        return self._inner.get_model_name()

    def get_dimensions(self) -> int:
        return self._inner.get_dimensions()

    def get_model_id(self) -> str:
        return self._inner.get_model_id()


def wrap_embedding_concurrency(embedder: base.Embedder, limit: int) -> base.Embedder:
    """Install the background concurrency governor around an embedder."""
    return ConcurrencyEmbedder(embedder, limit)


__all__ = [
    "ConcurrencyEmbedder",
    "ConcurrencyLimiter",
    "LocalLimiter",
    "Release",
    "TaskContext",
    "gate_named_n",
    "set_global_limit",
    "set_governor",
    "wrap_embedding_concurrency",
]
