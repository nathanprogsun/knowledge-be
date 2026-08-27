"""Tests for the per-key concurrency limiter (local + Redis) and governor.

The Redis limiter is exercised against an in-memory fake client that
reimplements the acquire script's sorted-set semantics, so no Redis server or
network is involved.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import timedelta
from typing import cast

import pytest
from redis.asyncio import Redis

from src.ai.llm.limiter import (
    LocalLimiter,
    RedisLimiter,
    background_task_context,
    gate,
    gate_n,
    gate_named_n,
    runtime_stats,
    set_global_limit,
    set_governor,
)


class FakeScript:
    def __init__(self, fake: FakeRedis) -> None:
        self._fake = fake

    async def __call__(
        self, keys: list[str] | None = None, args: list[object] | None = None
    ) -> int:
        if self._fake.fail_script:
            raise RuntimeError("redis down")
        zkey = (keys or [""])[0]
        values = args or []
        now_ms = int(str(values[0]))
        limit = int(str(values[1]))
        ttl_ms = int(str(values[3]))
        token = str(values[2])
        zset = self._fake.zsets.setdefault(zkey, {})
        for member, score in list(zset.items()):
            if score <= now_ms:
                del zset[member]
        if len(zset) < limit:
            zset[token] = now_ms + ttl_ms
            return 1
        return 0


class FakeRedis:
    """In-memory sorted-set stand-in for the limiter's Redis surface."""

    def __init__(self) -> None:
        self.zsets: dict[str, dict[str, int]] = {}
        self.fail_script = False

    def register_script(self, script: str) -> FakeScript:
        return FakeScript(self)

    async def zadd(self, name: str, mapping: dict[str, float]) -> int:
        zset = self.zsets.setdefault(name, {})
        for member, score in mapping.items():
            zset[member] = int(score)
        return len(mapping)

    async def zrem(self, name: str, *values: str) -> int:
        zset = self.zsets.get(name, {})
        removed = 0
        for value in values:
            if value in zset:
                del zset[value]
                removed += 1
        return removed

    async def zcount(self, name: str, min: str, max: str) -> int:
        zset = self.zsets.get(name, {})
        low = int(min) if min != "-inf" else float("-inf")
        high = int(max) if max != "+inf" else float("inf")
        return sum(1 for score in zset.values() if low < score < high)

    async def pexpire(self, name: str, time_ms: int) -> bool:
        return name in self.zsets


@pytest.fixture(autouse=True)
def _reset_governor() -> Iterator[None]:
    yield
    set_governor(None, 0)


# ── LocalLimiter ─────────────────────────────────────────────────────


async def test_local_limiter_caps_concurrency() -> None:
    limiter = LocalLimiter()
    release1 = await limiter.acquire("m1", 1)
    assert release1 is not None

    waiting = asyncio.create_task(limiter.acquire("m1", 1))
    await asyncio.sleep(0.02)
    assert not waiting.done()

    release1()
    release2 = await asyncio.wait_for(waiting, timeout=1)
    release2()

    stats = await limiter.runtime_stats()
    assert stats[0].model_id == "m1"
    assert stats[0].active == 0
    assert stats[0].limit == 1


async def test_local_limiter_fail_open_for_non_positive_limit() -> None:
    limiter = LocalLimiter()
    release = await limiter.acquire("m1", 0)
    release()
    assert await limiter.runtime_stats() == []


async def test_local_limiter_set_model_name_and_stats() -> None:
    limiter = LocalLimiter()
    release = await limiter.acquire("m1", 2)
    limiter.set_model_name("m1", "model-one")
    stats = await limiter.runtime_stats()
    assert stats[0].name == "model-one"
    assert stats[0].active == 1
    release()
    stats = await limiter.runtime_stats()
    assert stats[0].active == 0


# ── RedisLimiter ─────────────────────────────────────────────────────


def _redis_limiter(fake: FakeRedis) -> RedisLimiter:
    return RedisLimiter(
        cast(Redis, fake),  # type: ignore[type-arg]
        lease_ttl=timedelta(seconds=1),
        poll_interval=timedelta(milliseconds=10),
    )


async def test_redis_limiter_acquire_and_release() -> None:
    fake = FakeRedis()
    limiter = _redis_limiter(fake)
    release = await limiter.acquire("m1", 1)
    assert release is not None
    stats = await limiter.runtime_stats()
    assert stats[0].active == 1
    release()
    await asyncio.sleep(0.01)
    stats = await limiter.runtime_stats()
    assert stats[0].active == 0


async def test_redis_limiter_waits_for_free_slot() -> None:
    fake = FakeRedis()
    limiter = _redis_limiter(fake)
    release1 = await limiter.acquire("m1", 1)

    waiting = asyncio.create_task(limiter.acquire("m1", 1))
    await asyncio.sleep(0.03)
    assert not waiting.done()

    release1()
    release2 = await asyncio.wait_for(waiting, timeout=1)
    release2()
    await asyncio.sleep(0.01)


async def test_redis_limiter_fails_open_on_backend_error() -> None:
    fake = FakeRedis()
    limiter = _redis_limiter(fake)
    fake.fail_script = True
    release = await limiter.acquire("m1", 1)
    release()  # no-op, must not raise


async def test_redis_limiter_nil_client_fails_open() -> None:
    limiter = RedisLimiter(None)
    release = await limiter.acquire("m1", 1)
    release()
    assert await limiter.runtime_stats() == []


# ── Governor ─────────────────────────────────────────────────────────


async def test_gate_passthrough_outside_background_task() -> None:
    set_governor(LocalLimiter(), 1)
    release = await gate_named_n("m1", "model-one", 1)
    release()
    release = await gate_n("m1", 1)
    release()
    release = await gate("m1")
    release()


async def test_gate_acquires_slot_inside_background_task() -> None:
    set_governor(LocalLimiter(), 2)
    with background_task_context():
        release = await gate_named_n("m1", "model-one", 0)
        assert release is not None
        release()


async def test_gate_uses_global_default_limit() -> None:
    set_governor(LocalLimiter(), 1)
    set_global_limit(1)
    with background_task_context():
        release1 = await gate_named_n("m1", "", 0)
        assert release1 is not None
        waiting = asyncio.create_task(gate_named_n("m1", "", 0))
        await asyncio.sleep(0.02)
        assert not waiting.done()
        release1()
        release2 = await asyncio.wait_for(waiting, timeout=1)
        release2()


async def test_runtime_stats_reports_availability() -> None:
    assert await runtime_stats() == ([], False)
    set_governor(LocalLimiter(), 1)
    stats, available = await runtime_stats()
    assert available is True
    assert stats == []
