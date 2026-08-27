"""Per-model concurrency governor for vision-language calls.

Maps the upstream concurrency wrapper: the outermost decorator holds a
per-model slot only around the real provider round-trip. A positive
``limit`` shares a process-local counting semaphore across every VLM
instance of the same model; zero or negative limits are a cheap
passthrough (no governor installed).
"""

from __future__ import annotations

import asyncio
from typing import Protocol


# Structural subset of the public ``VLM`` protocol, kept local so this
# module stays import-light (the two protocol shapes are identical, so
# ``ConcurrencyVLM`` is assignable to ``VLM``).
class _InnerVLM(Protocol):
    async def predict(self, img_bytes: list[bytes], prompt: str) -> str: ...

    def get_model_name(self) -> str: ...

    def get_model_id(self) -> str: ...


class _ModelGate:
    """Process-local per-model counting semaphores shared across instances."""

    def __init__(self) -> None:
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._lock = asyncio.Lock()

    async def semaphore_for(self, key: str, limit: int) -> asyncio.Semaphore:
        async with self._lock:
            semaphore = self._semaphores.get(key)
            if semaphore is None:
                semaphore = asyncio.Semaphore(limit)
                self._semaphores[key] = semaphore
            return semaphore


_GATE = _ModelGate()


class ConcurrencyVLM:
    """Decorator holding the model's concurrency slot around ``predict``."""

    def __init__(
        self,
        *,
        inner: _InnerVLM,
        limit: int,
        gate: _ModelGate | None = None,
    ) -> None:
        self._inner = inner
        self._limit = limit
        self._gate = gate if gate is not None else _GATE

    def get_model_name(self) -> str:
        return self._inner.get_model_name()

    def get_model_id(self) -> str:
        return self._inner.get_model_id()

    async def predict(self, img_bytes: list[bytes], prompt: str) -> str:
        if self._limit <= 0:
            return await self._inner.predict(img_bytes, prompt)
        semaphore = await self._gate.semaphore_for(self._inner.get_model_id(), self._limit)
        async with semaphore:
            return await self._inner.predict(img_bytes, prompt)


def wrap_vlm_concurrency(vlm: _InnerVLM, limit: int) -> _InnerVLM:
    """Install the concurrency governor as the outermost VLM decorator.

    Always applied; a cheap passthrough when ``limit`` is not positive.
    """
    return ConcurrencyVLM(inner=vlm, limit=limit)


__all__ = [
    "ConcurrencyVLM",
    "wrap_vlm_concurrency",
]
