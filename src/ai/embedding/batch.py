"""Batch embedding dispatcher (batch.go).

``BatchEmbedder`` fans a large text list out into sub-batches of
``BATCH_EMBED_SIZE`` (default 5) and dispatches each sub-batch through the
model's ``batch_embed`` under a bounded worker pool (default 5, matching
the upstream pool size). Results are reassembled in the original order; the
first error recorded aborts the whole dispatch. The upstream goroutine pool
is mirrored here with an ``asyncio.Semaphore``.
"""

from __future__ import annotations

import asyncio
import os

from src.ai.embedding.base import Context, Embedder, EmbedderPooler
from src.common.exception import AIProviderError, ApplicationError, ValidationError

_BATCH_EMBED_SIZE_ENV = "BATCH_EMBED_SIZE"
_CONCURRENCY_POOL_SIZE_ENV = "CONCURRENCY_POOL_SIZE"
_DEFAULT_BATCH_SIZE = 5
_DEFAULT_MAX_WORKERS = 5


def _as_batch_error(exc: BaseException) -> ApplicationError:
    """Project the first sub-batch error onto a sanctioned exception.

    Application-layer errors keep their type; anything else (e.g. a
    provider transport failure) is wrapped so the batch dispatch never
    leaks an arbitrary exception type.
    """
    if isinstance(exc, ApplicationError):
        return exc
    return AIProviderError(
        str(exc),
        code="embedding.batch_failed",
    )


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValidationError(
            code="embedding.invalid_env_value",
            message=f"{name} must be an integer",
        ) from exc


def _chunk_ranges(size: int, batch_size: int) -> list[tuple[int, int]]:
    """Return ``(start, end)`` half-open ranges over ``range(size)``."""
    return [(start, min(start + batch_size, size)) for start in range(0, size, batch_size)]


class BatchEmbedder:
    """Concurrent sub-batch dispatcher (upstream ``batchEmbedder``)."""

    def __init__(self, *, batch_size: int, max_workers: int) -> None:
        if batch_size <= 0:
            raise ValidationError(
                code="embedding.invalid_batch_size",
                message="batch size must be positive",
            )
        if max_workers <= 0:
            raise ValidationError(
                code="embedding.invalid_pool_size",
                message="pool size must be positive",
            )
        self._batch_size = batch_size
        self._max_workers = max_workers

    async def batch_embed_with_pool(
        self,
        ctx: Context,
        model: Embedder,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float] | None] = [None] * len(texts)
        semaphore = asyncio.Semaphore(self._max_workers)
        first_error: BaseException | None = None
        error_lock = asyncio.Lock()

        async def process(start: int, end: int) -> None:
            nonlocal first_error
            async with error_lock:
                if first_error is not None:
                    return
            chunk = texts[start:end]
            async with semaphore:
                try:
                    embeddings = await model.batch_embed(ctx, chunk)
                except Exception as exc:
                    async with error_lock:
                        if first_error is None:
                            first_error = exc
                    return
            for offset, embedding in enumerate(embeddings):
                results[start + offset] = embedding

        await asyncio.gather(
            *(process(start, end) for start, end in _chunk_ranges(len(texts), self._batch_size))
        )
        if first_error is not None:
            raise _as_batch_error(first_error)
        return [result if result is not None else [] for result in results]


def new_batch_embedder(
    *,
    batch_size: int | None = None,
    max_workers: int | None = None,
) -> EmbedderPooler:
    """Build the batch dispatcher (upstream ``NewBatchEmbedder``).

    Unset values fall back to the ``BATCH_EMBED_SIZE`` /
    ``CONCURRENCY_POOL_SIZE`` environment variables, then to 5. The
    upstream reads the batch size from the environment per call; here the
    values are resolved once at construction for determinism.
    """
    return BatchEmbedder(
        batch_size=(
            _env_int(_BATCH_EMBED_SIZE_ENV, _DEFAULT_BATCH_SIZE)
            if batch_size is None
            else batch_size
        ),
        max_workers=(
            _env_int(_CONCURRENCY_POOL_SIZE_ENV, _DEFAULT_MAX_WORKERS)
            if max_workers is None
            else max_workers
        ),
    )


__all__ = ["BatchEmbedder", "new_batch_embedder"]
