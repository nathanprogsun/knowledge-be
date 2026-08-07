"""Local Ollama embedding provider (ollama.go).

Implements text vectorization over a local Ollama instance via the shared
``OllamaService``. The model is ensured available (pulled on demand) before
every batch, and the embedding vectors come from the ``/api/embed``
endpoint. The module references the base protocols through a module-level
import of ``src.ai.embedding.base`` (rather than a from-import) to break
the base↔provider import cycle.
"""

from __future__ import annotations

import src.ai.embedding.base as base
from src.ai.utils.ollama_service import OllamaEmbedRequest, OllamaService
from src.common.exception import AIProviderError
from src.common.json import JsonValue

_DEFAULT_MODEL_NAME = "nomic-embed-text"
_DEFAULT_TRUNCATE_PROMPT_TOKENS = 511


class OllamaEmbedder:
    """Ollama text vectorizer (upstream ``OllamaEmbedder``)."""

    def __init__(
        self,
        *,
        model_name: str,
        truncate_prompt_tokens: int,
        ollama_service: OllamaService,
        dimensions: int,
        model_id: str,
        pooler: base.EmbedderPooler | None,
    ) -> None:
        self._model_name = model_name
        self._truncate_prompt_tokens = truncate_prompt_tokens
        self._ollama_service = ollama_service
        self._dimensions = dimensions
        self._model_id = model_id
        self._pooler = pooler
        self._supports_dimension_override = False

    def set_supports_dimension_override(self, supported: bool) -> None:
        self._supports_dimension_override = supported

    async def embed(self, ctx: base.Context, text: str) -> list[float]:
        embedding = await self.batch_embed(ctx, [text])
        if not embedding:
            raise AIProviderError(
                "failed to embed text: no embedding returned",
                code="embedding.no_embedding_returned",
            )
        return embedding[0]

    async def batch_embed(self, ctx: base.Context, texts: list[str]) -> list[list[float]]:
        await self._ollama_service.ensure_model_available(self._model_name)
        options: dict[str, JsonValue] = {}
        truncate: bool | None = None
        if self._truncate_prompt_tokens > 0:
            options["num_ctx"] = self._truncate_prompt_tokens
            truncate = True
        request = OllamaEmbedRequest(
            model=self._model_name,
            input=list(texts),
            options=options,
            truncate=truncate,
            dimensions=(
                self._dimensions
                if self._supports_dimension_override and self._dimensions > 0
                else None
            ),
        )
        return await self._ollama_service.embeddings(request)

    async def batch_embed_with_pool(
        self,
        ctx: base.Context,
        model: base.Embedder,
        texts: list[str],
    ) -> list[list[float]]:
        if self._pooler is None:
            raise AIProviderError(
                "embedder pooler is not configured",
                code="embedding.pooler_missing",
            )
        return await self._pooler.batch_embed_with_pool(ctx, model, texts)

    def get_model_name(self) -> str:
        return self._model_name

    def get_dimensions(self) -> int:
        return self._dimensions

    def get_model_id(self) -> str:
        return self._model_id


async def new_ollama_embedder(
    *,
    base_url: str,
    model_name: str,
    truncate_prompt_tokens: int,
    dimensions: int,
    model_id: str,
    pooler: base.EmbedderPooler | None,
    ollama_service: OllamaService | None,
) -> OllamaEmbedder:
    """Build an Ollama embedder (upstream ``NewOllamaEmbedder``).

    ``base_url`` seeds a default service when none is injected; the
    injected service owns its own dial address. The upstream signature is
    synchronous; this stays synchronous because no async validation is
    involved, and the factory awaits it for a uniform call shape.
    """
    if model_name == "":
        model_name = _DEFAULT_MODEL_NAME
    if truncate_prompt_tokens == 0:
        truncate_prompt_tokens = _DEFAULT_TRUNCATE_PROMPT_TOKENS
    if ollama_service is None:
        ollama_service = OllamaService(base_url=base_url or None)
    return OllamaEmbedder(
        model_name=model_name,
        truncate_prompt_tokens=truncate_prompt_tokens,
        ollama_service=ollama_service,
        dimensions=dimensions,
        model_id=model_id,
        pooler=pooler,
    )


__all__ = ["OllamaEmbedder", "new_ollama_embedder"]
