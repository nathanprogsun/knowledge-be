"""RerankService — resolves and drives rerank model instances.

A per-request service that loads a tenant-visible model row, maps it to
``RerankerConfig`` via ``config_from_model``, and constructs the reranker
through ``new_reranker``. The base URL is SSRF-validated during
construction, so an unsafe endpoint fails fast at resolution time rather
than on the first call.

Constructed per request (mirroring ``ModelService``); the repository owns
the per-request session. ``http_client`` may be an APP-scope pooled
client injected by the caller; when omitted the reranker opens its own
client.
"""

from __future__ import annotations

import httpx

from src.ai.rerank import Reranker, config_from_model, new_reranker
from src.ai.rerank.remote_api import RankResult
from src.common.exception import NotFoundError, ValidationError
from src.common.json import JsonValue
from src.db.dao.model_repository import ModelRepository


def _as_str(value: JsonValue | None) -> str:
    return value if isinstance(value, str) else ""


class RerankService:
    """Stateless rerank service, constructed per request."""

    def __init__(
        self,
        *,
        models_repo: ModelRepository,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._models_repo = models_repo
        self._http_client = http_client

    # ── Resolution ───────────────────────────────────────────────────

    async def get_rerank_model(self, *, tenant_id: int, model_id: str) -> Reranker:
        """Load ``model_id`` and construct its reranker instance.

        The row must be visible to ``tenant_id`` (``model.not_found``
        otherwise). The resolved base URL is SSRF-validated during
        construction, so blocked endpoints surface here.
        """
        if tenant_id <= 0:
            raise ValidationError(
                code="model.invalid_tenant_id",
                message="Tenant ID must be positive",
            )
        row = await self._models_repo.find_by_tenant_and_id_or_fail(
            tenant_id=tenant_id,
            id=model_id,
        )
        parameters = row.parameters or {}
        app_id = _as_str(parameters.get("app_id"))
        app_secret = _as_str(parameters.get("app_secret"))
        config = config_from_model(row, app_id=app_id, app_secret=app_secret)
        if config is None:
            raise NotFoundError(
                code="model.not_found",
                message=f"Model {model_id} not found",
            )
        return await new_reranker(config, client=self._http_client)

    # ── Drive ────────────────────────────────────────────────────────

    async def rerank(
        self,
        *,
        tenant_id: int,
        model_id: str,
        query: str,
        documents: list[str],
    ) -> list[RankResult]:
        """Rerank ``documents`` against ``query`` with the configured model."""
        reranker = await self.get_rerank_model(tenant_id=tenant_id, model_id=model_id)
        return await reranker.rerank(query, documents)


__all__ = ["RerankService"]
