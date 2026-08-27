"""VectorStore persistence — raw SQL only, no ORM.

Maps the methods declared in the upstream
``internal/types/interfaces/vectorstore.go::VectorStoreRepository``
interface. Each method scopes by ``tenant_id`` so a caller can never
read or mutate another workspace's rows.

Reads filter ``deleted_at IS NULL`` (via the base ``GenericRepository``
helpers) so a soft-deleted row behaves as if it no longer exists.

``ExistsByEngineTypeEndpointIndex`` is the duplicate guard used by
``VectorStoreService.create_store`` before a write. The Go counterpart
iterates the tenant's rows in Python because the JSONB-field extraction
syntax differs between PostgreSQL and SQLite; we mirror that path so
the row count kept by an operator (a few per tenant) is the cost we
pay, not a fan-out join.
"""

from __future__ import annotations

from sqlalchemy import text

from src.common.json import JsonObject
from src.db.dao.generic_repository import GenericRepository
from src.db.models.infra.vector_store import VectorStore


class VectorStoreRepository(GenericRepository[VectorStore]):
    """`vector_stores`-table SQL — tenant-scoped CRUD + duplicate check."""

    model_class = VectorStore

    # ── Reads ───────────────────────────────────────────────────────

    async def get_by_id(
        self,
        tenant_id: int,
        store_id: str,
    ) -> VectorStore | None:
        """Return one live vector store by primary key + tenant scope."""
        return await self.find_unique_by_column_values(
            {"id": store_id, "tenant_id": tenant_id},
        )

    async def list_for_tenant(self, tenant_id: int) -> list[VectorStore]:
        """Return every live vector store of the tenant, newest first.

        Mirrors ``VectorStoreRepository.List`` on the Go side (newest
        first ordering).
        """
        stmt = text(
            "select * from vector_stores "
            "where tenant_id = :tenant_id and deleted_at is null "
            "order by created_at desc"
        ).bindparams(tenant_id=tenant_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def exists_by_engine_type_endpoint_index(
        self,
        *,
        tenant_id: int,
        engine_type: str,
        endpoint: str,
        index_name: str,
    ) -> bool:
        """Return True if a live row matches the same engine+endpoint+index.

        The endpoint / index-name comparison is performed at the
        application layer because JSONB-field extraction syntax differs
        between PostgreSQL and SQLite, and the row count per tenant is
        small (a handful of stores). Mirrors the Go ``ExistsByEndpointAndIndex``.
        """
        stmt = text(
            "select * from vector_stores "
            "where tenant_id = :tenant_id and engine_type = :engine_type "
            "and deleted_at is null"
        ).bindparams(tenant_id=tenant_id, engine_type=engine_type)
        result = await self._session.execute(stmt)
        for row in result.mappings().all():
            store = self._hydrate(row)
            if (
                _endpoint_of(store.connection_config) == endpoint
                and _index_name_of(store.index_config, engine_type) == index_name
            ):
                return True
        return False


# ── Helpers ──────────────────────────────────────────────────────────


def _endpoint_of(connection_config: JsonObject | None) -> str:
    """Return the normalised endpoint string for a stored connection_config.

    Mirrors ``(types.ConnectionConfig).GetEndpoint`` on the Go side. The
    rules are intentionally tolerant: a stored row may have been
    written by an older version of the form, so we read whichever
    field looks populated and fall back to the empty string on
    exclusion. Keep the rules in sync with the Go helper — drift here
    silently weakens the duplicate guard.
    """
    if not connection_config:
        return ""
    addr = connection_config.get("addr")
    if isinstance(addr, str) and addr:
        database = connection_config.get("database")
        if isinstance(database, str) and database:
            return f"{addr}/{database}"
        return addr
    host = connection_config.get("host")
    if isinstance(host, str) and host:
        port = connection_config.get("port")
        port_int = int(port) if isinstance(port, int) else 6334
        return f"{host}:{port_int}"
    if connection_config.get("use_default_connection") is True:
        return "__default_postgres__"
    return ""


def _index_name_of(index_config: JsonObject | None, engine_type: str) -> str:
    """Return the effective index/collection name for a stored index_config.

    Mirrors ``(types.IndexConfig).GetIndexNameOrDefault`` on the Go side
    across the seven engine types we expose.
    """
    if not index_config:
        index_config = {}
    if engine_type == "elasticsearch":
        name = index_config.get("index_name")
        if isinstance(name, str) and name:
            return name
        return "xwrag_default"
    if engine_type == "qdrant":
        prefix = index_config.get("collection_prefix")
        if isinstance(prefix, str) and prefix:
            return prefix
        return "kb_embeddings"
    if engine_type == "milvus":
        name = index_config.get("collection_name")
        if isinstance(name, str) and name:
            return name
        return "kb_embeddings"
    if engine_type == "tencent_vectordb":
        name = index_config.get("collection_name")
        if isinstance(name, str) and name:
            return name
        return "kb_embeddings"
    if engine_type == "weaviate":
        prefix = index_config.get("collection_prefix")
        if isinstance(prefix, str) and prefix:
            return prefix
        return "kb_embeddings"
    if engine_type == "doris":
        prefix = index_config.get("collection_prefix")
        if isinstance(prefix, str) and prefix:
            return prefix
        name = index_config.get("collection_name")
        if isinstance(name, str) and name:
            return name
        return "kb_embeddings"
    if engine_type == "opensearch":
        name = index_config.get("index_name")
        if isinstance(name, str) and name:
            return name
        return "kb"
    name = index_config.get("index_name")
    return name if isinstance(name, str) else ""


__all__ = [
    "VectorStoreRepository",
    "_endpoint_of",
    "_index_name_of",
]
