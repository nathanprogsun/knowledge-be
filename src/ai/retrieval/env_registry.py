"""Environment-driven retrieval engine registry (upstream ``container.go`` env path).

``init_retrieve_engine_registry`` registers one engine service per
``RETRIEVE_DRIVER`` entry, then loads DB-managed vector stores into the
registry. Mirror the upstream behavior: per-driver failures are logged and
skipped, never fatal, and the registry still serves every engine that did
register.

Concrete engine repositories land in followup engine PRs; until then the
per-driver repository builders are placeholders that raise
``NotImplementedError``, so an unset or unimplemented driver degrades to a
logged skip rather than a startup failure.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeAlias

from src.ai.embedding import TaskContext
from src.ai.retrieval.base import (
    AppConfig,
    AuditSink,
    Context,
    Database,
    EngineFactory,
    RetrieveEngineRepository,
    StoreRegistry,
    VectorStoreRepositoryLike,
)
from src.ai.retrieval.factory import create_engine_service_from_store
from src.ai.retrieval.kv_hybrid import new_kv_hybrid_retrieve_engine
from src.ai.retrieval.registry import RetrieveEngineRegistry, new_retrieve_engine_registry
from src.ai.retrieval.types import (
    ConnectionConfig,
    RetrieverEngineType,
    VectorStoreLike,
)
from src.app_logging import logger
from src.common.exception import ConflictError

#: ``RETRIEVE_DRIVER`` values registered by the env path, mapped to the
#: engine type each registers. The upstream env path also handles
#: ``tencent_vectordb``; it is delivered with its engine PR.
_DRIVER_ENGINE_TYPES: dict[str, RetrieverEngineType] = {
    "postgres": RetrieverEngineType.POSTGRES,
    "sqlite": RetrieverEngineType.SQLITE,
    "elasticsearch_v8": RetrieverEngineType.ELASTICSEARCH,
    "elasticsearch_v7": RetrieverEngineType.ELASTICSEARCH,
    "opensearch": RetrieverEngineType.OPENSEARCH,
    "qdrant": RetrieverEngineType.QDRANT,
    "weaviate": RetrieverEngineType.WEAVIATE,
    "milvus": RetrieverEngineType.MILVUS,
    "doris": RetrieverEngineType.DORIS,
}


#: Loads every DB-managed vector store for registry hydration.
StoreLoaderFn: TypeAlias = Callable[[], Awaitable[Sequence[VectorStoreLike]]]

#: Background context used for startup-time builds (upstream ``context.Background()``).
_BACKGROUND_CONTEXT: Context = TaskContext(is_background_task=True)


def parse_retrieve_driver(raw: str) -> list[str]:
    """Split the ``RETRIEVE_DRIVER`` value into trimmed driver names.

    Empty segments are skipped. This trims segments (upstream
    ``BuildEnvVectorStores`` also trims) so ``"postgres, sqlite"`` registers
    both engines.
    """
    if not raw:
        return []
    return [driver.strip() for driver in raw.split(",") if driver.strip()]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_tls_enabled(name: str) -> bool:
    raw = os.getenv(name, "").strip().lower()
    return raw != "" and raw != "false" and raw != "0"


def _env_weaviate_api_key() -> str:
    if os.getenv("WEAVIATE_AUTH_ENABLED", "").strip().lower() == "true":
        return os.getenv("WEAVIATE_API_KEY", "").strip()
    return ""


def _opensearch_connection_config() -> ConnectionConfig:
    return ConnectionConfig(
        addr=os.getenv("OPENSEARCH_ADDR", ""),
        username=os.getenv("OPENSEARCH_USERNAME", ""),
        password=os.getenv("OPENSEARCH_PASSWORD", ""),
        insecure_skip_verify=(
            os.getenv("OPENSEARCH_INSECURE_SKIP_VERIFY", "").strip().lower() == "true"
        ),
    )


# ── Per-driver repository builders (placeholders) ───────────────────
#
# Each mirrors the env values the concrete engine PR will consume. The
# client/repository construction is deferred to the engine PRs.


async def _new_postgres_repository(_db: Database) -> RetrieveEngineRepository:
    raise NotImplementedError("postgres engine repository lands with the pgvector engine")


async def _new_sqlite_repository(_db: Database) -> RetrieveEngineRepository:
    raise NotImplementedError("sqlite engine repository lands with the sqlite-vec engine")


async def _new_elasticsearch_v8_repository(_cfg: AppConfig) -> RetrieveEngineRepository:
    raise NotImplementedError("elasticsearch v8 repository lands with the elasticsearch engine")


async def _new_elasticsearch_v7_repository(_cfg: AppConfig) -> RetrieveEngineRepository:
    raise NotImplementedError("elasticsearch v7 repository lands with the elasticsearch engine")


async def _new_opensearch_repository(
    _cc: ConnectionConfig, _audit_sink: AuditSink | None
) -> RetrieveEngineRepository:
    raise NotImplementedError("opensearch repository lands with the opensearch engine")


async def _new_qdrant_repository(
    _host: str, _port: int, _api_key: str, _use_tls: bool
) -> RetrieveEngineRepository:
    raise NotImplementedError("qdrant repository lands with the qdrant engine")


async def _new_weaviate_repository(
    _host: str, _grpc_address: str, _scheme: str, _api_key: str
) -> RetrieveEngineRepository:
    raise NotImplementedError("weaviate repository lands with the weaviate engine")


async def _new_milvus_repository(
    _address: str, _username: str, _password: str, _db_name: str
) -> RetrieveEngineRepository:
    raise NotImplementedError("milvus repository lands with the milvus engine")


async def _new_doris_repository(
    _addr: str, _http_port: int, _database: str, _username: str, _password: str
) -> RetrieveEngineRepository:
    raise NotImplementedError("doris repository lands with the doris engine")


async def _build_driver_repository(
    driver: str,
    db: Database,
    cfg: AppConfig,
    audit_sink: AuditSink | None,
) -> RetrieveEngineRepository:
    if driver == "postgres":
        return await _new_postgres_repository(db)
    if driver == "sqlite":
        return await _new_sqlite_repository(db)
    if driver == "elasticsearch_v8":
        return await _new_elasticsearch_v8_repository(cfg)
    if driver == "elasticsearch_v7":
        return await _new_elasticsearch_v7_repository(cfg)
    if driver == "opensearch":
        return await _new_opensearch_repository(_opensearch_connection_config(), audit_sink)
    if driver == "qdrant":
        return await _new_qdrant_repository(
            os.getenv("QDRANT_HOST", "localhost"),
            _env_int("QDRANT_PORT", 6334),
            os.getenv("QDRANT_API_KEY", ""),
            _env_tls_enabled("QDRANT_USE_TLS"),
        )
    if driver == "weaviate":
        return await _new_weaviate_repository(
            os.getenv("WEAVIATE_HOST", "weaviate:8080"),
            os.getenv("WEAVIATE_GRPC_ADDRESS", "weaviate:50051"),
            os.getenv("WEAVIATE_SCHEME", "http"),
            _env_weaviate_api_key(),
        )
    if driver == "milvus":
        return await _new_milvus_repository(
            os.getenv("MILVUS_ADDRESS", "localhost:19530"),
            os.getenv("MILVUS_USERNAME", ""),
            os.getenv("MILVUS_PASSWORD", ""),
            os.getenv("MILVUS_DB_NAME", ""),
        )
    if driver == "doris":
        return await _new_doris_repository(
            os.getenv("DORIS_ADDR", "doris-fe:9030"),
            _env_int("DORIS_HTTP_PORT", 8030),
            os.getenv("DORIS_DATABASE", "weknora"),
            os.getenv("DORIS_USERNAME", "root"),
            os.getenv("DORIS_PASSWORD", ""),
        )
    raise ValueError(f"unsupported retrieve driver: {driver}")


# ── Registration loop ───────────────────────────────────────────────


async def _register_driver(
    registry: RetrieveEngineRegistry,
    driver: str,
    db: Database,
    cfg: AppConfig,
    audit_sink: AuditSink | None,
) -> None:
    engine_type = _DRIVER_ENGINE_TYPES.get(driver)
    if engine_type is None:
        return
    try:
        repo = await _build_driver_repository(driver, db, cfg, audit_sink)
    except NotImplementedError as exc:
        logger.error(
            "Create {} retrieve engine failed (engine not implemented yet): {}",
            driver,
            exc,
        )
        return
    except Exception as exc:
        logger.error("Create {} retrieve engine failed: {}", driver, exc)
        return
    try:
        registry.register(new_kv_hybrid_retrieve_engine(repo, engine_type))
    except ConflictError as exc:
        logger.error("Register {} retrieve engine failed: {}", driver, exc)
    else:
        logger.info("Register {} retrieve engine success", driver)


# ── DB store hydration ──────────────────────────────────────────────


async def load_db_stores_into_registry(
    store_registry: StoreRegistry,
    stores: Sequence[VectorStoreLike],
    db: Database,
    cfg: AppConfig,
    audit_sink: AuditSink | None = None,
) -> None:
    """Register one engine service per DB-managed store (upstream ``loadDBStoresIntoRegistry``).

    Failures are logged and skipped; a broken store never blocks the others.
    """
    if not stores:
        return
    for store in stores:
        try:
            svc = await create_engine_service_from_store(
                _BACKGROUND_CONTEXT, store, db, cfg, audit_sink
            )
        except Exception as exc:
            logger.error(
                "Failed to create engine for store {} ({}): {}",
                store.id,
                store.name,
                exc,
            )
            continue
        store_registry.register_with_store_id(store.id, svc)
        logger.info(
            "Registered DB vector store: id={}, name={}, engine={}",
            store.id,
            store.name,
            store.engine_type,
        )


async def init_retrieve_engine_registry(
    db: Database,
    cfg: AppConfig,
    audit_svc: AuditSink | None = None,
    store_repo: VectorStoreRepositoryLike | None = None,
    engine_factory: EngineFactory | None = None,
    store_loader: StoreLoaderFn | None = None,
) -> RetrieveEngineRegistry:
    """Build the retrieval engine registry (upstream ``initRetrieveEngineRegistry``).

    Registers every engine named by ``RETRIEVE_DRIVER``, then hydrates
    DB-managed stores when ``store_loader`` is provided. ``store_repo`` and
    ``engine_factory`` let the registry rebuild a store engine on demand.
    """
    registry = new_retrieve_engine_registry(store_repo, engine_factory)
    drivers = parse_retrieve_driver(os.getenv("RETRIEVE_DRIVER", ""))
    for driver in drivers:
        await _register_driver(registry, driver, db, cfg, audit_svc)
    if store_loader is not None:
        stores = await store_loader()
        await load_db_stores_into_registry(registry, stores, db, cfg, audit_svc)
    return registry


__all__ = [
    "StoreLoaderFn",
    "init_retrieve_engine_registry",
    "load_db_stores_into_registry",
    "parse_retrieve_driver",
]
