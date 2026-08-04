"""Internal DTOs and registry metadata for the vector-store domain.

Three surfaces live here:

- ``VectorStoreInfo`` — service-side projection of a ``vector_stores``
  row, mirroring the Go ``VectorStore``. Boundary translation (e.g.
  masking the ``password`` / ``api_key`` columns) lives at the web layer;
  this DTO is the internal carrier.
- ``VECTOR_STORE_TYPES`` — registry metadata for every supported engine
  type, mirroring
  ``internal/types/vectorstore.go::GetVectorStoreTypes``. The web layer
  returns this verbatim for ``GET /vector-stores/types``.
- ``SUPPORTED_ENGINE_TYPES`` — frozenset of valid engine ids, used by
  the service to validate the ``engine_type`` field on create and
  during the raw connection probe.

Note on the ``default`` field
-----------------------------

The frozen contract types ``VectorStoreConnectionField.default`` as
``JsonObject | None`` (a dict), but the Go wire format ships scalar
defaults (``"http://localhost:9200"``, ``4``, ``false``). The contract
is the source of truth for the column type; we wrap every scalar in a
single-entry ``{"value": ...}`` object so the contract validates.
A drift note is queued for the next contract reconciliation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self, cast

from pydantic import BaseModel, ConfigDict

from src.common.json import JsonObject, JsonValue
from src.core.contracts.infra import (
    VectorStoreConnectionField,
    VectorStoreTypeInfo,
)
from src.db.models.infra.vector_store import VectorStore


def _scalar_default(value: str | int | bool) -> JsonObject:
    """Wrap a scalar default as a single-entry JsonObject.

    See the module docstring for the contract-vs-wire drift context.
    """
    return cast("JsonObject", {"value": cast("JsonValue", value)})


# ── Vector-store type metadata (registry) ────────────────────────────
#
# Mirrors ``internal/types/vectorstore.go::GetVectorStoreTypes`` with the
# same seven engine types (Postgres and SQLite are excluded because they
# only support the app's default DB connection). Each block declares the
# connection fields + optional index fields exposed to the UI; the wire
# contract is the frozen ``VectorStoreTypeInfo`` in
# ``src/core/contracts/infra.py`` so a type drift against Go surfaces as
# a check_contract_invariants failure rather than a runtime crash.

_VECTOR_STORE_TYPES: tuple[VectorStoreTypeInfo, ...] = (
    VectorStoreTypeInfo(
        type="elasticsearch",
        display_name="Elasticsearch",
        connection_fields=[
            VectorStoreConnectionField(
                name="addr",
                type="string",
                required=True,
                default=_scalar_default("http://localhost:9200"),
                description="URL",
            ),
            VectorStoreConnectionField(
                name="username",
                type="string",
                required=False,
                default=_scalar_default("elastic"),
                description="Username",
            ),
            VectorStoreConnectionField(
                name="password",
                type="string",
                required=False,
                sensitive=True,
                description="Password",
            ),
        ],
        index_fields=[
            VectorStoreConnectionField(
                name="index_name",
                type="string",
                required=False,
                default=_scalar_default("weknora"),
                description="Index Name",
            ),
            VectorStoreConnectionField(
                name="number_of_shards",
                type="number",
                required=False,
                default=_scalar_default(4),
                description="Shards",
            ),
            VectorStoreConnectionField(
                name="number_of_replicas",
                type="number",
                required=False,
                default=_scalar_default(1),
                description="Replicas",
            ),
        ],
    ),
    VectorStoreTypeInfo(
        type="qdrant",
        display_name="Qdrant",
        connection_fields=[
            VectorStoreConnectionField(
                name="host",
                type="string",
                required=True,
                default=_scalar_default("localhost"),
                description="Host",
            ),
            VectorStoreConnectionField(
                name="port",
                type="number",
                required=False,
                default=_scalar_default(6334),
                description="Port",
            ),
            VectorStoreConnectionField(
                name="api_key",
                type="string",
                required=False,
                sensitive=True,
                description="API Key",
            ),
            VectorStoreConnectionField(
                name="use_tls",
                type="boolean",
                required=False,
                default=_scalar_default(False),
                description="Use TLS",
            ),
        ],
        index_fields=[
            VectorStoreConnectionField(
                name="collection_prefix",
                type="string",
                required=False,
                default=_scalar_default("weknora_embeddings"),
                description="Collection Prefix",
            ),
            VectorStoreConnectionField(
                name="shard_number",
                type="number",
                required=False,
                default=_scalar_default(1),
                description="Shard Number",
            ),
            VectorStoreConnectionField(
                name="replication_factor",
                type="number",
                required=False,
                default=_scalar_default(1),
                description="Replication Factor",
            ),
        ],
    ),
    VectorStoreTypeInfo(
        type="milvus",
        display_name="Milvus",
        connection_fields=[
            VectorStoreConnectionField(
                name="addr",
                type="string",
                required=True,
                default=_scalar_default("localhost:19530"),
                description="Address",
            ),
            VectorStoreConnectionField(
                name="database",
                type="string",
                required=False,
                description="Database Name",
            ),
            VectorStoreConnectionField(
                name="username",
                type="string",
                required=False,
                default=_scalar_default("root"),
                description="Username",
            ),
            VectorStoreConnectionField(
                name="password",
                type="string",
                required=False,
                sensitive=True,
                description="Password",
            ),
        ],
        index_fields=[
            VectorStoreConnectionField(
                name="collection_name",
                type="string",
                required=False,
                default=_scalar_default("weknora_embeddings"),
                description="Collection Name",
            ),
            VectorStoreConnectionField(
                name="shards_num",
                type="number",
                required=False,
                default=_scalar_default(1),
                description="Shards (write parallelism)",
            ),
            VectorStoreConnectionField(
                name="replica_number",
                type="number",
                required=False,
                default=_scalar_default(1),
                description="In-memory Replicas (read HA)",
            ),
        ],
    ),
    VectorStoreTypeInfo(
        type="tencent_vectordb",
        display_name="Tencent VectorDB",
        connection_fields=[
            VectorStoreConnectionField(
                name="addr",
                type="string",
                required=True,
                default=_scalar_default("http://localhost:8080"),
                description="Address",
            ),
            VectorStoreConnectionField(
                name="username",
                type="string",
                required=True,
                description="Username",
            ),
            VectorStoreConnectionField(
                name="api_key",
                type="string",
                required=True,
                sensitive=True,
                description="API Key",
            ),
            VectorStoreConnectionField(
                name="database",
                type="string",
                required=False,
                default=_scalar_default("weknora"),
                description="Database",
            ),
        ],
        index_fields=[
            VectorStoreConnectionField(
                name="collection_name",
                type="string",
                required=False,
                default=_scalar_default("weknora_embeddings"),
                description="Collection Name",
            ),
            VectorStoreConnectionField(
                name="shards_num",
                type="number",
                required=False,
                default=_scalar_default(1),
                description="Shards",
            ),
            VectorStoreConnectionField(
                name="replica_number",
                type="number",
                required=False,
                default=_scalar_default(1),
                description="Replicas",
            ),
        ],
    ),
    VectorStoreTypeInfo(
        type="weaviate",
        display_name="Weaviate",
        connection_fields=[
            VectorStoreConnectionField(
                name="host",
                type="string",
                required=True,
                default=_scalar_default("weaviate:8080"),
                description="Host",
            ),
            VectorStoreConnectionField(
                name="grpc_address",
                type="string",
                required=False,
                default=_scalar_default("weaviate:50051"),
                description="gRPC Address",
            ),
            VectorStoreConnectionField(
                name="scheme",
                type="string",
                required=False,
                default=_scalar_default("http"),
                description="Scheme",
            ),
            VectorStoreConnectionField(
                name="api_key",
                type="string",
                required=False,
                sensitive=True,
                description="API Key",
            ),
        ],
        index_fields=[
            VectorStoreConnectionField(
                name="collection_prefix",
                type="string",
                required=False,
                default=_scalar_default("Weknora_embeddings"),
                description="Collection Prefix",
            ),
            VectorStoreConnectionField(
                name="desired_shard_count",
                type="number",
                required=False,
                default=_scalar_default(1),
                description="Shard Count",
            ),
            VectorStoreConnectionField(
                name="replication_factor",
                type="number",
                required=False,
                default=_scalar_default(1),
                description="Replication Factor",
            ),
        ],
    ),
    VectorStoreTypeInfo(
        type="doris",
        display_name="Apache Doris",
        connection_fields=[
            VectorStoreConnectionField(
                name="addr",
                type="string",
                required=True,
                default=_scalar_default("doris-fe:9030"),
                description="FE MySQL Address (host:port)",
            ),
            VectorStoreConnectionField(
                name="http_port",
                type="number",
                required=False,
                default=_scalar_default(8030),
                description="FE HTTP Port (Stream Load)",
            ),
            VectorStoreConnectionField(
                name="database",
                type="string",
                required=True,
                default=_scalar_default("weknora"),
                description="Database",
            ),
            VectorStoreConnectionField(
                name="username",
                type="string",
                required=False,
                default=_scalar_default("root"),
                description="Username",
            ),
            VectorStoreConnectionField(
                name="password",
                type="string",
                required=False,
                sensitive=True,
                description="Password",
            ),
        ],
        index_fields=[
            VectorStoreConnectionField(
                name="collection_prefix",
                type="string",
                required=False,
                default=_scalar_default("weknora_embeddings"),
                description="Table Prefix",
            ),
            VectorStoreConnectionField(
                name="buckets_num",
                type="number",
                required=False,
                default=_scalar_default(10),
                description="Buckets per table",
            ),
            VectorStoreConnectionField(
                name="replication_num",
                type="number",
                required=False,
                default=_scalar_default(1),
                description="Replication Num",
            ),
        ],
    ),
    VectorStoreTypeInfo(
        type="opensearch",
        display_name="OpenSearch",
        connection_fields=[
            VectorStoreConnectionField(
                name="addr",
                type="string",
                required=True,
                default=_scalar_default("https://localhost:9200"),
                description="URL",
            ),
            VectorStoreConnectionField(
                name="username",
                type="string",
                required=False,
                default=_scalar_default("admin"),
                description="Username",
            ),
            VectorStoreConnectionField(
                name="password",
                type="string",
                required=False,
                sensitive=True,
                description="Password",
            ),
            VectorStoreConnectionField(
                name="insecure_skip_verify",
                type="boolean",
                required=False,
                default=_scalar_default(False),
                description=(
                    "Skip TLS certificate verification. For self-signed "
                    "dev clusters only — never enable in production."
                ),
            ),
        ],
        index_fields=[
            VectorStoreConnectionField(
                name="index_name",
                type="string",
                required=False,
                default=_scalar_default("weknora"),
                description="Index Name",
            ),
            VectorStoreConnectionField(
                name="number_of_shards",
                type="number",
                required=False,
                default=_scalar_default(4),
                description="Shards",
            ),
            VectorStoreConnectionField(
                name="number_of_replicas",
                type="number",
                required=False,
                default=_scalar_default(1),
                description="Replicas",
            ),
        ],
    ),
)

SUPPORTED_ENGINE_TYPES: frozenset[str] = frozenset(info.type for info in _VECTOR_STORE_TYPES)


def vector_store_types() -> tuple[VectorStoreTypeInfo, ...]:
    """Return the registry metadata for every supported engine type."""
    return _VECTOR_STORE_TYPES


# ── Wire-side projection ─────────────────────────────────────────────


# Placeholder used by the service boundary to mask sensitive fields
# before they cross the wire. Keeps the same string as the Go
# ``internal/types.RedactedSecretPlaceholder`` so a downstream UI that
# already special-cases the placeholder stays compatible.
REDACTED_SECRET_PLACEHOLDER: str = "***"

# Sensitive keys whose stored value must be replaced with the
# placeholder above before the service layer projects the row.
_SENSITIVE_KEYS: frozenset[str] = frozenset({"password", "api_key"})


class VectorStoreInfo(BaseModel):
    """Service-side projection of a `vector_stores` row.

    Mirrors ``internal/types/vectorstore.go::VectorStore``. The wire
    contract (``VectorStore``) is identical at the field level; the
    service layer masks sensitive fields so the wire never sees
    plaintext credentials on the list response. The ``source`` and
    ``readonly`` columns are persisted on the row but mirrored from the
    contract for the wire layer's purposes.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: int
    name: str
    engine_type: str
    connection_config: JsonObject | None = None
    index_config: JsonObject | None = None
    source: str = "user"
    readonly: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None
    deleted_at: datetime | None = None

    @classmethod
    def map_from_db(cls, db: VectorStore) -> Self:
        """Build a projection from the raw storage row."""
        return cls(
            id=db.id,
            tenant_id=db.tenant_id,
            name=db.name,
            engine_type=db.engine_type,
            connection_config=db.connection_config,
            index_config=db.index_config,
            source=db.source,
            readonly=db.readonly,
            created_at=db.created_at,
            updated_at=db.updated_at,
            deleted_at=db.deleted_at,
        )


def mask_sensitive_fields(
    connection_config: JsonObject | None,
) -> JsonObject | None:
    """Replace ``password``/``api_key`` values with the redacted placeholder.

    Empty values stay empty so the frontend can distinguish "set (hidden)"
    from "not set" without an extra flag — mirrors the Go
    ``(types.ConnectionConfig).MaskSensitiveFields`` helper.
    """
    if connection_config is None:
        return None
    masked: dict[str, JsonValue] = {}
    for key, value in connection_config.items():
        if key in _SENSITIVE_KEYS and isinstance(value, str) and value:
            masked[key] = REDACTED_SECRET_PLACEHOLDER
        else:
            masked[key] = value
    return masked


__all__ = [
    "REDACTED_SECRET_PLACEHOLDER",
    "SUPPORTED_ENGINE_TYPES",
    "VectorStoreInfo",
    "mask_sensitive_fields",
    "vector_store_types",
]
