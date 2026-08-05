from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject, JsonValue


class EmbeddingParameters(BaseModel):
    model_config = ConfigDict(frozen=True)

    dimension: int | None = Field(default=None)
    truncate_prompt_tokens: int | None = Field(default=0)
    supports_dimension_override: bool | None = Field(default=False)


class ModelParameters(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_url: str | None = Field(default=None)
    api_key: str | None = Field(default=None)
    provider: str | None = Field(default=None)
    interface_type: str | None = Field(default=None)
    embedding_parameters: EmbeddingParameters | None = Field(default=None)
    parameter_size: str | None = Field(default=None)
    extra_config: dict[str, str] | None = Field(default=None)
    custom_headers: dict[str, str] | None = Field(default=None)
    supports_vision: bool | None = Field(default=False)
    max_concurrency: int | None = Field(default=None)
    app_id: str | None = Field(default=None)
    app_secret: str | None = Field(default=None)


class Model(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: int
    name: str
    display_name: str | None = Field(default=None)
    type: str
    source: str
    description: str | None = Field(default=None)
    parameters: ModelParameters
    is_default: bool | None = Field(default=False)
    is_builtin: bool | None = Field(default=False)
    status: str | None = Field(default="active")
    created_at: datetime
    updated_at: datetime
    # Per-field "configured?" map mirroring Go's
    # ``dto.ModelResponse.Credentials``; the values never carry the
    # secret itself.
    credentials: dict[str, CredentialFieldMetadata] | None = Field(default=None)


class CreateModelRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    type: str
    source: str
    description: str | None = Field(default=None)
    display_name: str | None = Field(default=None)
    parameters: ModelParameters


class UpdateModelRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str | None = Field(default=None)
    description: str | None = Field(default=None)
    type: str | None = Field(default=None)
    source: str | None = Field(default=None)
    display_name: str | None = Field(default=None)
    parameters: ModelParameters | None = Field(default=None)


class ProviderTypeMeta(BaseModel):
    """Catalog metadata for one model provider.

    Mirrors ``internal/models/provider/provider.go::ProviderInfo`` on
    the Go side; the wire JSON keeps camelCase (``defaultUrls``,
    ``modelTypes``, ``requiresAuth``, ``extraFields``) so a Go UI
    consumer can drop in unchanged.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    value: str
    label: str
    description: str | None = Field(default=None)
    default_urls: dict[str, str] | None = Field(default=None, alias="defaultUrls")
    model_types: list[str] | None = Field(default=None, alias="modelTypes")
    requires_auth: bool | None = Field(default=None, alias="requiresAuth")
    extra_fields: list[JsonObject] | None = Field(default=None, alias="extraFields")


class MCPMcpServiceAuthConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    auth_type: str | None = Field(default=None)
    api_key: str | None = Field(default=None)
    api_key_header: str | None = Field(default=None)
    token: str | None = Field(default=None)
    custom_headers: dict[str, str] | None = Field(default=None)
    scopes: list[str] | None = Field(default=None)
    auth_server_metadata_url: str | None = Field(default=None)


class MCPServiceAdvancedConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    timeout: int | None = Field(default=None)
    retry_count: int | None = Field(default=None)
    retry_delay: int | None = Field(default=None)


class MCPStdioConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    command: str
    args: list[str] | None = Field(default=None)


class MCPService(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: int
    name: str
    description: str | None = Field(default=None)
    enabled: bool | None = Field(default=True)
    transport_type: str
    url: str | None = Field(default=None)
    headers: dict[str, str] | None = Field(default=None)
    auth_config: MCPMcpServiceAuthConfig | None = Field(default=None)
    advanced_config: MCPServiceAdvancedConfig | None = Field(default=None)
    stdio_config: MCPStdioConfig | None = Field(default=None)
    env_vars: dict[str, str] | None = Field(default=None)
    is_builtin: bool | None = Field(default=False)
    created_at: datetime
    updated_at: datetime
    # Per-field "configured?" map mirroring Go's
    # ``dto.MCPServiceResponse.Credentials``.
    credentials: dict[str, CredentialFieldMetadata] | None = Field(default=None)


class CreateMCPServiceRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str | None = Field(default=None)
    transport_type: str
    url: str | None = Field(default=None)
    headers: dict[str, str] | None = Field(default=None)
    auth_config: MCPMcpServiceAuthConfig | None = Field(default=None)
    advanced_config: MCPServiceAdvancedConfig | None = Field(default=None)
    stdio_config: MCPStdioConfig | None = Field(default=None)
    env_vars: dict[str, str] | None = Field(default=None)
    enabled: bool | None = Field(default=True)


class UpdateMCPServiceRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str | None = Field(default=None)
    description: str | None = Field(default=None)
    enabled: bool | None = Field(default=None)
    transport_type: str | None = Field(default=None)
    url: str | None = Field(default=None)
    stdio_config: MCPStdioConfig | None = Field(default=None)
    env_vars: dict[str, str] | None = Field(default=None)
    headers: dict[str, str] | None = Field(default=None)
    auth_config: MCPMcpServiceAuthConfig | None = Field(default=None)
    advanced_config: MCPServiceAdvancedConfig | None = Field(default=None)


class MCPTool(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    name: str
    description: str | None = Field(default=None)
    input_schema: JsonObject | None = Field(default=None, alias="inputSchema")
    require_approval: bool | None = Field(default=False)


class MCPResource(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    uri: str
    name: str
    description: str | None = Field(default=None)
    mime_type: str | None = Field(default=None, alias="mimeType")


class MCPTestResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool
    message: str
    description: str | None = Field(default=None)
    oauth_required: bool | None = Field(default=False)
    tools: list[MCPTool] = Field(default_factory=list)
    resources: list[MCPResource] = Field(default_factory=list)


class MCPToolApproval(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_name: str
    require_approval: bool
    updated_at: datetime


class SetMCPToolApprovalRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    require_approval: bool


class MCPToolApprovalDecisionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    decision: str
    modified_args: JsonObject | None = Field(default=None)
    reason: str | None = Field(default=None)


class VectorStoreConnectionField(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    type: str
    required: bool | None = Field(default=False)
    default: JsonValue | None = Field(default=None)
    description: str | None = Field(default=None)
    sensitive: bool | None = Field(default=False)


class VectorStoreTypeInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str
    display_name: str
    connection_fields: list[VectorStoreConnectionField] | None = Field(default=None)
    index_fields: list[VectorStoreConnectionField] | None = Field(default=None)


class VectorStore(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: int
    name: str
    engine_type: str
    connection_config: JsonObject | None = Field(default=None)
    index_config: JsonObject | None = Field(default=None)
    source: str | None = Field(default="user")
    readonly: bool | None = Field(default=False)
    created_at: datetime | None = Field(default=None)
    updated_at: datetime | None = Field(default=None)
    deleted_at: datetime | None = Field(default=None)


class TestVectorStoreRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    engine_type: str
    connection_config: JsonObject


class CreateVectorStoreRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    engine_type: str
    connection_config: JsonObject
    index_config: JsonObject | None = Field(default=None)


class UpdateVectorStoreRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str


class TestVectorStoreResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool
    version: str | None = Field(default=None)
    error: str | None = Field(default=None)


class StorageBackendConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: str | None = Field(default=None)
    endpoint: str | None = Field(default=None)
    region: str | None = Field(default=None)
    access_key_id: str | None = Field(default=None)
    secret_access_key: str | None = Field(default=None)
    bucket_name: str | None = Field(default=None)
    path_prefix: str | None = Field(default=None)
    app_id: str | None = Field(default=None)
    use_ssl: bool | None = Field(default=None)
    force_path_style: bool | None = Field(default=None)
    use_temp_bucket: bool | None = Field(default=None)
    temp_bucket_name: str | None = Field(default=None)
    temp_region: str | None = Field(default=None)


class StorageBackend(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: int
    name: str
    provider: str
    config: StorageBackendConfig | None = Field(default=None)
    source: str | None = Field(default=None)
    status: str | None = Field(default="active")
    legacy_alias: bool | None = Field(default=False)
    created_at: datetime | None = Field(default=None)
    updated_at: datetime | None = Field(default=None)
    deleted_at: datetime | None = Field(default=None)


class StorageBackendListResponse(BaseModel):
    """Wire for ``GET /storage-backends``.

    Go returns ``{"success": true, "data": [...], "default_storage_backend_id": ...}``
    — the list lives under ``data``, the default id at the top level.
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    data: list[StorageBackend] = Field(default_factory=list)
    default_storage_backend_id: str | None = Field(default=None)


class TestStorageBackendRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    provider: str
    config: StorageBackendConfig | None = Field(default=None)


class CreateStorageBackendRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    provider: str
    config: StorageBackendConfig | None = Field(default=None)
    status: str | None = Field(default="active")


class UpdateStorageBackendRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str | None = Field(default=None)
    provider: str | None = Field(default=None)
    config: StorageBackendConfig | None = Field(default=None)
    status: str | None = Field(default=None)


class WebSearchProviderTypeInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    label: str
    description: str | None = Field(default=None)
    parameter_schema: list[JsonObject] | None = Field(default=None)


class WebSearchBuiltinProvider(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    label: str
    description: str
    enabled: bool


class WebSearchProvider(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: int | None = Field(default=None)
    name: str
    provider: str
    description: str | None = Field(default=None)
    is_default: bool | None = Field(default=False)
    parameters: JsonObject | None = Field(default=None)
    created_at: datetime | None = Field(default=None)
    updated_at: datetime | None = Field(default=None)
    # Per-field "configured?" map mirroring Go's
    # ``dto.WebSearchProviderResponse.Credentials``.
    credentials: dict[str, CredentialFieldMetadata] | None = Field(default=None)


class TestWebSearchProviderRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    parameters: JsonObject


class CreateWebSearchProviderRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    provider: str
    description: str | None = Field(default=None)
    parameters: JsonObject | None = Field(default=None)
    is_default: bool | None = Field(default=False)


class UpdateWebSearchProviderRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str | None = Field(default=None)
    description: str | None = Field(default=None)
    parameters: JsonObject | None = Field(default=None)
    is_default: bool | None = Field(default=None)


class WebSearchProviderParameters(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    api_key: str | None = Field(default=None)
    # Google CSE uses ``cx`` (custom-search engine id) per Go docs/api/web-search.md.
    # Accept ``engine_id`` / ``engineId`` as alias for legacy callers.
    cx: str | None = Field(default=None, alias="cx")
    base_url: str | None = Field(default=None)
    proxy_url: str | None = Field(default=None)
    extra_config: dict[str, str] | None = Field(default=None)


# ── DataSource ────────────────────────────────────────────────────────


class DataSourceConfig(BaseModel):
    """Structured data-source connection config (parsed ``config`` JSON)."""

    model_config = ConfigDict(frozen=True)

    type: str | None = Field(default=None)
    resource_ids: list[str] | None = Field(default=None)
    settings: JsonObject | None = Field(default=None)


class CredentialFieldMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    configured: bool


class DataSource(BaseModel):
    """Wire shape for one data source (``DataSourceResponse``)."""

    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: int
    knowledge_base_id: str
    name: str
    type: str
    config: DataSourceConfig | None = Field(default=None)
    sync_schedule: str | None = Field(default=None)
    sync_mode: str | None = Field(default=None)
    status: str | None = Field(default=None)
    conflict_strategy: str | None = Field(default=None)
    sync_deletions: bool | None = Field(default=False)
    last_sync_at: datetime | None = Field(default=None)
    last_sync_cursor: JsonObject | None = Field(default=None)
    last_sync_result: JsonObject | None = Field(default=None)
    error_message: str | None = Field(default=None)
    sync_log_retention_days: int | None = Field(default=None)
    created_at: datetime
    updated_at: datetime
    total_items_synced: int | None = Field(default=None)
    latest_sync_log: SyncLog | None = Field(default=None)
    credentials: dict[str, CredentialFieldMetadata] | None = Field(default=None)


class CreateDataSourceRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    knowledge_base_id: str
    name: str
    type: str
    config: JsonObject | None = Field(default=None)
    sync_schedule: str | None = Field(default=None)
    sync_mode: str | None = Field(default=None)
    conflict_strategy: str | None = Field(default=None)
    sync_deletions: bool | None = Field(default=False)
    sync_log_retention_days: int | None = Field(default=None)


class UpdateDataSourceRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str | None = Field(default=None)
    config: JsonObject | None = Field(default=None)
    sync_schedule: str | None = Field(default=None)
    sync_mode: str | None = Field(default=None)
    conflict_strategy: str | None = Field(default=None)
    sync_deletions: bool | None = Field(default=None)
    sync_log_retention_days: int | None = Field(default=None)


class DataSourceConnectorMetadata(BaseModel):
    """Wire shape for ``GET /datasources/types`` (``ConnectorMetadata``)."""

    model_config = ConfigDict(frozen=True)

    type: str
    name: str
    description: str | None = Field(default=None)
    icon: str | None = Field(default=None)
    priority: int
    auth_type: str | None = Field(default=None)
    capabilities: list[str] = Field(default_factory=list)


class ValidateCredentialsRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str
    credentials: JsonObject


class ResolveResourceAncestorsRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    resource_ids: list[str] = Field(default_factory=list)


class SyncLog(BaseModel):
    """Wire shape for one data-source sync log entry."""

    model_config = ConfigDict(frozen=True)

    id: str
    data_source_id: str
    tenant_id: int
    status: str
    started_at: datetime
    finished_at: datetime | None = Field(default=None)
    items_total: int = 0
    items_created: int = 0
    items_updated: int = 0
    items_deleted: int = 0
    items_skipped: int = 0
    items_failed: int = 0
    error_message: str | None = Field(default=None)
    result: JsonObject | None = Field(default=None)
    created_at: datetime
    updated_at: datetime


# ── Initialization ────────────────────────────────────────────────────


class OllamaStatusResponse(BaseModel):
    """Wire for ``GET /initialization/ollama/status``.

    Go returns ``{"success": true, "data": {"available", "version", "baseUrl"}}``.
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    data: OllamaStatusData | None = Field(default=None)


class OllamaStatusData(BaseModel):
    model_config = ConfigDict(frozen=True)

    available: bool | None = Field(default=None)
    version: str | None = Field(default=None)
    base_url: str | None = Field(default=None, alias="baseUrl")


class OllamaModelInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    size: int | None = Field(default=None)
    digest: str | None = Field(default=None)
    modified_at: datetime | None = Field(default=None)


class OllamaModelsListResponse(BaseModel):
    """Wire for ``GET /initialization/ollama/models``.

    Go returns ``{"success": true, "data": {"models": [...]}}``.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    success: bool
    data: OllamaModelsData | None = Field(default=None)


class OllamaModelsData(BaseModel):
    model_config = ConfigDict(frozen=True)

    models: list[OllamaModelInfo] = Field(default_factory=list)


class CheckOllamaModelsRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    models: list[str] = Field(default_factory=list)


class DownloadOllamaModelRequest(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    model_name: str = Field(alias="modelName")


class ModelTestRequest(BaseModel):
    """Wire shape for ``/initialization/*/test|check`` model probes.

    Go's ``ModelTestRequest`` binds ``json:"modelName"`` (see
    ``internal/handler/initialization.go``); ``model`` is kept as an
    alias for legacy callers.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    source: str | None = Field(default=None)
    model: str = Field(default="", alias="modelName")
    base_url: str | None = Field(default=None, alias="baseUrl")
    api_key: str | None = Field(default=None, alias="apiKey")
    provider: str | None = Field(default=None)
    interface_type: str | None = Field(default=None, alias="interfaceType")
    dimension: int | None = Field(default=None)
    supports_dimension_override: bool | None = Field(
        default=False, alias="supportsDimensionOverride"
    )
    custom_headers: dict[str, str] | None = Field(default=None, alias="customHeaders")
    extra_config: dict[str, str] | None = Field(default=None, alias="extraConfig")
    app_secret: str | None = Field(default=None, alias="appSecret")
    model_id: str | None = Field(default=None, alias="modelId")


# ── Storage provider (SystemService) ──────────────────────────────────


class StorageProviderStatus(BaseModel):
    """Wire shape for a storage engine status item (``StorageEngineStatusItem``)."""

    model_config = ConfigDict(frozen=True)

    name: str
    allowed: bool
    available: bool
    description: str


# ── WeKnoraCloud ──────────────────────────────────────────────────────


class SaveWeKnoraCloudCredentialsRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    app_id: str
    app_secret: str


class WeKnoraCloudStatusResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    has_models: bool
    needs_reinit: bool
    reason: str | None = Field(default=None)


class Skill(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str


class SkillsListResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    data: list[Skill]
    skills_available: bool


__all__ = [
    "CheckOllamaModelsRequest",
    "CreateDataSourceRequest",
    "CreateMCPServiceRequest",
    "CreateModelRequest",
    "CreateStorageBackendRequest",
    "CreateVectorStoreRequest",
    "CreateWebSearchProviderRequest",
    "CredentialFieldMetadata",
    "DataSource",
    "DataSourceConfig",
    "DataSourceConnectorMetadata",
    "DownloadOllamaModelRequest",
    "EmbeddingParameters",
    "MCPMcpServiceAuthConfig",
    "MCPResource",
    "MCPService",
    "MCPServiceAdvancedConfig",
    "MCPStdioConfig",
    "MCPTestResult",
    "MCPTool",
    "MCPToolApproval",
    "MCPToolApprovalDecisionRequest",
    "Model",
    "ModelParameters",
    "ModelTestRequest",
    "OllamaModelInfo",
    "OllamaModelsData",
    "OllamaModelsListResponse",
    "OllamaStatusData",
    "OllamaStatusResponse",
    "ProviderTypeMeta",
    "ResolveResourceAncestorsRequest",
    "SaveWeKnoraCloudCredentialsRequest",
    "SetMCPToolApprovalRequest",
    "Skill",
    "SkillsListResponse",
    "StorageBackend",
    "StorageBackendConfig",
    "StorageBackendListResponse",
    "StorageProviderStatus",
    "SyncLog",
    "TestStorageBackendRequest",
    "TestVectorStoreRequest",
    "TestVectorStoreResponse",
    "TestWebSearchProviderRequest",
    "UpdateDataSourceRequest",
    "UpdateMCPServiceRequest",
    "UpdateModelRequest",
    "UpdateStorageBackendRequest",
    "UpdateVectorStoreRequest",
    "UpdateWebSearchProviderRequest",
    "ValidateCredentialsRequest",
    "VectorStore",
    "VectorStoreConnectionField",
    "VectorStoreTypeInfo",
    "WeKnoraCloudStatusResponse",
    "WebSearchBuiltinProvider",
    "WebSearchProvider",
    "WebSearchProviderParameters",
    "WebSearchProviderTypeInfo",
]
