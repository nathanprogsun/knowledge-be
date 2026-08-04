from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject


class EmbeddingParameters(BaseModel):
    model_config = ConfigDict(frozen=True)

    dimension: int | None = Field(default=None)
    truncate_prompt_tokens: int | None = Field(default=0)


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
    app_secret: str | None = Field(default=None)


class Model(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: int
    name: str
    type: str
    source: str
    description: str | None = Field(default=None)
    parameters: ModelParameters
    is_default: bool | None = Field(default=False)
    is_builtin: bool | None = Field(default=False)
    status: str | None = Field(default="active")
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = Field(default=None)


class CreateModelRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    type: str
    source: str
    description: str | None = Field(default=None)
    parameters: ModelParameters


class UpdateModelRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str | None = Field(default=None)
    description: str | None = Field(default=None)
    type: str | None = Field(default=None)
    source: str | None = Field(default=None)
    parameters: ModelParameters | None = Field(default=None)


class ProviderTypeMeta(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    value: str
    label: str
    description: str | None = Field(default=None)
    default_urls: dict[str, str] | None = Field(default=None, alias="defaultUrls")
    model_types: list[str] | None = Field(default=None, alias="modelTypes")


class MCPMcpServiceAuthConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    api_key: str | None = Field(default=None)
    token: str | None = Field(default=None)


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
    default: JsonObject | None = Field(default=None)
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
    name: str
    engine_type: str
    connection_config: JsonObject | None = Field(default=None)
    index_config: JsonObject | None = Field(default=None)
    source: str | None = Field(default="user")
    readonly: bool | None = Field(default=False)
    created_at: datetime | None = Field(default=None)
    updated_at: datetime | None = Field(default=None)


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
    name: str
    provider: str
    config: StorageBackendConfig | None = Field(default=None)
    source: str | None = Field(default=None)
    status: str | None = Field(default="active")
    legacy_alias: bool | None = Field(default=False)
    created_at: datetime | None = Field(default=None)
    updated_at: datetime | None = Field(default=None)


class StorageBackendListResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    items: list[StorageBackend]
    default_storage_backend_id: str


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


class Skill(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str


class SkillsListResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    data: list[Skill]
    skills_available: bool


__all__ = [
    "CreateMCPServiceRequest",
    "CreateModelRequest",
    "CreateStorageBackendRequest",
    "CreateVectorStoreRequest",
    "CreateWebSearchProviderRequest",
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
    "ProviderTypeMeta",
    "SetMCPToolApprovalRequest",
    "Skill",
    "SkillsListResponse",
    "StorageBackend",
    "StorageBackendConfig",
    "StorageBackendListResponse",
    "TestStorageBackendRequest",
    "TestVectorStoreRequest",
    "TestVectorStoreResponse",
    "TestWebSearchProviderRequest",
    "UpdateMCPServiceRequest",
    "UpdateModelRequest",
    "UpdateStorageBackendRequest",
    "UpdateVectorStoreRequest",
    "UpdateWebSearchProviderRequest",
    "VectorStore",
    "VectorStoreConnectionField",
    "VectorStoreTypeInfo",
    "WebSearchBuiltinProvider",
    "WebSearchProvider",
    "WebSearchProviderTypeInfo",
]
