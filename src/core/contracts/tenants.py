from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class RetrieverEngineEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    retriever_type: str
    retriever_engine_type: str


class RetrieverEnginesConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    engines: list[RetrieverEngineEntry]


class Tenant(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    name: str
    description: str | None = Field(default=None)
    api_key: str | None = Field(default=None)
    status: str
    retriever_engines: RetrieverEnginesConfig | None = Field(default=None)
    business: str | None = Field(default=None)
    storage_quota: int | None = Field(default=None)
    storage_used: int | None = Field(default=None)
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = Field(default=None)


class TenantList(BaseModel):
    model_config = ConfigDict(frozen=True)

    items: list[Tenant]
    total: int | None = Field(default=None)
    page: int | None = Field(default=None)
    page_size: int | None = Field(default=None)


class CreateTenantRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str | None = Field(default=None)
    business: str | None = Field(default=None)
    retriever_engines: RetrieverEnginesConfig | None = Field(default=None)
    storage_quota: int | None = Field(default=None)


class UpdateTenantRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str | None = Field(default=None)
    description: str | None = Field(default=None)
    business: str | None = Field(default=None)
    retriever_engines: RetrieverEnginesConfig | None = Field(default=None)
    storage_quota: int | None = Field(default=None)
    status: str | None = Field(default=None)


class TenantAPIKey(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    role: str
    token: str | None = Field(default=None)
    key_prefix: str | None = Field(default=None)
    knowledge_base_ids: list[str] | None = Field(default=None)
    expires_at_unix: int | None = Field(default=None)
    created_at: datetime
    revoked_at: datetime | None = Field(default=None)


class CreateAPIKeyRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    role: str
    knowledge_base_ids: list[str] | None = Field(default=None)
    expires_at_unix: int | None = Field(default=None)


class APIPrincipalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: str
    direct_header_name: str | None = Field(default="X-External-User-ID")
    signed_token_header_name: str | None = Field(default="X-External-User-Token")
    require_direct_header: bool = False
    has_hmac_secret: bool = False


class UpdateAPIPrincipalConfigRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: str
    direct_header_name: str | None = Field(default=None)
    signed_token_header_name: str | None = Field(default=None)
    require_direct_header: bool | None = Field(default=None)
    hmac_secret: str | None = Field(default=None)


class SearchTenantsQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    keyword: str | None = Field(default=None)
    tenant_id: int | None = Field(default=None)
    page: int = 1
    page_size: int = 20


class TenantKVAgentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_iterations: int
    allowed_tools: list[str] | None = Field(default=None)
    temperature: float
    system_prompt: str
    use_custom_system_prompt: bool = False
    available_tools: list[dict[str, object]] | None = Field(default=None)
    available_placeholders: list[dict[str, object]] | None = Field(default=None)


class TenantKVWebSearchConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_results: int | None = Field(default=None)


class TenantKVConversationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    keyword_threshold: float | None = Field(default=None)
    vector_threshold: float | None = Field(default=None)
    rerank_threshold: float | None = Field(default=None)
    temperature: float | None = Field(default=None)
    max_completion_tokens: int | None = Field(default=None)


class TenantKVRetrievalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    embedding_top_k: int | None = Field(default=None)
    rerank_top_k: int | None = Field(default=None)
    keyword_threshold: float | None = Field(default=None)
    vector_threshold: float | None = Field(default=None)
    rerank_threshold: float | None = Field(default=None)


class TenantKVParserEngineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    engines: list[dict[str, object]] | None = Field(default=None)


class TenantKVStorageEngineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_provider: str | None = Field(default=None)
    providers: list[dict[str, object]] | None = Field(default=None)


class TenantKVChatHistoryConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool
    embedding_model_id: str | None = Field(default=None)
    knowledge_base_id: str | None = Field(default=None)


class TenantKVPromptTemplates(BaseModel):
    model_config = ConfigDict(frozen=True)

    templates: list[dict[str, object]]


__all__ = [
    "APIPrincipalConfig",
    "CreateAPIKeyRequest",
    "CreateTenantRequest",
    "RetrieverEngineEntry",
    "RetrieverEnginesConfig",
    "SearchTenantsQuery",
    "Tenant",
    "TenantAPIKey",
    "TenantKVAgentConfig",
    "TenantKVChatHistoryConfig",
    "TenantKVConversationConfig",
    "TenantKVParserEngineConfig",
    "TenantKVPromptTemplates",
    "TenantKVRetrievalConfig",
    "TenantKVStorageEngineConfig",
    "TenantKVWebSearchConfig",
    "TenantList",
    "UpdateAPIPrincipalConfigRequest",
    "UpdateTenantRequest",
]
