from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject

TenantRole = Literal["owner", "admin", "contributor", "viewer"]


class RetrieverEngineEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    retriever_type: str
    retriever_engine_type: str


class RetrieverEnginesConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    engines: list[RetrieverEngineEntry]


class Membership(BaseModel):
    """A user's role inside one tenant."""

    model_config = ConfigDict(frozen=True)

    tenant_id: int
    tenant_name: str
    role: TenantRole


class Tenant(BaseModel):
    """HTTP wire shape for a tenant.

    All configuration blobs are optional and are populated only when the
    caller's role permits access to workspace-level secrets.
    """

    model_config = ConfigDict(frozen=True)

    id: int
    name: str
    description: str | None = Field(default=None)
    status: str
    retriever_engines: RetrieverEnginesConfig | None = Field(default=None)
    business: str | None = Field(default=None)
    storage_quota: int | None = Field(default=None)
    storage_used: int | None = Field(default=None)
    context_config: JsonObject | None = Field(default=None)
    web_search_config: JsonObject | None = Field(default=None)
    parser_engine_config: JsonObject | None = Field(default=None)
    credentials: JsonObject | None = Field(default=None)
    storage_engine_config: JsonObject | None = Field(default=None)
    chat_history_config: JsonObject | None = Field(default=None)
    retrieval_config: JsonObject | None = Field(default=None)
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = Field(default=None)


class TenantList(BaseModel):
    """Paginated tenant listing.

    Field names mirror Go's ``ListTenantsResponse`` / pagination
    contract: ``data`` is the array of tenants (not ``items``);
    ``total``, ``page``, ``page_size`` are siblings at the envelope
    level.
    """

    model_config = ConfigDict(frozen=True)

    data: list[Tenant]
    total: int = 0
    page: int = 1
    page_size: int = 20


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

    id: int
    scope_type: str
    name: str
    full_access: bool
    knowledge_base_ids: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    last_used_at: datetime | None = Field(default=None)
    expires_at: datetime | None = Field(default=None)
    created_at: datetime


class CreateAPIKeyRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    full_access: bool = False
    knowledge_base_ids: list[str] | None = Field(default=None)
    capabilities: list[str] | None = Field(default=None)
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
    available_tools: list[JsonObject] | None = Field(default=None)
    available_placeholders: list[JsonObject] | None = Field(default=None)


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

    engines: list[JsonObject] | None = Field(default=None)


class TenantKVStorageEngineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_provider: str | None = Field(default=None)
    providers: list[JsonObject] | None = Field(default=None)


class TenantKVChatHistoryConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool
    embedding_model_id: str | None = Field(default=None)
    knowledge_base_id: str | None = Field(default=None)


class TenantKVPromptTemplates(BaseModel):
    model_config = ConfigDict(frozen=True)

    templates: list[JsonObject]


__all__ = [
    "APIPrincipalConfig",
    "CreateAPIKeyRequest",
    "CreateTenantRequest",
    "Membership",
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
    "TenantRole",
    "UpdateAPIPrincipalConfigRequest",
    "UpdateTenantRequest",
]
