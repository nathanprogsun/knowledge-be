"""Stage-2 frozen contract surface: the infrastructure domains.

Re-exported for the stage-2 contract tests and for downstream Stage 3
PRs. Each entry names the frozen Pydantic model, its wire endpoint, and
the fixture key it is compared against.

The fixture field sets are captured from the Go side:

- ``internal/handler/dto/*.go`` — Model, MCPService, WebSearchProvider,
  DataSource response shapes (the response DTOs that strip secret
  fields by construction);
- ``internal/types/*.go`` — VectorStore, StorageBackend, SyncLog and
  the request parameter models;
- ``docs/api/*.md`` — the vector-store response shape (``source`` /
  ``readonly`` are documented even though the storage type omits them);
- ``docs/swagger.json`` — the handler request shapes (``display_name``
  on the model requests, camelCase ``ModelTestRequest`` fields).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias

from pydantic import BaseModel

from src.core.contracts import infra

# (contract_name, model, wire_endpoint) — the Model domain.
MODEL_CONTRACTS: tuple[tuple[str, type[BaseModel], str], ...] = (
    ("Model", infra.Model, "GET /models/{id}"),
    ("ModelParameters", infra.ModelParameters, "models.parameters"),
    ("CreateModelRequest", infra.CreateModelRequest, "POST /models"),
    ("UpdateModelRequest", infra.UpdateModelRequest, "PUT /models/{id}"),
    ("ProviderTypeMeta", infra.ProviderTypeMeta, "GET /models/providers"),
)

# (contract_name, model, wire_endpoint) — the MCP service domain.
MCP_CONTRACTS: tuple[tuple[str, type[BaseModel], str], ...] = (
    ("MCPService", infra.MCPService, "GET /mcp-services/{id}"),
    ("MCPMcpServiceAuthConfig", infra.MCPMcpServiceAuthConfig, "mcp.auth-config"),
    ("MCPServiceAdvancedConfig", infra.MCPServiceAdvancedConfig, "mcp.advanced-config"),
    ("MCPStdioConfig", infra.MCPStdioConfig, "mcp.stdio-config"),
    ("CreateMCPServiceRequest", infra.CreateMCPServiceRequest, "POST /mcp-services"),
    ("UpdateMCPServiceRequest", infra.UpdateMCPServiceRequest, "PUT /mcp-services/{id}"),
    ("MCPTool", infra.MCPTool, "mcp.tool"),
    ("MCPResource", infra.MCPResource, "mcp.resource"),
    ("MCPTestResult", infra.MCPTestResult, "POST /mcp-services/{id}/test"),
    ("MCPToolApproval", infra.MCPToolApproval, "mcp.tool-approval"),
    (
        "SetMCPToolApprovalRequest",
        infra.SetMCPToolApprovalRequest,
        "PUT /mcp-services/{id}/tool-approvals/{tool_name}",
    ),
)

# (contract_name, model, wire_endpoint) — the VectorStore domain.
VECTOR_STORE_CONTRACTS: tuple[tuple[str, type[BaseModel], str], ...] = (
    ("VectorStore", infra.VectorStore, "GET /vector-stores/{id}"),
    ("TestVectorStoreRequest", infra.TestVectorStoreRequest, "POST /vector-stores/test"),
    ("CreateVectorStoreRequest", infra.CreateVectorStoreRequest, "POST /vector-stores"),
    ("UpdateVectorStoreRequest", infra.UpdateVectorStoreRequest, "PUT /vector-stores/{id}"),
    ("TestVectorStoreResponse", infra.TestVectorStoreResponse, "vector-stores.test"),
    ("VectorStoreConnectionField", infra.VectorStoreConnectionField, "vector-stores.field"),
    ("VectorStoreTypeInfo", infra.VectorStoreTypeInfo, "GET /vector-stores/types"),
)

# (contract_name, model, wire_endpoint) — the StorageBackend domain.
STORAGE_BACKEND_CONTRACTS: tuple[tuple[str, type[BaseModel], str], ...] = (
    ("StorageBackend", infra.StorageBackend, "GET /storage-backends/{id}"),
    ("StorageBackendConfig", infra.StorageBackendConfig, "storage.config"),
    ("StorageBackendListResponse", infra.StorageBackendListResponse, "GET /storage-backends"),
    ("TestStorageBackendRequest", infra.TestStorageBackendRequest, "POST /storage-backends/test"),
    ("CreateStorageBackendRequest", infra.CreateStorageBackendRequest, "POST /storage-backends"),
    (
        "UpdateStorageBackendRequest",
        infra.UpdateStorageBackendRequest,
        "PUT /storage-backends/{id}",
    ),
)

# (contract_name, model, wire_endpoint) — the WebSearch domain.
WEB_SEARCH_CONTRACTS: tuple[tuple[str, type[BaseModel], str], ...] = (
    ("WebSearchProvider", infra.WebSearchProvider, "GET /web-search-providers/{id}"),
    (
        "WebSearchProviderTypeInfo",
        infra.WebSearchProviderTypeInfo,
        "GET /web-search-providers/types",
    ),
    ("WebSearchBuiltinProvider", infra.WebSearchBuiltinProvider, "web-search.builtin"),
    (
        "TestWebSearchProviderRequest",
        infra.TestWebSearchProviderRequest,
        "POST /web-search-providers/test",
    ),
    (
        "CreateWebSearchProviderRequest",
        infra.CreateWebSearchProviderRequest,
        "POST /web-search-providers",
    ),
    (
        "UpdateWebSearchProviderRequest",
        infra.UpdateWebSearchProviderRequest,
        "PUT /web-search-providers/{id}",
    ),
)

# (contract_name, model, wire_endpoint) — the DataSource domain.
DATASOURCE_CONTRACTS: tuple[tuple[str, type[BaseModel], str], ...] = (
    ("DataSource", infra.DataSource, "GET /datasource/{id}"),
    ("DataSourceConfig", infra.DataSourceConfig, "datasource.config"),
    ("CreateDataSourceRequest", infra.CreateDataSourceRequest, "POST /datasource"),
    ("UpdateDataSourceRequest", infra.UpdateDataSourceRequest, "PUT /datasource/{id}"),
    ("DataSourceConnectorMetadata", infra.DataSourceConnectorMetadata, "GET /datasource/types"),
    (
        "ValidateCredentialsRequest",
        infra.ValidateCredentialsRequest,
        "POST /datasource/validate-credentials",
    ),
    (
        "ResolveResourceAncestorsRequest",
        infra.ResolveResourceAncestorsRequest,
        "POST /datasource/{id}/resource-ancestors",
    ),
    ("SyncLog", infra.SyncLog, "datasource.sync-log"),
    ("CredentialFieldMetadata", infra.CredentialFieldMetadata, "credentials.metadata"),
)

# (contract_name, model, wire_endpoint) — the Initialization domain.
INITIALIZATION_CONTRACTS: tuple[tuple[str, type[BaseModel], str], ...] = (
    ("OllamaStatusResponse", infra.OllamaStatusResponse, "GET /initialization/ollama/status"),
    ("OllamaStatusData", infra.OllamaStatusData, "ollama.status-data"),
    (
        "OllamaModelsListResponse",
        infra.OllamaModelsListResponse,
        "GET /initialization/ollama/models",
    ),
    ("OllamaModelsData", infra.OllamaModelsData, "ollama.models-data"),
    ("OllamaModelInfo", infra.OllamaModelInfo, "ollama.model-info"),
    (
        "CheckOllamaModelsRequest",
        infra.CheckOllamaModelsRequest,
        "POST /initialization/ollama/models/check",
    ),
    (
        "DownloadOllamaModelRequest",
        infra.DownloadOllamaModelRequest,
        "POST /initialization/ollama/models/download",
    ),
    ("ModelTestRequest", infra.ModelTestRequest, "POST /initialization/embedding/test"),
)

# Every stage-2 wire contract, flattened for uniform iteration.
ALL_STAGE2_CONTRACTS: tuple[tuple[str, type[BaseModel], str], ...] = (
    MODEL_CONTRACTS
    + MCP_CONTRACTS
    + VECTOR_STORE_CONTRACTS
    + STORAGE_BACKEND_CONTRACTS
    + WEB_SEARCH_CONTRACTS
    + DATASOURCE_CONTRACTS
    + INITIALIZATION_CONTRACTS
)

FIXTURE_PATH: Path = Path(__file__).parent / "fixtures" / "infra_responses.json"


def load_fixture_fields() -> dict[str, list[str]]:
    """Return the contract-name -> expected wire field-name map from the fixture."""
    raw = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for name, fields in raw.items():
        if isinstance(name, str) and isinstance(fields, list):
            out[name] = [f for f in fields if isinstance(f, str)]
    return out


def model_wire_fields(model: type[BaseModel]) -> list[str]:
    """Return the wire (serialization) field names of a frozen contract.

    Re-exported from the stage-1 module so both contract surfaces share
    the alias-aware projection.
    """
    from tests.contracts.stage1_contract import model_wire_fields as _project

    return _project(model)


JsonFixture: TypeAlias = dict[str, object]

__all__ = [
    "ALL_STAGE2_CONTRACTS",
    "DATASOURCE_CONTRACTS",
    "FIXTURE_PATH",
    "INITIALIZATION_CONTRACTS",
    "MCP_CONTRACTS",
    "MODEL_CONTRACTS",
    "STORAGE_BACKEND_CONTRACTS",
    "VECTOR_STORE_CONTRACTS",
    "WEB_SEARCH_CONTRACTS",
    "JsonFixture",
    "load_fixture_fields",
    "model_wire_fields",
]
