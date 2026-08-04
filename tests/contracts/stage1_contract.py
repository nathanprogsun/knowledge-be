"""Stage-1 frozen contract surface: auth + tenants + system.

Re-exported for the stage-1 contract tests and for downstream Stage 2 PRs.
Each entry names the frozen Pydantic model, its wire endpoint, and the
fixture key it is compared against.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias

from pydantic import BaseModel

from src.core.contracts import auth, system, tenants

# (contract_name, model, wire_endpoint) — the auth stage-1 surface.
AUTH_CONTRACTS: tuple[tuple[str, type[BaseModel], str], ...] = (
    ("RegisterRequest", auth.RegisterRequest, "POST /auth/register"),
    ("RegisterResponse", auth.RegisterResponse, "POST /auth/register"),
    ("LoginRequest", auth.LoginRequest, "POST /auth/login"),
    ("LoginResponse", auth.LoginResponse, "POST /auth/login"),
    ("RefreshTokenRequest", auth.RefreshTokenRequest, "POST /auth/refresh"),
    ("RefreshTokenResponse", auth.RefreshTokenResponse, "POST /auth/refresh"),
    ("ChangePasswordRequest", auth.ChangePasswordRequest, "POST /auth/change-password"),
    ("MeResponse", auth.MeResponse, "GET /auth/me"),
    ("ValidateTokenResponse", auth.ValidateTokenResponse, "GET /auth/validate"),
    ("OIDCMetaConfig", auth.OIDCMetaConfig, "GET /auth/oidc/config"),
    ("OIDCAuthorizeURLResponse", auth.OIDCAuthorizeURLResponse, "GET /auth/oidc/url"),
    ("OIDCCallbackResponse", auth.OIDCCallbackResponse, "GET /auth/oidc/callback"),
    ("AuthUser", auth.AuthUser, "auth.user"),
)

# (contract_name, model, wire_endpoint) — the tenants stage-1 surface.
TENANT_CONTRACTS: tuple[tuple[str, type[BaseModel], str], ...] = (
    ("Tenant", tenants.Tenant, "tenant"),
    ("TenantList", tenants.TenantList, "GET /tenants"),
    ("CreateTenantRequest", tenants.CreateTenantRequest, "POST /tenants"),
    ("UpdateTenantRequest", tenants.UpdateTenantRequest, "PUT /tenants/{id}"),
    ("TenantAPIKey", tenants.TenantAPIKey, "tenant.api-key"),
    ("CreateAPIKeyRequest", tenants.CreateAPIKeyRequest, "POST /tenants/{id}/api-keys"),
    ("APIPrincipalConfig", tenants.APIPrincipalConfig, "tenant.api-principal-config"),
    (
        "UpdateAPIPrincipalConfigRequest",
        tenants.UpdateAPIPrincipalConfigRequest,
        "PUT /tenants/{id}/api-principal-config",
    ),
    ("Membership", tenants.Membership, "tenant.membership"),
)

# (contract_name, model, wire_endpoint) — the system stage-1 surface.
SYSTEM_CONTRACTS: tuple[tuple[str, type[BaseModel], str], ...] = (
    ("SystemInfo", system.SystemInfo, "GET /system/info"),
    ("ParserEnginesList", system.ParserEnginesList, "GET /system/parser-engines"),
    (
        "StorageEngineCheckRequest",
        system.StorageEngineCheckRequest,
        "POST /system/storage-engine-check",
    ),
)

# Every stage-1 wire contract, flattened for uniform iteration.
ALL_STAGE1_CONTRACTS: tuple[tuple[str, type[BaseModel], str], ...] = (
    AUTH_CONTRACTS + TENANT_CONTRACTS + SYSTEM_CONTRACTS
)

FIXTURE_PATH: Path = Path(__file__).parent / "fixtures" / "auth_tenant_responses.json"


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

    Respects Pydantic ``alias``/``serialization_alias`` so the comparison
    is against the actual JSON keys the Go API emits.
    """
    out: list[str] = []
    for fname, field in model.model_fields.items():
        if fname == "model_config":
            continue
        out.append(field.serialization_alias or field.alias or fname)
    return out


JsonFixture: TypeAlias = dict[str, object]

__all__ = [
    "ALL_STAGE1_CONTRACTS",
    "AUTH_CONTRACTS",
    "FIXTURE_PATH",
    "SYSTEM_CONTRACTS",
    "TENANT_CONTRACTS",
    "JsonFixture",
    "load_fixture_fields",
    "model_wire_fields",
]
