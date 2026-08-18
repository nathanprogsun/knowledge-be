"""Cloud credential endpoints.

``POST /weknoracloud/credentials`` saves the cloud APPID/APPSECRET to
the tenant config blob (no models are auto-created); the status check
lives on the models router (``GET /models/weknoracloud/status``).
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict

from src.common.exception import ValidationError
from src.web.deps import AuthDep, RoleAdminDep
from src.web.deps.tenants import TenantServiceDep

router = APIRouter(prefix="/weknoracloud", tags=["weknoracloud"])

#: Key under ``tenants.credentials`` that holds the cloud credentials.
CREDENTIALS_KEY = "weknoracloud"


class SaveCredentialsRequest(BaseModel):
    """Body for ``POST /weknoracloud/credentials``."""

    model_config = ConfigDict(frozen=True)

    app_id: str
    app_secret: str


class SaveCredentialsResponse(BaseModel):
    """Ack for the credential save."""

    model_config = ConfigDict(frozen=True)

    success: bool
    message: str


def _require_tenant_id(request: Request) -> int:
    tenant_id = int(request.state.tenant_id or 0)
    if tenant_id <= 0:
        raise ValidationError(
            code="weknoracloud.no_tenant",
            message="workspace context missing",
        )
    return tenant_id


@router.post("/credentials", response_model=SaveCredentialsResponse)
async def save_credentials(
    _auth: AuthDep,
    _role: RoleAdminDep,
    body: SaveCredentialsRequest,
    tenant_service: TenantServiceDep,
    request: Request,
) -> SaveCredentialsResponse:
    """Save the cloud APPID/APPSECRET to the current workspace config."""
    app_id = body.app_id.strip()
    app_secret = body.app_secret.strip()
    if not app_id or not app_secret:
        raise ValidationError(
            code="weknoracloud.missing_credentials",
            message="app_id and app_secret are required",
        )
    tenant_id = _require_tenant_id(request)
    tenant = await tenant_service.get_tenant(tenant_id)
    credentials = dict(tenant.credentials or {})
    credentials[CREDENTIALS_KEY] = {"app_id": app_id, "app_secret": app_secret}
    await tenant_service.update_tenant(tenant_id, credentials=credentials)
    return SaveCredentialsResponse(success=True, message="凭证保存成功")


__all__ = ["router"]
