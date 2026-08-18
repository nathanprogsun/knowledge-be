"""Public MCP OAuth provider redirect callback.

The third-party authorization server redirects the browser here after
the user approves/denies an MCP OAuth flow. No bearer token is carried
on this redirect, so the route is unauthenticated: the single-use CSRF
``state`` parameter is the authorisation. The callback exchanges the
``code`` for a token set and bounces the browser back to the SPA with
a fragment marker the UI renders as a toast.
"""

from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

from src.app_logging import logger
from src.core.infra.mcp_services.factory import build_mcp_service
from src.web.deps.session import SessionDep

router = APIRouter(prefix="/mcp-oauth", tags=["mcp-oauth"])

_FALLBACK_REDIRECT = "/"


def _error_redirect(message: str) -> RedirectResponse:
    return RedirectResponse(
        f"{_FALLBACK_REDIRECT}#mcp_oauth_error={quote(message)}",
        status_code=302,
    )


@router.get("/callback")
async def mcp_oauth_callback(
    request: Request,
    session: SessionDep,
    state: str = "",
    code: str = "",
    error: str = "",
) -> RedirectResponse:
    """Handle the provider redirect: exchange the code, bounce to the SPA."""
    if error:
        return _error_redirect(error)
    if not state or not code:
        return _error_redirect("missing_code_or_state")

    lifespan_service = getattr(request.app.state, "lifespan_service", None)
    state_store = (
        getattr(lifespan_service, "mcp_oauth_state_store", None)
        if lifespan_service is not None
        else None
    )
    oauth_factory = (
        getattr(lifespan_service, "mcp_oauth_manager_factory", None)
        if lifespan_service is not None
        else None
    )
    if state_store is None or oauth_factory is None:
        return _error_redirect("authorization_failed")

    entry = state_store.peek(state=state)
    if entry is None:
        return _error_redirect("authorization_failed")

    try:
        mcp_service = build_mcp_service(session, oauth_manager_factory=oauth_factory)
        manager = await mcp_service.fetch_oauth_manager(
            tenant_id=entry.tenant_id,
            service_id=entry.service_id,
        )
        await manager.exchange_code(
            user_id=entry.user_id,
            code=code,
            state=state,
            redirect_uri=entry.redirect_uri,
        )
    except Exception as exc:  # noqa: BLE001 — any failure maps to the SPA error marker
        logger.warning("MCP OAuth callback exchange failed: {}", exc)
        return _error_redirect("authorization_failed")

    return RedirectResponse(
        f"{_FALLBACK_REDIRECT}#mcp_oauth_result=success",
        status_code=302,
    )


__all__ = ["router"]
