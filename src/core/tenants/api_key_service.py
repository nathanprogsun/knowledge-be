"""Tenant API-key service — issuance, authentication, revocation.

A key is a machine credential: the caller sees the plaintext token
exactly once (at creation), and every later lookup goes through the
SHA-256 hash stored in ``key_hash``.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from src.common.exception import NotFoundError, ValidationError
from src.core.tenants.types import TenantAPIKeyInfo
from src.db.dao.tenant_api_keys_repository import TenantAPIKeyRepository
from src.db.models.tenants.tenant_api_keys import TenantAPIKey

SCOPE_TENANT = "tenant"
SCOPE_PLATFORM = "platform"

# Token shape: `sk-` + 32 random bytes, base64url without padding.
_TOKEN_PREFIX = "sk-"
_TOKEN_ENTROPY_BYTES = 32

# ``last_used_at`` is persisted at most once per key per minute: the UI
# only needs minute-level freshness and auth is a hot path.
_LAST_USED_MIN_INTERVAL = timedelta(minutes=1)

# Bounded grants a non-full-access key may carry. Unknown values are
# dropped rather than rejected.
_KNOWN_CAPABILITIES: frozenset[str] = frozenset(
    {
        "retrieve",
        "chat",
        "read_agents",
        "ingest",
        "manage_kbs",
        "manage_agents",
        "message_history",
        "manage_models",
        "manage_mcp_services",
        "manage_datasources",
        "manage_channels",
        "manage_vector_stores",
        "manage_storage_backends",
        "manage_web_search",
        "run_evaluations",
        "manage_members",
        "manage_spaces",
        "manage_tenant_settings",
        "system_tenants_read",
        "system_tenants_manage",
        "system_settings_read",
        "system_settings_manage",
        "system_runtime_read",
        "system_runtime_manage",
        "system_audit_read",
    }
)


def generate_api_key_token() -> str:
    """Mint a fresh opaque token."""
    raw = secrets.token_bytes(_TOKEN_ENTROPY_BYTES)
    return _TOKEN_PREFIX + base64.urlsafe_b64encode(raw).decode().rstrip("=")


def hash_api_key_token(token: str) -> str:
    """Hash a token the way the authentication lookup expects."""
    return hashlib.sha256(token.encode()).hexdigest()


def normalize_scope_type(scope_type: str | None) -> str:
    """Map free-form input onto a known scope, defaulting to ``tenant``."""
    normalized = (scope_type or "").strip().lower()
    return SCOPE_PLATFORM if normalized == SCOPE_PLATFORM else SCOPE_TENANT


def normalize_capabilities(capabilities: list[str] | None) -> list[str]:
    """Lower-case, de-duplicate and drop unknown capabilities, order kept."""
    return _normalize_unique(capabilities, known=_KNOWN_CAPABILITIES)


def normalize_knowledge_base_ids(ids: list[str] | None) -> list[str]:
    """Trim, de-duplicate and drop blank KB ids, order kept."""
    return _normalize_unique(ids, known=None, lower=False)


def _normalize_unique(
    values: list[str] | None,
    *,
    known: frozenset[str] | None,
    lower: bool = True,
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        item = value.strip().lower() if lower else value.strip()
        if not item or (known is not None and item not in known) or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


@dataclass(frozen=True, slots=True)
class APIKeyCreateResult:
    """Returned by ``TenantAPIKeyService.create_api_key``.

    ``token`` is the only time the plaintext credential is available.
    """

    key: TenantAPIKeyInfo
    token: str


class TenantAPIKeyService:
    """Stateless API-key service, constructed per request."""

    def __init__(self, *, api_keys_repo: TenantAPIKeyRepository) -> None:
        self._api_keys_repo = api_keys_repo

    # ── Issuance ────────────────────────────────────────────────────

    async def create_api_key(
        self,
        *,
        name: str,
        tenant_id: int | None = None,
        scope_type: str = SCOPE_TENANT,
        full_access: bool = False,
        knowledge_base_ids: list[str] | None = None,
        capabilities: list[str] | None = None,
        expires_at: datetime | None = None,
    ) -> APIKeyCreateResult:
        """Mint, hash and persist a key; return it with its plaintext token.

        Scope rules enforced below: ``tenant`` scope requires a workspace
        id; ``platform`` scope forbids ``full_access`` and requires at
        least one capability (a platform key is not bounded by a
        workspace). ``full_access`` keys carry neither a KB allow-list
        nor capabilities — the grant is already unbounded within the
        workspace. The plaintext token is returned only here; lookups
        go through the SHA-256 hash.
        """
        scope = normalize_scope_type(scope_type)
        clean_capabilities = normalize_capabilities(capabilities)
        self._validate_scope(
            scope,
            tenant_id=tenant_id,
            full_access=full_access,
            capabilities=clean_capabilities,
        )
        clean_name = name.strip()
        if not clean_name:
            raise ValidationError(
                code="tenant_api_key.name_required",
                message="Name is required",
            )

        token = generate_api_key_token()
        now = datetime.now(UTC)
        row = TenantAPIKey(
            tenant_id=tenant_id if scope == SCOPE_TENANT else None,
            scope_type=scope,
            name=clean_name,
            key_hash=hash_api_key_token(token),
            api_key=token,
            full_access=full_access,
            # A full-access key is already unbounded inside its
            # workspace, so both narrowing lists are cleared.
            knowledge_base_ids=(
                [] if full_access else normalize_knowledge_base_ids(knowledge_base_ids)
            ),
            capabilities=[] if full_access else clean_capabilities,
            expires_at=expires_at.astimezone(UTC) if expires_at is not None else None,
            created_at=now,
            updated_at=now,
        )
        stored = await self._api_keys_repo.insert(row)
        return APIKeyCreateResult(key=TenantAPIKeyInfo.map_from_db(stored), token=token)

    @staticmethod
    def _validate_scope(
        scope: str,
        *,
        tenant_id: int | None,
        full_access: bool,
        capabilities: list[str],
    ) -> None:
        """Enforce the tenant/platform rules on create."""
        if scope == SCOPE_TENANT and not tenant_id:
            raise ValidationError(
                code="tenant_api_key.tenant_required",
                message="tenant_id is required",
            )
        if scope != SCOPE_PLATFORM:
            return
        if full_access:
            raise ValidationError(
                code="tenant_api_key.platform_full_access",
                message="Platform API keys require explicit capabilities",
            )
        if not capabilities:
            raise ValidationError(
                code="tenant_api_key.capabilities_required",
                message="Platform API keys require at least one capability",
            )

    # ── Authentication ──────────────────────────────────────────────

    async def authenticate(self, token: str) -> TenantAPIKeyInfo:
        """Resolve a plaintext token to its key, or raise ``not_found``.

        A revoked or expired key is reported as missing so a caller
        cannot distinguish the two. On success ``last_used_at`` is
        refreshed at most once per minute per key.
        """
        clean_token = token.strip()
        if not clean_token:
            raise self._not_found()
        row = await self._api_keys_repo.find_by_hash(hash_api_key_token(clean_token))
        now = datetime.now(UTC)
        if row.expires_at is not None and now > row.expires_at:
            raise self._not_found()
        if self._should_touch(row.last_used_at, now):
            await self._api_keys_repo.touch_last_used(row.id, used_at=now)
        return TenantAPIKeyInfo.map_from_db(row)

    @staticmethod
    def _should_touch(last_used_at: datetime | None, now: datetime) -> bool:
        """Throttle ``last_used_at`` writes to one per key per interval."""
        return last_used_at is None or now - last_used_at >= _LAST_USED_MIN_INTERVAL

    # ── Listing ─────────────────────────────────────────────────────

    async def list_api_keys(self, tenant_id: int) -> list[TenantAPIKeyInfo]:
        """Live keys of one workspace, newest first."""
        rows = await self._api_keys_repo.list_for_tenant(tenant_id)
        return [TenantAPIKeyInfo.map_from_db(row) for row in rows]

    async def list_platform_api_keys(self) -> list[TenantAPIKeyInfo]:
        """Live platform-scoped keys, newest first."""
        rows = await self._api_keys_repo.list_platform()
        return [TenantAPIKeyInfo.map_from_db(row) for row in rows]

    # ── Revocation ──────────────────────────────────────────────────

    async def revoke_api_key(self, key_id: int, *, tenant_id: int) -> None:
        """Revoke a workspace key; raise ``not_found`` if already gone."""
        await self._api_keys_repo.revoke(
            key_id,
            tenant_id=tenant_id,
            revoked_at=datetime.now(UTC),
        )

    async def revoke_platform_api_key(self, key_id: int) -> None:
        """Revoke a platform key; raise ``not_found`` if already gone."""
        await self._api_keys_repo.revoke_platform(key_id, revoked_at=datetime.now(UTC))

    # ── Maintenance ─────────────────────────────────────────────────

    async def backfill_missing_key_hashes(self) -> int:
        """Replace placeholder hashes with real ones.

        Rows carried over from a per-tenant ``api_key`` column start
        with a placeholder hash; they cannot authenticate until the
        stored token is hashed. Returns how many rows were fixed.
        """
        if not await self._api_keys_repo.has_placeholder_hash():
            return 0
        backfilled = 0
        for row in await self._api_keys_repo.list_with_placeholder_hash():
            token = row.api_key.strip()
            if not token:
                continue
            key_hash = hash_api_key_token(token)
            if key_hash == row.key_hash:
                continue
            await self._api_keys_repo.update_hash(row.id, key_hash=key_hash)
            backfilled += 1
        return backfilled

    # ── Internal helpers ────────────────────────────────────────────

    @staticmethod
    def _not_found() -> NotFoundError:
        return NotFoundError(
            code="tenant_api_key.not_found",
            message="Tenant API key not found",
        )


__all__ = [
    "SCOPE_PLATFORM",
    "SCOPE_TENANT",
    "APIKeyCreateResult",
    "TenantAPIKeyService",
    "generate_api_key_token",
    "hash_api_key_token",
    "normalize_capabilities",
    "normalize_knowledge_base_ids",
    "normalize_scope_type",
]
