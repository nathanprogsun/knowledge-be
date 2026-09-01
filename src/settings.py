"""Application settings — single source of truth for runtime configuration.

Loaded from environment variables and an optional `.env` file (no prefix;
the canonical variable names match the keys exactly). Read via
`get_settings()` which memoizes a process-wide singleton via
`functools.lru_cache` (no module-level mutable globals).

Database URL composition: when `DATABASE_URL_OVERRIDE` is not set,
`database_url` is auto-built from the `DB_*` component variables
(`DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME`,
`DB_DRIVER`). The password component is URL-encoded so values
containing reserved characters (`!`, `@`, `#`, …) round-trip safely.
"""

from __future__ import annotations

from functools import lru_cache
from urllib.parse import quote

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "knowledge-be"
    environment: str = "development"

    # Database connection components — consumed by `database_url` below.
    db_user: str = "postgres"
    db_password: str = "postgres"
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "knowledge_be"
    db_driver: str = "postgresql+asyncpg"

    # Optional explicit override for `database_url`. When set, the
    # computed URL below falls back to this value verbatim.
    database_url_override: str | None = None

    redis_url: str = "redis://localhost:6379"

    jwt_secret_key: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 10080
    refresh_token_expire_days: int = 7

    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_echo: bool = False
    db_conn_prewarm: bool = True

    cors_allow_origins: list[str] = ["*"]
    default_main_thread_pool_size: int = 40

    # ── Observability (OpenTelemetry tracing) ──────────────────────────
    # Disabled by default so local dev and tests carry zero OTel overhead.
    # When enabled without an OTLP endpoint, spans go to the console
    # exporter (useful for local verification); with an endpoint they are
    # batched to the collector over OTLP/HTTP.
    otel_enabled: bool = False
    otel_service_name: str = "knowledge-be"
    otel_exporter_otlp_endpoint: str | None = None
    # ``file`` writes spans as newline-delimited JSON to ``otel_file_dir``
    # so an agent can ``jq`` / grep a trace locally without a collector.
    otel_exporter: str = "console"  # one of: console | otlp | file
    otel_file_dir: str = "traces"

    docreader_addr: str = "localhost:50051"
    docreader_transport: str = "grpc"

    system_aes_key: str = ""

    # ── Tenant RBAC ────────────────────────────────────────────────────
    # Rollout switches. When ``rbac_enforced`` is false the role guards
    # log but do not reject (preserves today's behaviour during
    # rollout); ``cross_tenant_access_enabled`` gates the
    # cross-workspace superuser endpoints (/tenants/all, /search).
    rbac_enforced: bool = True
    cross_tenant_access_enabled: bool = False

    # ── OIDC SSO ──────────────────────────────────────────────────────
    # Endpoints may be left blank when ``oidc_discovery_url`` is set -
    # the OIDC client fills them from the discovery document at request
    # time.
    oidc_enable: bool = False
    oidc_issuer_url: str = ""
    oidc_discovery_url: str = ""
    oidc_provider_display_name: str = ""
    oidc_client_id: str = ""
    oidc_client_secret: str = ""
    oidc_authorization_endpoint: str = ""
    oidc_token_endpoint: str = ""
    oidc_user_info_endpoint: str = ""
    oidc_scopes: list[str] = ["openid", "profile", "email"]
    # Optional host whitelist for the OIDC callback redirect URI; empty
    # defers to the provider's registered-URI enforcement.
    oidc_redirect_allowed_hosts: list[str] = []
    # Claim keys to read username/email from the userinfo / id_token.
    oidc_user_info_mapping_username: str = "name"
    oidc_user_info_mapping_email: str = "email"

    # ── Auth header names ─────────────────────────────────────────────
    # HTTP header names used by the auth layer when a request does not
    # carry a JWT or API key (for example, internal service-to-service
    # calls that authenticate via headers rather than bearer tokens).
    # Defaults mirror the upstream contract (``X-Tenant-ID``,
    # ``X-User-Id``, ``X-Roles``).
    #
    # ``header_auth_enabled`` is the deploy-time kill switch for that
    # channel: the headers assert identity and roles verbatim, so the
    # channel must stay off unless a trusted gateway strips/overwrites
    # them for all inbound traffic.
    header_auth_enabled: bool = False
    # ``auto_setup_enabled`` gates the unauthenticated /auth/auto-setup
    # bootstrap that mints a token pair for the default account; off by
    # default so a public deployment cannot be anonymously signed in.
    auto_setup_enabled: bool = False
    auth_header_user_id: str = "X-User-Id"
    auth_header_tenant_id: str = "X-Tenant-ID"
    auth_header_roles: str = "X-Roles"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> str:
        """Composed asyncpg URL from DB_* components (or DATABASE_URL_OVERRIDE)."""
        if self.database_url_override is not None:
            return self.database_url_override
        user = quote(self.db_user, safe="")
        password = quote(self.db_password, safe="")
        return f"{self.db_driver}://{user}:{password}@{self.db_host}:{self.db_port}/{self.db_name}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings singleton.

    Lazy-initialized on first call and cached via `lru_cache`. Tests can
    reset with `reset_settings_cache()`.
    """
    return Settings()


def reset_settings_cache() -> None:
    """Drop the memoized singleton. Used by tests that mutate env vars."""
    get_settings.cache_clear()
