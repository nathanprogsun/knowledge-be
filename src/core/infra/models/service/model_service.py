"""ModelService — CRUD for tenant-scoped AI model configurations.

Mirrors ``internal/application/service/model.go`` on the Go side for
the basic CRUD surface. Inference / debug paths (the
``GetEmbeddingModel`` / ``GetChatModel`` / ``DebugModel`` family)
are deferred until the inference providers land; the service stays
focused on persistence + tenant scoping.

Constructed per request; the repository owns the per-request session.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import cast

from src.common.exception import NotFoundError, ValidationError
from src.common.json import BindParams, JsonObject
from src.core.contracts.infra import (
    CreateModelRequest,
    ModelParameters,
    UpdateModelRequest,
)
from src.core.infra.models.types import ModelInfo
from src.db.dao.model_repository import ModelRepository
from src.db.models.infra.model import Model

# Mirrors WeKnora's ``types.ModelStatusActive`` constant.
_STATUS_ACTIVE = "active"


def _now() -> datetime:
    """Return a timezone-aware ``now`` for stamping rows."""
    return datetime.now(UTC)


def _new_id() -> str:
    """Generate a UUID for a freshly created model row."""
    return str(uuid.uuid4())


def _coerce_parameters(raw: JsonObject | ModelParameters | None) -> JsonObject:
    """Coerce a parameters input to a ``JsonObject`` for the JSONB column.

    Accepts the wire-shape ``ModelParameters`` (round-trips as JSON)
    or a plain ``dict`` (already on the storage form). ``None``
    becomes an empty dict so the column never carries ``null``.
    """
    if raw is None:
        return {}
    if isinstance(raw, ModelParameters):
        # ``mode="json"`` excludes the None-valued fields so the stored
        # shape matches Go's `omitempty` JSON tags on
        # ``ModelParameters`` (the Go wire format drops empty
        # collections as well).
        dumped = raw.model_dump(mode="json", exclude_none=True)
        # ``extra_config`` / ``custom_headers`` are dicts whose empty
        # form is a valid storage value; leave them as-is. ``None``
        # values have already been excluded above.
        return cast("JsonObject", dumped)
    if isinstance(raw, dict):
        return raw
    raise ValidationError(
        code="model.parameters_invalid",
        message="parameters must be an object or a ModelParameters instance",
    )


class ModelService:
    """Stateless model service, constructed per request."""

    def __init__(self, *, models_repo: ModelRepository) -> None:
        self._models_repo = models_repo

    # ── Create ──────────────────────────────────────────────────────

    async def create_model(
        self,
        *,
        tenant_id: int,
        body: CreateModelRequest,
        model_id: str | None = None,
    ) -> ModelInfo:
        """Insert a new model row.

        Caller can supply ``model_id`` (the built-in loader does this
        to keep a stable id); otherwise a UUID is generated. ``status``
        defaults to ``active`` — both local and remote models are
        immediately queryable; the local Ollama download path lives on
        the Go side and is out of scope here.
        """
        if tenant_id <= 0:
            raise ValidationError(
                code="model.invalid_tenant_id",
                message="Tenant ID must be positive",
            )
        clean_name = body.name.strip()
        if not clean_name:
            raise ValidationError(
                code="model.name_required",
                message="Model name cannot be empty",
            )
        now = _now()
        parameters = _coerce_parameters(body.parameters)
        row = Model(
            id=model_id or _new_id(),
            tenant_id=tenant_id,
            name=clean_name,
            display_name=body.name,
            type=body.type,
            source=body.source,
            description=body.description,
            parameters=parameters,
            is_default=False,
            is_builtin=False,
            managed_by="",
            status=_STATUS_ACTIVE,
            created_at=now,
            updated_at=now,
        )
        return ModelInfo.map_from_db(await self._models_repo.insert(row))

    # ── Read ────────────────────────────────────────────────────────

    async def get_model(self, *, tenant_id: int, model_id: str) -> ModelInfo:
        """Return one model visible to ``tenant_id``; ``model.not_found`` if absent."""
        if tenant_id <= 0:
            raise ValidationError(
                code="model.invalid_tenant_id",
                message="Tenant ID must be positive",
            )
        return ModelInfo.map_from_db(
            await self._models_repo.find_by_tenant_and_id_or_fail(
                tenant_id=tenant_id,
                id=model_id,
            )
        )

    async def list_models(
        self,
        *,
        tenant_id: int,
        model_type: str | None = None,
        source: str | None = None,
        include_builtin: bool = True,
    ) -> list[ModelInfo]:
        """List every model visible to ``tenant_id``."""
        if tenant_id <= 0:
            raise ValidationError(
                code="model.invalid_tenant_id",
                message="Tenant ID must be positive",
            )
        rows = await self._models_repo.list_by_tenant(
            tenant_id=tenant_id,
            model_type=model_type or None,
            source=source or None,
            include_builtin=include_builtin,
        )
        return [ModelInfo.map_from_db(row) for row in rows]

    # ── Update ──────────────────────────────────────────────────────

    async def update_model(
        self,
        *,
        tenant_id: int,
        model_id: str,
        body: UpdateModelRequest,
        is_system_admin: bool = False,
    ) -> ModelInfo:
        """Patch an existing model row; built-ins require a system admin."""
        if tenant_id <= 0:
            raise ValidationError(
                code="model.invalid_tenant_id",
                message="Tenant ID must be positive",
            )
        existing = await self._models_repo.find_by_tenant_and_id_or_fail(
            tenant_id=tenant_id,
            id=model_id,
        )
        if existing.is_builtin and not is_system_admin:
            raise ValidationError(
                code="model.builtin_protected",
                message="Only system administrators can update builtin models",
            )
        updates = self._build_update_columns(
            existing=existing,
            body=body,
        )
        updates["updated_at"] = _now()
        if existing.is_builtin:
            # UI edits are runtime overrides of YAML-managed data — clear
            # ownership so the reconciler doesn't silently restore the
            # YAML form on the next boot.
            updates["managed_by"] = ""
        row = await self._models_repo.update_row(
            Model.model_validate(
                {
                    **existing.model_dump(),
                    **updates,
                }
            )
        )
        if row is None:
            raise NotFoundError(
                code="model.not_found",
                message=f"Model {model_id} not found",
            )
        return ModelInfo.map_from_db(row)

    @staticmethod
    def _build_update_columns(
        *,
        existing: Model,
        body: UpdateModelRequest,
    ) -> BindParams:
        """Collect the supplied columns, never allowing credential edits.

        Credentials (``api_key`` / ``app_secret``) never flow through
        this endpoint — they live behind the credentials subresource.
        The service rejects a request body that tries to mutate them
        so a stale caller cannot clobber a stored secret.
        """
        columns: BindParams = {}
        if body.name is not None:
            clean_name = body.name.strip()
            if not clean_name:
                raise ValidationError(
                    code="model.name_required",
                    message="Model name cannot be empty",
                )
            columns["name"] = clean_name
        if body.description is not None:
            columns["description"] = body.description
        if body.type is not None:
            columns["type"] = body.type
        if body.source is not None:
            columns["source"] = body.source
        if body.parameters is not None:
            existing_params = dict(existing.parameters)
            new_params = _coerce_parameters(body.parameters)
            # Preserve stored credentials across the update.
            for key in ("api_key", "app_secret"):
                if key in existing_params:
                    new_params[key] = existing_params[key]
            # Preserve any stored parameter key the caller did not send
            # — Go's `omitempty` contract drops *empty* fields, not
            # caller-omitted ones, so the merged blob keeps
            # ``base_url`` / ``embedding_parameters`` / etc.
            for key, value in existing_params.items():
                if key not in new_params:
                    new_params[key] = value
            columns["parameters"] = new_params
        return columns

    # ── Delete ──────────────────────────────────────────────────────

    async def delete_model(
        self,
        *,
        tenant_id: int,
        model_id: str,
    ) -> bool:
        """Hard-delete a non-built-in model row.

        Returns ``True`` when a row was deleted, ``False`` when no row
        matched (idempotent for unknown ids). The Go service surfaces
        a separate ``model.not_found`` error for absent ids; we keep
        idempotent semantics here so the view layer can render a
        uniform 200 / 204.
        """
        if tenant_id <= 0:
            raise ValidationError(
                code="model.invalid_tenant_id",
                message="Tenant ID must be positive",
            )
        # Look up with builtins included so we can surface the
        # builtin-specific error instead of a generic "not found".
        existing = await self._models_repo.find_by_tenant_and_id_or_fail(
            tenant_id=tenant_id,
            id=model_id,
            include_builtin=True,
        )
        if existing.is_builtin:
            raise ValidationError(
                code="model.builtin_protected",
                message="Builtin models cannot be deleted",
            )
        rows = await self._models_repo.delete_by_tenant_and_id(
            tenant_id=tenant_id,
            id=model_id,
        )
        return rows > 0


__all__ = ["ModelService"]
