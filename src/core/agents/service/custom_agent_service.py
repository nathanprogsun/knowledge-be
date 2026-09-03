# Chinese API messages use fullwidth punctuation.

"""Custom-agent service — CRUD over the ``custom_agents`` table.

Request-scoped: constructed per request by ``factory.build_custom_agent_service``
with a fresh repository on the shared ``AsyncSession``; the web layer
never imports ``db`` directly. Mirrors the upstream custom-agent service
semantics: tenant-isolated CRUD, built-in protection, config defaulting
and validation, and the copy flow.

The config blob is opaque here — the agent's model bindings, tool
allow-list, knowledge-base selection and suggestion policy travel inside
the persisted ``config`` JSONB. The service applies the contract
defaults (quick-answer mode on create, suggestion blocks, retrieval
strategy numbers) and validates the suggestion settings before persist;
typed parsing of the blob happens further down the chat layer.

Deferred seams (neutral wording): the built-in agent registry (a
startup-loaded preset catalog) and the suggested-question generator
(which needs the chunk / wiki / tag / knowledge scopes and the
knowledge-base service). Until the registry is ported the service
recognises built-in ids from a fixed constant so built-in rows can never
be edited or deleted, and every read falls back to the stored row.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from src.common.exception import ConflictError, NotFoundError, ValidationError
from src.common.json import JsonObject, JsonValue
from src.core.agents.builtin_registry import BUILTIN_AGENT_ORDER, get_builtin_agent
from src.core.agents.types import (
    AGENT_MODE_QUICK_ANSWER,
    AGENT_MODE_SMART_REASONING,
    BUILTIN_AGENT_IDS,
    SUGGESTION_CATEGORY_ACTION,
    SUGGESTION_CATEGORY_CLARIFY,
    SUGGESTION_CATEGORY_DEEPEN,
    SUGGESTION_MODE_CURATED,
    SUGGESTION_MODE_GENERATED,
    SUGGESTION_MODE_HYBRID,
    SUGGESTION_MODE_KNOWLEDGE,
    CustomAgentInfo,
)
from src.db.dao.custom_agent_repository import CustomAgentRepository
from src.db.models.custom_agent import CustomAgent

_NOT_FOUND_CODE = "agent.not_found"

# Numeric config fields defaulted when absent or zero, mirroring the
# upstream entity's EnsureDefaults.
_NUMERIC_DEFAULTS: tuple[tuple[str, int | float], ...] = (
    ("max_iterations", 10),
    ("web_search_max_results", 5),
    ("history_turns", 5),
    ("embedding_top_k", 10),
    ("keyword_threshold", 0.3),
    ("vector_threshold", 0.5),
    ("rerank_top_k", 5),
    ("max_completion_tokens", 2048),
)

_STARTER_MODES: tuple[str, ...] = (
    SUGGESTION_MODE_CURATED,
    SUGGESTION_MODE_KNOWLEDGE,
    SUGGESTION_MODE_HYBRID,
)
_FOLLOW_UP_MODES: tuple[str, ...] = (
    SUGGESTION_MODE_GENERATED,
    SUGGESTION_MODE_KNOWLEDGE,
    SUGGESTION_MODE_HYBRID,
)
_SUGGESTION_CATEGORIES: tuple[str, ...] = (
    SUGGESTION_CATEGORY_CLARIFY,
    SUGGESTION_CATEGORY_DEEPEN,
    SUGGESTION_CATEGORY_ACTION,
)

_COPY_NAME_SUFFIX = " （副本）"


class CustomAgentService:
    """Stateless custom-agent service, constructed per request."""

    def __init__(self, *, agent_repo: CustomAgentRepository) -> None:
        self._agent_repo = agent_repo

    # ── Create ──────────────────────────────────────────────────────

    async def create_agent(
        self,
        *,
        tenant_id: int,
        name: str,
        config: JsonObject | None = None,
        description: str | None = None,
        avatar: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> CustomAgentInfo:
        """Insert a new custom agent and return its projected shape.

        Stamps id / tenant / timestamps, forces ``is_builtin`` to false,
        records the creator (skipping synthetic API-key users) and
        applies the contract config defaults (quick-answer mode when the
        config declares none). Suggestion settings are validated before
        the row is persisted.
        """
        _require_tenant_id(tenant_id)
        clean_name = _require_name(name)
        resolved_config = _apply_config_defaults(
            _with_mode_default(config if config is not None else {})
        )
        _validate_config(resolved_config)
        now = _now()
        row = CustomAgent(
            id=agent_id or str(uuid.uuid4()),
            name=clean_name,
            description=description,
            avatar=avatar,
            is_builtin=False,
            tenant_id=tenant_id,
            created_by=_creator_id(user_id),
            config=resolved_config,
            created_at=now,
            updated_at=now,
        )
        persisted = await self._agent_repo.create(row)
        return CustomAgentInfo.from_row(persisted)

    # ── Reads ───────────────────────────────────────────────────────

    async def get_agent_by_id(self, *, tenant_id: int, agent_id: str) -> CustomAgentInfo:
        """Return one agent for the tenant, or raise ``NotFoundError``.

        Built-in ids resolve to the registry default when no customized
        row exists; custom ids resolve to the stored row.
        """
        _require_tenant_id(tenant_id)
        _require_agent_id(agent_id)
        row = await self._agent_repo.get_by_id_and_tenant(id=agent_id, tenant_id=tenant_id)
        if row is not None:
            return CustomAgentInfo.from_row(_ensure_defaults(row))
        builtin = get_builtin_agent(agent_id, tenant_id)
        if builtin is not None:
            return builtin
        raise NotFoundError(
            code=_NOT_FOUND_CODE,
            message=f"custom agent {agent_id} not found",
        )

    async def get_agent_by_id_and_tenant(
        self,
        *,
        agent_id: str,
        tenant_id: int,
    ) -> CustomAgentInfo:
        """Return one agent scoped to an explicit tenant (shared paths).

        Does not resolve built-in presets; used where the caller already
        resolved the ownership scope.
        """
        _require_tenant_id(tenant_id)
        _require_agent_id(agent_id)
        row = await self._get_live_agent(tenant_id=tenant_id, agent_id=agent_id)
        return CustomAgentInfo.from_row(_ensure_defaults(row))

    async def list_agents(self, *, tenant_id: int) -> list[CustomAgentInfo]:
        """Return the tenant's agents: built-in presets first, then custom.

        Built-in presets appear in the fixed registry order; a preset
        with a customized row in storage is replaced by that row. Custom
        agents follow newest-first.
        """
        _require_tenant_id(tenant_id)
        rows = await self._agent_repo.list_by_tenant(tenant_id)
        by_id = {row.id: CustomAgentInfo.from_row(_ensure_defaults(row)) for row in rows}

        result: list[CustomAgentInfo] = []
        for builtin_id in BUILTIN_AGENT_ORDER:
            if builtin_id in by_id:
                result.append(by_id[builtin_id])
            else:
                builtin = get_builtin_agent(builtin_id, tenant_id)
                if builtin is not None:
                    result.append(builtin)
        result.extend(
            info for agent_id, info in by_id.items() if agent_id not in BUILTIN_AGENT_ORDER
        )
        return result

    # ── Update ──────────────────────────────────────────────────────

    async def update_agent(
        self,
        *,
        tenant_id: int,
        agent_id: str,
        name: str | None = None,
        config: JsonObject | None = None,
        description: str | None = None,
        avatar: str | None = None,
    ) -> CustomAgentInfo:
        """Partial-update an agent's mutable fields and return the result.

        Every parameter is optional — ``None`` means "leave the existing
        value alone". This lets the same request shape serve PUT (full
        body) and PATCH (subset). Built-in rows reject basic-info edits;
        supplied ``config`` is re-defaulted and re-validated before
        persist; if ``config`` is omitted the existing config is kept
        verbatim (no defaults reapplied, no validation rerun).
        """
        _require_tenant_id(tenant_id)
        _require_agent_id(agent_id)
        existing = await self._get_live_agent(tenant_id=tenant_id, agent_id=agent_id)
        if existing.is_builtin:
            raise ConflictError(
                code="agent.cannot_modify_builtin",
                message="cannot modify built-in agent basic info",
            )
        patch: dict[str, JsonObject | str | datetime] = {"updated_at": _now()}
        if name is not None:
            patch["name"] = _require_name(name)
        if description is not None:
            patch["description"] = description
        if avatar is not None:
            patch["avatar"] = avatar
        if config is not None:
            resolved_config = _apply_config_defaults(config)
            _validate_config(resolved_config)
            patch["config"] = resolved_config
        updated = existing.model_copy(update=patch)
        persisted = await self._agent_repo.update(updated)
        return CustomAgentInfo.from_row(persisted)

    # ── Delete ──────────────────────────────────────────────────────

    async def delete_agent(self, *, tenant_id: int, agent_id: str) -> None:
        """Soft-delete a custom agent, raising on built-in / missing rows."""
        _require_tenant_id(tenant_id)
        _require_agent_id(agent_id)
        if agent_id in BUILTIN_AGENT_IDS:
            raise ConflictError(
                code="agent.cannot_delete_builtin",
                message="cannot delete built-in agent",
            )
        existing = await self._get_live_agent(tenant_id=tenant_id, agent_id=agent_id)
        if existing.is_builtin:
            raise ConflictError(
                code="agent.cannot_delete_builtin",
                message="cannot delete built-in agent",
            )
        await self._agent_repo.soft_delete(id=agent_id, tenant_id=tenant_id, now=_now())

    # ── Copy ────────────────────────────────────────────────────────

    async def copy_agent(
        self,
        *,
        tenant_id: int,
        agent_id: str,
        user_id: str | None = None,
    ) -> CustomAgentInfo:
        """Create a copy of an existing agent, owned by the caller.

        The clone is never built-in and inherits the source config with
        defaults applied; its creator is whoever ran the copy, not the
        original author.
        """
        _require_tenant_id(tenant_id)
        _require_agent_id(agent_id)
        source = await self.get_agent_by_id(tenant_id=tenant_id, agent_id=agent_id)
        now = _now()
        row = CustomAgent(
            id=str(uuid.uuid4()),
            name=f"{source.name}{_COPY_NAME_SUFFIX}",
            description=source.description,
            avatar=source.avatar,
            is_builtin=False,
            tenant_id=tenant_id,
            created_by=_creator_id(user_id),
            config=_apply_config_defaults(source.config),
            created_at=now,
            updated_at=now,
        )
        persisted = await self._agent_repo.create(row)
        return CustomAgentInfo.from_row(persisted)

    # ── Shared fetch ────────────────────────────────────────────────

    async def _get_live_agent(self, *, tenant_id: int, agent_id: str) -> CustomAgent:
        """Fetch a live row, raising ``NotFoundError`` when absent."""
        row = await self._agent_repo.get_by_id_and_tenant(id=agent_id, tenant_id=tenant_id)
        if row is None:
            raise NotFoundError(
                code=_NOT_FOUND_CODE,
                message=f"custom agent {agent_id} not found",
            )
        return row


# ── Boundary validators ─────────────────────────────────────────────


def _require_tenant_id(tenant_id: int) -> None:
    """Reject a non-positive tenant id."""
    if not isinstance(tenant_id, int) or tenant_id <= 0:
        raise ValidationError(
            code="agent.tenant_required",
            message="tenant ID is required",
        )


def _require_agent_id(agent_id: str) -> None:
    """Reject an empty agent id."""
    if not agent_id.strip():
        raise ValidationError(
            code="agent.id_required",
            message="agent ID cannot be empty",
        )


def _require_name(name: str) -> str:
    """Reject a blank agent name, returning the trimmed value."""
    clean = name.strip()
    if not clean:
        raise ValidationError(
            code="agent.name_required",
            message="agent name is required",
        )
    return clean


def _creator_id(user_id: str | None) -> str | None:
    """Record the creator, skipping synthetic API-key users."""
    if user_id is None or _is_synthetic_user_id(user_id):
        return None
    return user_id


def _is_synthetic_user_id(user_id: str) -> bool:
    """True for the ``system-<digits>`` id of the API-key auth path.

    Such users have no real membership row behind them, so resources
    they create are tenant-owned and carry no ``created_by``.
    """
    prefix = "system-"
    if not user_id.startswith(prefix) or len(user_id) <= len(prefix):
        return False
    suffix = user_id[len(prefix) :]
    return all(char in "0123456789" for char in suffix)


# ── Config defaults ─────────────────────────────────────────────────


def _with_mode_default(config: JsonObject) -> JsonObject:
    """Return a copy of ``config`` whose mode defaults to quick-answer.

    Applied on the create path only — the update path preserves whatever
    mode the caller supplies, mirroring the upstream service.
    """
    out: JsonObject = dict(config)
    if not out.get("agent_mode"):
        out["agent_mode"] = AGENT_MODE_QUICK_ANSWER
    return out


def _apply_config_defaults(config: JsonObject) -> JsonObject:
    """Return a copy of ``config`` with the contract defaults applied.

    Pure function — the input dict is never mutated. Defaults mirror the
    upstream entity: full suggestion block when absent (otherwise the
    suggestion sub-config defaults), negative temperature reset, zero /
    absent numeric strategy fields, the fallback strategy, smart-mode
    multi-turn pin, and the thinking / citation opt-ins.
    """
    out: JsonObject = dict(config)

    suggestions = out.get("question_suggestions")
    if suggestions is None:
        out["question_suggestions"] = _default_question_suggestions()
    elif isinstance(suggestions, dict):
        out["question_suggestions"] = _ensure_suggestion_defaults(suggestions)

    temperature = out.get("temperature")
    if isinstance(temperature, (int, float)) and temperature < 0:
        out["temperature"] = 0.7

    for key, default in _NUMERIC_DEFAULTS:
        value = out.get(key)
        if value is None or value == 0:
            out[key] = default

    fallback = out.get("fallback_strategy")
    if not isinstance(fallback, str) or not fallback:
        out["fallback_strategy"] = "model"

    if out.get("agent_mode") == AGENT_MODE_SMART_REASONING:
        out["multi_turn_enabled"] = True

    if out.get("thinking") is None:
        out["thinking"] = False

    if out.get("citation_enabled") is None:
        out["citation_enabled"] = True

    return out


def _ensure_defaults(row: CustomAgent) -> CustomAgent:
    """Return ``row`` with config defaults applied (no mutation)."""
    config = _apply_config_defaults(row.config)
    if config == row.config:
        return row
    return row.model_copy(update={"config": config})


def _default_question_suggestions() -> JsonObject:
    """Build the default suggestion policy for a fresh agent."""
    return {
        "starters": {
            "enabled": True,
            "mode": SUGGESTION_MODE_HYBRID,
            "items": [],
            "count": 6,
        },
        "follow_ups": {
            "enabled": False,
            "mode": SUGGESTION_MODE_HYBRID,
            "count": 3,
            "max_context_turns": 2,
            "suppress_on_fallback": True,
            "suppress_when_answer_asks_question": True,
            "knowledge_fallback": True,
            "categories": [
                SUGGESTION_CATEGORY_CLARIFY,
                SUGGESTION_CATEGORY_DEEPEN,
                SUGGESTION_CATEGORY_ACTION,
            ],
        },
    }


def _ensure_suggestion_defaults(suggestions: JsonObject) -> JsonObject:
    """Apply the suggestion sub-config defaults to a copy of ``suggestions``."""
    out: JsonObject = dict(suggestions)

    starters_raw = out.get("starters")
    starters: JsonObject = dict(starters_raw) if isinstance(starters_raw, dict) else {}
    if not starters.get("mode"):
        starters["mode"] = SUGGESTION_MODE_HYBRID
    if starters.get("count") is None or starters.get("count") == 0:
        starters["count"] = 6
    if starters.get("items") is None:
        starters["items"] = []
    out["starters"] = starters

    follow_raw = out.get("follow_ups")
    follow: JsonObject = dict(follow_raw) if isinstance(follow_raw, dict) else {}
    if not follow.get("mode"):
        follow["mode"] = SUGGESTION_MODE_HYBRID
    if follow.get("count") is None or follow.get("count") == 0:
        follow["count"] = 3
    if follow.get("max_context_turns") is None or follow.get("max_context_turns") == 0:
        follow["max_context_turns"] = 2
    if not follow.get("categories"):
        follow["categories"] = [
            SUGGESTION_CATEGORY_CLARIFY,
            SUGGESTION_CATEGORY_DEEPEN,
            SUGGESTION_CATEGORY_ACTION,
        ]
    out["follow_ups"] = follow

    return out


# ── Suggestion validation ───────────────────────────────────────────


def _validate_config(config: JsonObject) -> None:
    """Reject invalid agent-authored suggestion settings before persist."""
    suggestions = config.get("question_suggestions")
    if not isinstance(suggestions, dict):
        return

    starters = suggestions.get("starters")
    if isinstance(starters, dict):
        _validate_suggestion_count(starters.get("count"), "starter suggestion", 1, 8)
        _validate_suggestion_mode(starters.get("mode"), "starter suggestion", _STARTER_MODES)
        items = starters.get("items")
        if isinstance(items, list):
            for index, item in enumerate(items, start=1):
                _validate_starter_item(item, index)

    follow_ups = suggestions.get("follow_ups")
    if isinstance(follow_ups, dict):
        _validate_suggestion_count(follow_ups.get("count"), "follow-up suggestion", 1, 5)
        _validate_suggestion_count(
            follow_ups.get("max_context_turns"),
            "follow-up max_context_turns",
            1,
            5,
        )
        _validate_suggestion_mode(follow_ups.get("mode"), "follow-up suggestion", _FOLLOW_UP_MODES)
        instruction = follow_ups.get("additional_instruction")
        if isinstance(instruction, str) and len(instruction.strip()) > 2000:
            raise ValidationError(
                code="agent.suggestion_instruction_too_long",
                message="follow-up additional_instruction exceeds 2000 characters",
            )
        categories = follow_ups.get("categories")
        if isinstance(categories, list):
            for category in categories:
                if not isinstance(category, str) or category not in _SUGGESTION_CATEGORIES:
                    raise ValidationError(
                        code="agent.suggestion_category_invalid",
                        message=f"invalid follow-up suggestion category {category!r}",
                    )


def _validate_suggestion_count(value: JsonValue, label: str, low: int, high: int) -> None:
    """Reject a suggestion count outside its allowed range."""
    if not isinstance(value, int) or isinstance(value, bool) or not (low <= value <= high):
        raise ValidationError(
            code="agent.suggestion_count_invalid",
            message=f"{label} count must be between {low} and {high}",
        )


def _validate_suggestion_mode(value: JsonValue, label: str, allowed: tuple[str, ...]) -> None:
    """Reject a suggestion mode outside its allowed set."""
    if not isinstance(value, str) or value not in allowed:
        raise ValidationError(
            code="agent.suggestion_mode_invalid",
            message=f"invalid {label} mode {value!r}",
        )


def _validate_starter_item(item: JsonValue, index: int) -> None:
    """Reject a blank or over-long curated starter prompt."""
    if not isinstance(item, str):
        raise ValidationError(
            code="agent.suggestion_item_invalid",
            message=f"starter suggestion {index} must be a string",
        )
    trimmed = item.strip()
    if not trimmed:
        raise ValidationError(
            code="agent.suggestion_item_empty",
            message=f"starter suggestion {index} cannot be empty",
        )
    if len(trimmed) > 200:
        raise ValidationError(
            code="agent.suggestion_item_too_long",
            message=f"starter suggestion {index} exceeds 200 characters",
        )


def _now() -> datetime:
    """Return a timezone-aware ``now`` for stamping rows."""
    return datetime.now(UTC)


__all__ = ["CustomAgentService"]
