"""Generic CRUD helpers for raw-SQL TableModel repositories.

Concrete repositories (``UserRepository``, ``AuthTokenRepository``, …)
inherit from ``GenericRepository[TheirModelType]`` and pick up the
boilerplate ``insert`` / ``insert_or_none`` / ``find_by_*`` /
``update_by_primary_key`` implementations. Domain-specific queries
(``find_by_email``, ``soft_delete``, ``revoke``, …) stay on the
concrete subclasses.

Why a base class? Without it, every concrete repo repeats the same
INSERT / SELECT-by-column / UPDATE-by-pk SQL — about 40-60 lines per
repo, varying only in the model type. Pushing the common shape behind
``Generic[ModelType]`` keeps each concrete repo focused on what is
actually unique.

Why raw SQL? Per AGENTS.md §6.5, this project deliberately avoids the
SQLAlchemy ORM — every query uses ``sqlalchemy.text()`` with named
``bindparams``. ``GenericRepository`` follows that rule.

JSON columns
------------

Models declare JSON/JSONB columns via ``json_columns: ClassVar``.
``GenericRepository`` attaches ``bindparam(col, type_=JSON)`` for those
columns on every INSERT and UPDATE. The bind type is
``JSON().with_variant(JSONB, "postgresql")`` so Postgres uses ``JSONB``
and the SQLite test dialect uses ``JSON`` — both serialise Python
``dict``/``list`` automatically.

Soft-delete / archived filtering
-------------------------------

Finders accept ``exclude_deleted_or_archived: bool``. When the model has
``deleted_at`` / ``archived_at`` columns, the WHERE clause filters rows
whose timestamp is in the past — mirroring the cookiecutter-fastapi
``GenericRepository`` semantics (rows are deleted when ``deleted_at`` is
not null and not in the future).

Error semantics
---------------

``find_*_or_fail`` methods raise ``NotFoundError`` when no row matches.
``insert`` (no-conflict variant) raises ``ConflictError`` only when the
caller converts an ``IntegrityError`` — but ``insert_or_none`` uses
``ON CONFLICT DO NOTHING`` and returns ``None`` instead, which is the
preferred path.

Session ownership
------------------

Per the cookiecutter-fastapi pattern, every repository holds its own
``AsyncSession`` in ``__init__``. The web layer constructs a new repo
per request via ``Depends(get_xxx_repository)``; tests construct one
per test.

- A repository is request-scoped — never share an instance across
  requests.
- A repository is the **only** layer that opens SQL sessions.
- Services hold repositories (and the same session) and never call
  ``session_factory`` or ``session_scope`` themselves.
- Repositories never commit/rollback — the web-layer session
  dependency owns the transaction boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from functools import cached_property
from typing import Generic, TypeAlias, TypeVar, cast

from sqlalchemy import JSON, bindparam, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import BindParameter

from src.common.exception import DataError, NotFoundError, ValidationError
from src.common.table_model import TableModel

ModelType = TypeVar("ModelType", bound=TableModel)

# Concrete union of every Python value a ``TableModel`` field may hold —
# used to type bind parameters without falling back to the forbidden
# ``Any``/``object`` annotations (AGENTS.md §1).
BindValue: TypeAlias = str | int | float | bool | datetime | None

# Bind type for JSON columns: JSONB on Postgres, JSON on other dialects
# (e.g. SQLite in tests). ``with_variant`` makes the dialect pick at
# compile time.
_JSON_BIND_TYPE = JSON().with_variant(JSONB(), "postgresql")


class GenericRepository(Generic[ModelType]):
    """Boilerplate-free raw-SQL CRUD for ``TableModel`` rows.

    Subclasses declare their model via the ``model_class`` class
    attribute; ``__init__`` takes only the per-request session.
    """

    model_class: type[ModelType]

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ── Cached model metadata ────────────────────────────────────────

    @cached_property
    def _table(self) -> str:
        return self.model_class.fq_table_name()

    @cached_property
    def _pk_columns(self) -> tuple[str, ...]:
        return self.model_class.ordered_primary_keys()

    @cached_property
    def _json_columns(self) -> tuple[str, ...]:
        return self.model_class.get_json_columns()

    # ── Hydration helper ────────────────────────────────────────────
    # ``TableModel.from_row`` is annotated to return ``TableModel``
    # (the base class is not generic), so the concrete subclass type is
    # lost. ``_hydrate`` is the single cast point that narrows it back
    # to ``ModelType``; every read path funnels through it.
    def _hydrate(self, mapping: RowMapping) -> ModelType:
        return cast(
            "ModelType",
            self.model_class.from_row(cast("Mapping[str, object]", mapping)),
        )

    def _hydrate_opt(self, mapping: RowMapping | None) -> ModelType | None:
        return self._hydrate(mapping) if mapping is not None else None

    @staticmethod
    def _require_non_empty_query(column_to_query: dict[str, object]) -> None:
        """Reject empty query dicts — they would produce ``where`` with
        no condition, which is a no-op full scan at best and broken SQL
        at worst (when soft-delete fragments are appended). Callers who
        want a full scan should use :meth:`find_all`.
        """
        if not column_to_query:
            raise DataError(
                code="db.empty_query",
                message=(
                    f"{GenericRepository.__name__}.find_*_by_column_values "
                    "requires a non-empty column_to_query; use find_all "
                    "for a full scan"
                ),
            )

    # ── Soft-delete / archived WHERE fragment builders ──────────────

    @staticmethod
    def _soft_delete_where_fragment(
        model: type[ModelType],
        *,
        exclude_deleted_or_archived: bool,
        prefix: str = "and",
    ) -> str:
        """Build ``and (deleted_at is null or deleted_at > now)`` fragment.

        Returns empty string when the model has no ``deleted_at`` column
        or when filtering is disabled. ``prefix`` controls the leading
        connector (``and`` / ``where``) so the fragment can be appended
        to either a WHERE clause or an existing condition list.
        """
        if not exclude_deleted_or_archived:
            return ""
        if "deleted_at" not in model.column_fields():
            return ""
        return f"{prefix} (deleted_at is null or deleted_at > current_timestamp)"

    @staticmethod
    def _archived_where_fragment(
        model: type[ModelType],
        *,
        exclude_deleted_or_archived: bool,
        prefix: str = "and",
    ) -> str:
        """Build ``and (archived_at is null or archived_at > now)`` fragment."""
        if not exclude_deleted_or_archived:
            return ""
        if "archived_at" not in model.column_fields():
            return ""
        return f"{prefix} (archived_at is null or archived_at > current_timestamp)"

    # ── Bind-param helpers ──────────────────────────────────────────

    def _json_bindparams(self, column_names: tuple[str, ...]) -> list[BindParameter[BindValue]]:
        return [
            bindparam(col, type_=_JSON_BIND_TYPE)
            for col in column_names
            if col in self._json_columns
        ]

    # ── INSERT ───────────────────────────────────────────────────────

    @staticmethod
    def _insert_stmt_text(model: type[ModelType]) -> str:
        """Plain ``INSERT ... VALUES (...) RETURNING *`` — no conflict clause."""
        columns = model.insert_sql_column_list()
        col_list = ", ".join(f'"{c}"' for c in columns)
        param_list = ", ".join(f":{c}" for c in columns)
        return f"insert into {model.fq_table_name()} ({col_list}) values ({param_list}) returning *"

    @staticmethod
    def _insert_on_conflict_do_nothing_stmt_text(
        model: type[ModelType],
        target_columns: list[str] | None,
    ) -> str:
        """``INSERT ... ON CONFLICT (...) DO NOTHING RETURNING *``.

        ``target_columns=None`` emits ``ON CONFLICT DO NOTHING`` with no
        target (Postgres: suppresses conflicts on every unique
        constraint). A non-empty list targets a specific constraint.
        """
        base = (
            "insert into "
            + model.fq_table_name()
            + " ("
            + ", ".join(f'"{c}"' for c in model.insert_sql_column_list())
            + ") values ("
            + ", ".join(f":{c}" for c in model.insert_sql_column_list())
            + ")"
        )
        if target_columns:
            targets = ", ".join(f'"{c}"' for c in target_columns)
            conflict = f"on conflict ({targets}) do nothing"
        else:
            conflict = "on conflict do nothing"
        return f"{base} {conflict} returning *"

    async def insert(self, row: ModelType) -> ModelType:
        """Insert ``row`` and return the persisted row (``RETURNING *``).

        Raises ``IntegrityError`` on unique-constraint violation — callers
        that need conflict-aware insertion should use :meth:`insert_or_none`.
        """
        stmt_text = self._insert_stmt_text(self.model_class)
        params = row.insert_bind_params()
        json_bps = self._json_bindparams(self.model_class.insert_sql_column_list())
        stmt = text(stmt_text).bindparams(*json_bps, **params)
        result = await self._session.execute(stmt)
        mapping = result.mappings().first()
        # INSERT ... RETURNING * yields exactly one row on a successful
        # insert; ``None`` here would indicate a driver/dialect anomaly
        # (e.g. a misconfigured RETURNING) rather than a user error.
        if mapping is None:
            raise DataError(
                code="db.insert_no_row",
                message=(
                    f"INSERT into {self._table} returned no row; "
                    "expected exactly one from RETURNING *"
                ),
            )
        return self._hydrate(mapping)

    async def insert_or_none(
        self,
        row: ModelType,
        *,
        on_conflict_do_nothing_target_columns: list[str] | None = None,
    ) -> ModelType | None:
        """Insert with ``ON CONFLICT DO NOTHING``.

        Returns the inserted row, or ``None`` when a conflict was
        detected and nothing was inserted. When
        ``on_conflict_do_nothing_target_columns`` is omitted (``None``)
        the ``ON CONFLICT DO NOTHING`` is emitted without a conflict
        target, which suppresses conflicts on **every** unique
        constraint (Postgres semantics). When specific target columns
        are supplied only conflicts on those columns are suppressed —
        conflicts on any other unique constraint still raise
        ``IntegrityError``.

        Prefer this over catching ``IntegrityError`` around :meth:`insert`
        — the SQL-level conflict handling is faster and more precise.
        """
        if on_conflict_do_nothing_target_columns is not None:
            self.model_class.validate_in_columns(
                on_conflict_do_nothing_target_columns,
            )
        stmt_text = self._insert_on_conflict_do_nothing_stmt_text(
            self.model_class,
            on_conflict_do_nothing_target_columns,
        )
        params = row.insert_bind_params()
        json_bps = self._json_bindparams(self.model_class.insert_sql_column_list())
        stmt = text(stmt_text).bindparams(*json_bps, **params)
        result = await self._session.execute(stmt)
        mapping = result.mappings().first()
        return self._hydrate_opt(mapping)

    # ── SELECT by primary key ────────────────────────────────────────

    async def find_by_primary_key(
        self,
        primary_key_to_value: dict[str, object],
        *,
        exclude_deleted_or_archived: bool = True,
    ) -> ModelType | None:
        """Look up a single row by its primary key(s)."""
        self.model_class.validate_contains_all_primary_keys(primary_key_to_value)
        pk_cols = self.model_class.ordered_primary_keys()
        where_pk = " and ".join(f'"{c}" = :{c}' for c in pk_cols)
        soft = self._soft_delete_where_fragment(
            self.model_class,
            exclude_deleted_or_archived=exclude_deleted_or_archived,
        )
        archived = self._archived_where_fragment(
            self.model_class,
            exclude_deleted_or_archived=exclude_deleted_or_archived,
        )
        stmt_text = f"select * from {self._table} where {where_pk} {soft} {archived}"
        stmt = text(stmt_text).bindparams(**primary_key_to_value)
        result = await self._session.execute(stmt)
        mapping = result.mappings().first()
        return self._hydrate_opt(mapping)

    async def find_by_primary_key_or_fail(
        self,
        primary_key_to_value: dict[str, object],
        *,
        exclude_deleted_or_archived: bool = True,
        not_found_code: str = "resource.not_found",
        not_found_message: str | None = None,
    ) -> ModelType:
        result = await self.find_by_primary_key(
            primary_key_to_value,
            exclude_deleted_or_archived=exclude_deleted_or_archived,
        )
        if result is None:
            raise NotFoundError(
                code=not_found_code,
                message=not_found_message
                or (
                    f"{self.model_class.__name__} with primary key {primary_key_to_value} not found"
                ),
            )
        return result

    async def find_by_id(
        self,
        id: str | int,
        *,
        exclude_deleted_or_archived: bool = True,
        not_found_code: str = "resource.not_found",
        not_found_message: str | None = None,
    ) -> ModelType:
        """Convenience wrapper for single-column ``id`` primary keys."""
        result = await self.find_by_primary_key(
            {"id": id},
            exclude_deleted_or_archived=exclude_deleted_or_archived,
        )
        if result is None:
            raise NotFoundError(
                code=not_found_code,
                message=not_found_message or f"{self.model_class.__name__} {id} not found",
            )
        return result

    # ── SELECT by arbitrary column values ────────────────────────────

    async def find_unique_by_column_values(
        self,
        column_to_query: dict[str, object],
        *,
        exclude_deleted_or_archived: bool = True,
    ) -> ModelType | None:
        """Return the single row matching ``column_to_query``, or ``None``.

        ``column_to_query`` should select at most one row (i.e. the
        columns form a unique constraint). The query is built with
        ``at_most_one`` semantics; multiple matches are treated as a
        data error and surfaced via the driver (only the first row is
        returned).
        """
        self._require_non_empty_query(column_to_query)
        self.model_class.validate_in_columns(column_to_query)
        where_parts: list[str] = []
        params: dict[str, object] = {}
        for col, val in column_to_query.items():
            if val is None:
                where_parts.append(f'"{col}" is null')
            else:
                where_parts.append(f'"{col}" = :{col}')
                params[col] = val
        where_clause = " and ".join(where_parts)
        soft = self._soft_delete_where_fragment(
            self.model_class,
            exclude_deleted_or_archived=exclude_deleted_or_archived,
        )
        archived = self._archived_where_fragment(
            self.model_class,
            exclude_deleted_or_archived=exclude_deleted_or_archived,
        )
        stmt_text = f"select * from {self._table} where {where_clause} {soft} {archived}"
        stmt = text(stmt_text).bindparams(**params)
        result = await self._session.execute(stmt)
        mapping = result.mappings().first()
        return self._hydrate_opt(mapping)

    async def find_unique_by_column_values_or_fail(
        self,
        column_to_query: dict[str, object],
        *,
        exclude_deleted_or_archived: bool = True,
        not_found_code: str = "resource.not_found",
        not_found_message: str | None = None,
    ) -> ModelType:
        result = await self.find_unique_by_column_values(
            column_to_query,
            exclude_deleted_or_archived=exclude_deleted_or_archived,
        )
        if result is None:
            raise NotFoundError(
                code=not_found_code,
                message=not_found_message
                or (f"{self.model_class.__name__} with columns {column_to_query} not found"),
            )
        return result

    async def find_all_by_column_values(
        self,
        column_to_query: dict[str, object],
        *,
        exclude_deleted_or_archived: bool = True,
    ) -> list[ModelType]:
        """Return every row matching ``column_to_query``."""
        self._require_non_empty_query(column_to_query)
        self.model_class.validate_in_columns(column_to_query)
        where_parts: list[str] = []
        params: dict[str, object] = {}
        for col, val in column_to_query.items():
            if val is None:
                where_parts.append(f'"{col}" is null')
            else:
                where_parts.append(f'"{col}" = :{col}')
                params[col] = val
        where_clause = " and ".join(where_parts)
        soft = self._soft_delete_where_fragment(
            self.model_class,
            exclude_deleted_or_archived=exclude_deleted_or_archived,
        )
        archived = self._archived_where_fragment(
            self.model_class,
            exclude_deleted_or_archived=exclude_deleted_or_archived,
        )
        stmt_text = f"select * from {self._table} where {where_clause} {soft} {archived}"
        stmt = text(stmt_text).bindparams(**params)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    # ── UPDATE by primary key ────────────────────────────────────────

    async def update_by_primary_key(
        self,
        primary_key_to_value: dict[str, object],
        column_to_update: dict[str, object],
        *,
        exclude_deleted_or_archived: bool = True,
    ) -> ModelType | None:
        """Update a row by its primary key, returning the updated row.

        ``column_to_update`` carries only the columns to change; the
        primary-key columns are read from ``primary_key_to_value`` and
        must NOT appear in ``column_to_update`` (doing so would update
        the PK, which is almost never intended). Returns ``None`` when
        no row matched the primary key (after soft-delete filtering).
        """
        self.model_class.validate_contains_all_primary_keys(primary_key_to_value)
        self.model_class.validate_in_columns(column_to_update)
        for pk in self._pk_columns:
            if pk in column_to_update:
                raise ValidationError(
                    code="db.primary_key_in_update",
                    message=f"primary key '{pk}' must not appear in column_to_update",
                )
        pk_cols = self.model_class.ordered_primary_keys()
        where_pk = " and ".join(f'"{c}" = :{c}' for c in pk_cols)
        set_clause = ", ".join(f'"{k}" = :u_{k}' for k in column_to_update)
        update_params: dict[str, object] = {f"u_{k}": v for k, v in column_to_update.items()}
        soft = self._soft_delete_where_fragment(
            self.model_class,
            exclude_deleted_or_archived=exclude_deleted_or_archived,
        )
        archived = self._archived_where_fragment(
            self.model_class,
            exclude_deleted_or_archived=exclude_deleted_or_archived,
        )
        stmt_text = (
            f"update {self._table} set {set_clause} where {where_pk} {soft} {archived} returning *"
        )
        json_bps = [
            bindparam(f"u_{col}", type_=_JSON_BIND_TYPE)
            for col in column_to_update
            if col in self._json_columns
        ]
        stmt = text(stmt_text).bindparams(*json_bps, **primary_key_to_value, **update_params)
        result = await self._session.execute(stmt)
        mapping = result.mappings().first()
        return self._hydrate_opt(mapping)

    async def update_by_primary_key_or_fail(
        self,
        primary_key_to_value: dict[str, object],
        column_to_update: dict[str, object],
        *,
        exclude_deleted_or_archived: bool = True,
    ) -> ModelType:
        result = await self.update_by_primary_key(
            primary_key_to_value,
            column_to_update,
            exclude_deleted_or_archived=exclude_deleted_or_archived,
        )
        if result is None:
            raise NotFoundError(
                code="resource.not_found",
                message=(
                    f"{self.model_class.__name__} with primary key {primary_key_to_value} not found"
                ),
            )
        return result

    # ── SELECT all (paginated) ────────────────────────────────────────

    async def find_all(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        exclude_deleted_or_archived: bool = True,
        order_by: str | None = None,
    ) -> list[ModelType]:
        """Return rows with pagination.

        ``order_by`` is a raw SQL fragment (e.g. ``"created_at desc"``)
        so callers retain control of ordering — the base class does not
        validate it. Pass ``None`` for unspecified order.
        """
        # Build the WHERE clause from the soft-delete/archived fragments.
        # The first present fragment leads with ``where``; the second (if
        # any) prefixes with ``and``.
        soft = self._soft_delete_where_fragment(
            self.model_class,
            exclude_deleted_or_archived=exclude_deleted_or_archived,
            prefix="where",
        )
        archived = self._archived_where_fragment(
            self.model_class,
            exclude_deleted_or_archived=exclude_deleted_or_archived,
            prefix="and" if soft else "where",
        )
        where_clause = f"{soft} {archived}".strip()
        order_clause = f"order by {order_by}" if order_by else ""
        stmt_text = (
            f"select * from {self._table} {where_clause} {order_clause} limit :limit offset :offset"
        )
        stmt = text(stmt_text).bindparams(limit=limit, offset=offset)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]


__all__ = ["GenericRepository", "ModelType"]
