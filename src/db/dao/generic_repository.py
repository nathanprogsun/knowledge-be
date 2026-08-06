"""Generic raw-SQL CRUD helpers for ``TableModel`` repositories.

Concrete repositories inherit from ``GenericRepository[TheirModelType]``
and pick up ``insert`` / ``insert_or_none`` / ``find_by_*`` /
``update_by_primary_key``. Domain-specific queries stay on the
concrete subclasses.

Every query is raw ``sqlalchemy.text()`` with named ``bindparams`` —
no ORM. Soft-delete and archived filters are applied at every read.

SQL is built from ``text(f"...")`` whose only interpolated values are
class-level constants: ``self._table`` (the model's fully-qualified
table name, a ``ClassVar[str]``) and column / fragment names taken
from :meth:`TableModel.column_fields` / :meth:`ordered_primary_keys`.
User input never reaches the SQL string — only ``bindparams`` slots —
and dynamic identifiers (e.g. ``order_by`` columns on
:meth:`find_all`) are validated against an allow-list before use.
"""

from __future__ import annotations

import re
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
from src.common.json import BindParams, SqlValue
from src.common.table_model import TableModel

ModelType = TypeVar("ModelType", bound=TableModel)

# Concrete union of every Python value a ``TableModel`` field may hold.
BindValue: TypeAlias = str | int | float | bool | datetime | None

# JSONB on Postgres, JSON on other dialects (e.g. SQLite in tests).
_JSON_BIND_TYPE = JSON().with_variant(JSONB(), "postgresql")

# Allow-list of column names :meth:`find_all` will accept on ``order_by``.
# Any column referenced in this set is considered a safe, read-only sort
# key; callers may still append ``asc`` / ``desc`` (the validator only
# checks the leading identifier). Expanding the set requires an explicit
# PR so the safe-by-default posture is preserved.
_ALLOWED_ORDER_COLUMNS: frozenset[str] = frozenset(
    {
        "id",
        "created_at",
        "updated_at",
        "name",
        "joined_at",
        "responded_at",
        "expires_at",
        "started_at",
        "last_used_at",
    },
)

# SQL identifier regex (Postgres-style: letter / underscore start, then
# letters / digits / underscores). The set is the gate; this regex is the
# final structural check before a value is interpolated into SQL.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class GenericRepository(Generic[ModelType]):
    """Raw-SQL CRUD for ``TableModel`` rows. Subclasses set
    ``model_class``; the per-request session comes from ``__init__``.
    """

    model_class: type[ModelType]

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ── Cached model metadata ────────────────────────────────────────

    @cached_property
    def _table(self) -> str:
        # ``fq_table_name`` returns the model's ``ClassVar[str] table``
        # attribute — a fixed compile-time literal. Validate the shape
        # once so a misconfigured model surfaces at first use rather
        # than as opaque SQL at execution time.
        name = self.model_class.fq_table_name()
        self._assert_safe_identifier(name, kind="table")
        return name

    @staticmethod
    def _assert_safe_identifier(value: str, *, kind: str) -> None:
        """Reject identifiers that don't match the SQL identifier shape.

        Called for any value that is about to be interpolated into the
        SQL string (table / column names from model metadata). User
        input is NEVER routed through this check — it is bound via
        ``bindparams`` instead. The check exists so that a typo on a
        ``ClassVar`` or a hand-edited row cannot smuggle arbitrary SQL
        through the f-string interpolation point.
        """
        if not _IDENTIFIER_RE.fullmatch(value):
            raise DataError(
                code="db.invalid_identifier",
                message=f"unsafe {kind} identifier rejected: {value!r}",
            )

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
            self.model_class.from_row(cast("Mapping[str, SqlValue]", mapping)),
        )

    def _hydrate_opt(self, mapping: RowMapping | None) -> ModelType | None:
        return self._hydrate(mapping) if mapping is not None else None

    @staticmethod
    def _require_non_empty_query(column_to_query: BindParams) -> None:
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
        """Build ``and deleted_at is null`` fragment.

        Returns empty string when the model has no ``deleted_at`` column
        or when filtering is disabled. ``prefix`` controls the leading
        connector (``and`` / ``where``) so the fragment can be appended
        to either a WHERE clause or an existing condition list.
        """
        if not exclude_deleted_or_archived:
            return ""
        if "deleted_at" not in model.column_fields():
            return ""
        # ``deleted_at`` is a literal column name declared in this
        # module — not user input — but it is interpolated into the
        # SQL string, so route it through the identifier guard for the
        # same audit posture as every other f-string interpolation.
        GenericRepository._assert_safe_identifier("deleted_at", kind="column")
        return f"{prefix} deleted_at is null"

    @staticmethod
    def _archived_where_fragment(
        model: type[ModelType],
        *,
        exclude_deleted_or_archived: bool,
        prefix: str = "and",
    ) -> str:
        """Build ``and archived_at is null`` fragment."""
        if not exclude_deleted_or_archived:
            return ""
        if "archived_at" not in model.column_fields():
            return ""
        GenericRepository._assert_safe_identifier("archived_at", kind="column")
        return f"{prefix} archived_at is null"

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
        # Validate every identifier that will be interpolated into the
        # SQL string. ``insert_sql_column_list`` returns field names
        # from the model, which are class-level constants, but the
        # guard catches a misconfigured model early.
        table = model.fq_table_name()
        GenericRepository._assert_safe_identifier(table, kind="table")
        for col in columns:
            GenericRepository._assert_safe_identifier(col, kind="column")
        col_list = ", ".join(f'"{c}"' for c in columns)
        param_list = ", ".join(f":{c}" for c in columns)
        return f"insert into {table} ({col_list}) values ({param_list}) returning *"

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
        table = model.fq_table_name()
        GenericRepository._assert_safe_identifier(table, kind="table")
        columns = model.insert_sql_column_list()
        for col in columns:
            GenericRepository._assert_safe_identifier(col, kind="column")
        base = (
            f"insert into {table} ("
            + ", ".join(f'"{c}"' for c in columns)
            + ") values ("
            + ", ".join(f":{c}" for c in columns)
            + ")"
        )
        if target_columns:
            for col in target_columns:
                GenericRepository._assert_safe_identifier(col, kind="column")
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
        primary_key_to_value: BindParams,
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
        primary_key_to_value: BindParams,
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
        column_to_query: BindParams,
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
        params: BindParams = {}
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
        column_to_query: BindParams,
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
        column_to_query: BindParams,
        *,
        exclude_deleted_or_archived: bool = True,
    ) -> list[ModelType]:
        """Return every row matching ``column_to_query``."""
        self._require_non_empty_query(column_to_query)
        self.model_class.validate_in_columns(column_to_query)
        where_parts: list[str] = []
        params: BindParams = {}
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
        primary_key_to_value: BindParams,
        column_to_update: BindParams,
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
        # The WHERE clause binds only the actual primary-key columns.
        # Drop any extra keys callers passed (e.g. a tenant_id scoping
        # key) so ``bindparams`` never sees an unbound parameter.
        primary_key_to_value = {
            k: v for k, v in primary_key_to_value.items() if k in self._pk_columns
        }
        pk_cols = self.model_class.ordered_primary_keys()
        where_pk = " and ".join(f'"{c}" = :{c}' for c in pk_cols)
        set_clause = ", ".join(f'"{k}" = :u_{k}' for k in column_to_update)
        update_params: BindParams = {f"u_{k}": v for k, v in column_to_update.items()}
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
        primary_key_to_value: BindParams,
        column_to_update: BindParams,
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

        ``order_by`` is the leading identifier of an ``ORDER BY``
        expression (e.g. ``"created_at"`` or ``"created_at desc"``).
        The identifier must be present in :data:`_ALLOWED_ORDER_COLUMNS`;
        the optional ``asc`` / ``desc`` token, if supplied, is passed
        through unchanged. Pass ``None`` for unspecified order.
        """
        order_clause = ""
        if order_by is not None:
            # ``order_by`` is caller-controlled text interpolated into
            # the SQL string; validate the leading identifier so a
            # crafted value cannot smuggle a sub-query or extra clause.
            tokens = order_by.split()
            if not tokens:
                raise ValidationError(
                    code="db.invalid_order_by",
                    message="order_by must contain at least one identifier",
                )
            column = tokens[0]
            if column not in _ALLOWED_ORDER_COLUMNS:
                raise ValidationError(
                    code="db.invalid_order_by",
                    message=(
                        f"order_by column {column!r} is not in the allow-list "
                        f"({sorted(_ALLOWED_ORDER_COLUMNS)})"
                    ),
                )
            self._assert_safe_identifier(column, kind="order_by_column")
            for token in tokens[1:]:
                # Acceptable direction tokens are literal SQL keywords,
                # not identifiers — keep the allow-list strict so any
                # future caller mistake fails closed.
                if token.lower() not in {"asc", "desc"}:
                    raise ValidationError(
                        code="db.invalid_order_by",
                        message=(
                            f"order_by modifier {token!r} is not allowed; "
                            "expected 'asc' or 'desc'"
                        ),
                    )
            order_clause = f"order by {order_by}"
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
        stmt_text = (
            f"select * from {self._table} {where_clause} {order_clause} limit :limit offset :offset"
        )
        stmt = text(stmt_text).bindparams(limit=limit, offset=offset)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]


__all__ = ["GenericRepository", "ModelType"]
