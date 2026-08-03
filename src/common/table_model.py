"""Pydantic-based TableModel - the project's chosen ORM substitute.

Persistence is done with raw SQL via `sqlalchemy.text()` and named
`bindparams`; see project conventions in AGENTS.md. TableModels are
frozen Pydantic models that double as row shapes and bindparam dicts.

Repositories receive an `AsyncSession`, build SQL with `sqlalchemy.text(...)`,
and pass `row.model_dump()` (or a subset) as bindparams. Subclasses must
remain frozen and use concrete field types.

The ``table`` attribute on every subclass is a ``ClassVar[str]`` - it is
metadata (the SQL table name), not a DB column. Using ``ClassVar``
ensures Pydantic v2 does NOT treat it as a model field, so it stays out
of ``model_fields`` / ``model_dump()`` / INSERT + SELECT column lists.

Primary-key metadata
--------------------

Every subclass declares its primary-key column name(s) via the
``primary_keys: ClassVar[tuple[str, ...]]`` attribute (defaults to
``("id",)``). ``GenericRepository`` reads it to build WHERE-by-pk,
ON CONFLICT, and validate clauses. Multi-column primary keys are
supported by listing every column in declaration order.

JSON columns
------------

Subclasses declare which columns are JSON/JSONB via the
``json_columns: ClassVar[tuple[str, ...]]`` attribute. The repository
attaches ``bindparam(col, type_=JSONB)`` to those so the driver does
not require implicit ``json.dumps``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from src.common.exception import ValidationError


class TableModel(BaseModel):
    """Base for row shapes used by repositories.

    Subclasses define column-equivalent fields. Use `model_dump()` to obtain
    a `dict[str, object]` suitable for SQL bindparams.

    Every subclass MUST declare::

        table: ClassVar[str] = "<sql_table_name>"
        primary_keys: ClassVar[tuple[str, ...]] = ("<pk_col>", ...)
        json_columns: ClassVar[tuple[str, ...]] = (...,)

    so ``GenericRepository`` knows which table to read/write, how to
    build WHERE-by-pk / ON CONFLICT clauses, and which bindparams need
    ``JSONB`` typing. The ``ClassVar`` annotation prevents Pydantic from
    treating metadata as a model field.
    """

    model_config = ConfigDict(frozen=True)

    # Subclasses override these ClassVar metadata fields.
    table: ClassVar[str] = ""
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ()

    # ── Read-side metadata ───────────────────────────────────────────

    @classmethod
    def fq_table_name(cls) -> str:
        """Fully-qualified table name (currently just the bare name)."""
        return cls.table

    @classmethod
    def column_fields(cls) -> tuple[str, ...]:
        """All DB column names (Pydantic field names)."""
        return tuple(cls.model_fields.keys())

    @classmethod
    def ordered_primary_keys(cls) -> tuple[str, ...]:
        """Primary-key column names in declaration order.

        Raises ``ValidationError`` if a declared pk is not a model field —
        this catches schema drift at class-creation time rather than at
        SQL execution.
        """
        fields = cls.column_fields()
        for pk in cls.primary_keys:
            if pk not in fields:
                raise ValidationError(
                    code="db.schema_drift",
                    message=f"primary key '{pk}' is not a field of {cls.__name__}",
                )
        return cls.primary_keys

    @classmethod
    def primary_key_column_list(cls) -> tuple[str, ...]:
        """Alias of :meth:`ordered_primary_keys` for SQL-builder clarity."""
        return cls.ordered_primary_keys()

    @classmethod
    def get_json_columns(cls) -> tuple[str, ...]:
        """Columns that must be bound as ``JSONB`` on Postgres."""
        return cls.json_columns

    # ── Insert-side metadata ──────────────────────────────────────────

    @classmethod
    def insert_sql_column_list(cls) -> tuple[str, ...]:
        """Columns participating in INSERT.

        Defaults to every model field. Subclasses with DB-generated
        columns (autoincrement, defaults populated by triggers) override
        this to exclude them.
        """
        return cls.column_fields()

    @classmethod
    def insert_sql_column_param_list(cls) -> tuple[str, ...]:
        """Bindparam placeholders for :meth:`insert_sql_column_list`."""
        return tuple(f":{c}" for c in cls.insert_sql_column_list())

    def insert_bind_params(self) -> dict[str, object]:
        """Bindparam dict for INSERT (subset of ``model_dump``)."""
        return self.model_dump(include=set(self.insert_sql_column_list()))

    # ── Validation helpers ───────────────────────────────────────────

    @classmethod
    def validate_in_columns(cls, columns: Mapping[str, object] | Iterable[str]) -> None:
        """Ensure each name in ``columns`` is a real column of this model.

        Accepts a list, dict keys, or any iterable of column names. Raises
        ``ValidationError`` on the first unknown column so DAOs fail fast
        on bad input rather than at SQL execution time. ``ValidationError``
        is an ``ApplicationError`` subclass (AGENTS.md §5): callers in the
        ``web`` layer translate it to a 4xx response.
        """
        fields = cls.column_fields()
        keys = columns.keys() if isinstance(columns, Mapping) else columns
        for c in keys:
            if c not in fields:
                raise ValidationError(
                    code="db.unknown_column",
                    message=f"column '{c}' is not a field of {cls.__name__}",
                )

    @classmethod
    def validate_contains_all_primary_keys(
        cls, columns: Mapping[str, object] | Iterable[str]
    ) -> None:
        """Ensure ``columns`` includes every primary-key column."""
        fields = cls.column_fields()
        keys = set(columns.keys()) if isinstance(columns, Mapping) else set(columns)
        for pk in cls.ordered_primary_keys():
            if pk not in keys:
                raise ValidationError(
                    code="db.missing_primary_key",
                    message=f"primary key '{pk}' of {cls.__name__} is missing",
                )
            if pk not in fields:
                raise ValidationError(
                    code="db.unknown_column",
                    message=f"primary key '{pk}' is not a field of {cls.__name__}",
                )

    # ── Read-side hydration ───────────────────────────────────────────

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> TableModel:
        """Hydrate a model from a SQLAlchemy mapping row.

        ``row`` is typically the ``.mappings().first()`` / ``.all()``
        result of a ``SELECT ... `` - either a ``RowMapping`` or a plain
        ``dict``. We normalise to ``dict`` then defer to Pydantic.
        """
        return cls.model_validate(dict(row))

    # ── PK extraction ─────────────────────────────────────────────────

    def primary_key_to_value(self) -> dict[str, object]:
        """Map primary-key column name -> current value."""
        return {pk: getattr(self, pk) for pk in self.ordered_primary_keys()}


__all__ = ["TableModel"]
