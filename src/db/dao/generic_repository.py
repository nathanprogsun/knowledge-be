"""Generic CRUD helpers for raw-SQL TableModel repositories.

Concrete repositories (``UserRepository``, ``AuthTokenRepository``, …)
inherit from ``GenericRepository[TheirModelType]`` and pick up the
boilerplate ``insert`` / ``find_by_id`` implementations. Domain-specific
queries (``find_by_email``, ``update``, ``soft_delete``, …) stay on the
concrete subclasses.

Why a base class? PR-1 and PR-2 each shipped a copy of the same INSERT
+ SELECT-by-id SQL — about 40 lines per repo, varying only in the model
type. Pushing the common shape behind ``Generic[ModelType]`` keeps each
concrete repo focused on what is actually unique.

Why raw SQL? Per AGENTS.md §6.5, this project deliberately avoids the
SQLAlchemy ORM — every query uses ``sqlalchemy.text()`` with named
``bindparams``. ``GenericRepository`` follows that rule.

Design notes
------------

- ``TableModel`` carries ``table: str = "..."`` (literal) and
  ``model_fields`` (Pydantic v2 introspection). The base class reads
  those to build the SQL — no ORM-style reflection.
- All columns participate in INSERT and SELECT. Pydantic ``model_dump``
  produces ``dict[str, object]`` directly usable as bindparams; on the
  read side, ``model_validate(dict(row))`` rebuilds the typed row.
- The base class is deliberately minimal: only ``insert`` and
  ``find_by_id``. Soft-delete, update, list-with-ordering, etc. stay
  on concrete subclasses where the SQL is domain-specific.
"""

from __future__ import annotations

from functools import cached_property
from typing import Generic, TypeVar

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.table_model import TableModel

ModelType = TypeVar("ModelType", bound=TableModel)


class GenericRepository(Generic[ModelType]):
    """Boilerplate-free raw-SQL CRUD for ``TableModel`` rows.

    Subclasses pass their model class to ``__init__`` and inherit
    ``insert`` and ``find_by_id``.
    """

    model_class: type[ModelType]

    def __init__(self, model_class: type[ModelType]) -> None:
        self.model_class = model_class

    @cached_property
    def _table(self) -> str:
        return self.model_class.table  # type: ignore[attr-defined,no-any-return]

    @cached_property
    def _column_names(self) -> tuple[str, ...]:
        return tuple(self.model_class.model_fields.keys())

    @cached_property
    def _columns_csv(self) -> str:
        return ", ".join(self._column_names)

    @cached_property
    def _placeholders_csv(self) -> str:
        return ", ".join(f":{name}" for name in self._column_names)

    async def insert(self, session: AsyncSession, row: ModelType) -> None:
        """Insert ``row`` into the table. All columns are populated."""
        stmt = text(
            f"INSERT INTO {self._table} ({self._columns_csv}) VALUES ({self._placeholders_csv})"
        )
        await session.execute(stmt.bindparams(**row.model_dump()))

    async def find_by_id(self, session: AsyncSession, id: str | int) -> ModelType | None:
        """Look up a single row by its primary key."""
        stmt = text(f"SELECT {self._columns_csv} FROM {self._table} WHERE id = :id")
        row = (await session.execute(stmt.bindparams(id=id))).mappings().first()
        if row is None:
            return None
        return self.model_class.model_validate(dict(row))


__all__ = ["GenericRepository", "ModelType"]
