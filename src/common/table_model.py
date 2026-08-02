"""Pydantic-based TableModel - the project's chosen ORM substitute.

Persistence is done with raw SQL via `sqlalchemy.text()` and named
`bindparams`; see project conventions in AGENTS.md. TableModels are
Pydantic models that double as row shapes and bindparam dicts.

Repositories receive an `AsyncSession`, build SQL with `sqlalchemy.text(...)`,
and pass `row.model_dump()` (or a subset) as bindparams. Subclasses must
remain frozen and use concrete field types.

The ``table`` attribute on every subclass is a ``ClassVar[str]`` - it is
metadata (the SQL table name), not a DB column. Using ``ClassVar``
ensures Pydantic v2 does NOT treat it as a model field, so it stays out
of ``model_fields`` / ``model_dump()`` / INSERT + SELECT column lists.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class TableModel(BaseModel):
    """Base for row shapes used by repositories.

    Subclasses define column-equivalent fields. Use `model_dump()` to obtain
    a `dict[str, object]` suitable for SQL bindparams.

    Every subclass MUST declare::

        table: ClassVar[str] = "<sql_table_name>"

    so ``GenericRepository`` knows which table to read/write. The
    ``ClassVar`` annotation prevents Pydantic from treating it as a
    model field.
    """

    model_config = ConfigDict(frozen=True)


__all__ = ["TableModel"]
