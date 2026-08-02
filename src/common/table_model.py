"""Pydantic-based TableModel — the project's chosen ORM substitute.

Persistence is done with raw SQL via `sqlalchemy.text()` and named
`bindparams`; see project conventions in AGENTS.md. TableModels are
Pydantic models that double as row shapes and bindparam dicts.

Repositories receive an `AsyncSession`, build SQL with `sqlalchemy.text(...)`,
and pass `row.model_dump()` (or a subset) as bindparams. Subclasses must
remain frozen and use concrete field types.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class TableModel(BaseModel):  # type: ignore[explicit-any]
    """Base for row shapes used by repositories.

    Subclasses define column-equivalent fields. Use `model_dump()` to obtain
    a `dict[str, object]` suitable for SQL bindparams.
    """

    model_config = ConfigDict(frozen=True)


__all__ = ["TableModel"]
