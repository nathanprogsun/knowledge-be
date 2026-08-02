"""Pagination request and response shapes.

Field names are part of the public HTTP contract; renaming any of them
is a breaking change for clients.

- Request: `page`, `page_size` (capped at 1000, default 20).
- Response: `total`, `page`, `page_size`, `data`.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class Pagination(BaseModel):  # type: ignore[explicit-any]
    model_config = ConfigDict(frozen=True)

    page: int = Field(ge=1, default=1)
    page_size: int = Field(ge=1, le=1000, default=20)


class PaginationResponse(BaseModel, Generic[T]):  # type: ignore[explicit-any]
    model_config = ConfigDict(frozen=True)

    total: int
    page: int
    page_size: int
    data: list[T]


__all__ = ["Pagination", "PaginationResponse"]
