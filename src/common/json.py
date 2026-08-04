"""JSON value type aliases for arbitrary JSON-backed data.

``JsonValue`` is re-exported from Pydantic — a recursive JSON type whose
schema generation terminates (unlike a hand-rolled recursive
``TypeAlias``). Using a named alias rather than bare ``object`` / ``Any``
keeps annotations concrete and satisfies the anti-drift rule that forbids
``Any`` / ``object`` annotations.

Used for JSONB columns, SQL bindparams, and config blobs whose concrete
nested shape is not modelled as a Pydantic type.
"""

from __future__ import annotations

from datetime import datetime
from typing import TypeAlias

from pydantic import JsonValue

JsonObject = dict[str, JsonValue]

# SQL bindparams may carry ``datetime`` values (JSON columns never do at
# runtime, but param maps for ``text()`` queries often include timestamps).
SqlValue: TypeAlias = JsonValue | datetime
BindParams = dict[str, SqlValue]

__all__ = ["BindParams", "JsonObject", "JsonValue", "SqlValue"]
