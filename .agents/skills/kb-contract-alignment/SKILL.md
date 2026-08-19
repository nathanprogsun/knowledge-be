---
name: kb-contract-alignment
description: Use when changing a contract-bearing file — pagination, error
  hierarchy, frozen contracts under src/core/contracts/, or domain view
  models. Verifies field names and wire shapes match the upstream contract
  exactly.
---

# kb-contract-alignment

Contract-bearing files define the public HTTP/wire shapes. Field names
are part of the contract; renaming any of them is a breaking change for
clients.

## Contract-bearing files

- `src/common/pagination.py` — request `page`/`page_size` (capped at
  100, default 20); response `total`/`page`/`page_size`/`data`.
- `src/common/exception.py` — error payload `code`/`message`/`details`;
  `details` is a declared field.
- `src/core/contracts/*.py` — frozen Pydantic models
  (`model_config = ConfigDict(frozen=True)`).
- Domain view models (service-output DTOs).

## Procedure

1. Open the upstream contract definition and compare field names,
   types, and optionality one-to-one.
2. Match JSON serialization names exactly. Default Pydantic snake_case
   is acceptable when it matches.
3. Confirm frozen `model_config` on contract models.
4. Run `make check-contract` (contract invariants) and
   `make check-schema` (schema compatibility); both must stay green.

## Drift that has shipped before

- `limit` instead of `page_size` on pagination requests.
- `items` instead of `data` on list responses.
- `details` as a runtime-only attribute instead of a declared field.

Each was caught by the contract checks. Do not repeat.
