# Agent Note: HTTP contract field alignment

Status: implemented
Tags: contracts, pagination, errors, http
Date: 2026-08-19
Scope: wire shapes for pagination and application errors
Related files: src/common/pagination.py, src/common/exception.py, scripts/check_pr_leak.py

## Context

Pagination request/response and error payload shapes are part of the
public HTTP contract; renaming any of them is a breaking change for
clients. Field names must match the upstream contract exactly.

## Decision

- Pagination request: `page`, `page_size` (capped at 100, default 20).
- Pagination response: `total`, `page`, `page_size`, `data`.
- Error payload: `code`, `message`, `details` — `details` is a declared
  field on the base error class, not a runtime-only attribute.

## Alternatives considered

- **`limit` instead of `page_size`** — rejected: drifted from the
  upstream name; the field-alignment check now enforces the canonical
  name.
- **`items` for list payloads** — rejected: response bodies must carry
  `data` per the contract.
- **Runtime-only `details`** — rejected: an undeclared attribute is
  invisible to schema/serialization tooling.

## Consequences

Clients see canonical field names; drift surfaces in the contract
checks instead of at runtime.

## Required verification

`make check-contract` and `make check-schema` must stay green; the
contract-invariants gate runs on every change.
