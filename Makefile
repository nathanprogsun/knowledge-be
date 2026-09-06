.PHONY: help install sync lint typecheck format format-fix test migrate clean dev-app openapi frontend-install frontend-typecheck frontend-test frontend-build check check-layer check-singleton check-endpoint check-schema check-imports check-sql check-pr-leak check-map-from-db check-exception-types check-agent-notes check-feature-map

help:
	@echo "Targets:"
	@echo "  install     create venv + install deps (uv)"
	@echo "  sync        uv sync (dev deps)"
	@echo "  lint        ruff check ."
	@echo "  format      ruff format --check .  (CI gate — fails if unformatted)"
	@echo "  format-fix  ruff format .          (rewrites files in place)"
	@echo "  typecheck   uv run mypy (strict from mypy.ini; project venv only)"
	@echo "  test        pytest tests/"
	@echo "  check       run all anti-drift checks"
	@echo "  migrate     alembic upgrade head"
	@echo "  clean       remove caches"

AUTH_TENANT_DOMAINS := auth,tenants,system

install:
	uv venv
	uv sync --all-extras

sync:
	uv sync --all-extras

lint:
	uv run ruff check .

format:
	uv run ruff format --check .

format-fix:
	uv run ruff format .

typecheck:
	uv run mypy

# Ratchet gate: mypy may only improve vs the recorded baseline. CI runs
# this instead of raw mypy until the backlog is burned down. Runs via uv so
# the mypy that produces the numbers is the project venv's, not whatever a
# global python happens to resolve (a mismatched mypy reports phantom
# import-not-found regressions and fails the gate incorrectly).
check-mypy-baseline:
	uv run python scripts/check_mypy_baseline.py

test:
	uv run pytest

migrate:
	uv run alembic upgrade head

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage

# ── Anti-drift checks ───────────────────────────────────────────────────

INFRA_DOMAINS := datasources,initialization,mcp_services,models,storage_backends,vector_stores,web_search
PRODUCT_DOMAINS := favorites,chat,organizations,channels,knowledge,knowledge_bases,agents,evaluation,sharing,me,files,cloud

check: check-layer check-singleton check-endpoint check-schema check-imports check-sql check-pr-leak check-map-from-db check-agent-notes check-mypy-baseline check-exception-types check-feature-map
	@echo "All anti-drift checks passed"

# Layer check covers auth/tenant, infra, and product domains. `ai` and
# `workers` stay out until the retrieval Any backlog is cleared.
check-layer:
	python scripts/check_layer_violation.py --src-root src/ --domains $(AUTH_TENANT_DOMAINS),$(INFRA_DOMAINS),$(PRODUCT_DOMAINS)

check-singleton:
	bash scripts/run_check_service_singleton.sh --src-root src/

check-endpoint:
	python scripts/check_endpoint_coverage.py --src-root src/ --docs-root docs

check-schema:
	python scripts/check_schema_compatibility.py --src-root src/

check-imports:
	python scripts/check_imports.py --src-root src/

check-sql:
	python scripts/check_sql_format.py --src-root src/

check-pr-leak:
	python scripts/check_pr_leak.py --repo-root .

check-map-from-db:
	python scripts/check_map_from_db.py --src-root src/

check-exception-types:
	python scripts/check_exception_types.py --src-root src/

check-agent-notes:
	python scripts/verify_agent_notes.py --repo-root .

check-feature-map:
	python scripts/check_feature_map.py --repo-root .

dev-app:
	uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# ── Frontend contract codegen ───────────────────────────────────────────
# Single source of truth: FastAPI OpenAPI schema. Frontend TS types are
# generated, never hand-written against the backend.

openapi:
	uv run python scripts/export_openapi.py --output docs/api/openapi.json
	cd frontend && npx openapi-typescript ../docs/api/openapi.json -o src/api/__generated__/schema.ts

frontend-install:
	cd frontend && npm ci

frontend-typecheck:
	cd frontend && npm run type-check

frontend-test:
	cd frontend && npm test

frontend-build:
	cd frontend && npm run build