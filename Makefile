.PHONY: help install sync lint typecheck format format-fix test migrate clean dev-app check check-layer check-singleton check-endpoint check-schema check-contract check-imports

help:
	@echo "Targets:"
	@echo "  install     create venv + install deps (uv)"
	@echo "  sync        uv sync (dev deps)"
	@echo "  lint        ruff check ."
	@echo "  format      ruff format --check .  (CI gate — fails if unformatted)"
	@echo "  format-fix  ruff format .          (rewrites files in place)"
	@echo "  typecheck   mypy --strict (uses mypy.ini)"
	@echo "  test        pytest tests/"
	@echo "  check       run all anti-drift checks"
	@echo "  migrate     alembic upgrade head"
	@echo "  clean       remove caches"

STAGE1_DOMAINS := auth,tenants,system

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

test:
	uv run pytest

migrate:
	uv run alembic upgrade head

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage

# ── Anti-drift checks (checkpoint) ──────────────────────────────────────

check: check-layer check-singleton check-endpoint check-contract check-imports
	@echo "All anti-drift checks passed"

check-layer:
	python scripts/check_layer_violation.py --src-root src/ --domains $(STAGE1_DOMAINS)

check-singleton:
	python scripts/check_service_singleton.py --src-root src/

check-endpoint:
	python scripts/check_endpoint_coverage.py --src-root src/ --domains auth,tenants

check-schema:
	python scripts/check_schema_compatibility.py --src-root src/

check-contract:
	python scripts/check_contract_invariants.py --src-root src/

check-imports:
	python scripts/check_imports.py --src-root src/

dev-app:
	uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000