.PHONY: help install sync lint typecheck format format-fix test migrate clean dev-app

help:
	@echo "Targets:"
	@echo "  install     create venv + install deps (uv)"
	@echo "  sync        uv sync (dev deps)"
	@echo "  lint        ruff check ."
	@echo "  format      ruff format --check .  (CI gate — fails if unformatted)"
	@echo "  format-fix  ruff format .          (rewrites files in place)"
	@echo "  typecheck   mypy --strict (uses mypy.ini)"
	@echo "  test        pytest tests/"
	@echo "  migrate     alembic upgrade head"
	@echo "  clean       remove caches"

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

dev-app:
	uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000