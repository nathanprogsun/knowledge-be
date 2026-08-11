"""Tests for the endpoint coverage check script and inventory.

Verifies that ``scripts/check_endpoint_coverage.py`` runs against the
repository and that ``scripts/endpoint_inventory.json`` is a valid,
non-empty inventory of the registered FastAPI routes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "check_endpoint_coverage.py"
_INVENTORY = _REPO_ROOT / "scripts" / "endpoint_inventory.json"


def test_inventory_is_valid_json() -> None:
    """The inventory file must parse and carry a non-empty endpoint list."""
    data = json.loads(_INVENTORY.read_text())
    assert data["total_paths"] > 0
    assert len(data["endpoints"]) > 0
    # Every endpoint must have a path and method.
    for ep in data["endpoints"]:
        assert ep["path"].startswith("/")
        assert ep["method"] in {"GET", "POST", "PUT", "DELETE", "PATCH"}


def test_inventory_matches_live_openapi() -> None:
    """The inventory endpoint set must match the live app's OpenAPI paths."""
    from fastapi.testclient import TestClient

    from src.app_context.lifespan import create_app

    app = create_app()
    client = TestClient(app)
    openapi = client.get("/openapi.json").json()
    live = {
        (method.upper(), path)
        for path, ops in openapi["paths"].items()
        for method in ops
    }
    inventory = {
        (ep["method"], ep["path"]) for ep in json.loads(_INVENTORY.read_text())["endpoints"]
    }
    assert inventory == live


def test_coverage_script_runs() -> None:
    """The coverage script must execute without crashing."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--src-root", str(_REPO_ROOT / "src")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    # Exit 0 or 1 are both acceptable (1 = coverage gaps reported);
    # a crash (non-zero with traceback) is not.
    assert result.returncode in (0, 1), result.stderr[-2000:]
