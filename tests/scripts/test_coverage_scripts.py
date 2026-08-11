"""Tests for the service/task/go-file coverage scripts.

Verifies that the three coverage scripts run against the repository and
that the service inventory is valid.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"
_INVENTORY = _SCRIPTS / "service_inventory.json"


def test_service_inventory_is_valid() -> None:
    """The inventory must parse and carry a non-empty service list."""
    data = json.loads(_INVENTORY.read_text())
    assert data["total_services"] > 0
    assert len(data["services"]) > 0
    for svc in data["services"]:
        assert svc["name"].endswith("Service")


def test_service_coverage_script_runs() -> None:
    """check_service_coverage.py must execute without crashing."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPTS / "check_service_coverage.py"), "--src-root", str(_REPO_ROOT)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode in (0, 1), result.stderr[-2000:]


def test_task_coverage_script_runs() -> None:
    """check_task_coverage.py must execute and report 19 tasks."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPTS / "check_task_coverage.py"), "--src-root", str(_REPO_ROOT)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode in (0, 1), result.stderr[-2000:]
    assert "19" in result.stdout


def test_go_file_coverage_script_runs() -> None:
    """check_go_file_coverage.py must execute without crashing."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPTS / "check_go_file_coverage.py"), "--src-root", str(_REPO_ROOT)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode in (0, 1), result.stderr[-2000:]
