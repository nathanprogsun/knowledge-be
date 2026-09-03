"""Tests for the feature-map generator and drift check."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"
_MAP = _REPO_ROOT / ".agents" / "feature-map" / "generated.json"


def _load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_feature_map",
        _SCRIPTS / "build_feature_map.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_map_contains_session_samples() -> None:
    builder = _load_builder()
    feature_map = builder.build_feature_map(_REPO_ROOT)
    endpoint_keys = {(item["method"], item["path"]) for item in feature_map["endpoints"]}
    service_names = {item["name"] for item in feature_map["services"]}
    assert ("POST", "/api/v1/sessions") in endpoint_keys
    assert "build_session_service" in service_names
    assert feature_map["tasks"]


def test_committed_map_matches_generation() -> None:
    builder = _load_builder()
    expected = builder.dump_feature_map(builder.build_feature_map(_REPO_ROOT))
    assert _MAP.is_file()
    assert _MAP.read_text(encoding="utf-8") == expected
    payload = json.loads(_MAP.read_text(encoding="utf-8"))
    assert payload["generated_by"] == "scripts/build_feature_map.py"


def test_check_script_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPTS / "check_feature_map.py"), "--repo-root", str(_REPO_ROOT)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
