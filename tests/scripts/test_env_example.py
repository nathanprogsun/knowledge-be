"""Tests for ``scripts/check_env_example.py`` helpers.

Live ``.env.example`` completeness is owned by the check script itself
and may lag a parallel template rewrite; these tests use tempfile
examples only.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

from pydantic_settings import BaseSettings

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_SCRIPT: Path = _REPO_ROOT / "scripts" / "check_env_example.py"


def _load_checker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_env_example", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _settings_cls(checker: ModuleType) -> type[BaseSettings]:
    loaded: type[BaseSettings] = checker.load_settings_class(_REPO_ROOT, "src.settings")
    return loaded


def _complete_body(checker: ModuleType, settings_cls: type[BaseSettings]) -> str:
    names: list[str] = sorted(checker.collect_settings_env_names(settings_cls))
    return "\n".join(f"{name}=" for name in names) + "\n"


def _write_example(tmp_path: Path, body: str) -> Path:
    path: Path = tmp_path / ".env.example"
    path.write_text(body, encoding="utf-8")
    return path


def test_parse_commented_assignment_counts() -> None:
    checker: ModuleType = _load_checker()
    keys: set[str] = checker.parse_env_example_keys(
        "APP_NAME=knowledge-be\n# OTEL_ENABLED=false\n# comment only\n"
    )
    assert keys == {"APP_NAME", "OTEL_ENABLED"}


def test_commented_key_counts_as_present(tmp_path: Path) -> None:
    checker: ModuleType = _load_checker()
    settings_cls: type[BaseSettings] = _settings_cls(checker)
    required: set[str] = checker.collect_settings_env_names(settings_cls)
    assert "OTEL_ENABLED" in required
    lines: list[str] = []
    for name in sorted(required):
        if name == "OTEL_ENABLED":
            lines.append("# OTEL_ENABLED=false")
        else:
            lines.append(f"{name}=")
    path: Path = _write_example(tmp_path, "\n".join(lines) + "\n")
    errors: list[str] = checker.check_example_file(path, settings_cls)
    assert errors == []


def test_missing_key_fails(tmp_path: Path) -> None:
    checker: ModuleType = _load_checker()
    settings_cls: type[BaseSettings] = _settings_cls(checker)
    required: set[str] = checker.collect_settings_env_names(settings_cls)
    omitted: str = "OTEL_ENABLED"
    body: str = "\n".join(f"{name}=" for name in sorted(required) if name != omitted) + "\n"
    path: Path = _write_example(tmp_path, body)
    errors: list[str] = checker.check_example_file(path, settings_cls)
    assert any("missing Settings env OTEL_ENABLED" in item for item in errors)


def test_extra_unknown_key_fails(tmp_path: Path) -> None:
    checker: ModuleType = _load_checker()
    settings_cls: type[BaseSettings] = _settings_cls(checker)
    body: str = _complete_body(checker, settings_cls) + "NOT_A_SETTINGS_KEY=1\n"
    path: Path = _write_example(tmp_path, body)
    errors: list[str] = checker.check_example_file(path, settings_cls)
    assert any("unknown env NOT_A_SETTINGS_KEY" in item for item in errors)


def test_database_url_extra_fails(tmp_path: Path) -> None:
    checker: ModuleType = _load_checker()
    settings_cls: type[BaseSettings] = _settings_cls(checker)
    body: str = (
        _complete_body(checker, settings_cls)
        + "# DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/knowledge_be\n"
    )
    path: Path = _write_example(tmp_path, body)
    errors: list[str] = checker.check_example_file(path, settings_cls)
    assert any("DATABASE_URL_OVERRIDE" in item for item in errors)
    assert any("DATABASE_URL is not a Settings field" in item for item in errors)


def test_settings_names_exclude_computed_and_worker_keys() -> None:
    checker: ModuleType = _load_checker()
    settings_cls: type[BaseSettings] = _settings_cls(checker)
    names: set[str] = checker.collect_settings_env_names(settings_cls)
    assert "DATABASE_URL_OVERRIDE" in names
    assert "OTEL_ENABLED" in names
    assert "DATABASE_URL" not in names
    assert all(not name.startswith("WORKER_") for name in names)


def test_cli_against_temp_example(tmp_path: Path) -> None:
    checker: ModuleType = _load_checker()
    settings_cls: type[BaseSettings] = _settings_cls(checker)
    path: Path = _write_example(tmp_path, _complete_body(checker, settings_cls))
    result: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--repo-root",
            str(_REPO_ROOT),
            "--example",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
