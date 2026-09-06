#!/usr/bin/env python3
"""Fail when ``.env.example`` drifts from ``src.settings.Settings``.

Settings is the source of truth for process env names. The onboarding
template must list every field Settings actually reads (as an assignment
or a commented assignment) and must not advertise keys Settings ignores.

``database_url`` is computed and is excluded. The compose override is
``DATABASE_URL_OVERRIDE``; documenting ``DATABASE_URL`` is a hard fail
so the wrong name cannot re-enter the template. WorkerSettings
``WORKER_*`` keys are out of scope — this gate only loads ``Settings``.

Usage::

    python scripts/check_env_example.py [--repo-root PATH]
        [--example PATH] [--settings-module MODULE]

Exit codes:
    0 = ``.env.example`` keys match Settings env names
    1 = missing Settings key, unknown extra key, or unreadable inputs
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path
from types import ModuleType

from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings

# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────

DEFAULT_SETTINGS_MODULE: str = "src.settings"
EXAMPLE_RELATIVE: str = ".env.example"
COMPUTED_FIELD_EXCLUDE: frozenset[str] = frozenset({"database_url"})
WRONG_DATABASE_URL_NAME: str = "DATABASE_URL"

ENV_ASSIGNMENT_RE: re.Pattern[str] = re.compile(r"^\s*#?\s*([A-Za-z_][A-Za-z0-9_]*)=")


# ─────────────────────────────────────────────────────────────────────────
# Resolution
# ─────────────────────────────────────────────────────────────────────────


def resolve_repo_root(explicit: str | None) -> Path | None:
    """Locate the repository root.

    Resolution order: CLI ``--repo-root`` -> walk up from this script
    (up to 6 ancestors) looking for ``src/`` and ``scripts/``.
    """
    if explicit:
        path: Path = Path(explicit).resolve()
        return path if path.is_dir() else None
    cur: Path = Path(__file__).resolve().parent
    for _ in range(6):
        if (cur / "src").is_dir() and (cur / "scripts").is_dir():
            return cur
        cur = cur.parent
    return None


def resolve_example_path(explicit: str | None, repo_root: Path | None) -> Path | None:
    if explicit:
        path: Path = Path(explicit)
        if not path.is_absolute() and repo_root is not None:
            path = repo_root / path
        return path.resolve()
    if repo_root is not None:
        return (repo_root / EXAMPLE_RELATIVE).resolve()
    return None


def ensure_repo_on_sys_path(repo_root: Path) -> None:
    resolved: str = str(repo_root.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


# ─────────────────────────────────────────────────────────────────────────
# Settings env names
# ─────────────────────────────────────────────────────────────────────────


def load_settings_class(repo_root: Path, module_name: str) -> type[BaseSettings]:
    """Import ``Settings`` from ``module_name`` with ``repo_root`` on ``sys.path``.

    Does not instantiate Settings, so no ``.env`` load and no database.
    """
    ensure_repo_on_sys_path(repo_root)
    module: ModuleType = importlib.import_module(module_name)
    loaded_attr: type[BaseSettings] | None = None
    maybe_settings = getattr(module, "Settings", None)
    if isinstance(maybe_settings, type) and issubclass(maybe_settings, BaseSettings):
        loaded_attr = maybe_settings
    if loaded_attr is None:
        raise TypeError(f"{module_name} does not define a pydantic-settings Settings class")
    return loaded_attr


def settings_env_prefix(settings_cls: type[BaseSettings]) -> str:
    prefix: str = str(settings_cls.model_config.get("env_prefix") or "")
    return prefix


def env_name_for_field(field_name: str, field: FieldInfo, env_prefix: str) -> str:
    """Map a Settings field to the env name Settings would read.

    pydantic-settings uses ``field.alias`` (or ``validation_alias`` when
    it is a string), otherwise the field name, then ``env_prefix``.
    Documented names are uppercase (``otel_enabled`` -> ``OTEL_ENABLED``).
    """
    alias: str | None = None
    if isinstance(field.alias, str) and field.alias:
        alias = field.alias
    elif isinstance(field.validation_alias, str) and field.validation_alias:
        alias = field.validation_alias
    raw: str = alias if alias is not None else field_name
    return f"{env_prefix}{raw}".upper()


def collect_settings_env_names(settings_cls: type[BaseSettings]) -> set[str]:
    """Env names for every Settings field except computed ``database_url``."""
    prefix: str = settings_env_prefix(settings_cls)
    computed: frozenset[str] = frozenset(settings_cls.model_computed_fields)
    names: set[str] = set()
    for field_name, field in settings_cls.model_fields.items():
        if field_name in COMPUTED_FIELD_EXCLUDE or field_name in computed:
            continue
        names.add(env_name_for_field(field_name, field, prefix))
    for computed_name in computed | COMPUTED_FIELD_EXCLUDE:
        names.discard(f"{prefix}{computed_name}".upper())
    return names


# ─────────────────────────────────────────────────────────────────────────
# Example file
# ─────────────────────────────────────────────────────────────────────────


def parse_env_example_keys(text: str) -> set[str]:
    """Return KEY tokens from ``KEY=`` / ``# KEY=`` assignment lines."""
    keys: set[str] = set()
    for line in text.splitlines():
        match: re.Match[str] | None = ENV_ASSIGNMENT_RE.match(line)
        if match is not None:
            keys.add(match.group(1))
    return keys


def collect_violations(settings_names: set[str], example_keys: set[str]) -> list[str]:
    """Compare Settings env names against keys parsed from the example."""
    errors: list[str] = []
    missing: list[str] = sorted(settings_names - example_keys)
    extra: list[str] = sorted(example_keys - settings_names)
    for name in missing:
        errors.append(f"missing Settings env {name} in .env.example")
    for name in extra:
        if name == WRONG_DATABASE_URL_NAME:
            errors.append("DATABASE_URL is not a Settings field; use DATABASE_URL_OVERRIDE")
            continue
        errors.append(f"unknown env {name} in .env.example is not a Settings field")
    return errors


def check_example_file(example_path: Path, settings_cls: type[BaseSettings]) -> list[str]:
    if not example_path.is_file():
        return [f"missing example file: {example_path}"]
    try:
        text: str = example_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"cannot read {example_path}: {exc}"]
    return collect_violations(
        collect_settings_env_names(settings_cls), parse_env_example_keys(text)
    )


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail when .env.example drifts from Settings env names.",
    )
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--example", default=None)
    parser.add_argument("--settings-module", default=DEFAULT_SETTINGS_MODULE)
    args = parser.parse_args(argv)

    repo_root: Path | None = resolve_repo_root(args.repo_root)
    if repo_root is None:
        print("[WARN] repo root not found — nothing to check (exit 0).")
        return 0

    example_path: Path | None = resolve_example_path(args.example, repo_root)
    if example_path is None:
        print("[FAIL] .env.example path could not be resolved")
        return 1

    settings_module: str = str(args.settings_module)
    try:
        settings_cls: type[BaseSettings] = load_settings_class(repo_root, settings_module)
    except (ImportError, TypeError) as exc:
        print(f"[FAIL] cannot import Settings from {settings_module}: {exc}")
        return 1

    errors: list[str] = check_example_file(example_path, settings_cls)
    if errors:
        for item in errors:
            print(f"[FAIL] {item}")
        print(f"[FAIL] {len(errors)} env-example violation(s)")
        return 1

    expected_count: int = len(collect_settings_env_names(settings_cls))
    print(
        f"[PASS] {example_path.name} matches {expected_count} Settings env "
        f"name(s) from {settings_module}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
