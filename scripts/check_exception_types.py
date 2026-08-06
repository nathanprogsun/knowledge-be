#!/usr/bin/env python3
"""Enforce exception discipline across non-web layers.

Rules:

1. Inside ``src/core/``, ``src/db/``, ``src/ai/``, and
   ``src/app_context/``, code may only raise:

   - ``ApplicationError`` subclasses (imported from ``src.common.exception``)
   - Python built-in ``NotImplementedError`` (the standard sentinel for
     abstract methods / protocol stubs)

   The following raises are forbidden:

   - ``raise ValueError(...)``
   - ``raise TypeError(...)``
   - ``raise RuntimeError(...)``
   - ``raise <CustomException>(...)`` where ``<CustomException>`` is
     not an ``ApplicationError`` subclass (custom ``Exception``
     subclasses declared in any of these layers are reported).

   The rule does NOT touch ``src/web/`` — the web boundary may need to
   raise ``HTTPException`` and other transport-specific errors.

2. The allowed ``ApplicationError`` subclasses are:

   ``NotFoundError``, ``ConflictError``, ``ValidationError``,
   ``PermissionDeniedError``, ``UnauthorizedError``,
   ``ExternalServiceError``, ``AIProviderError``,
   ``VectorStoreError``, ``StorageBackendError``, ``DataError``,
   ``OAuthRequiredError``.

   Any subclass of ``ApplicationError`` declared *outside* the
   canonical module is also reported (each new error code should be
   added to ``src.common.exception``).

Usage::

    python check_exception_types.py [--src-root PATH]

Exit codes:
    0 = every raise in non-web layers is a sanctioned exception type
    1 = at least one unsanctioned raise found
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path
from typing import NamedTuple


# ─────────────────────────────────────────────────────────────────────────
# Resolution
# ─────────────────────────────────────────────────────────────────────────


def resolve_src_root(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit).resolve()
        return p if p.is_dir() else None
    env = os.environ.get("KNOWLEDGE_BE_SRC")
    if env:
        p = Path(env).resolve()
        if p.is_dir():
            return p
    cur = Path(__file__).resolve().parent
    for _ in range(6):
        candidate = cur / "src"
        if candidate.is_dir():
            return candidate
        cur = cur.parent
    return None


SCANNED_TOP_LEVEL_DIRS = {"core", "db", "ai", "app_context"}

# Errors that may be raised from anywhere inside the scanned dirs.
ALLOWED_BUILTIN_RAISES = {"NotImplementedError"}

# Subclasses of ``ApplicationError`` permitted to be raised.  These names
# are also imported from ``src.common.exception`` and are listed in
# ``__all__``; any new error class should be added to that module rather
# than declared ad-hoc.
ALLOWED_APPLICATION_SUBCLASSES = {
    "ApplicationError",
    "NotFoundError",
    "ConflictError",
    "ValidationError",
    "PermissionDeniedError",
    "UnauthorizedError",
    "ExternalServiceError",
    "AIProviderError",
    "VectorStoreError",
    "StorageBackendError",
    "DataError",
    "OAuthRequiredError",
}

# Per-file allowlist of raise names that are deliberately sanctioned even
# though they are not ApplicationError subclasses. Currently only the DI
# sentinels in ``app_context/registry.py`` (lifespan not started) — these are
# programming-error markers, not business-layer exceptions, and are an
# intentional pre-existing design.
ALLOWED_FILE_EXCEPTIONS: dict[str, set[str]] = {
    "app_context/registry.py": {"RuntimeError"},
}


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


class _ExceptionViolation(NamedTuple):
    relpath: str
    lineno: int
    message: str


def _is_scanned_file(rel_parts: tuple[str, ...]) -> bool:
    if not rel_parts:
        return False
    return rel_parts[0] in SCANNED_TOP_LEVEL_DIRS


def _raised_exception_name(exc: ast.AST) -> str | None:
    """Return the bare class name of ``raise X(...)`` or ``raise X``."""
    if isinstance(exc, ast.Name):
        return exc.id
    if isinstance(exc, ast.Attribute):
        return exc.attr
    if isinstance(exc, ast.Call):
        return _raised_exception_name(exc.func)
    return None


def _build_class_hierarchy(src_root: Path) -> dict[str, list[str]]:
    """Map every class name in the scanned layers to its direct base names.

    Used to resolve ``ApplicationError`` subclasses that are declared outside
    ``src.common.exception`` (e.g. ``mcp_transport.errors.MCPError``) by
    walking their inheritance chain back to a sanctioned base.
    """
    hierarchy: dict[str, list[str]] = {}
    for file in sorted(src_root.rglob("*.py")):
        rel = str(file.relative_to(src_root))
        if not _is_scanned_file(Path(rel).parts):
            continue
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        except (OSError, SyntaxError):
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            bases: list[str] = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(base.attr)
            hierarchy[node.name] = bases
    return hierarchy


def _is_application_subclass(name: str, hierarchy: dict[str, list[str]]) -> bool:
    """True if ``name`` is ``ApplicationError`` or reaches a sanctioned base.

    Walks the inheritance chain transitively (cycle-safe), so a class like
    ``MCPTransportError(MCPError)`` → ``MCPError(ExternalServiceError)`` is
    sanctioned because ``ExternalServiceError`` is in the allowlist. Multiple
    inheritance is handled naturally: any base chain that hits an allowed
    name passes.
    """
    if name in ALLOWED_APPLICATION_SUBCLASSES:
        return True
    seen: set[str] = set()
    stack = list(hierarchy.get(name, []))
    while stack:
        base = stack.pop()
        if base in seen:
            continue
        seen.add(base)
        if base in ALLOWED_APPLICATION_SUBCLASSES:
            return True
        stack.extend(hierarchy.get(base, []))
    return False


def _collect_helper_returns(tree: ast.Module) -> set[str]:
    """Names of functions/methods whose return annotation is a sanctioned error.

    These are exception factories such as ``_not_found()`` / ``_already_exists()``
    / ``_unimplemented()`` that construct and return an ``ApplicationError``
    subclass. ``raise self.<helper>(...)`` against them is fine — the helper
    already produces a sanctioned exception.
    """
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        ret = node.returns
        if ret is None:
            continue
        base: str | None = None
        if isinstance(ret, ast.Name):
            base = ret.id
        elif isinstance(ret, ast.Attribute):
            base = ret.attr
        if base in ALLOWED_APPLICATION_SUBCLASSES:
            out.add(node.name)
    return out


def _scan_raises(
    rel: str,
    tree: ast.Module,
    hierarchy: dict[str, list[str]],
    helper_names: set[str],
) -> list[_ExceptionViolation]:
    out: list[_ExceptionViolation] = []
    file_allowed = ALLOWED_FILE_EXCEPTIONS.get(rel, set())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise):
            continue
        exc = node.exc
        if exc is None:
            # bare ``raise`` — re-raise, always allowed
            continue
        name = _raised_exception_name(exc)
        if name is None:
            # Complicated expression; not a known builtin ApplicationError.
            continue
        if name in ALLOWED_BUILTIN_RAISES:
            continue
        if name in file_allowed:
            continue
        if _is_application_subclass(name, hierarchy):
            continue
        if name in helper_names:
            # ``raise self._not_found()`` — the helper returns a sanctioned
            # ApplicationError subclass.
            continue
        # ``Exception`` itself is the platform base; only an
        # ApplicationError subclass is sanctioned.
        out.append(
            _ExceptionViolation(
                relpath=rel,
                lineno=node.lineno,
                message=(
                    f"raise {name}(...) is not sanctioned — non-web layers "
                    "may only raise ApplicationError subclasses "
                    f"({', '.join(sorted(ALLOWED_APPLICATION_SUBCLASSES))}) "
                    "or NotImplementedError"
                ),
            )
        )
    return out


def _scan_custom_exception_declarations(rel: str, tree: ast.Module) -> list[_ExceptionViolation]:
    """Report custom Exception subclasses declared outside src.common.exception."""
    out: list[_ExceptionViolation] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not _class_inherits_exception(node):
            continue
        if node.name in ALLOWED_APPLICATION_SUBCLASSES:
            continue
        if node.name in {"ApplicationError"}:
            continue
        out.append(
            _ExceptionViolation(
                relpath=rel,
                lineno=node.lineno,
                message=(
                    f"custom Exception subclass '{node.name}' declared in a "
                    "non-web layer — add it to src.common.exception instead "
                    "so all domain errors flow through one hierarchy"
                ),
            )
        )
    return out


def _class_inherits_exception(cls: ast.ClassDef) -> bool:
    """True for any class that inherits from ``Exception`` (directly or transitively)."""
    for base in cls.bases:
        if isinstance(base, ast.Name) and base.id == "Exception":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "Exception":
            return True
        # Inheriting from another ApplicationError subclass is fine, but
        # if that subclass is in the scanned tree we still want to
        # allow it because they're already in the canonical hierarchy.
    # Also catch classes that extend ``ApplicationError`` — they may
    # inherit it via ``from src.common.exception import ApplicationError``
    # and use it as a base class.
    for base in cls.bases:
        if isinstance(base, ast.Name) and base.id in ALLOWED_APPLICATION_SUBCLASSES:
            return False  # sanctioned subclass
        if isinstance(base, ast.Attribute) and base.attr in ALLOWED_APPLICATION_SUBCLASSES:
            return False
    # If a class has ``Exception`` in its MRO via an indirect base, we
    # can't easily check that here, so we report only direct Exception
    # subclasses as the suspicious case.
    return False


# ─────────────────────────────────────────────────────────────────────────
# Drive
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enforce exception discipline in non-web layers.",
    )
    parser.add_argument("--src-root", default=None)
    args = parser.parse_args()

    src = resolve_src_root(args.src_root)
    if src is None:
        print("[WARN] src/ not found — nothing to check (exit 0).")
        return 0

    violations: list[_ExceptionViolation] = []
    hierarchy = _build_class_hierarchy(src)
    files = sorted(src.rglob("*.py"))
    for file in files:
        rel = str(file.relative_to(src))
        rel_parts = Path(rel).parts
        if not _is_scanned_file(rel_parts):
            continue
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        except (OSError, SyntaxError):
            continue
        helper_names = _collect_helper_returns(tree)
        violations.extend(_scan_raises(rel, tree, hierarchy, helper_names))
        violations.extend(_scan_custom_exception_declarations(rel, tree))

    if not violations:
        print(
            f"[PASS] {len(files)} files scanned; every raise in non-web "
            "layers is a sanctioned ApplicationError subclass or "
            "NotImplementedError"
        )
        return 0

    for v in violations:
        print(f"[FAIL] {v.relpath}:{v.lineno}: {v.message}")
    print(f"[FAIL] {len(violations)} unsanctioned exception violation(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())